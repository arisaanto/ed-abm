"""Run-level validation and outcome metrics used by the canonical runner."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Dict, Iterable, Mapping

import numpy as np

import config
from src.analysis import compute_kde_grid, cosine_similarity
from src.empirical import classify_zone_supergroup


def _normalize_counter(
    counter: Mapping[str, int | float],
) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {str(key): 0.0 for key in counter}
    return {str(key): float(value) / total for key, value in counter.items()}


def compute_summary_statistics(interactions: Iterable[dict], simulation) -> dict:
    """Compute the validation statistics retained for one simulation run."""

    interactions = list(interactions)
    zone_counts = simulation.interaction_zone_counts(interactions)
    role_pair_counts = simulation.interaction_role_pair_counts(interactions)
    topic_counts = simulation.interaction_topic_counts(interactions)

    topic_by_zone: Dict[str, Dict[str, int]] = {}
    duration_by_pair: Dict[str, list[float]] = {}
    for event in interactions:
        zone_id = (
            simulation.which_zone(float(event["x"]), float(event["y"]))
            or "Outside named zones"
        )
        topic = str(event.get("topic", "unknown_topic"))
        topic_by_zone.setdefault(zone_id, {})
        topic_by_zone[zone_id][topic] = (
            topic_by_zone[zone_id].get(topic, 0) + 1
        )

        pair = tuple(sorted((str(event["role_1"]), str(event["role_2"]))))
        pair_key = f"{pair[0]}|{pair[1]}"
        duration_by_pair.setdefault(pair_key, []).append(
            float(event.get("duration_seconds", 0.0))
        )

    duration_quantiles = {}
    for pair_key, durations in duration_by_pair.items():
        if durations:
            duration_quantiles[pair_key] = {
                "q25": float(np.quantile(durations, 0.25)),
                "q50": float(np.quantile(durations, 0.50)),
                "q75": float(np.quantile(durations, 0.75)),
            }

    interaction_points = [
        (float(event["x"]), float(event["y"]))
        for event in interactions
        if event.get("x") not in (None, "") and event.get("y") not in (None, "")
    ]

    return {
        "zone_histogram": _normalize_counter(zone_counts),
        "role_pair_matrix": _normalize_counter(
            {"|".join(pair): count for pair, count in role_pair_counts.items()}
        ),
        "topic_distribution": _normalize_counter(topic_counts),
        "topic_by_zone": topic_by_zone,
        "duration_quantiles": duration_quantiles,
        "kde_grid": np.asarray(
            compute_kde_grid(
                interaction_points,
                simulation.environment.plot_bounds,
                resolution=1.0,
                bandwidth=1.5,
            ),
            dtype=float,
        ).tolist(),
        "movement_metrics": simulation.movement_metrics(),
        "llm_metrics": simulation.llm_metrics(),
        "interaction_count": len(interactions),
        "visibility_metrics": simulation.visibility_metrics_snapshot(),
    }


def _jensen_shannon_distance(
    left_distribution: Mapping[str, float],
    right_distribution: Mapping[str, float],
) -> float:
    labels = sorted(set(left_distribution) | set(right_distribution))
    if not labels:
        return 0.0
    left = np.asarray(
        [float(left_distribution.get(label, 0.0)) for label in labels],
        dtype=float,
    )
    right = np.asarray(
        [float(right_distribution.get(label, 0.0)) for label in labels],
        dtype=float,
    )
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    midpoint = 0.5 * (left + right)

    def safe_kl(probabilities: np.ndarray, reference: np.ndarray) -> float:
        mask = probabilities > 0
        return float(
            np.sum(
                probabilities[mask]
                * np.log2(
                    probabilities[mask]
                    / np.maximum(reference[mask], 1e-12)
                )
            )
        )

    return 0.5 * safe_kl(left, midpoint) + 0.5 * safe_kl(right, midpoint)


def _topic_distance(
    simulated: Mapping[str, Mapping[str, int]],
    empirical: Mapping[str, Mapping[str, int]],
) -> float:
    zones = sorted(set(simulated) | set(empirical))
    if not zones:
        return 0.0
    return mean(
        _jensen_shannon_distance(
            {
                label: float(value)
                for label, value in simulated.get(zone_id, {}).items()
            },
            {
                label: float(value)
                for label, value in empirical.get(zone_id, {}).items()
            },
        )
        for zone_id in zones
    )


def _duration_distance(
    simulated: Mapping[str, Mapping[str, float]],
    empirical: Mapping[str, Mapping[str, float]],
) -> float:
    labels = sorted(set(simulated) | set(empirical))
    if not labels:
        return 0.0

    reference_scale = max(
        (
            float(value)
            for quantiles in empirical.values()
            for value in quantiles.values()
        ),
        default=300.0,
    )
    distances = []
    for label in labels:
        sim_quantiles = simulated.get(label, {})
        emp_quantiles = empirical.get(label, {})
        sim_vector = np.asarray(
            [
                float(sim_quantiles.get(key, 0.0))
                for key in ("q25", "q50", "q75")
            ],
            dtype=float,
        )
        emp_vector = np.asarray(
            [
                float(emp_quantiles.get(key, 0.0))
                for key in ("q25", "q50", "q75")
            ],
            dtype=float,
        )
        distances.append(
            float(np.mean(np.abs(sim_vector - emp_vector)))
            / max(reference_scale, 1.0)
        )
    return mean(distances)


def compute_distance_components(
    simulated_summary: Mapping[str, object],
    empirical_summary: Mapping[str, object],
) -> dict:
    """Return each component of the frozen Part 1 validation distance."""

    zone_component = _jensen_shannon_distance(
        simulated_summary.get("zone_histogram", {}),
        empirical_summary.get("zone_histogram", {}),
    )
    role_pair_component = _jensen_shannon_distance(
        simulated_summary.get("role_pair_matrix", {}),
        empirical_summary.get("role_pair_matrix", {}),
    )
    topic_component = _topic_distance(
        simulated_summary.get("topic_by_zone", {}),
        empirical_summary.get("topic_by_zone", {}),
    )
    duration_component = _duration_distance(
        simulated_summary.get("duration_quantiles", {}),
        empirical_summary.get("duration_quantiles", {}),
    )

    simulated_kde = np.asarray(
        simulated_summary.get("kde_grid", []), dtype=float
    )
    empirical_kde = np.asarray(
        empirical_summary.get("kde_grid", []), dtype=float
    )
    if np.allclose(simulated_kde, 0.0) and np.allclose(
        empirical_kde, 0.0
    ):
        kde_component = 0.0
    else:
        kde_component = 1.0 - cosine_similarity(
            simulated_kde, empirical_kde
        )

    weights = config.ABC_DISTANCE_WEIGHTS
    weighted_total = (
        (weights["zone_histogram"] * zone_component)
        + (weights["role_pair_matrix"] * role_pair_component)
        + (weights["topic_by_zone"] * topic_component)
        + (weights["duration_quantiles"] * duration_component)
        + (weights["kde_grid"] * kde_component)
    )
    return {
        "kde_component": kde_component,
        "zone_component": zone_component,
        "role_pair_component": role_pair_component,
        "topic_component": topic_component,
        "duration_component": duration_component,
        "weighted_total": weighted_total,
    }


def compute_outcome_metrics(
    interactions: Iterable[dict],
    missed_opportunity_count: int,
) -> dict:
    """Summarize the behavioral outcomes retained in completed study outputs."""

    interactions = list(interactions)
    interaction_count = len(interactions)
    role_pair_counts = Counter(
        "|".join(
            sorted((str(event["role_1"]), str(event["role_2"])))
        )
        for event in interactions
    )
    topic_counts = Counter(
        str(event.get("topic", "unknown")) for event in interactions
    )
    zone_counts = Counter(
        str(event.get("zone_id", "Outside named zones"))
        for event in interactions
    )
    hcw_patient_count = sum(
        1
        for event in interactions
        if "Patient" in {event["role_1"], event["role_2"]}
    )
    high_acuity_count = sum(
        1 for event in interactions if event.get("esi_level") in {1, 2}
    )
    zone_group_counts = Counter(
        classify_zone_supergroup(
            str(event.get("zone_id", "Outside named zones")),
            float(event.get("x", 0.0)),
            float(event.get("y", 0.0)),
        )
        for event in interactions
    )
    durations = [
        float(event.get("duration_seconds", 0.0))
        for event in interactions
    ]
    denominator = max(interaction_count, 1)
    return {
        "interaction_count": interaction_count,
        "missed_opportunity_episodes": int(missed_opportunity_count),
        "hcw_hcw_share": (
            interaction_count - hcw_patient_count
        )
        / denominator,
        "hcw_patient_share": hcw_patient_count / denominator,
        "patient_facing_share": sum(
            1
            for event in interactions
            if event.get("patient_id") is not None
        )
        / denominator,
        "high_acuity_interaction_share": high_acuity_count / denominator,
        "mean_duration": float(mean(durations)) if durations else 0.0,
        "median_duration": float(median(durations)) if durations else 0.0,
        "station_or_desk_interaction_share": (
            zone_group_counts.get("station_or_desk", 0) / denominator
        ),
        "corridor_interaction_share": (
            zone_group_counts.get("corridor", 0) / denominator
        ),
        "bedside_or_patient_room_interaction_share": (
            zone_group_counts.get("bedside_or_patient_room", 0)
            / denominator
        ),
        "other_or_uncoded_interaction_share": (
            zone_group_counts.get("other_or_uncoded", 0) / denominator
        ),
        "outside_named_zones_interaction_share": sum(
            1
            for event in interactions
            if str(event.get("zone_id", "Outside named zones"))
            == "Outside named zones"
        )
        / denominator,
        "role_pair_distribution": _normalize_counter(role_pair_counts),
        "topic_distribution": _normalize_counter(topic_counts),
        "zone_distribution": _normalize_counter(zone_counts),
        "zone_supergroup_distribution": _normalize_counter(
            zone_group_counts
        ),
    }
