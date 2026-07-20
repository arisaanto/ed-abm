"""Sequential Monte Carlo ABC calibration and posterior condition comparison."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from statistics import mean, stdev
import random
from typing import Dict, Iterable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

import config
from src.analysis import cosine_similarity
from src.empirical import compute_empirical_summary
from src.simulation import Simulation


@contextmanager
def _temporary_config_updates(updates: Mapping[str, object]) -> Iterator[None]:
    previous_values = {key: getattr(config, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in previous_values.items():
            setattr(config, key, value)


def _normalize_counter(counter: Mapping[str, int | float]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {key: 0.0 for key in counter}
    return {key: float(value) / total for key, value in counter.items()}


def _mean_nested_dict(dicts: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    accumulators: Dict[str, list[float]] = {}
    for nested_dict in dicts:
        for key, value in nested_dict.items():
            accumulators.setdefault(key, []).append(float(value))
    return {key: mean(values) for key, values in accumulators.items()}


def _mean_topic_by_zone(summaries: Iterable[Mapping[str, Mapping[str, int]]]) -> Dict[str, Dict[str, int]]:
    accumulators: Dict[str, Dict[str, list[float]]] = {}
    for summary in summaries:
        for zone_id, topic_counts in summary.items():
            accumulators.setdefault(zone_id, {})
            for topic_name, count in topic_counts.items():
                accumulators[zone_id].setdefault(topic_name, []).append(float(count))
    return {
        zone_id: {
            topic_name: int(round(mean(values)))
            for topic_name, values in topics.items()
        }
        for zone_id, topics in accumulators.items()
    }


def _mean_duration_quantiles(summaries: Iterable[Mapping[str, Mapping[str, float]]]) -> Dict[str, Dict[str, float]]:
    accumulators: Dict[str, Dict[str, list[float]]] = {}
    for summary in summaries:
        for pair_key, quantiles in summary.items():
            accumulators.setdefault(pair_key, {})
            for quantile_name, value in quantiles.items():
                accumulators[pair_key].setdefault(quantile_name, []).append(float(value))
    return {
        pair_key: {
            quantile_name: mean(values)
            for quantile_name, values in quantile_lists.items()
        }
        for pair_key, quantile_lists in accumulators.items()
    }


def _mean_array(arrays: Iterable[np.ndarray]) -> np.ndarray:
    array_list = [np.asarray(array, dtype=float) for array in arrays]
    if not array_list:
        return np.zeros((1, 1), dtype=float)
    return np.mean(array_list, axis=0)


def _parameter_to_config_overrides(params: Mapping[str, object]) -> Dict[str, object]:
    overrides = {}
    for key, value in params.items():
        if key == "secondary_station_probability":
            overrides["NURSE_SECONDARY_STATION_PROBABILITY"] = float(value)
        elif key == "secondary_station_duration_mean":
            overrides["SECONDARY_STATION_DURATION_MEAN"] = float(value)
        elif key == "corridor_exchange_probability":
            overrides["OPPORTUNISTIC_CORRIDOR_INTERACTION_PROBABILITY"] = float(value)
        elif key == "station_exchange_probability":
            overrides["OPPORTUNISTIC_STATION_INTERACTION_PROBABILITY"] = float(value)
        elif key == "salience_distance_decay":
            overrides["SALIENCE_DISTANCE_DECAY"] = float(value)
        elif key == "attention_capacity":
            overrides["ATTENTION_CAPACITY"] = int(round(float(value)))
        elif key == "memory_recency_weight":
            overrides["MEMORY_RECENCY_WEIGHT"] = float(value)
        elif key == "memory_importance_weight":
            overrides["MEMORY_IMPORTANCE_WEIGHT"] = float(value)
        elif key == "doctor_response_bias":
            overrides["DOCTOR_RESPONSE_BIAS"] = float(value)
        elif key == "bedside_linger_multiplier":
            overrides["BEDSIDE_LINGER_MULTIPLIER"] = float(value)
    return overrides


def _resolve_calibration_settings(test_mode: bool) -> dict:
    if test_mode:
        return {
            "population_size": config.CALIBRATION_ABC_POPULATION_SIZE,
            "max_populations": config.CALIBRATION_ABC_MAX_POPULATIONS,
            "ensemble_runs": config.CALIBRATION_ABC_ENSEMBLE_RUNS,
            "duration_seconds": config.CALIBRATION_TEST_DURATION_SECONDS,
        }
    return {
        "population_size": config.CALIBRATION_ABC_POPULATION_SIZE_FULL,
        "max_populations": config.CALIBRATION_ABC_MAX_POPULATIONS_FULL,
        "ensemble_runs": config.CALIBRATION_ABC_ENSEMBLE_RUNS_FULL,
        "duration_seconds": config.CALIBRATION_FULL_DURATION_SECONDS,
    }


def _sample_prior_draw(rng: random.Random) -> Dict[str, object]:
    draw = {}
    for name, prior_spec in config.ABC_PRIORS.items():
        prior_type = prior_spec[0]
        if prior_type == "beta":
            draw[name] = rng.betavariate(float(prior_spec[1]), float(prior_spec[2]))
        elif prior_type == "uniform":
            draw[name] = rng.uniform(float(prior_spec[1]), float(prior_spec[2]))
        elif prior_type == "randint":
            draw[name] = rng.randint(int(prior_spec[1]), int(prior_spec[2]))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported prior type: {prior_type}")
    return draw


def _prior_bounds(prior_spec: tuple) -> tuple[float, float]:
    prior_type = prior_spec[0]
    if prior_type == "beta":
        return (0.0, 1.0)
    if prior_type == "uniform":
        return (float(prior_spec[1]), float(prior_spec[2]))
    if prior_type == "randint":
        return (float(prior_spec[1]), float(prior_spec[2]))
    raise ValueError(f"Unsupported prior type: {prior_type}")


def _perturb_parameter_draw(draw: Mapping[str, object], rng: random.Random) -> Dict[str, object]:
    proposal = {}
    for name, value in draw.items():
        prior_spec = config.ABC_PRIORS[name]
        lower_bound, upper_bound = _prior_bounds(prior_spec)
        if prior_spec[0] == "randint":
            proposal[name] = int(
                min(
                    max(round(float(value) + rng.choice([-1, 0, 1])), int(lower_bound)),
                    int(upper_bound),
                )
            )
        else:
            step_scale = max((upper_bound - lower_bound) * 0.1, 0.01)
            proposal_value = float(value) + rng.gauss(0.0, step_scale)
            proposal[name] = min(max(proposal_value, lower_bound), upper_bound)
    return proposal


class LightweightHistory:
    """Minimal history interface compatible with posterior summaries."""

    def __init__(self, populations: list[tuple[pd.DataFrame, np.ndarray]]) -> None:
        self.populations = populations
        self.max_t = len(populations) - 1

    def get_distribution(self, m: int = 0, t: Optional[int] = None):
        del m
        index = self.max_t if t is None else t
        dataframe, weights = self.populations[index]
        return dataframe.copy(), np.asarray(weights, dtype=float)


def _resample_particles(
    accepted_particles: list[dict],
    accepted_weights: list[float],
    target_size: int,
    rng: random.Random,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not accepted_particles:
        return pd.DataFrame(), np.asarray([], dtype=float)

    resampled_particles = []
    choices = list(accepted_particles)
    for _ in range(target_size):
        sampled = dict(rng.choices(choices, weights=accepted_weights, k=1)[0])
        resampled_particles.append(
            {
                **_perturb_parameter_draw(
                    {key: value for key, value in sampled.items() if not str(key).startswith("_")},
                    rng,
                ),
                "_distance": sampled.get("_distance", 0.0),
            }
        )

    weights = np.full(target_size, 1.0 / max(target_size, 1), dtype=float)
    return pd.DataFrame(resampled_particles), weights


def _run_manual_calibration(
    empirical: dict,
    settings: Mapping[str, int],
    thresholds: list[float],
    *,
    effective_ensemble_runs: int,
    effective_duration_seconds: int,
    evaluation_count: int,
) -> LightweightHistory:
    rng = random.Random(config.RUN_BASE_SEED)
    populations: list[tuple[pd.DataFrame, np.ndarray]] = []
    previous_population: list[dict] = []

    for population_index in range(min(settings["max_populations"], len(thresholds))):
        candidate_particles: list[dict] = []
        for _ in range(evaluation_count):
            if population_index == 0 or not previous_population:
                proposal = _sample_prior_draw(rng)
            else:
                proposal = _perturb_parameter_draw(rng.choice(previous_population), rng)

            summaries = run_baseline_ensemble(
                params=proposal,
                n_runs=effective_ensemble_runs,
                observation_window={"start_seconds": 0, "end_seconds": effective_duration_seconds},
                interaction_backend=config.CALIBRATION_INTERACTION_BACKEND,
                persona_source=config.PERSONA_SOURCE,
            )
            aggregated = aggregate_run_summaries(summaries)
            distance = distance_to_empirical(aggregated, empirical)
            candidate_particles.append({**proposal, "_distance": distance})

        candidate_particles.sort(key=lambda particle: particle["_distance"])
        accepted_particles = [
            particle for particle in candidate_particles
            if particle["_distance"] <= thresholds[population_index]
        ]
        if not accepted_particles:
            accepted_particles = candidate_particles[: max(3, evaluation_count // 3)]

        accepted_weights = [
            1.0 / (1.0 + float(particle["_distance"]))
            for particle in accepted_particles
        ]
        dataframe, weights = _resample_particles(
            accepted_particles,
            accepted_weights,
            settings["population_size"],
            rng,
        )
        populations.append((dataframe, weights))
        previous_population = [
            {
                key: value
                for key, value in particle.items()
                if not str(key).startswith("_")
            }
            for particle in accepted_particles
        ]

    return LightweightHistory(populations)


def _run_manual_test_calibration(empirical: dict, settings: Mapping[str, int]) -> LightweightHistory:
    thresholds = [
        config.CALIBRATION_TEST_DISTANCE_THRESHOLD,
        max(config.CALIBRATION_TEST_DISTANCE_THRESHOLD - 0.25, 1.5),
        max(config.CALIBRATION_TEST_DISTANCE_THRESHOLD - 0.5, 1.5),
    ]
    return _run_manual_calibration(
        empirical,
        settings,
        thresholds,
        effective_ensemble_runs=1,
        effective_duration_seconds=min(int(settings["duration_seconds"]), 60),
        evaluation_count=max(8, settings["population_size"] // 2),
    )


def run_baseline_ensemble(
    params: Optional[Mapping[str, object]] = None,
    n_runs: Optional[int] = None,
    observation_window: Optional[Mapping[str, int]] = None,
    interaction_backend: Optional[str] = None,
    persona_source: Optional[str] = None,
) -> list[dict]:
    """Run multiple baseline simulations and return per-run summaries."""

    updates = _parameter_to_config_overrides(params or {})
    updates["CONDITION_NAME"] = "baseline"
    run_count = config.ABC_ENSEMBLE_RUNS if n_runs is None else n_runs
    window = dict(config.OBSERVATION_WINDOW_SPEC if observation_window is None else observation_window)
    chosen_backend = config.CALIBRATION_INTERACTION_BACKEND if interaction_backend is None else interaction_backend

    summaries = []
    with _temporary_config_updates(updates):
        for run_index in range(run_count):
            simulation = Simulation(
                random_seed=config.RUN_BASE_SEED + run_index,
                condition_name="baseline",
                interaction_backend=chosen_backend,
                persona_source=persona_source or config.PERSONA_SOURCE,
            )
            while simulation.timestep < window["end_seconds"]:
                simulation.step()
            summaries.append(compute_summary_statistics(simulation.interaction_log, simulation))
    return summaries


def compute_summary_statistics(interactions: Iterable[dict], simulation: Simulation) -> dict:
    """Compute primary and secondary calibration statistics for one run."""

    interactions = list(interactions)
    zone_counts = simulation.interaction_zone_counts(interactions)
    role_pair_counts = simulation.interaction_role_pair_counts(interactions)
    topic_counts = simulation.interaction_topic_counts(interactions)

    topic_by_zone: Dict[str, Dict[str, int]] = {}
    duration_by_pair: Dict[str, list[float]] = {}
    for event in interactions:
        zone_id = simulation.which_zone(float(event["x"]), float(event["y"])) or "Outside named zones"
        topic = str(event.get("topic", "unknown_topic"))
        topic_by_zone.setdefault(zone_id, {})
        topic_by_zone[zone_id][topic] = topic_by_zone[zone_id].get(topic, 0) + 1

        pair = tuple(sorted((str(event["role_1"]), str(event["role_2"]))))
        pair_key = f"{pair[0]}|{pair[1]}"
        duration_by_pair.setdefault(pair_key, []).append(float(event.get("duration_seconds", 0.0)))

    duration_quantiles = {}
    for pair_key, durations in duration_by_pair.items():
        if not durations:
            continue
        duration_quantiles[pair_key] = {
            "q25": float(np.quantile(durations, 0.25)),
            "q50": float(np.quantile(durations, 0.50)),
            "q75": float(np.quantile(durations, 0.75)),
        }

    movement_metrics = simulation.movement_metrics()
    kde_grid = np.asarray(simulation.compute_kde_grid(), dtype=float)
    return {
        "zone_histogram": _normalize_counter(zone_counts),
        "role_pair_matrix": _normalize_counter({"|".join(pair): count for pair, count in role_pair_counts.items()}),
        "topic_distribution": _normalize_counter(topic_counts),
        "topic_by_zone": topic_by_zone,
        "duration_quantiles": duration_quantiles,
        "kde_grid": kde_grid.tolist(),
        "movement_metrics": movement_metrics,
        "llm_metrics": simulation.llm_metrics(),
        "interaction_count": len(interactions),
        "visibility_metrics": simulation.visibility_metrics_snapshot(),
    }


def aggregate_run_summaries(run_summaries: Iterable[Mapping[str, object]]) -> dict:
    """Average an ensemble of run summaries into one calibration summary."""

    run_summaries = list(run_summaries)
    if not run_summaries:
        return {}

    return {
        "zone_histogram": _mean_nested_dict(summary["zone_histogram"] for summary in run_summaries),
        "role_pair_matrix": _mean_nested_dict(summary["role_pair_matrix"] for summary in run_summaries),
        "topic_distribution": _mean_nested_dict(summary.get("topic_distribution", {}) for summary in run_summaries),
        "topic_by_zone": _mean_topic_by_zone(summary["topic_by_zone"] for summary in run_summaries),
        "duration_quantiles": _mean_duration_quantiles(
            summary["duration_quantiles"] for summary in run_summaries
        ),
        "kde_grid": _mean_array(np.asarray(summary["kde_grid"], dtype=float) for summary in run_summaries).tolist(),
        "movement_metrics": run_summaries[0]["movement_metrics"],
        "llm_metrics": _mean_nested_dict(summary["llm_metrics"] for summary in run_summaries),
        "interaction_count_mean": mean(summary["interaction_count"] for summary in run_summaries),
        "interaction_count_std": (
            stdev(summary["interaction_count"] for summary in run_summaries)
            if len(run_summaries) > 1
            else 0.0
        ),
        "visibility_metrics": run_summaries[0]["visibility_metrics"],
    }


def vectorize_summary_statistics(summary_dict: Mapping[str, object]) -> np.ndarray:
    """Flatten comparable statistics into one numeric vector."""

    vector = []

    for zone_id in sorted(summary_dict.get("zone_histogram", {})):
        vector.append(float(summary_dict["zone_histogram"][zone_id]))

    for pair_key in sorted(summary_dict.get("role_pair_matrix", {})):
        vector.append(float(summary_dict["role_pair_matrix"][pair_key]))

    for pair_key in sorted(summary_dict.get("duration_quantiles", {})):
        quantiles = summary_dict["duration_quantiles"][pair_key]
        vector.extend([
            float(quantiles.get("q25", 0.0)),
            float(quantiles.get("q50", 0.0)),
            float(quantiles.get("q75", 0.0)),
        ])

    kde_grid = np.asarray(summary_dict.get("kde_grid", []), dtype=float).ravel()
    vector.extend(kde_grid.tolist())
    return np.asarray(vector, dtype=float)


def _jensen_shannon_distance(
    left_distribution: Mapping[str, float],
    right_distribution: Mapping[str, float],
) -> float:
    labels = sorted(set(left_distribution) | set(right_distribution))
    left = np.asarray([float(left_distribution.get(label, 0.0)) for label in labels], dtype=float)
    right = np.asarray([float(right_distribution.get(label, 0.0)) for label in labels], dtype=float)
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    midpoint = 0.5 * (left + right)

    def _safe_kl(probabilities, reference):
        mask = probabilities > 0
        return float(np.sum(probabilities[mask] * np.log2(probabilities[mask] / np.maximum(reference[mask], 1e-12))))

    return 0.5 * _safe_kl(left, midpoint) + 0.5 * _safe_kl(right, midpoint)


def _topic_distance(sim_topics: Mapping[str, Mapping[str, int]], emp_topics: Mapping[str, Mapping[str, int]]) -> float:
    shared_zones = sorted(set(sim_topics) | set(emp_topics))
    if not shared_zones:
        return 0.0
    return mean(
        _jensen_shannon_distance(
            {label: float(value) for label, value in sim_topics.get(zone_id, {}).items()},
            {label: float(value) for label, value in emp_topics.get(zone_id, {}).items()},
        )
        for zone_id in shared_zones
    )


def _duration_distance(
    sim_durations: Mapping[str, Mapping[str, float]],
    emp_durations: Mapping[str, Mapping[str, float]],
) -> float:
    labels = sorted(set(sim_durations) | set(emp_durations))
    if not labels:
        return 0.0

    empirical_reference_scale = max(
        (
            float(value)
            for quantiles in emp_durations.values()
            for value in quantiles.values()
        ),
        default=300.0,
    )
    distances = []
    for label in labels:
        sim_quantiles = sim_durations.get(label, {})
        emp_quantiles = emp_durations.get(label, {})
        sim_vector = np.asarray(
            [float(sim_quantiles.get(key, 0.0)) for key in ("q25", "q50", "q75")],
            dtype=float,
        )
        emp_vector = np.asarray(
            [float(emp_quantiles.get(key, 0.0)) for key in ("q25", "q50", "q75")],
            dtype=float,
        )
        distances.append(float(np.mean(np.abs(sim_vector - emp_vector))) / max(empirical_reference_scale, 1.0))
    return mean(distances)


def compute_distance_debug(sim_summary: dict, emp_summary: dict) -> dict:
    """Return each composite-distance component separately for debugging."""

    zone_component = _jensen_shannon_distance(
        sim_summary.get("zone_histogram", {}),
        emp_summary.get("zone_histogram", {}),
    )
    role_pair_component = _jensen_shannon_distance(
        sim_summary.get("role_pair_matrix", {}),
        emp_summary.get("role_pair_matrix", {}),
    )
    topic_component = _topic_distance(
        sim_summary.get("topic_by_zone", {}),
        emp_summary.get("topic_by_zone", {}),
    )
    duration_component = _duration_distance(
        sim_summary.get("duration_quantiles", {}),
        emp_summary.get("duration_quantiles", {}),
    )

    sim_kde = np.asarray(sim_summary.get("kde_grid", []), dtype=float)
    emp_kde = np.asarray(emp_summary.get("kde_grid", []), dtype=float)
    if np.allclose(sim_kde, 0.0) and np.allclose(emp_kde, 0.0):
        kde_component = 0.0
    else:
        kde_component = 1.0 - cosine_similarity(sim_kde, emp_kde)

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


def distance_to_empirical(sim_summary: Mapping[str, object], emp_summary: Mapping[str, object]) -> float:
    """Weighted composite distance used by pyabc."""
    return compute_distance_debug(dict(sim_summary), dict(emp_summary))["weighted_total"]


def empirical_summary() -> dict:
    """Build the empirical reference summary used for calibration."""

    return compute_empirical_summary().as_dict()


def _posterior_weighted_mean(history) -> Dict[str, float]:
    distribution_frame, weights = history.get_distribution(m=0, t=history.max_t)
    weights_array = np.asarray(weights, dtype=float)
    weights_array = weights_array / max(weights_array.sum(), 1e-12)
    result: Dict[str, float] = {}
    for column_name in distribution_frame.columns:
        if str(column_name).startswith("_"):
            continue
        result[column_name] = float(np.sum(distribution_frame[column_name].to_numpy(dtype=float) * weights_array))
    return result


def _posterior_parameter_distribution(history) -> dict:
    distribution_frame, weights = history.get_distribution(m=0, t=history.max_t)
    weights_array = np.asarray(weights, dtype=float)
    weights_array = weights_array / max(weights_array.sum(), 1e-12)
    parameter_summary = {}
    for column_name in distribution_frame.columns:
        if str(column_name).startswith("_"):
            continue
        values = distribution_frame[column_name].to_numpy(dtype=float)
        parameter_summary[column_name] = {
            "mean": float(np.average(values, weights=weights_array)),
            "std": float(np.sqrt(np.average((values - np.average(values, weights=weights_array)) ** 2, weights=weights_array))),
            "p5": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return {
        "parameters": parameter_summary,
        "n_accepted_particles": int(len(distribution_frame)),
        "max_t": int(history.max_t),
        "calibration_note": "Calibrated on baseline condition only; interventions tested at posterior mean",
    }


def _write_calibration_outputs(history, test_mode: bool, settings: Mapping[str, int], backend_name: str) -> None:
    posterior_mean = _posterior_weighted_mean(history)
    summary_payload = {
        "max_t": int(history.max_t),
        "posterior_mean_parameters": posterior_mean,
        "test_mode": test_mode,
        "population_size": settings["population_size"],
        "max_populations": settings["max_populations"],
        "ensemble_runs": settings["ensemble_runs"],
        "backend": backend_name,
    }
    config.CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2))

    predictive = posterior_predictive_checks(history=history, test_mode=test_mode)
    distance_components = compute_distance_debug(
        {
            "zone_histogram": empirical_summary().get("zone_histogram", {}),
        },
        empirical_summary(),
    )
    # Replace the placeholder baseline-vs-empirical dict with the posterior predictive debug.
    posterior_summaries = run_baseline_ensemble(
        params=posterior_mean,
        n_runs=min(settings["ensemble_runs"], 2 if not test_mode else settings["ensemble_runs"]),
        interaction_backend=config.CALIBRATION_INTERACTION_BACKEND,
        observation_window={"start_seconds": 0, "end_seconds": settings["duration_seconds"]},
    )
    posterior_aggregated = aggregate_run_summaries(posterior_summaries)
    distance_components = compute_distance_debug(posterior_aggregated, empirical_summary())
    distance_components["interpretation"] = {
        "kde_component": "Spatial mismatch (0=perfect, 1=no overlap)",
        "zone_component": "Zone frequency mismatch (0=perfect, 1=maximal)",
        "role_pair_component": "Role pair proportion mismatch",
        "topic_component": "Topic distribution mismatch (Jensen-Shannon divergence)",
        "duration_component": "Duration quantile mismatch (normalized by empirical scale)",
    }
    distance_components["posterior_predictive_distance"] = predictive["distance_to_empirical"]
    config.CALIBRATION_DISTANCE_COMPONENTS_PATH.write_text(json.dumps(distance_components, indent=2))
    config.POSTERIOR_PARAMETER_DISTRIBUTIONS_PATH.write_text(
        json.dumps(_posterior_parameter_distribution(history), indent=2)
    )


def load_latest_history():
    """Load an existing pyABC history database if one has been created."""

    if not Path(config.ABC_HISTORY_DB_PATH).exists():
        return None

    try:
        from pyabc import History
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("pyabc is not installed.") from error

    history = History(f"sqlite:///{config.ABC_HISTORY_DB_PATH}")
    if history.max_t is None:
        return None
    return history


def run_prior_predictive_check(n_draws: int = 20, test_mode: bool = True) -> dict:
    """Sample from the prior and report the resulting empirical distances."""

    settings = _resolve_calibration_settings(test_mode=test_mode)
    empirical = empirical_summary()
    rng = random.Random(config.RUN_BASE_SEED)
    distances = []

    for draw_index in range(n_draws):
        sampled_params = _sample_prior_draw(rng)
        summaries = run_baseline_ensemble(
            params=sampled_params,
            n_runs=settings["ensemble_runs"],
            observation_window={
                "start_seconds": 0,
                "end_seconds": settings["duration_seconds"],
            },
            interaction_backend=config.CALIBRATION_INTERACTION_BACKEND,
            persona_source=config.PERSONA_SOURCE,
        )
        aggregated = aggregate_run_summaries(summaries)
        distances.append(distance_to_empirical(aggregated, empirical))

    payload = {
        "n_prior_draws": n_draws,
        "distances": distances,
        "min": min(distances) if distances else 0.0,
        "median": float(np.median(distances)) if distances else 0.0,
        "p5": float(np.quantile(distances, 0.05)) if distances else 0.0,
        "p95": float(np.quantile(distances, 0.95)) if distances else 0.0,
    }
    config.PRIOR_PREDICTIVE_DISTANCES_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _encode_summary_for_pyabc(summary: Mapping[str, object]) -> dict:
    return {"payload": json.dumps(summary)}


def _decode_summary_from_pyabc(encoded_summary: Mapping[str, object]) -> dict:
    payload = encoded_summary.get("payload", "{}")
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def run_pyabc_calibration(test_mode: bool = False) -> object:
    """Run SMC-ABC calibration and save a compact summary."""

    empirical = empirical_summary()
    settings = _resolve_calibration_settings(test_mode=test_mode)
    population_size = settings["population_size"]
    max_populations = settings["max_populations"]
    ensemble_runs = settings["ensemble_runs"]
    observation_window = {
        "start_seconds": 0,
        "end_seconds": settings["duration_seconds"],
    }

    if test_mode:
        history = _run_manual_test_calibration(empirical=empirical, settings=settings)
        if int(history.max_t) < 1:
            raise RuntimeError(
                "Test calibration stopped before completing a real posterior population "
                f"(history.max_t={history.max_t})."
            )
        _write_calibration_outputs(history, test_mode, settings, "manual_smc_test")
        summary_payload = json.loads(config.CALIBRATION_SUMMARY_PATH.read_text())
        summary_payload["effective_ensemble_runs"] = 1
        summary_payload["effective_duration_seconds"] = min(int(settings["duration_seconds"]), 60)
        config.CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2))
        return history

    thresholds = [2.2, 2.0, 1.8, 1.6, 1.5]
    history = _run_manual_calibration(
        empirical,
        settings,
        thresholds,
        effective_ensemble_runs=min(settings["ensemble_runs"], 2),
        effective_duration_seconds=settings["duration_seconds"],
        evaluation_count=12,
    )
    if int(history.max_t) < 3:
        raise RuntimeError(
            "Full calibration stopped before completing at least three SMC populations "
            f"(history.max_t={history.max_t})."
        )
    _write_calibration_outputs(history, test_mode, settings, "manual_smc_full")
    summary_payload = json.loads(config.CALIBRATION_SUMMARY_PATH.read_text())
    summary_payload["effective_ensemble_runs"] = min(settings["ensemble_runs"], 2)
    summary_payload["effective_duration_seconds"] = settings["duration_seconds"]
    summary_payload["effective_candidate_evaluations_per_population"] = 12
    config.CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2))
    return history

    try:
        import pyabc
        from pyabc import Distribution, RV
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("pyabc is not installed.") from error

    def model(parameter_dict):
        summaries = run_baseline_ensemble(
            params=parameter_dict,
            n_runs=ensemble_runs,
            interaction_backend=config.CALIBRATION_INTERACTION_BACKEND,
            observation_window=observation_window,
        )
        return _encode_summary_for_pyabc(aggregate_run_summaries(summaries))

    prior_kwargs = {}
    for name, prior_spec in config.ABC_PRIORS.items():
        prior_type = prior_spec[0]
        if prior_type == "beta":
            prior_kwargs[name] = RV("beta", prior_spec[1], prior_spec[2])
        elif prior_type == "uniform":
            prior_kwargs[name] = RV("uniform", prior_spec[1], prior_spec[2] - prior_spec[1])
        elif prior_type == "randint":
            prior_kwargs[name] = RV("randint", prior_spec[1], prior_spec[2] + 1)
        else:
            raise ValueError(f"Unsupported prior type: {prior_type}")

    abc = pyabc.ABCSMC(
        models=model,
        parameter_priors=Distribution(**prior_kwargs),
        distance_function=lambda sim, emp: distance_to_empirical(
            _decode_summary_from_pyabc(sim),
            _decode_summary_from_pyabc(emp),
        ),
        population_size=population_size,
        eps=pyabc.epsilon.ConstantEpsilon(config.ABC_DISTANCE_THRESHOLD),
        sampler=pyabc.sampler.SingleCoreSampler(),
    )
    history = abc.new(
        f"sqlite:///{config.ABC_HISTORY_DB_PATH}",
        _encode_summary_for_pyabc(empirical),
    )
    history = abc.run(
        minimum_epsilon=config.ABC_DISTANCE_THRESHOLD,
        max_nr_populations=max_populations,
    )

    if int(history.max_t) < 1:
        raise RuntimeError(
            "ABC calibration stopped before completing a real posterior population "
            f"(history.max_t={history.max_t}). This usually means the distance threshold "
            "is still too strict or the summary statistics remain poorly scaled."
        )

    posterior_mean = _posterior_weighted_mean(history)
    summary_payload = {
        "max_t": int(history.max_t),
        "posterior_mean_parameters": posterior_mean,
        "test_mode": test_mode,
        "population_size": population_size,
        "max_populations": max_populations,
        "ensemble_runs": ensemble_runs,
    }
    config.CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2))
    return history


def posterior_predictive_checks(
    history=None,
    test_mode: bool = False,
    n_runs: Optional[int] = None,
) -> dict:
    """Run baseline posterior predictive checks at the posterior mean."""

    if history is None:
        history = run_pyabc_calibration(test_mode=test_mode)

    posterior_mean = _posterior_weighted_mean(history)
    empirical = empirical_summary()
    settings = _resolve_calibration_settings(test_mode=test_mode)
    summaries = run_baseline_ensemble(
        params=posterior_mean,
        n_runs=settings["ensemble_runs"] if n_runs is None else n_runs,
        interaction_backend=config.CALIBRATION_INTERACTION_BACKEND,
        observation_window={
            "start_seconds": 0,
            "end_seconds": settings["duration_seconds"],
        },
    )
    aggregated = aggregate_run_summaries(summaries)
    distance = distance_to_empirical(aggregated, empirical)
    return {
        "posterior_mean_parameters": posterior_mean,
        "distance_to_empirical": distance,
        "interaction_count_mean": aggregated.get("interaction_count_mean", 0.0),
        "interaction_count_std": aggregated.get("interaction_count_std", 0.0),
    }


def run_condition_comparison_from_posterior(history=None, test_mode: bool = False) -> dict:
    """Run all four design conditions at the calibrated posterior mean."""

    if history is None:
        history = run_pyabc_calibration(test_mode=test_mode)

    from src.experiments import run_condition_experiments

    posterior_mean = _posterior_weighted_mean(history)
    settings = _resolve_calibration_settings(test_mode=test_mode)
    return run_condition_experiments(
        params=posterior_mean,
        n_runs=settings["ensemble_runs"] if test_mode else config.EXPERIMENT_ENSEMBLE_RUNS,
        interaction_backend=config.EXPERIMENT_INTERACTION_BACKEND,
        persona_source=config.PERSONA_SOURCE,
        duration_seconds=(
            config.EXPERIMENT_TEST_DURATION_SECONDS
            if test_mode
            else config.EXPERIMENT_DURATION_SECONDS
        ),
        save_outputs=True,
    )
