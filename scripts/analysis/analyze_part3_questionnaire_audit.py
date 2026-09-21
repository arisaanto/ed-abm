#!/usr/bin/env python3
"""Analyze synthetic questionnaire responses and optionally compare hospital scales.

The optional empirical comparison reads the confidential source CSV (or ARIS ZIP)
locally, collapses repeated interaction rows to person-shifts, and exports aggregate
statistics only. Raw questionnaire rows are never copied to the output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import zipfile
from typing import Any

import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.questionnaire_audit import QUESTIONNAIRE_ITEMS, QUESTION_IDS


ROLE_MAP = {
    "nurse": "Nurse",
    "assistant_doctor": "Doctor",
    "coordination": "CoordinationNurse",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _numeric(question_id: str, row: dict[str, Any]) -> float | None:
    value = row.get("share_percent") if question_id == "q2" else row.get("score_1_to_7")
    return float(value) if value is not None else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _wasserstein_distance(left: list[float], right: list[float]) -> float:
    """Return the exact one-dimensional earth-mover distance."""

    if not left or not right:
        raise ValueError("Wasserstein distance requires two non-empty samples")
    support = sorted(set(left) | set(right))
    if len(support) == 1:
        return 0.0
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    left_index = right_index = 0
    distance = 0.0
    for current, following in zip(support, support[1:]):
        while left_index < len(left_sorted) and left_sorted[left_index] <= current:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= current:
            right_index += 1
        left_cdf = left_index / len(left_sorted)
        right_cdf = right_index / len(right_sorted)
        distance += abs(left_cdf - right_cdf) * (following - current)
    return distance


def _total_variation_distance(left: list[float], right: list[float]) -> float:
    """Compare the complete discrete response-category composition."""

    if not left or not right:
        raise ValueError("Total variation requires two non-empty samples")
    support = set(left) | set(right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    return 0.5 * sum(
        abs(left_counts[value] / len(left) - right_counts[value] / len(right))
        for value in support
    )


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = (
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    ) ** 0.5
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _is_scientific_result(
    verification: dict[str, Any],
    empirical_comparison: list[dict[str, Any]],
    empirical_adequacy: dict[str, Any] | None = None,
) -> bool:
    """Require the frozen internal and empirical convergence gates."""

    return bool(
        empirical_comparison
        and verification.get("scientific_gate_pass") is True
        and empirical_adequacy
        and empirical_adequacy.get("descriptive_convergence_gate_pass") is True
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _long_rows(
    packets: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    packet_by_id = {str(row["prompt_id"]): row for row in packets}
    rows = []
    for result in responses:
        packet = packet_by_id[str(result["prompt_id"])]
        metadata = packet["trace_evidence"]["metadata"]
        normalized = result.get("response") or {}
        for response in normalized.get("responses", []):
            question_id = str(response["question_id"])
            rows.append(
                {
                    "prompt_id": result["prompt_id"],
                    "control_variant": packet["control_variant"],
                    "assessment_variant": packet.get("assessment_variant", "balanced"),
                    "scenario": metadata["scenario_mode"],
                    "seed": metadata["seed"],
                    "persona_id": metadata["persona_id"],
                    "role": metadata["role"],
                    "run_id": metadata["run_id"],
                    "agent_id": metadata["agent_id"],
                    "question_id": question_id,
                    "rateability": response["rateability"],
                    "value": _numeric(question_id, response),
                    "confidence_1_to_5": response["confidence_1_to_5"],
                    "short_rationale": response["short_rationale"],
                    "uncertainty_note": response["uncertainty_note"],
                    "evidence_event_ids": "|".join(response["evidence_event_ids"]),
                }
            )
    return rows


def _collapse_assessment_variants(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain one scientific response per question and simulated shift."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    output = []
    for row in rows:
        if row["control_variant"] != "full_trace_persona":
            output.append({**row, "assessment_count": 1, "assessment_range": 0.0})
            continue
        grouped[
            (str(row["run_id"]), str(row["agent_id"]), str(row["question_id"]))
        ].append(row)
    for key, members in sorted(grouped.items()):
        variants = {str(row["assessment_variant"]) for row in members}
        if variants != {"balanced"} or len(members) != 1:
            raise ValueError(
                f"Expected one balanced assessment for {key}; got "
                f"{len(members)} rows with variants {sorted(variants)}"
            )
        output.append(
            {
                **members[0],
                "assessment_variant": "absolute_adequacy_balanced",
                "assessment_count": 1,
                "assessment_range": 0.0,
            }
        )
    return output


def _synthetic_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["control_variant"]), str(row["scenario"]),
                str(row["persona_id"]), str(row["role"]), str(row["question_id"]),
            )
        ].append(row)
    output = []
    for key, members in sorted(grouped.items()):
        values = [float(row["value"]) for row in members if row["value"] is not None]
        output.append(
            {
                "control_variant": key[0],
                "scenario": key[1],
                "persona_id": key[2],
                "role": key[3],
                "question_id": key[4],
                "n_packets": len(members),
                "n_rateable": len(values),
                "rateability_rate": len(values) / len(members),
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
        )
    return output


def _item_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    personas = sorted(
        {str(row["persona_id"]) for row in rows if row["control_variant"] == "full_trace_persona"}
    )
    for question_id in QUESTION_IDS:
        main = [
            row
            for row in rows
            if row["control_variant"] == "full_trace_persona"
            and row["question_id"] == question_id
        ]
        neutral = [
            row
            for row in rows
            if row["control_variant"] == "full_trace_neutral"
            and row["question_id"] == question_id
        ]
        persona_rates = []
        for persona in personas:
            members = [row for row in main if row["persona_id"] == persona]
            persona_rates.append(
                sum(row["value"] is not None for row in members) / len(members)
            )
        scenario_values = {
            scenario: [
                float(row["value"])
                for row in main
                if row["scenario"] == scenario and row["value"] is not None
            ]
            for scenario in ("normal_load", "high_load_high_acuity")
        }
        normal = scenario_values["normal_load"]
        high = scenario_values["high_load_high_acuity"]
        normal_mean = statistics.mean(normal) if normal else None
        high_mean = statistics.mean(high) if high else None
        output.append(
            {
                "question_id": question_id,
                "full_persona_rateability": (
                    sum(row["value"] is not None for row in main) / len(main)
                ),
                "full_neutral_rateability": (
                    sum(row["value"] is not None for row in neutral) / len(neutral)
                ),
                "minimum_persona_rateability": min(persona_rates),
                "maximum_persona_rateability": max(persona_rates),
                "normal_n_rateable": len(normal),
                "normal_mean": normal_mean,
                "normal_sd": _sample_sd(normal),
                "high_load_n_rateable": len(high),
                "high_load_mean": high_mean,
                "high_load_sd": _sample_sd(high),
                "high_minus_normal": (
                    high_mean - normal_mean
                    if high_mean is not None and normal_mean is not None
                    else None
                ),
            }
        )
    return output


def _empirical_reader(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.empirical_zip:
        with zipfile.ZipFile(args.empirical_zip) as archive:
            with archive.open(args.empirical_member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig") as handle:
                    return list(csv.DictReader(handle))
    if args.empirical_csv:
        with Path(args.empirical_csv).open(encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("No empirical source supplied")


def _load_empirical_shifts(args: argparse.Namespace) -> list[dict[str, Any]]:
    shifts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _empirical_reader(args):
        key = (str(row.get("date", "")), str(row.get("study_id", "")))
        if key in shifts:
            continue
        source_role = str(row.get("role_survey", "")).strip()
        role = ROLE_MAP.get(source_role)
        values = {}
        for question_id in QUESTION_IDS:
            raw = str(row.get(question_id, "")).strip()
            if not raw:
                values[question_id] = None
            elif question_id == "q2":
                values[question_id] = float(raw.rstrip("%"))
            else:
                values[question_id] = float(raw)
        shifts[key] = {
            "role": role,
            "source_role": source_role,
            "values": values,
        }
    return list(shifts.values())


def _bootstrap_comparison(
    synthetic: list[dict[str, Any]], empirical: list[dict[str, Any]], *, draws: int
) -> list[dict[str, Any]]:
    rng = random.Random(20260731)
    candidates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in synthetic:
        if row["control_variant"] != "full_trace_persona" or row["value"] is None:
            continue
        candidates[(str(row["scenario"]), str(row["question_id"]), str(row["role"]))].append(float(row["value"]))
    output = []
    for question_id in QUESTION_IDS:
        if question_id == "q8":
            continue
        observed = [
            (row["role"], row["values"][question_id])
            for row in empirical
            if row["role"] in ROLE_MAP.values() and row["values"][question_id] is not None
        ]
        observed_values = [float(value) for _role, value in observed]
        role_counts = Counter(role for role, _value in observed)
        for scenario in ("normal_load", "high_load_high_acuity"):
            standardized_pool = []
            for role, count in role_counts.items():
                pool = candidates[(scenario, question_id, role)]
                if not pool:
                    raise ValueError(f"No synthetic pool for {scenario}/{question_id}/{role}")
                standardized_pool.extend(
                    pool[index % len(pool)] for index in range(count * 100)
                )
            bootstrap_means = []
            bootstrap_medians = []
            bootstrap_sds = []
            null_wasserstein = []
            null_total_variation = []
            for _ in range(draws):
                sampled = []
                for role, count in role_counts.items():
                    pool = candidates[(scenario, question_id, role)]
                    sampled.extend(rng.choice(pool) for _index in range(count))
                bootstrap_means.append(statistics.mean(sampled))
                bootstrap_medians.append(statistics.median(sampled))
                bootstrap_sds.append(statistics.stdev(sampled))
                null_wasserstein.append(
                    _wasserstein_distance(sampled, standardized_pool)
                )
                null_total_variation.append(
                    _total_variation_distance(sampled, standardized_pool)
                )
            observed_mean = statistics.mean(observed_values)
            observed_median = statistics.median(observed_values)
            observed_sd = statistics.stdev(observed_values)
            synthetic_mean = statistics.mean(bootstrap_means)
            synthetic_sd = statistics.mean(bootstrap_sds)
            mean_margin = 12.5 if question_id == "q2" else 0.5
            severe_margin = 25.0 if question_id == "q2" else 1.0
            mean_low = _quantile(bootstrap_means, 0.025)
            mean_high = _quantile(bootstrap_means, 0.975)
            median_low = _quantile(bootstrap_medians, 0.025)
            median_high = _quantile(bootstrap_medians, 0.975)
            wasserstein = _wasserstein_distance(
                observed_values, standardized_pool
            )
            scale_span = 100.0 if question_id == "q2" else 6.0
            normalized_wasserstein = wasserstein / scale_span
            null_wasserstein_95 = float(_quantile(null_wasserstein, 0.95) or 0.0)
            total_variation = _total_variation_distance(
                observed_values, standardized_pool
            )
            null_total_variation_95 = float(
                _quantile(null_total_variation, 0.95) or 0.0
            )
            distribution_pass = bool(
                wasserstein <= null_wasserstein_95
                and total_variation <= null_total_variation_95
                and normalized_wasserstein <= 0.20
            )
            convergence_evidence_eligible = question_id != "q2"
            descriptive_mean_within_margin = (
                abs(synthetic_mean - observed_mean) <= mean_margin
            )
            sd_ratio = synthetic_sd / observed_sd if observed_sd else None
            output.append(
                {
                    "question_id": question_id,
                    "scenario": scenario,
                    "empirical_n": len(observed_values),
                    "empirical_role_mix": ";".join(f"{role}:{count}" for role, count in sorted(role_counts.items())),
                    "empirical_mean": observed_mean,
                    "empirical_median": observed_median,
                    "empirical_sd": observed_sd,
                    "empirical_minimum": min(observed_values),
                    "empirical_maximum": max(observed_values),
                    "synthetic_role_standardized_mean": synthetic_mean,
                    "synthetic_mean_bias": synthetic_mean - observed_mean,
                    "synthetic_absolute_mean_difference": abs(synthetic_mean - observed_mean),
                    "prespecified_mean_margin": mean_margin,
                    "value_convergence_evidence_eligible": convergence_evidence_eligible,
                    "q2_anchor_note": (
                        "The ordinary-load mean is structurally anchored by the "
                        "top-quartile critical-window definition and is descriptive only."
                        if question_id == "q2" else None
                    ),
                    "descriptive_mean_within_margin": descriptive_mean_within_margin,
                    "mean_equivalence_pass": (
                        descriptive_mean_within_margin
                        if convergence_evidence_eligible else None
                    ),
                    "severe_mean_failure": bool(
                        convergence_evidence_eligible
                        and abs(synthetic_mean - observed_mean) > severe_margin
                    ),
                    "synthetic_mean_95pct_low": mean_low,
                    "synthetic_mean_95pct_high": mean_high,
                    "empirical_mean_within_synthetic_95pct": bool(mean_low <= observed_mean <= mean_high),
                    "synthetic_role_standardized_median": statistics.median(bootstrap_medians),
                    "synthetic_role_standardized_sd": synthetic_sd,
                    "synthetic_to_empirical_sd_ratio": sd_ratio,
                    "dispersion_ratio_pass": bool(
                        sd_ratio is not None and 0.67 <= sd_ratio <= 1.50
                    ),
                    "wasserstein_distance": wasserstein,
                    "normalized_wasserstein_distance": normalized_wasserstein,
                    "finite_sample_wasserstein_95pct": null_wasserstein_95,
                    "total_variation_distance": total_variation,
                    "finite_sample_total_variation_95pct": null_total_variation_95,
                    "distribution_distance_ceiling": 0.20,
                    "distribution_compatibility_pass": distribution_pass,
                    "distribution_convergence_evidence_eligible": True,
                    "synthetic_median_95pct_low": median_low,
                    "synthetic_median_95pct_high": median_high,
                    "empirical_median_within_synthetic_95pct": bool(median_low <= observed_median <= median_high),
                }
            )
    return output


def _synthetic_shift_vectors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    roles: dict[tuple[str, str, str, str], str] = {}
    for row in rows:
        if (
            row["control_variant"] != "full_trace_persona"
            or row["value"] is None
            or row["question_id"] == "q8"
        ):
            continue
        key = (
            str(row["scenario"]), str(row["run_id"]),
            str(row["agent_id"]), str(row["persona_id"]),
        )
        grouped[key][str(row["question_id"])] = float(row["value"])
        roles[key] = str(row["role"])
    return [
        {"scenario": key[0], "role": roles[key], "values": values}
        for key, values in grouped.items()
        if all(question_id in values for question_id in QUESTION_IDS[:7])
    ]


def _questionnaire_relationships(
    synthetic: list[dict[str, Any]], empirical: list[dict[str, Any]], *, draws: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    empirical_vectors = [
        row for row in empirical
        if row["role"] in ROLE_MAP.values()
        and all(row["values"].get(question_id) is not None for question_id in QUESTION_IDS[:7])
    ]
    role_counts = Counter(str(row["role"]) for row in empirical_vectors)
    normal_synthetic = [
        row for row in _synthetic_shift_vectors(synthetic)
        if row["scenario"] == "normal_load"
    ]
    role_standardized_synthetic = []
    for role, count in role_counts.items():
        pool = [row for row in normal_synthetic if row["role"] == role]
        if not pool:
            raise ValueError(f"No complete synthetic vectors for role {role}")
        role_standardized_synthetic.extend(
            pool[index % len(pool)] for index in range(count * 100)
        )
    if not empirical_vectors or not role_standardized_synthetic:
        raise ValueError("Complete empirical and synthetic vectors are required")

    rng = random.Random(20260801)
    relationship_rows = []
    for left_index, left_id in enumerate(QUESTION_IDS[:7]):
        for right_id in QUESTION_IDS[left_index + 1:7]:
            empirical_left = [float(row["values"][left_id]) for row in empirical_vectors]
            empirical_right = [float(row["values"][right_id]) for row in empirical_vectors]
            synthetic_left = [float(row["values"][left_id]) for row in role_standardized_synthetic]
            synthetic_right = [float(row["values"][right_id]) for row in role_standardized_synthetic]
            empirical_r = _pearson(empirical_left, empirical_right)
            synthetic_r = _pearson(synthetic_left, synthetic_right)
            bootstrap_r = []
            for _ in range(min(draws, 5000)):
                sampled = [rng.choice(empirical_vectors) for _index in empirical_vectors]
                value = _pearson(
                    [float(row["values"][left_id]) for row in sampled],
                    [float(row["values"][right_id]) for row in sampled],
                )
                if value is not None:
                    bootstrap_r.append(value)
            absolute_difference = (
                abs(float(synthetic_r) - float(empirical_r))
                if synthetic_r is not None and empirical_r is not None else None
            )
            relationship_rows.append(
                {
                    "left_item": left_id,
                    "right_item": right_id,
                    "empirical_n": len(empirical_vectors),
                    "empirical_r": empirical_r,
                    "empirical_r_95pct_low": _quantile(bootstrap_r, 0.025),
                    "empirical_r_95pct_high": _quantile(bootstrap_r, 0.975),
                    "synthetic_role_standardized_r": synthetic_r,
                    "absolute_correlation_difference": absolute_difference,
                    "sign_agreement": bool(
                        empirical_r is not None and synthetic_r is not None
                        and empirical_r * synthetic_r >= 0
                    ),
                    "empirically_informative": bool(
                        empirical_r is not None and abs(empirical_r) >= 0.30
                    ),
                }
            )

    comparable = [
        row for row in relationship_rows
        if row["absolute_correlation_difference"] is not None
    ]
    empirical_profile = [float(row["empirical_r"]) for row in comparable]
    synthetic_profile = [float(row["synthetic_role_standardized_r"]) for row in comparable]
    matrix_mae = statistics.mean(
        float(row["absolute_correlation_difference"]) for row in comparable
    )
    profile_correlation = _pearson(empirical_profile, synthetic_profile)
    core_pairs = {
        ("q1", "q2"), ("q4", "q5"), ("q4", "q7"),
        ("q5", "q7"), ("q6", "q7"),
    }
    core_rows = [
        row for row in comparable
        if (str(row["left_item"]), str(row["right_item"])) in core_pairs
    ]
    core_sign_count = sum(bool(row["sign_agreement"]) for row in core_rows)
    informative_rows = [row for row in comparable if row["empirically_informative"]]
    informative_sign_count = sum(
        bool(row["sign_agreement"]) for row in informative_rows
    )
    informative_sign_required = math.ceil(0.75 * len(informative_rows))
    summary = {
        "relationship_gate_pass": bool(
            matrix_mae <= 0.35
            and profile_correlation is not None
            and profile_correlation >= 0.50
            and core_sign_count >= 4
            and informative_sign_count >= informative_sign_required
        ),
        "correlation_matrix_mean_absolute_difference": matrix_mae,
        "correlation_profile_correlation": profile_correlation,
        "matrix_mae_ceiling": 0.35,
        "profile_correlation_floor": 0.50,
        "core_pair_sign_agreement_count": core_sign_count,
        "core_pair_count": len(core_rows),
        "core_pair_sign_agreement_required": 4,
        "informative_pair_sign_agreement_count": informative_sign_count,
        "informative_pair_count": len(informative_rows),
        "informative_pair_sign_agreement_required": informative_sign_required,
        "empirical_complete_shift_count": len(empirical_vectors),
        "small_empirical_sample_warning": len(empirical_vectors) < 30,
    }
    item_profiles: dict[str, dict[str, Any]] = {}
    for question_id in QUESTION_IDS[:7]:
        members = [
            row for row in comparable
            if question_id in {row["left_item"], row["right_item"]}
        ]
        mae = statistics.mean(
            float(row["absolute_correlation_difference"]) for row in members
        )
        informative = [row for row in members if row["empirically_informative"]]
        informative_required = math.ceil(0.75 * len(informative))
        informative_agreement = sum(
            bool(row["sign_agreement"]) for row in informative
        )
        item_profiles[question_id] = {
            "relationship_profile_mae": mae,
            "relationship_profile_pass": bool(
                mae <= 0.35
                and informative_agreement >= informative_required
            ),
            "informative_pair_count": len(informative),
            "informative_sign_agreement_count": informative_agreement,
            "informative_sign_agreement_required": informative_required,
        }
    return relationship_rows, summary, item_profiles


def _empirical_adequacy(
    verification: dict[str, Any],
    comparison: list[dict[str, Any]],
    item_diagnostics: list[dict[str, Any]],
    relationship_summary: dict[str, Any],
    relationship_item_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        by_question[str(row["question_id"])].append(row)
    normal_rows = [
        row for row in comparison
        if row["scenario"] == "normal_load"
        and row.get("value_convergence_evidence_eligible", row["question_id"] != "q2")
    ]
    distribution_items = {
        question_id: bool(
            any(
                row["scenario"] == "normal_load"
                and row["distribution_compatibility_pass"]
                for row in rows
            )
        )
        for question_id, rows in sorted(by_question.items())
    }
    high_load_reversal = {
        str(row["question_id"]): bool(float(row["high_minus_normal"] or 0.0) > 0.5)
        for row in item_diagnostics
        if str(row["question_id"]) in {"q3", "q4", "q5", "q6", "q7"}
    }
    per_item_rateability = verification.get("supported_rateability_rate_by_item") or {}
    mean_pass_count = sum(bool(row["mean_equivalence_pass"]) for row in normal_rows)
    mean_pass = len(normal_rows) == 6 and mean_pass_count >= 5
    severe_failure = any(row["severe_mean_failure"] for row in normal_rows)
    distribution_pass_count = sum(distribution_items.values())
    relationship_pass_count = sum(
        bool(row["relationship_profile_pass"])
        for row in relationship_item_profiles.values()
    )
    rateability_pass = all(
        per_item_rateability.get(question_id) is not None
        and float(per_item_rateability[question_id]) >= 0.90
        for question_id in QUESTION_IDS[:7]
    )
    gate = bool(
        mean_pass
        and not severe_failure
        and distribution_pass_count >= 5
        and relationship_pass_count >= 5
        and relationship_summary.get("relationship_gate_pass") is True
        and rateability_pass
    )
    return {
        "descriptive_convergence_gate_pass": gate,
        "mean_equivalence_omnibus_pass": mean_pass,
        "empirical_equivalence_scope": "normal-load synthetic shifts only",
        "high_load_scope": "directional stress sensitivity, not equivalence to the observed average-day sample",
        "mean_equivalence_pass_count": mean_pass_count,
        "mean_equivalence_check_count": len(normal_rows),
        "mean_equivalence_required_check_count": 5,
        "mean_equivalence_excluded_items": ["q2"],
        "q2_value_role": "anchored descriptive and relationship item; not value-convergence evidence",
        "severe_mean_failure_present": severe_failure,
        "distribution_compatibility_by_item": distribution_items,
        "distribution_compatibility_item_count": distribution_pass_count,
        "distribution_compatibility_required_item_count": 5,
        "distribution_compatibility_excluded_items": [],
        "relationship_profile_by_item": relationship_item_profiles,
        "relationship_profile_item_count": relationship_pass_count,
        "relationship_profile_required_item_count": 5,
        "relationship_summary": relationship_summary,
        "per_item_rateability_pass": rateability_pass,
        "high_load_improvement_over_half_point_by_item": high_load_reversal,
        "high_load_direction_is_descriptive_not_gated": True,
    }


def _compact_comparison_table(
    comparison: list[dict[str, Any]],
    relationship_item_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = {row["question_id"]: row["short_name"] for row in QUESTIONNAIRE_ITEMS}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in comparison:
        grouped[str(row["question_id"])][str(row["scenario"])] = row
    output = []
    for question_id in QUESTION_IDS[:7]:
        normal = grouped[question_id]["normal_load"]
        high = grouped[question_id]["high_load_high_acuity"]
        output.append(
            {
                "item": question_id.upper(),
                "construct": labels[question_id].replace("_", " "),
                "observed_mean": round(float(normal["empirical_mean"]), 2),
                "synthetic_normal_mean": round(
                    float(normal["synthetic_role_standardized_mean"]), 2
                ),
                "synthetic_high_load_mean": round(
                    float(high["synthetic_role_standardized_mean"]), 2
                ),
                "normal_absolute_difference": round(
                    float(normal["synthetic_absolute_mean_difference"]), 2
                ),
                "value_convergence_role": (
                    "mean anchored; distribution eligible" if question_id == "q2"
                    else "convergence evidence"
                ),
                "normal_mean_equivalence": normal["mean_equivalence_pass"],
                "normal_normalized_wasserstein": round(
                    float(normal["normalized_wasserstein_distance"]), 3
                ),
                "normal_distribution_compatibility": bool(
                    normal["distribution_compatibility_pass"]
                ),
                "relationship_profile_mae": round(
                    float(
                        relationship_item_profiles[question_id][
                            "relationship_profile_mae"
                        ]
                    ),
                    3,
                ),
                "relationship_profile_compatibility": bool(
                    relationship_item_profiles[question_id][
                        "relationship_profile_pass"
                    ]
                ),
                "high_minus_normal": round(
                    float(high["synthetic_role_standardized_mean"])
                    - float(normal["synthetic_role_standardized_mean"]),
                    2,
                ),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output-dir", required=True)
    empirical = parser.add_mutually_exclusive_group()
    empirical.add_argument("--empirical-csv")
    empirical.add_argument("--empirical-zip")
    parser.add_argument(
        "--empirical-member",
        default="ARIS_2/shadowing_with-participant-info.csv",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = _read_jsonl(Path(args.packets))
    responses = _read_jsonl(Path(args.responses))
    verification = json.loads(Path(args.verification).read_text())
    if verification.get("technical_verification_pass") is not True:
        raise SystemExit("Questionnaire responses did not pass technical verification")
    raw_long_rows = _long_rows(packets, responses)
    long_rows = _collapse_assessment_variants(raw_long_rows)
    summary_rows = _synthetic_summary(long_rows)
    item_diagnostics = _item_diagnostics(long_rows)
    _write_csv(
        output_dir / "questionnaire_response_assessment_raw.csv", raw_long_rows
    )
    _write_csv(output_dir / "questionnaire_response_long.csv", long_rows)
    _write_csv(output_dir / "synthetic_questionnaire_summary.csv", summary_rows)
    _write_csv(output_dir / "questionnaire_item_diagnostics.csv", item_diagnostics)

    empirical_comparison = []
    relationship_rows: list[dict[str, Any]] = []
    relationship_summary: dict[str, Any] = {}
    relationship_item_profiles: dict[str, dict[str, Any]] = {}
    empirical_shift_count = None
    empirical_matched_shift_count = None
    empirical_unmatched_role_counts: dict[str, int] = {}
    if args.empirical_csv or args.empirical_zip:
        empirical = _load_empirical_shifts(args)
        empirical_shift_count = len(empirical)
        empirical_matched_shift_count = sum(
            row["role"] in ROLE_MAP.values() for row in empirical
        )
        empirical_unmatched_role_counts = dict(
            sorted(
                Counter(
                    str(row.get("source_role") or "missing")
                    for row in empirical
                    if row["role"] not in ROLE_MAP.values()
                ).items()
            )
        )
        empirical_comparison = _bootstrap_comparison(
            long_rows, empirical, draws=max(1000, args.bootstrap_draws)
        )
        (
            relationship_rows,
            relationship_summary,
            relationship_item_profiles,
        ) = _questionnaire_relationships(
            long_rows, empirical, draws=max(1000, args.bootstrap_draws)
        )
        _write_csv(
            output_dir / "empirical_role_standardized_comparison.csv",
            empirical_comparison,
        )
        _write_csv(
            output_dir / "questionnaire_relationships.csv",
            relationship_rows,
        )
    empirical_adequacy = (
        _empirical_adequacy(
            verification,
            empirical_comparison,
            item_diagnostics,
            relationship_summary,
            relationship_item_profiles,
        )
        if empirical_comparison else {}
    )
    if empirical_comparison:
        _write_csv(
            output_dir / "questionnaire_convergence_table.csv",
            _compact_comparison_table(
                empirical_comparison, relationship_item_profiles
            ),
        )
    report = {
        "analysis_pass": True,
        "prespecified_scientific_gate_pass": verification.get("scientific_gate_pass"),
        "scientific_result": _is_scientific_result(
            verification, empirical_comparison, empirical_adequacy
        ),
        "empirical_adequacy": empirical_adequacy,
        "scientific_scope": "aggregate_baseline_convergence_and_intervention_plausibility_audit",
        "raw_assessment_response_row_count": len(raw_long_rows),
        "synthetic_response_row_count": len(long_rows),
        "assessment_aggregation": "one absolute-adequacy assessment per simulated shift; repeated assessment variants are not treated as independent observations",
        "empirical_comparison_completed": bool(empirical_comparison),
        "empirical_shift_count_read_in_memory": empirical_shift_count,
        "empirical_matched_shift_count": empirical_matched_shift_count,
        "empirical_unmatched_role_counts": empirical_unmatched_role_counts,
        "raw_empirical_rows_exported": False,
        "q8_primary_empirical_comparison": False,
        "q8_role": "internal acoustic negative control",
        "q6_scope": "operational helpfulness proxy, not clinical quality",
        "persona_weighting": "equal-weight designed-orientation mixture within each matched role",
        "comparison_row_count": len(empirical_comparison),
        "relationship_row_count": len(relationship_rows),
        "relationship_summary": relationship_summary,
        "relationship_results": relationship_rows,
        "item_diagnostics": item_diagnostics,
        "empirical_source_sha256": (
            hashlib.sha256(Path(args.empirical_zip or args.empirical_csv).read_bytes()).hexdigest()
            if empirical_comparison else None
        ),
        "empirical_mean_overlap_count": sum(
            bool(row["empirical_mean_within_synthetic_95pct"])
            for row in empirical_comparison
        ),
        "empirical_median_overlap_count": sum(
            bool(row["empirical_median_within_synthetic_95pct"])
            for row in empirical_comparison
        ),
        "comparison_results": empirical_comparison,
        "interpretation_boundary": (
            "Baseline convergence supports the aggregate plausibility of synthetic staff "
            "appraisals and, because the same grounded architecture generates the condition "
            "contrasts, lends bounded credibility to predicted intervention reactions. It "
            "does not directly validate unobserved counterfactual effects, establish "
            "individual-level prediction, validate the five designed orientations as human "
            "psychometric categories, or estimate their prevalence in the workforce."
        ),
    }
    (output_dir / "questionnaire_audit_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
