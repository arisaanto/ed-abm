#!/usr/bin/env python3
"""Tabulate trace-grounded Part 3 appraisals without treating them as testimony."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_SPEC = PROJECT_DIR / "manifests" / "part3_appraisal_analysis_spec.json"

SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
INTERVENTIONS = CONDITIONS[1:]
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metadata(packet: Mapping[str, Any]) -> dict[str, Any]:
    trace = packet.get("trace_evidence", {})
    source = trace.get("metadata", {})
    experience = trace.get("source_experience_map", {})
    profile = packet.get("user_model_profile", {})
    scenario = str(source.get("scenario_mode"))
    seed = int(source.get("seed"))
    assignment_round = int(experience.get("assignment_round", 0))
    persona_id = str(profile.get("persona_id"))
    role = str(packet.get("role"))
    agent_id = int(source.get("agent_id"))
    pair_id = "|".join(
        (
            scenario,
            str(seed),
            str(assignment_round),
            persona_id,
            role,
            str(agent_id),
        )
    )
    return {
        "prompt_id": packet["prompt_id"],
        "run_id": source.get("run_id"),
        "scenario": scenario,
        "condition": source.get("condition"),
        "seed": seed,
        "assignment_round": assignment_round,
        "agent_id": agent_id,
        "role": role,
        "persona_id": persona_id,
        "pair_id": pair_id,
        "not_human_data": True,
    }


def _mean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values]
    return statistics.fmean(selected) if selected else math.nan


def _paired_stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = _mean(values)
    sd = statistics.stdev(values) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    critical = T_975.get(n - 1, 1.96) if n > 1 else math.nan
    half_width = critical * se if n > 1 else math.nan
    return {
        "inference_unit_count": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "se_delta": se,
        "ci95_low": mean - half_width if n > 1 else math.nan,
        "ci95_high": mean + half_width if n > 1 else math.nan,
        "median_delta": statistics.median(values) if values else math.nan,
        "positive_unit_share": (
            sum(value > 0 for value in values) / n if n else math.nan
        ),
        "negative_unit_share": (
            sum(value < 0 for value in values) / n if n else math.nan
        ),
    }


def _survey_pair_records(
    survey_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, dict[str, Any]]]:
    pairs: dict[
        tuple[str, str, str, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in survey_rows:
        if row.get("rateability") != "rateable" or row.get("score_1_to_7") is None:
            continue
        key = (
            str(row["scenario"]),
            str(row["persona_id"]),
            str(row["dimension"]),
            str(row["pair_id"]),
        )
        condition = str(row["condition"])
        if condition in pairs[key]:
            raise ValueError(f"Duplicate survey pair member: {key}/{condition}")
        pairs[key][condition] = row
    return pairs


def _paired_survey_effects(
    survey_rows: list[dict[str, Any]],
    analysis_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pairs = _survey_pair_records(survey_rows)
    output = []
    for scenario in analysis_spec["scenarios"]:
        for persona in analysis_spec["personas"]:
            for dimension in analysis_spec["primary_dimensions"]:
                for intervention in analysis_spec["interventions"]:
                    members = [
                        row
                        for (pair_scenario, pair_persona, pair_dimension, _), row in pairs.items()
                        if pair_scenario == scenario
                        and pair_persona == persona
                        and pair_dimension == dimension
                        and "baseline" in row
                        and intervention in row
                    ]
                    deltas = [
                        float(row[intervention]["score_1_to_7"])
                        - float(row["baseline"]["score_1_to_7"])
                        for row in members
                    ]
                    deltas_by_seed: dict[int, list[float]] = defaultdict(list)
                    for member, delta in zip(members, deltas):
                        deltas_by_seed[int(member["baseline"]["seed"])].append(delta)
                    seed_mean_deltas = [
                        _mean(seed_deltas)
                        for _, seed_deltas in sorted(deltas_by_seed.items())
                    ]
                    seed_stats = _paired_stats(seed_mean_deltas)
                    role_means = {}
                    for role in analysis_spec["roles"]:
                        role_deltas = [
                            float(row[intervention]["score_1_to_7"])
                            - float(row["baseline"]["score_1_to_7"])
                            for row in members
                            if row["baseline"]["role"] == role
                        ]
                        role_means[role] = _mean(role_deltas)
                    overall_sign = 1 if _mean(deltas) > 0 else -1 if _mean(deltas) < 0 else 0
                    role_direction_agreement = sum(
                        math.isfinite(value)
                        and (1 if value > 0 else -1 if value < 0 else 0) == overall_sign
                        for value in role_means.values()
                    )
                    output.append(
                        {
                            "scenario": scenario,
                            "persona_id": persona,
                            "dimension": dimension,
                            "preferred_direction": analysis_spec[
                                "dimension_directions"
                            ][dimension],
                            "contrast": f"{intervention} - baseline",
                            "baseline_mean": _mean(
                                float(row["baseline"]["score_1_to_7"])
                                for row in members
                            ),
                            "intervention_mean": _mean(
                                float(row[intervention]["score_1_to_7"])
                                for row in members
                            ),
                            "paired_shift_count": len(deltas),
                            "seed_count": seed_stats["inference_unit_count"],
                            "mean_delta": seed_stats["mean_delta"],
                            "sd_delta_across_seed_means": seed_stats["sd_delta"],
                            "se_delta_across_seed_means": seed_stats["se_delta"],
                            "ci95_low": seed_stats["ci95_low"],
                            "ci95_high": seed_stats["ci95_high"],
                            "median_seed_mean_delta": seed_stats["median_delta"],
                            "positive_seed_share": seed_stats["positive_unit_share"],
                            "negative_seed_share": seed_stats["negative_unit_share"],
                            "role_direction_agreement_count": role_direction_agreement,
                            "role_direction_denominator": len(analysis_spec["roles"]),
                            "coordination_nurse_mean_delta": role_means.get(
                                "CoordinationNurse", math.nan
                            ),
                            "nurse_mean_delta": role_means.get("Nurse", math.nan),
                            "doctor_mean_delta": role_means.get("Doctor", math.nan),
                            "claim_boundary": (
                                "Paired synthetic appraisal difference; not a human "
                                "preference estimate."
                            ),
                        }
                    )
    return output


def _polarization_rows(
    survey_rows: list[dict[str, Any]], analysis_spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in survey_rows:
        if row.get("rateability") != "rateable" or row.get("score_1_to_7") is None:
            continue
        grouped[
            (str(row["scenario"]), str(row["condition"]), str(row["dimension"]))
        ][str(row["persona_id"])].append(float(row["score_1_to_7"]))

    output = []
    for scenario in analysis_spec["scenarios"]:
        for dimension in analysis_spec["primary_dimensions"]:
            baseline_means = [
                _mean(grouped[(scenario, "baseline", dimension)].get(persona, []))
                for persona in analysis_spec["personas"]
            ]
            baseline_means = [value for value in baseline_means if math.isfinite(value)]
            baseline_sd = statistics.stdev(baseline_means) if len(baseline_means) > 1 else math.nan
            baseline_range = max(baseline_means) - min(baseline_means) if baseline_means else math.nan
            for condition in analysis_spec["conditions"]:
                persona_means = [
                    _mean(grouped[(scenario, condition, dimension)].get(persona, []))
                    for persona in analysis_spec["personas"]
                ]
                persona_means = [value for value in persona_means if math.isfinite(value)]
                sd = statistics.stdev(persona_means) if len(persona_means) > 1 else math.nan
                span = max(persona_means) - min(persona_means) if persona_means else math.nan
                output.append(
                    {
                        "scenario": scenario,
                        "condition": condition,
                        "dimension": dimension,
                        "persona_count": len(persona_means),
                        "between_persona_mean": _mean(persona_means),
                        "between_persona_sd": sd,
                        "between_persona_range": span,
                        "sd_change_from_baseline": sd - baseline_sd,
                        "range_change_from_baseline": span - baseline_range,
                        "interpretation": (
                            "Descriptive dispersion among five designed orientations, "
                            "not population psychological polarization."
                        ),
                    }
                )
    return output


def _worst_case_fit_rows(
    effect_rows: list[dict[str, Any]], analysis_spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for scenario in analysis_spec["scenarios"]:
        for intervention in analysis_spec["interventions"]:
            contrast = f"{intervention} - baseline"
            rows = [
                row
                for row in effect_rows
                if row["scenario"] == scenario
                and row["contrast"] == contrast
                and row["dimension"] == "overall_person_space_fit"
            ]
            if not rows:
                continue
            worst = min(rows, key=lambda row: float(row["mean_delta"]))
            best = max(rows, key=lambda row: float(row["mean_delta"]))
            output.append(
                {
                    "scenario": scenario,
                    "contrast": contrast,
                    "persona_count": len(rows),
                    "personas_with_positive_mean_delta": sum(
                        float(row["mean_delta"]) > 0 for row in rows
                    ),
                    "personas_with_negative_mean_delta": sum(
                        float(row["mean_delta"]) < 0 for row in rows
                    ),
                    "worst_fit_persona": worst["persona_id"],
                    "worst_fit_mean_delta": worst["mean_delta"],
                    "worst_fit_ci95_low": worst["ci95_low"],
                    "best_fit_persona": best["persona_id"],
                    "best_fit_mean_delta": best["mean_delta"],
                    "interpretation": (
                        "Worst observed synthetic orientation response; not a safety "
                        "or human acceptability threshold."
                    ),
                }
            )
    return output


def _rounded(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, 6) if isinstance(value, float) and math.isfinite(value) else ""
        if isinstance(value, float)
        else value
        for key, value in row.items()
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packets).resolve()
    response_path = Path(args.responses).resolve()
    verification_path = Path(args.verification).resolve()
    output_dir = Path(args.output_dir).resolve()
    analysis_spec_path = Path(
        getattr(args, "analysis_spec", None) or DEFAULT_ANALYSIS_SPEC
    ).resolve()
    analysis_spec = json.loads(analysis_spec_path.read_text())
    verification = json.loads(verification_path.read_text())
    if verification.get("technical_verification_pass") is not True:
        raise ValueError("Appraisal response verification has not passed")

    packets = {
        str(row["prompt_id"]): row for row in _read_jsonl(packet_path)
    }
    responses = {
        str(row["prompt_id"]): row for row in _read_jsonl(response_path)
    }
    if set(packets) != set(responses):
        raise ValueError("Appraisal packet and response IDs do not match exactly")

    survey_rows = []
    interview_rows = []
    interview_prompt_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt_id, packet in sorted(packets.items()):
        response = responses[prompt_id].get("response", {})
        base = _metadata(packet)
        if packet.get("packet_type") == "end_of_shift_survey_bundle":
            for row in response.get("responses", []):
                survey_rows.append(
                    {
                        **base,
                        "dimension": row.get("dimension"),
                        "rateability": row.get("rateability"),
                        "score_1_to_7": row.get("score_1_to_7"),
                        "short_rationale": row.get("short_rationale"),
                        "confidence_1_to_5": row.get("confidence_1_to_5"),
                        "uncertainty_note": row.get("uncertainty_note"),
                        "evidence_event_ids": "|".join(row.get("evidence_event_ids", [])),
                    }
                )
        elif packet.get("packet_type") == "end_of_shift_interview_bundle":
            for row in response.get("answers", []):
                rendered = {
                    **base,
                    "question_id": row.get("question_id"),
                    "answer": row.get("answer"),
                    "grounded_pattern": row.get("grounded_pattern"),
                    "persona_conditioned_interpretation": row.get(
                        "persona_conditioned_interpretation"
                    ),
                    "latent_need": row.get("latent_need"),
                    "design_hypothesis": row.get("design_hypothesis"),
                    "tradeoff": row.get("tradeoff"),
                    "uncertainty_note": row.get("uncertainty_note"),
                    "evidence_event_ids": "|".join(row.get("evidence_event_ids", [])),
                }
                interview_rows.append(rendered)
                interview_prompt_rows[prompt_id].append(rendered)

    survey_fields = [
        "prompt_id", "run_id", "scenario", "condition", "seed",
        "assignment_round", "agent_id", "pair_id",
        "role", "persona_id", "dimension", "rateability", "score_1_to_7",
        "short_rationale", "confidence_1_to_5", "uncertainty_note",
        "evidence_event_ids",
        "not_human_data",
    ]
    interview_fields = [
        "prompt_id", "run_id", "scenario", "condition", "seed",
        "assignment_round", "agent_id", "pair_id",
        "role", "persona_id", "question_id", "answer", "grounded_pattern",
        "persona_conditioned_interpretation", "latent_need", "design_hypothesis",
        "tradeoff", "uncertainty_note", "evidence_event_ids", "not_human_data",
    ]
    _write_csv(output_dir / "survey_responses.csv", survey_rows, survey_fields)
    _write_csv(output_dir / "interview_responses.csv", interview_rows, interview_fields)

    paired_effects = _paired_survey_effects(survey_rows, analysis_spec)
    polarization = _polarization_rows(survey_rows, analysis_spec)
    worst_case_fit = _worst_case_fit_rows(paired_effects, analysis_spec)
    paired_fields = list(paired_effects[0]) if paired_effects else [
        "scenario", "persona_id", "dimension", "contrast", "paired_shift_count",
        "seed_count",
        "mean_delta", "ci95_low", "ci95_high",
    ]
    polarization_fields = list(polarization[0]) if polarization else [
        "scenario", "condition", "dimension", "persona_count",
        "between_persona_sd", "between_persona_range",
    ]
    worst_case_fields = list(worst_case_fit[0]) if worst_case_fit else [
        "scenario", "contrast", "persona_count", "worst_fit_persona",
        "worst_fit_mean_delta", "best_fit_persona", "best_fit_mean_delta",
    ]
    _write_csv(
        output_dir / "paired_survey_effects.csv",
        [_rounded(row) for row in paired_effects],
        paired_fields,
    )
    _write_csv(
        output_dir / "between_persona_polarization.csv",
        [_rounded(row) for row in polarization],
        polarization_fields,
    )
    _write_csv(
        output_dir / "worst_case_person_space_fit.csv",
        [_rounded(row) for row in worst_case_fit],
        worst_case_fields,
    )

    # One full interview per scenario-condition-persona cell (40 bundles) is
    # selected deterministically for mandatory human grounding/naturalness review.
    review_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for prompt_id, rows in interview_prompt_rows.items():
        first = rows[0]
        review_groups[
            (str(first["scenario"]), str(first["condition"]), str(first["persona_id"]))
        ].append(prompt_id)
    review_prompt_ids = {
        min(
            prompt_ids,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )
        for prompt_ids in review_groups.values()
    }
    review_rows = [
        {
            **row,
            "review_grounding_pass": "",
            "review_naturalness_1_to_3": "",
            "review_direct_answer_pass": "",
            "review_context_labeling_pass": "",
            "review_claim_layer_pass": "",
            "review_tradeoff_pass": "",
            "review_notes": "",
            "include_in_explorer": "",
            "explorer_display_order": "",
        }
        for prompt_id in sorted(review_prompt_ids)
        for row in interview_prompt_rows[prompt_id]
    ]
    _write_csv(
        output_dir / "interview_manual_review_sample.csv",
        review_rows,
        interview_fields
        + [
            "review_grounding_pass",
            "review_naturalness_1_to_3",
            "review_direct_answer_pass",
            "review_context_labeling_pass",
            "review_claim_layer_pass",
            "review_tradeoff_pass",
            "review_notes",
            "include_in_explorer",
            "explorer_display_order",
        ],
    )

    coding_rows = [
        {
            **row,
            "primary_theme": "",
            "secondary_theme": "",
            "spatial_feature": "",
            "workplace_consequence": "",
            "orientation_priority": "",
            "design_hypothesis_category": "",
            "grounding_status": "",
            "coder_notes": "",
        }
        for row in interview_rows
    ]
    coding_fields = interview_fields + [
        "primary_theme",
        "secondary_theme",
        "spatial_feature",
        "workplace_consequence",
        "orientation_priority",
        "design_hypothesis_category",
        "grounding_status",
        "coder_notes",
    ]
    _write_csv(
        output_dir / "interview_manual_coding_template.csv",
        coding_rows,
        coding_fields,
    )

    rateability = Counter(str(row["rateability"]) for row in survey_rows)
    complete_effect_rows = [
        row
        for row in paired_effects
        if int(row["paired_shift_count"]) > 0 and int(row["seed_count"]) >= 2
    ]
    seed_counts = [int(row["seed_count"]) for row in complete_effect_rows]
    expected_effect_rows_per_dimension = (
        len(analysis_spec["scenarios"])
        * len(analysis_spec["personas"])
        * len(analysis_spec["interventions"])
    )
    dimension_completeness = {}
    for dimension in analysis_spec["primary_dimensions"]:
        dimension_rows = [
            row for row in paired_effects if row["dimension"] == dimension
        ]
        complete_dimension_rows = [
            row
            for row in dimension_rows
            if int(row["paired_shift_count"]) > 0 and int(row["seed_count"]) >= 2
        ]
        dimension_rateability = Counter(
            str(row["rateability"])
            for row in survey_rows
            if row["dimension"] == dimension
        )
        complete_count = len(complete_dimension_rows)
        if complete_count == expected_effect_rows_per_dimension:
            status = "fully_estimable"
        elif complete_count == 0:
            status = "non_estimable"
        else:
            status = "partially_estimable"
        dimension_completeness[dimension] = {
            "status": status,
            "expected_paired_effect_row_count": expected_effect_rows_per_dimension,
            "complete_paired_effect_row_count": complete_count,
            "incomplete_paired_effect_row_count": len(dimension_rows) - complete_count,
            "rateable_response_count": dimension_rateability.get("rateable", 0),
            "insufficient_evidence_response_count": dimension_rateability.get(
                "insufficient_evidence", 0
            ),
        }
    fully_estimable_dimensions = [
        dimension
        for dimension, audit in dimension_completeness.items()
        if audit["status"] == "fully_estimable"
    ]
    partially_estimable_dimensions = [
        dimension
        for dimension, audit in dimension_completeness.items()
        if audit["status"] == "partially_estimable"
    ]
    non_estimable_dimensions = [
        dimension
        for dimension, audit in dimension_completeness.items()
        if audit["status"] == "non_estimable"
    ]
    summary = {
        "analysis_pass": bool(survey_rows and interview_rows),
        "packet_count": len(packets),
        "survey_bundle_count": sum(
            packet.get("packet_type") == "end_of_shift_survey_bundle"
            for packet in packets.values()
        ),
        "interview_bundle_count": len(interview_prompt_rows),
        "survey_response_row_count": len(survey_rows),
        "interview_answer_row_count": len(interview_rows),
        "survey_rateability_counts": dict(sorted(rateability.items())),
        "manual_review_bundle_count": len(review_prompt_ids),
        "manual_review_answer_count": len(review_rows),
        "manual_review_required": True,
        "manual_review_scope": (
            "One complete interview per scenario-condition-persona cell; review "
            "grounding, naturalness, direct answers, unobtrusive context framing, "
            "conjecture labeling, and design tradeoffs."
        ),
        "automated_theme_inference_performed": False,
        "analysis_spec": str(analysis_spec_path),
        "analysis_spec_version": analysis_spec["spec_version"],
        "paired_analysis_ready": len(complete_effect_rows) == len(paired_effects),
        "paired_analysis_ready_for_fully_estimable_dimensions": bool(
            fully_estimable_dimensions
        ),
        "dimension_completeness": dimension_completeness,
        "fully_estimable_dimensions": fully_estimable_dimensions,
        "partially_estimable_dimensions": partially_estimable_dimensions,
        "non_estimable_dimensions": non_estimable_dimensions,
        "paired_survey_effect_row_count": len(paired_effects),
        "complete_paired_survey_effect_row_count": len(complete_effect_rows),
        "incomplete_paired_survey_effect_row_count": (
            len(paired_effects) - len(complete_effect_rows)
        ),
        "minimum_seed_count_per_complete_effect": min(seed_counts) if seed_counts else 0,
        "polarization_row_count": len(polarization),
        "worst_case_fit_row_count": len(worst_case_fit),
        "primary_inferential_unit": analysis_spec["primary_inferential_unit"],
        "primary_uncertainty": analysis_spec["uncertainty_method"],
        "output_files": [
            "survey_responses.csv",
            "interview_responses.csv",
            "paired_survey_effects.csv",
            "between_persona_polarization.csv",
            "worst_case_person_space_fit.csv",
            "interview_manual_review_sample.csv",
            "interview_manual_coding_template.csv",
        ],
        "human_testimony": False,
        "claim_boundary": (
            "Synthetic persona-conditioned appraisals of simulated experience; "
            "not Zurich ED staff preferences or psychological measurements."
        ),
    }
    (output_dir / "appraisal_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--analysis-spec",
        default=str(DEFAULT_ANALYSIS_SPEC),
        help="Prespecified appraisal estimands and manual-review contract.",
    )
    return parser.parse_args()


def main() -> None:
    result = analyze(parse_args())
    print(json.dumps(result, indent=2))
    if not result["analysis_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
