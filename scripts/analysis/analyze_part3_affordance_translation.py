#!/usr/bin/env python3
"""Join Part 3 exposure, behavior, ecology, and appraisal effects.

This CPU-only analysis reads completed main-study and appraisal summaries. It
does not rerun the ABM or call an LLM. The output is a transparent relational
account of how a spatial intervention is encountered, taken up, and appraised
by each designed cognitive orientation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable


SCENARIO_LABELS = {
    "normal_load": "Normal load",
    "high_load_high_acuity": "High load",
    "Normal load": "Normal load",
    "High load": "High load",
}
PERSONA_LABELS = {
    "adaptive_generalist": "Adaptive Generalist",
    "focus_protector": "Focus Protector",
    "patient_advocate": "Patient Advocate",
    "team_connector": "Team Connector",
    "vigilant_monitor": "Vigilant Monitor",
}
CONTRAST_LABELS = {
    "cockpit_only - baseline": "COCPIT only - Baseline",
    "nursta_only - baseline": "NURSTA only - Baseline",
    "both - baseline": "Both - Baseline",
    "COCPIT only - Baseline": "COCPIT only - Baseline",
    "NURSTA only - Baseline": "NURSTA only - Baseline",
    "Both - Baseline": "Both - Baseline",
}

MAIN_OUTCOMES = {
    "Share of time with a mutually visible colleague": (
        "encountered_affordance",
        "Mutual staff visibility exposure",
        "percentage points",
        100.0,
        "Higher means more of the shift was spent with at least one mutually visible colleague.",
    ),
    "Movement distance per agent-hour": (
        "workflow_exposure",
        "Movement distance",
        "meters per agent-hour",
        1.0,
        "Describes modeled movement demand; it is not a measure of speed or efficiency.",
    ),
    "Role-standardized model engagement-rate change": (
        "behavioral_uptake",
        "Role-standardized engagement rate",
        "percentage points",
        100.0,
        "Persona-conditioned uptake of feasible optional contacts, standardized across roles.",
    ),
    "Interaction participations per agent-hour": (
        "realized_ecology",
        "Interaction participations",
        "participations per agent-hour",
        1.0,
        "Realized staff interaction participations; not a clinical outcome.",
    ),
}
APPRAISAL_DIMENSIONS = {
    "team_awareness": ("appraised_fit", "Team awareness", "points on 1-7 scale", 1.0),
    "spatial_legibility": ("appraised_fit", "Spatial legibility", "points on 1-7 scale", 1.0),
    "overall_person_space_fit": (
        "appraised_fit",
        "Overall person-space fit",
        "points on 1-7 scale",
        1.0,
    ),
}

CONSTRUCT_BOUNDARIES = {
    "team_awareness": (
        "Mutual visibility and realized staff-contact evidence",
        "Rateable when evidence shows recurring colleague visibility or coordination.",
        "primary",
    ),
    "interruption_burden": (
        "Optional-contact decisions, timing, and task-transition evidence",
        "Synthetic appraisal of modeled interruption exposure, not subjective workload.",
        "primary",
    ),
    "task_continuity": (
        "Task, mode, travel, and transition evidence",
        "Describes modeled continuity; it does not establish human cognitive flow.",
        "primary",
    ),
    "patient_accessibility": (
        "Patient-facing tasks, destinations, and interaction evidence",
        "Accessibility within the modeled workflow, not patient-reported access.",
        "primary",
    ),
    "privacy_and_control": (
        "Optional-contact acceptance and exposure evidence",
        "Contact autonomy is partly observable; acoustic and visual confidentiality are not logged.",
        "exclude_from_primary_inference",
    ),
    "spatial_legibility": (
        "Place recurrence, transitions, travel, and visibility evidence",
        "Synthetic interpretation of modeled navigation patterns, not a human wayfinding score.",
        "primary",
    ),
    "overall_person_space_fit": (
        "Cross-domain experience evidence",
        "Exploratory synthetic appraisal; not a validated psychometric scale.",
        "exploratory_primary",
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _label_persona(value: str) -> str:
    return PERSONA_LABELS.get(value, value)


def _main_rows(main_dir: Path) -> list[dict[str, Any]]:
    paths = (
        main_dir / "paired_persona_effects.csv",
        main_dir / "paired_persona_experience_effects.csv",
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        for source in _read_csv(path):
            definition = MAIN_OUTCOMES.get(source["outcome"])
            if definition is None:
                continue
            stage, label, unit, multiplier, boundary = definition
            rows.append(
                {
                    "scenario": SCENARIO_LABELS[source["scenario"]],
                    "persona": _label_persona(source["persona"]),
                    "contrast": CONTRAST_LABELS[source["contrast"]],
                    "stage": stage,
                    "outcome": label,
                    "unit": unit,
                    "n_paired_seeds": int(source["n_paired_seeds"]),
                    "mean_delta": round(float(source["mean_delta"]) * multiplier, 6),
                    "ci95_low": round(float(source["ci95_low"]) * multiplier, 6),
                    "ci95_high": round(float(source["ci95_high"]) * multiplier, 6),
                    "preferred_direction": "context_dependent",
                    "claim_boundary": boundary,
                    "source_table": path.name,
                }
            )
    return rows


def _appraisal_rows(appraisal_dir: Path) -> list[dict[str, Any]]:
    path = appraisal_dir / "paired_survey_effects.csv"
    rows = []
    for source in _read_csv(path):
        definition = APPRAISAL_DIMENSIONS.get(source["dimension"])
        if definition is None:
            continue
        stage, label, unit, multiplier = definition
        rows.append(
            {
                "scenario": SCENARIO_LABELS[source["scenario"]],
                "persona": _label_persona(source["persona_id"]),
                "contrast": CONTRAST_LABELS[source["contrast"]],
                "stage": stage,
                "outcome": label,
                "unit": unit,
                "n_paired_seeds": int(source["seed_count"]),
                "mean_delta": round(float(source["mean_delta"]) * multiplier, 6),
                "ci95_low": round(float(source["ci95_low"]) * multiplier, 6),
                "ci95_high": round(float(source["ci95_high"]) * multiplier, 6),
                "preferred_direction": source["preferred_direction"],
                "claim_boundary": source["claim_boundary"],
                "source_table": path.name,
            }
        )
    return rows


def construct_coverage(survey_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in survey_rows:
        grouped[row["dimension"]].append(row)
    result = []
    for dimension, (evidence, boundary, status) in CONSTRUCT_BOUNDARIES.items():
        rows = grouped.get(dimension, [])
        rateable = sum(row["rateability"] == "rateable" for row in rows)
        insufficient = sum(row["rateability"] != "rateable" for row in rows)
        total = len(rows)
        result.append(
            {
                "dimension": dimension,
                "rateable_count": rateable,
                "insufficient_evidence_count": insufficient,
                "total_count": total,
                "rateable_share": round(rateable / total, 6) if total else 0.0,
                "supporting_trace_channels": evidence,
                "claim_boundary": boundary,
                "analysis_status": status,
            }
        )
    return result


def interaction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["contrast"], row["outcome"])].append(row)
    result = []
    for (scenario, contrast, outcome), members in sorted(grouped.items()):
        values = [(row["persona"], float(row["mean_delta"])) for row in members]
        minimum = min(values, key=lambda item: item[1])
        maximum = max(values, key=lambda item: item[1])
        result.append(
            {
                "scenario": scenario,
                "contrast": contrast,
                "outcome": outcome,
                "persona_count": len(values),
                "minimum_persona": minimum[0],
                "minimum_delta": minimum[1],
                "maximum_persona": maximum[0],
                "maximum_delta": maximum[1],
                "between_persona_range": round(maximum[1] - minimum[1], 6),
                "sign_divergence": any(value < 0 for _, value in values)
                and any(value > 0 for _, value in values),
                "interpretation": (
                    "Descriptive persona-by-environment heterogeneity in designed "
                    "synthetic orientations; not a population personality estimate."
                ),
            }
        )
    return result


def rule_reference_comparison(main_results_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(main_results_root.rglob("part3_provider_inference.jsonl")):
        records.extend(_read_jsonl(path))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped[("overall", "all")] = records
    for field in ("scenario_mode", "condition", "role"):
        values = sorted(
            {
                str(row["packet"]["trace_evidence"]["metadata"][field])
                for row in records
            }
        )
        for value in values:
            grouped[(field, value)] = [
                row
                for row in records
                if str(row["packet"]["trace_evidence"]["metadata"][field]) == value
            ]
    persona_values = sorted(
        {
            str(row["packet"]["user_model_profile"]["persona_id"])
            for row in records
        }
    )
    for value in persona_values:
        grouped[("persona", value)] = [
            row
            for row in records
            if str(row["packet"]["user_model_profile"]["persona_id"]) == value
        ]

    result = []
    for (scope, value), members in grouped.items():
        model_actions = [row["result"]["response"]["selected_action"] for row in members]
        rule_actions = [
            row["packet"]["trace_evidence"]["rule_reference"]["selected_action"]
            for row in members
        ]
        count = len(members)
        result.append(
            {
                "scope": scope,
                "scope_value": _label_persona(value) if scope == "persona" else value,
                "decision_count": count,
                "action_agreement_rate": round(
                    sum(model == rule for model, rule in zip(model_actions, rule_actions))
                    / count,
                    6,
                ),
                "qwen_engage_rate": round(
                    sum(action == "engage" for action in model_actions) / count, 6
                ),
                "rule_engage_rate": round(
                    sum(action == "engage" for action in rule_actions) / count, 6
                ),
                "engage_rate_difference": round(
                    (
                        sum(action == "engage" for action in model_actions)
                        - sum(action == "engage" for action in rule_actions)
                    )
                    / count,
                    6,
                ),
                "claim_boundary": (
                    "Descriptive comparison on opportunities sampled for Qwen; "
                    "agreement is not a validity target and the rule policy has no "
                    "persona construct."
                ),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    main_dir = Path(args.main_analysis_dir).resolve()
    appraisal_dir = Path(args.appraisal_analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    translation = _main_rows(main_dir) + _appraisal_rows(appraisal_dir)
    coverage = construct_coverage(_read_csv(appraisal_dir / "survey_responses.csv"))
    interactions = interaction_summary(translation)
    rule_comparison = rule_reference_comparison(
        Path(args.main_results_root).resolve()
    )

    translation_fields = [
        "scenario", "persona", "contrast", "stage", "outcome", "unit",
        "n_paired_seeds", "mean_delta", "ci95_low", "ci95_high",
        "preferred_direction", "claim_boundary", "source_table",
    ]
    interaction_fields = [
        "scenario", "contrast", "outcome", "persona_count", "minimum_persona",
        "minimum_delta", "maximum_persona", "maximum_delta",
        "between_persona_range", "sign_divergence", "interpretation",
    ]
    coverage_fields = [
        "dimension", "rateable_count", "insufficient_evidence_count", "total_count",
        "rateable_share", "supporting_trace_channels", "claim_boundary", "analysis_status",
    ]
    rule_fields = [
        "scope", "scope_value", "decision_count", "action_agreement_rate",
        "qwen_engage_rate", "rule_engage_rate", "engage_rate_difference",
        "claim_boundary",
    ]
    _write_csv(output_dir / "affordance_translation.csv", translation, translation_fields)
    _write_csv(
        output_dir / "persona_environment_interactions.csv",
        interactions,
        interaction_fields,
    )
    _write_csv(output_dir / "construct_evidence_coverage.csv", coverage, coverage_fields)
    _write_csv(
        output_dir / "rule_reference_comparison.csv", rule_comparison, rule_fields
    )
    summary = {
        "analysis_pass": bool(translation) and bool(coverage),
        "abm_rerun": False,
        "llm_run": False,
        "translation_row_count": len(translation),
        "interaction_summary_row_count": len(interactions),
        "construct_count": len(coverage),
        "rule_reference_comparison_row_count": len(rule_comparison),
        "primary_privacy_inference_allowed": False,
        "privacy_boundary": CONSTRUCT_BOUNDARIES["privacy_and_control"][1],
        "claim": (
            "Completed outputs support a modeled affordance-translation analysis: "
            "spatial exposure can be compared with persona-conditioned uptake and "
            "synthetic appraisal, without treating personas as human personality types."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "affordance_translation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-analysis-dir", required=True)
    parser.add_argument("--main-results-root", required=True)
    parser.add_argument("--appraisal-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))
    if not summary["analysis_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
