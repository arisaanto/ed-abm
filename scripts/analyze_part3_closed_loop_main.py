#!/usr/bin/env python3
"""Analyze the paired Part 3 closed-loop main study.

The seed is the inferential unit. The five persona-assignment rounds balance
personas across staff roles within each scenario-condition-seed cell; they are
not treated as five independent replications. Model decisions are summarized
within seed before condition contrasts are computed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
INTERVENTIONS = CONDITIONS[1:]
PERSONAS = (
    "adaptive_generalist",
    "focus_protector",
    "patient_advocate",
    "team_connector",
    "vigilant_monitor",
)
ROLES = ("CoordinationNurse", "Nurse", "Doctor")
# The fixed-composition estimator mirrors the simulated staff complement:
# one coordination nurse, five nurses, and three doctors. It prevents a
# condition-specific mix of sampled roles from masquerading as a persona effect.
ROLE_WEIGHTS = {
    "CoordinationNurse": 1.0 / 9.0,
    "Nurse": 5.0 / 9.0,
    "Doctor": 3.0 / 9.0,
}
MACRO_METRICS = {
    "f2f_per_hour": "F2F interactions per hour",
    "hcw_hcw_share": "Staff-staff interaction share",
    "patient_facing_share": "Patient-facing interaction share",
    "corridor_share": "Corridor interaction share",
    "bedside_or_patient_room_share": "Bedside/patient-room interaction share",
}
EXPERIENCE_METRICS = {
    "movement_m_per_agent_hour": "Movement distance per agent-hour",
    "mean_mutually_visible_colleagues": "Mean mutually visible colleagues",
    "share_time_any_visible_colleague": "Share of time with a mutually visible colleague",
    "eligible_cognitive_opportunities_per_agent_hour": "Eligible cognitive opportunities per agent-hour",
    "sampled_cognitive_opportunities_per_agent_hour": "Model-sampled opportunities per agent-hour",
    "model_engage_decisions_per_agent_hour": "Model engage decisions per agent-hour",
    "model_realized_initiated_interactions_per_agent_hour": "Model-realized initiated interactions per agent-hour",
    "interaction_participations_per_agent_hour": "Interaction participations per agent-hour",
    "staff_staff_participations_per_agent_hour": "Staff-staff participations per agent-hour",
    "patient_facing_participations_per_agent_hour": "Patient-facing participations per agent-hour",
    "mean_unique_staff_partners": "Mean unique staff partners",
    "model_engage_rate": "Model engagement rate",
    "model_realization_yield": "Realized interactions per model engage decision",
    "memory_retrievals_per_model_decision": "Grounded memory retrievals per model decision",
}
SCENARIO_LABELS = {
    "normal_load": "Normal load",
    "high_load_high_acuity": "High load",
}
CONDITION_LABELS = {
    "baseline": "Baseline",
    "cockpit_only": "COCPIT only",
    "nursta_only": "NURSTA only",
    "both": "Both",
}
PERSONA_LABELS = {
    "adaptive_generalist": "Adaptive Generalist",
    "focus_protector": "Focus Protector",
    "patient_advocate": "Patient Advocate",
    "team_connector": "Team Connector",
    "vigilant_monitor": "Vigilant Monitor",
}

# Two-sided 95% Student-t critical values. The main design uses df=9; the
# complete table keeps the analyzer useful for bounded follow-up designs.
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
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _rate(count: int, total: int) -> float:
    return count / total if total else math.nan


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a descriptive action proportion."""

    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _exact_sign_flip_p(deltas: Sequence[float]) -> float:
    """Exact two-sided paired randomization p-value for a mean delta."""

    values = _finite(deltas)
    if not values:
        return math.nan
    observed = abs(_mean(values))
    extreme = 0
    total = 1 << len(values)
    tolerance = 1e-12
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(_mean([sign * value for sign, value in zip(signs, values)]))
        extreme += permuted + tolerance >= observed
    return extreme / total


def _paired_statistics(deltas: Sequence[float]) -> dict[str, float | int]:
    values = _finite(deltas)
    n = len(values)
    mean = _mean(values)
    sd = statistics.stdev(values) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    critical = T_975.get(n - 1, 1.96) if n > 1 else math.nan
    half_width = critical * se if n > 1 else math.nan
    return {
        "n_paired_seeds": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "se_delta": se,
        "ci95_low": mean - half_width if n > 1 else math.nan,
        "ci95_high": mean + half_width if n > 1 else math.nan,
        "exact_sign_flip_p": _exact_sign_flip_p(values),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_row(row: Mapping[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = "" if not math.isfinite(value) else round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _assignment_round(run_dir: Path) -> int:
    name = run_dir.name
    if not name.startswith("assignment_round_"):
        raise ValueError(f"Cannot parse assignment round from {run_dir}")
    return int(name.rsplit("_", 1)[-1])


def _diversion_band(value: float) -> str:
    if value < 1.0 / 3.0:
        return "Low"
    if value < 2.0 / 3.0:
        return "Moderate"
    return "High"


def _collect_runs(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.rglob("summary.json")):
        summary = _read_json(summary_path)
        metadata = summary["metadata"]
        validation = summary["validation_metrics"]
        row: dict[str, Any] = {
            "scenario": metadata["scenario_mode"],
            "condition": metadata["condition"],
            "seed": int(metadata["seed"]),
            "assignment_round": _assignment_round(summary_path.parent),
            "workflow_status": summary["workflow_health"][
                "workflow_health_status"
            ],
            "provider_calls": int(
                summary.get("part3_closed_loop", {}).get("stats", {}).get(
                    "provider_calls", 0
                )
            ),
            "eligible_decisions": int(
                summary.get("part3_closed_loop", {}).get(
                    "eligible_decision_count", 0
                )
            ),
        }
        row.update({metric: float(validation[metric]) for metric in MACRO_METRICS})
        rows.append(row)
    return rows


def _collect_model_decisions(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for inference_path in sorted(output_root.rglob("part3_provider_inference.jsonl")):
        assignment_round = _assignment_round(inference_path.parent)
        for record in _read_jsonl(inference_path):
            packet = record["packet"]
            evidence = packet["llm_visible_evidence"]
            opportunity = evidence["contact_opportunity"]
            metadata = packet["trace_evidence"]["metadata"]
            response = record["result"]["response"]
            rule_reference = packet["trace_evidence"]["rule_reference"]
            rule_action = str(
                record.get("request", {}).get("rule_reference_action")
                or rule_reference["selected_action"]
            )
            memory = evidence.get("memory_state_before") or []
            trace_memory = packet["trace_evidence"].get("memory_state_before") or []
            top_memory = memory[0] if memory else {}
            same_colleague_memory = any(
                bool(row.get("same_current_colleague")) for row in memory
            )
            same_patient_memory = any(
                bool(row.get("same_patient_context")) for row in memory
            )
            memory_context = (
                "Same colleague"
                if same_colleague_memory
                else "Same patient context"
                if same_patient_memory
                else "Other prior contact"
                if memory
                else "No retrieved memory"
            )
            diversion = float(opportunity["diversion_cost"])
            rows.append(
                {
                    "scenario": metadata["scenario_mode"],
                    "condition": metadata["condition"],
                    "seed": int(metadata["seed"]),
                    "assignment_round": assignment_round,
                    "persona": packet["user_model_profile"]["persona_id"],
                    "staff_id": int(metadata["staff_id"]),
                    "role": metadata["role"],
                    "action": response["selected_action"],
                    "rule_action": rule_action,
                    "action_agreement": response["selected_action"] == rule_action,
                    "engagement_agreement": (
                        (response["selected_action"] == "engage")
                        == (rule_action == "engage")
                    ),
                    "rule_engagement_probability": float(
                        rule_reference.get("engagement_probability", math.nan)
                    ),
                    "rule_random_draw": float(
                        rule_reference.get("random_draw", math.nan)
                    ),
                    "reason": response["selected_reason"],
                    "topic": response["topic_family"],
                    "abm_reason": packet["trace_evidence"]["decision_context"][
                        "abm_reason_type"
                    ],
                    "diversion_cost": diversion,
                    "diversion_band": _diversion_band(diversion),
                    "urgency": float(opportunity["urgency"]),
                    "pressure": float(opportunity["ed_pressure_index"]),
                    "zone": opportunity["zone"],
                    "patient_context_present": bool(
                        opportunity["patient_context_present"]
                    ),
                    "active_task_present": bool(
                        opportunity.get("active_task_present", False)
                    ),
                    "mutual_visibility": bool(
                        opportunity.get("mutual_visibility", False)
                    ),
                    "memory_count": len(memory),
                    "memory_full": len(memory) >= 5,
                    "mean_memory_age_seconds": _mean(
                        [float(row["seconds_ago"]) for row in trace_memory]
                    ),
                    "mean_memory_retrieval_score": _mean(
                        [float(row["retrieval_score"]) for row in trace_memory]
                    ),
                    "memory_context": memory_context,
                    "same_colleague_memory": same_colleague_memory,
                    "same_patient_context_memory": same_patient_memory,
                    "recent_memory_within_15_minutes": any(
                        int(row.get("seconds_ago", 10**9)) <= 900
                        for row in memory
                    ),
                    "retrieved_topic_context": top_memory.get(
                        "topic_family", "No retrieved memory"
                    ),
                    "retrieved_reason_context": top_memory.get(
                        "reason_context", "No retrieved memory"
                    ),
                }
            )
    return rows


def _rule_qwen_seed_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare Qwen and the hidden rule on the same sampled opportunities."""

    grouped: dict[
        tuple[str, str, int, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in decisions:
        grouped[
            (
                str(row["scenario"]),
                str(row["condition"]),
                int(row["seed"]),
                str(row["persona"]),
            )
        ].append(row)

    output = []
    for (scenario, condition, seed, persona), rows in sorted(grouped.items()):
        qwen_engages = sum(row["action"] == "engage" for row in rows)
        rule_engages = sum(row["rule_action"] == "engage" for row in rows)
        rule_to_qwen_engage = sum(
            row["rule_action"] != "engage" and row["action"] == "engage"
            for row in rows
        )
        rule_engage_to_qwen_nonengage = sum(
            row["rule_action"] == "engage" and row["action"] != "engage"
            for row in rows
        )
        output.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": seed,
                "persona": persona,
                "sampled_opportunity_count": len(rows),
                "action_agreement_rate": _rate(
                    sum(bool(row["action_agreement"]) for row in rows), len(rows)
                ),
                "engagement_agreement_rate": _rate(
                    sum(bool(row["engagement_agreement"]) for row in rows),
                    len(rows),
                ),
                "qwen_engage_rate": _rate(qwen_engages, len(rows)),
                "rule_engage_rate": _rate(rule_engages, len(rows)),
                "qwen_minus_rule_engage_rate": _rate(
                    qwen_engages - rule_engages, len(rows)
                ),
                "rule_nonengage_to_qwen_engage_rate": _rate(
                    rule_to_qwen_engage, len(rows)
                ),
                "rule_engage_to_qwen_nonengage_rate": _rate(
                    rule_engage_to_qwen_nonengage, len(rows)
                ),
            }
        )
    return output


def _rule_qwen_effect_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize local substitutions and their spatial modulation by paired seed."""

    lookup = {
        (row["scenario"], row["condition"], row["seed"], row["persona"]): row
        for row in seed_rows
    }
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    output = []
    for scenario in SCENARIOS:
        for persona in PERSONAS:
            for condition in CONDITIONS:
                rows = [lookup[(scenario, condition, seed, persona)] for seed in seeds]
                deltas = [
                    float(row["qwen_minus_rule_engage_rate"]) for row in rows
                ]
                output.append(
                    {
                        "analysis_type": "same-opportunity local substitution",
                        "scenario": SCENARIO_LABELS[scenario],
                        "persona": PERSONA_LABELS[persona],
                        "condition": CONDITION_LABELS[condition],
                        "contrast": "Qwen - hidden rule",
                        "mean_qwen_engage_rate": _mean(
                            [float(row["qwen_engage_rate"]) for row in rows]
                        ),
                        "mean_rule_engage_rate": _mean(
                            [float(row["rule_engage_rate"]) for row in rows]
                        ),
                        "mean_action_agreement_rate": _mean(
                            [float(row["action_agreement_rate"]) for row in rows]
                        ),
                        "mean_engagement_agreement_rate": _mean(
                            [
                                float(row["engagement_agreement_rate"])
                                for row in rows
                            ]
                        ),
                        **_paired_statistics(deltas),
                    }
                )
            for intervention in INTERVENTIONS:
                deltas = [
                    float(
                        lookup[(scenario, intervention, seed, persona)][
                            "qwen_minus_rule_engage_rate"
                        ]
                    )
                    - float(
                        lookup[(scenario, "baseline", seed, persona)][
                            "qwen_minus_rule_engage_rate"
                        ]
                    )
                    for seed in seeds
                ]
                output.append(
                    {
                        "analysis_type": "paired spatial modulation of local substitution",
                        "scenario": SCENARIO_LABELS[scenario],
                        "persona": PERSONA_LABELS[persona],
                        "condition": CONDITION_LABELS[intervention],
                        "contrast": (
                            f"(Qwen - rule) in {CONDITION_LABELS[intervention]} "
                            "- (Qwen - rule) in Baseline"
                        ),
                        "mean_qwen_engage_rate": math.nan,
                        "mean_rule_engage_rate": math.nan,
                        "mean_action_agreement_rate": math.nan,
                        "mean_engagement_agreement_rate": math.nan,
                        **_paired_statistics(deltas),
                    }
                )
    return output


def _rule_qwen_strata_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose where Qwen replaces the rule without treating decisions as iid."""

    dimensions = {
        "persona": ("persona",),
        "role": ("role",),
        "ABM opportunity reason": ("abm_reason",),
        "diversion-cost band": ("diversion_band",),
        "persona x opportunity reason": ("persona", "abm_reason"),
    }
    output = []
    for stratum_type, keys in dimensions.items():
        grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in decisions:
            grouped[tuple(str(row[key]) for key in keys)].append(row)
        for key, rows in sorted(grouped.items()):
            qwen_engages = sum(row["action"] == "engage" for row in rows)
            rule_engages = sum(row["rule_action"] == "engage" for row in rows)
            output.append(
                {
                    "stratum_type": stratum_type,
                    "stratum": " | ".join(key),
                    "sampled_opportunity_count": len(rows),
                    "action_agreement_rate": _rate(
                        sum(bool(row["action_agreement"]) for row in rows),
                        len(rows),
                    ),
                    "engagement_agreement_rate": _rate(
                        sum(bool(row["engagement_agreement"]) for row in rows),
                        len(rows),
                    ),
                    "qwen_engage_rate": _rate(qwen_engages, len(rows)),
                    "rule_engage_rate": _rate(rule_engages, len(rows)),
                    "qwen_minus_rule_engage_rate": _rate(
                        qwen_engages - rule_engages, len(rows)
                    ),
                    "rule_nonengage_to_qwen_engage_count": sum(
                        row["rule_action"] != "engage"
                        and row["action"] == "engage"
                        for row in rows
                    ),
                    "rule_engage_to_qwen_nonengage_count": sum(
                        row["rule_action"] == "engage"
                        and row["action"] != "engage"
                        for row in rows
                    ),
                    "inference_boundary": (
                        "Descriptive same-opportunity diagnostic; uncertainty is "
                        "reported only after aggregation within seed."
                    ),
                }
            )
    return output


def _collect_agent_experience(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("part3_agent_experience.jsonl")):
        for row in _read_jsonl(path):
            if int(row.get("experience_schema_version", 0)) != 1:
                raise ValueError(f"Unsupported agent-experience schema in {path}")
            rows.append(row)
    return rows


def _seed_persona_experience_rows(
    experience_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in experience_rows:
        grouped[
            (
                str(row["scenario"]),
                str(row["condition"]),
                int(row["seed"]),
                str(row["persona_id"]),
            )
        ].append(row)

    output = []
    for (scenario, condition, seed, persona), rows in sorted(grouped.items()):
        total_seconds = sum(float(row["window_duration_seconds"]) for row in rows)
        total_agent_hours = total_seconds / 3600.0
        total_interactions = sum(int(row["interaction_count"]) for row in rows)
        eligible_cognitive_opportunities = sum(
            int(row["eligible_cognitive_opportunity_count"]) for row in rows
        )
        sampled_cognitive_opportunities = sum(
            int(row["sampled_cognitive_opportunity_count"]) for row in rows
        )
        model_decisions = sum(int(row["model_decision_count"]) for row in rows)
        model_engages = sum(
            int(row["model_engage_decision_count"]) for row in rows
        )
        realized_initiated = sum(
            int(row["model_realized_initiated_interaction_count"])
            for row in rows
        )
        output.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": seed,
                "persona": persona,
                "agent_shift_count": len(rows),
                "role_coverage_count": len({str(row["role"]) for row in rows}),
                "eligible_cognitive_opportunity_count": eligible_cognitive_opportunities,
                "sampled_cognitive_opportunity_count": sampled_cognitive_opportunities,
                "model_decision_count": model_decisions,
                "model_engage_decision_count": model_engages,
                "model_realized_initiated_interaction_count": realized_initiated,
                "grounded_memory_retrieval_count": sum(
                    int(row["grounded_memory_retrieval_count"]) for row in rows
                ),
                "movement_m_per_agent_hour": sum(
                    float(row["movement_distance_m"]) for row in rows
                )
                / total_agent_hours,
                "mean_mutually_visible_colleagues": sum(
                    float(row["mean_mutually_visible_colleagues"])
                    * float(row["window_duration_seconds"])
                    for row in rows
                )
                / total_seconds,
                "share_time_any_visible_colleague": sum(
                    float(row["share_of_time_with_any_mutually_visible_colleague"])
                    * float(row["window_duration_seconds"])
                    for row in rows
                )
                / total_seconds,
                "eligible_cognitive_opportunities_per_agent_hour": (
                    eligible_cognitive_opportunities / total_agent_hours
                ),
                "sampled_cognitive_opportunities_per_agent_hour": (
                    sampled_cognitive_opportunities / total_agent_hours
                ),
                "model_engage_decisions_per_agent_hour": (
                    model_engages / total_agent_hours
                ),
                "model_realized_initiated_interactions_per_agent_hour": (
                    realized_initiated / total_agent_hours
                ),
                "interaction_participations_per_agent_hour": (
                    total_interactions / total_agent_hours
                ),
                "staff_staff_participations_per_agent_hour": sum(
                    int(row["staff_staff_interaction_count"]) for row in rows
                )
                / total_agent_hours,
                "patient_facing_participations_per_agent_hour": sum(
                    int(row["patient_facing_interaction_count"]) for row in rows
                )
                / total_agent_hours,
                "mean_unique_staff_partners": _mean(
                    [
                        float(row["unique_staff_interaction_partner_count"])
                        for row in rows
                    ]
                ),
                "model_engage_rate": _rate(model_engages, model_decisions),
                "model_realization_yield": _rate(
                    realized_initiated, model_engages
                ),
                "memory_retrievals_per_model_decision": (
                    sum(int(row["grounded_memory_retrieval_count"]) for row in rows)
                    / model_decisions
                    if model_decisions
                    else math.nan
                ),
            }
        )
    return output


def _persona_experience_effect_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["scenario"], row["condition"], int(row["seed"]), row["persona"]): row
        for row in seed_rows
    }
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    output = []
    for scenario in SCENARIOS:
        for persona in PERSONAS:
            for intervention in INTERVENTIONS:
                for metric, label in EXPERIENCE_METRICS.items():
                    deltas = [
                        float(lookup[(scenario, intervention, seed, persona)][metric])
                        - float(lookup[(scenario, "baseline", seed, persona)][metric])
                        for seed in seeds
                    ]
                    output.append(
                        {
                            "scenario": SCENARIO_LABELS[scenario],
                            "persona": PERSONA_LABELS[persona],
                            "contrast": f"{CONDITION_LABELS[intervention]} - Baseline",
                            "outcome": label,
                            **_paired_statistics(deltas),
                        }
                    )
    return output


def _seed_macro_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["scenario"], row["condition"], row["seed"])].append(row)
    result = []
    for (scenario, condition, seed), rows in sorted(grouped.items()):
        result.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": seed,
                "assignment_round_count": len(rows),
                **{
                    metric: _mean([float(row[metric]) for row in rows])
                    for metric in MACRO_METRICS
                },
            }
        )
    return result


def _macro_effect_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["scenario"], row["condition"], row["seed"]): row
        for row in seed_rows
    }
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    available_conditions = {str(row["condition"]) for row in seed_rows}
    output = []
    for scenario in SCENARIOS:
        for intervention in INTERVENTIONS:
            if intervention not in available_conditions:
                continue
            for metric, label in MACRO_METRICS.items():
                deltas = [
                    float(lookup[(scenario, intervention, seed)][metric])
                    - float(lookup[(scenario, "baseline", seed)][metric])
                    for seed in seeds
                ]
                output.append(
                    {
                        "scenario": SCENARIO_LABELS[scenario],
                        "contrast": (
                            f"{CONDITION_LABELS[intervention]} - Baseline"
                        ),
                        "metric": label,
                        **_paired_statistics(deltas),
                    }
                )
    return output


def _seed_persona_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in decisions:
        grouped[(row["scenario"], row["condition"], row["seed"], row["persona"])].append(row)
    output = []
    for (scenario, condition, seed, persona), rows in sorted(grouped.items()):
        actions = Counter(str(row["action"]) for row in rows)
        role_rows = {
            role: [row for row in rows if row["role"] == role] for role in ROLES
        }
        role_rates = {
            role: _rate(
                sum(row["action"] == "engage" for row in selected),
                len(selected),
            )
            for role, selected in role_rows.items()
        }
        role_coverage = sum(math.isfinite(rate) for rate in role_rates.values())
        role_standardized_rate = (
            sum(ROLE_WEIGHTS[role] * role_rates[role] for role in ROLES)
            if role_coverage == len(ROLES)
            else math.nan
        )
        output.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": seed,
                "persona": persona,
                "model_decisions": len(rows),
                "engage_rate": _rate(actions["engage"], len(rows)),
                "role_standardized_engage_rate": role_standardized_rate,
                "role_coverage_count": role_coverage,
                "coordination_nurse_decisions": len(role_rows["CoordinationNurse"]),
                "coordination_nurse_engage_rate": role_rates["CoordinationNurse"],
                "nurse_decisions": len(role_rows["Nurse"]),
                "nurse_engage_rate": role_rates["Nurse"],
                "doctor_decisions": len(role_rows["Doctor"]),
                "doctor_engage_rate": role_rates["Doctor"],
                "defer_rate": _rate(actions["defer"], len(rows)),
                "decline_rate": _rate(actions["decline"], len(rows)),
                "mean_diversion_cost": _mean(
                    [float(row["diversion_cost"]) for row in rows]
                ),
                "mean_urgency": _mean([float(row["urgency"]) for row in rows]),
                "mean_pressure": _mean([float(row["pressure"]) for row in rows]),
                "patient_context_share": _rate(
                    sum(bool(row["patient_context_present"]) for row in rows),
                    len(rows),
                ),
                "memory_present_share": _rate(
                    sum(int(row["memory_count"]) > 0 for row in rows), len(rows)
                ),
                "same_colleague_memory_share": _rate(
                    sum(bool(row["same_colleague_memory"]) for row in rows),
                    len(rows),
                ),
                "same_patient_context_memory_share": _rate(
                    sum(bool(row["same_patient_context_memory"]) for row in rows),
                    len(rows),
                ),
            }
        )
    return output


def _persona_effect_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["scenario"], row["condition"], row["seed"], row["persona"]): row
        for row in seed_rows
    }
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    available_conditions = {str(row["condition"]) for row in seed_rows}
    output = []
    for scenario in SCENARIOS:
        for persona in PERSONAS:
            for intervention in INTERVENTIONS:
                if intervention not in available_conditions:
                    continue
                for estimator, outcome in (
                    (
                        "engage_rate",
                        "Opportunity-weighted model engagement-rate change",
                    ),
                    (
                        "role_standardized_engage_rate",
                        "Role-standardized model engagement-rate change",
                    ),
                ):
                    deltas = [
                        float(
                            lookup[(scenario, intervention, seed, persona)][
                                estimator
                            ]
                        )
                        - float(
                            lookup[(scenario, "baseline", seed, persona)][
                                estimator
                            ]
                        )
                        for seed in seeds
                    ]
                    output.append(
                        {
                            "scenario": SCENARIO_LABELS[scenario],
                            "persona": PERSONA_LABELS[persona],
                            "contrast": (
                                f"{CONDITION_LABELS[intervention]} - Baseline"
                            ),
                            "outcome": outcome,
                            **_paired_statistics(deltas),
                        }
                    )
    return output


def _persona_role_seed_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in decisions:
        grouped[
            (
                row["scenario"],
                row["condition"],
                int(row["seed"]),
                row["persona"],
                row["role"],
            )
        ].append(row)
    output = []
    for (scenario, condition, seed, persona, role), rows in sorted(
        grouped.items()
    ):
        actions = Counter(str(row["action"]) for row in rows)
        output.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": seed,
                "persona": persona,
                "role": role,
                "model_decisions": len(rows),
                "engage_rate": _rate(actions["engage"], len(rows)),
                "defer_rate": _rate(actions["defer"], len(rows)),
                "decline_rate": _rate(actions["decline"], len(rows)),
                "mean_diversion_cost": _mean(
                    [float(row["diversion_cost"]) for row in rows]
                ),
                "mean_urgency": _mean([float(row["urgency"]) for row in rows]),
            }
        )
    return output


def _context_rows(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "scenario",
        "condition",
        "seed",
        "persona",
        "role",
        "abm_reason",
        "diversion_band",
        "memory_context",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        actions = Counter(str(row["action"]) for row in rows)
        output.append(
            {
                **dict(zip(keys, key)),
                "opportunities": len(rows),
                "engage_rate": _rate(actions["engage"], len(rows)),
                "defer_rate": _rate(actions["defer"], len(rows)),
                "decline_rate": _rate(actions["decline"], len(rows)),
                "mean_diversion_cost": _mean(
                    [float(row["diversion_cost"]) for row in rows]
                ),
                "mean_urgency": _mean([float(row["urgency"]) for row in rows]),
            }
        )
    return output


def _behavioral_saturation_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose extreme designed-policy rates without treating them as human rates."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        persona = str(row["persona"])
        grouped[(persona, "All sampled opportunities", "All")].append(row)
        grouped[(persona, "ABM opportunity type", str(row["abm_reason"]))].append(row)
        grouped[(persona, "Diversion-cost band", str(row["diversion_band"]))].append(row)

    output = []
    for (persona, stratum_type, stratum), rows in sorted(grouped.items()):
        total = len(rows)
        engaged = sum(row["action"] == "engage" for row in rows)
        engage_rate = _rate(engaged, total)
        ci_low, ci_high = _wilson_interval(engaged, total)
        enough_evidence = total >= 30
        extreme = enough_evidence and (
            engage_rate <= 0.05 or engage_rate >= 0.95
        )
        output.append(
            {
                "persona": PERSONA_LABELS[persona],
                "stratum_type": stratum_type,
                "stratum": stratum,
                "sampled_opportunities": total,
                "engage_count": engaged,
                "engage_rate": engage_rate,
                "wilson_ci95_low": ci_low,
                "wilson_ci95_high": ci_high,
                "extreme_rate_review_flag": extreme,
                "interpretation_boundary": (
                    "Designed policy response among Qwen-sampled opportunities; "
                    "not a population estimate for ED staff"
                ),
            }
        )
    return output


def _construct_strata(
    persona: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if persona == "patient_advocate":
        return (
            "Patient-linked vs other opportunity",
            "Patient-linked",
            [row for row in rows if row["patient_context_present"]],
            [row for row in rows if not row["patient_context_present"]],
        )
    if persona == "vigilant_monitor":
        return (
            "High vs low urgency",
            "High urgency",
            [row for row in rows if float(row["urgency"]) >= 2.0 / 3.0],
            [row for row in rows if float(row["urgency"]) < 1.0 / 3.0],
        )
    if persona == "focus_protector":
        return (
            "Low vs high diversion cost",
            "Low diversion cost",
            [row for row in rows if float(row["diversion_cost"]) < 1.0 / 3.0],
            [row for row in rows if float(row["diversion_cost"]) >= 2.0 / 3.0],
        )
    if persona == "team_connector":
        coordination = {"BED_FLOW_COORDINATION", "HANDOFF_NEED"}
        return (
            "Operational coordination vs station check-in",
            "Operational coordination",
            [row for row in rows if row["abm_reason"] in coordination],
            [
                row
                for row in rows
                if row["abm_reason"] == "SAME_STATION_BRIEF_CHECKIN"
            ],
        )
    if persona == "adaptive_generalist":
        return (
            "Low vs high diversion cost",
            "Low diversion cost",
            [row for row in rows if float(row["diversion_cost"]) < 1.0 / 3.0],
            [row for row in rows if float(row["diversion_cost"]) >= 2.0 / 3.0],
        )
    raise ValueError(f"No construct-coherence check for {persona}")


def _construct_check_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(int(row["seed"]), row["persona"])].append(row)
    output = []
    for (seed, persona), rows in sorted(grouped.items()):
        check, favored_label, favored, comparison = _construct_strata(persona, rows)
        favored_rate = _rate(
            sum(row["action"] == "engage" for row in favored), len(favored)
        )
        comparison_rate = _rate(
            sum(row["action"] == "engage" for row in comparison),
            len(comparison),
        )
        output.append(
            {
                "seed": seed,
                "persona": persona,
                "construct_check": check,
                "favored_stratum": favored_label,
                "favored_n": len(favored),
                "favored_engage_rate": favored_rate,
                "comparison_n": len(comparison),
                "comparison_engage_rate": comparison_rate,
                "directional_delta": (
                    favored_rate - comparison_rate
                    if math.isfinite(favored_rate)
                    and math.isfinite(comparison_rate)
                    else math.nan
                ),
            }
        )
    return output


def _construct_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["persona"], row["construct_check"])].append(row)
    output = []
    for (persona, check), selected in sorted(grouped.items()):
        deltas = _finite(float(row["directional_delta"]) for row in selected)
        output.append(
            {
                "persona": PERSONA_LABELS[persona],
                "construct_check": check,
                "expected_direction": "Positive",
                "seeds_with_both_strata": len(deltas),
                "positive_seed_count": sum(value > 0 for value in deltas),
                **_paired_statistics(deltas),
            }
        )
    return output


def _memory_audit_rows(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(row["scenario"], row["condition"], int(row["seed"]))].append(row)
    output = []
    for (scenario, condition, seed), rows in sorted(grouped.items()):
        with_memory = [row for row in rows if int(row["memory_count"]) > 0]
        ages = _finite(row["mean_memory_age_seconds"] for row in with_memory)
        scores = _finite(
            row["mean_memory_retrieval_score"] for row in with_memory
        )
        output.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": seed,
                "model_decisions": len(rows),
                "memory_present_share": _rate(len(with_memory), len(rows)),
                "mean_retrieved_memories": _mean(
                    [float(row["memory_count"]) for row in rows]
                ),
                "full_five_memory_share": _rate(
                    sum(bool(row["memory_full"]) for row in rows), len(rows)
                ),
                "same_colleague_memory_share": _rate(
                    sum(bool(row["same_colleague_memory"]) for row in rows),
                    len(rows),
                ),
                "same_patient_memory_share": _rate(
                    sum(bool(row["same_patient_context_memory"]) for row in rows),
                    len(rows),
                ),
                "generic_only_memory_share": _rate(
                    sum(
                        int(row["memory_count"]) > 0
                        and not bool(row["same_colleague_memory"])
                        and not bool(row["same_patient_context_memory"])
                        for row in rows
                    ),
                    len(rows),
                ),
                "median_mean_memory_age_seconds": (
                    statistics.median(ages) if ages else math.nan
                ),
                "mean_retrieval_score": _mean(scores),
            }
        )
    return output


def _topic_memory_rows(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[
            (
                row["scenario"],
                row["condition"],
                row["persona"],
                row["retrieved_topic_context"],
                row["topic"],
            )
        ].append(row)
    output = []
    for (scenario, condition, persona, retrieved_topic, topic), rows in sorted(grouped.items()):
        actions = Counter(str(row["action"]) for row in rows)
        output.append(
            {
                "scenario": scenario,
                "condition": condition,
                "persona": persona,
                "retrieved_topic_context": retrieved_topic,
                "topic": topic,
                "decisions": len(rows),
                "engage_rate": _rate(actions["engage"], len(rows)),
            }
        )
    return output


def _team_connector_diagnostic(
    decisions: Sequence[Mapping[str, Any]], expected_seed_count: int
) -> dict[str, Any]:
    selected = [
        row
        for row in decisions
        if row["persona"] == "team_connector"
        and row["abm_reason"] == "BED_FLOW_COORDINATION"
        and row["condition"] in {"baseline", "both"}
    ]
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["scenario"], row["condition"], row["seed"])].append(row)
    per_seed = []
    extreme_seed_count = 0
    eligible_seed_count = 0
    for scenario in SCENARIOS:
        for seed in range(1, expected_seed_count + 1):
            baseline = grouped.get((scenario, "baseline", seed), [])
            both = grouped.get((scenario, "both", seed), [])
            if not baseline or not both:
                continue
            baseline_rate = _rate(
                sum(row["action"] == "engage" for row in baseline), len(baseline)
            )
            both_rate = _rate(sum(row["action"] == "engage" for row in both), len(both))
            extreme = baseline_rate >= 0.90 and both_rate <= 0.10
            eligible_seed_count += 1
            extreme_seed_count += int(extreme)
            per_seed.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "baseline_n": len(baseline),
                    "both_n": len(both),
                    "baseline_engage_rate": baseline_rate,
                    "both_engage_rate": both_rate,
                    "baseline_mean_diversion": _mean(
                        [float(row["diversion_cost"]) for row in baseline]
                    ),
                    "both_mean_diversion": _mean(
                        [float(row["diversion_cost"]) for row in both]
                    ),
                    "extreme_contrast": extreme,
                }
            )
    persistent = (
        eligible_seed_count >= expected_seed_count * len(SCENARIOS)
        and extreme_seed_count / eligible_seed_count >= 0.80
    )
    return {
        "diagnostic_not_integrity_gate": True,
        "eligible_scenario_seed_cells": eligible_seed_count,
        "extreme_scenario_seed_cells": extreme_seed_count,
        "extreme_cell_share": _rate(extreme_seed_count, eligible_seed_count),
        "persistent_extreme_contrast": persistent,
        "interpretation": (
            "If persistent after stratifying by diversion band, ABM reason, and "
            "grounded memory context, treat this as a persona-by-environment result. "
            "If it attenuates within strata, treat the one-seed contrast as "
            "opportunity-composition and experience-context dependence."
        ),
        "per_seed": per_seed,
    }


def _markdown_report(
    integrity: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    macro_effects: Sequence[Mapping[str, Any]],
    persona_effects: Sequence[Mapping[str, Any]],
    rule_qwen_effects: Sequence[Mapping[str, Any]],
    construct_summary: Sequence[Mapping[str, Any]],
    saturation_rows: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    exploratory: bool,
) -> str:
    seed_count = len({int(row["seed"]) for row in run_rows})
    observed_conditions = [
        condition
        for condition in CONDITIONS
        if any(row["condition"] == condition for row in run_rows)
    ]
    lines = [
        "# Part 3 closed-loop cognitive study",
        "",
        "## Status",
        "",
        (
            "This is a sizing-pilot diagnostic and is not a scientific result."
            if exploratory
            else "The paired main-study integrity and design gates passed."
        ),
        "",
        f"- Complete runs: {len(run_rows)}",
        f"- Paired seeds: {seed_count}",
        f"- Model decisions analyzed: {len(decisions)}",
        f"- Conditions: {', '.join(CONDITION_LABELS[value] for value in observed_conditions)}",
        "- Five persona-to-role rotations are averaged within seed.",
        "- The seed, not the individual model decision, is the inferential unit.",
        "- Persona engagement is reported both for the observed opportunity ecology "
        "and with a fixed 1:5:3 coordination-nurse, nurse, doctor composition.",
        "",
        "## Statistical approach",
        "",
        "Condition effects are paired within scenario and seed. Tables report the "
        "mean paired delta, SD and SE of paired deltas, a Student-t 95% confidence "
        "interval, and an exact two-sided sign-flip p-value. Model decisions are "
        "first aggregated within seed; they are not treated as independent samples. "
        "The p-values are secondary diagnostics, not a substitute for effect sizes "
        "and uncertainty intervals.",
        "",
        "## What Qwen changed relative to the hidden rule",
        "",
        "The rule and Qwen outputs are compared only on the exact opportunities "
        "that were sent to the model. `Qwen - hidden rule` is therefore a local "
        "same-opportunity substitution diagnostic, not a second trajectory and not "
        "an estimate of what an all-rule rerun would have produced. Rates are first "
        "aggregated within seed before uncertainty is calculated.",
        "",
        f"Exact three-action agreement: {_rate(sum(bool(row['action_agreement']) for row in decisions), len(decisions)):.3f}.",
        f"Binary engage/non-engage agreement: {_rate(sum(bool(row['engagement_agreement']) for row in decisions), len(decisions)):.3f}.",
        f"Hidden-rule non-engage to Qwen engage: {sum(row['rule_action'] != 'engage' and row['action'] == 'engage' for row in decisions)} opportunities.",
        f"Hidden-rule engage to Qwen non-engage: {sum(row['rule_action'] == 'engage' and row['action'] != 'engage' for row in decisions)} opportunities.",
        "These descriptive counts establish behavioral non-equivalence; whether "
        "that substitution is substantively useful is evaluated by the paired "
        "seed-level and evidence-stratified tables.",
        "",
        f"Same-opportunity comparison rows: {sum(row['analysis_type'] == 'same-opportunity local substitution' for row in rule_qwen_effects)}.",
        f"Spatial-modulation rows: {sum(row['analysis_type'] == 'paired spatial modulation of local substitution' for row in rule_qwen_effects)}.",
        "",
        "## Team Connector diagnostic",
        "",
        diagnostic["interpretation"],
        "",
        f"Persistent extreme contrast flag: `{diagnostic['persistent_extreme_contrast']}`.",
        "The context-stratified table exposes diversion band, ABM reason, and grounded "
        "experience-memory context, with role retained as a blocking variable, so a "
        "condition contrast is not interpreted without its changing opportunity ecology.",
        "",
        "## Construct-coherence diagnostics",
        "",
        "Each designed orientation has one preregisterable directional sanity check "
        "based on evidence it is intended to prioritize. These checks are falsification "
        "diagnostics, not psychometric validation or additional primary outcomes.",
        "",
        *[
            f"- {row['persona']}: {row['construct_check']}; mean directional delta "
            f"{float(row['mean_delta']):.3f} across {int(row['n_paired_seeds'])} seed(s)."
            for row in construct_summary
        ],
        "",
        "## Outputs",
        "",
        "- `run_summary.csv`: one row per complete ABM run.",
        "- `paired_macro_effects.csv`: seed-level paired system effects.",
        "- `persona_seed_summary.csv`: persona decisions aggregated within seed.",
        "- `persona_role_seed_summary.csv`: role-stratified persona decisions.",
        "- `paired_persona_effects.csv`: opportunity-weighted and role-standardized persona contrasts.",
        "- `rule_qwen_seed_comparison.csv`: hidden-rule and Qwen actions aggregated within seed on matched opportunities.",
        "- `rule_qwen_paired_effects.csv`: seed-level local substitutions and paired spatial modulation.",
        "- `rule_qwen_substitution_strata.csv`: descriptive replacement directions by persona and evidence stratum.",
        "- `persona_experience_seed_summary.csv`: post-warmup movement, visibility, interaction, and memory exposure aggregated within seed and persona.",
        "- `paired_persona_experience_effects.csv`: paired condition effects on the agent-experience layer.",
        "- `persona_context_strata.csv`: role-blocked opportunity-composition diagnostic.",
        "- `persona_construct_checks.csv`: seed-level orientation sanity checks.",
        "- `persona_construct_summary.csv`: directional construct-coherence summary.",
        "- `memory_retrieval_audit.csv`: bounded memory exposure and saturation audit.",
        "- `behavioral_saturation_audit.csv`: descriptive extreme-rate review with Wilson intervals.",
        "- `grounded_memory_topic_summary.csv`: descriptive retrieved-topic and current-topic summary.",
        "- `team_connector_diagnostic.json`: explicit extreme-contrast audit.",
        "",
        "## Validity boundary",
        "",
        "These are results from designed synthetic cognitive-interaction orientations, "
        "not human personality types or Zurich ED staff testimony. They can test "
        "whether bounded cognitive policies interact with spatial conditions in the "
        "model. They cannot establish real staff preferences, psychometric validity, "
        "clinical outcomes, or causal effects in the physical ED.",
        "",
        "## Behavioral saturation audit",
        "",
        "Absolute action rates describe the bounded synthetic policy only. They are not "
        "calibrated estimates of how often a real clinician would engage. Strata with "
        "at least 30 sampled opportunities and an engagement rate at or below 5% or at "
        "or above 95% are flagged for substantive review rather than automatically "
        "accepted or rejected.",
        "",
        f"Flagged strata: {sum(bool(row['extreme_rate_review_flag']) for row in saturation_rows)}.",
        "",
        "## Integrity",
        "",
        f"Verifier pass: `{integrity.get('verification_pass')}`; technical integrity: "
        f"`{integrity.get('technical_integrity_pass')}`; design complete: "
        f"`{integrity.get('study_design_complete', integrity.get('pilot_design_complete'))}`.",
    ]
    return "\n".join(lines) + "\n"


def analyze(
    output_root: Path,
    integrity_path: Path,
    output_dir: Path,
    *,
    allow_sizing_pilot: bool = False,
) -> dict[str, Any]:
    integrity = _read_json(integrity_path)
    if integrity.get("verification_pass") is not True:
        raise ValueError("Refusing to analyze an integrity-failing result")
    scientific_result = integrity.get("scientific_result") is True
    if not scientific_result and not allow_sizing_pilot:
        raise ValueError(
            "Refusing to analyze a sizing pilot as a main result; use "
            "--allow-sizing-pilot only for local analyzer testing"
        )

    run_rows = _collect_runs(output_root)
    decisions = _collect_model_decisions(output_root)
    agent_experience = _collect_agent_experience(output_root)
    expected_seed_count = int(
        integrity.get("expected_seed_count")
        or len({int(row["seed"]) for row in run_rows})
    )
    if scientific_result:
        expected_run_count = (
            len(SCENARIOS)
            * len(CONDITIONS)
            * expected_seed_count
            * 5
        )
        if len(run_rows) != expected_run_count:
            raise ValueError(
                f"Expected {expected_run_count} complete main runs, got "
                f"{len(run_rows)}"
            )
        observed_cells = {
            (row["scenario"], row["condition"], int(row["seed"]))
            for row in run_rows
        }
        expected_cells = {
            (scenario, condition, seed)
            for scenario in SCENARIOS
            for condition in CONDITIONS
            for seed in range(1, expected_seed_count + 1)
        }
        if observed_cells != expected_cells:
            raise ValueError("Main run cells are incomplete or unexpected")
    seed_macro = _seed_macro_rows(run_rows)
    if scientific_result and any(
        int(row["assignment_round_count"]) != 5 for row in seed_macro
    ):
        raise ValueError("One or more main cells do not contain five rotations")
    macro_effects = _macro_effect_rows(seed_macro)
    seed_persona = _seed_persona_rows(decisions)
    if scientific_result:
        observed_persona_cells = {
            (
                row["scenario"],
                row["condition"],
                int(row["seed"]),
                row["persona"],
            )
            for row in seed_persona
        }
        expected_persona_cells = {
            (scenario, condition, seed, persona)
            for scenario in SCENARIOS
            for condition in CONDITIONS
            for seed in range(1, expected_seed_count + 1)
            for persona in PERSONAS
        }
        if observed_persona_cells != expected_persona_cells:
            raise ValueError(
                "At least one main scenario-condition-seed-persona cell has no "
                "sampled model decision"
            )
    persona_effects = _persona_effect_rows(seed_persona)
    rule_qwen_seed = _rule_qwen_seed_rows(decisions)
    rule_qwen_effects = _rule_qwen_effect_rows(rule_qwen_seed)
    rule_qwen_strata = _rule_qwen_strata_rows(decisions)
    seed_persona_experience = _seed_persona_experience_rows(agent_experience)
    if scientific_result:
        expected_experience_cells = {
            (scenario, condition, seed, persona)
            for scenario in SCENARIOS
            for condition in CONDITIONS
            for seed in range(1, expected_seed_count + 1)
            for persona in PERSONAS
        }
        observed_experience_cells = {
            (
                row["scenario"],
                row["condition"],
                int(row["seed"]),
                row["persona"],
            )
            for row in seed_persona_experience
        }
        if observed_experience_cells != expected_experience_cells or any(
            int(row["agent_shift_count"]) != 9
            or int(row["role_coverage_count"]) != len(ROLES)
            for row in seed_persona_experience
        ):
            raise ValueError(
                "Agent-experience evidence is incomplete or unbalanced"
            )
    persona_experience_effects = _persona_experience_effect_rows(
        seed_persona_experience
    )
    persona_role = _persona_role_seed_rows(decisions)
    context = _context_rows(decisions)
    construct_checks = _construct_check_rows(decisions)
    construct_summary = _construct_summary_rows(construct_checks)
    saturation_rows = _behavioral_saturation_rows(decisions)
    memory_audit = _memory_audit_rows(decisions)
    topic_memory = _topic_memory_rows(decisions)
    diagnostic = _team_connector_diagnostic(decisions, expected_seed_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "run_summary.csv": run_rows,
        "seed_macro_summary.csv": seed_macro,
        "paired_macro_effects.csv": macro_effects,
        "persona_seed_summary.csv": seed_persona,
        "persona_role_seed_summary.csv": persona_role,
        "paired_persona_effects.csv": persona_effects,
        "rule_qwen_seed_comparison.csv": rule_qwen_seed,
        "rule_qwen_paired_effects.csv": rule_qwen_effects,
        "rule_qwen_substitution_strata.csv": rule_qwen_strata,
        "persona_experience_seed_summary.csv": seed_persona_experience,
        "paired_persona_experience_effects.csv": persona_experience_effects,
        "persona_context_strata.csv": context,
        "persona_construct_checks.csv": construct_checks,
        "persona_construct_summary.csv": construct_summary,
        "behavioral_saturation_audit.csv": saturation_rows,
        "memory_retrieval_audit.csv": memory_audit,
        "grounded_memory_topic_summary.csv": topic_memory,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / name, [_round_row(row) for row in rows])
    (output_dir / "team_connector_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True)
    )
    (output_dir / "part3_closed_loop_report.md").write_text(
        _markdown_report(
            integrity,
            run_rows,
            decisions,
            macro_effects,
            persona_effects,
            rule_qwen_effects,
            construct_summary,
            saturation_rows,
            diagnostic,
            exploratory=not scientific_result,
        )
    )
    result = {
        "analysis_pass": True,
        "scientific_result": scientific_result,
        "run_count": len(run_rows),
        "seed_count": expected_seed_count,
        "model_decision_count": len(decisions),
        "rule_qwen_action_agreement_rate": _rate(
            sum(bool(row["action_agreement"]) for row in decisions),
            len(decisions),
        ),
        "rule_qwen_engagement_agreement_rate": _rate(
            sum(bool(row["engagement_agreement"]) for row in decisions),
            len(decisions),
        ),
        "rule_nonengage_to_qwen_engage_count": sum(
            row["rule_action"] != "engage" and row["action"] == "engage"
            for row in decisions
        ),
        "rule_engage_to_qwen_nonengage_count": sum(
            row["rule_action"] == "engage" and row["action"] != "engage"
            for row in decisions
        ),
        "rule_qwen_comparison_scope": (
            "Same model-sampled opportunities only; not a counterfactual rerun."
        ),
        "role_standardization_weights": ROLE_WEIGHTS,
        "complete_role_standardization_cell_count": sum(
            int(row["role_coverage_count"]) == len(ROLES)
            for row in seed_persona
        ),
        "persona_seed_cell_count": len(seed_persona),
        "agent_experience_row_count": len(agent_experience),
        "persona_experience_seed_cell_count": len(seed_persona_experience),
        "construct_check_count": len(construct_summary),
        "construct_checks_with_positive_mean": sum(
            float(row["mean_delta"]) > 0 for row in construct_summary
        ),
        "behavioral_saturation_review_flag_count": sum(
            bool(row["extreme_rate_review_flag"])
            for row in saturation_rows
        ),
        "behavioral_saturation_is_human_calibration": False,
        "output_files": sorted([*tables, "team_connector_diagnostic.json", "part3_closed_loop_report.md"]),
        "team_connector_persistent_extreme_contrast": diagnostic[
            "persistent_extreme_contrast"
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--integrity-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-sizing-pilot",
        action="store_true",
        help="Permit exploratory local testing on a verified non-scientific sizing pilot",
    )
    args = parser.parse_args()
    result = analyze(
        Path(args.output_root),
        Path(args.integrity_json),
        Path(args.output_dir),
        allow_sizing_pilot=args.allow_sizing_pilot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
