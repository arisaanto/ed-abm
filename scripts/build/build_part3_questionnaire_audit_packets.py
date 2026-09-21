#!/usr/bin/env python3
"""Build the blinded Part 3 empirical-questionnaire audit packets.

The builder reads completed baseline trajectories only. It does not rerun the
ABM, call an LLM, or read the confidential hospital questionnaire. The full
design contains 300 balanced persona-role-scenario-seed shifts and a 30-shift
matched control set, for 390 packets in total.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.build.build_part3_appraisal_packets import (
    MAIN_STUDY_SEEDS,
    _digest,
    _estimate_model_visible_tokens,
    _load_agent_shifts,
    _load_selected_agent_events,
    _orientation_text,
)
from src.personas import default_cognitive_personas
from src.questionnaire_audit import QUESTIONNAIRE_ITEMS


PACKET_TYPE = "empirical_questionnaire_bundle"
AUDIT_VERSION = "construct_translation_observed_outcomes"
ROLES = ("CoordinationNurse", "Nurse", "Doctor")
SCENARIOS = ("normal_load", "high_load_high_acuity")
ASSESSMENT_VARIANTS = ("balanced",)
CONTROL_VARIANTS = (
    "full_trace_persona",
    "full_trace_neutral",
    "evidence_withheld_persona",
    "evidence_withheld_neutral",
)

SYSTEM_MESSAGE = (
    "Complete a bounded end-of-shift questionnaire for one simulated emergency-"
    "department staff shift. This is synthetic model output, never human testimony. "
    "Use only supplied shift evidence. Do not infer missing facts from role or "
    "orientation labels. The item evidence cards are designed to support Q1-Q7; Q8 is an "
    "acoustic negative control and must remain unrated. Treat confidence as epistemic "
    "certainty, not as a substitute for using the full response scale. Return "
    "schema-valid JSON only."
)

MODEL_SCORING_RUBRIC = {
    "q1": "1 exceptionally light, 2 clearly lighter, 3 somewhat lighter, 4 typical, 5 somewhat heavier, 6 clearly heavier, 7 exceptionally heavy relative to comparable simulated days",
    "q2": "report the supplied category exactly: 0, 25, 50, 75, or 100 percent",
    "q3": "1 concentration repeatedly broke down, 2 very difficult, 3 difficult, 4 mixed, 5 generally sustained, 6 sustained well, 7 unusually uninterrupted",
    "q4": "1 repeated communication failure, 2 severe recurring difficulty, 3 friction clearly outweighed support, 4 support and friction were genuinely balanced, 5 communication needs were generally met despite ordinary imperfections, 6 consistently strong communication with only limited friction, 7 exceptional near-seamless communication",
    "q5": "1 strongly disconnected, 2 severe recurring coordination failure, 3 fragmentation clearly outweighed coordination, 4 coordination and friction were genuinely balanced, 5 the team generally worked together well despite ordinary imperfections, 6 consistently strong teamwork with only limited friction, 7 exceptional near-seamless teamwork",
    "q6": "1 operational support repeatedly failed, 2 severe recurring constraint, 3 constraints clearly outweighed support, 4 support and constraint were genuinely balanced, 5 the department was generally operationally helpful despite ordinary imperfections, 6 consistently strong operational support with only limited constraint, 7 exceptional operational support",
    "q7": "1 information repeatedly absent, 2 major recurring gaps, 3 gaps clearly outweighed access, 4 access and gaps were genuinely balanced, 5 needed information was generally available despite ordinary imperfections, 6 consistently strong information access with only limited gaps, 7 exceptional near-complete information access",
    "q8": "1 = very low noise; 7 = very high noise",
}

EVIDENCE_IDS = (
    "shift:aggregate",
    "shift:workload_windows",
    "shift:communication",
    "department:patient_flow",
    "reference:other_days",
    "scope:acoustics_unobserved",
)

INFORMATION_TOPICS = {
    "bed_and_capacity_management",
    "diagnostic_findings",
    "next_steps_and_planning",
    "patient_flow_and_transfer",
    "patient_status_update",
    "treatment_and_orders",
}


def _bounded_share(numerator: float, denominator: float) -> float:
    """Return a finite proportion while preventing impossible proxy values."""

    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / float(denominator)))


def _phase_balance(values: list[float]) -> float:
    """Return 0-1 balance across phases; presence alone is not sufficient."""

    if not values or sum(values) <= 0:
        return 0.0
    mean = statistics.fmean(values)
    return round(1.0 / (1.0 + statistics.pstdev(values) / mean), 3)


def _inverse_percentile(value: float | None) -> float | None:
    return None if value is None else round(100.0 - float(value), 1)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))


def _select_baseline_shifts(
    shifts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    personas = tuple(row.persona_id for row in default_cognitive_personas())
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for shift in shifts:
        row = shift["map"]
        if str(row["condition"]) != "baseline":
            continue
        grouped[
            (
                str(row["scenario"]),
                int(row["seed"]),
                str(row["persona_id"]),
                str(row["role"]),
            )
        ].append(shift)

    expected = {
        (scenario, seed, persona, role)
        for scenario in SCENARIOS
        for seed in MAIN_STUDY_SEEDS
        for persona in personas
        for role in ROLES
    }
    missing = sorted(expected - set(grouped))
    if missing:
        raise ValueError(f"Missing baseline questionnaire strata: {missing[:10]}")
    selected = []
    for key in sorted(expected):
        candidates = sorted(
            grouped[key],
            key=lambda shift: _digest(
                shift["map"]["run_id"],
                shift["map"]["agent_id"],
                "questionnaire_audit",
            ),
        )
        selected.append(candidates[0])
    return selected, {
        "selection_policy": (
            "one sha256-stable baseline shift per scenario-seed-persona-role stratum"
        ),
        "available_agent_shift_count": len(shifts),
        "selected_shift_count": len(selected),
        "expected_stratum_count": len(expected),
        "observed_stratum_count": len(expected) - len(missing),
        "missing_strata": [list(row) for row in missing],
        "scenario_count": len(SCENARIOS),
        "seed_count": len(MAIN_STUDY_SEEDS),
        "persona_count": len(personas),
        "role_count": len(ROLES),
        "conditions": ["baseline"],
    }


def _feature_values(summary: Mapping[str, Any]) -> dict[str, float]:
    dwell = {str(key): float(value) for key, value in summary.get("mode_dwell_seconds", {}).items()}
    total = max(1.0, sum(dwell.values()))
    return {
        "non_idle_share": 1.0 - dwell.get("idle", 0.0) / total,
        "interaction_count": float(summary.get("interaction_count", 0)),
        "staff_interaction_count": float(summary.get("staff_staff_interaction_count", 0)),
        "unique_staff_partner_count": float(
            summary.get("unique_staff_interaction_partner_count", 0)
        ),
        "missed_optional_contact_count": float(
            summary.get("model_missed_opportunity_count", 0)
        ),
        "visible_colleague_minutes": float(
            summary.get("minutes_with_any_mutually_visible_colleague", 0.0)
        ),
        "patient_interaction_count": float(summary.get("patient_facing_interaction_count", 0)),
        "eligible_contact_count": float(summary.get("eligible_cognitive_opportunity_count", 0)),
        "movement_distance_m": float(summary.get("movement_distance_m", 0.0)),
    }


def _phase_index(timestep: int, *, start: int, duration: int) -> int:
    return min(2, max(0, 3 * (timestep - start) // max(1, duration)))


def _temporal_trace(
    events: list[dict[str, Any]], *, start: int, duration: int
) -> dict[str, Any]:
    """Reconstruct work continuity and communication across three shift phases."""

    end = start + duration
    ordered = sorted(events, key=lambda row: (int(row.get("timestep", start)), str(row.get("event_type", ""))))
    starts = [row for row in ordered if row.get("event_type") == "experience_window_started"]
    initial_mode = str(starts[0].get("mode", "idle")) if starts else "idle"
    transitions = [
        row for row in ordered
        if row.get("event_type") == "mode_transition"
        and start <= int(row.get("timestep", start)) <= end
    ]
    transition_times = [int(row["timestep"]) for row in transitions]
    transition_modes = [str(row.get("to_mode", "idle")) for row in transitions]

    def mode_at(timestep: int) -> str:
        position = bisect.bisect_right(transition_times, timestep) - 1
        return transition_modes[position] if position >= 0 else initial_mode

    intervals = []
    cursor = start
    mode = initial_mode
    for row in transitions:
        timestep = min(end, max(start, int(row["timestep"])))
        if timestep > cursor:
            intervals.append((cursor, timestep, mode))
        cursor = max(cursor, timestep)
        mode = str(row.get("to_mode", mode))
    if cursor < end:
        intervals.append((cursor, end, mode))

    task_intervals = [(left, right) for left, right, value in intervals if value == "performing_task"]
    disruptions = []
    phase_rows = [
        {
            "phase": label,
            "staff_interactions": 0,
            "information_bearing_interactions": 0,
            "missed_optional_contacts": 0,
            "cognitive_contact_decisions": 0,
            "task_seconds": 0,
            "task_contact_disruptions": 0,
        }
        for label in ("early", "middle", "late")
    ]
    phase_partners = [set(), set(), set()]
    for left, right in task_intervals:
        phase_rows[_phase_index((left + right) // 2, start=start, duration=duration)]["task_seconds"] += right - left
    for row in ordered:
        timestep = int(row.get("timestep", start))
        if not start <= timestep < end:
            continue
        event_type = str(row.get("event_type", ""))
        phase = _phase_index(timestep, start=start, duration=duration)
        staff_interaction = (
            event_type == "interaction"
            and str(row.get("partner_role")) in ROLES
        )
        if staff_interaction:
            phase_rows[phase]["staff_interactions"] += 1
            if str(row.get("topic_family") or row.get("topic") or "") in INFORMATION_TOPICS:
                phase_rows[phase]["information_bearing_interactions"] += 1
            if row.get("partner_id") is not None:
                phase_partners[phase].add(int(row["partner_id"]))
        elif event_type == "missed_opportunity":
            phase_rows[phase]["missed_optional_contacts"] += 1
        elif event_type == "cognitive_opportunity_decision":
            phase_rows[phase]["cognitive_contact_decisions"] += 1
        contact_disruption = staff_interaction or event_type == "cognitive_opportunity_decision"
        if contact_disruption and mode_at(timestep) == "performing_task":
            duration_seconds = float(row.get("duration_seconds", 0.0)) if staff_interaction else 0.0
            disruptions.append((timestep, duration_seconds, event_type))
            phase_rows[phase]["task_contact_disruptions"] += 1

    uninterrupted_spans = []
    for left, right in task_intervals:
        local = sorted((t, d) for t, d, _kind in disruptions if left <= t < right)
        cursor = left
        for timestep, disruption_duration in local:
            uninterrupted_spans.append(max(0.0, timestep - cursor))
            cursor = max(cursor, timestep + disruption_duration)
        uninterrupted_spans.append(max(0.0, right - cursor))
    positive_spans = [value for value in uninterrupted_spans if value > 0]
    task_seconds = sum(right - left for left, right in task_intervals)
    for index, row in enumerate(phase_rows):
        row["unique_staff_partners"] = len(phase_partners[index])
        row["task_minutes"] = round(float(row.pop("task_seconds")) / 60.0, 1)
    staff_by_phase = [float(row["staff_interactions"]) for row in phase_rows]
    information_by_phase = [
        float(row["information_bearing_interactions"]) for row in phase_rows
    ]
    staff_total = sum(staff_by_phase)
    information_total = sum(information_by_phase)
    return {
        "task_episode_count": len(task_intervals),
        "task_minutes": round(task_seconds / 60.0, 1),
        "task_contact_disruption_count": len(disruptions),
        "task_contact_disruptions_per_task_hour": round(
            len(disruptions) / max(task_seconds / 3600.0, 1e-9), 2
        ),
        "median_uninterrupted_task_span_seconds": round(
            statistics.median(positive_spans), 1
        ) if positive_spans else 0.0,
        "longest_uninterrupted_task_span_seconds": round(max(positive_spans), 1) if positive_spans else 0.0,
        "staff_interaction_phase_balance": _phase_balance(staff_by_phase),
        "information_interaction_phase_balance": _phase_balance(information_by_phase),
        "information_phase_coverage": round(
            sum(value > 0 for value in information_by_phase) / 3.0, 3
        ),
        "information_bearing_interaction_share": round(
            _bounded_share(information_total, staff_total), 3
        ),
        "phase_evidence": phase_rows,
        "occupancy_is_not_concentration_evidence": True,
    }


def _team_trace(run_dir: str, *, start: int, duration: int) -> dict[str, Any]:
    """Summarize realized collaboration for the complete simulated team."""

    path = Path(run_dir)
    summaries = [json.loads(line) for line in (path / "part3_agent_experience.jsonl").read_text().splitlines() if line.strip()]
    staff_ids = {int(row["agent_id"]) for row in summaries}
    events = [json.loads(line) for line in (path / "part3_experience_events.jsonl").read_text().splitlines() if line.strip()]
    realized: dict[str, dict[str, Any]] = {}
    missed: dict[str, dict[str, Any]] = {}
    for row in events:
        event_type = str(row.get("event_type", ""))
        if event_type not in {"interaction", "missed_opportunity"}:
            continue
        if row.get("partner_id") is None:
            continue
        if int(row["agent_id"]) not in staff_ids or int(row["partner_id"]) not in staff_ids:
            continue
        sources = row.get("source_event_ids") or [row.get("event_id")]
        key = str(sources[0])
        target = realized if event_type == "interaction" else missed
        target.setdefault(key, row)
    dyads = {
        tuple(sorted((int(row["agent_id"]), int(row["partner_id"]))))
        for row in realized.values() if row.get("partner_id") is not None
    }
    cross_role = [
        row for row in realized.values()
        if row.get("partner_role") and str(row.get("role")) != str(row.get("partner_role"))
    ]
    handoffs = [
        row for row in realized.values()
        if "handoff" in str(row.get("interaction_type", "")).lower()
    ]
    information_bearing = [
        row for row in realized.values()
        if str(row.get("topic", "")) in INFORMATION_TOPICS
    ]
    covered_staff = {
        agent
        for row in realized.values()
        for agent in (int(row["agent_id"]), int(row["partner_id"]))
        if row.get("partner_id") is not None
    }
    interaction_degree = Counter()
    for row in realized.values():
        interaction_degree[int(row["agent_id"])] += 1
        interaction_degree[int(row["partner_id"])] += 1
    degree_values = [float(interaction_degree.get(agent_id, 0)) for agent_id in staff_ids]
    degree_mean = statistics.fmean(degree_values) if degree_values else 0.0
    degree_cv = statistics.pstdev(degree_values) / degree_mean if degree_mean else 0.0
    phase_rows = [
        {"phase": label, "realized_staff_interactions": 0, "missed_staff_opportunities": 0}
        for label in ("early", "middle", "late")
    ]
    for collection, field in (
        (realized.values(), "realized_staff_interactions"),
        (missed.values(), "missed_staff_opportunities"),
    ):
        for row in collection:
            phase_rows[_phase_index(int(row.get("timestep", start)), start=start, duration=duration)][field] += 1
    opportunity_total = len(realized) + len(missed)
    return {
        "team_staff_count": len(staff_ids),
        "realized_staff_interaction_count": len(realized),
        "missed_staff_opportunity_count": len(missed),
        "realization_rate": round(len(realized) / max(1, opportunity_total), 3),
        "unique_staff_dyad_count": len(dyads),
        "cross_role_interaction_count": len(cross_role),
        "cross_role_interaction_share": round(len(cross_role) / max(1, len(realized)), 3),
        "handoff_interaction_count": len(handoffs),
        "information_bearing_interaction_count": len(information_bearing),
        "information_bearing_interaction_share": round(
            len(information_bearing) / max(1, len(realized)), 3
        ),
        "staff_interaction_coverage": round(len(covered_staff & staff_ids) / max(1, len(staff_ids)), 3),
        "staff_interaction_balance": round(1.0 / (1.0 + degree_cv), 3),
        "phase_evidence": phase_rows,
    }


def _percentile(value: float, reference: list[float]) -> float | None:
    if not reference:
        return None
    lower = sum(row < value for row in reference)
    equal = sum(row == value for row in reference)
    return round(100.0 * (lower + 0.5 * equal) / len(reference), 1)


def _percentile_band(percentile: float | None) -> str:
    if percentile is None:
        return "unavailable"
    boundaries = (
        (5.0, "exceptionally low"),
        (20.0, "clearly below typical"),
        (40.0, "somewhat below typical"),
        (60.0, "typical"),
        (80.0, "somewhat above typical"),
        (95.0, "clearly above typical"),
    )
    return next((label for ceiling, label in boundaries if percentile <= ceiling), "exceptionally high")


def _q2_category(critical_share: float) -> int:
    if critical_share < 0.125:
        return 0
    if critical_share < 0.375:
        return 25
    if critical_share < 0.625:
        return 50
    if critical_share < 0.875:
        return 75
    return 100


def _reference_profiles(
    selected: list[dict[str, Any]],
    window_cache: Mapping[tuple[str, str], list[dict[str, Any]]],
    temporal_cache: Mapping[tuple[str, str], dict[str, Any]],
    team_cache: Mapping[str, dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for shift in selected:
        row = shift["map"]
        grouped[(str(row["persona_id"]), str(row["role"]))].append(shift)
    profiles = {}
    for persona_role, members in grouped.items():
        # The survey asks about personal workload. Department pressure remains
        # consequential, but the focal worker's observed activity must dominate.
        workload_weights = {
            "system_demand_pressure": 0.40,
            "interaction_events": 0.08,
            "workflow_events": 0.14,
            "cognitive_contact_decisions": 0.08,
            "task_or_mode_transitions": 0.12,
            "patient_linked_events": 0.10,
            "travel_distance_m": 0.08,
        }
        workload_components = tuple(workload_weights)
        derived: dict[tuple[str, str], dict[str, Any]] = {}
        for member in members:
            member_key = (str(member["map"]["run_id"]), str(member["map"]["agent_id"]))
            comparison_windows = [
                window
                for reference_member in members
                if str(reference_member["map"]["scenario"]) == "normal_load"
                and (
                    str(reference_member["map"]["run_id"]),
                    str(reference_member["map"]["agent_id"]),
                ) != member_key
                for window in window_cache[
                    (
                        str(reference_member["map"]["run_id"]),
                        str(reference_member["map"]["agent_id"]),
                    )
                ]
            ]
            component_references = {
                key: [float(window[key]) for window in comparison_windows]
                for key in workload_components
            }
            windows = []
            for window in window_cache[member_key]:
                component_percentiles = {
                    key: float(_percentile(float(window[key]), component_references[key]) or 0.0)
                    for key in workload_components
                }
                composite_score = sum(
                    workload_weights[key] * component_percentiles[key]
                    for key in workload_components
                )
                windows.append({
                    **window,
                    "local_workload_composite_score": round(composite_score, 1),
                })
            derived[member_key] = {
                "windows": windows,
                "temporal_trace": temporal_cache[member_key],
                "team_trace": team_cache[str(member["run_dir"])],
            }
        normal_member_keys = [
            (str(member["map"]["run_id"]), str(member["map"]["agent_id"]))
            for member in members
            if str(member["map"]["scenario"]) == "normal_load"
        ]
        for member in members:
            member_key = (
                str(member["map"]["run_id"]),
                str(member["map"]["agent_id"]),
            )
            comparison_keys = [
                key for key in normal_member_keys if key != member_key
            ]
            composite_reference = [
                float(window["local_workload_composite_score"])
                for key in comparison_keys
                for window in derived[key]["windows"]
            ]
            ranked_windows = []
            for window in derived[member_key]["windows"]:
                composite_percentile = _percentile(
                    float(window["local_workload_composite_score"]),
                    composite_reference,
                )
                ranked_windows.append({
                    **window,
                    "local_workload_composite_percentile": composite_percentile,
                    "locally_critical": bool(
                        composite_percentile is not None
                        and composite_percentile >= 75.0
                    ),
                })
            critical_count = sum(
                bool(window["locally_critical"]) for window in ranked_windows
            )
            derived[member_key].update({
                "windows": ranked_windows,
                "maximum_local_workload_percentile": max(
                    float(window["local_workload_composite_percentile"])
                    for window in ranked_windows
                    if window["local_workload_composite_percentile"] is not None
                ),
                "critical_window_count": critical_count,
                "critical_share": critical_count / len(ranked_windows),
            })
        outcome_by_member = {
            (str(member["map"]["run_id"]), str(member["map"]["agent_id"])):
            _run_outcomes(str(member["run_dir"]))
            for member in members
        }
        outcome_keys = (
            "completed_patient_count",
            "patient_completion_share",
            "first_nurse_coverage",
            "first_doctor_coverage",
            "clinical_contact_coverage",
            "workflow_task_completion_share",
            "doctor_task_completion_share",
            "deferred_arrival_share",
            "assignment_repair_count",
            "handoff_wait_repair_count",
            "backlog_patient_count_at_end",
            "ordinary_beds_full_share",
            "waiting_queue_present_share",
            "mean_waiting_queue_length",
            "mean_pressure_index",
            "doctor_task_wait_mean_minutes",
        )
        for member in members:
            row = member["map"]
            current = _feature_values(member["summary"])
            percentiles = {}
            for key, value in current.items():
                others = [
                    _feature_values(other["summary"])[key]
                    for other in members
                    if other is not member
                    and str(other["map"]["scenario"]) == "normal_load"
                ]
                percentiles[key] = _percentile(value, others)
            member_key = (str(row["run_id"]), str(row["agent_id"]))
            current_derived = derived[member_key]
            workload_reference = [
                derived[key]["maximum_local_workload_percentile"]
                for key in normal_member_keys
                if key != member_key
            ]
            max_percentile = _percentile(
                current_derived["maximum_local_workload_percentile"],
                workload_reference,
            )
            temporal = current_derived["temporal_trace"]
            disruption_percentile = _percentile(
                float(temporal["task_contact_disruptions_per_task_hour"]),
                [
                    float(derived[key]["temporal_trace"]["task_contact_disruptions_per_task_hour"])
                    for key in normal_member_keys if key != member_key
                ],
            )
            uninterrupted_percentile = _percentile(
                float(temporal["median_uninterrupted_task_span_seconds"]),
                [
                    float(derived[key]["temporal_trace"]["median_uninterrupted_task_span_seconds"])
                    for key in normal_member_keys if key != member_key
                ],
            )
            disruption_rank = 50.0 if disruption_percentile is None else disruption_percentile
            uninterrupted_rank = 50.0 if uninterrupted_percentile is None else uninterrupted_percentile
            focus_continuity = statistics.fmean(
                [100.0 - float(disruption_rank), float(uninterrupted_rank)]
            )
            communication_metrics = (
                "staff_interaction_phase_balance",
                "information_interaction_phase_balance",
                "information_phase_coverage",
                "information_bearing_interaction_share",
            )
            communication_percentiles = {
                metric: _percentile(
                    float(temporal[metric]),
                    [
                        float(derived[key]["temporal_trace"][metric])
                        for key in normal_member_keys if key != member_key
                    ],
                )
                for metric in communication_metrics
            }
            team = current_derived["team_trace"]
            team_percentiles = {
                metric: _percentile(
                    float(team[metric]),
                    [float(derived[key]["team_trace"][metric]) for key in normal_member_keys if key != member_key],
                )
                for metric in (
                    "unique_staff_dyad_count",
                    "cross_role_interaction_share",
                    "staff_interaction_balance",
                    "handoff_interaction_count",
                    "information_bearing_interaction_share",
                )
            }
            outcome_percentiles = {
                key: _percentile(
                    float(outcome_by_member[member_key][key]),
                    [
                        float(outcome_by_member[reference_key][key])
                        for reference_key in normal_member_keys
                        if reference_key != member_key
                    ],
                )
                for key in outcome_keys
            }
            profiles[(str(row["run_id"]), str(row["agent_id"]), "reference")] = {
                "comparison_group": "ordinary-load simulated days for the same staff role and designed orientation",
                "comparison_day_count": len(workload_reference),
                "current_shift_percentiles": percentiles,
                "workload_translation": {
                    "reference_pool": "hourly windows from normal-load days for the same role and orientation",
                    "critical_definition": "composite individual-burden percentile at or above 75 relative to ordinary-load composite scores; department pressure weight 0.40 and focal activity components weight 0.60",
                    "hourly_local_workload_composite_scores": [
                        window["local_workload_composite_score"]
                        for window in current_derived["windows"]
                    ],
                    "hourly_local_workload_percentiles": [
                        window["local_workload_composite_percentile"]
                        for window in current_derived["windows"]
                    ],
                    "critical_window_count": current_derived["critical_window_count"],
                    "window_count": len(current_derived["windows"]),
                    "raw_critical_share_percent": round(100.0 * current_derived["critical_share"], 1),
                    "questionnaire_category_percent": _q2_category(current_derived["critical_share"]),
                    "maximum_workload_percentile_across_days": max_percentile,
                    "maximum_workload_band": _percentile_band(max_percentile),
                },
                "focus_translation": {
                    **temporal,
                    "task_disruption_percentile_across_days": disruption_percentile,
                    "uninterrupted_task_span_percentile_across_days": uninterrupted_percentile,
                    "focus_continuity_index": round(focus_continuity, 1),
                    "focus_continuity_band": _percentile_band(focus_continuity),
                    "construct_definition": "continuity between observed task-contact disruptions; task occupancy is excluded",
                },
                "communication_translation": {
                    **{
                        metric: temporal[metric]
                        for metric in communication_metrics
                    },
                    "normal_day_percentiles": communication_percentiles,
                },
                "team_translation": {**team, "normal_day_percentiles": team_percentiles},
                "department_outcome_percentiles": outcome_percentiles,
            }
    return profiles


def _workload_windows(
    events: list[dict[str, Any]], *, start: int, duration: int
) -> list[dict[str, Any]]:
    window_count = 10
    width = max(1, duration // window_count)
    rows = []
    for index in range(window_count):
        rows.append(
            {
                "shift_hour": index + 1,
                "interaction_events": 0,
                "workflow_events": 0,
                "cognitive_contact_decisions": 0,
                "task_or_mode_transitions": 0,
                "patient_linked_events": 0,
                "travel_distance_m": 0.0,
            }
        )
    for event in events:
        timestep = int(event.get("timestep", start))
        index = min(window_count - 1, max(0, (timestep - start) // width))
        target = rows[index]
        event_type = str(event.get("event_type", ""))
        target["interaction_events"] += int(event_type == "interaction")
        target["workflow_events"] += int(event_type == "workflow_event")
        target["cognitive_contact_decisions"] += int(
            event_type == "cognitive_opportunity_decision"
        )
        target["task_or_mode_transitions"] += int(
            event_type in {"task_context_transition", "mode_transition"}
        )
        target["patient_linked_events"] += int(event.get("target_patient_id") is not None)
        if event_type == "travel_episode":
            target["travel_distance_m"] += float(event.get("distance_meters", 0.0))
    for row in rows:
        row["travel_distance_m"] = round(row["travel_distance_m"], 1)
    return rows


def _system_demand_windows(run_dir: str, window_count: int = 10) -> list[float]:
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    workflow = summary["workflow_health"]
    definition = workflow["scenario_definition"]
    hourly = [float(value) for value in definition["hourly_arrival_multipliers"]]
    start_hour = int(workflow["scenario_start_hour"])
    base = float(definition["arrival_rate_multiplier"])
    pressure = float(summary["operational_pressure_metrics"]["ed_pressure_index_mean"])
    return [
        base * hourly[(start_hour + index) % len(hourly)] * pressure
        for index in range(window_count)
    ]


def _run_outcomes(run_dir: str) -> dict[str, Any]:
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    pressure = summary["operational_pressure_metrics"]
    flow = summary["patient_flow_metrics"]
    workflow = summary["workflow_health"]
    completed = sum(flow.get("completed_patients_by_esi", {}).values())
    active = sum(flow.get("active_patients_by_esi", {}).values())
    admitted = int(workflow.get("admitted_arrivals", 0))
    arrival_attempts = int(workflow.get("arrival_attempts", admitted))
    first_nurse = sum(workflow.get("first_nurse_seen_by_esi", {}).values())
    first_doctor = sum(workflow.get("first_doctor_seen_by_esi", {}).values())
    handled = completed + active
    tasks_started = int(workflow.get("workflow_tasks_started", 0))
    tasks_completed = int(workflow.get("workflow_tasks_completed", 0))
    doctor_tasks_started = int(workflow.get("doctor_tasks_started", 0))
    doctor_tasks_completed = int(workflow.get("doctor_tasks_completed", 0))
    deferred = int(workflow.get("deferred_arrivals", 0))
    return {
        "arrival_attempt_count": arrival_attempts,
        "admitted_patient_count": admitted,
        "completed_patient_count": completed,
        "active_patient_count_at_end": active,
        "backlog_patient_count_at_end": sum(flow.get("backlog_by_esi", {}).values()),
        "patient_completion_share": round(_bounded_share(completed, handled), 3),
        "first_nurse_coverage": round(_bounded_share(first_nurse, handled), 3),
        "first_doctor_coverage": round(_bounded_share(first_doctor, handled), 3),
        "clinical_contact_coverage": round(
            statistics.fmean(
                [
                    _bounded_share(first_nurse, handled),
                    _bounded_share(first_doctor, handled),
                ]
            ),
            3,
        ),
        "workflow_task_completion_share": round(
            _bounded_share(tasks_completed, tasks_started), 3
        ),
        "doctor_task_completion_share": round(
            _bounded_share(doctor_tasks_completed, doctor_tasks_started), 3
        ),
        "deferred_arrival_share": round(
            _bounded_share(deferred, arrival_attempts), 3
        ),
        "assignment_repair_count": int(workflow.get("assignment_repair_events", 0)),
        "handoff_wait_repair_count": int(workflow.get("handoff_wait_repair_events", 0)),
        "ordinary_beds_full_share": round(float(flow.get("percent_time_all_ordinary_beds_full", 0.0)), 3),
        "waiting_queue_present_share": round(float(flow.get("percent_time_waiting_queue_gt_zero", 0.0)), 3),
        "mean_waiting_queue_length": round(float(flow.get("mean_waiting_queue_length", 0.0)), 2),
        "mean_pressure_index": round(float(pressure.get("ed_pressure_index_mean", 0.0)), 3),
        "doctor_task_wait_mean_minutes": round(float(workflow.get("average_doctor_task_wait_seconds", 0.0)) / 60.0, 1),
        "scope_note": (
            "These are operational throughput and delay indicators. They do not measure "
            "clinical quality, patient satisfaction, or treatment success."
        ),
    }


def _visible_evidence(
    shift: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    summary = shift["summary"]
    dwell = {str(key): round(float(value) / 60.0, 1) for key, value in summary.get("mode_dwell_seconds", {}).items()}
    workload = dict(reference["workload_translation"])
    focus = dict(reference["focus_translation"])
    communication_trace = dict(reference["communication_translation"])
    team = dict(reference["team_translation"])
    staff_percentiles = dict(reference["current_shift_percentiles"])
    department_percentiles = dict(reference["department_outcome_percentiles"])
    topic_counts = {
        str(key): int(value)
        for key, value in summary.get("interaction_counts_by_topic", {}).items()
    }
    information_topic_counts = {
        key: value for key, value in topic_counts.items()
        if key in INFORMATION_TOPICS and value > 0
    }
    communication = {
        "staff_interaction_count": int(summary.get("staff_staff_interaction_count", 0)),
        "unique_staff_partner_count": int(summary.get("unique_staff_interaction_partner_count", 0)),
        "interaction_topics": topic_counts,
        "information_topic_counts": information_topic_counts,
        "information_topic_coverage": round(
            len(information_topic_counts) / len(INFORMATION_TOPICS), 3
        ),
        "information_bearing_interaction_count": sum(
            information_topic_counts.values()
        ),
        "model_missed_opportunity_count": int(summary.get("model_missed_opportunity_count", 0)),
        "minutes_with_any_mutually_visible_colleague": round(
            float(summary.get("minutes_with_any_mutually_visible_colleague", 0.0)), 1
        ),
        "phase_evidence": focus["phase_evidence"],
    }
    flow = _run_outcomes(str(shift["run_dir"]))
    return {
        "staff_role": str(summary["role"]),
        "shift_aggregate": {
            "mode_minutes": dwell,
            "movement_distance_m": round(float(summary.get("movement_distance_m", 0.0)), 1),
            "interaction_count": int(summary.get("interaction_count", 0)),
            "patient_facing_interaction_count": int(summary.get("patient_facing_interaction_count", 0)),
            "eligible_optional_contact_count": int(summary.get("eligible_cognitive_opportunity_count", 0)),
        },
        "other_day_reference": {
            "comparison_group": reference["comparison_group"],
            "comparison_day_count": reference["comparison_day_count"],
        },
        "item_evidence_cards": {
            "q1": {
                "maximum_workload_percentile_across_comparable_days": workload[
                    "maximum_workload_percentile_across_days"
                ],
                "plain_language_band": workload["maximum_workload_band"],
                "interpretation": "This is a relative rank across days, not a share of the shift.",
            },
            "q2": {
                "required_share_percent": workload["questionnaire_category_percent"],
                "raw_critical_share_percent": workload["raw_critical_share_percent"],
                "critical_window_count": workload["critical_window_count"],
                "window_count": workload["window_count"],
                "critical_definition": workload["critical_definition"],
            },
            "q3": {
                **focus,
                "positive_cue": "longer uninterrupted task spans relative to comparable simulated days",
                "friction_cue": "realized contacts and cognitive contact decisions occurring during task work",
                "excluded_cue": "the share of time marked performing_task is workload occupancy, not concentration",
            },
            "q4": {
                "absolute_adequacy_anchor": "Judge whether communication needs were met during this shift. Percentiles compare simulated shifts; the 50th percentile is not a neutral satisfaction score.",
                "support_percentiles": {
                    "staff_interaction_percentile": staff_percentiles["staff_interaction_count"],
                    "unique_partner_percentile": staff_percentiles["unique_staff_partner_count"],
                    "information_bearing_share_percentile": communication_trace["normal_day_percentiles"]["information_bearing_interaction_share"],
                    "communication_phase_balance_percentile": communication_trace["normal_day_percentiles"]["staff_interaction_phase_balance"],
                },
                "friction_percentiles": {
                    "task_contact_disruption_percentile": focus["task_disruption_percentile_across_days"],
                    "communication_phase_imbalance_percentile": _inverse_percentile(communication_trace["normal_day_percentiles"]["staff_interaction_phase_balance"]),
                },
                "raw_context": {
                    "staff_interactions": communication["staff_interaction_count"],
                    "unique_staff_partners": communication["unique_staff_partner_count"],
                    "information_bearing_interaction_share": communication_trace["information_bearing_interaction_share"],
                    "communication_phase_balance": communication_trace["staff_interaction_phase_balance"],
                },
                "demand_context": {
                    "maximum_workload_percentile_across_comparable_days": workload["maximum_workload_percentile_across_days"],
                    "critical_workload_share_percent": workload["questionnaire_category_percent"],
                    "interpretation": "Judge communication sufficiency relative to demand. More contact under greater demand is not automatically better communication.",
                },
                "phase_evidence": communication["phase_evidence"],
                "construct_boundary": "Optional contact decisions are not treated as communication failures. Only realized communication and observed task disruption inform this item.",
            },
            "q5": {
                "absolute_adequacy_anchor": "Judge whether the complete team generally worked together well. Median comparative performance can still be satisfactory; percentiles do not define the questionnaire midpoint.",
                "support_percentiles": {
                    "cross_role_coordination_percentile": team["normal_day_percentiles"]["cross_role_interaction_share"],
                    "interaction_balance_percentile": team["normal_day_percentiles"]["staff_interaction_balance"],
                    "handoff_interaction_percentile": team["normal_day_percentiles"]["handoff_interaction_count"],
                    "information_bearing_share_percentile": team["normal_day_percentiles"]["information_bearing_interaction_share"],
                    "workflow_completion_percentile": department_percentiles["workflow_task_completion_share"],
                },
                "friction_percentiles": {
                    "doctor_task_wait_percentile": department_percentiles["doctor_task_wait_mean_minutes"],
                    "deferred_arrival_percentile": department_percentiles["deferred_arrival_share"],
                },
                "raw_context": {
                    "whole_team_realized_interactions": team["realized_staff_interaction_count"],
                    "whole_team_unique_dyads": team["unique_staff_dyad_count"],
                    "whole_team_cross_role_interaction_share": team["cross_role_interaction_share"],
                    "whole_team_interaction_balance": team["staff_interaction_balance"],
                    "handoff_interactions": team["handoff_interaction_count"],
                    "workflow_task_completion_share": flow["workflow_task_completion_share"],
                    "doctor_task_wait_mean_minutes": flow["doctor_task_wait_mean_minutes"],
                },
                "phase_evidence": team["phase_evidence"],
                "interpretation": "Judge how the complete simulated team worked together, not only how sociable the focal staff member was.",
                "construct_boundary": "Optional contact decisions are excluded. Teamwork is inferred only from realized cross-role coordination, handoffs, interaction balance, and workflow outcomes.",
            },
            "q6": {
                "absolute_adequacy_anchor": "Judge whether the department generally remained operationally helpful to patients. Ordinary queues or unfinished work do not by themselves imply an unsatisfactory shift.",
                "support_percentiles": {
                    "completion_share_percentile": department_percentiles["patient_completion_share"],
                    "clinical_contact_coverage_percentile": department_percentiles["clinical_contact_coverage"],
                    "workflow_completion_percentile": department_percentiles["workflow_task_completion_share"],
                },
                "friction_percentiles": {
                    "deferred_arrival_percentile": department_percentiles["deferred_arrival_share"],
                    "backlog_percentile": department_percentiles["backlog_patient_count_at_end"],
                    "waiting_queue_presence_percentile": department_percentiles["waiting_queue_present_share"],
                    "doctor_task_wait_percentile": department_percentiles["doctor_task_wait_mean_minutes"],
                },
                "raw_context": {
                    "completed_patients": flow["completed_patient_count"],
                    "active_patients_at_end": flow["active_patient_count_at_end"],
                    "patient_completion_share": flow["patient_completion_share"],
                    "first_nurse_coverage": flow["first_nurse_coverage"],
                    "first_doctor_coverage": flow["first_doctor_coverage"],
                    "deferred_arrival_share": flow["deferred_arrival_share"],
                    "backlog_patients_at_end": flow["backlog_patient_count_at_end"],
                    "waiting_queue_present_share": flow["waiting_queue_present_share"],
                    "doctor_task_wait_mean_minutes": flow["doctor_task_wait_mean_minutes"],
                },
                "boundary": "Operational helpfulness only; no clinical quality or patient satisfaction is observed.",
                "construct_boundary": "No single queue or unfinished-patient cue determines the score; judge the complete support-and-constraint pattern.",
            },
            "q7": {
                "absolute_adequacy_anchor": "Judge whether workflow-required information was generally available. Percentiles compare simulated shifts; the 50th percentile is not a neutral information-access score.",
                "support_percentiles": {
                    "unique_partner_percentile": staff_percentiles["unique_staff_partner_count"],
                    "information_bearing_share_percentile": communication_trace["normal_day_percentiles"]["information_bearing_interaction_share"],
                    "information_phase_coverage_percentile": communication_trace["normal_day_percentiles"]["information_phase_coverage"],
                    "information_phase_balance_percentile": communication_trace["normal_day_percentiles"]["information_interaction_phase_balance"],
                    "handoff_interaction_percentile": team["normal_day_percentiles"]["handoff_interaction_count"],
                },
                "friction_percentiles": {
                    "information_bearing_deficit_percentile": _inverse_percentile(communication_trace["normal_day_percentiles"]["information_bearing_interaction_share"]),
                    "information_phase_gap_percentile": _inverse_percentile(communication_trace["normal_day_percentiles"]["information_phase_coverage"]),
                    "handoff_scarcity_percentile": _inverse_percentile(team["normal_day_percentiles"]["handoff_interaction_count"]),
                    "partner_narrowness_percentile": _inverse_percentile(staff_percentiles["unique_staff_partner_count"]),
                },
                "raw_context": {
                    "unique_staff_partners": communication["unique_staff_partner_count"],
                    "information_topic_counts": communication["information_topic_counts"],
                    "information_bearing_interactions": communication["information_bearing_interaction_count"],
                    "information_bearing_interaction_share": communication_trace["information_bearing_interaction_share"],
                    "information_phase_coverage": communication_trace["information_phase_coverage"],
                    "handoff_interactions": team["handoff_interaction_count"],
                },
                "demand_context": {
                    "maximum_workload_percentile_across_comparable_days": workload["maximum_workload_percentile_across_days"],
                    "critical_workload_share_percent": workload["questionnaire_category_percent"],
                    "interpretation": "Judge information sufficiency relative to workflow demand. More information-bearing contact under greater demand is not automatically better access.",
                },
                "phase_evidence": communication["phase_evidence"],
                "boundary": "This is a workflow-linked information-access proxy; the simulation does not encode every private information need.",
                "construct_boundary": "Optional contact decisions and generic waiting are excluded. Only realized information-bearing communication, handoffs, partner breadth, and temporal coverage inform this item.",
            },
            "q8": {
                "status": "unrateable",
                "reason": "No acoustic measurement or sound-generating process exists in the simulation.",
            },
        },
        "scope_limits": {
            "acoustic_or_noise_measurements": "not recorded by the simulation",
            "patient_reported_outcomes": "not recorded by the simulation",
        },
    }


def _withheld_evidence(role: str) -> dict[str, Any]:
    return {
        "staff_role": role,
        "shift_evidence": "withheld for a matched negative-control packet",
        "scope_limits": {
            "acoustic_or_noise_measurements": "not recorded by the simulation"
        },
    }


def _build_packet(
    shift: Mapping[str, Any],
    *,
    variant: str,
    visible: Mapping[str, Any],
    assessment_variant: str = "balanced",
) -> dict[str, Any]:
    row = shift["map"]
    persona = {item.persona_id: item for item in default_cognitive_personas()}[
        str(row["persona_id"])
    ]
    persona_enabled = variant.endswith("persona")
    evidence_withheld = variant.startswith("evidence_withheld")
    profile = persona.prompt_profile() if persona_enabled else {
        "persona_id": "neutral_orientation",
        "summary": "No additional designed workplace orientation is applied.",
    }
    evidence_ids = (
        ["scope:shift_evidence_withheld", "scope:acoustics_unobserved"]
        if evidence_withheld
        else list(EVIDENCE_IDS)
    )
    if assessment_variant not in ASSESSMENT_VARIANTS:
        raise ValueError(f"Unknown assessment variant: {assessment_variant}")
    stable_id = _digest(
        row["run_id"], row["agent_id"], variant, assessment_variant
    )[:20]
    orientation_instruction = (
        "Apply this designed workplace orientation silently: " + _orientation_text(profile)
        if persona_enabled
        else "Use a neutral orientation and judge only the supplied evidence."
    )
    fixed_answers = {}
    if not evidence_withheld:
        fixed_answers["q2"] = int(
            visible["item_evidence_cards"]["q2"]["required_share_percent"]
        )
    packet = {
        "prompt_id": f"questionnaire_{variant}_{stable_id}",
        "packet_type": PACKET_TYPE,
        "questionnaire_audit_version": AUDIT_VERSION,
        "system_message": SYSTEM_MESSAGE,
        "role": str(row["role"]),
        "user_model_profile": profile,
        "questionnaire_items": list(QUESTIONNAIRE_ITEMS),
        "model_scoring_rubric": MODEL_SCORING_RUBRIC,
        "control_variant": variant,
        "assessment_variant": assessment_variant,
        "evidence_withheld": evidence_withheld,
        "persona_orientation_enabled": persona_enabled,
        "trace_evidence": {
            "metadata": {
                "run_id": row["run_id"],
                "scenario_mode": row["scenario"],
                "condition": row["condition"],
                "seed": row["seed"],
                "assignment_round": row.get("assignment_round"),
                "agent_id": row["agent_id"],
                "persona_id": row["persona_id"],
                "role": row["role"],
            }
        },
        "llm_visible_evidence": dict(visible),
        "evidence_ids": evidence_ids,
        "required_rateable_question_ids": [] if evidence_withheld else ["q2"],
        "required_insufficient_question_ids": (
            [item["question_id"] for item in QUESTIONNAIRE_ITEMS]
            if evidence_withheld
            else ["q8"]
        ),
        "fixed_answers": fixed_answers,
        "fixture_only_not_scientific_data": False,
        "synthetic_design_probe_not_human_data": True,
        "source_is_simulated_experience": True,
        "empirical_answers_visible_to_model": False,
    }
    rating_instructions = (
        [
            "Shift evidence is deliberately withheld. Return insufficient_evidence with null scores for Q1-Q8.",
            "Do not use the role or designed orientation as a substitute for observed shift evidence.",
        ]
        if evidence_withheld
        else [
            "The construct cards are designed to support Q1-Q7. Rate an item when they support a bounded operational translation; otherwise return insufficient_evidence rather than inventing an answer.",
            "Use the full 1-7 scale when evidence warrants it. Do not default to 5 or 6 merely because the evidence is synthetic.",
            "Q1 must translate the supplied across-day percentile and must never describe that percentile as a share of the shift.",
            "For Q2, copy required_share_percent exactly; the critical-window rule was frozen before inference.",
            "For Q3, use interruption-free task spans and task-contact disruptions. Do not treat task occupancy, busyness, or idle time as concentration.",
            "For Q1-Q3, use the frozen base rubrics identically; the assessment emphasis below applies only to Q4-Q7.",
            "For Q4 and Q7, inspect the early, middle, and late phase evidence as well as the aggregate cues.",
            "For Q4, judge realized communication coverage and continuity; an unused optional contact is not a communication failure.",
            "For Q5, judge complete-team coordination from cross-role coverage, handoffs, task-chain completion, and workflow repairs; do not substitute sociability, queue pressure, or optional contacts for teamwork.",
            "For Q4-Q7, first judge absolute operational adequacy, then use percentiles to distinguish stronger and weaker shifts. A 50th percentile means typical among simulated shifts; it does not mean a neutral or unsatisfactory human rating.",
            "For Q4-Q7, use 4 only when consequential support and friction are genuinely balanced. Use 5 when needs were generally met and support clearly outweighed ordinary imperfections. Use 6 for consistently strong support with limited friction; perfection or negligible friction is not required. Reserve 7 for exceptional near-seamless evidence.",
            "For Q4 and Q7, judge sufficiency relative to operational demand. A larger contact or information count during a busier shift does not by itself imply better communication or information access.",
            "For Q6, judge operational helpfulness from patients served and clinician coverage; do not infer clinical quality or patient satisfaction, and do not treat queue presence alone as failure.",
            "For Q7, judge workflow-linked information coverage and handoffs; do not treat unused optional contacts as missing required information.",
            "For Q8, return insufficient_evidence with null scores; do not infer sound from busyness, interaction counts, or staff role.",
        ]
    )
    assessment_instruction = (
        "Q4-Q7 assessment: compare every support cue with every consequential friction "
        "cue, but do not count ordinary imperfection as failure."
    )
    packet["user_message"] = "\n".join(
        [
            "Complete all eight end-of-shift questions in question order.",
            orientation_instruction,
            *rating_instructions,
            assessment_instruction,
            "Questions: " + json.dumps(QUESTIONNAIRE_ITEMS, sort_keys=True),
            "Model scoring rubric: " + json.dumps(MODEL_SCORING_RUBRIC, sort_keys=True),
            "Shift evidence: " + json.dumps(visible, sort_keys=True),
        ]
    )
    packet["estimated_tokens"] = _estimate_model_visible_tokens(packet)
    return packet


def _measurement_discriminability(
    packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject impossible or non-discriminating Q4-Q7 evidence before inference."""

    result: dict[str, Any] = {}
    all_checks = []
    balanced = [
        packet for packet in packets
        if packet["control_variant"] == "full_trace_persona"
        and packet["assessment_variant"] == "balanced"
    ]
    for scenario in SCENARIOS:
        scenario_result = {}
        members = [
            packet for packet in balanced
            if packet["trace_evidence"]["metadata"]["scenario_mode"] == scenario
        ]
        for question_id in ("q4", "q5", "q6", "q7"):
            support = []
            friction = []
            net = []
            invalid = []
            for packet in members:
                card = packet["llm_visible_evidence"]["item_evidence_cards"][question_id]
                support_values = [
                    float(value) for value in card["support_percentiles"].values()
                    if value is not None
                ]
                friction_values = [
                    float(value) for value in card["friction_percentiles"].values()
                    if value is not None
                ]
                values = support_values + friction_values
                if not support_values or not friction_values or any(
                    not 0.0 <= value <= 100.0 for value in values
                ):
                    invalid.append(str(packet["prompt_id"]))
                    continue
                support_index = statistics.fmean(support_values)
                friction_index = statistics.fmean(friction_values)
                support.append(support_index)
                friction.append(friction_index)
                net.append((support_index + 100.0 - friction_index) / 2.0)
            occupied_bands = {
                min(6, max(0, int(value // 15.0))) for value in net
            }
            support_sd_floor = 8.0 if scenario == "normal_load" else 6.0
            friction_sd_floor = 8.0 if scenario == "normal_load" else 5.0
            band_floor = 4 if scenario == "normal_load" else 3
            checks = {
                "all_percentiles_bounded": not invalid,
                "support_unique_value_count_at_least_ten": len({round(value, 3) for value in support}) >= 10,
                "friction_unique_value_count_at_least_ten": len({round(value, 3) for value in friction}) >= 10,
                "support_sd_meets_scenario_floor": len(support) > 1 and statistics.pstdev(support) >= support_sd_floor,
                "friction_sd_meets_scenario_floor": len(friction) > 1 and statistics.pstdev(friction) >= friction_sd_floor,
                "net_evidence_band_count_meets_scenario_floor": len(occupied_bands) >= band_floor,
            }
            item_pass = all(checks.values())
            all_checks.append(item_pass)
            scenario_result[question_id] = {
                "audit_pass": item_pass,
                "packet_count": len(members),
                "support_mean": statistics.fmean(support) if support else None,
                "support_sd": statistics.pstdev(support) if len(support) > 1 else None,
                "friction_mean": statistics.fmean(friction) if friction else None,
                "friction_sd": statistics.pstdev(friction) if len(friction) > 1 else None,
                "net_minimum": min(net) if net else None,
                "net_maximum": max(net) if net else None,
                "occupied_net_band_count": len(occupied_bands),
                "support_sd_floor": support_sd_floor,
                "friction_sd_floor": friction_sd_floor,
                "occupied_net_band_floor": band_floor,
                "invalid_packet_count": len(invalid),
                "checks": checks,
            }
        result[scenario] = scenario_result
    return {"audit_pass": all(all_checks), "by_scenario_and_item": result}


def build_packets(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    shifts, selection = _select_baseline_shifts(_load_agent_shifts(results_root))
    window_durations = {
        int(shift["summary"]["window_duration_seconds"])
        for shift in shifts
    }
    window_cache = {}
    temporal_cache = {}
    team_cache = {}
    for shift in shifts:
        summary = shift["summary"]
        key = (str(shift["map"]["run_id"]), str(shift["map"]["agent_id"]))
        selected_events = _load_selected_agent_events(shift)
        windows = _workload_windows(
            selected_events,
            start=int(summary["window_start_seconds"]),
            duration=int(summary["window_duration_seconds"]),
        )
        system_demand = _system_demand_windows(str(shift["run_dir"]), len(windows))
        window_cache[key] = [
            {**window, "system_demand_pressure": round(system_demand[index], 4)}
            for index, window in enumerate(windows)
        ]
        temporal_cache[key] = _temporal_trace(
            selected_events,
            start=int(summary["window_start_seconds"]),
            duration=int(summary["window_duration_seconds"]),
        )
        run_dir = str(shift["run_dir"])
        if run_dir not in team_cache:
            team_cache[run_dir] = _team_trace(
                run_dir,
                start=int(summary["window_start_seconds"]),
                duration=int(summary["window_duration_seconds"]),
            )
    references = _reference_profiles(
        shifts, window_cache, temporal_cache, team_cache
    )

    packets = []
    inventory = []
    for shift in shifts:
        row = shift["map"]
        key = (str(row["run_id"]), str(row["agent_id"]), "reference")
        visible = _visible_evidence(shift, references[key])
        for assessment_variant in ASSESSMENT_VARIANTS:
            packet = _build_packet(
                shift,
                variant="full_trace_persona",
                visible=visible,
                assessment_variant=assessment_variant,
            )
            packets.append(packet)
            inventory.append((packet, shift))

    control_sources: dict[tuple[str, str, str], dict[str, Any]] = {}
    for shift in shifts:
        row = shift["map"]
        stratum = (str(row["scenario"]), str(row["persona_id"]), str(row["role"]))
        current = control_sources.get(stratum)
        if current is None or _digest(row["run_id"], row["agent_id"], "control") < _digest(
            current["map"]["run_id"], current["map"]["agent_id"], "control"
        ):
            control_sources[stratum] = shift
    for shift in [control_sources[key] for key in sorted(control_sources)]:
        row = shift["map"]
        reference = references[(str(row["run_id"]), str(row["agent_id"]), "reference")]
        full_visible = _visible_evidence(shift, reference)
        withheld = _withheld_evidence(str(row["role"]))
        for variant, visible in (
            ("full_trace_neutral", full_visible),
            ("evidence_withheld_persona", withheld),
            ("evidence_withheld_neutral", withheld),
        ):
            packet = _build_packet(shift, variant=variant, visible=visible)
            packets.append(packet)
            inventory.append((packet, shift))

    packet_path = output_dir / "prompt_packets" / "questionnaire_audit_prompt_packets.jsonl"
    inventory_path = output_dir / "questionnaire_audit_inventory.csv"
    preflight_path = output_dir / "questionnaire_audit_preflight.json"
    _write_jsonl(packet_path, packets)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "prompt_id", "control_variant", "assessment_variant", "scenario", "seed", "persona_id",
        "role", "run_id", "agent_id", "estimated_tokens",
    )
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for packet, shift in inventory:
            row = shift["map"]
            writer.writerow(
                {
                    "prompt_id": packet["prompt_id"],
                    "control_variant": packet["control_variant"],
                    "assessment_variant": packet["assessment_variant"],
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "persona_id": row["persona_id"],
                    "role": row["role"],
                    "run_id": row["run_id"],
                    "agent_id": row["agent_id"],
                    "estimated_tokens": packet["estimated_tokens"],
                }
            )
    counts = Counter(packet["control_variant"] for packet in packets)
    full_packets = [
        packet for packet in packets
        if packet["control_variant"] == "full_trace_persona"
        and packet["assessment_variant"] == "balanced"
    ]
    q2_categories_by_scenario = {
        scenario: Counter(
            int(packet["fixed_answers"]["q2"])
            for packet in full_packets
            if packet["trace_evidence"]["metadata"]["scenario_mode"] == scenario
        )
        for scenario in SCENARIOS
    }
    q2_means = {
        scenario: statistics.fmean(
            category
            for category, count in q2_categories_by_scenario[scenario].items()
            for _ in range(count)
        )
        for scenario in SCENARIOS
    }
    q2_construct_sanity = {
        "normal_mean_within_top_quartile_band": (
            15.0 <= q2_means["normal_load"] <= 35.0
        ),
        "normal_category_count_at_least_three": (
            len(q2_categories_by_scenario["normal_load"]) >= 3
        ),
        "high_load_mean_exceeds_normal": (
            q2_means["high_load_high_acuity"] > q2_means["normal_load"]
        ),
        "high_load_mean_at_least_35_percent": (
            q2_means["high_load_high_acuity"] >= 35.0
        ),
    }
    q2_construct_sanity_pass = all(q2_construct_sanity.values())
    q1_percentile_means = {
        scenario: statistics.fmean(
            float(
                packet["llm_visible_evidence"]["item_evidence_cards"]["q1"][
                    "maximum_workload_percentile_across_comparable_days"
                ]
            )
            for packet in full_packets
            if packet["trace_evidence"]["metadata"]["scenario_mode"] == scenario
        )
        for scenario in SCENARIOS
    }
    assessment_counts = Counter(packet["assessment_variant"] for packet in packets)
    measurement_discriminability = _measurement_discriminability(packets)
    preflight = {
        "preflight_pass": (
            len(packets) == 390
            and counts == {
                "full_trace_persona": 300,
                "full_trace_neutral": 30,
                "evidence_withheld_persona": 30,
                "evidence_withheld_neutral": 30,
            }
            and assessment_counts == {
                "balanced": 390,
            }
            and not selection["missing_strata"]
            and all(packet["empirical_answers_visible_to_model"] is False for packet in packets)
            and window_durations == {36000}
            and q2_construct_sanity_pass
            and measurement_discriminability["audit_pass"]
            and q1_percentile_means["high_load_high_acuity"] > q1_percentile_means["normal_load"]
        ),
        "abm_rerun": False,
        "llm_run": False,
        "questionnaire_answers_loaded": False,
        "confidential_hospital_data_in_packets": False,
        "scientific_scope": "aggregate_baseline_convergence_and_intervention_plausibility_audit",
        "questionnaire_audit_version": AUDIT_VERSION,
        "selection": selection,
        "packet_count": len(packets),
        "control_variant_counts": dict(sorted(counts.items())),
        "assessment_variant_counts": dict(sorted(assessment_counts.items())),
        "matched_control_stratum_count": len(control_sources),
        "simulation_window_duration_seconds": sorted(window_durations),
        "workload_window_definition": "ten equal one-hour windows within each ten-hour simulated shift",
        "q2_category_counts_by_scenario": {
            scenario: dict(sorted(counts.items()))
            for scenario, counts in q2_categories_by_scenario.items()
        },
        "q2_category_mean_by_scenario": q2_means,
        "q2_construct_sanity_pass": q2_construct_sanity_pass,
        "q2_construct_sanity_checks": q2_construct_sanity,
        "q1_workload_percentile_mean_by_scenario": q1_percentile_means,
        "measurement_discriminability": measurement_discriminability,
        "maximum_estimated_tokens": max(packet["estimated_tokens"] for packet in packets),
        "prompt_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "prespecified_scientific_checks": {
            "q1_q2_load_direction": "high-load aggregate should exceed normal-load aggregate",
            "q2_construct_sanity": "ordinary-load Q2 should retain at least three categories and center between 15% and 35%, as expected from a top-quartile critical-window definition",
            "evidence_withheld_refusal": "unsupported items should be insufficient_evidence",
            "q8_negative_control": "noise should be insufficient_evidence because acoustics are absent",
            "persona_increment": "full-trace persona responses are compared with matched neutral responses",
            "per_item_rateability": "Q1-Q7 each require at least 90% rateability",
            "mean_equivalence": "for normal-load shifts, at least five of Q1 and Q3-Q7 require absolute mean differences <=0.5; Q2 is excluded because its ordinary-load mean is anchored by construction",
            "mean_equivalence_omnibus": "at least five of six value-eligible normal-load item means must meet their prespecified margins",
            "distribution_compatibility": "for normal-load shifts, at least five of Q1-Q7 must pass finite-sample Wasserstein and category-composition checks plus normalized distance ceiling 0.20; Q2 remains eligible because its across-shift shape is not fixed by the aggregate top-quartile anchor",
            "relationship_compatibility": "the role-standardized Q1-Q7 correlation profile must meet matrix-MAE, profile-correlation, core-pair, and 75% informative-sign checks",
            "severe_failure": "in normal-load shifts, no value-eligible 1-7 item may differ by >1 point",
            "empirical_comparison": "performed locally after inference with role-standardized aggregates; observed answers never enter prompts",
            "intervention_inference_boundary": "baseline convergence can support the plausibility of counterfactual intervention reactions generated by the same grounded architecture, but does not directly validate those unobserved effects",
        },
    }
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
    return preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    result = build_packets(parse_args())
    print(json.dumps(result, indent=2))
    if not result["preflight_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
