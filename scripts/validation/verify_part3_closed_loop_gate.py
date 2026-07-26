#!/usr/bin/env python3
"""Verify one bounded live Part 3 mechanics gate.

The gate establishes technical closed-loop integrity only. It is not evidence
of persona effects, intervention effects, or scientific readiness for Part 3.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.vllm_backend import scientific_packet_prompt_leakage


HARD_ZERO_KEYS = (
    "route_failures",
    "route_failure_count",
    "station_check_route_failures",
    "stuck_agent_events",
    "assignment_repair_events",
    "transit_stuck_repair_events",
    "terminal_stuck_agent_events",
    "unsupported_assignment_rejection_count",
    "route_movement_issue_count",
    "oscillation_warnings",
    "nonproximate_logged_interaction_count",
    "synthetic_or_relocated_interaction_coordinate_count",
    "midpoint_logged_validation_interaction_count",
    "validation_counted_interactions_beyond_close_threshold_count",
)
DECISION_KEYS = {
    "decision_id",
    "evidence_id",
    "persona_id",
    "selected_action",
    "selected_reason",
    "topic_family",
    "evidence_ids",
    "policy_name",
    "was_fallback",
    "rejected_reason",
    "agent_id",
    "timestep",
    "condition",
    "scenario",
    "seed",
    "role",
    "partner_id",
    "partner_role",
    "zone",
    "interaction_type",
    "abm_reason_type",
    "allowed_reasons",
    "allowed_topic_families",
    "patient_context_present",
    "sampling_method",
    "sampling_replication_id",
    "sampling_key",
    "model_decision_sample_rate",
    "sampling_value",
    "sampled_for_model",
    "decision_output_contract",
    "free_text_used_as_causal_input",
}
MEMORY_SOURCE_EVENT_FIELDS = {
    "source_event_id",
    "timestamp",
    "agent_1_id",
    "agent_1_role",
    "agent_2_id",
    "agent_2_role",
    "initiated_by",
    "patient_context_id",
    "zone",
    "topic_family",
    "interaction_type",
    "reason_type",
    "outcome",
    "x",
    "y",
    "distance_m",
}
GROUNDED_MEMORY_FIELDS = {
    "agent_id",
    "decision_id",
    "decision_evidence_id",
    "memory_id",
    "timestamp",
    "seconds_ago",
    "event_type",
    "outcome",
    "partner_id",
    "partner_role",
    "owner_was_initiator",
    "patient_context_id",
    "zone",
    "topic_family",
    "interaction_type",
    "reason_type",
    "importance",
    "same_partner",
    "same_patient_context",
    "retrieval_rank",
    "retrieval_score",
    "retrieval_components",
    "salience_features",
    "retrieval_match_features",
    "source_event_id",
    "support_ids",
    "source",
    "memory_policy",
}
VISIBLE_MEMORY_FIELDS = {
    "memory_id",
    "seconds_ago",
    "outcome",
    "colleague_key",
    "colleague_role",
    "interaction_direction",
    "same_current_colleague",
    "patient_context_key",
    "same_patient_context",
    "topic_family",
    "interaction_type",
    "reason_context",
    "zone",
    "salience_features",
    "retrieval_match_features",
}
MEMORY_POLICY = "grounded_realized_interactions_v1"
ALLOWED_MEMORY_MATCH_FEATURES = {
    "same_colleague",
    "same_patient_context",
    "same_operational_reason",
    "same_interaction_context",
    "same_recent_zone",
}
PROHIBITED_MEMORY_FIELDS = {
    "selected_action",
    "selected_reason",
    "prior_action",
    "prior_reason",
    "summary",
    "partner_name",
    "owner_agent_name",
}
AGENT_EXPERIENCE_KEYS = {
    "experience_schema_version",
    "run_id",
    "scenario",
    "condition",
    "seed",
    "assignment_round",
    "agent_id",
    "role",
    "persona_id",
    "window_start_seconds",
    "window_duration_seconds",
    "movement_distance_m",
    "zone_dwell_seconds",
    "mode_dwell_seconds",
    "visible_colleague_person_minutes",
    "mean_mutually_visible_colleagues",
    "minutes_with_any_mutually_visible_colleague",
    "share_of_time_with_any_mutually_visible_colleague",
    "visible_colleague_person_seconds_by_zone",
    "interaction_count",
    "staff_staff_interaction_count",
    "patient_facing_interaction_count",
    "unique_staff_interaction_partner_count",
    "interaction_duration_seconds",
    "interaction_counts_by_zone",
    "interaction_counts_by_topic",
    "interaction_counts_by_reason",
    "eligible_cognitive_opportunity_count",
    "sampled_cognitive_opportunity_count",
    "model_decision_count",
    "model_action_counts",
    "model_engage_decision_count",
    "model_reason_counts",
    "model_topic_counts",
    "model_realized_interaction_count",
    "model_realized_initiated_interaction_count",
    "model_realization_yield",
    "model_missed_opportunity_count",
    "grounded_memory_retrieval_count",
    "unique_grounded_memory_event_count",
    "not_human_data",
    "generated_appraisal_or_interview",
}

EXPERIENCE_EVENT_REQUIRED_KEYS = {
    "experience_schema_version",
    "event_id",
    "event_type",
    "timestep",
    "agent_id",
    "agent_name",
    "role",
    "persona_id",
    "place",
    "source_event_ids",
    "directly_observed_simulation_event",
    "not_human_data",
    "event_summary",
}

SPATIAL_EXPERIENCE_KEYS = {
    "experience_map_schema_version",
    "run_id",
    "scenario",
    "condition",
    "seed",
    "assignment_round",
    "agent_id",
    "agent_name",
    "role",
    "persona_id",
    "spatial_features",
    "place_nodes",
    "transition_edges",
    "salient_event_ids",
    "experience_event_count",
    "source_is_simulated_experience",
    "generated_interpretation_present",
    "not_human_data",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hard_gate(summary: dict[str, Any], key: str) -> float:
    hard = summary.get("hard_gate_diagnostics", {})
    workflow = summary.get("workflow_health", {})
    try:
        return float(hard.get(key, workflow.get(key, 0)) or 0)
    except (TypeError, ValueError):
        return float("inf")


def verify(
    run_dir: Path, *, require_model_interaction: bool = True
) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "summary.json",
        "interaction_events.csv",
        "part3_cognitive_decisions.jsonl",
        "part3_cognitive_memory.jsonl",
        "part3_memory_source_events.jsonl",
        "part3_provider_errors.jsonl",
        "part3_provider_inference.jsonl",
        "part3_cognitive_interactions.jsonl",
        "part3_cognitive_missed_opportunities.jsonl",
        "part3_agent_experience.jsonl",
        "part3_experience_events.jsonl",
        "part3_spatial_experience.jsonl",
    )
    missing_files = [name for name in required if not (run_dir / name).exists()]
    if missing_files:
        errors.append(f"Missing required files: {missing_files}")

    summary = json.loads((run_dir / "summary.json").read_text())
    decisions = _jsonl(run_dir / "part3_cognitive_decisions.jsonl")
    memory = _jsonl(run_dir / "part3_cognitive_memory.jsonl")
    memory_sources = _jsonl(run_dir / "part3_memory_source_events.jsonl")
    provider_errors = _jsonl(run_dir / "part3_provider_errors.jsonl")
    inference = _jsonl(run_dir / "part3_provider_inference.jsonl")
    interactions = _jsonl(run_dir / "part3_cognitive_interactions.jsonl")
    missed = _jsonl(run_dir / "part3_cognitive_missed_opportunities.jsonl")
    agent_experience = _jsonl(run_dir / "part3_agent_experience.jsonl")
    experience_events = _jsonl(run_dir / "part3_experience_events.jsonl")
    spatial_experience = _jsonl(run_dir / "part3_spatial_experience.jsonl")
    closed_loop = summary.get("part3_closed_loop", {})

    persona_by_agent = {
        int(agent_id): str(persona_id)
        for agent_id, persona_id in closed_loop.get("persona_by_agent", {}).items()
    }
    experience_agent_ids = [int(row.get("agent_id", -1)) for row in agent_experience]
    malformed_experience = [
        row.get("agent_id")
        for row in agent_experience
        if set(row) != AGENT_EXPERIENCE_KEYS
        or int(row.get("experience_schema_version", 0)) != 1
        or row.get("scenario") != summary.get("metadata", {}).get("scenario_mode")
        or row.get("condition") != summary.get("metadata", {}).get("condition")
        or int(row.get("seed", -1)) != int(summary.get("metadata", {}).get("seed", -2))
        or int(row.get("assignment_round", -1))
        != int(closed_loop.get("sampling_replication_id", -2))
        or row.get("persona_id")
        != persona_by_agent.get(int(row.get("agent_id", -1)))
        or int(row.get("window_duration_seconds", 0)) <= 0
        or float(row.get("movement_distance_m", -1)) < 0
        or int(row.get("model_engage_decision_count", -1))
        != int(row.get("model_action_counts", {}).get("engage", 0))
        or int(row.get("model_realized_initiated_interaction_count", -1))
        > int(row.get("model_engage_decision_count", 0))
        or not 0.0
        <= float(row.get("share_of_time_with_any_mutually_visible_colleague", -1))
        <= 1.0
        or row.get("not_human_data") is not True
        or row.get("generated_appraisal_or_interview") is not False
    ]
    if len(agent_experience) != len(persona_by_agent) or len(set(experience_agent_ids)) != len(
        persona_by_agent
    ):
        errors.append("Agent-experience export does not cover each staff member exactly once")
    if malformed_experience:
        errors.append(f"Malformed agent-experience rows: {malformed_experience}")

    experience_event_ids = [str(row.get("event_id", "")) for row in experience_events]
    malformed_event_ids = [
        str(row.get("event_id", ""))
        for row in experience_events
        if not EXPERIENCE_EVENT_REQUIRED_KEYS <= set(row)
        or int(row.get("experience_schema_version", 0)) != 1
        or int(row.get("agent_id", -1)) not in persona_by_agent
        or row.get("persona_id")
        != persona_by_agent.get(int(row.get("agent_id", -1)))
        or row.get("directly_observed_simulation_event") is not True
        or row.get("not_human_data") is not True
        or not isinstance(row.get("place"), dict)
        or not str(row.get("event_summary", "")).strip()
    ]
    if not experience_events:
        errors.append("Part 3 episodic experience export is empty")
    if len(experience_event_ids) != len(set(experience_event_ids)):
        errors.append("Part 3 episodic experience event ids are duplicated")
    if malformed_event_ids:
        errors.append(
            f"Malformed episodic experience rows: {malformed_event_ids[:12]}"
        )

    events_by_agent: dict[int, set[str]] = {
        agent_id: set() for agent_id in persona_by_agent
    }
    event_types_by_agent: dict[int, set[str]] = {
        agent_id: set() for agent_id in persona_by_agent
    }
    for row in experience_events:
        agent_id = int(row.get("agent_id", -1))
        if agent_id in events_by_agent:
            events_by_agent[agent_id].add(str(row.get("event_id", "")))
            event_types_by_agent[agent_id].add(str(row.get("event_type", "")))
    incomplete_event_agents = [
        agent_id
        for agent_id in persona_by_agent
        if not {
            "experience_window_started",
            "experience_window_ended",
        }
        <= event_types_by_agent.get(agent_id, set())
    ]
    if incomplete_event_agents:
        errors.append(
            "Episodic experience windows are incomplete for agents: "
            f"{incomplete_event_agents}"
        )

    map_agent_ids = [int(row.get("agent_id", -1)) for row in spatial_experience]
    malformed_map_agents = []
    for row in spatial_experience:
        agent_id = int(row.get("agent_id", -1))
        known_ids = events_by_agent.get(agent_id, set())
        cited_ids = set(map(str, row.get("salient_event_ids", [])))
        for node in row.get("place_nodes", []):
            cited_ids.update(map(str, node.get("evidence_event_ids", [])))
        for edge in row.get("transition_edges", []):
            cited_ids.update(map(str, edge.get("evidence_event_ids", [])))
        if (
            set(row) != SPATIAL_EXPERIENCE_KEYS
            or int(row.get("experience_map_schema_version", 0)) != 1
            or agent_id not in persona_by_agent
            or row.get("persona_id") != persona_by_agent.get(agent_id)
            or row.get("scenario") != summary.get("metadata", {}).get("scenario_mode")
            or row.get("condition") != summary.get("metadata", {}).get("condition")
            or int(row.get("seed", -1))
            != int(summary.get("metadata", {}).get("seed", -2))
            or int(row.get("assignment_round", -1))
            != int(closed_loop.get("sampling_replication_id", -2))
            or row.get("source_is_simulated_experience") is not True
            or row.get("generated_interpretation_present") is not False
            or row.get("not_human_data") is not True
            or int(row.get("experience_event_count", -1)) != len(known_ids)
            or bool(cited_ids - known_ids)
        ):
            malformed_map_agents.append(agent_id)
    if len(spatial_experience) != len(persona_by_agent) or len(set(map_agent_ids)) != len(
        persona_by_agent
    ):
        errors.append("Spatial-experience maps do not cover each staff member once")
    if malformed_map_agents:
        errors.append(f"Malformed spatial-experience maps: {malformed_map_agents}")

    post_run_evidence = closed_loop.get("post_run_evidence", {})
    if (
        int(post_run_evidence.get("agent_summary_count", -1))
        != len(agent_experience)
        or int(post_run_evidence.get("experience_event_count", -1))
        != len(experience_events)
        or int(post_run_evidence.get("spatial_experience_map_count", -1))
        != len(spatial_experience)
        or post_run_evidence.get("generated_interpretation_present") is not False
    ):
        errors.append("Summary post-run evidence counts or markers do not match files")

    decision_ids = [str(row.get("decision_id")) for row in decisions]
    decisions_by_id = {
        str(row.get("decision_id")): row
        for row in decisions
        if row.get("decision_id") is not None
    }
    if len(decisions_by_id) != len(decision_ids):
        errors.append("Closed-loop decision ids are missing or duplicated")

    if summary.get("metadata", {}).get("part3_closed_loop_enabled") is not True:
        errors.append("Summary does not identify an enabled closed-loop controller")
    if summary.get("metadata", {}).get(
        "part3_exogenous_arrival_stream_isolated"
    ) is not True:
        errors.append("Part 3 exogenous arrival stream was not isolated")
    if summary.get("metadata", {}).get(
        "part3_exogenous_arrival_stream_version"
    ) != "scenario_seed_common_random_numbers_v1":
        errors.append("Unexpected Part 3 exogenous arrival-stream version")
    if closed_loop.get("free_text_used_as_causal_input") is not False:
        errors.append("Closed-loop summary does not prohibit free-text causal input")
    if not decisions:
        errors.append("No live closed-loop decisions were recorded")
    if provider_errors or int(closed_loop.get("provider_error_count", 0)):
        errors.append("One or more provider errors occurred")

    malformed = [
        row.get("decision_id")
        for row in decisions
        if set(row) != DECISION_KEYS
        or row.get("decision_output_contract") != "categorical_causal_v1"
        or row.get("free_text_used_as_causal_input") is not False
        or row.get("selected_action") not in {"engage", "defer", "decline"}
        or row.get("selected_reason") not in row.get("allowed_reasons", [])
        or row.get("topic_family") not in row.get("allowed_topic_families", [])
        or row.get("evidence_ids") != [row.get("evidence_id")]
        or row.get("sampling_method")
        != "deterministic_sha256_matched_opportunity_threshold_v1"
        or not 0.0 < float(row.get("model_decision_sample_rate", 0.0)) <= 1.0
        or not 0.0 <= float(row.get("sampling_value", -1.0)) < 1.0
        or (
            row.get("sampled_for_model") is False
            and (
                row.get("was_fallback") is not True
                or row.get("rejected_reason") != "deterministic_sampling_skip"
            )
        )
    ]
    if malformed:
        errors.append(f"Malformed bounded decisions: {malformed}")

    model_decisions = [row for row in decisions if not row.get("was_fallback")]
    if not model_decisions:
        errors.append("No validated model decision reached the controller")
    if len(inference) != len(model_decisions):
        errors.append("Provider inference count does not match model decision count")

    start_seconds = int(closed_loop.get("start_seconds", -1))
    if start_seconds < 0 or any(int(row.get("timestep", -1)) < start_seconds for row in decisions):
        errors.append("A controller decision occurred before its preregistered start")
    sampling_replication_id = int(
        closed_loop.get("sampling_replication_id", -1)
    )
    if sampling_replication_id < 0 or any(
        int(row.get("sampling_replication_id", -1)) != sampling_replication_id
        for row in decisions
    ):
        errors.append("Decision sampling-replication metadata is inconsistent")
    maximum = int(closed_loop.get("max_decisions_per_run", -1))
    provider_calls = int(closed_loop.get("stats", {}).get("provider_calls", 0))
    if provider_calls <= 0 or provider_calls > maximum:
        errors.append("Provider call count is zero or exceeds the configured run cap")

    prompt_leakage = {}
    model_revisions = set()
    for row in inference:
        packet = row.get("packet", {})
        leakage = scientific_packet_prompt_leakage(packet)
        if leakage:
            prompt_leakage[str(packet.get("prompt_id"))] = leakage
        if packet.get("decision_output_contract") != "categorical_causal_v1":
            errors.append("A provider packet used the wrong output contract")
        raw = row.get("result", {}).get("raw_response", "")
        try:
            raw_payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            errors.append("A provider raw response is not valid JSON")
            continue
        expected_raw_keys = {
            "decision_id",
            "selected_action",
            "selected_reason",
            "topic_family",
            "evidence_ids",
        }
        if set(raw_payload) != expected_raw_keys:
            errors.append("A provider response contains fields outside the categorical contract")
        revision = row.get("model_revision")
        if revision:
            model_revisions.add(str(revision))
    if prompt_leakage:
        errors.append(f"Evaluator-only prompt leakage: {prompt_leakage}")
    if len(model_revisions) != 1:
        errors.append("Exactly one pinned model revision must be recorded")

    if closed_loop.get("memory_policy") != MEMORY_POLICY:
        errors.append("Closed-loop summary used the wrong grounded-memory policy")
    if closed_loop.get("decision_history_used_as_memory") is not False:
        errors.append("Previous categorical choices were used as causal memory")
    if closed_loop.get("generated_reflection_used_as_causal_input") is not False:
        errors.append("Generated reflection text was used as causal memory")
    if int(closed_loop.get("memory_count", -1)) != len(memory):
        errors.append("Closed-loop summary memory count disagrees with export")

    invalid_memory = [
        row
        for row in memory
        if set(row) != GROUNDED_MEMORY_FIELDS
        or bool(set(row) & PROHIBITED_MEMORY_FIELDS)
        or row.get("event_type") != "communicative_interaction"
        or row.get("outcome") != "realized"
        or row.get("source") != "realized_communicative_interaction"
        or row.get("memory_policy") != MEMORY_POLICY
        or not row.get("source_event_id")
        or row.get("support_ids") != [row.get("source_event_id")]
        or int(row.get("seconds_ago", -1)) < 0
        or int(row.get("seconds_ago", -1)) > 7200
        or not row.get("retrieval_match_features")
        or not isinstance(row.get("owner_was_initiator"), bool)
        or not set(row.get("retrieval_match_features", []))
        <= ALLOWED_MEMORY_MATCH_FEATURES
    ]
    if invalid_memory:
        errors.append("Causal memory contains ungrounded or unexpected records")

    memory_sources_by_id = {
        str(row.get("source_event_id")): row for row in memory_sources
    }
    if len(memory_sources_by_id) != len(memory_sources):
        errors.append("Part 3 memory source event ids are missing or duplicated")
    required_source_ids = {str(row.get("source_event_id")) for row in memory}
    if set(memory_sources_by_id) != required_source_ids:
        errors.append(
            "Part 3 memory source ledger does not exactly cover retrieved memory"
        )
    malformed_sources = [
        row.get("source_event_id")
        for row in memory_sources
        if set(row) != MEMORY_SOURCE_EVENT_FIELDS
        or row.get("outcome") != "realized"
        or not row.get("source_event_id")
        or int(row.get("agent_1_id", -1)) < 0
        or int(row.get("agent_2_id", -1)) < 0
        or int(row.get("initiated_by", -1))
        not in {int(row.get("agent_1_id", -1)), int(row.get("agent_2_id", -1))}
    ]
    if malformed_sources:
        errors.append(f"Malformed Part 3 memory source events: {malformed_sources}")

    memory_source_mismatches = []
    for row in memory:
        source = memory_sources_by_id.get(str(row.get("source_event_id")))
        if source is None:
            continue
        decision = decisions_by_id.get(str(row.get("decision_id")))
        if decision is None:
            memory_source_mismatches.append(str(row.get("memory_id")))
            continue
        agent_id = int(row.get("agent_id", -1))
        agent_1_id = int(source.get("agent_1_id", -1))
        agent_2_id = int(source.get("agent_2_id", -1))
        if agent_id == agent_1_id:
            expected_partner_id = agent_2_id
            expected_partner_role = source.get("agent_2_role")
        elif agent_id == agent_2_id:
            expected_partner_id = agent_1_id
            expected_partner_role = source.get("agent_1_role")
        else:
            memory_source_mismatches.append(str(row.get("memory_id")))
            continue
        expected_patient = source.get("patient_context_id")
        observed_patient = row.get("patient_context_id")
        if expected_patient in (-1, "-1", ""):
            expected_patient = None
        if observed_patient in (-1, "-1", ""):
            observed_patient = None
        expected_age = int(decision.get("timestep", -1)) - int(
            source.get("timestamp", -2)
        )
        if (
            int(row.get("timestamp", -1)) != int(source.get("timestamp", -2))
            or expected_age < 0
            or expected_age > 7200
            or int(row.get("seconds_ago", -1)) != expected_age
            or int(row.get("partner_id", -1)) != expected_partner_id
            or row.get("partner_role") != expected_partner_role
            or row.get("owner_was_initiator")
            != (int(source.get("initiated_by", -1)) == agent_id)
            or observed_patient != expected_patient
            or row.get("zone") != source.get("zone")
            or row.get("topic_family") != source.get("topic_family")
            or row.get("interaction_type") != source.get("interaction_type")
            or row.get("reason_type") != source.get("reason_type")
        ):
            memory_source_mismatches.append(str(row.get("memory_id")))
    if memory_source_mismatches:
        errors.append(
            "Retrieved memory disagrees with its realized source event: "
            f"{memory_source_mismatches}"
        )
    memory_by_decision: dict[str, list[dict[str, Any]]] = {}
    for row in memory:
        memory_by_decision.setdefault(str(row.get("decision_id")), []).append(row)
    duplicate_exposures = [
        decision_id
        for decision_id, rows in memory_by_decision.items()
        if len({str(row.get("memory_id")) for row in rows}) != len(rows)
    ]
    if duplicate_exposures:
        errors.append(
            f"A decision received duplicate grounded memories: {duplicate_exposures}"
        )

    inference_by_decision_id: dict[str, dict[str, Any]] = {}
    memory_context_packet_count = 0
    memory_context_record_count = 0
    for row in inference:
        request = row.get("request", {})
        packet = row.get("packet", {})
        result_payload = row.get("result", {})
        decision_id = str(request.get("decision_id"))
        if decision_id in inference_by_decision_id:
            errors.append(f"Duplicate provider inference for {decision_id}")
        inference_by_decision_id[decision_id] = row

        visible_memory = packet.get("llm_visible_evidence", {}).get(
            "memory_state_before", []
        )
        trace_memory = packet.get("trace_evidence", {}).get(
            "memory_state_before", []
        )
        request_memory = request.get("evidence", {}).get(
            "memory_state_before", []
        )
        if visible_memory:
            memory_context_packet_count += 1
            memory_context_record_count += len(visible_memory)
        if len(visible_memory) != len(trace_memory):
            errors.append(f"Visible and trace memory disagree for {decision_id}")
        if trace_memory != request_memory:
            errors.append(f"Request and trace memory disagree for {decision_id}")
        exported_memory = sorted(
            memory_by_decision.get(decision_id, []),
            key=lambda value: int(value.get("retrieval_rank", 0)),
        )
        exported_trace = [
            {
                key: value
                for key, value in row_value.items()
                if key not in {"agent_id", "decision_id", "decision_evidence_id"}
            }
            for row_value in exported_memory
        ]
        if exported_trace != trace_memory:
            errors.append(f"Exported and trace memory disagree for {decision_id}")
        for visible_row, trace_row in zip(visible_memory, trace_memory):
            expected_memory_id = "evidence-" + hashlib.sha256(
                str(trace_row.get("memory_id", "")).encode("utf-8")
            ).hexdigest()[:16]
            if (
                set(visible_row) != VISIBLE_MEMORY_FIELDS
                or bool(set(visible_row) & PROHIBITED_MEMORY_FIELDS)
                or visible_row.get("memory_id") != expected_memory_id
                or visible_row.get("outcome") != "realized"
                or int(visible_row.get("seconds_ago", -1)) < 0
                or not isinstance(visible_row.get("same_current_colleague"), bool)
                or not isinstance(visible_row.get("same_patient_context"), bool)
                or visible_row.get("interaction_direction")
                not in {"initiated_by_self", "received_from_colleague"}
                or visible_row.get("interaction_direction")
                != (
                    "initiated_by_self"
                    if trace_row.get("owner_was_initiator")
                    else "received_from_colleague"
                )
                or visible_row.get("same_current_colleague")
                != trace_row.get("same_partner")
                or visible_row.get("same_patient_context")
                != trace_row.get("same_patient_context")
                or visible_row.get("retrieval_match_features")
                != trace_row.get("retrieval_match_features")
                or not set(visible_row.get("retrieval_match_features", []))
                <= ALLOWED_MEMORY_MATCH_FEATURES
            ):
                errors.append(f"Invalid LLM-visible grounded memory for {decision_id}")

        raw = result_payload.get("raw_response", "")
        try:
            raw_payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        response = result_payload.get("response", {})
        expected_decision = decisions_by_id.get(decision_id, {})
        if raw_payload != response:
            errors.append(f"Parsed provider response differs from raw JSON for {decision_id}")
        if raw_payload.get("decision_id") != packet.get("prompt_id"):
            errors.append(f"Provider response cites the wrong prompt id for {decision_id}")
        if raw_payload.get("evidence_ids") != packet.get("evidence_ids"):
            errors.append(f"Provider response cites the wrong evidence for {decision_id}")
        if any(
            raw_payload.get(raw_key) != expected_decision.get(decision_key)
            for raw_key, decision_key in (
                ("selected_action", "selected_action"),
                ("selected_reason", "selected_reason"),
                ("topic_family", "topic_family"),
            )
        ):
            errors.append(f"Provider response differs from applied decision for {decision_id}")

    if set(inference_by_decision_id) != {
        str(row.get("decision_id")) for row in model_decisions
    }:
        errors.append("Provider inference does not map one-to-one to model decisions")
    repeated_model_agents = {
        row.get("agent_id")
        for row in model_decisions
        if sum(
            other.get("agent_id") == row.get("agent_id") for other in model_decisions
        )
        > 1
    }
    # Grounded retrieval is deliberately context-matched, so a run may have
    # repeated decisions without a relevant prior realized interaction. The
    # aggregate study verifier enforces memory exposure across every cell.
    if memory_context_record_count != len(memory):
        errors.append("Visible memory exposure count disagrees with memory export")

    invalid_interactions: list[str] = []
    model_applied_interactions: list[dict[str, Any]] = []
    fallback_rule_interactions: list[dict[str, Any]] = []
    for row in interactions:
        event_id = str(row.get("event_id"))
        linked_decision = decisions_by_id.get(str(row.get("part3_decision_id")))
        fallback = row.get("part3_decision_fallback")
        applied = row.get("part3_decision_applied_to_interaction")
        invalid = (
            row.get("part3_decision_output_contract") != "categorical_causal_v1"
            or row.get("raw_llm_response") is not None
            or row.get("part3_free_text_used_as_causal_input") is not False
            or row.get("part3_selected_action") != "engage"
            or linked_decision is None
            or fallback is not bool(linked_decision and linked_decision.get("was_fallback"))
            or applied is not (not bool(fallback))
        )
        if linked_decision is not None and any(
            row.get(event_key) != linked_decision.get(decision_key)
            for event_key, decision_key in (
                ("part3_evidence_id", "evidence_id"),
                ("part3_persona_id", "persona_id"),
                ("part3_selected_action", "selected_action"),
                ("part3_selected_reason", "selected_reason"),
                ("part3_topic_family", "topic_family"),
            )
        ):
            invalid = True
        if applied is True and row.get("topic") != row.get("part3_topic_family"):
            invalid = True
        if invalid:
            invalid_interactions.append(event_id)
        elif applied is True:
            model_applied_interactions.append(row)
        else:
            fallback_rule_interactions.append(row)
    if invalid_interactions:
        errors.append(f"Invalid model-linked interactions: {invalid_interactions}")
    if require_model_interaction and not model_applied_interactions:
        errors.append("No live model engage decision reached an interaction")

    model_nonengage_ids = {
        str(row.get("decision_id"))
        for row in model_decisions
        if row.get("selected_action") != "engage"
    }
    model_nonengage_interactions = [
        row
        for row in model_applied_interactions
        if str(row.get("part3_decision_id")) in model_nonengage_ids
    ]
    if model_nonengage_interactions:
        errors.append("A model defer/decline decision produced an applied interaction")

    model_missed_opportunities: list[dict[str, Any]] = []
    fallback_missed_opportunities: list[dict[str, Any]] = []
    invalid_missed_opportunities: list[str] = []
    for row in missed:
        state = row.get("perception_state", {})
        linked_decision = decisions_by_id.get(str(state.get("part3_decision_id")))
        expected_action = str(row.get("reason", "")).removeprefix(
            "part3_cognitive_"
        )
        if (
            linked_decision is None
            or expected_action not in {"defer", "decline"}
            or linked_decision.get("selected_action") != expected_action
            or any(
                state.get(state_key) != linked_decision.get(decision_key)
                for state_key, decision_key in (
                    ("part3_persona_id", "persona_id"),
                    ("part3_selected_reason", "selected_reason"),
                    ("part3_topic_family", "topic_family"),
                )
            )
        ):
            invalid_missed_opportunities.append(str(row.get("event_id")))
        elif bool(linked_decision.get("was_fallback")):
            fallback_missed_opportunities.append(row)
        else:
            model_missed_opportunities.append(row)
    if invalid_missed_opportunities:
        errors.append(
            "Invalid cognitive missed-opportunity links: "
            f"{invalid_missed_opportunities}"
        )

    workflow_status = summary.get("workflow_health", {}).get(
        "workflow_health_status", "UNKNOWN"
    )
    if workflow_status != "PASS":
        errors.append(f"Workflow status is {workflow_status}, expected PASS")
    hard_gates = {key: _hard_gate(summary, key) for key in HARD_ZERO_KEYS}
    nonzero_hard_gates = {key: value for key, value in hard_gates.items() if value != 0}
    if nonzero_hard_gates:
        errors.append(f"Nonzero integrity gates: {nonzero_hard_gates}")

    if not model_applied_interactions and not model_missed_opportunities:
        errors.append("No model decision reached a recorded interaction outcome")

    result = {
        "verification_pass": not errors,
        "gate_scope": "technical_closed_loop_mechanics_only",
        "scientific_result": False,
        "exogenous_arrival_stream_isolated": summary.get("metadata", {}).get(
            "part3_exogenous_arrival_stream_isolated"
        ),
        "exogenous_arrival_stream_version": summary.get("metadata", {}).get(
            "part3_exogenous_arrival_stream_version"
        ),
        "errors": errors,
        "decision_count": len(decisions),
        "model_decision_count": len(model_decisions),
        "fallback_count": sum(bool(row.get("was_fallback")) for row in decisions),
        "provider_call_count": provider_calls,
        "provider_calls_by_window": closed_loop.get(
            "provider_calls_by_window", {}
        ),
        "provider_error_count": len(provider_errors),
        "memory_count": len(memory),
        "memory_source_event_count": len(memory_sources),
        "memory_source_lineage_pass": (
            not memory_source_mismatches
            and not malformed_sources
            and len(memory_sources_by_id) == len(memory_sources)
            and set(memory_sources_by_id) == required_source_ids
        ),
        "memory_policy": MEMORY_POLICY,
        "decision_history_used_as_memory": False,
        "generated_reflection_used_as_causal_input": False,
        "agent_experience_schema_version": 1,
        "agent_experience_row_count": len(agent_experience),
        "agent_experience_complete": (
            len(agent_experience) == len(persona_by_agent)
            and len(set(experience_agent_ids)) == len(persona_by_agent)
            and not malformed_experience
        ),
        "experience_event_schema_version": 1,
        "experience_event_count": len(experience_events),
        "experience_event_export_pass": (
            bool(experience_events)
            and len(experience_event_ids) == len(set(experience_event_ids))
            and not malformed_event_ids
            and not incomplete_event_agents
        ),
        "spatial_experience_map_schema_version": 1,
        "spatial_experience_map_count": len(spatial_experience),
        "spatial_experience_map_export_pass": (
            len(spatial_experience) == len(persona_by_agent)
            and len(set(map_agent_ids)) == len(persona_by_agent)
            and not malformed_map_agents
        ),
        "cognitive_tagged_interaction_count": len(interactions),
        "model_applied_interaction_count": len(model_applied_interactions),
        "fallback_rule_interaction_count": len(fallback_rule_interactions),
        "cognitive_tagged_missed_opportunity_count": len(missed),
        "model_missed_opportunity_count": len(model_missed_opportunities),
        "fallback_rule_missed_opportunity_count": len(
            fallback_missed_opportunities
        ),
        "model_engage_decision_count": sum(
            row.get("selected_action") == "engage" for row in model_decisions
        ),
        "model_nonengage_decision_count": sum(
            row.get("selected_action") != "engage" for row in model_decisions
        ),
        "memory_context_packet_count": memory_context_packet_count,
        "memory_context_record_count": memory_context_record_count,
        "repeated_model_agent_count": len(repeated_model_agents),
        "grounded_memory_exercised": memory_context_packet_count > 0,
        "action_counts": dict(Counter(row.get("selected_action") for row in model_decisions)),
        "model_revisions": sorted(model_revisions),
        "workflow_status": workflow_status,
        "hard_gates": hard_gates,
        "free_text_used_as_causal_input": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = verify(Path(args.run_dir))
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
