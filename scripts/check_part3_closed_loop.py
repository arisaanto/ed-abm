#!/usr/bin/env python3
"""Exercise the bounded Part 3 control point without loading an LLM.

This is a plumbing and integrity check, not a scientific experiment. It runs a
short deterministic mock-controlled trajectory and confirms that the default
controller-disabled trajectory remains unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.part3_closed_loop import (
    DeterministicMockCausalProvider,
    Part3ClosedLoopController,
)
from src.personas import balanced_cognitive_persona_assignments
from src.simulation import Simulation


def _persona_assignment(simulation: Simulation) -> dict[int, str]:
    staff_ids = [int(agent.gid) for agent in simulation.staff_agents]
    rows = balanced_cognitive_persona_assignments(staff_ids)
    return {
        int(row["staff_id"]): str(row["persona_id"])
        for row in rows
        if int(row["assignment_round"]) == 1
    }


def _trajectory_snapshot(simulation: Simulation) -> dict[str, Any]:
    return {
        "timestep": int(simulation.timestep),
        "interactions": simulation.interaction_log,
        "workflow_events": simulation.workflow_event_log,
        "missed_opportunities": simulation.missed_opportunity_log,
        "staff": [
            {
                "id": int(agent.gid),
                "position": tuple(float(value) for value in agent.position),
                "mode": getattr(agent, "mode", None),
                "task": getattr(agent, "current_task_name", None),
                "target_patient_id": getattr(agent, "target_patient_id", None),
            }
            for agent in simulation.staff_agents
        ],
        "patients": [
            {
                "id": int(patient.gid),
                "position": tuple(float(value) for value in patient.position),
                "task": getattr(patient, "current_task_name", None),
                "bed_index": getattr(patient, "bed_index", None),
                "discharged": bool(getattr(patient, "discharged", False)),
            }
            for patient in simulation.active_patients
        ],
    }


def _new_simulation(
    args: argparse.Namespace,
    *,
    controller: Part3ClosedLoopController | None = None,
) -> Simulation:
    return Simulation(
        random_seed=args.seed,
        condition_name=args.condition,
        interaction_backend="rule_stub",
        persona_source=config.PERSONA_SOURCE,
        model_variant=config.BASELINE_MODEL_ID,
        scenario_mode=args.scenario,
        scenario_start_hour=10,
        part3_episode_logging_enabled=False,
        part3_cognitive_controller=controller,
    )


def _run_disabled_equivalence(args: argparse.Namespace) -> bool:
    implicit = Simulation(
        random_seed=args.seed,
        condition_name=args.condition,
        interaction_backend="rule_stub",
        persona_source=config.PERSONA_SOURCE,
        model_variant=config.BASELINE_MODEL_ID,
        scenario_mode=args.scenario,
        scenario_start_hour=10,
        part3_episode_logging_enabled=False,
    )
    explicit = _new_simulation(args, controller=None)
    implicit.run(args.equivalence_seconds // config.TIMESTEP_SECONDS)
    explicit.run(args.equivalence_seconds // config.TIMESTEP_SECONDS)
    return _trajectory_snapshot(implicit) == _trajectory_snapshot(explicit)


def check(args: argparse.Namespace) -> dict[str, Any]:
    disabled_equivalence = _run_disabled_equivalence(args)

    simulation = _new_simulation(args)
    controller = Part3ClosedLoopController(
        DeterministicMockCausalProvider(),
        _persona_assignment(simulation),
        max_decisions_per_run=args.max_decisions,
        max_decisions_per_agent=args.max_decisions_per_agent,
    )
    simulation.part3_cognitive_controller = controller
    simulation.run(args.seconds // config.TIMESTEP_SECONDS)

    decisions = controller.decision_log
    decision_actions = Counter(str(row["selected_action"]) for row in decisions)
    contract_events = [
        event
        for event in simulation.interaction_log
        if event.get("part3_decision_output_contract") == "categorical_causal_v1"
    ]
    causal_events = [
        event
        for event in contract_events
        if not bool(event.get("part3_decision_fallback", False))
    ]
    fallback_events = [
        event
        for event in contract_events
        if bool(event.get("part3_decision_fallback", False))
    ]
    expected_decision_keys = {
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
    decision_shape_pass = bool(decisions) and all(
        set(row) == expected_decision_keys for row in decisions
    )
    causal_event_pass = all(
        event.get("part3_selected_action") == "engage"
        and event.get("topic") == event.get("part3_topic_family")
        and event.get("raw_llm_response") is None
        and event.get("part3_free_text_used_as_causal_input") is False
        and event.get("part3_decision_applied_to_interaction") is True
        for event in causal_events
    )
    fallback_event_pass = all(
        event.get("part3_decision_applied_to_interaction") is False
        and event.get("raw_llm_response") is None
        for event in fallback_events
    )
    memory_count = sum(
        len(rows) for rows in controller.memory_exposures_by_agent.values()
    )
    workflow = simulation.workflow_health_summary()
    hard_gate_values = {
        "stuck_agent_events": int(simulation.stuck_agent_events),
        "assignment_repair_events": int(simulation.assignment_repair_events),
        "transit_stuck_repair_events": int(
            workflow.get("transit_stuck_repair_events", 0)
        ),
        "terminal_stuck_agent_events": int(
            workflow.get("terminal_stuck_agent_events", 0)
        ),
        "unsupported_assignment_rejection_count": len(
            simulation.unsupported_assignment_rejections
        ),
        "route_movement_issue_count": len(simulation.route_movement_issues),
        "route_failures": int(
            simulation.ablation_diagnostics.get("route_failures", 0)
        ),
        "route_failure_count": int(
            simulation.ablation_diagnostics.get("route_failure_count", 0)
        ),
        "station_check_route_failures": int(
            simulation.ablation_diagnostics.get("station_check_route_failures", 0)
        ),
        "nonproximate_logged_interaction_count": int(
            simulation.ablation_diagnostics.get(
                "nonproximate_logged_interaction_count", 0
            )
        ),
        "synthetic_or_relocated_interaction_coordinate_count": int(
            simulation.ablation_diagnostics.get(
                "synthetic_or_relocated_interaction_coordinate_count", 0
            )
        ),
        "midpoint_logged_validation_interaction_count": int(
            simulation.ablation_diagnostics.get(
                "midpoint_logged_validation_interaction_count", 0
            )
        ),
        "validation_counted_interactions_beyond_close_threshold_count": int(
            simulation.ablation_diagnostics.get(
                "validation_counted_interactions_beyond_close_threshold_count", 0
            )
        ),
    }
    errors = []
    if not disabled_equivalence:
        errors.append("controller-disabled trajectory changed")
    if not decisions:
        errors.append("mock controller received no eligible decisions")
    if not decision_shape_pass:
        errors.append("causal decision record contains missing or extra fields")
    memory_rows = [
        row
        for rows in controller.memory_exposures_by_agent.values()
        for row in rows
    ]
    if any(
        row.get("source") != "realized_communicative_interaction"
        or row.get("outcome") != "realized"
        or "selected_action" in row
        or "selected_reason" in row
        for row in memory_rows
    ):
        errors.append("closed-loop memory contains choices or ungrounded records")
    controller_summary = controller.summary()
    if (
        controller_summary.get("memory_policy")
        != "grounded_realized_interactions_v1"
        or controller_summary.get("decision_history_used_as_memory") is not False
        or controller_summary.get("generated_reflection_used_as_causal_input")
        is not False
    ):
        errors.append("grounded-memory controller contract failed")
    if not causal_events:
        errors.append("no mock-controlled engage decision reached an interaction")
    if not causal_event_pass:
        errors.append("causal interaction event failed bounded-output checks")
    if not fallback_event_pass:
        errors.append("rule fallback was incorrectly marked as model-applied")
    if len(decision_actions) < 2:
        errors.append("mock check did not exercise multiple action categories")
    if any(hard_gate_values.values()):
        errors.append(f"movement/contact integrity gate failed: {hard_gate_values}")
    if str(workflow.get("workflow_health_status")) == "FAIL":
        errors.append("workflow health failed")

    return {
        "check_name": "part3_closed_loop_mock_integrity",
        "scientific_result": False,
        "controller_disabled_equivalence": disabled_equivalence,
        "scenario": args.scenario,
        "condition": args.condition,
        "seed": args.seed,
        "duration_seconds": args.seconds,
        "controller_summary": controller_summary,
        "decision_action_counts": dict(sorted(decision_actions.items())),
        "causal_interaction_event_count": len(causal_events),
        "fallback_interaction_event_count": len(fallback_events),
        "decision_shape_pass": decision_shape_pass,
        "causal_event_pass": causal_event_pass,
        "fallback_event_pass": fallback_event_pass,
        "hard_gate_values": hard_gate_values,
        "workflow_health_status": workflow.get("workflow_health_status"),
        "errors": errors,
        "verification_pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--equivalence-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario", default="normal_load")
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--max-decisions-per-agent", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if result["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
