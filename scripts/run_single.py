#!/usr/bin/env python3
"""Run one reproducible ABM scenario/condition/seed and write a compact summary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.ablation import _outcome_metrics
from src.calibration import compute_distance_debug, compute_summary_statistics
from src.empirical import classify_zone_supergroup, compute_empirical_summary, describe_empirical_observation_window
from src.simulation import Simulation


HARD_GATE_KEYS = (
    "validation_counted_interactions_beyond_close_threshold_count",
    "synthetic_or_relocated_interaction_coordinate_count",
    "midpoint_logged_validation_interaction_count",
    "nonproximate_logged_interaction_count",
    "task_transition_update_max_logged_distance",
    "staff_staff_logged_max_distance",
    "patient_facing_logged_max_distance",
    "route_failure_count",
    "route_failures",
    "station_check_route_failures",
)


PERCEPTION_FUNNEL_KEYS = (
    "raw_visible_staff_percepts_count",
    "eligible_visible_staff_opportunity_count",
    "selected_visible_staff_opportunity_count",
    "raw_visible_staff_cocpit_related_count",
    "eligible_cocpit_related_opportunity_count",
    "approach_intent_cocpit_related_count",
    "completed_cocpit_related_count",
    "raw_visible_staff_nursta_related_count",
    "eligible_nursta_related_opportunity_count",
    "approach_intent_nursta_related_count",
    "completed_nursta_related_count",
    "visible_with_reason_count",
    "approach_intent_count",
    "approach_success_count",
    "approach_failure_count",
    "missed_not_close_opportunity_count",
    "pressure_suppressed_unrelated_interactions",
    "pressure_gate_active_evaluation_count",
    "pressure_suppression_candidate_count",
    "pressure_suppression_block_count",
    "pressure_suppression_probability_application_count",
    "pressure_created_bed_flow_reasons",
    "pressure_created_high_acuity_reasons",
    "high_acuity_preemption_attempt_count",
    "high_acuity_preemption_success_count",
    "high_acuity_preemption_rejected_count",
    "route_override_count",
    "route_resume_count",
)


INTERACTION_EVENT_FIELDS = (
    "seed",
    "scenario",
    "condition",
    "interaction_id",
    "time_seconds",
    "time_after_warmup_seconds",
    "hour",
    "x",
    "y",
    "zone",
    "zone_group",
    "role_a",
    "role_b",
    "role_pair",
    "is_hcw_hcw",
    "is_patient_facing",
    "topic",
    "reason",
    "duration_seconds",
    "distance_m",
    "counted_for_validation",
    "coordinate_source",
)


SENSITIVITY_PARAMETER_TARGETS = {
    "PERCEIVED_STAFF_INTERACTION_PROBABILITY": ("config", "PERCEIVED_STAFF_INTERACTION_PROBABILITY", float),
    "PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS": (
        "config",
        "PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS",
        int,
    ),
    "POST_TASK_STATION_CHECK_PROBABILITY_NURSE": (
        "config",
        "POST_TASK_STATION_CHECK_PROBABILITY_NURSE",
        float,
    ),
    "SCENARIO_PRESSURE_ACTION_THRESHOLD": (
        "scenario",
        "pressure_action_threshold",
        float,
    ),
}


def _apply_sensitivity_override(args: argparse.Namespace) -> dict[str, Any] | None:
    parameter = getattr(args, "sensitivity_parameter", None)
    if not parameter:
        return None
    required = {
        "sensitivity_level": getattr(args, "sensitivity_level", None),
        "sensitivity_value": getattr(args, "sensitivity_value", None),
        "sensitivity_default": getattr(args, "sensitivity_default", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(
            "Incomplete sensitivity override: --sensitivity-parameter requires "
            "--sensitivity-level, --sensitivity-value, and --sensitivity-default; "
            f"missing {flags}."
        )
    if parameter not in SENSITIVITY_PARAMETER_TARGETS:
        raise ValueError(f"Unsupported sensitivity parameter: {parameter}")
    target_kind, target_name, caster = SENSITIVITY_PARAMETER_TARGETS[parameter]
    value = caster(float(args.sensitivity_value))
    metadata = {
        "parameter": parameter,
        "level": args.sensitivity_level,
        "default_value": _safe_float(args.sensitivity_default),
        "applied_value": value,
        "target_kind": target_kind,
        "target_name": target_name,
    }
    if target_kind == "config":
        setattr(config, target_name, value)
    else:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Scenario sensitivity value must be within [0, 1]: {value}")
        config.SCENARIO_MODES[args.scenario_mode][target_name] = value
    return metadata


def _effective_sensitivity_value(
    simulation: Simulation,
    metadata: dict[str, Any] | None,
) -> Any:
    if not metadata:
        return None
    if metadata["target_kind"] == "scenario":
        return simulation.scenario_definition.get(metadata["target_name"])
    return getattr(config, metadata["target_name"], None)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Counter):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _filter_events(events: list[dict], warmup_seconds: int) -> list[dict]:
    return [
        event for event in events
        if int(event.get("timestamp", event.get("timestep", 0))) >= int(warmup_seconds)
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _role_pair_shares(outcome_metrics: Mapping[str, Any]) -> dict[str, float]:
    distribution = outcome_metrics.get("role_pair_distribution", {})
    return {str(key): _safe_float(value) for key, value in dict(distribution).items()}


def _diagnostic_subset(diagnostics: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _jsonable(diagnostics.get(key, 0)) for key in keys}


def _counts_with_prefix(diagnostics: Mapping[str, Any], prefix: str) -> dict[str, int]:
    return {
        str(key).replace(prefix, ""): _safe_int(value)
        for key, value in diagnostics.items()
        if str(key).startswith(prefix)
    }


def _event_value(event: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = event.get(key)
        if value not in (None, ""):
            return value
    return default


def _csv_float(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _role_pair_label(role_a: str, role_b: str) -> str:
    return "|".join(sorted((role_a, role_b)))


def _interaction_coordinate_source(event: Mapping[str, Any]) -> str:
    if event.get("x") in (None, "") or event.get("y") in (None, ""):
        return "missing"
    if bool(event.get("synthetic_or_relocated_coordinate", False)):
        return "source_flagged_synthetic_or_relocated"
    if bool(event.get("midpoint_logged_validation_interaction", False)):
        return "source_flagged_midpoint"
    if bool(event.get("logged_coordinate_is_actual_contact", False)):
        if str(event.get("role_1", "")) == "Patient":
            return "participant_actual_position_agent_2"
        return "participant_actual_position_agent_1"
    return "event_logged_coordinate_unspecified"


def _interaction_event_rows(
    interactions: list[dict],
    *,
    seed: int,
    scenario: str,
    condition: str,
    warmup_seconds: int,
    scenario_start_hour: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(interactions, start=1):
        timestamp = _safe_float(_event_value(event, "timestamp", "timestep", default=0.0))
        role_a = str(_event_value(event, "role_1", "agent_1_role", default="unknown"))
        role_b = str(_event_value(event, "role_2", "agent_2_role", default="unknown"))
        x_value = event.get("x")
        y_value = event.get("y")
        zone = str(_event_value(event, "zone_id", "zone", default="Outside named zones"))
        zone_group = ""
        if x_value not in (None, "") and y_value not in (None, ""):
            try:
                zone_group = classify_zone_supergroup(zone, float(x_value), float(y_value))
            except (TypeError, ValueError):
                zone_group = ""
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "condition": condition,
                "interaction_id": _event_value(event, "event_id", "interaction_id", default=f"interaction_{index}"),
                "time_seconds": _csv_float(timestamp),
                "time_after_warmup_seconds": _csv_float(max(timestamp - float(warmup_seconds), 0.0)),
                "hour": _csv_float((float(scenario_start_hour) + timestamp / 3600.0) % 24.0),
                "x": _csv_float(x_value),
                "y": _csv_float(y_value),
                "zone": zone,
                "zone_group": zone_group,
                "role_a": role_a,
                "role_b": role_b,
                "role_pair": _role_pair_label(role_a, role_b),
                "is_hcw_hcw": "Patient" not in {role_a, role_b},
                "is_patient_facing": "Patient" in {role_a, role_b},
                "topic": _event_value(event, "topic", default=""),
                "reason": _event_value(
                    event,
                    "reason_for_interaction",
                    "interaction_reason_type",
                    "interaction_source",
                    "interaction_type",
                    default="",
                ),
                "duration_seconds": _csv_float(_event_value(event, "duration_seconds", default="")),
                "distance_m": _csv_float(
                    _event_value(event, "final_logged_distance_m", "distance_m", "initiation_distance_m", default="")
                ),
                "counted_for_validation": True,
                "coordinate_source": _interaction_coordinate_source(event),
            }
        )
    return rows


def _write_interaction_events_csv(
    output_path: Path,
    interactions: list[dict],
    *,
    seed: int,
    scenario: str,
    condition: str,
    warmup_seconds: int,
    scenario_start_hour: int,
) -> None:
    rows = _interaction_event_rows(
        interactions,
        seed=seed,
        scenario=scenario,
        condition=condition,
        warmup_seconds=warmup_seconds,
        scenario_start_hour=scenario_start_hour,
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERACTION_EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows)
    )


_PART3_MOVEMENT_MODES = {
    "returning_home",
    "patrolling",
    "placing_patient",
    "escorting_patient",
    "leading_doctor_to_patient",
}


def _part3_place_label(zone_id: str, cluster: str) -> str:
    """Return a reader-facing place label without changing model geometry."""

    labels = {
        "COCPIT": "COCPIT coordination station",
        "NURSTA": "NURSTA nurse station",
        "OFFSPA": "doctor work area",
        "NUROPE": "nursing work area",
        "NEUOFF": "clinical office",
        "LEFT_ROOM_CLUSTER": "left patient-room area",
        "RIGHT_ROOM_CLUSTER": "right patient-room area",
    }
    if zone_id in labels:
        return labels[zone_id]
    if cluster in labels:
        return labels[cluster]
    if zone_id.startswith("CORR"):
        return "corridor and circulation area"
    if zone_id == "Outside named zones":
        return "shared clinical circulation area"
    return zone_id.replace("_", " ").strip().lower() or "clinical area"


def _part3_destination_type(mode: str) -> str:
    if mode == "moving_to_patient":
        return "patient_bed_access"
    if mode in {"moving_to_doctor", "leading_doctor_to_patient"}:
        return "staff_coordination_or_patient_access"
    if mode == "moving_to_post_task_station_check":
        return "post_task_station_check"
    if mode == "moving_to_secondary_station":
        return "secondary_staff_station"
    if mode == "returning_home":
        return "station_home"
    if mode == "patrolling":
        return "patrol_point"
    if mode in {"placing_patient", "escorting_patient"}:
        return "patient_escort_destination"
    return "other_movement_destination"


def _part3_work_state_label(value: Any) -> str:
    labels = {
        "idle": "available at the staff station",
        "working": "carrying out clinical work",
        "moving_to_patient": "heading to a patient-care area",
        "returning_home": "returning to the staff station",
        "patrolling": "circulating through the department",
        "moving_to_doctor": "seeking a doctor for coordination",
        "leading_doctor_to_patient": "accompanying a doctor to a patient",
        "moving_to_post_task_station_check": "heading to a post-task station check",
        "moving_to_secondary_station": "heading to another staff station",
        "placing_patient": "placing a patient in a care area",
        "escorting_patient": "escorting a patient",
    }
    text = str(value or "unspecified work state")
    return labels.get(text, text.replace("_", " "))


def _is_part3_movement_mode(mode: str) -> bool:
    return mode.startswith("moving_to_") or mode in _PART3_MOVEMENT_MODES


class _Part3ExperienceRecorder:
    """Observe sparse event boundaries without participating in ABM decisions."""

    visible_minimum_seconds = 10
    no_visible_colleague_minimum_seconds = 60

    def __init__(self, simulation: Simulation, start_seconds: int) -> None:
        self.start_seconds = int(start_seconds)
        self.events: list[dict[str, Any]] = []
        self._event_index = 0
        self._states: dict[int, dict[str, Any]] = {}
        self._travel: dict[int, dict[str, Any]] = {}
        self._visibility: dict[int, dict[str, Any]] = {}
        self._initialize(simulation)

    def _persona_id(self, simulation: Simulation, agent_id: int) -> str:
        controller = simulation.part3_cognitive_controller
        return str(getattr(controller, "persona_by_agent", {}).get(agent_id, "unknown"))

    def _state(self, simulation: Simulation, agent: Any) -> dict[str, Any]:
        zone_id = simulation.which_zone(*agent.position) or "Outside named zones"
        cluster = simulation.location_cluster(agent.position)
        destination = getattr(agent, "destination", None)
        return {
            "agent_id": int(agent.gid),
            "agent_name": str(getattr(agent, "name", f"{agent.role} {agent.gid}")),
            "role": str(agent.role),
            "persona_id": self._persona_id(simulation, int(agent.gid)),
            "timestep": int(simulation.timestep),
            "mode": str(getattr(agent, "mode", "unknown")),
            "task": getattr(agent, "current_task_name", None),
            "target_patient_id": getattr(agent, "target_patient_id", None),
            "position": [round(float(agent.position[0]), 3), round(float(agent.position[1]), 3)],
            "zone_id": zone_id,
            "cluster": cluster,
            "place_label": _part3_place_label(zone_id, cluster),
            "destination": (
                [round(float(destination[0]), 3), round(float(destination[1]), 3)]
                if destination is not None
                else None
            ),
        }

    def _next_event_id(self, simulation: Simulation, agent_id: int) -> str:
        self._event_index += 1
        return (
            f"experience:{simulation.random_seed}:{simulation.condition_spec.name}:"
            f"{agent_id}:{self._event_index}"
        )

    def _append(
        self,
        simulation: Simulation,
        state: Mapping[str, Any],
        event_type: str,
        **payload: Any,
    ) -> None:
        self.events.append(
            {
                "experience_schema_version": 1,
                "event_id": self._next_event_id(simulation, int(state["agent_id"])),
                "event_type": event_type,
                "timestep": int(payload.pop("timestep", state["timestep"])),
                "evaluation_seconds": max(
                    int(payload.get("end_timestep", state["timestep"])) - self.start_seconds,
                    0,
                ),
                "agent_id": int(state["agent_id"]),
                "agent_name": str(state["agent_name"]),
                "role": str(state["role"]),
                "persona_id": str(state["persona_id"]),
                "place": {
                    "zone_id": state["zone_id"],
                    "cluster": state["cluster"],
                    "label": state["place_label"],
                    "position": list(state["position"]),
                },
                "mode": state["mode"],
                "task": state["task"],
                "target_patient_id": state["target_patient_id"],
                "directly_observed_simulation_event": True,
                "not_human_data": True,
                **payload,
            }
        )

    def _mutually_visible_partners(
        self, simulation: Simulation
    ) -> dict[int, tuple[tuple[int, str], ...]]:
        partners: dict[int, list[tuple[int, str]]] = defaultdict(list)
        staff = list(simulation.staff_agents)
        for left_index, left in enumerate(staff):
            for right in staff[left_index + 1 :]:
                if not simulation.perception.can_agents_mutually_perceive(
                    left, right, simulation
                ):
                    continue
                partners[int(left.gid)].append((int(right.gid), str(right.role)))
                partners[int(right.gid)].append((int(left.gid), str(left.role)))
        return {
            int(agent.gid): tuple(sorted(partners.get(int(agent.gid), [])))
            for agent in staff
        }

    def _initialize(self, simulation: Simulation) -> None:
        visible = self._mutually_visible_partners(simulation)
        for agent in simulation.staff_agents:
            state = self._state(simulation, agent)
            self._states[int(agent.gid)] = state
            self._append(simulation, state, "experience_window_started")
            self._start_visibility(state, visible[int(agent.gid)])
            if _is_part3_movement_mode(state["mode"]):
                self._start_travel(state)

    def _start_travel(self, state: Mapping[str, Any]) -> None:
        self._travel[int(state["agent_id"])] = {
            "start_timestep": int(state["timestep"]),
            "start_position": list(state["position"]),
            "start_zone_id": state["zone_id"],
            "start_place_label": state["place_label"],
            "last_position": list(state["position"]),
            "distance_meters": 0.0,
            "mode": state["mode"],
            "destination": state["destination"],
            "destination_type": _part3_destination_type(str(state["mode"])),
            "target_patient_id": state["target_patient_id"],
        }

    def _close_travel(
        self,
        simulation: Simulation,
        state: Mapping[str, Any],
        outcome: str,
    ) -> None:
        travel = self._travel.pop(int(state["agent_id"]), None)
        if travel is None:
            return
        self._append(
            simulation,
            state,
            "travel_episode",
            timestep=travel["start_timestep"],
            start_timestep=travel["start_timestep"],
            end_timestep=int(state["timestep"]),
            duration_seconds=max(int(state["timestep"]) - travel["start_timestep"], 0),
            start_position=travel["start_position"],
            end_position=list(state["position"]),
            start_zone_id=travel["start_zone_id"],
            end_zone_id=state["zone_id"],
            start_place_label=travel["start_place_label"],
            end_place_label=state["place_label"],
            distance_meters=round(float(travel["distance_meters"]), 3),
            travel_mode=travel["mode"],
            destination=travel["destination"],
            destination_type=travel["destination_type"],
            travel_outcome=outcome,
        )

    def _start_visibility(
        self,
        state: Mapping[str, Any],
        partners: tuple[tuple[int, str], ...],
    ) -> None:
        self._visibility[int(state["agent_id"])] = {
            "start_timestep": int(state["timestep"]),
            "start_position": list(state["position"]),
            "start_zone_id": state["zone_id"],
            "start_place_label": state["place_label"],
            "partners": partners,
        }

    def _close_visibility(
        self,
        simulation: Simulation,
        state: Mapping[str, Any],
    ) -> None:
        episode = self._visibility.pop(int(state["agent_id"]), None)
        if episode is None:
            return
        duration = max(int(state["timestep"]) - episode["start_timestep"], 0)
        partners = list(episode["partners"])
        minimum = (
            self.visible_minimum_seconds
            if partners
            else self.no_visible_colleague_minimum_seconds
        )
        if duration < minimum:
            return
        self._append(
            simulation,
            state,
            "visibility_episode",
            timestep=episode["start_timestep"],
            start_timestep=episode["start_timestep"],
            end_timestep=int(state["timestep"]),
            duration_seconds=duration,
            visibility_state=(
                "mutually_visible_colleagues" if partners else "no_mutually_visible_colleague"
            ),
            partner_ids=[partner_id for partner_id, _ in partners],
            partner_roles=[role for _, role in partners],
            start_position=episode["start_position"],
            end_position=list(state["position"]),
            start_zone_id=episode["start_zone_id"],
            end_zone_id=state["zone_id"],
            start_place_label=episode["start_place_label"],
            end_place_label=state["place_label"],
        )

    def observe(self, simulation: Simulation) -> None:
        visible = self._mutually_visible_partners(simulation)
        for agent in simulation.staff_agents:
            agent_id = int(agent.gid)
            previous = self._states[agent_id]
            current = self._state(simulation, agent)

            travel = self._travel.get(agent_id)
            if travel is not None:
                travel["distance_meters"] += math.dist(
                    tuple(travel["last_position"]), tuple(current["position"])
                )
                travel["last_position"] = list(current["position"])

            if current["zone_id"] != previous["zone_id"]:
                self._append(
                    simulation,
                    current,
                    "zone_transition",
                    from_zone_id=previous["zone_id"],
                    to_zone_id=current["zone_id"],
                    from_place_label=previous["place_label"],
                    to_place_label=current["place_label"],
                )
            if current["mode"] != previous["mode"]:
                self._append(
                    simulation,
                    current,
                    "mode_transition",
                    from_mode=previous["mode"],
                    to_mode=current["mode"],
                )
            if (
                current["task"] != previous["task"]
                or current["target_patient_id"] != previous["target_patient_id"]
            ):
                self._append(
                    simulation,
                    current,
                    "task_context_transition",
                    from_task=previous["task"],
                    to_task=current["task"],
                    from_patient_id=previous["target_patient_id"],
                    to_patient_id=current["target_patient_id"],
                )

            was_moving = _is_part3_movement_mode(str(previous["mode"]))
            is_moving = _is_part3_movement_mode(str(current["mode"]))
            destination_changed = current["destination"] != previous["destination"]
            if was_moving and (not is_moving or destination_changed):
                self._close_travel(
                    simulation,
                    current,
                    "arrived_or_state_transitioned" if not is_moving else "destination_changed",
                )
            if is_moving and agent_id not in self._travel:
                self._start_travel(current)

            visibility = self._visibility.get(agent_id)
            visibility_key = (
                current["zone_id"],
                visible[agent_id],
            )
            previous_visibility_key = (
                visibility["start_zone_id"],
                visibility["partners"],
            )
            if visibility_key != previous_visibility_key:
                self._close_visibility(simulation, current)
                self._start_visibility(current, visible[agent_id])

            self._states[agent_id] = current

    def finalize(self, simulation: Simulation) -> list[dict[str, Any]]:
        for agent in simulation.staff_agents:
            state = self._state(simulation, agent)
            self._close_travel(simulation, state, "evaluation_window_ended")
            self._close_visibility(simulation, state)
            self._append(simulation, state, "experience_window_ended")
        return sorted(
            self.events,
            key=lambda row: (
                int(row["timestep"]),
                int(row["agent_id"]),
                str(row["event_id"]),
            ),
        )


def _part3_spatial_feature_context(condition: str) -> list[dict[str, Any]]:
    cockpit_transparent = condition in {"cockpit_only", "both"}
    nursta_relocated = condition in {"nursta_only", "both"}
    return [
        {
            "feature_id": "cocpit_partitions",
            "label": "COCPIT partitions",
            "experienced_configuration": (
                "transparent for staff intervisibility"
                if cockpit_transparent
                else "opaque baseline partitions"
            ),
        },
        {
            "feature_id": "nursta_station",
            "label": "NURSTA nurse station",
            "experienced_configuration": (
                "relocated desk obstacle with a separate walkable staff standing area"
                if nursta_relocated
                else "baseline desk and staff standing-area configuration"
            ),
        },
        {
            "feature_id": "patient_rooms",
            "label": "patient rooms",
            "experienced_configuration": "fixed clinical destinations with staff-access points",
        },
        {
            "feature_id": "circulation",
            "label": "corridors and shared circulation",
            "experienced_configuration": "shared movement and opportunistic-contact space",
        },
    ]


def _staff_by_id(simulation: Simulation) -> dict[int, Any]:
    return {int(agent.gid): agent for agent in simulation.staff_agents}


def _experience_common_row(
    simulation: Simulation,
    agent: Any,
    *,
    event_id: str,
    event_type: str,
    timestep: int,
    x: float,
    y: float,
    persona_id: str,
    source_event_ids: list[str],
) -> dict[str, Any]:
    zone_id = simulation.which_zone(float(x), float(y)) or "Outside named zones"
    cluster = simulation.location_cluster((float(x), float(y)))
    return {
        "experience_schema_version": 1,
        "event_id": event_id,
        "event_type": event_type,
        "timestep": int(timestep),
        "agent_id": int(agent.gid),
        "agent_name": str(getattr(agent, "name", f"{agent.role} {agent.gid}")),
        "role": str(agent.role),
        "persona_id": persona_id,
        "place": {
            "zone_id": zone_id,
            "cluster": cluster,
            "label": _part3_place_label(zone_id, cluster),
            "position": [round(float(x), 3), round(float(y), 3)],
        },
        "source_event_ids": source_event_ids,
        "directly_observed_simulation_event": True,
        "not_human_data": True,
    }


def _part3_event_summary(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type", "event"))
    place = str(event.get("place", {}).get("label", "the clinical area"))
    if event_type == "interaction":
        partner = str(event.get("partner_role", "colleague")).lower()
        topic = str(event.get("topic_family") or event.get("topic") or "work coordination")
        return f"A contact with a {partner} occurred in {place} about {topic.replace('_', ' ')}."
    if event_type == "cognitive_opportunity_decision":
        action = str(event.get("selected_action", "responded"))
        partner = str(event.get("partner_role", "colleague")).lower()
        return f"At an optional contact opportunity with a {partner} in {place}, the selected action was {action}."
    if event_type == "missed_opportunity":
        partner = str(event.get("partner_role", "colleague")).lower()
        reason = str(event.get("missed_reason", "the contact did not proceed")).replace("_", " ")
        return f"A potential contact with a {partner} in {place} did not proceed because {reason}."
    if event_type == "workflow_event":
        action = str(event.get("workflow_event_type", "workflow changed")).replace("_", " ")
        task = event.get("workflow_task")
        return f"{action.capitalize()} in {place}" + (f" for {task}." if task else ".")
    if event_type == "travel_episode":
        return (
            f"Travelled {float(event.get('distance_meters', 0.0)):.1f} m from "
            f"{event.get('start_place_label', 'one area')} to "
            f"{event.get('end_place_label', place)} over "
            f"{int(event.get('duration_seconds', 0))} seconds."
        )
    if event_type == "visibility_episode":
        duration = int(event.get("duration_seconds", 0))
        partner_count = len(event.get("partner_ids") or [])
        if partner_count:
            return f"Had mutual visibility with {partner_count} colleague(s) in {place} for {duration} seconds."
        return f"Had no mutually visible colleague in {place} for {duration} seconds."
    if event_type == "zone_transition":
        return f"Moved from {event.get('from_place_label')} to {event.get('to_place_label')}."
    if event_type == "mode_transition":
        return (
            f"Work changed from {_part3_work_state_label(event.get('from_mode'))} "
            f"to {_part3_work_state_label(event.get('to_mode'))} in {place}."
        )
    if event_type == "task_context_transition":
        return f"Task context changed from {event.get('from_task') or 'none'} to {event.get('to_task') or 'none'} in {place}."
    return f"A {event_type.replace('_', ' ')} was recorded in {place}."


def _part3_experience_event_rows(
    simulation: Simulation,
    recorder_rows: list[dict[str, Any]],
    interactions: list[dict],
    missed_opportunities: list[dict],
    workflow_events: list[dict],
) -> list[dict[str, Any]]:
    controller = simulation.part3_cognitive_controller
    persona_by_agent = {
        int(key): str(value)
        for key, value in getattr(controller, "persona_by_agent", {}).items()
    }
    staff_by_id = _staff_by_id(simulation)
    rows = [dict(row) for row in recorder_rows]
    episodes_by_evidence_id = {
        str(episode.get("evidence_id")): episode
        for episode in simulation.part3_decision_episode_log
    }

    for event in interactions:
        source_id = str(event.get("event_id", ""))
        for own_key, partner_key in (("agent_1_id", "agent_2_id"), ("agent_2_id", "agent_1_id")):
            agent_id = _safe_int(event.get(own_key), -1)
            if agent_id not in staff_by_id:
                continue
            partner_id = _safe_int(event.get(partner_key), -1)
            partner_role = str(
                event.get("agent_2_role" if own_key == "agent_1_id" else "agent_1_role", "unknown")
            )
            row = _experience_common_row(
                simulation,
                staff_by_id[agent_id],
                event_id=f"experience:interaction:{source_id}:{agent_id}",
                event_type="interaction",
                timestep=_safe_int(event.get("timestep")),
                x=_safe_float(event.get("x")),
                y=_safe_float(event.get("y")),
                persona_id=persona_by_agent.get(agent_id, "unknown"),
                source_event_ids=[source_id],
            )
            row.update(
                {
                    "partner_id": partner_id if partner_id >= 0 else None,
                    "partner_role": partner_role,
                    "patient_context_id": event.get("patient_id"),
                    "interaction_type": event.get("interaction_type"),
                    "topic_family": event.get("part3_topic_family") or event.get("topic"),
                    "interaction_reason": event.get("part3_selected_reason")
                    or event.get("interaction_reason_type")
                    or event.get("reason_for_interaction"),
                    "duration_seconds": _safe_float(event.get("duration_seconds")),
                    "was_mutually_visible_before_contact": bool(
                        event.get("was_visible_before_interaction", False)
                    ),
                    "model_decision_applied": event.get(
                        "part3_decision_applied_to_interaction"
                    )
                    is True,
                }
            )
            row["event_summary"] = _part3_event_summary(row)
            rows.append(row)

    for event in missed_opportunities:
        agent_id = _safe_int(event.get("agent_id"), -1)
        if agent_id not in staff_by_id:
            continue
        source_id = str(event.get("event_id", ""))
        row = _experience_common_row(
            simulation,
            staff_by_id[agent_id],
            event_id=f"experience:missed:{source_id}:{agent_id}",
            event_type="missed_opportunity",
            timestep=_safe_int(event.get("timestep")),
            x=_safe_float(event.get("x")),
            y=_safe_float(event.get("y")),
            persona_id=persona_by_agent.get(agent_id, "unknown"),
            source_event_ids=[source_id],
        )
        row.update(
            {
                "partner_id": event.get("potential_partner_id"),
                "partner_role": event.get("potential_partner_role"),
                "patient_context_id": event.get("patient_id"),
                "missed_reason": event.get("reason"),
                "duration_seconds": _safe_int(event.get("episode_duration_seconds")),
                "task_state": event.get("task_state"),
            }
        )
        row["event_summary"] = _part3_event_summary(row)
        rows.append(row)

    for event in workflow_events:
        agent_id = _safe_int(event.get("agent_id"), -1)
        if agent_id not in staff_by_id:
            continue
        source_id = str(event.get("event_id", ""))
        row = _experience_common_row(
            simulation,
            staff_by_id[agent_id],
            event_id=f"experience:workflow:{source_id}:{agent_id}",
            event_type="workflow_event",
            timestep=_safe_int(event.get("timestep")),
            x=_safe_float(event.get("x")),
            y=_safe_float(event.get("y")),
            persona_id=persona_by_agent.get(agent_id, "unknown"),
            source_event_ids=[source_id],
        )
        row.update(
            {
                "workflow_event_type": event.get("event_type"),
                "workflow_task": event.get("task_name"),
                "patient_context_id": event.get("patient_id"),
                "esi_level": event.get("esi_level"),
                "old_state": event.get("old_state"),
                "new_state": event.get("new_state"),
            }
        )
        row["event_summary"] = _part3_event_summary(row)
        rows.append(row)

    for decision in getattr(controller, "decision_log", []):
        agent_id = _safe_int(decision.get("agent_id"), -1)
        if agent_id not in staff_by_id:
            continue
        source_id = str(decision.get("evidence_id", ""))
        source_episode = episodes_by_evidence_id.get(source_id, {})
        spatial = source_episode.get("spatial_context", {})
        position = spatial.get("initiator_position") or staff_by_id[agent_id].position
        row = _experience_common_row(
            simulation,
            staff_by_id[agent_id],
            event_id=f"experience:decision:{decision.get('decision_id')}:{agent_id}",
            event_type="cognitive_opportunity_decision",
            timestep=_safe_int(decision.get("timestep")),
            x=_safe_float(position[0]),
            y=_safe_float(position[1]),
            persona_id=persona_by_agent.get(agent_id, "unknown"),
            source_event_ids=[source_id],
        )
        row.update(
            {
                "decision_id": decision.get("decision_id"),
                "partner_id": decision.get("partner_id"),
                "partner_role": decision.get("partner_role"),
                "interaction_type": decision.get("interaction_type"),
                "selected_action": decision.get("selected_action"),
                "selected_reason": decision.get("selected_reason"),
                "topic_family": decision.get("topic_family"),
                "sampled_for_model": bool(decision.get("sampled_for_model")),
                "was_fallback": bool(decision.get("was_fallback")),
                "patient_context_present": bool(
                    decision.get("patient_context_present")
                ),
            }
        )
        row["event_summary"] = _part3_event_summary(row)
        rows.append(row)

    for row in rows:
        row.setdefault("source_event_ids", [])
        row.setdefault("event_summary", _part3_event_summary(row))
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("timestep", 0)),
            int(row.get("agent_id", -1)),
            str(row.get("event_id", "")),
        ),
    )


def _experience_salience(event: Mapping[str, Any]) -> float:
    event_type = str(event.get("event_type", ""))
    base = {
        "interaction": 6.0,
        "cognitive_opportunity_decision": 5.0,
        "missed_opportunity": 4.5,
        "workflow_event": 3.5,
        "travel_episode": 3.0,
        "visibility_episode": 2.5,
        "task_context_transition": 2.0,
        "zone_transition": 1.5,
        "mode_transition": 1.0,
    }.get(event_type, 0.5)
    if event.get("patient_context_id") not in (None, -1, "-1"):
        base += 1.0
    if _safe_int(event.get("esi_level"), 5) in {1, 2}:
        base += 1.0
    if event.get("model_decision_applied") is True:
        base += 1.0
    if event_type == "travel_episode":
        base += min(_safe_float(event.get("distance_meters")) / 10.0, 2.0)
    if event_type == "visibility_episode":
        base += min(_safe_float(event.get("duration_seconds")) / 600.0, 2.0)
    return base


def _part3_spatial_experience_rows(
    simulation: Simulation,
    experience_events: list[dict[str, Any]],
    agent_experience_rows: list[dict[str, Any]],
    *,
    run_id: str,
    scenario: str,
    condition: str,
    seed: int,
) -> list[dict[str, Any]]:
    events_by_agent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in experience_events:
        events_by_agent[_safe_int(event.get("agent_id"), -1)].append(event)
    summary_by_agent = {
        int(row["agent_id"]): row for row in agent_experience_rows
    }
    rows = []
    for agent in simulation.staff_agents:
        agent_id = int(agent.gid)
        events = events_by_agent.get(agent_id, [])
        summary = summary_by_agent.get(agent_id, {})
        place_nodes: dict[str, dict[str, Any]] = {}
        transition_edges: dict[tuple[str, str], dict[str, Any]] = {}
        for event in events:
            place = event.get("place", {})
            place_id = str(place.get("zone_id") or place.get("cluster") or "unknown")
            node = place_nodes.setdefault(
                place_id,
                {
                    "place_id": place_id,
                    "label": str(place.get("label", "clinical area")),
                    "event_count": 0,
                    "interaction_count": 0,
                    "task_event_count": 0,
                    "visibility_seconds": 0,
                    "unique_colleague_ids": set(),
                    "evidence_event_ids": [],
                },
            )
            node["event_count"] += 1
            node["interaction_count"] += int(event.get("event_type") == "interaction")
            node["task_event_count"] += int(
                event.get("event_type") in {"workflow_event", "task_context_transition"}
            )
            if event.get("event_type") == "visibility_episode" and event.get("partner_ids"):
                node["visibility_seconds"] += _safe_int(event.get("duration_seconds"))
                node["unique_colleague_ids"].update(event.get("partner_ids") or [])
            if event.get("partner_id") not in (None, -1, "-1"):
                node["unique_colleague_ids"].add(_safe_int(event.get("partner_id")))
            node["evidence_event_ids"].append(str(event.get("event_id")))

            if event.get("event_type") == "travel_episode":
                from_place = str(event.get("start_place_label") or "clinical area")
                to_place = str(event.get("end_place_label") or "clinical area")
                edge = transition_edges.setdefault(
                    (from_place, to_place),
                    {
                        "from_place": from_place,
                        "to_place": to_place,
                        "trip_count": 0,
                        "distance_meters": 0.0,
                        "duration_seconds": 0,
                        "evidence_event_ids": [],
                    },
                )
                edge["trip_count"] += 1
                edge["distance_meters"] += _safe_float(event.get("distance_meters"))
                edge["duration_seconds"] += _safe_int(event.get("duration_seconds"))
                edge["evidence_event_ids"].append(str(event.get("event_id")))

        zone_dwell = summary.get("zone_dwell_seconds", {})
        for zone_id, seconds in zone_dwell.items():
            cluster = str(zone_id)
            node = place_nodes.setdefault(
                str(zone_id),
                {
                    "place_id": str(zone_id),
                    "label": _part3_place_label(str(zone_id), cluster),
                    "event_count": 0,
                    "interaction_count": 0,
                    "task_event_count": 0,
                    "visibility_seconds": 0,
                    "unique_colleague_ids": set(),
                    "evidence_event_ids": [],
                },
            )
            node["dwell_seconds"] = _safe_int(seconds)

        normalized_nodes = []
        for node in place_nodes.values():
            node["unique_colleague_count"] = len(node.pop("unique_colleague_ids"))
            node["evidence_event_ids"] = sorted(set(node["evidence_event_ids"]))
            node.setdefault("dwell_seconds", 0)
            normalized_nodes.append(node)
        normalized_edges = []
        for edge in transition_edges.values():
            edge["distance_meters"] = round(float(edge["distance_meters"]), 3)
            edge["evidence_event_ids"] = sorted(set(edge["evidence_event_ids"]))
            normalized_edges.append(edge)

        salient = sorted(
            events,
            key=lambda event: (
                _experience_salience(event),
                int(event.get("timestep", 0)),
            ),
            reverse=True,
        )[:24]
        rows.append(
            {
                "experience_map_schema_version": 1,
                "run_id": run_id,
                "scenario": scenario,
                "condition": condition,
                "seed": int(seed),
                "assignment_round": _safe_int(summary.get("assignment_round")),
                "agent_id": agent_id,
                "agent_name": str(getattr(agent, "name", f"{agent.role} {agent_id}")),
                "role": str(agent.role),
                "persona_id": str(summary.get("persona_id", "unknown")),
                "spatial_features": _part3_spatial_feature_context(condition),
                "place_nodes": sorted(normalized_nodes, key=lambda row: row["label"]),
                "transition_edges": sorted(
                    normalized_edges,
                    key=lambda row: (row["from_place"], row["to_place"]),
                ),
                "salient_event_ids": [str(event["event_id"]) for event in salient],
                "experience_event_count": len(events),
                "source_is_simulated_experience": True,
                "generated_interpretation_present": False,
                "not_human_data": True,
            }
        )
    return rows


def _part3_memory_source_rows(
    memory_rows: list[Mapping[str, Any]],
    interactions: list[dict],
) -> list[dict[str, Any]]:
    """Return one evaluator-only source row for every retrieved memory event."""

    required_ids = {str(row["source_event_id"]) for row in memory_rows}
    events_by_id = {
        str(_event_value(event, "event_id", "interaction_id")): event
        for event in interactions
    }
    missing_ids = sorted(required_ids - set(events_by_id))
    if missing_ids:
        raise ValueError(
            "Part 3 memory source events are missing from the realized "
            f"interaction log: {missing_ids}"
        )

    rows = []
    for source_event_id in required_ids:
        event = events_by_id[source_event_id]
        patient_context_id = _event_value(
            event, "interaction_reason_patient_id", "patient_id", default=None
        )
        if patient_context_id in (-1, "-1", "", None):
            patient_context_id = None
        rows.append(
            {
                "source_event_id": source_event_id,
                "timestamp": int(
                    _safe_float(
                        _event_value(event, "timestamp", "timestep", default=0)
                    )
                ),
                "agent_1_id": int(
                    _event_value(event, "agent_1_id", "agent_1", default=-1)
                ),
                "agent_1_role": str(
                    _event_value(event, "agent_1_role", "role_1", default="")
                ),
                "agent_2_id": int(
                    _event_value(event, "agent_2_id", "agent_2", default=-1)
                ),
                "agent_2_role": str(
                    _event_value(event, "agent_2_role", "role_2", default="")
                ),
                "initiated_by": _safe_int(event.get("initiated_by"), -1),
                "patient_context_id": patient_context_id,
                "zone": str(_event_value(event, "zone_id", "zone", default="")),
                "topic_family": str(_event_value(event, "topic", default="")),
                "interaction_type": str(
                    _event_value(event, "interaction_type", default="")
                ),
                "reason_type": _event_value(
                    event, "interaction_reason_type", default=None
                ),
                "outcome": "realized",
                "x": _event_value(event, "x", default=None),
                "y": _event_value(event, "y", default=None),
                "distance_m": _event_value(
                    event,
                    "final_logged_distance_m",
                    "distance_m",
                    "initiation_distance_m",
                    default=None,
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["timestamp"], row["source_event_id"]))


def _part3_experience_snapshot(simulation: Simulation) -> dict[str, Any]:
    """Snapshot logging counters at the start of the Part 3 evidence window."""

    return {
        "movement_distance_by_agent": dict(simulation.movement_distance_by_agent),
        "zone_dwell_by_agent": {
            int(agent_id): dict(counter)
            for agent_id, counter in simulation.zone_dwell_by_agent.items()
        },
        "mode_dwell_by_agent": {
            int(agent_id): dict(counter)
            for agent_id, counter in simulation.mode_dwell_by_agent.items()
        },
        "visible_person_seconds_by_agent": dict(
            simulation.mutual_visibility_exposure_seconds_by_agent
        ),
        "any_visible_seconds_by_agent": dict(
            simulation.any_mutually_visible_staff_seconds_by_agent
        ),
        "visible_person_seconds_by_agent_zone": {
            int(agent_id): dict(counter)
            for agent_id, counter in (
                simulation.mutual_visibility_exposure_seconds_by_agent_zone.items()
            )
        },
    }


def _counter_difference(
    current: Mapping[Any, Any], baseline: Mapping[Any, Any]
) -> dict[str, float]:
    keys = set(current) | set(baseline)
    return {
        str(key): float(current.get(key, 0)) - float(baseline.get(key, 0))
        for key in sorted(keys, key=str)
        if float(current.get(key, 0)) - float(baseline.get(key, 0))
    }


def _part3_agent_experience_rows(
    simulation: Simulation,
    *,
    run_id: str,
    scenario: str,
    condition: str,
    seed: int,
    warmup_seconds: int,
    duration_seconds: int,
    baseline: Mapping[str, Any],
    interactions: list[dict],
    missed_opportunities: list[dict],
) -> list[dict[str, Any]]:
    """Build trace-grounded post-run evidence without generating testimony."""

    controller = simulation.part3_cognitive_controller
    if controller is None:
        return []
    evaluation_seconds = max(duration_seconds - warmup_seconds, 1)
    staff_ids = {int(agent.gid) for agent in simulation.staff_agents}
    decisions = list(controller.decision_log)
    memory_by_agent = controller.memory_exposures_by_agent
    assignment_round = int(controller.sampling_replication_id)
    rows: list[dict[str, Any]] = []

    for agent in simulation.staff_agents:
        agent_id = int(agent.gid)
        persona_id = str(controller.persona_by_agent[agent_id])
        agent_interactions = [
            event
            for event in interactions
            if agent_id
            in {
                _safe_int(_event_value(event, "agent_1_id", "agent_1", default=-1), -1),
                _safe_int(_event_value(event, "agent_2_id", "agent_2", default=-1), -1),
            }
        ]
        staff_interactions = [
            event
            for event in agent_interactions
            if {
                _safe_int(_event_value(event, "agent_1_id", "agent_1", default=-1), -1),
                _safe_int(_event_value(event, "agent_2_id", "agent_2", default=-1), -1),
            }
            <= staff_ids
        ]
        patient_interactions = [
            event for event in agent_interactions if event not in staff_interactions
        ]
        cognitive_interactions = [
            event
            for event in agent_interactions
            if event.get("part3_decision_output_contract")
            == "categorical_causal_v1"
            and event.get("part3_decision_applied_to_interaction") is True
        ]
        initiated_cognitive_interactions = [
            event
            for event in cognitive_interactions
            if _safe_int(event.get("initiated_by"), -1) == agent_id
            and str(event.get("part3_persona_id", "")) == persona_id
        ]
        agent_missed = [
            event
            for event in missed_opportunities
            if _safe_int(event.get("agent_id"), -1) == agent_id
            and str(event.get("reason", "")).startswith("part3_cognitive_")
        ]
        agent_decisions = [
            row for row in decisions if int(row.get("agent_id", -1)) == agent_id
        ]
        sampled_decisions = [
            row for row in agent_decisions if bool(row.get("sampled_for_model"))
        ]
        model_decisions = [
            row for row in sampled_decisions if not bool(row.get("was_fallback"))
        ]
        model_action_counts = Counter(
            str(row["selected_action"]) for row in model_decisions
        )
        memories = list(memory_by_agent.get(agent_id, []))

        movement_m = float(simulation.movement_distance_by_agent.get(agent_id, 0.0)) - float(
            baseline["movement_distance_by_agent"].get(agent_id, 0.0)
        )
        zone_dwell = _counter_difference(
            simulation.zone_dwell_by_agent.get(agent_id, {}),
            baseline["zone_dwell_by_agent"].get(agent_id, {}),
        )
        mode_dwell = _counter_difference(
            simulation.mode_dwell_by_agent.get(agent_id, {}),
            baseline["mode_dwell_by_agent"].get(agent_id, {}),
        )
        visible_person_seconds = float(
            simulation.mutual_visibility_exposure_seconds_by_agent.get(agent_id, 0)
        ) - float(baseline["visible_person_seconds_by_agent"].get(agent_id, 0))
        any_visible_seconds = float(
            simulation.any_mutually_visible_staff_seconds_by_agent.get(agent_id, 0)
        ) - float(baseline["any_visible_seconds_by_agent"].get(agent_id, 0))
        visible_by_zone = _counter_difference(
            simulation.mutual_visibility_exposure_seconds_by_agent_zone.get(
                agent_id, {}
            ),
            baseline["visible_person_seconds_by_agent_zone"].get(agent_id, {}),
        )

        partner_ids = set()
        for event in staff_interactions:
            pair = {
                _safe_int(_event_value(event, "agent_1_id", "agent_1", default=-1), -1),
                _safe_int(_event_value(event, "agent_2_id", "agent_2", default=-1), -1),
            }
            partner_ids.update(pair - {agent_id})

        rows.append(
            {
                "experience_schema_version": 1,
                "run_id": run_id,
                "scenario": scenario,
                "condition": condition,
                "seed": int(seed),
                "assignment_round": assignment_round,
                "agent_id": agent_id,
                "role": str(agent.role),
                "persona_id": persona_id,
                "window_start_seconds": int(warmup_seconds),
                "window_duration_seconds": int(evaluation_seconds),
                "movement_distance_m": max(movement_m, 0.0),
                "zone_dwell_seconds": zone_dwell,
                "mode_dwell_seconds": mode_dwell,
                "visible_colleague_person_minutes": max(
                    visible_person_seconds, 0.0
                )
                / 60.0,
                "mean_mutually_visible_colleagues": max(
                    visible_person_seconds, 0.0
                )
                / evaluation_seconds,
                "minutes_with_any_mutually_visible_colleague": max(
                    any_visible_seconds, 0.0
                )
                / 60.0,
                "share_of_time_with_any_mutually_visible_colleague": min(
                    max(any_visible_seconds / evaluation_seconds, 0.0), 1.0
                ),
                "visible_colleague_person_seconds_by_zone": visible_by_zone,
                "interaction_count": len(agent_interactions),
                "staff_staff_interaction_count": len(staff_interactions),
                "patient_facing_interaction_count": len(patient_interactions),
                "unique_staff_interaction_partner_count": len(partner_ids),
                "interaction_duration_seconds": sum(
                    float(_event_value(event, "duration_seconds", default=0) or 0)
                    for event in agent_interactions
                ),
                "interaction_counts_by_zone": dict(
                    Counter(
                        str(_event_value(event, "zone_id", "zone", default="unknown"))
                        for event in agent_interactions
                    )
                ),
                "interaction_counts_by_topic": dict(
                    Counter(
                        str(_event_value(event, "topic", default="unknown"))
                        for event in agent_interactions
                    )
                ),
                "interaction_counts_by_reason": dict(
                    Counter(
                        str(
                            _event_value(
                                event,
                                "interaction_reason_type",
                                "reason_for_interaction",
                                default="unknown",
                            )
                        )
                        for event in agent_interactions
                    )
                ),
                "eligible_cognitive_opportunity_count": len(agent_decisions),
                "sampled_cognitive_opportunity_count": len(sampled_decisions),
                "model_decision_count": len(model_decisions),
                "model_action_counts": dict(model_action_counts),
                "model_engage_decision_count": int(model_action_counts["engage"]),
                "model_reason_counts": dict(
                    Counter(str(row["selected_reason"]) for row in model_decisions)
                ),
                "model_topic_counts": dict(
                    Counter(str(row["topic_family"]) for row in model_decisions)
                ),
                "model_realized_interaction_count": len(cognitive_interactions),
                "model_realized_initiated_interaction_count": len(
                    initiated_cognitive_interactions
                ),
                "model_realization_yield": (
                    len(initiated_cognitive_interactions)
                    / model_action_counts["engage"]
                    if model_action_counts["engage"]
                    else None
                ),
                "model_missed_opportunity_count": len(agent_missed),
                "grounded_memory_retrieval_count": len(memories),
                "unique_grounded_memory_event_count": len(
                    {str(row["memory_id"]) for row in memories}
                ),
                "not_human_data": True,
                "generated_appraisal_or_interview": False,
            }
        )
    return rows


def run_single(
    args: argparse.Namespace,
    *,
    part3_controller_factory: Callable[[Simulation], object] | None = None,
) -> dict[str, Any]:
    sensitivity_metadata = _apply_sensitivity_override(args)
    run_id = args.run_id or (
        f"{args.scenario_mode}_{args.condition}_seed_{args.seed}_"
        f"{args.duration}s_warmup_{args.warmup_seconds}s"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation = Simulation(
        random_seed=args.seed,
        condition_name=args.condition,
        interaction_backend="rule_stub",
        persona_source=config.PERSONA_SOURCE,
        model_variant=config.BASELINE_MODEL_ID,
        scenario_mode=args.scenario_mode,
        scenario_start_hour=args.scenario_start_hour,
        part3_episode_logging_enabled=args.export_part3_episodes,
        part3_max_decision_episodes=args.part3_max_episodes,
        part3_episode_logging_start_seconds=args.warmup_seconds,
        part3_isolate_exogenous_arrival_stream=(
            part3_controller_factory is not None
        ),
    )
    if part3_controller_factory is not None:
        simulation.part3_cognitive_controller = part3_controller_factory(simulation)
    part3_experience_baseline = None
    part3_experience_recorder: _Part3ExperienceRecorder | None = None
    while simulation.timestep < args.duration:
        if (
            simulation.part3_cognitive_controller is not None
            and part3_experience_baseline is None
            and simulation.timestep >= args.warmup_seconds
        ):
            part3_experience_baseline = _part3_experience_snapshot(simulation)
            part3_experience_recorder = _Part3ExperienceRecorder(
                simulation, args.warmup_seconds
            )
        simulation.step()
        if part3_experience_recorder is not None:
            part3_experience_recorder.observe(simulation)
    if simulation.part3_cognitive_controller is not None and part3_experience_baseline is None:
        part3_experience_baseline = _part3_experience_snapshot(simulation)
        part3_experience_recorder = _Part3ExperienceRecorder(
            simulation, args.warmup_seconds
        )

    part3_recorder_rows = (
        part3_experience_recorder.finalize(simulation)
        if part3_experience_recorder is not None
        else []
    )

    evaluation_interactions = _filter_events(
        list(simulation.interaction_log), args.warmup_seconds
    )
    missed_opportunities = _filter_events(
        list(simulation.missed_opportunity_log), args.warmup_seconds
    )
    evaluation_workflow_events = _filter_events(
        list(simulation.workflow_event_log), args.warmup_seconds
    )
    part3_episodes = sorted(
        _filter_events(
            list(simulation.part3_decision_episode_log), args.warmup_seconds
        ),
        key=lambda event: (
            int(event.get("timestep", 0)),
            str(event.get("evidence_id", "")),
        ),
    )
    part3_logging_summary = {
        "enabled": bool(args.export_part3_episodes),
        "episode_schema_version": 2,
        "behavior_policy": "rule_stub",
        "behavior_unchanged": True,
        "sampling_method": "deterministic_sha256_reservoir",
        "window_start_seconds": int(args.warmup_seconds),
        "candidate_count": int(simulation.part3_decision_episode_candidates_seen),
        "retained_episode_count": len(part3_episodes),
        "maximum_retained_per_run": int(args.part3_max_episodes),
        "active_task_episode_count": sum(
            bool(
                event.get("workflow_context", {}).get(
                    "initiator_is_task_busy", False
                )
            )
            for event in part3_episodes
        ),
        "patient_context_episode_count": sum(
            event.get("workflow_context", {}).get("patient_id")
            not in (None, -1, "-1")
            for event in part3_episodes
        ),
        "selected_action_counts": dict(
            Counter(
                str(event.get("rule_reference", {}).get("selected_action", "unknown"))
                for event in part3_episodes
            )
        ),
        "initiator_role_counts": dict(
            Counter(
                str(event.get("metadata", {}).get("role", "unknown"))
                for event in part3_episodes
            )
        ),
        "reason_type_counts": dict(
            Counter(
                str(event.get("decision_context", {}).get("abm_reason_type", "unknown"))
                for event in part3_episodes
            )
        ),
        "zone_counts": dict(
            Counter(
                str(event.get("spatial_context", {}).get("zone", "unknown"))
                for event in part3_episodes
            )
        ),
    }
    empirical = compute_empirical_summary(
        validation_target=args.validation_target
    ).as_dict()
    empirical_window = describe_empirical_observation_window(
        write_output=False,
        validation_target=args.validation_target,
    )
    run_summary = compute_summary_statistics(evaluation_interactions, simulation)
    distance_components = compute_distance_debug(run_summary, empirical)
    outcome_metrics = _outcome_metrics(
        evaluation_interactions, len(missed_opportunities)
    )
    workflow_health = simulation.workflow_health_summary(
        warmup_seconds=args.warmup_seconds
    )
    diagnostics = dict(simulation.ablation_diagnostics)
    effective_sensitivity_value = _effective_sensitivity_value(
        simulation, sensitivity_metadata
    )
    sensitivity_override_matches_effective = None
    if sensitivity_metadata is not None:
        try:
            sensitivity_override_matches_effective = (
                abs(
                    float(effective_sensitivity_value)
                    - float(sensitivity_metadata["applied_value"])
                )
                <= 1e-12
            )
        except (TypeError, ValueError):
            sensitivity_override_matches_effective = False
    visibility_metrics = run_summary.get(
        "visibility_metrics", simulation.visibility_metrics_snapshot()
    )
    movement_metrics = run_summary.get(
        "movement_metrics", simulation.movement_metrics()
    )
    evaluated_hours = max((args.duration - args.warmup_seconds) / 3600.0, 1e-9)

    interaction_count = len(evaluation_interactions)
    validation_metrics = {
        "interaction_count": interaction_count,
        "evaluated_hours": evaluated_hours,
        "f2f_per_hour": float(interaction_count / evaluated_hours),
        "composite_distance": _safe_float(distance_components.get("weighted_total")),
        "kde_component": _safe_float(distance_components.get("kde_component")),
        "zone_component": _safe_float(distance_components.get("zone_component")),
        "role_pair_component": _safe_float(
            distance_components.get("role_pair_component")
        ),
        "topic_component": _safe_float(distance_components.get("topic_component")),
        "duration_component": _safe_float(
            distance_components.get("duration_component")
        ),
        "hcw_hcw_share": _safe_float(outcome_metrics.get("hcw_hcw_share")),
        "patient_facing_share": _safe_float(
            outcome_metrics.get("patient_facing_share")
        ),
        "station_or_desk_share": _safe_float(
            outcome_metrics.get("station_or_desk_interaction_share")
        ),
        "corridor_share": _safe_float(
            outcome_metrics.get("corridor_interaction_share")
        ),
        "bedside_or_patient_room_share": _safe_float(
            outcome_metrics.get("bedside_or_patient_room_interaction_share")
        ),
        "empirical_f2f_per_hour": _safe_float(
            empirical_window.get("f2f_records_per_hour")
        ),
        "empirical_record_count": _safe_int(
            empirical_window.get("validation_record_count")
        ),
        "empirical_observed_hours": _safe_float(
            empirical_window.get("total_observed_clock_span_hours")
        ),
    }

    pressure_metrics = {
        key: workflow_health.get(key, 0)
        for key in (
            "ed_pressure_index_mean",
            "ed_pressure_index_p95",
            "waiting_pressure_minutes",
            "ordinary_beds_full_minutes",
            "open_ordinary_bed_minutes",
            "zero_waiting_minutes",
            "high_acuity_active_minutes",
            "active_esi1_patient_minutes",
            "active_esi2_patient_minutes",
            "max_concurrent_esi1",
            "max_concurrent_esi1_esi2",
            "mean_concurrent_esi1",
            "mean_concurrent_esi1_esi2",
        )
    }
    patient_flow_metrics = {
        key: workflow_health.get(key, 0)
        for key in (
            "arrival_attempts",
            "arrival_attempts_by_esi",
            "patient_arrivals",
            "admitted_arrivals",
            "admitted_arrivals_by_esi",
            "deferred_arrivals",
            "deferred_arrivals_by_esi",
            "rejected_external_by_esi",
            "accepted_waiting_by_esi",
            "waiting_for_bed_by_esi",
            "bed_assignment_by_esi",
            "completed_patients_by_esi",
            "active_patients_by_esi",
            "backlog_by_esi",
            "time_to_bed_by_esi",
            "time_to_first_doctor_by_esi",
            "time_to_first_nurse_by_esi",
            "average_ordinary_bed_occupancy",
            "p95_ordinary_bed_occupancy",
            "percent_time_all_ordinary_beds_full",
            "percent_time_at_least_one_ordinary_bed_open",
            "ordinary_bed_open_minutes_by_bed",
            "average_special_high_acuity_bed_occupancy",
            "p95_special_high_acuity_bed_occupancy",
            "percent_time_waiting_queue_gt_zero",
            "percent_time_waiting_queue_zero",
            "mean_waiting_queue_length",
            "p95_waiting_queue_length",
        )
    }

    hard_gates = {
        **_diagnostic_subset(diagnostics, HARD_GATE_KEYS),
        "workflow_health_status": workflow_health.get(
            "workflow_health_status", "UNKNOWN"
        ),
        "assignment_repair_events": workflow_health.get("assignment_repair_events", 0),
        "handoff_wait_repair_events": workflow_health.get(
            "handoff_wait_repair_events", 0
        ),
        "transit_stuck_repair_events": workflow_health.get(
            "transit_stuck_repair_events", 0
        ),
        "terminal_stuck_agent_events": workflow_health.get(
            "terminal_stuck_agent_events", 0
        ),
        "unsupported_assignment_rejection_count": workflow_health.get(
            "unsupported_assignment_rejection_count", 0
        ),
        "route_movement_issue_count": workflow_health.get(
            "route_movement_issue_count", 0
        ),
        "stuck_agent_events": workflow_health.get("stuck_agent_events", 0),
        "oscillation_warnings": workflow_health.get("oscillation_warnings", 0),
        "station_check_route_failures": diagnostics.get(
            "station_check_route_failures", 0
        ),
    }

    part3_controller = simulation.part3_cognitive_controller
    part3_closed_loop_summary = (
        part3_controller.summary() if part3_controller is not None else {}
    )
    agent_experience_rows: list[dict[str, Any]] = []
    experience_event_rows: list[dict[str, Any]] = []
    spatial_experience_rows: list[dict[str, Any]] = []
    if part3_controller is not None:
        agent_experience_rows = _part3_agent_experience_rows(
            simulation,
            run_id=run_id,
            scenario=args.scenario_mode,
            condition=args.condition,
            seed=args.seed,
            warmup_seconds=args.warmup_seconds,
            duration_seconds=args.duration,
            baseline=part3_experience_baseline or {},
            interactions=evaluation_interactions,
            missed_opportunities=missed_opportunities,
        )
        experience_event_rows = _part3_experience_event_rows(
            simulation,
            part3_recorder_rows,
            evaluation_interactions,
            missed_opportunities,
            evaluation_workflow_events,
        )
        spatial_experience_rows = _part3_spatial_experience_rows(
            simulation,
            experience_event_rows,
            agent_experience_rows,
            run_id=run_id,
            scenario=args.scenario_mode,
            condition=args.condition,
            seed=args.seed,
        )
        part3_closed_loop_summary["post_run_evidence"] = {
            "agent_summary_count": len(agent_experience_rows),
            "experience_event_count": len(experience_event_rows),
            "spatial_experience_map_count": len(spatial_experience_rows),
            "experience_event_schema_version": 1,
            "spatial_experience_map_schema_version": 1,
            "generated_interpretation_present": False,
            "not_human_data": True,
        }
    payload = {
        "metadata": {
            "run_id": run_id,
            "batch_name": args.batch_name,
            "scenario_mode": args.scenario_mode,
            "condition": args.condition,
            "seed": args.seed,
            "duration_seconds": args.duration,
            "warmup_seconds": args.warmup_seconds,
            "scenario_start_hour": args.scenario_start_hour,
            "validation_target": args.validation_target,
            "staff_count": len(simulation.staff_agents),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "sensitivity_parameter": (
                sensitivity_metadata.get("parameter") if sensitivity_metadata else None
            ),
            "sensitivity_level": (
                sensitivity_metadata.get("level") if sensitivity_metadata else None
            ),
            "sensitivity_default_value": (
                sensitivity_metadata.get("default_value")
                if sensitivity_metadata
                else None
            ),
            "sensitivity_applied_value": (
                sensitivity_metadata.get("applied_value")
                if sensitivity_metadata
                else None
            ),
            "part3_episode_logging_enabled": bool(args.export_part3_episodes),
            "part3_max_decision_episodes": int(args.part3_max_episodes),
            "part3_episode_logging_start_seconds": int(args.warmup_seconds),
            "part3_closed_loop_enabled": part3_controller is not None,
            "part3_exogenous_arrival_stream_isolated": bool(
                simulation.part3_exogenous_arrival_stream_isolated
            ),
            "part3_exogenous_arrival_stream_version": (
                simulation.part3_exogenous_arrival_stream_version
            ),
        },
        "sensitivity": sensitivity_metadata or {},
        "sensitivity_runtime_diagnostics": {
            "sensitivity_parameter": (
                sensitivity_metadata.get("parameter") if sensitivity_metadata else None
            ),
            "sensitivity_level": (
                sensitivity_metadata.get("level") if sensitivity_metadata else None
            ),
            "sensitivity_value": (
                sensitivity_metadata.get("applied_value")
                if sensitivity_metadata
                else None
            ),
            "sensitivity_default": (
                sensitivity_metadata.get("default_value")
                if sensitivity_metadata
                else None
            ),
            "effective_sensitivity_value": effective_sensitivity_value,
            "override_matches_effective_value": sensitivity_override_matches_effective,
            "effective_pressure_station_social_suppression": float(
                simulation.scenario_definition.get(
                    "pressure_station_social_suppression", 0.0
                )
            ),
            "effective_pressure_action_threshold": float(
                simulation.pressure_action_threshold()
            ),
            "pressure_gate_active_evaluation_count": int(
                diagnostics.get("pressure_gate_active_evaluation_count", 0)
            ),
            "pressure_suppression_candidate_count": int(
                diagnostics.get("pressure_suppression_candidate_count", 0)
            ),
            "pressure_suppression_block_count": int(
                diagnostics.get("pressure_suppression_block_count", 0)
            ),
            "pressure_suppression_probability_used": float(
                simulation.scenario_definition.get(
                    "pressure_station_social_suppression", 0.0
                )
            ),
            "pressure_suppression_probability_application_count": int(
                diagnostics.get("pressure_suppression_probability_application_count", 0)
            ),
            "legacy_pressure_suppressed_unrelated_interactions": int(
                diagnostics.get("pressure_suppressed_unrelated_interactions", 0)
            ),
        },
        "scenario_clock": simulation.scenario_clock_metadata(
            args.duration, args.warmup_seconds
        ),
        "scenario_definition": dict(simulation.scenario_definition),
        "validation_metrics": validation_metrics,
        "distance_components": distance_components,
        "interaction_metrics": {
            **outcome_metrics,
            "role_pair_shares": _role_pair_shares(outcome_metrics),
            "interactions_by_source": dict(
                Counter(
                    str(
                        event.get(
                            "reason_for_interaction",
                            event.get("interaction_type", "unknown"),
                        )
                    )
                    for event in evaluation_interactions
                )
            ),
            "interactions_by_type": dict(
                Counter(
                    str(event.get("interaction_type", "unknown"))
                    for event in evaluation_interactions
                )
            ),
        },
        "visibility_metrics": visibility_metrics,
        "experienced_visibility_metrics": {
            "mutual_visibility_counts_by_zone": visibility_metrics.get(
                "mutual_visibility_counts_by_zone", {}
            ),
            "mean_isovist_area_by_key_point": visibility_metrics.get(
                "mean_isovist_area_by_key_point", {}
            ),
            "through_vision": visibility_metrics.get("through_vision", {}),
        },
        "perception_reason_funnel_metrics": {
            **_diagnostic_subset(diagnostics, PERCEPTION_FUNNEL_KEYS),
            "high_acuity_preemption_rejection_reasons": _counts_with_prefix(
                diagnostics,
                "high_acuity_preemption_rejection_reason_",
            ),
            "approach_failure_reasons": _counts_with_prefix(
                diagnostics, "approach_failure_reason_"
            ),
        },
        "movement_dwell_metrics": movement_metrics,
        "patient_flow_metrics": patient_flow_metrics,
        "operational_pressure_metrics": pressure_metrics,
        "workflow_health": workflow_health,
        "workflow_repair_diagnostics": {
            "assignment_repair_events": workflow_health.get(
                "assignment_repair_events", 0
            ),
            "handoff_wait_repair_events": workflow_health.get(
                "handoff_wait_repair_events", 0
            ),
            "transit_stuck_repair_events": workflow_health.get(
                "transit_stuck_repair_events", 0
            ),
            "terminal_stuck_agent_events": workflow_health.get(
                "terminal_stuck_agent_events", 0
            ),
            "unsupported_assignment_rejection_count": workflow_health.get(
                "unsupported_assignment_rejection_count", 0
            ),
            "route_movement_issue_count": workflow_health.get(
                "route_movement_issue_count", 0
            ),
            "assignment_repair_event_details": workflow_health.get(
                "assignment_repair_event_details", []
            ),
            "unsupported_assignment_rejection_details": workflow_health.get(
                "unsupported_assignment_rejection_details", []
            ),
            "route_movement_issue_details": workflow_health.get(
                "route_movement_issue_details", []
            ),
        },
        "hard_gate_diagnostics": hard_gates,
        "part3_episode_logging": part3_logging_summary,
        "part3_closed_loop": part3_closed_loop_summary,
    }
    payload = _jsonable(payload)
    _write_interaction_events_csv(
        output_dir / "interaction_events.csv",
        evaluation_interactions,
        seed=args.seed,
        scenario=args.scenario_mode,
        condition=args.condition,
        warmup_seconds=args.warmup_seconds,
        scenario_start_hour=args.scenario_start_hour,
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    if args.export_part3_episodes:
        _write_jsonl(output_dir / "part3_decision_episodes.jsonl", part3_episodes)
    if part3_controller is not None:
        _write_jsonl(
            output_dir / "part3_cognitive_decisions.jsonl",
            list(part3_controller.decision_log),
        )
        memory_rows = [
            {"agent_id": int(agent_id), **dict(row)}
            for agent_id, rows in sorted(
                part3_controller.memory_exposures_by_agent.items()
            )
            for row in rows
        ]
        _write_jsonl(output_dir / "part3_cognitive_memory.jsonl", memory_rows)
        memory_source_rows = _part3_memory_source_rows(
            memory_rows,
            list(simulation.interaction_log),
        )
        _write_jsonl(
            output_dir / "part3_memory_source_events.jsonl",
            memory_source_rows,
        )
        _write_jsonl(
            output_dir / "part3_provider_errors.jsonl",
            list(part3_controller.provider_error_log),
        )
        provider_log = list(
            getattr(part3_controller.provider, "inference_log", [])
        )
        _write_jsonl(
            output_dir / "part3_provider_inference.jsonl", provider_log
        )
        part3_interactions = [
            event
            for event in evaluation_interactions
            if event.get("part3_decision_output_contract")
            == "categorical_causal_v1"
        ]
        part3_missed = [
            event
            for event in missed_opportunities
            if str(event.get("reason", "")).startswith("part3_cognitive_")
        ]
        _write_jsonl(
            output_dir / "part3_cognitive_interactions.jsonl", part3_interactions
        )
        _write_jsonl(
            output_dir / "part3_cognitive_missed_opportunities.jsonl",
            part3_missed,
        )
        _write_jsonl(
            output_dir / "part3_agent_experience.jsonl",
            agent_experience_rows,
        )
        _write_jsonl(
            output_dir / "part3_experience_events.jsonl",
            experience_event_rows,
        )
        _write_jsonl(
            output_dir / "part3_spatial_experience.jsonl",
            spatial_experience_rows,
        )
    print(output_dir / "summary.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-mode", default=config.DEFAULT_SCENARIO_MODE, choices=sorted(config.SCENARIO_MODES))
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration", type=int, default=config.SIMULATION_DURATION_SECONDS)
    parser.add_argument("--warmup-seconds", type=int, default=0)
    parser.add_argument("--scenario-start-hour", type=int, default=config.DEFAULT_SCENARIO_START_HOUR)
    parser.add_argument("--validation-target", default="care_area", choices=["care_area", "full_empirical"])
    parser.add_argument("--batch-name", default="manual")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sensitivity-parameter",
        choices=sorted(SENSITIVITY_PARAMETER_TARGETS),
        default=None,
    )
    parser.add_argument("--sensitivity-level", choices=("low", "high"), default=None)
    parser.add_argument("--sensitivity-value", type=float, default=None)
    parser.add_argument("--sensitivity-default", type=float, default=None)
    parser.add_argument(
        "--export-part3-episodes",
        action="store_true",
        help="Write sparse logging-only decision evidence; rule behavior remains unchanged",
    )
    parser.add_argument(
        "--part3-max-episodes",
        type=int,
        default=config.PART3_MAX_DECISION_EPISODES_PER_RUN,
    )
    return parser.parse_args()


def main() -> None:
    run_single(parse_args())


if __name__ == "__main__":
    main()
