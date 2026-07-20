"""Condition experiment execution and comparative metric reporting."""

from __future__ import annotations

import csv
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, Mapping, Optional

import config

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(config.MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import compute_kde_grid, cosine_similarity
from src.calibration import aggregate_run_summaries, compute_summary_statistics, distance_to_empirical
from src.conditions import ConditionManager, get_condition_spec
from src.empirical import compute_empirical_summary
from src.environment import Environment
from src.simulation import Simulation
from src.ablation import _aggregate_workflow_health, _outcome_metrics


def _duration_label(duration_seconds: int) -> str:
    hours = duration_seconds / 3600.0
    if abs(hours - round(hours)) <= 1e-9:
        return f"{int(round(hours))}h"
    return f"{int(round(duration_seconds / 60.0))}m"


def _labeled_output_path(base_path: Path, duration_label: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{duration_label}{base_path.suffix}")


@contextmanager
def _temporary_config_updates(updates: Mapping[str, object]) -> Iterator[None]:
    previous = {key: getattr(config, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(config, key, value)


def _parameter_to_config_overrides(params: Mapping[str, object]) -> Dict[str, object]:
    overrides = {}
    for key, value in params.items():
        if key == "secondary_station_probability":
            overrides["NURSE_SECONDARY_STATION_PROBABILITY"] = float(value)
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
        elif key == "secondary_station_duration_mean":
            overrides["SECONDARY_STATION_DURATION_MEAN"] = float(value)
    return overrides


def _plot_condition_heatmap(axis, simulation: Simulation, interactions: Iterable[dict], title: str) -> None:
    simulation.condition_manager.draw_floorplan(axis)
    simulation.condition_manager.draw_zone_overlays(axis)
    points_xy = [(float(event["x"]), float(event["y"])) for event in interactions]
    if points_xy:
        kde_grid = compute_kde_grid(
            points_xy,
            simulation.environment.plot_bounds,
            resolution=1.0,
            bandwidth=1.5,
        )
        min_x, max_x, min_y, max_y = simulation.environment.plot_bounds
        axis.imshow(
            np.asarray(kde_grid, dtype=float),
            extent=(min_x, max_x, min_y, max_y),
            origin="lower",
            cmap=config.HEATMAP_CMAP,
            alpha=config.HEATMAP_ALPHA,
            zorder=4,
            aspect="auto",
        )
    axis.set_xlim(*simulation.environment.plot_bounds[:2])
    axis.set_ylim(*simulation.environment.plot_bounds[2:])
    axis.set_aspect("equal")
    axis.set_title(title)


def _mean_approach_distance(interactions: Iterable[dict], predicate) -> float:
    distances = [
        float(event.get("approach_distance_meters", 0.0))
        for event in interactions
        if predicate(event)
    ]
    return float(mean(distances)) if distances else 0.0


def _zone_interaction_count(interactions: Iterable[dict], zone_ids: set[str]) -> int:
    return sum(
        1 for event in interactions
        if str(event.get("zone_id", "")) in zone_ids
    )


def _corr04_corr05_interaction_count(interactions: Iterable[dict]) -> int:
    corridor_door_points = [config.ROUTING_WAYPOINTS["corr04_door"], config.ROUTING_WAYPOINTS["corr05_door"]]
    door_radius_meters = 3.0
    count = 0
    for event in interactions:
        if str(event.get("zone_id", "")) in {"CORR04", "CORR05"}:
            count += 1
            continue
        event_point = (float(event["x"]), float(event["y"]))
        if any(
            np.hypot(event_point[0] - door[0], event_point[1] - door[1]) <= door_radius_meters
            for door in corridor_door_points
        ):
            count += 1
    return count


def _interaction_cloud_metrics(interactions: Iterable[dict]) -> Dict[str, float]:
    interactions = list(interactions)
    if not interactions:
        return {
            "interaction_cloud_centroid_x": 0.0,
            "interaction_cloud_centroid_y": 0.0,
            "hcw_hcw_interaction_count": 0,
            "hcw_patient_interaction_count": 0,
            "nurse_doctor_interaction_count": 0,
            "nurse_doctor_mean_duration_seconds": 0.0,
            "mean_nurse_doctor_approach_distance": 0.0,
            "mean_cocpit_interaction_approach_distance": 0.0,
            "coordination_nurse_nurse_count": 0,
            "corridor_interaction_share": 0.0,
            "corr04_corr05_interaction_count": 0,
        }

    centroid_x = mean(float(event["x"]) for event in interactions)
    centroid_y = mean(float(event["y"]) for event in interactions)
    hcw_hcw = [
        event for event in interactions
        if event["role_1"] != "Patient" and event["role_2"] != "Patient"
    ]
    hcw_patient = [
        event for event in interactions
        if "Patient" in {event["role_1"], event["role_2"]}
    ]
    nurse_doctor = [
        event for event in hcw_hcw
        if {event["role_1"], event["role_2"]} == {"Nurse", "Doctor"}
    ]
    coordination_nurse = [
        event for event in hcw_hcw
        if {event["role_1"], event["role_2"]} == {"CoordinationNurse", "Nurse"}
    ]
    corridor_count = sum(
        1 for event in interactions
        if str(event.get("interaction_type", "")).startswith("opportunistic_corridor")
    )
    return {
        "interaction_cloud_centroid_x": float(centroid_x),
        "interaction_cloud_centroid_y": float(centroid_y),
        "hcw_hcw_interaction_count": len(hcw_hcw),
        "hcw_patient_interaction_count": len(hcw_patient),
        "nurse_doctor_interaction_count": len(nurse_doctor),
        "nurse_doctor_mean_duration_seconds": (
            float(mean(float(event["duration_seconds"]) for event in nurse_doctor))
            if nurse_doctor
            else 0.0
        ),
        "mean_nurse_doctor_approach_distance": _mean_approach_distance(
            interactions,
            lambda event: {event["role_1"], event["role_2"]} == {"Nurse", "Doctor"},
        ),
        "mean_cocpit_interaction_approach_distance": _mean_approach_distance(
            interactions,
            lambda event: str(event.get("zone_id", "")) == "COCPIT",
        ),
        "coordination_nurse_nurse_count": len(coordination_nurse),
        "corridor_interaction_share": corridor_count / max(len(interactions), 1),
        "corr04_corr05_interaction_count": _corr04_corr05_interaction_count(interactions),
    }


def _visible_staff_from_point(simulation: Simulation, point: tuple[float, float]) -> int:
    return sum(
        1
        for agent in simulation.staff_agents
        if simulation.perception._has_visibility(point, agent.position)
    )


def _visibility_comparison_metrics(simulation: Simulation) -> Dict[str, object]:
    cockpit_point = simulation.condition_manager.zone_centroid("COCPIT")
    corr07_point = config.PATIENT_ENTRY_POINT
    corr01_point = simulation.condition_manager.zone_centroid("CORR01")
    corr03_point = simulation.condition_manager.zone_centroid("CORR03")
    nurope_point = simulation.condition_manager.zone_centroid("NUROPE")
    snapshot = simulation.visibility_metrics_snapshot()
    return {
        "mean_isovist_area_cocpit_centroid": float(snapshot["mean_isovist_area_by_key_point"]["COCPIT"]),
        "mean_isovist_area_nursta_centroid": float(snapshot["mean_isovist_area_by_key_point"]["NURSTA"]),
        "mean_isovist_area_corr07_entry": float(
            simulation._polygon_area(
                simulation.perception.compute_isovist(corr07_point, heading=0.0, fov_degrees=360.0)
            )
        ),
        "cocpit_to_corr01_through_vision": simulation.perception._has_visibility(cockpit_point, corr01_point),
        "cocpit_to_corr03_through_vision": simulation.perception._has_visibility(cockpit_point, corr03_point),
        "cocpit_to_nurope_through_vision": simulation.perception._has_visibility(cockpit_point, nurope_point),
        "agents_visible_from_cocpit_mean": float(_visible_staff_from_point(simulation, cockpit_point)),
    }


def _nurse_dwell_centroid_y(summary: Mapping[str, object], simulation: Simulation) -> float:
    nurse_dwell = summary.get("movement_metrics", {}).get("zone_dwell_by_role", {}).get("Nurse", {})
    weighted_positions = []
    weights = []
    for zone_id, seconds in nurse_dwell.items():
        if zone_id == "Outside named zones":
            continue
        if zone_id in simulation.condition_manager.zone_centroids:
            weighted_positions.append(simulation.condition_manager.zone_centroid(zone_id)[1])
            weights.append(float(seconds))
    if not weights:
        return 0.0
    return float(np.average(weighted_positions, weights=weights))


def _mean_duration_for_types(interactions: Iterable[dict], allowed_types: set[str]) -> float:
    durations = [
        float(event["duration_seconds"])
        for event in interactions
        if str(event.get("interaction_type")) in allowed_types
    ]
    return float(mean(durations)) if durations else 0.0


def _build_transcript_sample(interactions: Iterable[dict], limit: int = 20) -> list[dict]:
    sample = []
    for event in interactions:
        if "Patient" in {event["role_1"], event["role_2"]}:
            continue
        sample.append(
            {
                "timestep": int(event["timestamp"]),
                "simulated_time_hhmm": f"{int(event['timestamp']) // 3600:02d}:{(int(event['timestamp']) % 3600) // 60:02d}",
                "role_1": event["role_1"],
                "role_2": event["role_2"],
                "zone": event.get("zone_id", "Outside named zones"),
                "x": float(event["x"]),
                "y": float(event["y"]),
                "topic": event.get("topic"),
                "duration_seconds": float(event.get("duration_seconds", 0.0)),
                "interaction_type": event.get("interaction_type"),
                "memory_summary": event.get("memory_summary"),
                "backend_used": event.get("backend_used", event.get("interaction_backend")),
                "was_visible_before_interaction": bool(event.get("was_visible_before_interaction", False)),
                "isovist_area_at_initiation": float(event.get("isovist_area_at_initiation", 0.0)),
                "approach_distance_meters": float(event.get("approach_distance_meters", 0.0)),
                "condition_name": event.get("condition_name"),
            }
        )
        if len(sample) >= limit:
            break
    return sample


def _format_delta_table(results: Mapping[str, Mapping[str, object]]) -> str:
    baseline = results["baseline"]
    def _pct_delta(metric_value: float, baseline_value: float) -> str:
        if abs(baseline_value) <= 1e-9:
            return "="
        return f"{((metric_value - baseline_value) / baseline_value) * 100.0:+.1f}%"

    lines = [
        "Metric                          | Baseline | COCPIT | NURSTA | Both",
        "--------------------------------|----------|--------|--------|------",
    ]
    comparison_specs = [
        (
            "Nurse-Doctor interactions/hr",
            lambda payload: payload["interaction_cloud_metrics"]["nurse_doctor_interaction_count"] / max(
                payload.get("duration_seconds", 3600) / 3600.0,
                1e-9,
            ),
        ),
        (
            "COCPIT isovist area (m²)",
            lambda payload: payload["visibility_comparison"]["mean_isovist_area_cocpit_centroid"],
        ),
        (
            "NURSTA isovist area (m²)",
            lambda payload: payload["visibility_comparison"]["mean_isovist_area_nursta_centroid"],
        ),
        (
            "Interaction centroid y (m)",
            lambda payload: payload["interaction_cloud_metrics"]["interaction_cloud_centroid_y"],
        ),
        (
            "Mean nurse-doctor duration (s)",
            lambda payload: payload["interaction_cloud_metrics"]["nurse_doctor_mean_duration_seconds"],
        ),
        (
            "Corridor interaction share",
            lambda payload: payload["interaction_cloud_metrics"]["corridor_interaction_share"],
        ),
    ]

    for metric_name, extractor in comparison_specs:
        baseline_value = extractor(baseline)
        row = [metric_name, f"{baseline_value:.2f}"]
        for condition_name in ("cockpit_only", "nursta_only", "both"):
            metric_value = extractor(results[condition_name])
            if metric_name in {
                "Interaction centroid y (m)",
                "Mean nurse-doctor duration (s)",
                "Corridor interaction share",
            }:
                row.append(f"{metric_value - baseline_value:+.2f}")
            else:
                row.append(_pct_delta(metric_value, baseline_value))
        lines.append(
            f"{row[0]:<31}| {row[1]:>8} | {row[2]:>6} | {row[3]:>6} | {row[4]:>6}"
        )
    return "\n".join(lines)


def _visibility_graph_metrics(simulation: Simulation) -> Dict[str, float]:
    graph = simulation.perception.build_visibility_graph(
        simulation.environment,
        resolution=config.VISIBILITY_GRAPH_RESOLUTION_METERS,
    )
    integration = simulation.perception.compute_visual_integration(graph)
    if not integration:
        return {"mean_visual_integration": 0.0, "max_visual_integration": 0.0}
    values = list(integration.values())
    return {
        "mean_visual_integration": float(mean(values)),
        "max_visual_integration": float(max(values)),
    }


def _events_after_warmup(events: Iterable[dict], warmup_seconds: int, duration_seconds: int) -> list[dict]:
    return [
        event for event in events
        if warmup_seconds <= int(event.get("timestamp", event.get("timestep", 0))) < duration_seconds
    ]


def _aggregate_part2_runs(
    *,
    condition_name: str,
    run_records: list[Mapping[str, object]],
    all_interactions: list[dict],
    all_missed: list[dict],
    duration_seconds: int,
    warmup_seconds: int,
    simulation_for_metrics: Simulation,
) -> dict:
    evaluation_seconds = max(duration_seconds - warmup_seconds, 1)
    outcome_metrics = _outcome_metrics(all_interactions, len(all_missed))
    outcome_metrics["f2f_per_hour"] = (
        len(all_interactions) / max((evaluation_seconds * max(len(run_records), 1)) / 3600.0, 1e-9)
    )
    workflow_health = _aggregate_workflow_health([
        record["workflow_health"] for record in run_records
    ])
    movement_by_role = Counter()
    route_exposure_by_zone = Counter()
    station_dwell = Counter()
    visibility_opportunities = Counter()
    perception_counter = Counter()
    for record in run_records:
        movement = record.get("movement_metrics", {})
        for role, distance in movement.get("movement_distance_by_role", {}).items():
            movement_by_role[role] += float(distance)
        for transition, count in movement.get("zone_transition_counts", {}).items():
            route_exposure_by_zone[transition] += int(count)
        for zone_id, seconds in movement.get("station_occupancy_seconds", {}).items():
            station_dwell[zone_id] += float(seconds)
        for zone_id, count in record.get("visibility_counts_by_zone", {}).items():
            visibility_opportunities[zone_id] += int(count)
        perception_counter.update(record.get("perception_diagnostics", {}))
    for diagnostic_key in (
        "raw_visible_staff_percepts_count",
        "eligible_visible_staff_opportunity_count",
        "selected_visible_staff_opportunity_count",
        "approach_intent_count",
        "approach_success_count",
        "approach_failure_count",
        "immediate_close_logged_count",
        "raw_visible_staff_cocpit_related_count",
        "eligible_cocpit_related_opportunity_count",
        "approach_intent_cocpit_related_count",
        "completed_cocpit_related_count",
        "raw_visible_staff_nursta_related_count",
        "eligible_nursta_related_opportunity_count",
        "approach_intent_nursta_related_count",
        "completed_nursta_related_count",
        "route_override_count",
        "route_resume_count",
        "nonproximate_logged_interaction_count",
        "validation_counted_interactions_beyond_close_threshold_count",
        "synthetic_or_relocated_interaction_coordinate_count",
        "midpoint_logged_validation_interaction_count",
        "task_transition_update_logged_count",
        "task_transition_update_not_close_count",
        "task_transition_update_episode_suppressed_count",
        "task_transition_update_new_episode_count",
        "immediate_close_interaction_count",
        "pending_intent_completed_interaction_count",
        "missed_not_close_opportunity_count",
        "station_same_dwell_repeat_suppressed_count",
        "station_same_episode_repeat_suppressed_count",
        "station_same_pair_same_episode_opportunity_count",
        "station_episode_started_count",
        "station_episode_ended_count",
        "station_same_pair_episode_count",
        "station_episode_new_encounter_count",
        "visible_staff_percepts_count",
        "visible_staff_with_reason_count",
        "visible_staff_no_reason_count",
        "reason_gate_rejected_count",
        "actionable_reason_opportunity_count",
        "actionable_reason_to_approach_count",
        "reason_to_completed_interaction_count",
        "task_update_episode_started_count",
        "task_update_episode_completed_count",
        "task_update_episode_suppressed_repeat_count",
        "task_update_episode_expired_count",
        "post_task_station_check_count",
        "post_task_station_check_completed_count",
        "station_check_route_failures",
    ):
        perception_counter.setdefault(diagnostic_key, 0)

    role_pair_distribution = outcome_metrics.get("role_pair_distribution", {})
    selected_role_pairs = {
        pair: float(role_pair_distribution.get(pair, 0.0))
        for pair in (
            "Doctor|Doctor",
            "Doctor|Nurse",
            "Doctor|Patient",
            "Nurse|Nurse",
            "Nurse|Patient",
            "CoordinationNurse|Nurse",
            "CoordinationNurse|Patient",
        )
    }
    zone_distribution = outcome_metrics.get("zone_distribution", {})
    selected_zones = {
        zone_id: float(zone_distribution.get(zone_id, 0.0))
        for zone_id in ("COCPIT", "NURSTA", "NUROPE", "CORR01", "CORR02", "CORR03", "CORR04", "CORR05", "CORR06", "CORR07")
    }
    visibility_metrics = simulation_for_metrics.visibility_metrics_snapshot()
    visibility_metrics.update(_visibility_graph_metrics(simulation_for_metrics))
    perception_diagnostics = dict(perception_counter)
    perception_diagnostics["final_logged_distance_max"] = max(
        (
            float(record.get("perception_diagnostics", {}).get("final_logged_distance_max", 0.0))
            for record in run_records
        ),
        default=0.0,
    )
    for max_key in (
        "task_transition_update_max_logged_distance",
        "staff_staff_logged_max_distance",
        "patient_facing_logged_max_distance",
    ):
        perception_diagnostics[max_key] = max(
            (
                float(record.get("perception_diagnostics", {}).get(max_key, 0.0))
                for record in run_records
            ),
            default=0.0,
        )
    perception_diagnostics["approach_failure_reasons"] = {
        str(key).replace("approach_failure_reason_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("approach_failure_reason_")
    }
    perception_diagnostics["post_task_station_check_by_role"] = {
        str(key).replace("post_task_station_check_role_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("post_task_station_check_role_")
    }
    perception_diagnostics["post_task_station_check_by_station"] = {
        str(key).replace("post_task_station_check_station_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("post_task_station_check_station_")
    }
    perception_diagnostics["station_episode_reset_reasons"] = {
        str(key).replace("station_episode_reset_reason_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("station_episode_reset_reason_")
    }
    perception_diagnostics["reason_type_counts"] = {
        str(key).replace("reason_type_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("reason_type_")
    }
    perception_diagnostics["reason_gate_rejection_reasons"] = {
        str(key).replace("reason_gate_rejection_reason_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("reason_gate_rejection_reason_")
    }
    perception_diagnostics["task_update_episode_by_role_pair"] = {
        str(key).replace("task_update_episode_role_pair_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("task_update_episode_role_pair_")
    }
    perception_diagnostics["task_update_episode_by_patient_context"] = {
        str(key).replace("task_update_episode_patient_context_", ""): int(value)
        for key, value in perception_counter.items()
        if str(key).startswith("task_update_episode_patient_context_")
    }
    return {
        "condition_name": condition_name,
        "scenario_definition": config.PART2_SCENARIO_REGISTRY.get(condition_name, {}),
        "n_runs": len(run_records),
        "duration_seconds": duration_seconds,
        "warmup_seconds": warmup_seconds,
        "evaluation_duration_seconds": evaluation_seconds,
        "paired_seeds": [int(record["seed"]) for record in run_records],
        "interaction_count": int(outcome_metrics["interaction_count"]),
        "f2f_per_hour": float(outcome_metrics["f2f_per_hour"]),
        "hcw_hcw_share": float(outcome_metrics["hcw_hcw_share"]),
        "patient_facing_share": float(outcome_metrics["patient_facing_share"]),
        "station_or_desk_share": float(outcome_metrics["station_or_desk_interaction_share"]),
        "corridor_share": float(outcome_metrics["corridor_interaction_share"]),
        "bedside_or_patient_room_share": float(outcome_metrics["bedside_or_patient_room_interaction_share"]),
        "missed_opportunity_episodes": int(outcome_metrics["missed_opportunity_episodes"]),
        "mean_duration": float(outcome_metrics["mean_duration"]),
        "median_duration": float(outcome_metrics["median_duration"]),
        "role_pair_distribution": role_pair_distribution,
        "selected_role_pair_shares": selected_role_pairs,
        "topic_distribution": outcome_metrics.get("topic_distribution", {}),
        "zone_distribution": zone_distribution,
        "selected_zone_shares": selected_zones,
        "zone_supergroup_distribution": outcome_metrics.get("zone_supergroup_distribution", {}),
        "workflow_health": workflow_health,
        "movement_distance_by_role_total": dict(movement_by_role),
        "movement_distance_by_role_mean_per_run": {
            role: float(distance) / max(len(run_records), 1)
            for role, distance in movement_by_role.items()
        },
        "route_exposure_by_zone_transition": dict(route_exposure_by_zone),
        "station_occupancy_seconds_total": dict(station_dwell),
        "visibility_opportunities_by_zone": dict(visibility_opportunities),
        "perception_diagnostics": perception_diagnostics,
        "visibility_metrics": visibility_metrics,
        "accepted_interaction_opportunities": len(all_interactions),
        "missed_interaction_opportunities": len(all_missed),
        "per_run": list(run_records),
    }


def _condition_delta(condition: Mapping[str, object], baseline: Mapping[str, object], metric: str) -> float:
    return float(condition.get(metric, 0.0)) - float(baseline.get(metric, 0.0))


def _build_part2_effects(results: Mapping[str, Mapping[str, object]], metrics: list[str]) -> dict:
    baseline = results["baseline"]
    effects: dict[str, dict[str, float]] = {}
    for condition_name, payload in results.items():
        effects[condition_name] = {
            metric: _condition_delta(payload, baseline, metric)
            for metric in metrics
        }
    main_effects = {}
    if {"baseline", "cockpit_only", "nursta_only", "both"}.issubset(results):
        for metric in metrics:
            baseline_value = float(results["baseline"].get(metric, 0.0))
            cockpit_value = float(results["cockpit_only"].get(metric, 0.0))
            nursta_value = float(results["nursta_only"].get(metric, 0.0))
            both_value = float(results["both"].get(metric, 0.0))
            main_effects[metric] = {
                "cockpit_main_effect": cockpit_value - baseline_value,
                "nursta_main_effect": nursta_value - baseline_value,
                "combined_effect": both_value - baseline_value,
                "interaction_effect": both_value - cockpit_value - nursta_value + baseline_value,
            }
    return {"deltas_from_baseline": effects, "factorial_effects": main_effects}


def _write_part2_scenario_matrix(conditions: list[str]) -> None:
    config.FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Part 2 Scenario Matrix",
        "",
        "Part 2 tests spatial/design changes as predicted effects relative to the validated Part 1 baseline. These are not post-intervention empirical validation results.",
        "",
        "| Condition | What changes | Mechanism tested | Interpretable effect |",
        "|---|---|---|---|",
    ]
    for condition_name in conditions:
        spec = config.PART2_SCENARIO_REGISTRY[condition_name]
        lines.append(
            f"| `{condition_name}` | {spec['description']} | {spec['mechanism_changed']} | {spec['expected_direction_of_effect']} |"
        )
    lines.extend([
        "",
        "The four conditions form a 2x2 design: COCPIT transparency off/on crossed with NURSTA station intervention off/on.",
        "The interaction effect is computed as `both - cockpit_only - nursta_only + baseline`.",
    ])
    config.PART2_SCENARIO_MATRIX_PATH.write_text("\n".join(lines) + "\n")


def _write_part2_mechanism_explanation() -> None:
    lines = [
        "# Part 2 Mechanism Explanation",
        "",
        "Part 2 changes the environment, simple spatial perception, and movement-opportunity layer. It does not change the clinical workflow, arrivals, beds, staffing, ESI logic, memory, LLM reasoning, or generative destination choice.",
        "",
        "- COCPIT transparency changes line-of-sight, isovists, visibility, and awareness around COCPIT while preserving physical barriers.",
        "- The NURSTA intervention changes the station polygon, station attractor, and local access waypoints, which can alter station return, dwell, route exposure, and co-presence.",
        "- Outcomes are interpreted as model-predicted deltas from the baseline, not empirical proof that one intervention is better.",
        "- The spatial mechanism is a simplified proxy: wall-valid routing, zone exposure, station-attractor behavior, line-of-sight, field-of-view, isovist area, and visibility graph centrality. It is not a full new space-syntax calibration framework.",
        "- Part 3 will later test whether richer memory/generative cognitive agents interpret and narrate these spatial differences differently.",
    ]
    config.PART2_MECHANISM_EXPLANATION_PATH.write_text("\n".join(lines) + "\n")


def _write_part2_cluster_plan(conditions: list[str]) -> None:
    condition_string = ",".join(conditions)
    lines = [
        "# Part 2 Cluster Plan",
        "",
        "Future cluster runs should use a paired-seed design: every seed is run once per condition, then condition deltas are computed within seed.",
        "",
        "Suggested later command pattern:",
        "",
        "```bash",
        f"python3 main.py --part2-scenarios --duration 43200 --warmup-seconds 7200 --n-runs 1 --conditions {condition_string} --seed <SEED> --output-run-id <RUN_ID>",
        "```",
        "",
        "For n=1000 on Snellius or another cluster, use a SLURM array with one seed-condition batch per task or one seed per task running all four conditions. Do not run long ensembles on a login node.",
        "",
        "Outputs from task runs should be mergeable into `outputs/latest/part2_scenario_summary.json` by condition and seed. Keep the seed list paired across baseline, cockpit_only, nursta_only, and both.",
    ]
    config.PART2_CLUSTER_PLAN_PATH.write_text("\n".join(lines) + "\n")


def _write_part2_condition_mechanism_audit(payload: Mapping[str, object]) -> None:
    conditions = payload.get("conditions", {})
    baseline = conditions.get("baseline", {})
    audit = {
        "purpose": "Part 2 condition mechanism audit; development output, not empirical proof.",
        "paired_seeds_used": payload.get("paired_seeds_used"),
        "conditions": {},
        "interpretation": [],
    }
    for name, row in conditions.items():
        definition = row.get("scenario_definition", {})
        visibility = row.get("visibility_metrics", {})
        movement = row.get("movement_distance_by_role_mean_per_run", {})
        station_dwell = row.get("station_occupancy_seconds_total", {})
        opportunities = row.get("visibility_opportunities_by_zone", {})
        base_visibility = baseline.get("visibility_metrics", {}) if isinstance(baseline, Mapping) else {}
        base_movement = baseline.get("movement_distance_by_role_mean_per_run", {}) if isinstance(baseline, Mapping) else {}
        audit["conditions"][name] = {
            "definition": definition,
            "workflow_health": row.get("workflow_health", {}).get("workflow_health_status", "UNKNOWN"),
            "bed8_violations": row.get("workflow_health", {}).get("reserved_bed_restriction_violations", 0),
            "visibility_changes_declared": bool(definition.get("visibility_changes", False)),
            "station_attractor_changes_declared": bool(definition.get("station_attractor_changes", False)),
            "wayfinding_or_movement_changes_declared": bool(definition.get("wayfinding_or_movement_changes", False)),
            "mean_isovist_area": visibility.get("mean_isovist_area"),
            "cockpit_isovist_area": visibility.get("COCPIT_isovist_area"),
            "nursta_isovist_area": visibility.get("NURSTA_isovist_area"),
            "mean_isovist_delta_vs_baseline": (
                float(visibility.get("mean_isovist_area", 0.0))
                - float(base_visibility.get("mean_isovist_area", 0.0))
                if name != "baseline" else 0.0
            ),
            "movement_distance_by_role_mean_per_run": movement,
            "movement_distance_delta_vs_baseline": {
                role: float(distance) - float(base_movement.get(role, 0.0))
                for role, distance in movement.items()
            } if name != "baseline" else {},
            "station_occupancy_seconds_total": station_dwell,
            "visibility_opportunities_by_zone": opportunities,
            "f2f_per_hour": row.get("f2f_per_hour"),
            "missed_opportunity_episodes": row.get("missed_opportunity_episodes"),
            "visibility_mediated_interactions": row.get("perception_diagnostics", {}).get("visibility_mediated_interaction_count"),
            "nonproximate_visible_interactions": row.get("perception_diagnostics", {}).get("nonproximate_visible_interaction_count"),
        }
    if "cockpit_only" in conditions:
        cockpit = audit["conditions"]["cockpit_only"]
        changed = abs(float(cockpit.get("mean_isovist_delta_vs_baseline", 0.0))) > 1e-6
        audit["interpretation"].append(
            "COCPIT transparency changed visibility metrics." if changed
            else "COCPIT transparency is declared but did not change aggregate scenario metrics in this preliminary run; inspect the visibility-to-action chain before claiming an effect."
        )
    if "nursta_only" in conditions:
        nursta_delta = payload.get("effects", {}).get("deltas_from_baseline", {}).get("nursta_only", {})
        f2f_delta = float(nursta_delta.get("f2f_per_hour", 0.0))
        audit["interpretation"].append(
            f"NURSTA condition changed F2F/hour by {f2f_delta:+.2f}; large effects should remain latest-only until station dwell, route exposure, and cooldown behavior are explained."
        )
    config.PART2_CONDITION_MECHANISM_AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2))

    lines = [
        "# Part 2 Condition Mechanism Audit",
        "",
        "Development output. These are predicted mechanism checks, not empirical post-intervention validation.",
        "",
    ]
    for name, row in audit["conditions"].items():
        lines.extend([
            f"## {name}",
            f"- Workflow health: {row['workflow_health']}",
            f"- Bed 8 violations: {row['bed8_violations']}",
            f"- Visibility change declared: {row['visibility_changes_declared']}",
            f"- Station/movement change declared: {row['station_attractor_changes_declared'] or row['wayfinding_or_movement_changes_declared']}",
            f"- Mean isovist delta vs baseline: {float(row['mean_isovist_delta_vs_baseline'] or 0.0):+.3f}",
            f"- F2F/hour: {float(row['f2f_per_hour'] or 0.0):.2f}",
            f"- Missed opportunities: {row['missed_opportunity_episodes']}",
            "",
        ])
    lines.extend(["## Interpretation", *[f"- {item}" for item in audit["interpretation"]], ""])
    config.PART2_CONDITION_MECHANISM_AUDIT_MD_PATH.write_text("\n".join(lines))


def _write_part2_results_markdown(payload: Mapping[str, object], metrics: list[str]) -> None:
    results = payload["conditions"]
    effects = payload["effects"]
    lines = [
        "# Part 2 Preliminary Results",
        "",
        "These are preliminary predicted effects relative to the validated Part 1 baseline. They are not empirical proof of intervention benefit.",
        "",
        f"- Protocol: duration {payload['duration_seconds']}s, warm-up {payload['warmup_seconds']}s, n-runs {payload['n_runs']}.",
        f"- Paired seeds used: {'yes' if payload['paired_seeds_used'] else 'no'}.",
        f"- SeniorDoctor oversight: {'ON' if payload.get('senior_doctor_oversight_enabled') else 'OFF'}.",
        "",
        "| Condition | F2F/hr | HCW-HCW | Patient-facing | Station/desk | Corridor | Bedside | Missed opps | Workflow | Bed 8 violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for condition_name in payload["condition_order"]:
        row = results[condition_name]
        workflow = row["workflow_health"]
        lines.append(
            f"| `{condition_name}` | {row['f2f_per_hour']:.2f} | {row['hcw_hcw_share']:.3f} | "
            f"{row['patient_facing_share']:.3f} | {row['station_or_desk_share']:.3f} | "
            f"{row['corridor_share']:.3f} | {row['bedside_or_patient_room_share']:.3f} | "
            f"{row['missed_opportunity_episodes']} | {workflow.get('workflow_health_status', 'UNKNOWN')} | "
            f"{workflow.get('reserved_bed_restriction_violations', 0)} |"
        )
    lines.extend([
        "",
        "## Deltas From Baseline",
        "",
        "| Condition | F2F/hr | HCW-HCW | Patient-facing | Station/desk | Corridor | Bedside | Missed opps | Doctor-Nurse | Nurse-Nurse | CoordinationNurse-Nurse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    delta_rows = effects["deltas_from_baseline"]
    for condition_name in payload["condition_order"]:
        deltas = delta_rows[condition_name]
        role_deltas = results[condition_name]["selected_role_pair_deltas_from_baseline"]
        lines.append(
            f"| `{condition_name}` | {deltas['f2f_per_hour']:+.2f} | {deltas['hcw_hcw_share']:+.3f} | "
            f"{deltas['patient_facing_share']:+.3f} | {deltas['station_or_desk_share']:+.3f} | "
            f"{deltas['corridor_share']:+.3f} | {deltas['bedside_or_patient_room_share']:+.3f} | "
            f"{deltas['missed_opportunity_episodes']:+.0f} | {role_deltas['Doctor|Nurse']:+.3f} | "
            f"{role_deltas['Nurse|Nurse']:+.3f} | {role_deltas['CoordinationNurse|Nurse']:+.3f} |"
        )
    lines.extend([
        "",
        "## Factorial Effects",
        "",
        "| Metric | COCPIT main effect | NURSTA main effect | Combined effect | Interaction effect |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric in metrics:
        effect = effects["factorial_effects"].get(metric, {})
        lines.append(
            f"| `{metric}` | {effect.get('cockpit_main_effect', 0.0):+.3f} | "
            f"{effect.get('nursta_main_effect', 0.0):+.3f} | "
            f"{effect.get('combined_effect', 0.0):+.3f} | "
            f"{effect.get('interaction_effect', 0.0):+.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "At n=3, effects should be read as a smoke-tested direction-of-effect check. Larger paired-seed ensembles are needed before making stable intervention claims.",
    ])
    config.PART2_PRELIMINARY_NOTES_LATEST_PATH.write_text("\n".join(lines) + "\n")


def _plot_part2_effects(payload: Mapping[str, object], metrics: list[str]) -> None:
    results = payload["conditions"]
    condition_order = [name for name in payload["condition_order"] if name != "baseline"]
    display_metrics = [
        ("f2f_per_hour", "F2F/hr"),
        ("hcw_hcw_share", "HCW-HCW"),
        ("patient_facing_share", "Patient-facing"),
        ("station_or_desk_share", "Station/desk"),
        ("corridor_share", "Corridor"),
        ("bedside_or_patient_room_share", "Bedside"),
        ("missed_opportunity_episodes", "Missed"),
        ("Doctor|Nurse", "Doctor|Nurse"),
        ("Nurse|Nurse", "Nurse|Nurse"),
        ("CoordinationNurse|Nurse", "CoordNurse|Nurse"),
    ]
    matrix = []
    for condition_name in condition_order:
        row = []
        deltas = payload["effects"]["deltas_from_baseline"][condition_name]
        role_deltas = results[condition_name]["selected_role_pair_deltas_from_baseline"]
        for metric, _label in display_metrics:
            if metric in role_deltas:
                row.append(float(role_deltas[metric]))
            else:
                row.append(float(deltas.get(metric, 0.0)))
        matrix.append(row)
    figure, axis = plt.subplots(figsize=(12, 4.8))
    arr = np.asarray(matrix, dtype=float)
    max_abs = max(float(np.max(np.abs(arr))) if arr.size else 0.0, 1e-6)
    image = axis.imshow(arr, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    axis.set_xticks(range(len(display_metrics)))
    axis.set_xticklabels([label for _metric, label in display_metrics], rotation=35, ha="right")
    axis.set_yticks(range(len(condition_order)))
    axis.set_yticklabels(condition_order)
    axis.set_title("Part 2 preliminary predicted deltas from baseline (n=3)")
    for row_index, condition_name in enumerate(condition_order):
        for col_index, value in enumerate(arr[row_index]):
            axis.text(col_index, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Delta from baseline")
    figure.tight_layout()
    figure.savefig(config.PART2_PRELIMINARY_EFFECTS_LATEST_PATH, dpi=220)
    config.PART2_PRELIMINARY_EFFECTS_HTML_LATEST_PATH.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Part 2 Preliminary Effects</title></head>"
        "<body><h1>Part 2 preliminary effects</h1><p>Latest-only development output.</p>"
        "<img src='part2_preliminary_effects.png' style='max-width:1100px;width:100%;'></body></html>"
    )
    plt.close(figure)


def run_part2_scenarios(
    *,
    duration_seconds: int = 43_200,
    warmup_seconds: int = 7_200,
    n_runs: int = 3,
    conditions: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
    output_run_id: Optional[str] = None,
    save_outputs: bool = True,
) -> dict:
    """Run paired-seed Part 2 design scenarios without empirical validation claims."""

    condition_order = list(conditions or config.EXPERIMENT_CONDITIONS)
    unknown = [name for name in condition_order if name not in config.PART2_SCENARIO_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown Part 2 condition(s): {', '.join(unknown)}")
    seeds = [int(seed)] if seed is not None else [config.RUN_BASE_SEED + index for index in range(n_runs)]
    results: dict[str, dict] = {}

    for condition_name in condition_order:
        run_records: list[dict] = []
        all_interactions: list[dict] = []
        all_missed: list[dict] = []
        simulation_for_metrics: Optional[Simulation] = None
        for run_seed in seeds:
            simulation = Simulation(
                random_seed=run_seed,
                condition_name=condition_name,
                interaction_backend=config.INTERACTION_BACKEND,
                persona_source=config.PERSONA_SOURCE,
                model_variant=config.MODEL_VARIANT,
                enable_senior_doctor_oversight=False,
            )
            simulation.run(duration_seconds)
            window_interactions = _events_after_warmup(simulation.interaction_log, warmup_seconds, duration_seconds)
            window_missed = _events_after_warmup(simulation.missed_opportunity_log, warmup_seconds, duration_seconds)
            all_interactions.extend(window_interactions)
            all_missed.extend(window_missed)
            run_perception_diagnostics = dict(simulation.ablation_diagnostics)
            final_logged_distances = [
                float(event.get("final_logged_distance_m", 0.0))
                for event in window_interactions
                if event.get("final_logged_distance_m") is not None
            ]
            final_staff_distances = [
                float(event.get("final_logged_distance_m", 0.0))
                for event in window_interactions
                if event.get("interaction_source") == "opportunistic_perception_interruption"
                and event.get("final_logged_distance_m") is not None
            ]
            staff_staff_distances = [
                float(event.get("final_logged_distance_m", 0.0))
                for event in window_interactions
                if event.get("final_logged_distance_m") is not None
                and event.get("role_1") != "Patient"
                and event.get("role_2") != "Patient"
            ]
            patient_facing_distances = [
                float(event.get("final_logged_distance_m", 0.0))
                for event in window_interactions
                if event.get("final_logged_distance_m") is not None
                and (event.get("role_1") == "Patient" or event.get("role_2") == "Patient")
            ]
            task_transition_distances = [
                float(event.get("final_logged_distance_m", 0.0))
                for event in window_interactions
                if event.get("interaction_type") == "task_transition_update"
                and event.get("final_logged_distance_m") is not None
            ]
            run_perception_diagnostics["final_logged_distance_max"] = (
                float(max(final_logged_distances)) if final_logged_distances else 0.0
            )
            run_perception_diagnostics["validation_counted_interactions_beyond_close_threshold_count"] = sum(
                1
                for distance in final_logged_distances
                if distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
            )
            run_perception_diagnostics["nonproximate_logged_interaction_count"] = sum(
                1
                for distance in final_staff_distances
                if distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
            )
            run_perception_diagnostics["synthetic_or_relocated_interaction_coordinate_count"] = sum(
                1 for event in window_interactions if bool(event.get("synthetic_or_relocated_coordinate", False))
            )
            run_perception_diagnostics["midpoint_logged_validation_interaction_count"] = sum(
                1 for event in window_interactions if bool(event.get("midpoint_logged_validation_interaction", False))
            )
            run_perception_diagnostics["task_transition_update_logged_count"] = len(task_transition_distances)
            run_perception_diagnostics["task_transition_update_max_logged_distance"] = (
                float(max(task_transition_distances)) if task_transition_distances else 0.0
            )
            run_perception_diagnostics["staff_staff_logged_max_distance"] = (
                float(max(staff_staff_distances)) if staff_staff_distances else 0.0
            )
            run_perception_diagnostics["patient_facing_logged_max_distance"] = (
                float(max(patient_facing_distances)) if patient_facing_distances else 0.0
            )
            run_perception_diagnostics["route_failures"] = int(
                run_perception_diagnostics.get("route_failure_count", 0)
            )
            run_records.append({
                "seed": run_seed,
                "interaction_count": len(window_interactions),
                "missed_opportunity_episodes": len(window_missed),
                "workflow_health": simulation.workflow_health_summary(warmup_seconds=warmup_seconds),
                "movement_metrics": simulation.movement_metrics(),
                "visibility_counts_by_zone": dict(simulation.mutual_visibility_counts_by_zone),
                "perception_diagnostics": run_perception_diagnostics,
            })
            simulation_for_metrics = simulation
        if simulation_for_metrics is None:
            continue
        results[condition_name] = _aggregate_part2_runs(
            condition_name=condition_name,
            run_records=run_records,
            all_interactions=all_interactions,
            all_missed=all_missed,
            duration_seconds=duration_seconds,
            warmup_seconds=warmup_seconds,
            simulation_for_metrics=simulation_for_metrics,
        )

    baseline_roles = results.get("baseline", {}).get("selected_role_pair_shares", {})
    for condition_name, payload in results.items():
        payload["selected_role_pair_deltas_from_baseline"] = {
            pair: float(payload["selected_role_pair_shares"].get(pair, 0.0)) - float(baseline_roles.get(pair, 0.0))
            for pair in payload["selected_role_pair_shares"]
        }

    effect_metrics = [
        "f2f_per_hour",
        "hcw_hcw_share",
        "patient_facing_share",
        "station_or_desk_share",
        "corridor_share",
        "bedside_or_patient_room_share",
        "missed_opportunity_episodes",
    ]
    payload = {
        "run_kind": "part2_scenario_comparison",
        "interpretation": "Preliminary predicted effects relative to the validated baseline; not empirical post-intervention validation.",
        "duration_seconds": duration_seconds,
        "warmup_seconds": warmup_seconds,
        "evaluation_duration_seconds": max(duration_seconds - warmup_seconds, 0),
        "n_runs": len(seeds),
        "paired_seeds_used": True,
        "seeds": seeds,
        "output_run_id": output_run_id,
        "condition_order": condition_order,
        "senior_doctor_oversight_enabled": False,
        "scenario_registry": {name: config.PART2_SCENARIO_REGISTRY[name] for name in condition_order},
        "spatial_method_note": "Uses existing wall-valid routing, station attractors, FOV/line-of-sight, isovists, visibility graph centrality, route exposure, and co-presence proxies; not a new full space-syntax calibration.",
        "conditions": results,
        "effects": _build_part2_effects(results, effect_metrics),
    }
    if save_outputs:
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
        config.PART2_SCENARIO_SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
        _write_part2_condition_mechanism_audit(payload)
        _write_part2_results_markdown(payload, effect_metrics)
        _plot_part2_effects(payload, effect_metrics)
    return payload


def run_cockpit_transparency_mechanism_audit(
    *,
    duration_seconds: int = 14_400,
    n_runs: int = 1,
    seeds: Optional[Iterable[int]] = None,
    conditions: Iterable[str] = ("baseline", "cockpit_only"),
    save_outputs: bool = True,
) -> dict:
    """Run a short paired-seed audit of the COCPIT visibility-to-action chain.

    This is intentionally not a Part 2 scenario comparison. It isolates whether
    COCPIT transparency changes perception-mediated opportunities while leaving
    movement, workflow, arrivals, beds, staffing, and empirical targets alone.
    Counts ending in ``observations`` are repeated timestep observations, not
    unique trips or unique people.
    """

    condition_order = list(conditions)
    seed_list = [int(seed) for seed in seeds] if seeds is not None else [
        config.RUN_BASE_SEED + index for index in range(n_runs)
    ]
    available_modes = {"idle", "waiting_for_doctor", "using_secondary_station", "returning_home", "patrolling"}
    stationary_modes = {"idle", "waiting_for_doctor", "using_secondary_station"}
    moving_modes = {"returning_home", "moving_to_secondary_station", "patrolling", "moving_to_doctor", "moving_to_patient"}
    cockpit_adjacent_zones = {"COCPIT", "CORR02", "CORR03", "CORR07"}
    near_radius = float(config.PERCEPTION_INTERACTION_RADIUS_METERS)

    baseline_environment = Environment(config.WALL_POSITIONS_PATH, config.ZONE_BOUNDARIES_PATH)
    baseline_manager = ConditionManager(baseline_environment, get_condition_spec("baseline"), [])

    def _zone(simulation: Simulation, position: tuple[float, float]) -> str:
        return simulation.which_zone(*position) or "Outside named zones"

    def _cockpit_centroid(simulation: Simulation) -> tuple[float, float]:
        return simulation.condition_manager.zone_centroid("COCPIT")

    def _near_cockpit(simulation: Simulation, position: tuple[float, float]) -> bool:
        return (
            math.dist(position, _cockpit_centroid(simulation)) <= near_radius
            or _zone(simulation, position) in cockpit_adjacent_zones
        )

    def _initiator_in_cockpit_context(simulation: Simulation, agent) -> bool:
        position = tuple(agent.position)
        return _zone(simulation, position) in cockpit_adjacent_zones

    def _baseline_blocks(start: tuple[float, float], end: tuple[float, float]) -> bool:
        return not baseline_manager.has_line_of_sight(start, end)

    def _condition_geometry_checks(condition_name: str) -> dict:
        spec = get_condition_spec(condition_name)
        manager = ConditionManager(baseline_environment, spec, [])
        return {
            "removed_wall_segments_count": len(spec.removed_wall_segments),
            "routing_waypoint_overrides_count": len(spec.routing_waypoint_overrides),
            "station_attractor_overrides_count": len(spec.station_attractor_overrides),
            "effective_wall_count_matches_baseline": len(manager.effective_walls) == len(baseline_manager.effective_walls),
            "visibility_wall_count": len(manager.visibility_walls),
            "baseline_visibility_wall_count": len(baseline_manager.visibility_walls),
            "transparent_wall_indices_count": len(manager.transparent_wall_indices),
            "collision_walls_unchanged": tuple(manager.effective_walls) == tuple(baseline_manager.effective_walls),
            "routing_waypoints_unchanged": manager.routing_waypoints == baseline_manager.routing_waypoints,
            "station_attractors_unchanged": manager.station_attractors == baseline_manager.station_attractors,
        }

    results: dict[str, dict] = {}
    for condition_name in condition_order:
        aggregate = Counter()
        per_seed: list[dict] = []
        examples: list[dict] = []
        for seed in seed_list:
            simulation = Simulation(
                random_seed=seed,
                condition_name=condition_name,
                interaction_backend=config.INTERACTION_BACKEND,
                persona_source=config.PERSONA_SOURCE,
                model_variant=config.MODEL_VARIANT,
                enable_senior_doctor_oversight=False,
            )
            counters: Counter = Counter()
            previous_interaction_count = 0
            previous_missed_count = 0
            while simulation.timestep < duration_seconds:
                staff_lookup = simulation.staff_by_id()
                state_by_id = {
                    agent.gid: {
                        "gid": agent.gid,
                        "name": getattr(agent, "name", str(agent.gid)),
                        "role": agent.role,
                        "mode": getattr(agent, "mode", None),
                        "position": tuple(agent.position),
                        "zone": _zone(simulation, tuple(agent.position)),
                    }
                    for agent in simulation.staff_agents
                }

                moving_near_ids = {
                    state["gid"]
                    for state in state_by_id.values()
                    if state["mode"] in moving_modes and _near_cockpit(simulation, state["position"])
                }
                counters["raw_moving_staff_near_cocpit_observations"] += len(moving_near_ids)

                perceived_pairs: set[tuple[int, int]] = set()
                attended_pairs: set[tuple[int, int]] = set()
                for initiator in simulation.staff_agents:
                    initiator_state = state_by_id[initiator.gid]
                    if initiator_state["mode"] not in available_modes:
                        continue
                    if not _initiator_in_cockpit_context(simulation, initiator):
                        continue
                    bundle = simulation.perception.build_bundle(initiator, simulation, [])
                    visible_moving_percepts = []
                    for percept in bundle.visible_agents:
                        if not isinstance(percept.entity_id, int):
                            continue
                        target = staff_lookup.get(int(percept.entity_id))
                        if target is None or target.gid == initiator.gid:
                            continue
                        target_state = state_by_id[target.gid]
                        if target_state["mode"] not in moving_modes:
                            continue
                        if not _near_cockpit(simulation, target_state["position"]):
                            continue
                        perceived_pairs.add((initiator.gid, target.gid))
                        visible_moving_percepts.append((percept, target))

                    visible_moving_percepts.sort(
                        key=lambda item: simulation._visibility_candidate_salience(initiator, item[1], item[0]),
                        reverse=True,
                    )
                    for percept, target in visible_moving_percepts[: max(int(config.PERCEPTION_STAFF_ATTENTION_CAPACITY), 0)]:
                        attended_pairs.add((initiator.gid, target.gid))

                counters["perceived_moving_staff_from_cocpit_observations"] += len(perceived_pairs)
                counters["attended_moving_staff_from_cocpit_observations"] += len(attended_pairs)

                simulation.step()
                new_interactions = simulation.interaction_log[previous_interaction_count:]
                new_missed = simulation.missed_opportunity_log[previous_missed_count:]
                previous_interaction_count = len(simulation.interaction_log)
                previous_missed_count = len(simulation.missed_opportunity_log)

                for event in new_interactions:
                    initiator_id = int(event.get("agent_1_id", -1))
                    target_id = int(event.get("agent_2_id", -1))
                    initiator_state = state_by_id.get(initiator_id)
                    target_state = state_by_id.get(target_id)
                    if initiator_state is None or target_state is None:
                        continue
                    if initiator_state["mode"] not in stationary_modes:
                        continue
                    if initiator_state["zone"] not in cockpit_adjacent_zones:
                        continue
                    if target_state["mode"] not in moving_modes or not _near_cockpit(simulation, target_state["position"]):
                        continue
                    distance = float(event.get("initiation_distance_m", math.dist(initiator_state["position"], target_state["position"])))
                    blocked_in_baseline = _baseline_blocks(initiator_state["position"], target_state["position"])
                    counters["cocpit_idle_to_passing_opportunities"] += 1
                    counters["cocpit_idle_to_passing_logged_interactions"] += 1
                    if bool(event.get("approach_intent_created", False)):
                        counters["cocpit_idle_to_passing_approach_pause_events"] += 1
                    if distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
                        counters["nonproximate_perception_initiated_cocpit_staff_interactions"] += 1
                    if blocked_in_baseline:
                        counters["transparent_only_blocked_in_baseline_examples"] += 1
                    if len(examples) < 8 and condition_name == "cockpit_only":
                        examples.append({
                            "condition": condition_name,
                            "seed": seed,
                            "timestep": int(event.get("timestamp", event.get("timestep", -1))),
                            "outcome": "logged_interaction",
                            "initiator": initiator_state["name"],
                            "initiator_role": initiator_state["role"],
                            "initiator_mode": initiator_state["mode"],
                            "initiator_zone": initiator_state["zone"],
                            "target": target_state["name"],
                            "target_role": target_state["role"],
                            "target_mode": target_state["mode"],
                            "target_zone": target_state["zone"],
                            "distance_m": round(distance, 3),
                            "line_of_sight_in_condition": True,
                            "mutual_visibility": bool(event.get("perception_context", {}).get("mutual_visibility", False)),
                            "would_be_blocked_in_baseline": bool(blocked_in_baseline),
                            "approach_or_pause_occurred": bool(event.get("approach_intent_created", False)),
                            "interaction_logged": True,
                            "interaction_type": event.get("interaction_type"),
                            "topic": event.get("topic"),
                            "zone": event.get("zone_id"),
                            "nonproximate_perception_initiated": distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS,
                        })

                for event in new_missed:
                    initiator_id = int(event.get("agent_id", -1))
                    target_id = int(event.get("potential_partner_id", -1))
                    initiator_state = state_by_id.get(initiator_id)
                    target_state = state_by_id.get(target_id)
                    if initiator_state is None or target_state is None:
                        continue
                    if initiator_state["mode"] not in stationary_modes:
                        continue
                    if initiator_state["zone"] not in cockpit_adjacent_zones:
                        continue
                    if target_state["mode"] not in moving_modes or not _near_cockpit(simulation, target_state["position"]):
                        continue
                    perception_state = event.get("perception_state", {}) or {}
                    distance = float(perception_state.get("distance_m", math.dist(initiator_state["position"], target_state["position"])))
                    blocked_in_baseline = _baseline_blocks(initiator_state["position"], target_state["position"])
                    counters["cocpit_idle_to_passing_opportunities"] += 1
                    if blocked_in_baseline:
                        counters["transparent_only_blocked_in_baseline_examples"] += 1
                    if len(examples) < 8 and condition_name == "cockpit_only":
                        examples.append({
                            "condition": condition_name,
                            "seed": seed,
                            "timestep": int(event.get("timestep", -1)),
                            "outcome": "missed_opportunity",
                            "initiator": initiator_state["name"],
                            "initiator_role": initiator_state["role"],
                            "initiator_mode": initiator_state["mode"],
                            "initiator_zone": initiator_state["zone"],
                            "target": target_state["name"],
                            "target_role": target_state["role"],
                            "target_mode": target_state["mode"],
                            "target_zone": target_state["zone"],
                            "distance_m": round(distance, 3),
                            "line_of_sight_in_condition": True,
                            "mutual_visibility": bool(perception_state.get("mutual_visibility", False)),
                            "would_be_blocked_in_baseline": bool(blocked_in_baseline),
                            "approach_or_pause_occurred": False,
                            "interaction_logged": False,
                            "interaction_type": perception_state.get("interaction_type"),
                            "topic": None,
                            "zone": event.get("zone_id"),
                            "nonproximate_perception_initiated": distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS,
                        })

            aggregate.update(counters)
            per_seed.append({"seed": seed, "counts": dict(counters)})

        results[condition_name] = {
            "counts": dict(aggregate),
            "per_seed": per_seed,
            "examples": examples,
            "condition_geometry_checks": _condition_geometry_checks(condition_name),
        }

    baseline_counts = results.get("baseline", {}).get("counts", {})
    cockpit_counts = results.get("cockpit_only", {}).get("counts", {})
    metric_rows = [
        ("raw_moving_staff_near_cocpit_observations", "raw moving staff near COCPIT observations"),
        ("perceived_moving_staff_from_cocpit_observations", "perceived moving staff from COCPIT"),
        ("attended_moving_staff_from_cocpit_observations", "attended moving staff from COCPIT"),
        ("cocpit_idle_to_passing_opportunities", "COCPIT idle-to-passing opportunities"),
        ("cocpit_idle_to_passing_approach_pause_events", "COCPIT idle-to-passing approach/pause events"),
        ("cocpit_idle_to_passing_logged_interactions", "COCPIT idle-to-passing logged interactions"),
        ("nonproximate_perception_initiated_cocpit_staff_interactions", "nonproximate perception-initiated COCPIT staff interactions"),
        ("transparent_only_blocked_in_baseline_examples", "transparent-only blocked-in-baseline examples"),
    ]

    comparison_rows = []
    for key, label in metric_rows:
        baseline_value = int(baseline_counts.get(key, 0))
        cockpit_value = int(cockpit_counts.get(key, 0))
        pct_change = None if baseline_value == 0 else ((cockpit_value - baseline_value) / baseline_value) * 100.0
        comparison_rows.append({
            "metric": key,
            "label": label,
            "baseline": baseline_value,
            "cockpit_only": cockpit_value,
            "percent_change_from_baseline": pct_change,
        })

    raw_baseline = int(baseline_counts.get("raw_moving_staff_near_cocpit_observations", 0))
    raw_cockpit = int(cockpit_counts.get("raw_moving_staff_near_cocpit_observations", 0))
    raw_delta_pct = None if raw_baseline == 0 else ((raw_cockpit - raw_baseline) / raw_baseline) * 100.0
    raw_stable = raw_delta_pct is not None and abs(raw_delta_pct) <= 15.0

    payload = {
        "purpose": "Short paired-seed COCPIT transparency mechanism audit; not a full Part 2 scenario batch.",
        "audit_code_location": "src/experiments.py::run_cockpit_transparency_mechanism_audit",
        "duration_seconds": duration_seconds,
        "warmup_seconds": 0,
        "n_runs": len(seed_list),
        "paired_seeds_used": True,
        "seeds": seed_list,
        "metric_definitions": {
            "raw_moving_staff_near_cocpit_observations": "Per-timestep staff-agent observations. Counts each moving/transit staff agent near COCPIT once per timestep. No initiator, visibility, perception, or attention filter.",
            "perceived_moving_staff_from_cocpit_observations": "Per-timestep initiator-target pair observations. Counts idle/available/stationary staff in/near COCPIT who actually perceive a moving/passing staff target under the condition geometry.",
            "attended_moving_staff_from_cocpit_observations": "Per-timestep initiator-target pair observations after the COCPIT initiator attention capacity selects visible moving/passing staff targets.",
            "cocpit_idle_to_passing_opportunities": "Event-level opportunities from logged interactions plus missed-opportunity episodes matching idle/available COCPIT initiator and moving/passing staff target.",
            "cocpit_idle_to_passing_approach_pause_events": "Event-level logged interactions from the above class where an approach intent was created.",
            "cocpit_idle_to_passing_logged_interactions": "Event-level validation-counted F2F interactions matching idle/available COCPIT initiator and moving/passing staff target.",
            "nonproximate_perception_initiated_cocpit_staff_interactions": "Event-level staff-staff interactions whose opportunity was initiated beyond PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS; this means seeing/calling/approaching/pausing before communication, not talking across space.",
            "transparent_only_blocked_in_baseline_examples": "Event-level opportunities/interactions whose initiator-target line would not have line of sight under baseline visibility geometry.",
        },
        "conditions": results,
        "comparison_rows": comparison_rows,
        "raw_movement_delta_percent": raw_delta_pct,
        "raw_movement_stable_within_15_percent": raw_stable,
        "movement_difference_interpretation": (
            "Raw movement exposure is stable under paired seed; differences are small enough to interpret perception changes directly."
            if raw_stable else
            "Raw moving staff observations differ by more than 15%. Because this is a dynamic simulation, perception-triggered approach/pause and different accepted interactions can change subsequent stochastic trajectories even without route or workflow changes. Inspect per-seed traces before expensive Part 2 runs."
        ),
        "cockpit_transparency_implementation_check": {
            "cockpit_only_has_no_removed_wall_segments": results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("removed_wall_segments_count") == 0,
            "cockpit_only_has_no_routing_waypoint_overrides": results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("routing_waypoint_overrides_count") == 0,
            "cockpit_only_has_no_station_attractor_overrides": results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("station_attractor_overrides_count") == 0,
            "collision_walls_unchanged": results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("collision_walls_unchanged") is True,
            "routing_waypoints_unchanged": results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("routing_waypoints_unchanged") is True,
            "station_attractors_unchanged": results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("station_attractors_unchanged") is True,
            "visibility_walls_reduced": (
                int(results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("visibility_wall_count", 0))
                < int(results.get("cockpit_only", {}).get("condition_geometry_checks", {}).get("baseline_visibility_wall_count", 0))
            ),
        },
        "mechanism_observed": bool(
            int(cockpit_counts.get("perceived_moving_staff_from_cocpit_observations", 0))
            > int(baseline_counts.get("perceived_moving_staff_from_cocpit_observations", 0))
            and int(cockpit_counts.get("transparent_only_blocked_in_baseline_examples", 0)) > 0
            and int(cockpit_counts.get("cocpit_idle_to_passing_approach_pause_events", 0)) > 0
        ),
        "transparent_condition_examples": results.get("cockpit_only", {}).get("examples", []),
    }

    if save_outputs:
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = config.LATEST_OUTPUT_DIR / "cockpit_transparency_mechanism_audit.json"
        md_path = config.LATEST_OUTPUT_DIR / "cockpit_transparency_mechanism_audit.md"
        json_path.write_text(json.dumps(payload, indent=2))

        lines = [
            "# COCPIT Transparency Mechanism Audit",
            "",
            "Short paired-seed audit only. This is not a full Part 2 scenario comparison.",
            "",
            f"- Audit code: `{payload['audit_code_location']}`",
            f"- Duration: {duration_seconds}s",
            f"- Seeds: {', '.join(str(seed) for seed in seed_list)}",
            "",
            "## Metric Definitions",
        ]
        for key, definition in payload["metric_definitions"].items():
            lines.append(f"- `{key}`: {definition}")
        lines.extend([
            "",
            "## Revised Audit Table",
            "| Metric | baseline | cockpit_only | change |",
            "|---|---:|---:|---:|",
        ])
        for row in comparison_rows:
            change = "n/a" if row["percent_change_from_baseline"] is None else f"{row['percent_change_from_baseline']:+.1f}%"
            lines.append(f"| {row['label']} | {row['baseline']} | {row['cockpit_only']} | {change} |")
        lines.extend([
            "",
            "## COCPIT Transparency Implementation Check",
        ])
        for key, value in payload["cockpit_transparency_implementation_check"].items():
            lines.append(f"- {key.replace('_', ' ')}: {'yes' if value else 'no'}")
        lines.extend([
            "",
            "## Raw Movement Interpretation",
            f"- Raw movement delta: {'n/a' if raw_delta_pct is None else f'{raw_delta_pct:+.1f}%'}",
            f"- Stable within +/-15%: {'yes' if raw_stable else 'no'}",
            f"- Interpretation: {payload['movement_difference_interpretation']}",
            "",
            "## Transparent-Condition Trace Examples",
        ])
        for index, example in enumerate(payload["transparent_condition_examples"][:8], 1):
            lines.extend([
                f"### Example {index}: {example['outcome']}",
                (
                    f"- t={example['timestep']}s, {example['initiator_role']} "
                    f"({example['initiator_mode']}, {example['initiator_zone']}) -> "
                    f"{example['target_role']} ({example['target_mode']}, {example['target_zone']})"
                ),
                (
                    f"- distance={example['distance_m']}m, mutual_visibility={example['mutual_visibility']}, "
                    f"would_be_blocked_in_baseline={example['would_be_blocked_in_baseline']}"
                ),
                (
                    f"- approach/pause={example['approach_or_pause_occurred']}, "
                    f"logged={example['interaction_logged']}, type={example['interaction_type']}, zone={example['zone']}"
                ),
                "",
            ])
        if not payload["transparent_condition_examples"]:
            lines.append("No transparent-condition examples matched this audit definition.")
        lines.extend([
            "",
            "## Interpretation",
            (
                "Mechanism observed: cockpit_only increases perceived/attended moving staff from COCPIT and produces transparent-only approach/pause or interaction traces that baseline visibility would block."
                if payload["mechanism_observed"] else
                "Mechanism not observed in this short run. Do not proceed to the full Part 2 batch until this is resolved."
            ),
        ])
        md_path.write_text("\n".join(lines))

    return payload


def run_condition_experiments(
    params: Optional[Mapping[str, object]] = None,
    n_runs: Optional[int] = None,
    interaction_backend: Optional[str] = None,
    persona_source: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    model_variant: Optional[str] = None,
    save_outputs: bool = True,
) -> dict:
    """Run Baseline / COCPIT / NURSTA / Both and compare against empirical data."""

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    empirical = compute_empirical_summary().as_dict()
    run_count = config.EXPERIMENT_ENSEMBLE_RUNS if n_runs is None else n_runs
    duration_seconds = config.EXPERIMENT_DURATION_SECONDS if duration_seconds is None else duration_seconds
    duration_label = _duration_label(duration_seconds)
    backend_name = config.EXPERIMENT_INTERACTION_BACKEND if interaction_backend is None else interaction_backend
    model_variant = config.MODEL_VARIANT if model_variant is None else model_variant
    overrides = _parameter_to_config_overrides(params or {})

    results = {}
    metric_rows = []
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    transcript_source_interactions = []
    condition_simulations: Dict[str, Simulation] = {}
    agent_shift_narratives: Dict[str, Dict[str, str]] = {}

    for condition_index, condition_name in enumerate(config.EXPERIMENT_CONDITIONS):
        run_summaries = []
        all_interactions = []
        simulation_for_plot = None

        with _temporary_config_updates(overrides):
            for run_index in range(run_count):
                simulation = Simulation(
                    random_seed=config.RUN_BASE_SEED + run_index,
                    condition_name=condition_name,
                    interaction_backend=backend_name,
                    persona_source=persona_source or config.PERSONA_SOURCE,
                    model_variant=model_variant,
                )
                assert simulation.condition_spec.name == condition_name, (
                    f"Simulation was given condition {simulation.condition_spec.name} "
                    f"but expected {condition_name}"
                )
                while simulation.timestep < duration_seconds:
                    simulation.step()
                run_summaries.append(compute_summary_statistics(simulation.interaction_log, simulation))
                all_interactions.extend(simulation.interaction_log)
                simulation_for_plot = simulation
                if not transcript_source_interactions and condition_name == "baseline":
                    transcript_source_interactions = list(simulation.interaction_log)

        aggregated = aggregate_run_summaries(run_summaries)
        if simulation_for_plot is None:
            continue

        visibility_metrics = simulation_for_plot.visibility_metrics_snapshot()
        visibility_metrics.update(_visibility_graph_metrics(simulation_for_plot))
        aggregated["visibility_metrics"] = visibility_metrics
        aggregated["visibility_comparison"] = _visibility_comparison_metrics(simulation_for_plot)
        aggregated["interaction_cloud_metrics"] = _interaction_cloud_metrics(all_interactions)
        aggregated["mean_bedside_duration"] = _mean_duration_for_types(
            all_interactions,
            {"nursing_task", "doctor_task"},
        )
        aggregated["mean_non_bedside_duration"] = _mean_duration_for_types(
            all_interactions,
            {"opportunistic_corridor", "opportunistic_station", "nurse_doctor_handoff"},
        )
        aggregated["distance_to_empirical"] = distance_to_empirical(aggregated, empirical)
        aggregated["spatial_cosine_similarity"] = cosine_similarity(
            np.asarray(aggregated["kde_grid"], dtype=float),
            np.asarray(empirical["kde_grid"], dtype=float),
        )
        aggregated["condition_name"] = condition_name
        aggregated["interaction_backend"] = backend_name
        aggregated["persona_source"] = persona_source or config.PERSONA_SOURCE
        aggregated["model_variant"] = model_variant
        aggregated["duration_seconds"] = duration_seconds
        aggregated["duration_label"] = duration_label
        results[condition_name] = aggregated
        condition_simulations[condition_name] = simulation_for_plot
        agent_shift_narratives[condition_name] = {
            str(agent.gid): simulation_for_plot.generate_agent_shift_narrative(agent.gid)
            for agent in simulation_for_plot.staff_agents
        }

        metric_rows.append(
            {
                "condition_name": condition_name,
                "duration_label": duration_label,
                "distance_to_empirical": aggregated["distance_to_empirical"],
                "spatial_cosine_similarity": aggregated["spatial_cosine_similarity"],
                "interaction_count_mean": aggregated.get("interaction_count_mean", 0.0),
                "interaction_count_std": aggregated.get("interaction_count_std", 0.0),
                "mean_visual_integration": visibility_metrics["mean_visual_integration"],
                "cockpit_isovist_area": visibility_metrics["mean_isovist_area_by_key_point"]["COCPIT"],
                "nursta_isovist_area": visibility_metrics["mean_isovist_area_by_key_point"]["NURSTA"],
                "interaction_centroid_y": aggregated["interaction_cloud_metrics"]["interaction_cloud_centroid_y"],
                "nurse_doctor_interaction_count": aggregated["interaction_cloud_metrics"]["nurse_doctor_interaction_count"],
                "corridor_interaction_share": aggregated["interaction_cloud_metrics"]["corridor_interaction_share"],
                "corr04_corr05_interaction_count": aggregated["interaction_cloud_metrics"]["corr04_corr05_interaction_count"],
                "mean_nurse_doctor_approach_distance": aggregated["interaction_cloud_metrics"]["mean_nurse_doctor_approach_distance"],
            }
        )

        axis = axes.ravel()[condition_index]
        _plot_condition_heatmap(
            axis,
            simulation_for_plot,
            all_interactions,
            f"{condition_name} (n={len(all_interactions)})",
        )

    if len(results) == 4:
        baseline = results["baseline"]
        baseline_visibility = baseline["visibility_comparison"]
        baseline_cloud = baseline["interaction_cloud_metrics"]
        results["hypothesis_checks"] = {
            "V1_cocpit_isovist_increases_20pct": {
                "result": results["cockpit_only"]["visibility_comparison"]["mean_isovist_area_cocpit_centroid"]
                >= baseline_visibility["mean_isovist_area_cocpit_centroid"] * 1.2,
                "baseline_area": baseline_visibility["mean_isovist_area_cocpit_centroid"],
                "cocpit_area": results["cockpit_only"]["visibility_comparison"]["mean_isovist_area_cocpit_centroid"],
                "pct_change": (
                    (
                        results["cockpit_only"]["visibility_comparison"]["mean_isovist_area_cocpit_centroid"]
                        - baseline_visibility["mean_isovist_area_cocpit_centroid"]
                    )
                    / max(baseline_visibility["mean_isovist_area_cocpit_centroid"], 1e-9)
                )
                * 100.0,
            },
            "V2_nursta_integration_increases": {
                "result": results["nursta_only"]["visibility_metrics"]["mean_visual_integration"]
                >= baseline["visibility_metrics"]["mean_visual_integration"],
                "baseline_integration": baseline["visibility_metrics"]["mean_visual_integration"],
                "nursta_integration": results["nursta_only"]["visibility_metrics"]["mean_visual_integration"],
            },
            "V3_both_strongest_visual_coupling": {
                "result": max(
                    ("baseline", baseline["visibility_comparison"]["agents_visible_from_cocpit_mean"]),
                    ("cockpit_only", results["cockpit_only"]["visibility_comparison"]["agents_visible_from_cocpit_mean"]),
                    ("nursta_only", results["nursta_only"]["visibility_comparison"]["agents_visible_from_cocpit_mean"]),
                    ("both", results["both"]["visibility_comparison"]["agents_visible_from_cocpit_mean"]),
                    key=lambda item: item[1],
                )[0]
                == "both",
                "visible_agents_from_cocpit": {
                    condition_name: results[condition_name]["visibility_comparison"]["agents_visible_from_cocpit_mean"]
                    for condition_name in ("baseline", "cockpit_only", "nursta_only", "both")
                },
            },
            "M1_cocpit_increases_approach_distance": {
                "result": results["cockpit_only"]["interaction_cloud_metrics"]["mean_nurse_doctor_approach_distance"]
                > baseline["interaction_cloud_metrics"]["mean_nurse_doctor_approach_distance"],
                "baseline_m": baseline["interaction_cloud_metrics"]["mean_nurse_doctor_approach_distance"],
                "cocpit_m": results["cockpit_only"]["interaction_cloud_metrics"]["mean_nurse_doctor_approach_distance"],
                "note": "Transparent walls allow agents to initiate contact from further away",
            },
            "M2_nursta_shifts_nurse_dwell_upward": {
                "result": _nurse_dwell_centroid_y(results["nursta_only"], condition_simulations["nursta_only"])
                > _nurse_dwell_centroid_y(baseline, condition_simulations["baseline"]),
                "baseline_centroid_y": _nurse_dwell_centroid_y(baseline, condition_simulations["baseline"]),
                "nursta_centroid_y": _nurse_dwell_centroid_y(results["nursta_only"], condition_simulations["nursta_only"]),
            },
            "I1_cocpit_increases_nurse_doctor_frequency": {
                "result": results["cockpit_only"]["interaction_cloud_metrics"]["nurse_doctor_interaction_count"]
                >= baseline_cloud["nurse_doctor_interaction_count"],
                "baseline_count": baseline_cloud["nurse_doctor_interaction_count"],
                "cocpit_count": results["cockpit_only"]["interaction_cloud_metrics"]["nurse_doctor_interaction_count"],
                "pct_change": (
                    (
                        results["cockpit_only"]["interaction_cloud_metrics"]["nurse_doctor_interaction_count"]
                        - baseline_cloud["nurse_doctor_interaction_count"]
                    )
                    / max(baseline_cloud["nurse_doctor_interaction_count"], 1)
                )
                * 100.0,
            },
            "I2_nursta_increases_corr04_corr05_interactions": {
                "result": results["nursta_only"]["interaction_cloud_metrics"]["corr04_corr05_interaction_count"]
                > baseline_cloud["corr04_corr05_interaction_count"],
                "baseline_corr04_corr05_count": baseline_cloud["corr04_corr05_interaction_count"],
                "nursta_corr04_corr05_count": results["nursta_only"]["interaction_cloud_metrics"]["corr04_corr05_interaction_count"],
                "note": "Tests interactions specifically near the new NURSTA position, not total corridor share",
            },
            "I3_bedside_remains_longest": {
                "result": all(
                    results[condition_name]["mean_bedside_duration"]
                    >= results[condition_name]["mean_non_bedside_duration"]
                    for condition_name in ("baseline", "cockpit_only", "nursta_only", "both")
                ),
                "bedside_means": {
                    condition_name: results[condition_name]["mean_bedside_duration"]
                    for condition_name in ("baseline", "cockpit_only", "nursta_only", "both")
                },
                "non_bedside_means": {
                    condition_name: results[condition_name]["mean_non_bedside_duration"]
                    for condition_name in ("baseline", "cockpit_only", "nursta_only", "both")
                },
            },
            "I4_both_best_spatial_match": {
                "result": (
                    max(
                        ("baseline", baseline["spatial_cosine_similarity"]),
                        ("cockpit_only", results["cockpit_only"]["spatial_cosine_similarity"]),
                        ("nursta_only", results["nursta_only"]["spatial_cosine_similarity"]),
                        ("both", results["both"]["spatial_cosine_similarity"]),
                        key=lambda item: item[1],
                    )[0]
                    == "both"
                ),
                "cosine_similarities": {
                    condition_name: results[condition_name]["spatial_cosine_similarity"]
                    for condition_name in ("baseline", "cockpit_only", "nursta_only", "both")
                },
            },
        }

    if save_outputs:
        summary_path = _labeled_output_path(config.EXPERIMENT_SUMMARY_PATH, duration_label)
        delta_table_path = _labeled_output_path(config.CONDITION_DELTA_TABLE_PATH, duration_label)
        metrics_path = _labeled_output_path(config.EXPERIMENT_METRICS_CSV_PATH, duration_label)
        heatmap_path = _labeled_output_path(config.CONDITION_HEATMAPS_PATH, duration_label)

        config.EXPERIMENT_SUMMARY_PATH.write_text(json.dumps(results, indent=2))
        summary_path.write_text(json.dumps(results, indent=2))
        delta_table_text = _format_delta_table(results)
        config.CONDITION_DELTA_TABLE_PATH.write_text(delta_table_text)
        delta_table_path.write_text(delta_table_text)
        print(delta_table_text)
        if metric_rows:
            with config.EXPERIMENT_METRICS_CSV_PATH.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(metric_rows[0].keys()))
                writer.writeheader()
                for row in metric_rows:
                    writer.writerow(row)
            with metrics_path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(metric_rows[0].keys()))
                writer.writeheader()
                for row in metric_rows:
                    writer.writerow(row)
        figure.tight_layout()
        figure.savefig(config.EXPERIMENT_HEATMAP_GRID_PATH, dpi=300)
        figure.savefig(config.CONDITION_HEATMAPS_PATH, dpi=300)
        figure.savefig(heatmap_path, dpi=300)
        transcript_payload = _build_transcript_sample(transcript_source_interactions or [])
        config.INTERACTION_TRANSCRIPT_SAMPLE_PATH.write_text(json.dumps(transcript_payload, indent=2))
        config.AGENT_SHIFT_NARRATIVES_PATH.write_text(json.dumps(agent_shift_narratives, indent=2))
    plt.close(figure)

    return results
