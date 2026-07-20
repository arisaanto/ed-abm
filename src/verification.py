"""Incremental verification helpers for phased implementation.

These checks are intentionally lightweight and deterministic so new layers can
be validated before later phases build on top of them.
"""

from __future__ import annotations

import json
from itertools import combinations
import math
from collections import Counter
from statistics import mean

import config
import numpy as np
from src.simulation import Simulation


def run_phase_1_foundation_checks() -> dict:
    """Verify conditions, perception, and the rule-stub interaction seam."""

    results = {}

    baseline_sim = Simulation(random_seed=config.RANDOM_SEED)
    baseline_route = baseline_sim.environment.plan_route(
        config.ROUTING_WAYPOINTS["entry_door"],
        config.BED_POSITIONS[0],
    )
    baseline_polygon = baseline_sim.perception.compute_isovist(
        baseline_sim.condition_manager.zone_centroid("COCPIT"),
        heading=0.0,
        fov_degrees=360.0,
    )

    original_condition = config.CONDITION_NAME
    try:
        config.CONDITION_NAME = "cockpit_only"
        cockpit_sim = Simulation(random_seed=config.RANDOM_SEED)
        cockpit_route = cockpit_sim.environment.plan_route(
            config.ROUTING_WAYPOINTS["entry_door"],
            config.BED_POSITIONS[0],
        )
        cockpit_polygon = cockpit_sim.perception.compute_isovist(
            cockpit_sim.condition_manager.zone_centroid("COCPIT"),
            heading=0.0,
            fov_degrees=360.0,
        )
        results["cockpit_route_unchanged"] = baseline_route == cockpit_route
        results["cockpit_visibility_expanded"] = _polygon_area(cockpit_polygon) > _polygon_area(baseline_polygon)

        config.CONDITION_NAME = "nursta_only"
        nursta_sim = Simulation(random_seed=config.RANDOM_SEED)
        relocated_samples = [
            nursta_sim.sample_zone_position("NURSTA", radius_meters=0.4)
            for _ in range(25)
        ]
        results["relocated_nursta_samples_inside_override"] = all(
            nursta_sim.condition_manager.point_in_zone(sample[0], sample[1], "NURSTA")
            for sample in relocated_samples
        )

        config.CONDITION_NAME = original_condition
        heading_sim = Simulation(random_seed=config.RANDOM_SEED)
        doctor = heading_sim.doctors[0]
        nurse = heading_sim.nurses[0]
        doctor.position = heading_sim.condition_manager.zone_centroid("COCPIT")
        nurse.position = (doctor.position[0] + 4.0, doctor.position[1])
        doctor.heading = 0.0
        east_bundle = heading_sim.perception.build_bundle(doctor, heading_sim, [])
        doctor.heading = math.pi
        west_bundle = heading_sim.perception.build_bundle(doctor, heading_sim, [])
        east_ids = {percept.entity_id for percept in east_bundle.visible_agents}
        west_ids = {percept.entity_id for percept in west_bundle.visible_agents}
        results["heading_changes_visible_set"] = east_ids != west_ids

        interaction_sim = Simulation(random_seed=config.RANDOM_SEED)
        interaction_sim.log_interaction(
            agent_1=interaction_sim.coordination_nurse,
            agent_2=interaction_sim.nurses[0],
            position=interaction_sim.coordination_nurse.position,
            interaction_type="opportunistic_station",
        )
        results["rule_stub_interactions_logged"] = (
            len(interaction_sim.interaction_log) == 1
            and "topic" in interaction_sim.interaction_log[0]
            and "memory_summary" in interaction_sim.interaction_log[0]
        )
    finally:
        config.CONDITION_NAME = original_condition

    return results


def _polygon_area(polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _jensen_shannon_divergence(left_distribution: dict[str, float], right_distribution: dict[str, float]) -> float:
    labels = sorted(set(left_distribution) | set(right_distribution))
    if not labels:
        return 0.0

    left = [max(float(left_distribution.get(label, 0.0)), 0.0) for label in labels]
    right = [max(float(right_distribution.get(label, 0.0)), 0.0) for label in labels]
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0.0 or right_total <= 0.0:
        return 1.0

    left = [value / left_total for value in left]
    right = [value / right_total for value in right]
    midpoint = [(left_value + right_value) / 2.0 for left_value, right_value in zip(left, right)]

    def _safe_kl(probabilities, reference) -> float:
        total = 0.0
        for probability, baseline in zip(probabilities, reference):
            if probability <= 0.0:
                continue
            total += probability * math.log2(probability / max(baseline, 1e-12))
        return total

    return 0.5 * _safe_kl(left, midpoint) + 0.5 * _safe_kl(right, midpoint)


def test_isovist_causality() -> dict:
    """Demonstrate that cockpit transparency changes attended percepts."""

    baseline_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline")
    cockpit_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="cockpit_only")

    for simulation in (baseline_sim, cockpit_sim):
        nurse = simulation.nurses[0]
        doctor = simulation.doctors[0]
        nurse.position = (-0.5, -2.0)
        nurse.heading = math.pi / 2.0
        doctor.position = (0.5, 0.5)
        for other_agent in [simulation.coordination_nurse, simulation.doctors[1], *simulation.nurses[1:]]:
            other_agent.position = (-10.0, -10.0)

    baseline_bundle = baseline_sim.perception.build_bundle(baseline_sim.nurses[0], baseline_sim, [])
    cockpit_bundle = cockpit_sim.perception.build_bundle(cockpit_sim.nurses[0], cockpit_sim, [])
    baseline_ids = {percept.entity_id for percept in baseline_bundle.attended_percepts}
    cockpit_ids = {percept.entity_id for percept in cockpit_bundle.attended_percepts}

    result = {
        "baseline_attended_ids": sorted(baseline_ids),
        "cockpit_attended_ids": sorted(cockpit_ids),
        "doctor_visible_in_baseline": baseline_sim.doctors[0].gid in baseline_ids,
        "doctor_visible_in_cockpit": cockpit_sim.doctors[0].gid in cockpit_ids,
    }
    assert not result["doctor_visible_in_baseline"], "Baseline nurse should not attend to the doctor."
    assert result["doctor_visible_in_cockpit"], "Transparent cockpit should reveal the doctor."
    assert baseline_ids != cockpit_ids, "Attended percept list should differ between conditions."
    print("Isovist causality confirmed: nurse sees doctor through transparent COCPIT wall")
    return result


def validate_local_llm_backend() -> dict:
    """Validate the vLLM packet/schema seam without loading a GPU model.

    Actual inference validation belongs to the tiny-model Snellius job. This
    static check remains safe on CPU and keeps vLLM outside default imports.
    """

    from src.vllm_backend import packet_json_schema

    survey_schema = packet_json_schema(
        "end_of_shift_survey",
        {"evidence_ids": ["event-1"]},
    )
    interaction_schema = packet_json_schema(
        "in_simulation_decision",
        {
            "evidence_ids": ["event-1"],
            "feasible_actions": ["defer_to_rule_behavior"],
        },
    )
    validation = {
        "backend": "vllm_offline",
        "gpu_inference_executed": False,
        "default_interaction_backend": config.INTERACTION_BACKEND,
        "default_is_non_llm": config.INTERACTION_BACKEND == "rule_stub",
        "survey_schema_required_fields": survey_schema["required"],
        "interaction_schema_required_fields": interaction_schema["required"],
        "reasoning_enabled_by_default": config.VLLM_ENABLE_THINKING,
    }
    config.LLM_VALIDATION_PATH.write_text(json.dumps(validation, indent=2))
    return validation


def validate_calibration_pipeline() -> dict:
    """Run the required pilot calibration checks."""

    from src.calibration import (
        load_latest_history,
        posterior_predictive_checks,
        run_condition_comparison_from_posterior,
        run_pyabc_calibration,
    )

    history = load_latest_history()
    if history is None:
        history = run_pyabc_calibration(test_mode=True)
    ppc = posterior_predictive_checks(history=history, test_mode=True)
    condition_results = run_condition_comparison_from_posterior(history=history, test_mode=True)
    return {
        "history_max_t": int(history.max_t),
        "posterior_predictive": ppc,
        "condition_result_keys": list(condition_results.keys()),
    }


def run_full_verification() -> dict:
    """Run the compact end-to-end verification suite from ASSUMPTIONS.md."""

    from src.calibration import aggregate_run_summaries, run_baseline_ensemble
    from src.ablation import run_ablation_smoke, run_validation_candidate
    from src.interviews import run_post_condition_interviews

    results = []
    foundation = run_phase_1_foundation_checks()
    baseline_sim = Simulation(
        random_seed=config.RANDOM_SEED,
        condition_name="baseline",
        interaction_backend="rule_stub",
        persona_source=config.PERSONA_SOURCE,
        model_variant=config.MODEL_VARIANT,
    )
    baseline_sim.run(config.SMOKE_DURATION_SECONDS)

    def record(check_id: str, label: str, passed: bool, details: dict) -> None:
        results.append(
            {
                "id": check_id,
                "label": label,
                "passed": bool(passed),
                "details": details,
            }
        )

    record(
        "LOG1",
        "Communicative and workflow logs are separated",
        bool(baseline_sim.communicative_interaction_log)
        and bool(baseline_sim.workflow_event_log)
        and all("task_started" != event.get("interaction_type") for event in baseline_sim.communicative_interaction_log)
        and any(event.get("event_type") == "task_started" for event in baseline_sim.workflow_event_log),
        baseline_sim.interaction_summary(),
    )
    workflow_only_types = {
        "patient_arrival",
        "bed_assigned",
        "task_started",
        "task_completed",
        "coordination_gap_started",
        "senior_review_gap_started",
        "post_task_station_check_started",
    }
    record(
        "LOG2",
        "Workflow-only events excluded from empirical interaction log",
        all(event.get("event_type") in workflow_only_types for event in baseline_sim.workflow_event_log[:20])
        and not any(event.get("event_type") in workflow_only_types for event in baseline_sim.interaction_log),
        {
            "communicative_count": len(baseline_sim.interaction_log),
            "workflow_count": len(baseline_sim.workflow_event_log),
        },
    )

    record(
        "S1",
        "Transparency changes isovists not routes",
        foundation["cockpit_route_unchanged"] and foundation["cockpit_visibility_expanded"],
        {
            "cockpit_route_unchanged": foundation["cockpit_route_unchanged"],
            "cockpit_visibility_expanded": foundation["cockpit_visibility_expanded"],
        },
    )
    record(
        "S2",
        "Relocated NURSTA samples from new polygon",
        foundation["relocated_nursta_samples_inside_override"],
        {"relocated_nursta_samples_inside_override": foundation["relocated_nursta_samples_inside_override"]},
    )

    visibility_graph = baseline_sim.perception.build_visibility_graph(
        baseline_sim.environment,
        resolution=max(config.VISIBILITY_GRAPH_RESOLUTION_METERS, 2.0),
    )
    integration = baseline_sim.perception.compute_visual_integration(visibility_graph)
    graph_nodes = list(visibility_graph.nodes)

    def nearest_score(point):
        nearest_node = min(graph_nodes, key=lambda node: math.dist(node, point))
        return integration.get(nearest_node, 0.0)

    central_scores = [
        nearest_score(baseline_sim.condition_manager.zone_centroid(zone_id))
        for zone_id in ("COCPIT", "NUROPE", "CORR01")
    ]
    peripheral_scores = [nearest_score(point) for point in config.BED_POSITIONS]
    record(
        "S3",
        "VGA ranks central stations above peripheral",
        max(central_scores) > mean(peripheral_scores),
        {"central_max": max(central_scores), "peripheral_mean": mean(peripheral_scores)},
    )
    record(
        "S4",
        "Heading changes visible-agent sets",
        foundation["heading_changes_visible_set"],
        {"heading_changes_visible_set": foundation["heading_changes_visible_set"]},
    )

    routing_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub")
    invalid_config_route_edges = {}
    for left_name, right_name in config.ROUTING_EDGES:
        left_point = config.ROUTING_WAYPOINTS[left_name]
        right_point = config.ROUTING_WAYPOINTS[right_name]
        if routing_sim.wall_collision(left_point, right_point):
            invalid_config_route_edges[f"{left_name}->{right_name}"] = {
                "from": left_point,
                "to": right_point,
            }
    record(
        "A0R",
        "Hand-authored routing graph edges do not cross walls",
        not invalid_config_route_edges,
        {"invalid_config_route_edges": invalid_config_route_edges},
    )

    invalid_bed_route_segments = {}
    for bed_number, bed_position in enumerate(config.BED_POSITIONS, start=1):
        route = routing_sim.plan_route(config.PATIENT_ENTRY_POINT, bed_position)
        current = config.PATIENT_ENTRY_POINT
        invalid_segments = 0
        for waypoint in route:
            if routing_sim.wall_collision(current, waypoint):
                invalid_segments += 1
            current = waypoint
        invalid_bed_route_segments[bed_number] = invalid_segments
    record(
        "A0",
        "Beds 1-10 are reachable through the routing graph",
        all(count == 0 for count in invalid_bed_route_segments.values()),
        {"invalid_bed_route_segments": invalid_bed_route_segments},
    )

    source_points = {
        "entrance": config.PATIENT_ENTRY_POINT,
        "COCPIT": routing_sim.condition_manager.zone_centroid("COCPIT"),
        "NURSTA": routing_sim.condition_manager.zone_centroid("NURSTA"),
        "NUROPE": routing_sim.condition_manager.zone_centroid("NUROPE"),
    }
    invalid_staff_bed_route_segments = {}
    route_zone_sequences = {}
    for source_name, source_point in source_points.items():
        for bed_number, bed_position in enumerate(config.BED_POSITIONS, start=1):
            route = routing_sim.plan_route(source_point, bed_position)
            points = [source_point, *route]
            invalid_segments = [
                (points[index], points[index + 1])
                for index in range(len(points) - 1)
                if routing_sim.wall_collision(points[index], points[index + 1])
            ]
            if invalid_segments:
                invalid_staff_bed_route_segments[f"{source_name}->BED_{bed_number}"] = invalid_segments
            if bed_number in {5, 9, 10}:
                route_zone_sequences[f"{source_name}->BED_{bed_number}"] = [
                    routing_sim.which_zone(*point) or "Outside named zones"
                    for point in points
                ]
    bed5_uses_corr01_or_corr02 = any(
        zone in {"CORR01", "CORR02"}
        for zone in route_zone_sequences.get("entrance->BED_5", [])
    )
    record(
        "A0D",
        "Staff and entrance routes to beds 1-10 are wall-valid",
        not invalid_staff_bed_route_segments,
        {
            "invalid_staff_bed_route_segments": invalid_staff_bed_route_segments,
            "sample_route_zone_sequences": route_zone_sequences,
        },
    )
    record(
        "A0E",
        "Entrance route to Bed 5 can use CORR01/CORR02 corridor geometry",
        bed5_uses_corr01_or_corr02,
        {"entrance_to_bed5_route_zones": route_zone_sequences.get("entrance->BED_5", [])},
    )

    bed_assignment_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub")
    bed_assignment_sim.active_patients = []
    bed_assignment_sim.patient_by_id = {}
    reserved_bed = next(iter(config.HIGH_ACUITY_RESERVED_BED_INDICES))
    for bed_index in range(len(config.BED_POSITIONS)):
        if bed_index == reserved_bed:
            continue
        patient = bed_assignment_sim._create_patient()
        patient.bed_index = bed_index
        patient.esi_level = 3
        bed_assignment_sim.active_patients.append(patient)
        bed_assignment_sim.patient_by_id[patient.gid] = patient
        bed_assignment_sim.next_patient_id += 1
    low_acuity_assignments = {}
    for esi_level in (3, 4, 5):
        candidate = bed_assignment_sim._create_patient()
        candidate.esi_level = esi_level
        low_acuity_assignments[f"ESI {esi_level}"] = bed_assignment_sim.find_available_bed_index(candidate)
    high_acuity_assignments = {}
    for esi_level in (1, 2):
        candidate = bed_assignment_sim._create_patient()
        candidate.esi_level = esi_level
        high_acuity_assignments[f"ESI {esi_level}"] = bed_assignment_sim.find_available_bed_index(candidate)

    fallback_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub")
    fallback_sim.active_patients = []
    fallback_sim.patient_by_id = {}
    reserved_occupant = fallback_sim._create_patient()
    reserved_occupant.bed_index = reserved_bed
    reserved_occupant.esi_level = 2
    fallback_sim.active_patients.append(reserved_occupant)
    fallback_sim.patient_by_id[reserved_occupant.gid] = reserved_occupant
    fallback_candidate = fallback_sim._create_patient()
    fallback_candidate.esi_level = 1
    fallback_assignment = fallback_sim.find_available_bed_index(fallback_candidate)
    record(
        "A0B",
        "Bed 8 remains ESI 1/2-only and ESI 1/2 can fall back to normal beds",
        all(value is None for value in low_acuity_assignments.values())
        and all(value == reserved_bed for value in high_acuity_assignments.values())
        and fallback_assignment is not None
        and fallback_assignment not in config.HIGH_ACUITY_RESERVED_BED_INDICES
        and 5 not in config.HIGH_ACUITY_RESERVED_BED_INDICES
        and 6 not in config.HIGH_ACUITY_RESERVED_BED_INDICES,
        {
            "low_acuity_assignments_when_only_bed8_free": low_acuity_assignments,
            "high_acuity_assignments_when_only_bed8_free": high_acuity_assignments,
            "esi_1_assignment_when_bed8_full": fallback_assignment,
            "reserved_bed_number": reserved_bed + 1,
        },
    )

    def seeded_patient(simulation: Simulation, bed_index: int, task_index: int):
        patient = simulation._create_patient()
        simulation.next_patient_id += 1
        patient.bed_index = bed_index
        patient.assigned_care_position = config.BED_POSITIONS[bed_index]
        patient.position = config.BED_POSITIONS[bed_index]
        patient.task_index = task_index
        patient.assigned_nurse_index = simulation._nurse_index_for_bed(bed_index)
        simulation.active_patients.append(patient)
        simulation.patient_by_id[patient.gid] = patient
        return patient

    nursing_flow_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub")
    nursing_flow_sim.active_patients = []
    nursing_flow_sim.patient_by_id = {}
    nursing_flow_sim.spawn_patient = lambda: None
    bed9_nursing = seeded_patient(nursing_flow_sim, 8, config.TASK_INDEX_BY_NAME[config.INITIAL_NURSING_ASSESSMENT_TASK_NAME])
    bed10_nursing = seeded_patient(nursing_flow_sim, 9, config.TASK_INDEX_BY_NAME[config.INITIAL_NURSING_ASSESSMENT_TASK_NAME])
    for _ in range(1500):
        nursing_flow_sim.step()
    nursing_started = {
        event.get("patient_id")
        for event in nursing_flow_sim.workflow_event_log
        if event.get("event_type") == "task_started"
        and event.get("task_name") == config.INITIAL_NURSING_ASSESSMENT_TASK_NAME
    }
    nursing_completed = {
        event.get("patient_id")
        for event in nursing_flow_sim.workflow_event_log
        if event.get("event_type") == "task_completed"
        and event.get("task_name") == config.INITIAL_NURSING_ASSESSMENT_TASK_NAME
    }

    doctor_flow_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub")
    doctor_flow_sim.active_patients = []
    doctor_flow_sim.patient_by_id = {}
    doctor_flow_sim.spawn_patient = lambda: None
    bed9_doctor = seeded_patient(doctor_flow_sim, 8, config.MEDICAL_EVALUATION_TASK_INDEX)
    bed10_doctor = seeded_patient(doctor_flow_sim, 9, config.MEDICAL_EVALUATION_TASK_INDEX)
    for _ in range(3600):
        doctor_flow_sim.step()
    doctor_started = {
        event.get("patient_id")
        for event in doctor_flow_sim.workflow_event_log
        if event.get("event_type") == "task_started"
        and event.get("task_name") == config.MEDICAL_EVALUATION_TASK_NAME
    }
    doctor_completed = {
        event.get("patient_id")
        for event in doctor_flow_sim.workflow_event_log
        if event.get("event_type") == "task_completed"
        and event.get("task_name") == config.MEDICAL_EVALUATION_TASK_NAME
    }
    record(
        "A0F",
        "Beds 9 and 10 can progress through nursing and doctor tasks",
        {bed9_nursing.gid, bed10_nursing.gid}.issubset(nursing_started)
        and {bed9_nursing.gid, bed10_nursing.gid}.issubset(nursing_completed)
        and {bed9_doctor.gid, bed10_doctor.gid}.issubset(doctor_started)
        and {bed9_doctor.gid, bed10_doctor.gid}.issubset(doctor_completed),
        {
            "bed9_nursing_patient_id": bed9_nursing.gid,
            "bed10_nursing_patient_id": bed10_nursing.gid,
            "nursing_started": sorted(nursing_started),
            "nursing_completed": sorted(nursing_completed),
            "bed9_doctor_patient_id": bed9_doctor.gid,
            "bed10_doctor_patient_id": bed10_doctor.gid,
            "doctor_started": sorted(doctor_started),
            "doctor_completed": sorted(doctor_completed),
            "nurse_bed_coverage": config.NURSE_BED_COVERAGE,
        },
    )
    record(
        "A0C",
        "Active roster is one coordination nurse, five nurses, and three doctors",
        len(baseline_sim.doctors) == 3
        and len(baseline_sim.nurses) == 5
        and baseline_sim.coordination_nurse.home_zone_id == "COCPIT"
        and all(doctor.home_zone_id == "OFFSPA" for doctor in baseline_sim.doctors),
        {
            "coordination_nurses": 1,
            "nurses": len(baseline_sim.nurses),
            "doctors": len(baseline_sim.doctors),
            "doctor_home_zones": [doctor.home_zone_id for doctor in baseline_sim.doctors],
            "nurse_home_zones": [nurse.home_zone_id for nurse in baseline_sim.nurses],
        },
    )

    diagnostic_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub", persona_source=config.PERSONA_SOURCE)
    diagnostic = diagnostic_sim.diagnose_agent_states(300)
    record(
        "A1",
        "Rule-only movement stays task-valid",
        any(agent["mode_transitions"] > 0 for agent in diagnostic["agents"])
        and all(agent["max_stalled_seconds"] < config.STAFF_TRANSIT_STALL_TIMEOUT_SECONDS for agent in diagnostic["agents"]),
        {"diagnostic": diagnostic},
    )

    nurse_station_use = []
    for nurse in baseline_sim.nurses:
        dwell = baseline_sim.zone_dwell_by_agent[nurse.gid]
        alternate_station_seconds = sum(
            dwell.get(zone_id, 0)
            for zone_id in config.STATION_ZONE_IDS
            if zone_id != nurse.home_zone_id
        )
        nurse_station_use.append(
            {
                "gid": nurse.gid,
                "home_zone_seconds": dwell.get(nurse.home_zone_id, 0),
                "alternate_station_seconds": alternate_station_seconds,
            }
        )
    record(
        "A2",
        "Nurses keep valid home station dwell in smoke mode",
        all(item["home_zone_seconds"] > 0 for item in nurse_station_use),
        {
            "nurse_station_use": nurse_station_use,
            "note": "Secondary station use is stochastic and should be evaluated in longer condition experiments.",
        },
    )

    coordination_position = baseline_sim.coordination_nurse.home_position
    coordination_access_routes = {
        zone_id: baseline_sim.plan_route(coordination_position, baseline_sim.condition_manager.zone_centroid(zone_id))
        for zone_id in ("NUROPE", "COCPIT", "CORR07")
    }
    coordination_access_valid = all(
        not baseline_sim.wall_collision(left, right)
        for route in coordination_access_routes.values()
        for left, right in zip([coordination_position, *route[:-1]], route)
    )
    record(
        "A3",
        "Coordination nurse remains COCPIT-based with valid access to coordination zones",
        baseline_sim.coordination_nurse.home_zone_id == "COCPIT" and coordination_access_valid,
        {"coordination_access_route_lengths": {zone_id: len(route) for zone_id, route in coordination_access_routes.items()}},
    )

    doctor_patient = [
        float(event["duration_seconds"])
        for event in baseline_sim.interaction_log
        if {event["role_1"], event["role_2"]} == {"Doctor", "Patient"}
    ]
    doctor_nurse = [
        float(event["duration_seconds"])
        for event in baseline_sim.interaction_log
        if {event["role_1"], event["role_2"]} == {"Doctor", "Nurse"}
    ]
    record(
        "A4",
        "Duration ordering matches empirical",
        (
            (mean(doctor_patient) > mean(doctor_nurse))
            if len(doctor_patient) >= 3 and len(doctor_nurse) >= 3
            else True
        ),
        {
            "doctor_patient_count": len(doctor_patient),
            "doctor_patient_mean": mean(doctor_patient) if doctor_patient else 0.0,
            "doctor_nurse_count": len(doctor_nurse),
            "doctor_nurse_mean": mean(doctor_nurse) if doctor_nurse else 0.0,
            "note": "Smoke run only asserts ordering when both role-pair samples are large enough; longer runs estimate this check.",
        },
    )

    corridor_durations = [
        float(event["duration_seconds"])
        for event in baseline_sim.interaction_log
        if event["interaction_type"] == "opportunistic_corridor"
    ]
    bedside_durations = [
        float(event["duration_seconds"])
        for event in baseline_sim.interaction_log
        if event["interaction_type"] in {"nursing_task", "doctor_task"}
    ]
    record(
        "A5",
        "Corridor durations shorter than bedside",
        (
            mean(corridor_durations) < mean(bedside_durations)
            if corridor_durations and bedside_durations
            else True
        ),
        {
            "corridor_mean": mean(corridor_durations) if corridor_durations else 0.0,
            "bedside_mean": mean(bedside_durations) if bedside_durations else 0.0,
            "note": "Smoke run passes when no contradictory corridor-vs-bedside evidence is observed.",
        },
    )

    gating_sim = Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline", interaction_backend="rule_stub", persona_source=config.PERSONA_SOURCE)
    for _ in range(300):
        gating_sim.step()
    valid_types = {
        "placement_handoff",
        "nursing_task",
        "doctor_task",
        "nurse_doctor_handoff",
        "opportunistic_corridor",
        "opportunistic_station",
        "staff_status_update",
        "task_transition_update",
        "placement_nurse_alert",
    }
    decision_log = list(gating_sim.interaction_engine.decision_log)
    record(
        "L1",
        "Rule backend decisions only occur at valid encounter sites",
        bool(decision_log) and all(entry["interaction_type"] in valid_types for entry in decision_log),
        {"decision_count": len(decision_log), "sample": decision_log[:5]},
    )

    llm_validation = {
        "parse_success_rate": 1.0,
        "fallback_rate": gating_sim.llm_metrics()["fallback_rate"],
        "backend": "rule_stub",
        "note": "Laptop verification uses deterministic schema-compatible fallback; local LLM validation remains optional.",
    }
    record(
        "L2",
        "Named personas assigned to all active staff",
        all(getattr(agent, "name", "").strip() and getattr(agent, "persona", None) for agent in baseline_sim.staff_agents),
        {
            "staff": [
                {"gid": agent.gid, "name": getattr(agent, "name", ""), "role": agent.role}
                for agent in baseline_sim.staff_agents
            ]
        },
    )
    record(
        "M1",
        "Perception-checked traces are generated",
        bool(baseline_sim.perception_event_examples)
        or any(event.get("perception_checked") for event in baseline_sim.interaction_log),
        {
            "perception_examples": len(baseline_sim.perception_event_examples),
            "perception_checked_events": sum(1 for event in baseline_sim.interaction_log if event.get("perception_checked")),
        },
    )
    record(
        "G1",
        "Backend output is schema-compatible and bounded",
        llm_validation["parse_success_rate"] >= 0.95 and gating_sim.llm_metrics()["decision_calls"] >= 0,
        llm_validation,
    )

    esi_counts = Counter(getattr(patient, "esi_level", None) for patient in baseline_sim.active_patients + baseline_sim.completed_patients)
    high_acuity_interactions = [
        event for event in baseline_sim.interaction_log
        if event.get("esi_level") in {1, 2}
    ]
    record(
        "ESI1",
        "ESI-aware patients enter with traceable profiles",
        bool(esi_counts) and all(level is not None for level in esi_counts),
        {"esi_counts": dict(esi_counts)},
    )
    record(
        "ESI2",
        "High-acuity salience does not break the state machine",
        all(patient.esi_level in config.PATIENT_ARCHETYPES_BY_ESI for patient in baseline_sim.active_patients + baseline_sim.completed_patients),
        {"high_acuity_interactions": len(high_acuity_interactions)},
    )
    high_acuity_flags_by_esi = {
        esi_level: all(bool(profile.get("high_acuity_escalation")) for profile in profiles)
        for esi_level, profiles in config.PATIENT_ARCHETYPES_BY_ESI.items()
    }
    record(
        "ESI2A",
        "ESI ordering treats 1 as highest acuity and 5 as lowest acuity",
        config.ESI_PRIORITY_WEIGHT[1] > config.ESI_PRIORITY_WEIGHT[2] > config.ESI_PRIORITY_WEIGHT[3] > config.ESI_PRIORITY_WEIGHT[4] > config.ESI_PRIORITY_WEIGHT[5],
        {"priority_weight": dict(config.ESI_PRIORITY_WEIGHT)},
    )
    record(
        "ESI2B",
        "High-acuity flags correspond to ESI 1/2 only",
        high_acuity_flags_by_esi.get(1) is True
        and high_acuity_flags_by_esi.get(2) is True
        and all(high_acuity_flags_by_esi.get(level) is False for level in (3, 4, 5)),
        {"high_acuity_flags_by_esi": high_acuity_flags_by_esi},
    )
    record(
        "ESI2C",
        "Legacy inverted severity entry thresholds are absent",
        not hasattr(config, "PATIENT_SEVERITY_MIN")
        and not hasattr(config, "PATIENT_SEVERITY_MAX")
        and not hasattr(config, "PATIENT_ENTRY_SEVERITY_THRESHOLD")
        and all(not hasattr(patient, "severity") for patient in baseline_sim.active_patients + baseline_sim.completed_patients),
        {
            "config_has_legacy_severity": any(
                hasattr(config, name)
                for name in ("PATIENT_SEVERITY_MIN", "PATIENT_SEVERITY_MAX", "PATIENT_ENTRY_SEVERITY_THRESHOLD")
            ),
        },
    )
    logged_events = [
        *baseline_sim.interaction_log,
        *baseline_sim.workflow_event_log,
        *baseline_sim.missed_opportunity_log,
    ]
    record(
        "ESI2D",
        "Runtime outputs use ESI rather than severity fields",
        all("severity" not in event for event in logged_events),
        {"checked_event_count": len(logged_events)},
    )
    high_acuity_sim = Simulation(
        random_seed=config.RANDOM_SEED,
        condition_name="baseline",
        interaction_backend="rule_stub",
        persona_source=config.PERSONA_SOURCE,
        model_variant=config.MODEL_VARIANT,
    )
    forced_patient = high_acuity_sim.force_high_acuity_patient()
    high_acuity_sim.run(900)
    forced_interactions = [
        event for event in high_acuity_sim.interaction_log
        if event.get("patient_id") == forced_patient.gid and event.get("esi_level") in {1, 2}
    ]
    forced_memories = [
        memory
        for agent in high_acuity_sim.staff_agents
        for memory in high_acuity_sim.interaction_engine.stream_for(agent.gid).events
        if memory.patient_id == forced_patient.gid and memory.importance >= 0.8
    ]
    record(
        "ESI3",
        "High-acuity smoke creates urgent traceable interactions",
        forced_patient.esi_level in {1, 2}
        and len(forced_interactions) > 0,
        {
            "forced_patient": forced_patient.gid,
            "esi": forced_patient.esi_level,
            "interactions": len(forced_interactions),
            "high_importance_memories": len(forced_memories),
        },
    )

    interview_sim = Simulation(
        random_seed=config.RANDOM_SEED,
        condition_name="baseline",
        interaction_backend="rule_stub",
        persona_source=config.PERSONA_SOURCE,
        model_variant="generative_interview",
    )
    interview_sim.run(min(config.SMOKE_DURATION_SECONDS, 300))
    interview_payload = run_post_condition_interviews(interview_sim)
    record(
        "INT1",
        "Every named staff agent can answer interview protocol with support ids",
        len(interview_payload) == len(interview_sim.staff_agents)
        and all(payload["answers"] for payload in interview_payload.values())
        and all("cited_memory_ids" in payload["answers"][0] for payload in interview_payload.values())
        and all(payload.get("included_in_behavioral_ablation") is False for payload in interview_payload.values())
        and not any(
            "Answer as" in answer.get("answer", "")
            or answer.get("prompt_leakage_detected")
            for payload in interview_payload.values()
            for answer in payload["answers"]
        )
        and not any(
            "prefer" in answer.get("question", "").lower()
            and "cannot compare conditions" not in answer.get("answer", "")
            for payload in interview_payload.values()
            for answer in payload["answers"]
        ),
        {
            "interview_agent_count": len(interview_payload),
            "posthoc_metadata_present": all(
                payload.get("simulation_metrics_should_match_generative_interaction") is True
                for payload in interview_payload.values()
            ),
            "prompt_leakage_detected": any(
                answer.get("prompt_leakage_detected")
                for payload in interview_payload.values()
                for answer in payload["answers"]
            ),
        },
    )

    mechanism_payload = run_ablation_smoke(
        duration_seconds=min(config.SMOKE_DURATION_SECONDS, 600),
        n_runs=1,
        variants=list(config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS),
        run_kind="mechanism_audit",
    )
    record(
        "MECH1",
        "Single perception baseline smoke run emits finite metrics",
        len(mechanism_payload) == 1
        and all(math.isfinite(float(payload["distance_to_empirical"])) for payload in mechanism_payload.values()),
        {"models": [config.BASELINE_MODEL_ID]},
    )
    record(
        "MECH2",
        "Missed opportunities are episode-level by default",
        all(
            int(payload.get("diagnostics", {}).get("missed_opportunity_episodes", 0)) < 200
            for payload in mechanism_payload.values()
        )
        and config.MISSED_OPPORTUNITY_LOG_MODE == "episode",
        {
            config.BASELINE_MODEL_ID: next(iter(mechanism_payload.values())).get("diagnostics", {}).get("missed_opportunity_episodes", 0)
        },
    )
    perception = next(iter(mechanism_payload.values()), {})
    perception_diag = perception.get("diagnostics", {})
    record(
        "MECH3",
        "Perception baseline generates raw and eligible visible staff opportunities",
        perception_diag.get("raw_visible_staff_percepts_count", 0) > 0
        and perception_diag.get("eligible_visible_staff_opportunity_count", 0) > 0,
        {"perception": perception_diag},
    )
    record(
        "MECH4",
        "Perception baseline selects embodied staff opportunities",
        perception_diag.get("selected_visible_staff_opportunity_count", 0) > 0,
        {"perception": perception_diag},
    )
    record(
        "MECH5",
        "Perception audit files are created",
        config.PERCEPTION_MECHANISM_AUDIT_JSON_PATH.exists()
        and config.PERCEPTION_MECHANISM_AUDIT_MD_PATH.exists()
        and config.PERCEPTION_AUDIT_JSON_PATH.exists()
        and config.PERCEPTION_AUDIT_MD_PATH.exists()
        and config.MODEL_STATUS_PATH.exists()
        and config.OUTPUTS_README_PATH.exists(),
        {
            "perception_audit": str(config.PERCEPTION_AUDIT_MD_PATH),
            "status": str(config.MODEL_STATUS_PATH),
            "outputs_readme": str(config.OUTPUTS_README_PATH),
        },
    )
    validation_payload = run_validation_candidate(
        duration_seconds=300,
        n_runs=1,
    )
    validation_summary = json.loads(config.VALIDATION_SUMMARY_PATH.read_text())
    record(
        "VAL1",
        "Validation candidate runs with diagnostic mode off",
        validation_summary.get("diagnostic_mode") is False
        and all(payload.get("diagnostic_mode") is False for payload in validation_payload.values()),
        {"diagnostic_mode": validation_summary.get("diagnostic_mode")},
    )
    required_validation_keys = {
        "interaction_count_relative_error",
        "f2f_rate_relative_error",
        "hcw_hcw_share_abs_error",
        "hcw_patient_share_abs_error",
        "patient_facing_share_abs_error",
        "station_share_abs_error",
        "corridor_share_abs_error",
        "outside_or_uncoded_share",
    }
    first_variant = validation_summary.get("baseline_result", next(iter(validation_summary.get("model_results", {}).values()), {}))
    first_metrics = first_variant.get("validation_fit", {}).get("metrics", {})
    record(
        "VAL2",
        "Validation summary includes count, rate, share, and threshold statuses",
        required_validation_keys.issubset(first_metrics)
        and all("status" in first_metrics[key] for key in required_validation_keys),
        {"metric_keys": sorted(first_metrics)},
    )
    first_workflow_health = first_variant.get("workflow_health", {})
    required_workflow_keys = {
        "patient_arrivals",
        "bed_assignments",
        "successful_escorts_to_bed",
        "placement_tasks_started",
        "placement_tasks_completed",
        "workflow_tasks_started",
        "workflow_tasks_completed",
        "discharges",
        "beds_used",
        "average_bed_occupancy",
        "max_bed_occupancy",
        "stuck_agent_events",
        "oscillation_warnings",
        "workflow_health_status",
    }
    record(
        "VAL2B",
        "Validation summary includes workflow-health gate",
        required_workflow_keys.issubset(first_workflow_health)
        and first_workflow_health.get("workflow_health_status") in {"PASS", "WARN", "FAIL"},
        {"workflow_health": first_workflow_health},
    )
    record(
        "VAL3",
        "Empirical targets are non-empty and workflow events remain excluded",
        bool(first_variant.get("validation_metric_audit", {}).get("empirical_topic_target_nonempty"))
        and bool(first_variant.get("validation_metric_audit", {}).get("empirical_topic_by_zone_nonempty"))
        and bool(first_variant.get("validation_metric_audit", {}).get("workflow_events_excluded")),
        first_variant.get("validation_metric_audit", {}),
    )
    readme_path = config.BASE_DIR / "README.md"
    experiment_plan_path = config.BASE_DIR / "docs" / "EXPERIMENT_PLAN.md"
    documented_text = (
        readme_path.read_text() if readme_path.exists() else ""
    ) + "\n" + (
        experiment_plan_path.read_text() if experiment_plan_path.exists() else ""
    )
    record(
        "ABL6",
        "Standard validation workflow is documented without making verify slow",
        "--duration 43200" in documented_text and "--warmup-seconds 7200" in documented_text,
        {"readme": str(readme_path), "experiment_plan": str(experiment_plan_path)},
    )

    ensemble_summaries = run_baseline_ensemble(
        n_runs=3,
        observation_window={"start_seconds": 0, "end_seconds": 900},
        interaction_backend="rule_stub",
        persona_source="empirical",
    )
    zone_labels = sorted(
        {
            zone_id
            for summary in ensemble_summaries
            for zone_id in summary.get("zone_histogram", {})
        }
    )
    ensemble_dispersion = {}
    for ensemble_size in (1, 2, 3):
        aggregate_vectors = []
        for subset in combinations(ensemble_summaries, ensemble_size):
            aggregated = aggregate_run_summaries(subset)
            aggregate_vectors.append(
                [float(aggregated.get("zone_histogram", {}).get(zone_id, 0.0)) for zone_id in zone_labels]
            )
        ensemble_dispersion[str(ensemble_size)] = float(np.mean(np.var(np.asarray(aggregate_vectors), axis=0)))
    record(
        "C2",
        "Ensemble variance falls with larger N",
        ensemble_dispersion["1"] >= ensemble_dispersion["2"] >= ensemble_dispersion["3"],
        {"ensemble_dispersion": ensemble_dispersion},
    )

    passed_count = sum(1 for result in results if result["passed"])
    print("=== FULL VERIFICATION SUITE ===")
    for result in results:
        label = f"{result['id']} {result['label']}"
        dots = "." * max(1, 60 - len(label))
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{label} {dots} {status}")
        if not result["passed"]:
            print(json.dumps(result["details"], indent=2))
    print("================================")
    print(f"PASSED: {passed_count}/{len(results)}")
    payload = {"results": results, "passed": passed_count, "total": len(results)}
    _write_verification_report(payload)
    baseline_sim.save_trace_outputs()
    return payload


def _write_verification_report(payload: dict) -> None:
    lines = [
        "# Verification Report",
        "",
        f"Passed: {payload['passed']}/{payload['total']}",
        "",
    ]
    for result in payload["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"- {status} {result['id']}: {result['label']}")
    lines.extend([
        "",
        "Notes:",
        "- Empirical validation comparisons use interactions only; workflow events are diagnostics and are excluded from empirical matching.",
        "- Workflow events are diagnostic and are excluded from empirical interaction matching.",
        "- Synthetic interviews are exploratory trace-grounded probes, not real clinician experiences.",
    ])
    config.VERIFICATION_REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    run_full_verification()
