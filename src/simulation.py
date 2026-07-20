"""Simulation orchestration for the Phase 1 ED ABM."""

from __future__ import annotations

import math
import os
import random
import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional

import config

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(config.MPL_CONFIG_DIR))

import matplotlib

if not (config.ANIMATION_MODE and not config.SAVE_ANIMATION):
    matplotlib.use(config.MATPLOTLIB_BACKEND)
import matplotlib.pyplot as plt
from src.analysis import compute_kde_grid
from src.agents import CoordinationNurse, Doctor, Nurse, Patient, StaffAgent
from src.conditions import ConditionManager, get_condition_spec
from src.environment import Environment
from src.interaction import InteractionEngine
from src.perception import Percept, PerceptionService
from src.personas import persona_for_staff


@dataclass
class StaffInteractionIntent:
    """Pending embodied staff-staff interaction created by a visible percept."""

    intent_id: str
    initiator_id: int
    target_id: int
    created_time: int
    initial_distance: float
    initial_initiator_position: tuple[float, float]
    initial_target_position: tuple[float, float]
    initial_initiator_zone: Optional[str]
    initial_target_zone: Optional[str]
    current_target_position: tuple[float, float]
    interaction_type: str
    interaction_source: str
    perception_gate_result: dict
    salience_score: Optional[float]
    initial_nonproximate: bool
    timeout_seconds: int
    status: str = "pending"
    failure_reason: Optional[str] = None
    saved_initiator_state: dict = field(default_factory=dict)
    saved_target_state: dict = field(default_factory=dict)
    stationary_initiated_to_moving: bool = False
    route_overridden: bool = False
    target_route_overridden: bool = False
    target_paused: bool = False
    target_moved_toward_initiator: bool = False
    target_continued_route: bool = False
    last_distance: float = 0.0
    completed_time: Optional[int] = None

    @property
    def target_hold_expires_at(self) -> int:
        return self.created_time + int(config.PERCEPTION_STAFF_INTERACTION_TARGET_HOLD_SECONDS)


class Simulation:
    """Owns environment, agents, interaction logging, and visualization."""

    def __init__(
        self,
        random_seed: Optional[int] = None,
        condition_name: Optional[str] = None,
        interaction_backend: Optional[str] = None,
        persona_source: Optional[str] = None,
        model_variant: Optional[str] = None,
        scope_mode: Optional[str] = None,
        enable_senior_doctor_oversight: Optional[bool] = None,
        scenario_mode: Optional[str] = None,
        scenario_start_hour: Optional[int] = None,
        part3_episode_logging_enabled: Optional[bool] = None,
        part3_max_decision_episodes: Optional[int] = None,
        part3_episode_logging_start_seconds: int = 0,
        part3_cognitive_controller: Optional[object] = None,
        part3_isolate_exogenous_arrival_stream: bool = False,
    ) -> None:
        self.random_seed = config.RANDOM_SEED if random_seed is None else random_seed
        self.random = random.Random(self.random_seed)
        self.scenario_mode = config.DEFAULT_SCENARIO_MODE if scenario_mode is None else scenario_mode
        if self.scenario_mode not in config.SCENARIO_MODES:
            raise ValueError(f"Unsupported scenario_mode: {self.scenario_mode}")
        self.scenario_definition = dict(config.SCENARIO_MODES[self.scenario_mode])
        self.scenario_start_hour = (
            int(config.DEFAULT_SCENARIO_START_HOUR)
            if scenario_start_hour is None
            else int(scenario_start_hour)
        ) % 24
        self.scope_mode = config.DEFAULT_SCOPE_MODE
        self.enable_senior_doctor_oversight = (
            bool(config.ENABLE_SENIOR_DOCTOR_OVERSIGHT)
            if enable_senior_doctor_oversight is None
            else bool(enable_senior_doctor_oversight)
        )
        self.model_variant = config.MODEL_VARIANT if model_variant is None else model_variant
        if self.model_variant not in config.MODEL_VARIANTS and self.model_variant not in config.POSTHOC_ANALYSIS_MODES:
            raise ValueError(f"Unsupported model_variant: {self.model_variant}")
        self.environment = Environment(
            wall_path=config.WALL_POSITIONS_PATH,
            zone_path=config.ZONE_BOUNDARIES_PATH,
        )
        self.condition_spec = get_condition_spec(config.CONDITION_NAME if condition_name is None else condition_name)
        self.active_care_area_objects = []
        self.condition_manager = ConditionManager(self.environment, self.condition_spec, self.active_care_area_objects)
        self.perception = PerceptionService(self.condition_manager)
        self.persona_source = config.PERSONA_SOURCE if persona_source is None else persona_source
        self.part3_episode_logging_enabled = (
            bool(config.PART3_EPISODE_LOGGING_ENABLED)
            if part3_episode_logging_enabled is None
            else bool(part3_episode_logging_enabled)
        )
        self.part3_max_decision_episodes = (
            int(config.PART3_MAX_DECISION_EPISODES_PER_RUN)
            if part3_max_decision_episodes is None
            else max(int(part3_max_decision_episodes), 0)
        )
        self.part3_episode_logging_start_seconds = max(
            int(part3_episode_logging_start_seconds),
            0,
        )
        # Part 1/2 behavior is unchanged unless a controller is explicitly injected.
        self.part3_cognitive_controller = part3_cognitive_controller
        self.part3_exogenous_arrival_stream_isolated = bool(
            part3_isolate_exogenous_arrival_stream
            or part3_cognitive_controller is not None
        )
        self.part3_exogenous_arrival_stream_version = (
            "scenario_seed_common_random_numbers_v1"
            if self.part3_exogenous_arrival_stream_isolated
            else None
        )
        if self.part3_exogenous_arrival_stream_isolated:
            arrival_seed_material = (
                "part3-exogenous-arrivals-v1|"
                f"{self.scenario_mode}|{self.random_seed}"
            )
            self.part3_exogenous_arrival_stream_seed = int.from_bytes(
                hashlib.sha256(
                    arrival_seed_material.encode("utf-8")
                ).digest()[:8],
                byteorder="big",
                signed=False,
            )
            self.arrival_random = random.Random(
                self.part3_exogenous_arrival_stream_seed
            )
        else:
            # Preserve the validated Part 1/2 random-number lifecycle exactly.
            self.part3_exogenous_arrival_stream_seed = None
            self.arrival_random = self.random
        self.interaction_backend_name = (
            config.INTERACTION_BACKEND if interaction_backend is None else interaction_backend
        )
        if self.model_variant in {config.BASELINE_MODEL_ID, "traditional_rule", "perception_rule", "memory_rule"}:
            self.interaction_backend_name = "rule_stub"
        self.routing_waypoints = (
            self.condition_manager.adjusted_routing_waypoints()
            if hasattr(self.condition_manager, "adjusted_routing_waypoints")
            else dict(self.environment.routing_waypoints)
        )

        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        self.timestep = 0
        self.variant_settings = self._variant_settings(self.model_variant)
        self.interaction_probability_multiplier = 1.0
        self.interaction_engine = InteractionEngine(
            self.interaction_backend_name,
            self.random,
            model_variant=self.model_variant,
            memory_enabled=self.variant_settings["memory_enabled"],
        )
        self.next_patient_id = config.PATIENT_ID_START
        self.communicative_interaction_log: List[Dict[str, object]] = []
        # Compatibility alias: user-facing reports call these "interactions";
        # workflow events are kept separately and are not validation records.
        self.interaction_log = self.communicative_interaction_log
        self.workflow_event_log: List[Dict[str, object]] = []
        self.missed_opportunity_log: List[Dict[str, object]] = []
        self.part3_decision_episode_log: List[Dict[str, object]] = []
        self.part3_decision_episode_candidates_seen = 0
        self._part3_decision_episode_priorities: Dict[str, int] = {}
        self.missed_opportunity_debug_counts: Counter = Counter()
        self.active_missed_opportunity_since: Dict[tuple, int] = {}
        self.last_missed_opportunity_logged_at: Dict[tuple, int] = {}
        self.interruption_log: List[Dict[str, object]] = []
        self.ablation_diagnostics: Counter = Counter()
        self.patient_facing_task_attendance_counts: Counter = Counter()
        self.patient_facing_validation_interaction_counts: Counter = Counter()
        self.last_patient_facing_validation_by_patient: Dict[int, int] = {}
        self.patient_facing_in_task_logged_keys: set[tuple[int, int, str]] = set()
        self.patient_facing_in_task_considered_keys: set[tuple[int, int, str]] = set()
        self.next_communicative_event_index = 1
        self.next_workflow_event_index = 1
        self.next_missed_event_index = 1
        self.active_patients: List[Patient] = []
        self.completed_patients: List[Patient] = []
        self.reserved_doctor_ids_this_step: set[int] = set()
        self.last_opportunistic_corridor_interaction_by_pair: Dict[tuple[int, int], int] = {}
        self.last_opportunistic_station_interaction_by_pair: Dict[tuple[int, int], int] = {}
        self.last_coordination_hub_interaction_by_pair: Dict[tuple[int, int], int] = {}
        self.last_doctor_coreview_by_pair: Dict[tuple[int, int], int] = {}
        self.last_task_transition_update_by_pair: Dict[tuple[int, int], int] = {}
        self.logged_task_transition_update_episode_keys: set[tuple[object, ...]] = set()
        self.last_bedside_cotask_interaction_by_pair_patient: Dict[tuple[int, int, int], int] = {}
        self.bedside_cotask_logged_keys: set[tuple[int, int, int, str]] = set()
        self.last_senior_doctor_review_by_pair: Dict[tuple[int, int], int] = {}
        self.last_perceived_staff_interaction_by_pair: Dict[tuple[int, int], int] = {}
        self.pending_staff_interaction_intents: Dict[str, StaffInteractionIntent] = {}
        self.completed_staff_interaction_intents: List[StaffInteractionIntent] = []
        self.failed_staff_interaction_intents: List[StaffInteractionIntent] = []
        self.next_staff_interaction_intent_index = 1
        self.perception_checked_interaction_count = 0
        self.interactions_logged_without_perception_check = 0
        self.pending_task_transition_updates: List[Dict[str, object]] = []
        self.pending_senior_doctor_reviews: List[Dict[str, object]] = []
        self.perception_event_examples: List[Dict[str, object]] = []
        self.currently_interacting_pairs: set[tuple[int, int]] = set()
        self.next_station_episode_index = 1
        self.active_station_episode_by_agent: Dict[int, dict] = {}
        self.station_same_episode_interaction_keys: set[tuple[object, ...]] = set()
        self.last_reason_interaction_by_key: Dict[tuple[object, ...], int] = {}
        self.recent_task_update_episode_by_context: Dict[tuple[object, ...], int] = {}
        self.high_acuity_preemption_attempted_at: Dict[tuple[int, str], int] = {}
        self.deferred_arrivals = 0
        self.deferred_arrivals_by_esi: Counter = Counter()
        self.bed_assignment_history: set[int] = set()
        self.next_bed_assignment_cursor = 0
        self.zone_dwell_by_role: Dict[str, Counter] = defaultdict(Counter)
        self.zone_dwell_by_agent: Dict[int, Counter] = defaultdict(Counter)
        self.mode_dwell_by_role: Dict[str, Counter] = defaultdict(Counter)
        self.mode_dwell_by_agent: Dict[int, Counter] = defaultdict(Counter)
        self.zone_transition_counts: Counter = Counter()
        self.movement_distance_by_role: Counter = Counter()
        self.movement_distance_by_agent: Counter = Counter()
        self.mutual_visibility_counts_by_zone: Counter = Counter()
        # Logging-only person-time exposure counters. A mutually visible pair
        # contributes one timestep to each staff member; mechanics never read
        # these values.
        self.mutual_visibility_exposure_seconds_by_agent: Counter = Counter()
        self.any_mutually_visible_staff_seconds_by_agent: Counter = Counter()
        self.mutual_visibility_exposure_seconds_by_agent_zone: Dict[
            int, Counter
        ] = defaultdict(Counter)
        self.last_zone_by_agent: Dict[int, Optional[str]] = {}
        self.zone_entered_at_by_agent: Dict[int, int] = {}
        self.last_cluster_by_agent: Dict[int, str] = {}
        self.trip_counts: Counter = Counter()
        self.trip_path_lengths_by_label: Dict[str, List[float]] = defaultdict(list)
        self.trip_durations_by_label: Dict[str, List[float]] = defaultdict(list)
        self.bed_occupancy_samples: List[int] = []
        self.ordinary_bed_occupancy_samples: List[int] = []
        self.special_bed_occupancy_samples: List[int] = []
        self.waiting_queue_samples: List[int] = []
        self.ed_pressure_index_samples: List[float] = []
        self.active_high_acuity_patient_samples: List[int] = []
        self.active_esi1_patient_samples: List[int] = []
        self.active_esi2_patient_samples: List[int] = []
        self.active_esi1_esi2_patient_samples: List[int] = []
        self.ordinary_bed_open_seconds_by_bed: Counter = Counter()
        self.ordinary_bed_open_samples_by_bed: Dict[str, List[int]] = defaultdict(list)
        self.zero_waiting_seconds = 0
        self.waiting_pressure_seconds = 0
        self.ordinary_beds_full_seconds = 0
        self.open_ordinary_bed_seconds = 0
        self.high_acuity_active_seconds = 0
        self.first_nurse_seen_patient_ids: set[int] = set()
        self.first_doctor_seen_patient_ids: set[int] = set()
        self.assignment_repair_events = 0
        self.handoff_wait_repair_events = 0
        self.stuck_agent_events = 0
        self.unsupported_assignment_rejections: list[dict] = []
        self.route_movement_issues: list[dict] = []

        self.coordination_nurse = self._create_coordination_nurse()
        self.nurses = self._create_nurses()
        self.doctors = self._create_doctors()

        self.staff_agents: List[StaffAgent] = [
            self.coordination_nurse,
            *self.doctors,
            *self.nurses,
        ]
        self._assign_staff_personas()

        self.patient_by_id: Dict[int, Patient] = {}
        self.nurse_by_id = {nurse.gid: nurse for nurse in self.nurses}
        self.doctor_by_id = {doctor.gid: doctor for doctor in self.doctors}
        for agent in self.all_agents:
            self.last_zone_by_agent[agent.gid] = self.which_zone(*agent.position)
            self.zone_entered_at_by_agent[agent.gid] = self.timestep
            self.last_cluster_by_agent[agent.gid] = self.location_cluster(agent.position)
        self._seed_initial_patients()

    def _variant_settings(self, model_variant: str) -> dict:
        return {
            "perception_required": True,
            "memory_enabled": model_variant in {"memory_rule", "generative_interaction", "generative_interview"},
            "generative_enabled": model_variant in {"generative_interaction", "generative_interview"},
            "interviews_enabled": model_variant in config.POSTHOC_ANALYSIS_MODES,
            "included_in_behavioral_ablation": model_variant in config.MODEL_VARIANTS,
            "simulation_metrics_should_match_generative_interaction": model_variant in config.POSTHOC_ANALYSIS_MODES,
        }

    def _scenario_hour_index(self) -> int:
        elapsed_hours = int(self.timestep // 3600)
        return int((self.scenario_start_hour + elapsed_hours) % 24)

    def scenario_clock_metadata(self, duration_seconds: int, warmup_seconds: int = 0) -> dict:
        duration_hours = int(math.ceil(max(duration_seconds, 0) / 3600))
        warmup_hours = int(math.ceil(max(warmup_seconds, 0) / 3600))
        end_hour = (self.scenario_start_hour + duration_hours) % 24
        warmup_end_hour = (self.scenario_start_hour + warmup_hours) % 24
        evaluation_hours = [
            (self.scenario_start_hour + hour) % 24
            for hour in range(warmup_hours, duration_hours)
        ]
        profile = list(self.scenario_definition.get("hourly_arrival_multipliers") or [1.0] * 24)
        return {
            "scenario_start_hour": int(self.scenario_start_hour),
            "simulated_clock_window": f"{self.scenario_start_hour:02d}:00-{end_hour:02d}:00",
            "warmup_clock_window": f"{self.scenario_start_hour:02d}:00-{warmup_end_hour:02d}:00",
            "evaluation_clock_window": f"{warmup_end_hour:02d}:00-{end_hour:02d}:00",
            "evaluation_hour_indices": evaluation_hours,
            "evaluation_hourly_arrival_multipliers": [
                float(profile[hour % len(profile)]) if profile else 1.0
                for hour in evaluation_hours
            ],
        }

    def _scenario_hourly_multiplier(self) -> float:
        profile = list(self.scenario_definition.get("hourly_arrival_multipliers") or [1.0] * 24)
        if not profile:
            return 1.0
        return float(profile[self._scenario_hour_index() % len(profile)])

    def _scenario_arrival_rate_per_second(self) -> float:
        return (
            float(config.ARRIVAL_RATE_PER_SECOND)
            * float(self.scenario_definition.get("arrival_rate_multiplier", 1.0))
            * self._scenario_hourly_multiplier()
        )

    def _scenario_high_acuity_probability_per_second(self) -> float:
        return (
            float(config.HIGH_ACUITY_EVENT_PROBABILITY_PER_SECOND)
            * float(self.scenario_definition.get("high_acuity_event_probability_multiplier", 1.0))
        )

    def _scenario_esi_arrival_weights(self) -> Mapping[int, float]:
        weights = self.scenario_definition.get("esi_arrival_weights")
        if isinstance(weights, Mapping):
            return {int(level): float(weight) for level, weight in weights.items()}
        return config.ESI_ARRIVAL_WEIGHTS

    def active_care_area_object_ids(self) -> set[str]:
        return set()

    def station_interaction_zone_ids(self) -> set[str]:
        return set(config.STATION_ZONE_IDS)

    def station_dwell_zone_ids(self, role: Optional[str] = None) -> set[str]:
        dwell_ids = set(config.STATION_ZONE_IDS)
        for item in self.active_care_area_objects:
            if not bool(item.get("supports_dwell", False)):
                continue
            if role is not None and role not in set(item.get("allowed_roles", [])):
                continue
            dwell_ids.add(str(item["id"]))
        return dwell_ids

    def corridor_or_threshold_interaction_zone_ids(self) -> set[str]:
        return {zone_id for zone_id in self.condition_manager.zone_polygons if str(zone_id).startswith("CORR")}

    def care_area_registry(self) -> List[dict]:
        return []

    def scope_roster(self) -> Mapping[str, object]:
        return {
            "staff_counts": config.STAFF_COUNTS,
            "coordination_nurse_home_zone_id": config.COORDINATION_NURSE_HOME_ZONE_ID,
            "doctor_home_zone_id": config.DOCTOR_HOME_ZONE_ID,
            "nurse_home_zone_ids": config.NURSE_HOME_ZONE_IDS,
            "patient_care_positions": config.BED_POSITIONS,
            "nurse_bed_coverage": config.NURSE_BED_COVERAGE,
        }

    def patient_care_positions(self) -> List[tuple[float, float]]:
        return [tuple(point) for point in self.scope_roster().get("patient_care_positions", config.BED_POSITIONS)]

    def staff_access_position_for_patient(self, patient: Patient) -> tuple[float, float]:
        if patient.bed_index is not None:
            access_positions = getattr(config, "BED_ACCESS_POSITIONS", config.BED_POSITIONS)
            if patient.bed_index < len(access_positions):
                return tuple(access_positions[patient.bed_index])
        return tuple(patient.bed_position or patient.position)

    def _patient_assignment_destination(self, patient: Patient) -> tuple[float, float]:
        return self.staff_access_position_for_patient(patient)

    def patient_assignment_route_supported(self, staff_member: StaffAgent, patient: Patient) -> bool:
        return self._route_supported(
            tuple(staff_member.position),
            self._patient_assignment_destination(patient),
        )

    def record_unsupported_assignment_rejection(
        self,
        staff_member: StaffAgent,
        patient: Patient,
        destination: tuple[float, float],
        context: str,
        reason: str = "route_unsupported",
    ) -> None:
        entry = {
            "timestep": int(self.timestep),
            "scenario_mode": self.scenario_mode,
            "condition_name": self.condition_spec.name,
            "seed": int(self.random_seed),
            "staff_role": getattr(staff_member, "role", None),
            "staff_agent_id": getattr(staff_member, "gid", None),
            "patient_id": getattr(patient, "gid", None),
            "patient_esi": getattr(patient, "esi_level", None),
            "patient_current_task": getattr(patient, "current_task_name", None),
            "patient_bed_index": getattr(patient, "bed_index", None),
            "position": tuple(getattr(staff_member, "position", (None, None))),
            "destination": tuple(destination),
            "context": str(context),
            "reason": str(reason),
        }
        self.unsupported_assignment_rejections.append(entry)
        self.ablation_diagnostics["unsupported_assignment_rejection_count"] += 1
        self.ablation_diagnostics[f"unsupported_assignment_rejection_role_{getattr(staff_member, 'role', 'unknown')}"] += 1
        self.ablation_diagnostics[f"unsupported_assignment_rejection_context_{context}"] += 1

    def nurse_bed_coverage(self) -> Mapping[int, List[int]]:
        return config.NURSE_BED_COVERAGE

    def staff_counts(self) -> Mapping[str, int]:
        return config.STAFF_COUNTS

    def _assign_staff_personas(self) -> None:
        for agent in self.staff_agents:
            persona = persona_for_staff(agent.gid, agent.role, source=self.persona_source)
            agent.name = persona.name
            if getattr(agent, "senior_oversight_only", False):
                agent.name = f"Senior Doctor {agent.gid}"
            agent.persona = persona
            agent.persona_source_trace = persona.source
            agent.memory_stream = self.interaction_engine.stream_for(agent.gid)

    def staff_by_id(self) -> Dict[int, StaffAgent]:
        return {agent.gid: agent for agent in self.staff_agents}

    def _zone_centroid(self, zone_id: str) -> tuple[float, float]:
        return self.condition_manager.zone_centroid(zone_id)

    def which_zone(self, x: float, y: float) -> Optional[str]:
        return self.condition_manager.which_zone(x, y)

    def _sample_start_position(
        self,
        zone_id: str,
        radius_meters: float,
        center_point: Optional[tuple[float, float]] = None,
    ) -> tuple[float, float]:
        if zone_id == "COCPIT":
            return self.sample_station_position(zone_id, avoid_point=center_point)
        return self.sample_zone_position(
            zone_id=zone_id,
            radius_meters=radius_meters,
            center_point=center_point,
        )

    def wall_collision(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        if hasattr(self.condition_manager, "wall_collision"):
            return self.condition_manager.wall_collision(start, end)
        return self.environment.wall_collision(start, end)

    def has_line_of_sight(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        if hasattr(self.condition_manager, "has_line_of_sight"):
            return self.condition_manager.has_line_of_sight(start, end)
        return self.environment.has_line_of_sight(start, end)

    def plan_route(self, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
        if hasattr(self.condition_manager, "plan_route"):
            return self.condition_manager.plan_route(start, end)
        return self.environment.plan_route(start, end)

    def route_is_walkable(self, start: tuple[float, float], route: list[tuple[float, float]]) -> bool:
        if hasattr(self.condition_manager, "route_is_walkable"):
            return self.condition_manager.route_is_walkable(start, route)
        previous = start
        for waypoint in route:
            if self.wall_collision(previous, waypoint):
                return False
            previous = waypoint
        return bool(route)

    def point_has_wall_clearance(self, point: tuple[float, float], clearance_meters: float) -> bool:
        if hasattr(self.condition_manager, "point_has_wall_clearance"):
            return self.condition_manager.point_has_wall_clearance(point, clearance_meters)
        return True

    def record_movement_route_issue(
        self,
        staff_member: StaffAgent,
        destination: tuple[float, float],
        *,
        reason: str,
        semantic_destination_type: str,
        cached_route_used: bool,
        cached_route_failed_validation: bool,
        replan_attempted: bool,
        replan_succeeded: bool,
        immediate_step_collided: bool,
        route: Optional[list[tuple[float, float]]] = None,
        next_waypoint: Optional[tuple[float, float]] = None,
    ) -> None:
        active_route = list(route if route is not None else getattr(staff_member, "route_plan", []))
        entry = {
            "scenario_mode": self.scenario_mode,
            "condition_name": self.condition_spec.name,
            "seed": int(self.random_seed),
            "timestep": int(self.timestep),
            "agent_id": getattr(staff_member, "gid", None),
            "agent_name": getattr(staff_member, "name", None),
            "agent_role": getattr(staff_member, "role", None),
            "mode": getattr(staff_member, "mode", None),
            "next_mode": getattr(staff_member, "next_mode", None),
            "current_task": getattr(staff_member, "current_task_name", None),
            "position": tuple(getattr(staff_member, "position", (None, None))),
            "destination": tuple(destination),
            "route_destination": getattr(staff_member, "route_destination", None),
            "route_plan": active_route,
            "route_plan_length": len(active_route),
            "next_waypoint": next_waypoint,
            "reason": str(reason),
            "semantic_destination_type": str(semantic_destination_type),
            "cached_route_used": bool(cached_route_used),
            "cached_route_failed_validation": bool(cached_route_failed_validation),
            "replan_attempted": bool(replan_attempted),
            "replan_succeeded": bool(replan_succeeded),
            "immediate_step_collided": bool(immediate_step_collided),
            "current_position_has_wall_clearance": self.point_has_wall_clearance(
                tuple(getattr(staff_member, "position", (0.0, 0.0))),
                config.WALL_CLEARANCE_METERS,
            ),
            "destination_has_wall_clearance": self.point_has_wall_clearance(
                tuple(destination),
                config.WALL_CLEARANCE_METERS,
            ),
        }
        self.route_movement_issues.append(entry)
        self.ablation_diagnostics["route_movement_issue_count"] += 1
        self.ablation_diagnostics[f"route_movement_issue_reason_{reason}"] += 1
        self.ablation_diagnostics[f"route_movement_issue_semantic_{semantic_destination_type}"] += 1

    def _create_coordination_nurse(self) -> CoordinationNurse:
        home_zone_id = str(self.scope_roster().get("coordination_nurse_home_zone_id", config.COORDINATION_NURSE_HOME_ZONE_ID))
        home_position = self._zone_centroid(home_zone_id)
        start_position = self._sample_start_position(
            home_zone_id,
            config.STAFF_IDLE_POSITION_NOISE_METERS,
        )
        return CoordinationNurse(
            gid=1,
            role="CoordinationNurse",
            position=start_position,
            home_position=home_position,
            home_zone_id=home_zone_id,
            idle_position_radius=config.STAFF_IDLE_POSITION_NOISE_METERS,
        )

    def _create_nurses(self) -> List[Nurse]:
        roster = self.scope_roster()
        nurse_count = int(self.staff_counts().get("Nurse", config.STAFF_COUNTS["Nurse"]))
        configured_zone_ids = list(roster.get("nurse_home_zone_ids", config.NURSE_HOME_ZONE_IDS))
        if len(configured_zone_ids) < nurse_count:
            configured_zone_ids.extend(configured_zone_ids[-1:] * (nurse_count - len(configured_zone_ids)))
        assigned_zone_ids = configured_zone_ids[:nurse_count]
        nurses: List[Nurse] = []
        coverage = self.nurse_bed_coverage()

        for nurse_index, zone_id in enumerate(assigned_zone_ids):
            home_position = self.condition_manager.station_attractor(zone_id)
            start_position = self._sample_start_position(
                zone_id,
                config.HOME_POSITION_WANDER_RADIUS,
                center_point=home_position,
            )
            nurses.append(
                Nurse(
                    gid=10 + nurse_index,
                    position=start_position,
                    home_position=home_position,
                    home_zone_id=zone_id,
                    covered_bed_indices=list(coverage.get(nurse_index, [])),
                )
            )

        return nurses

    def _create_doctors(self) -> List[Doctor]:
        home_zone_id = str(self.scope_roster().get("doctor_home_zone_id", config.DOCTOR_HOME_ZONE_ID))
        if home_zone_id == "OFFSPA":
            home_position = tuple(config.OFFSPA_DOCTOR_HOME_POINT)
            start_center = home_position
        else:
            home_position = self._zone_centroid(home_zone_id)
            start_center = None
        doctors = [
            Doctor(
                gid=20 + doctor_index,
                position=self._sample_start_position(
                    home_zone_id,
                    config.STAFF_IDLE_POSITION_NOISE_METERS,
                    center_point=start_center,
                ),
                home_position=home_position,
                home_zone_id=home_zone_id,
            )
            for doctor_index in range(int(self.staff_counts().get("Doctor", config.STAFF_COUNTS["Doctor"])))
        ]
        if self.enable_senior_doctor_oversight:
            senior_home_zone_id = config.SENIOR_DOCTOR_HOME_ZONE_ID
            senior_home_position = self._zone_centroid(senior_home_zone_id)
            senior_doctor = Doctor(
                gid=config.SENIOR_DOCTOR_GID,
                position=self._sample_start_position(
                    senior_home_zone_id,
                    config.STAFF_IDLE_POSITION_NOISE_METERS,
                ),
                home_position=senior_home_position,
                home_zone_id=senior_home_zone_id,
            )
            senior_doctor.senior_oversight_only = True
            doctors.append(senior_doctor)
        return doctors

    def _sample_patient_profile(self, force_high_acuity: bool = False) -> dict:
        if force_high_acuity:
            esi_level = self.arrival_random.choice([1, 2])
        else:
            arrival_weights = self._scenario_esi_arrival_weights()
            levels = list(arrival_weights)
            weights = [float(arrival_weights[level]) for level in levels]
            esi_level = self.arrival_random.choices(
                levels, weights=weights, k=1
            )[0]
        archetypes = config.PATIENT_ARCHETYPES_BY_ESI.get(esi_level, config.PATIENT_ARCHETYPES_BY_ESI[3])
        archetype = dict(self.arrival_random.choice(archetypes))
        return {
            "esi_level": esi_level,
            "complaint_type": archetype["complaint"],
            "urgency": archetype["urgency"],
            "expected_resources": list(archetype["expected_resources"]),
            "likely_pathway": archetype["likely_pathway"],
            "required_staff_roles": list(archetype["required_staff_roles"]),
            "high_acuity_escalation": bool(archetype["high_acuity_escalation"]),
        }

    def _create_patient(self, force_high_acuity: bool = False) -> Patient:
        profile = self._sample_patient_profile(force_high_acuity=force_high_acuity)
        return Patient(
            gid=self.next_patient_id,
            position=config.PATIENT_ENTRY_POINT,
            spawned_at=self.timestep,
            **profile,
        )

    def _seed_initial_patients(self) -> None:
        for _ in range(max(config.INITIAL_PATIENT_COUNT, 0)):
            patient = self._create_patient()
            self.next_patient_id += 1
            self.active_patients.append(patient)
            self.patient_by_id[patient.gid] = patient
            self.log_workflow_event(
                agent=None,
                patient=patient,
                event_type="patient_arrival",
                task_name=patient.current_task_name,
                old_state=None,
                new_state="waiting_for_placement",
                position=patient.position,
                notes="Initial patient seeded after upstream admin/triage.",
            )

    def force_high_acuity_patient(self) -> Patient:
        """Inject one upstream-triaged ESI 1/2 patient for high-acuity smoke checks."""

        patient = self._create_patient(force_high_acuity=True)
        self.next_patient_id += 1
        self.active_patients.append(patient)
        self.patient_by_id[patient.gid] = patient
        self.log_workflow_event(
            agent=None,
            patient=patient,
            event_type="patient_arrival",
            task_name=patient.current_task_name,
            old_state=None,
            new_state="waiting_for_placement",
            position=patient.position,
            notes="Forced high-acuity smoke patient entered after upstream triage.",
        )
        return patient

    @property
    def all_agents(self) -> List[object]:
        return [*self.staff_agents, *self.active_patients]

    def sample_zone_position(
        self,
        zone_id: str,
        radius_meters: float,
        avoid_point: Optional[tuple[float, float]] = None,
        center_point: Optional[tuple[float, float]] = None,
    ) -> tuple[float, float]:
        centroid = self.condition_manager.zone_centroid(zone_id)
        sampling_center = centroid if center_point is None else center_point
        visibility_origin = centroid if center_point is None else center_point

        for _ in range(256):
            offset_x = self.random.uniform(-radius_meters, radius_meters)
            offset_y = self.random.uniform(-radius_meters, radius_meters)
            if (offset_x * offset_x) + (offset_y * offset_y) > (radius_meters * radius_meters):
                continue

            candidate = (sampling_center[0] + offset_x, sampling_center[1] + offset_y)
            if not self.condition_manager.point_in_zone(candidate[0], candidate[1], zone_id):
                continue
            if not self.has_line_of_sight(visibility_origin, candidate):
                continue
            if not self.point_has_wall_clearance(candidate, config.WALL_CLEARANCE_METERS):
                continue
            if (
                avoid_point is not None
                and math.dist(candidate, avoid_point) <= config.ARRIVAL_TOLERANCE_METERS
            ):
                continue
            return candidate

        # Rejection sampling can fail in thin polygons. Falling back to the
        # centroid keeps the agent in a valid, line-of-sight position without
        # touching environment.py.
        if (
            self.point_has_wall_clearance(sampling_center, config.WALL_CLEARANCE_METERS)
            and (
                avoid_point is None
                or math.dist(sampling_center, avoid_point) > config.ARRIVAL_TOLERANCE_METERS
            )
        ):
            return sampling_center

        return (sampling_center[0] + (radius_meters * 0.25), sampling_center[1])

    def sample_station_position(
        self,
        zone_id: str,
        avoid_point: Optional[tuple[float, float]] = None,
    ) -> tuple[float, float]:
        if zone_id != "COCPIT":
            return self.sample_zone_position(
                zone_id=zone_id,
                radius_meters=config.STAFF_IDLE_POSITION_NOISE_METERS,
                avoid_point=avoid_point,
                center_point=self.condition_manager.station_attractor(zone_id),
            )

        regions = (
            ("region_1", config.COCPIT_WORKSTATION_REGION_1),
            ("region_2", config.COCPIT_WORKSTATION_REGION_2),
        )
        region_order = list(regions)
        self.random.shuffle(region_order)
        jitter = float(config.COCPIT_WORKSTATION_BAND_Y_JITTER_METERS)
        for region_name, region in region_order:
            for _ in range(128):
                candidate = (
                    self.random.uniform(float(region["x_min"]), float(region["x_max"])),
                    float(region["y"]) + self.random.uniform(-jitter, jitter),
                )
                if not self.condition_manager.point_in_zone(candidate[0], candidate[1], zone_id):
                    continue
                if not self.point_has_wall_clearance(candidate, config.WALL_CLEARANCE_METERS):
                    continue
                if avoid_point is not None and math.dist(candidate, avoid_point) <= config.ARRIVAL_TOLERANCE_METERS:
                    continue
                self.ablation_diagnostics[f"cocpit_workstation_{region_name}_assignment_count"] += 1
                return candidate

        fallback_points = (
            ("region_1", ((float(config.COCPIT_WORKSTATION_REGION_1["x_min"]) + float(config.COCPIT_WORKSTATION_REGION_1["x_max"])) / 2.0, float(config.COCPIT_WORKSTATION_REGION_1["y"]))),
            ("region_2", ((float(config.COCPIT_WORKSTATION_REGION_2["x_min"]) + float(config.COCPIT_WORKSTATION_REGION_2["x_max"])) / 2.0, float(config.COCPIT_WORKSTATION_REGION_2["y"]))),
        )
        for region_name, point in fallback_points:
            if self.condition_manager.point_in_zone(point[0], point[1], zone_id):
                self.ablation_diagnostics[f"cocpit_workstation_{region_name}_assignment_count"] += 1
                return point
        return self.condition_manager.zone_centroid(zone_id)

    def find_available_bed_index(self, patient: Optional[Patient] = None) -> Optional[int]:
        occupied_bed_indices = {
            patient.bed_index
            for patient in self.active_patients
            if patient.bed_index is not None and not patient.discharged
        }

        bed_count = len(self.patient_care_positions())
        reserved_beds = set(getattr(config, "HIGH_ACUITY_RESERVED_BED_INDICES", set()))
        if patient is not None and getattr(patient, "esi_level", 5) in {1, 2}:
            for bed_index in sorted(reserved_beds):
                if bed_index < bed_count and bed_index not in occupied_bed_indices:
                    return bed_index
        for offset in range(bed_count):
            bed_index = (self.next_bed_assignment_cursor + offset) % bed_count
            if bed_index in reserved_beds:
                continue
            if bed_index not in occupied_bed_indices:
                return bed_index
        return None

    def assign_bed_to_patient(self, patient: Patient) -> Optional[int]:
        available_bed_index = self.find_available_bed_index(patient)
        if available_bed_index is None:
            return None

        patient.bed_index = available_bed_index
        patient.assigned_care_position = self.patient_care_positions()[available_bed_index]
        patient.assigned_nurse_index = self._nurse_index_for_bed(available_bed_index)
        self.bed_assignment_history.add(available_bed_index)
        self.next_bed_assignment_cursor = (available_bed_index + 1) % len(self.patient_care_positions())
        self.log_workflow_event(
            agent=self.coordination_nurse,
            patient=patient,
            event_type="bed_assigned",
            task_name=config.PLACEMENT_TASK_NAME,
            old_state="waiting_for_bed",
            new_state=f"bed_{available_bed_index + 1}_assigned",
            position=patient.position,
            notes=(
                "Internal bed assignment from upstream-triaged arrival into care area; "
                f"patient remains at entry until escorted to bed {available_bed_index + 1}."
            ),
        )
        return available_bed_index

    def record_esi_deprioritization(
        self,
        selected_patient: Optional[Patient],
        candidate_patients: Iterable[Patient],
        context: str,
    ) -> None:
        if selected_patient is None:
            return
        selected_esi = int(getattr(selected_patient, "esi_level", 5))
        for patient in candidate_patients:
            esi_level = int(getattr(patient, "esi_level", 5))
            if patient.gid == selected_patient.gid or esi_level <= selected_esi:
                continue
            self.ablation_diagnostics[f"low_acuity_deprioritized_ESI_{esi_level}"] += 1
            self.ablation_diagnostics[f"low_acuity_deprioritized_{context}_ESI_{esi_level}"] += 1

    def _nurse_index_for_bed(self, bed_index: int) -> Optional[int]:
        for nurse_index, bed_indices in self.nurse_bed_coverage().items():
            if bed_index in bed_indices:
                return nurse_index
        return None

    def find_free_doctor(self) -> Optional[Doctor]:
        for doctor in self.doctors:
            if (
                doctor.is_available()
                and not getattr(doctor, "senior_oversight_only", False)
                and doctor.handoff_patient_id is None
                and doctor.gid not in self.reserved_doctor_ids_this_step
            ):
                return doctor
        return None

    def spawn_patient(self) -> Optional[Patient]:
        force_high_acuity = (
            config.ENABLE_HIGH_ACUITY_EVENTS
            and self.arrival_random.random()
            < self._scenario_high_acuity_probability_per_second()
        )
        if (
            not force_high_acuity
            and self.arrival_random.random()
            >= self._scenario_arrival_rate_per_second()
        ):
            return None

        patient = self._create_patient(force_high_acuity=force_high_acuity)
        if patient.esi_level not in config.ESI_ENTRY_LEVELS:
            return None
        self.next_patient_id += 1
        if getattr(patient, "esi_level", 5) in {1, 2}:
            self.ablation_diagnostics["high_acuity_escalation_count"] += 1
            if force_high_acuity:
                self.ablation_diagnostics["high_acuity_event_arrival_count"] += 1
                self.ablation_diagnostics["high_acuity_shock_event_count"] += 1
                self.ablation_diagnostics[f"high_acuity_shock_event_ESI_{patient.esi_level}"] += 1
        if self._active_system_load() >= config.MAX_ACTIVE_SYSTEM_LOAD:
            if patient.esi_level in {1, 2}:
                self.active_patients.append(patient)
                self.patient_by_id[patient.gid] = patient
                self.ablation_diagnostics["urgent_waiting_esi1_esi2_count"] += 1
                self.ablation_diagnostics[f"accepted_waiting_ESI_{patient.esi_level}"] += 1
                self.log_workflow_event(
                    agent=None,
                    patient=patient,
                    event_type="patient_arrival",
                    task_name=patient.current_task_name,
                    old_state=None,
                    new_state="urgent_waiting_for_placement",
                    position=patient.position,
                    notes=(
                        "ESI 1/2 patient accepted into modeled care stream as urgent waiting "
                        "overflow because all modeled beds/load slots were occupied."
                    ),
                )
                return patient
            self.deferred_arrivals += 1
            self.deferred_arrivals_by_esi[f"ESI {patient.esi_level}"] += 1
            self.ablation_diagnostics[f"rejected_external_ESI_{patient.esi_level}"] += 1
            self.log_workflow_event(
                agent=None,
                patient=patient,
                event_type="deferred_arrival",
                task_name=patient.current_task_name,
                old_state="upstream_triage_complete",
                new_state="deferred_outside_modeled_care_area",
                position=patient.position,
                notes=(
                    "Arrival deferred outside the bounded modeled care area "
                    f"because active load reached {config.MAX_ACTIVE_SYSTEM_LOAD}."
                ),
            )
            return None
        self.active_patients.append(patient)
        self.patient_by_id[patient.gid] = patient
        self.log_workflow_event(
            agent=None,
            patient=patient,
            event_type="patient_arrival",
            task_name=patient.current_task_name,
            old_state=None,
            new_state="waiting_for_placement",
            position=patient.position,
            notes=(
                "Patient entered modeled care area after upstream admin/triage"
                + (" with high-acuity event flag." if force_high_acuity else ".")
            ),
        )
        return patient

    def _active_system_load(self) -> int:
        occupied_beds = {
            patient.bed_index
            for patient in self.active_patients
            if patient.bed_index is not None and not patient.discharged
        }
        waiting_patients = [
            patient
            for patient in self.active_patients
            if patient.bed_index is None and not patient.discharged
        ]
        return len(occupied_beds) + len(waiting_patients)

    def ordinary_bed_indices(self) -> set[int]:
        reserved = set(getattr(config, "HIGH_ACUITY_RESERVED_BED_INDICES", set()))
        return {index for index in range(len(self.patient_care_positions())) if index not in reserved}

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        sorted_values = sorted(float(value) for value in values)
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = (len(sorted_values) - 1) * max(0.0, min(float(percentile), 100.0)) / 100.0
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return sorted_values[lower]
        fraction = rank - lower
        return sorted_values[lower] + ((sorted_values[upper] - sorted_values[lower]) * fraction)

    def _patient_has_first_nurse(self, patient: Patient) -> bool:
        return patient.gid in self.first_nurse_seen_patient_ids

    def _patient_has_first_doctor(self, patient: Patient) -> bool:
        return patient.gid in self.first_doctor_seen_patient_ids

    def ed_pressure_snapshot(self) -> dict[str, object]:
        ordinary_beds = self.ordinary_bed_indices()
        occupied_beds = {
            patient.bed_index
            for patient in self.active_patients
            if patient.bed_index is not None and not patient.discharged
        }
        ordinary_occupied = len(ordinary_beds & occupied_beds)
        open_ordinary_beds = max(len(ordinary_beds) - ordinary_occupied, 0)
        waiting_patients = [
            patient for patient in self.active_patients
            if patient.bed_index is None and not patient.discharged
        ]
        active_high_acuity = [
            patient for patient in self.active_patients
            if not patient.discharged and int(getattr(patient, "esi_level", 5)) in {1, 2}
        ]
        waiting_without_first_nurse = [
            patient for patient in self.active_patients
            if not patient.discharged and not self._patient_has_first_nurse(patient)
        ]
        waiting_without_first_doctor = [
            patient for patient in self.active_patients
            if not patient.discharged and not self._patient_has_first_doctor(patient)
        ]
        oldest_waiting_seconds = max(
            (self.timestep - patient.spawned_at for patient in waiting_patients),
            default=0,
        )
        waiting_count = len(waiting_patients)
        ordinary_capacity = max(len(ordinary_beds), 1)
        pressure_index = min(
            2.0,
            (
                0.34 * (ordinary_occupied / ordinary_capacity)
                + 0.25 * min(waiting_count / 3.0, 1.5)
                + 0.15 * min(oldest_waiting_seconds / 3600.0, 1.5)
                + 0.16 * min(len(active_high_acuity) / 2.0, 1.5)
                + 0.10 * min(len(waiting_without_first_doctor) / 3.0, 1.5)
            ),
        )
        return {
            "pressure_index": float(pressure_index),
            "waiting_count": int(waiting_count),
            "waiting_by_esi": dict(Counter(self._esi_label(patient.esi_level) for patient in waiting_patients)),
            "ordinary_bed_occupancy": int(ordinary_occupied),
            "ordinary_bed_capacity": int(len(ordinary_beds)),
            "open_ordinary_beds": int(open_ordinary_beds),
            "all_ordinary_beds_full": bool(open_ordinary_beds == 0),
            "oldest_waiting_seconds": int(oldest_waiting_seconds),
            "active_high_acuity_patients": int(len(active_high_acuity)),
            "patients_without_first_nurse": int(len(waiting_without_first_nurse)),
            "patients_without_first_doctor": int(len(waiting_without_first_doctor)),
        }

    def current_ed_pressure_index(self) -> float:
        if self.ed_pressure_index_samples:
            return float(self.ed_pressure_index_samples[-1])
        return float(self.ed_pressure_snapshot()["pressure_index"])

    def pressure_action_threshold(self) -> float:
        return float(self.scenario_definition.get("pressure_action_threshold", 0.65))

    def _high_acuity_preemption_patient_candidates(self) -> list[Patient]:
        candidates = [
            patient
            for patient in self.active_patients
            if not patient.discharged
            and getattr(patient, "bed_position", None) is not None
            and getattr(patient, "claiming_staff_id", None) is None
            and int(getattr(patient, "esi_level", 5)) in {1, 2}
            and (
                int(getattr(patient, "esi_level", 5)) == 1
                or bool(getattr(patient, "high_acuity_escalation", False))
            )
            and patient.task_index in (
                set(config.ROLE_PERMISSIONS.get("Nurse", []))
                | set(config.ROLE_PERMISSIONS.get("Doctor", []))
            )
        ]
        candidates.sort(key=lambda patient: (getattr(patient, "esi_level", 5), patient.spawned_at))
        return candidates

    def _clinically_preemptible_by_high_acuity(self, agent, patient: Patient) -> tuple[str, str]:
        if getattr(agent, "senior_oversight_only", False):
            return "not_preemptible", "senior_doctor_unchanged"
        if agent.role not in {"Nurse", "Doctor"}:
            return "not_preemptible", "role_not_targeted"
        if patient.task_index not in config.ROLE_PERMISSIONS.get(agent.role, []):
            return "not_preemptible", "role_cannot_satisfy_task"
        if getattr(agent, "target_patient_id", None) is not None:
            current_patient = self.patient_by_id.get(getattr(agent, "target_patient_id", None))
            if current_patient is not None and int(getattr(current_patient, "esi_level", 5)) in {1, 2}:
                return "not_preemptible", "already_serving_high_acuity"
        if agent.is_task_busy():
            current_patient = self.patient_by_id.get(getattr(agent, "target_patient_id", None))
            if current_patient is not None and int(getattr(current_patient, "esi_level", 5)) in {3, 4, 5}:
                return "conditional_deferred", "active_task_preemption_deferred"
            return "not_preemptible", "protected_active_task"
        if self._agent_has_active_staff_interaction_intent(agent):
            return "soft", "opportunistic_interaction"
        if getattr(agent, "mode", None) in {
            "idle",
            "returning_home",
            "moving_to_post_task_station_check",
            "post_task_station_check",
            "moving_to_secondary_station",
            "using_secondary_station",
            "patrolling",
            "waiting_for_doctor",
        }:
            return "soft", str(getattr(agent, "mode", "soft_state"))
        if getattr(agent, "mode", None) in {"moving_to_patient", "moving_to_doctor", "leading_doctor_to_patient", "performing_task"}:
            return "conditional_deferred", "active_or_core_workflow_preemption_deferred"
        return "not_preemptible", f"mode_{getattr(agent, 'mode', 'unknown')}"

    def _apply_high_acuity_soft_preemptions(self) -> None:
        if not bool(getattr(config, "HIGH_ACUITY_SOFT_PREEMPTION_ENABLED", True)):
            return
        if self.scenario_mode != "high_load_high_acuity":
            return
        for patient in self._high_acuity_preemption_patient_candidates():
            needed_roles = [
                role for role in ("Nurse", "Doctor")
                if patient.task_index in config.ROLE_PERMISSIONS.get(role, [])
            ]
            for role in needed_roles:
                key = (patient.gid, role)
                last_attempt = self.high_acuity_preemption_attempted_at.get(key)
                if (
                    last_attempt is not None
                    and self.timestep - last_attempt < int(config.HIGH_ACUITY_PREEMPTION_COOLDOWN_SECONDS)
                ):
                    continue
                self.high_acuity_preemption_attempted_at[key] = self.timestep
                self.ablation_diagnostics["high_acuity_preemption_attempt_count"] += 1
                candidates = [
                    agent for agent in self.staff_agents
                    if getattr(agent, "role", None) == role
                    and not getattr(agent, "senior_oversight_only", False)
                ]
                soft_candidates = []
                saw_conditional = False
                rejection_reasons: Counter = Counter()
                for agent in candidates:
                    category, reason = self._clinically_preemptible_by_high_acuity(agent, patient)
                    if category == "soft":
                        if self.patient_assignment_route_supported(agent, patient):
                            soft_candidates.append(agent)
                        else:
                            rejection_reasons["route_unavailable"] += 1
                    elif category == "conditional_deferred":
                        saw_conditional = True
                        rejection_reasons[reason] += 1
                    else:
                        rejection_reasons[reason] += 1
                if saw_conditional:
                    self.ablation_diagnostics["high_acuity_active_task_preemption_deferred_count"] += 1
                if not soft_candidates:
                    self.ablation_diagnostics["high_acuity_preemption_rejected_count"] += 1
                    reason = max(rejection_reasons, key=rejection_reasons.get, default="no_soft_preemptible_staff")
                    self.ablation_diagnostics[f"high_acuity_preemption_rejection_reason_{reason}"] += 1
                    continue
                selected = min(
                    soft_candidates,
                    key=lambda agent: (
                        math.dist(agent.position, self._patient_assignment_destination(patient)),
                        agent.gid,
                    ),
                )
                active_intent = self._active_staff_intent_for_agent(selected)
                if active_intent is not None:
                    self._fail_staff_interaction_intent(active_intent, "high_acuity_preemption")
                if hasattr(selected, "_clear_secondary_station_visit"):
                    selected._clear_secondary_station_visit()
                if hasattr(selected, "_clear_post_task_station_check"):
                    selected._clear_post_task_station_check()
                preempted_mode = str(getattr(selected, "mode", "unknown"))
                patient.claiming_staff_id = selected.gid
                selected.mode = "moving_to_patient"
                selected.next_mode = "moving_to_patient"
                selected.target_patient_id = patient.gid
                selected.next_target_patient_id = patient.gid
                selected.current_task_name = None
                selected.next_current_task_name = None
                selected.task_remaining = 0
                selected.next_task_remaining = 0
                selected.destination = self._patient_assignment_destination(patient)
                selected.next_destination = selected.destination
                selected.route_plan = []
                selected.next_route_plan = []
                selected.route_destination = None
                selected.next_route_destination = None
                if patient.current_task_name is not None:
                    selected.begin_trip(patient.current_task_name, self)
                self.ablation_diagnostics["high_acuity_preemption_success_count"] += 1
                self.ablation_diagnostics["high_acuity_soft_preemption_count"] += 1
                self.ablation_diagnostics[f"high_acuity_soft_preemption_role_{role}"] += 1
                self.ablation_diagnostics[f"high_acuity_preempted_from_state_{preempted_mode}"] += 1
                self.ablation_diagnostics[f"high_acuity_preemption_destination_task_{patient.current_task_name or 'unknown'}"] += 1
                break

    def step(self) -> None:
        self.spawn_patient()
        self.reserved_doctor_ids_this_step = set()

        for agent in self.all_agents:
            agent.begin_step()

        self._apply_station_idle_headings()
        self._apply_high_acuity_soft_preemptions()

        for agent in self.staff_agents:
            if self._agent_has_active_staff_interaction_intent(agent):
                self._act_staff_interaction_intent(agent)
                continue
            perception = agent.perceive(self)
            attention = agent.attend(perception, self)
            agent.act(attention, self)

        for patient in self.active_patients:
            patient.act(None, self)

        for agent in self.all_agents:
            agent.update()

        self._record_step_metrics()
        self._update_station_episodes()
        self._update_interaction_pairs()
        self._update_staff_interaction_intents()
        self._log_perception_staff_interactions()
        self._log_opportunistic_corridor_interactions()
        self._log_bedside_cotask_interactions()
        self._log_task_transition_updates()
        self._log_senior_doctor_oversight_interactions()
        self._finalize_completed_tasks()
        self.timestep += config.TIMESTEP_SECONDS
        if self.timestep % config.PATIENT_STATE_SWEEP_INTERVAL_SECONDS == 0:
            self._repair_state_inconsistencies()

    def diagnose_agent_states(self, after_seconds: int = 300) -> dict:
        """Run a short diagnostic window and report state progression per agent."""

        tracked_agents = list(self.staff_agents)
        tracked_agent_ids = [agent.gid for agent in tracked_agents]
        mode_transitions = {agent_id: 0 for agent_id in tracked_agent_ids}
        previous_modes = {agent.gid: getattr(agent, "mode", "unknown") for agent in tracked_agents}
        previous_positions = {agent.gid: agent.position for agent in tracked_agents}
        total_distance = {agent_id: 0.0 for agent_id in tracked_agent_ids}
        max_stalled = {agent_id: 0 for agent_id in tracked_agent_ids}

        steps = max(int(after_seconds / config.TIMESTEP_SECONDS), 0)
        for _ in range(steps):
            self.step()
            for agent in tracked_agents:
                total_distance[agent.gid] += math.dist(previous_positions[agent.gid], agent.position)
                if agent.mode != previous_modes[agent.gid]:
                    mode_transitions[agent.gid] += 1
                    previous_modes[agent.gid] = agent.mode
                previous_positions[agent.gid] = agent.position
                max_stalled[agent.gid] = max(max_stalled[agent.gid], int(getattr(agent, "stalled_seconds", 0)))

        diagnostic = {
            "timestep": self.timestep,
            "after_seconds": after_seconds,
            "agents": [],
        }
        for agent in tracked_agents:
            agent_payload = {
                "gid": agent.gid,
                "role": agent.role,
                "mode": getattr(agent, "mode", "unknown"),
                "current_task": getattr(agent, "current_task_name", None),
                "target_patient_id": getattr(agent, "target_patient_id", None),
                "position": [round(agent.position[0], 3), round(agent.position[1], 3)],
                "total_distance_moved": round(total_distance[agent.gid], 3),
                "mode_transitions": mode_transitions[agent.gid],
                "current_stalled_seconds": int(getattr(agent, "stalled_seconds", 0)),
                "max_stalled_seconds": max_stalled[agent.gid],
            }
            diagnostic["agents"].append(agent_payload)
            print(
                f"{agent.role} {agent.gid}: mode={agent_payload['mode']} "
                f"task={agent_payload['current_task']} "
                f"pos=({agent_payload['position'][0]:.3f}, {agent_payload['position'][1]:.3f}) "
                f"distance={agent_payload['total_distance_moved']:.3f}m "
                f"transitions={agent_payload['mode_transitions']} "
                f"stalled={agent_payload['current_stalled_seconds']}s "
                f"max_stalled={agent_payload['max_stalled_seconds']}s"
            )
        return diagnostic

    def _update_interaction_pairs(self) -> None:
        for left_index, left_agent in enumerate(self.staff_agents):
            for right_agent in self.staff_agents[left_index + 1:]:
                pair_key = tuple(sorted((left_agent.gid, right_agent.gid)))
                if left_agent.distance_to_agent(right_agent) > config.INTERACTION_DISTANCE_METERS + 1.5:
                    self.currently_interacting_pairs.discard(pair_key)

    def _station_episode_zone_ids(self) -> set[str]:
        return set(config.STATION_ZONE_IDS) | {"OFFSPA"}

    def _functional_station_context(self, agent) -> Optional[str]:
        mode = getattr(agent, "mode", None)
        if mode in {
            "performing_task",
            "moving_to_patient",
            "placing_patient",
            "escorting_patient",
            "leading_doctor_to_patient",
        }:
            return None
        if agent.is_task_busy():
            return None

        zone_id = self.which_zone(*agent.position)
        if zone_id in self._station_episode_zone_ids():
            return str(zone_id)

        station_modes = {
            "idle",
            "returning_home",
            "waiting_for_doctor",
            "moving_to_secondary_station",
            "using_secondary_station",
            "moving_to_post_task_station_check",
            "post_task_station_check",
            "patrolling",
        }
        if mode not in station_modes:
            return None

        station_candidates: list[tuple[float, str]] = []
        for candidate_zone in self._station_episode_zone_ids():
            if candidate_zone == "OFFSPA":
                point = tuple(config.OFFSPA_DOCTOR_HOME_POINT)
            else:
                point = self.condition_manager.station_attractor(candidate_zone)
            station_candidates.append((math.dist(agent.position, point), candidate_zone))
        distance, station_zone = min(station_candidates, key=lambda item: item[0])
        if distance <= float(config.STATION_EPISODE_RADIUS_METERS):
            return station_zone
        return None

    def _start_station_episode(self, agent, station_zone: str) -> None:
        episode_id = f"{agent.gid}:{station_zone}:{self.next_station_episode_index}"
        self.next_station_episode_index += 1
        self.active_station_episode_by_agent[agent.gid] = {
            "episode_id": episode_id,
            "station_zone": station_zone,
            "started_at": self.timestep,
            "last_seen_at": self.timestep,
        }
        self.ablation_diagnostics["station_episode_started_count"] += 1

    def _end_station_episode(self, agent, reason: str) -> None:
        if agent.gid not in self.active_station_episode_by_agent:
            return
        self.active_station_episode_by_agent.pop(agent.gid, None)
        self.ablation_diagnostics["station_episode_ended_count"] += 1
        self.ablation_diagnostics[f"station_episode_reset_reason_{reason}"] += 1

    def _update_station_episodes(self) -> None:
        grace_seconds = int(config.STATION_EPISODE_LEAVE_GRACE_SECONDS)
        for agent in self.staff_agents:
            station_zone = self._functional_station_context(agent)
            active = self.active_station_episode_by_agent.get(agent.gid)
            if station_zone is None:
                if active is None:
                    continue
                if self.timestep - int(active.get("last_seen_at", self.timestep)) > grace_seconds:
                    self._end_station_episode(agent, "left_station")
                continue
            if active is None:
                self._start_station_episode(agent, station_zone)
                continue
            if str(active.get("station_zone")) != station_zone:
                self._end_station_episode(agent, "changed_station")
                self._start_station_episode(agent, station_zone)
                continue
            active["last_seen_at"] = self.timestep

    def _station_idle_heading(self, agent, zone_id: str) -> float:
        if zone_id == "COCPIT":
            region_boundary_y = (
                float(config.COCPIT_WORKSTATION_REGION_1["y"])
                + float(config.COCPIT_WORKSTATION_REGION_2["y"])
            ) / 2.0
            if agent.position[1] >= region_boundary_y:
                return math.radians(float(config.COCPIT_WORKSTATION_REGION_1["heading_degrees"]))
            return math.radians(float(config.COCPIT_WORKSTATION_REGION_2["heading_degrees"]))
        if zone_id == "NUROPE":
            return 0.0
        if zone_id == "NURSTA":
            return math.pi / 2.0
        return getattr(agent, "heading", 0.0)

    def _heading_bucket(self, heading: float) -> str:
        normalized = (heading + (2.0 * math.pi)) % (2.0 * math.pi)
        directions = [
            ("east", 0.0),
            ("north", math.pi / 2.0),
            ("west", math.pi),
            ("south", 3.0 * math.pi / 2.0),
        ]
        return min(
            directions,
            key=lambda item: abs(((normalized - item[1] + math.pi) % (2.0 * math.pi)) - math.pi),
        )[0]

    def _apply_station_idle_headings(self) -> None:
        station_modes = {"idle", "using_secondary_station", "post_task_station_check", "awaiting_nurse"}
        for agent in self.staff_agents:
            if getattr(agent, "mode", None) not in station_modes:
                continue
            zone_id = self.which_zone(*agent.position)
            if zone_id not in self.station_interaction_zone_ids():
                continue
            heading = self._station_idle_heading(agent, str(zone_id))
            agent.heading = heading
            agent.next_heading = heading
            bucket = self._heading_bucket(heading)
            self.ablation_diagnostics[f"idle_heading_zone_{zone_id}_{bucket}"] += 1
            if zone_id == "COCPIT":
                if bucket == "north":
                    self.ablation_diagnostics["cocpit_idle_heading_north_count"] += 1
                elif bucket == "south":
                    self.ablation_diagnostics["cocpit_idle_heading_south_count"] += 1
                region = "region_1" if heading > 0.0 else "region_2"
                self.ablation_diagnostics[f"cocpit_idle_{region}_occupancy_count"] += 1

    def _encounter_visible(self, left_agent, right_agent) -> bool:
        return self._perception_gate(
            left_agent,
            right_agent,
            interaction_type="generic",
        )["accepted"]

    def _classify_perception_rejection(self, left_agent, right_agent) -> str:
        if not self.variant_settings["perception_required"]:
            return "none"
        if not self.perception._within_fov(left_agent, right_agent.position) or not self.perception._within_fov(right_agent, left_agent.position):
            return "rejected_fov_hard"
        if not self.perception._has_visibility(left_agent.position, right_agent.position):
            return "rejected_by_visibility"
        return "rejected_by_attention_capacity"

    def _shared_station_context(self, left_agent, right_agent, position: Optional[tuple[float, float]] = None) -> bool:
        midpoint = position or (
            (left_agent.position[0] + right_agent.position[0]) / 2.0,
            (left_agent.position[1] + right_agent.position[1]) / 2.0,
        )
        midpoint_zone = self.which_zone(*midpoint)
        left_zone = self.which_zone(*left_agent.position)
        right_zone = self.which_zone(*right_agent.position)
        station_or_adjacent = self.station_interaction_zone_ids() | set(config.STATION_ADJACENT_ZONE_IDS)
        return (
            midpoint_zone in station_or_adjacent
            and (left_zone in station_or_adjacent or right_zone in station_or_adjacent)
            and left_agent.distance_to_agent(right_agent) <= config.SHARED_STATION_AWARENESS_DISTANCE_METERS
        )

    def _task_driven_context(self, agent, partner, interaction_type: str, task_name: Optional[str]) -> bool:
        patient = self.patient_by_id.get(getattr(agent, "target_patient_id", None))
        high_acuity = patient is not None and getattr(patient, "esi_level", None) in {1, 2}
        task_need = bool(task_name) or interaction_type in {
            "nurse_doctor_handoff",
            "placement_handoff",
            "doctor_task",
            "nursing_task",
        }
        role_need = (
            getattr(agent, "mode", "") in {"waiting_for_doctor", "leading_doctor_to_patient", "moving_to_doctor"}
            or getattr(partner, "mode", "") in {"waiting_for_doctor", "leading_doctor_to_patient", "moving_to_doctor"}
        )
        return high_acuity or task_need or role_need

    def _near_mutual_heading(self, left_agent, right_agent) -> bool:
        left_sees = self.perception._within_fov(left_agent, right_agent.position)
        right_sees = self.perception._within_fov(right_agent, left_agent.position)
        close = left_agent.distance_to_agent(right_agent) <= config.CORRIDOR_CLOSE_PASS_DISTANCE_METERS
        return (left_sees and right_sees) or (close and (left_sees or right_sees))

    def _can_perceive_interaction_partner(
        self,
        left_agent,
        right_agent,
        interaction_type: str,
        position: Optional[tuple[float, float]] = None,
        task_name: Optional[str] = None,
    ) -> dict:
        """Universal perception gate for every validation-counted F2F event."""

        if not self.variant_settings["perception_required"]:
            return {"accepted": False, "category": "interactions_logged_without_perception_check", "reason": "perception_disabled"}

        distance = left_agent.distance_to_agent(right_agent)
        close = distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
        if getattr(right_agent, "role", None) == "Patient" or getattr(left_agent, "role", None) == "Patient":
            staff_agent = left_agent if getattr(left_agent, "role", None) != "Patient" else right_agent
            patient = right_agent if getattr(right_agent, "role", None) == "Patient" else left_agent
            patient_distance = staff_agent.distance_to_agent(patient)
            at_bedside = patient_distance <= max(config.INTERACTION_DISTANCE_METERS, 1.8)
            placement_or_task = interaction_type in {
                "placement_handoff",
                "nursing_task",
                "doctor_task",
                "task_driven_patient_facing",
            } or bool(task_name)
            line_of_sight = self.perception._has_visibility(staff_agent.position, patient.position)
            if at_bedside or (placement_or_task and line_of_sight and patient_distance <= 3.0):
                return {
                    "accepted": True,
                    "category": "accepted_patient_facing",
                    "reason": "patient_perceived_for_communicative_task",
                    "perception_checked": True,
                    "salience_score": max(0.2, 1.0 / max(patient_distance, 0.5)),
                }
            return {
                "accepted": False,
                "category": "rejected_not_perceived",
                "reason": "patient_not_perceived_for_f2f_logging",
                "perception_checked": True,
            }

        if interaction_type in {
            "bedside_cotask_update",
            "nurse_doctor_bedside_coordination",
            "nurse_nurse_bedside_assist",
            "doctor_doctor_bedside_review",
        } and self._co_task_interaction_allowed(left_agent, right_agent) is not None:
            return {
                "accepted": True,
                "category": "accepted_cotask_local_communication",
                "reason": "same_patient_close_bedside_context",
                "perception_checked": True,
            }

        if interaction_type.startswith("opportunistic") and (
            not self._staff_interruptible_for_opportunistic_perception(left_agent)
            or not self._staff_interruptible_for_opportunistic_perception(right_agent)
        ):
            return {"accepted": False, "category": "rejected_protected_state", "reason": "staff_in_protected_state", "perception_checked": True}

        has_visibility = self.perception._has_visibility(left_agent.position, right_agent.position)
        left_fov = self.perception._within_fov(left_agent, right_agent.position)
        right_fov = self.perception._within_fov(right_agent, left_agent.position)
        direct = has_visibility and left_fov and right_fov
        if direct:
            return {"accepted": True, "category": "accepted_direct_fov", "reason": "direct_mutual_fov", "perception_checked": True}
        if not has_visibility and distance > config.CORRIDOR_CLOSE_PASS_DISTANCE_METERS:
            return {"accepted": False, "category": "rejected_not_perceived", "reason": "no_line_of_sight", "perception_checked": True}

        if self._shared_station_context(left_agent, right_agent, position):
            if has_visibility or close or distance <= config.CORRIDOR_CLOSE_PASS_DISTANCE_METERS:
                return {
                    "accepted": True,
                    "category": "accepted_shared_station_awareness",
                    "reason": "same_station_or_station_adjacent",
                    "perception_checked": True,
                }

        if self._task_driven_context(left_agent, right_agent, interaction_type, task_name):
            partner_known_station = self.which_zone(*right_agent.position) in self.station_interaction_zone_ids()
            if has_visibility or partner_known_station:
                if self.random.random() < config.TASK_DRIVEN_FOV_RELAXATION_PROBABILITY:
                    return {"accepted": True, "category": "accepted_task_driven", "reason": "task_need_relaxed_fov", "perception_checked": True}
                return {"accepted": False, "category": "rejected_by_fov", "reason": "task_need_soft_fov_roll", "perception_checked": True}

        if interaction_type == "opportunistic_corridor":
            if has_visibility and self._near_mutual_heading(left_agent, right_agent):
                return {"accepted": True, "category": "accepted_corridor_opportunistic", "reason": "corridor_close_or_mutual_heading", "perception_checked": True}
            if has_visibility and distance <= config.CORRIDOR_CLOSE_PASS_DISTANCE_METERS:
                if self.random.random() < config.CORRIDOR_SOFT_FOV_RELAXATION_PROBABILITY:
                    return {"accepted": True, "category": "accepted_corridor_opportunistic", "reason": "corridor_close_soft_fov", "perception_checked": True}
                return {"accepted": False, "category": "rejected_by_fov", "reason": "corridor_soft_fov_roll", "perception_checked": True}

        if not left_fov or not right_fov:
            return {"accepted": False, "category": "rejected_by_fov", "reason": "outside_contextual_fov", "perception_checked": True}
        return {"accepted": False, "category": "rejected_salience", "reason": "attention_or_salience_gate", "perception_checked": True}

    def _perception_gate(
        self,
        left_agent,
        right_agent,
        interaction_type: str,
        position: Optional[tuple[float, float]] = None,
        task_name: Optional[str] = None,
    ) -> dict:
        return self._can_perceive_interaction_partner(
            left_agent,
            right_agent,
            interaction_type=interaction_type,
            position=position,
            task_name=task_name,
        )

    def _role_pair_key(self, left_agent, right_agent) -> str:
        return "|".join(sorted((str(left_agent.role), str(right_agent.role))))

    def should_log_patient_facing_task_interaction(
        self,
        staff_agent,
        patient,
        task_name: str,
        communicative_type: str,
        communication_probability: float,
    ) -> bool:
        """Gate validation-counted patient communication without changing workflow execution."""

        del communicative_type
        counter_key = f"{staff_agent.role}|{task_name}"
        self.patient_facing_task_attendance_counts[counter_key] += 1

        allowed_tasks = config.PATIENT_FACING_VALIDATION_TASKS.get(staff_agent.role, set())
        if task_name not in allowed_tasks:
            self.ablation_diagnostics["patient_facing_task_not_validation_semantic"] += 1
            return False

        last_timestamp = self.last_patient_facing_validation_by_patient.get(patient.gid)
        if (
            last_timestamp is not None
            and (self.timestep - last_timestamp) < config.PATIENT_FACING_VALIDATION_MIN_INTERVAL_SECONDS
            and task_name not in {config.DISPOSITION_DECISION_TASK_NAME, config.DISCHARGE_TASK_NAME}
        ):
            self.ablation_diagnostics["patient_facing_suppressed_by_patient_interval"] += 1
            self.ablation_diagnostics["patient_facing_suppressed_by_interval_count"] += 1
            return False

        if getattr(patient, "esi_level", None) in {1, 2} and last_timestamp is None:
            self.last_patient_facing_validation_by_patient[patient.gid] = self.timestep
            self.patient_facing_validation_interaction_counts[counter_key] += 1
            self.ablation_diagnostics["patient_facing_task_start_interaction_count"] += 1
            self.ablation_diagnostics["high_acuity_patient_facing_required_trace_count"] += 1
            return True

        if self.random.random() >= communication_probability:
            self.ablation_diagnostics["patient_facing_task_no_explicit_communication"] += 1
            return False

        self.last_patient_facing_validation_by_patient[patient.gid] = self.timestep
        self.patient_facing_validation_interaction_counts[counter_key] += 1
        self.ablation_diagnostics["patient_facing_task_start_interaction_count"] += 1
        return True

    def patient_facing_task_interaction_position(
        self,
        staff_agent,
        patient,
        task_name: str,
    ) -> tuple[float, float]:
        """Return the staff member's actual embodied contact position."""

        return tuple(staff_agent.position)

    def maybe_log_patient_facing_in_task_interaction(self, staff_agent, patient) -> None:
        task_name = getattr(staff_agent, "current_task_name", None)
        if not task_name:
            return
        role_probabilities = getattr(config, "PATIENT_FACING_IN_TASK_INTERACTION_PROBABILITY", {}).get(
            getattr(staff_agent, "role", ""), {}
        )
        probability = float(role_probabilities.get(task_name, 0.0))
        if probability <= 0.0:
            return
        if not self._staff_locally_communicative(staff_agent, patient):
            return
        if staff_agent.distance_to_agent(patient) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            return
        if patient.seconds_at_current_task(self.timestep) < int(config.PATIENT_FACING_IN_TASK_MIN_SECONDS):
            return
        last_timestamp = self.last_patient_facing_validation_by_patient.get(patient.gid)
        if (
            last_timestamp is not None
            and (self.timestep - last_timestamp) < config.PATIENT_FACING_VALIDATION_MIN_INTERVAL_SECONDS
            and task_name not in {config.DISPOSITION_DECISION_TASK_NAME, config.DISCHARGE_TASK_NAME}
        ):
            self.ablation_diagnostics["patient_facing_suppressed_by_interval_count"] += 1
            return
        key = (int(staff_agent.gid), int(patient.gid), str(task_name))
        if key in self.patient_facing_in_task_logged_keys or key in self.patient_facing_in_task_considered_keys:
            return
        self.patient_facing_in_task_considered_keys.add(key)
        if self.random.random() >= probability:
            self.ablation_diagnostics["patient_facing_task_no_explicit_communication"] += 1
            return

        gate_result = {
            "accepted": True,
            "category": "accepted_patient_facing",
            "reason": "locally_communicative_patient_facing_in_task",
            "perception_checked": True,
            "interaction_source": "patient_facing_in_task",
            "force_interaction_decision": True,
        }
        before_count = len(self.interaction_log)
        self.log_interaction(
            agent_1=staff_agent,
            agent_2=patient,
            position=tuple(staff_agent.position),
            interaction_type=self._workflow_interaction_type_for_role(staff_agent.role),
            task_name=task_name,
            reason_for_interaction=f"patient_facing_in_task_{task_name}",
            perception_gate_result=gate_result,
        )
        if len(self.interaction_log) > before_count:
            self.patient_facing_in_task_logged_keys.add(key)
            self.last_patient_facing_validation_by_patient[patient.gid] = self.timestep
            self.ablation_diagnostics["patient_facing_in_task_interaction_count"] += 1
            self.ablation_diagnostics[f"patient_facing_in_task_role_{staff_agent.role}"] += 1
            zone_id = self.which_zone(*staff_agent.position) or "Outside named zones"
            self.ablation_diagnostics[f"patient_facing_in_task_zone_{zone_id}"] += 1

    def _workflow_interaction_type_for_role(self, role: str) -> str:
        if role == "CoordinationNurse":
            return "placement_handoff"
        if role == "Nurse":
            return "nursing_task"
        if role == "Doctor":
            return "doctor_task"
        return "workflow_task"

    def maybe_log_escort_corridor_interaction(self, staff_agent, patient) -> None:
        if getattr(staff_agent, "role", None) != "CoordinationNurse":
            return
        if getattr(patient, "escort_corridor_interaction_logged", False):
            return
        if staff_agent.current_task_name != config.PLACEMENT_TASK_NAME:
            return
        if staff_agent.distance_to_agent(patient) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            return
        zone_id = self.which_zone(*staff_agent.position)
        if zone_id not in self.corridor_or_threshold_interaction_zone_ids():
            return
        per_tick_probability = 1.0 - (
            (1.0 - float(config.PLACEMENT_ESCORT_CORRIDOR_INTERACTION_PROBABILITY))
            ** (config.TIMESTEP_SECONDS / 120.0)
        )
        if self.random.random() >= per_tick_probability:
            return

        gate_result = {
            "accepted": True,
            "category": "accepted_patient_facing",
            "reason": "coordination_nurse_patient_escort_close_corridor",
            "perception_checked": True,
            "interaction_source": "patient_escort_corridor_update",
            "force_interaction_decision": True,
        }
        before_count = len(self.interaction_log)
        self.log_interaction(
            agent_1=staff_agent,
            agent_2=patient,
            position=tuple(staff_agent.position),
            interaction_type="placement_handoff",
            task_name=config.PLACEMENT_TASK_NAME,
            reason_for_interaction="placement_walk_patient_guidance",
            perception_gate_result=gate_result,
        )
        if len(self.interaction_log) > before_count:
            patient.escort_corridor_interaction_logged = True
            self.ablation_diagnostics["escort_patient_corridor_interaction_count"] += 1
            self.ablation_diagnostics["coordination_nurse_patient_escort_interactions"] += 1
            self.ablation_diagnostics[f"coordination_nurse_patient_interactions_zone_{zone_id}"] += 1

    def _zone_weight_key(self, zone_id: Optional[str]) -> str:
        if zone_id in config.STATION_ZONE_IDS:
            return "station"
        if zone_id is not None and str(zone_id).startswith("CORR"):
            return "corridor"
        return str(zone_id or "Outside named zones")

    def _hcw_hcw_interaction_probability(
        self,
        left_agent,
        right_agent,
        zone_id: Optional[str],
        base_probability: float,
    ) -> float:
        pair_weight = float(
            config.HCW_HCW_ROLE_PAIR_INTERACTION_WEIGHTS.get(
                self._role_pair_key(left_agent, right_agent),
                1.0,
            )
        )
        zone_weight = max(
            float(config.HCW_HCW_ZONE_INTERACTION_WEIGHTS.get(str(zone_id), 1.0)),
            float(config.HCW_HCW_ZONE_INTERACTION_WEIGHTS.get(self._zone_weight_key(zone_id), 1.0)),
        )
        return min(
            1.0,
            base_probability
            * pair_weight
            * zone_weight
            * getattr(self, "interaction_probability_multiplier", 1.0),
        )

    def _finalize_completed_tasks(self) -> None:
        discharged_patients: List[Patient] = []

        for staff_member in self.staff_agents:
            if not staff_member.task_completed_this_step:
                continue

            patient = self.patient_by_id.get(staff_member.target_patient_id)
            if patient is None:
                staff_member.clear_assignment()
                continue

            if patient.claiming_staff_id == staff_member.gid:
                patient.claiming_staff_id = None

            old_state = patient.current_task_name
            patient.complete_current_task(current_timestep=self.timestep)
            self.log_workflow_event(
                agent=staff_member,
                patient=patient,
                event_type="task_completed",
                task_name=old_state,
                old_state=old_state,
                new_state=patient.current_task_name or "discharged",
                position=patient.position,
                notes=f"{staff_member.role} completed {old_state} for patient {patient.gid}",
            )
            self._schedule_task_transition_update(staff_member, patient, old_state)
            self._maybe_start_intertask_coordination_gap(patient, old_state)
            self._schedule_senior_doctor_review(staff_member, patient, old_state)
            if (
                isinstance(staff_member, CoordinationNurse)
                and old_state == config.PLACEMENT_TASK_NAME
                and patient.assigned_nurse_index is not None
                and self.random.random() < config.PLACEMENT_NURSE_ALERT_PROBABILITY
            ):
                assigned_nurse = (
                    self.nurses[patient.assigned_nurse_index]
                    if patient.assigned_nurse_index < len(self.nurses)
                    else None
                )
                if assigned_nurse is not None:
                    self.log_interaction(
                        agent_1=staff_member,
                        agent_2=assigned_nurse,
                        position=assigned_nurse.position,
                        interaction_type="placement_nurse_alert",
                        task_name=config.PLACEMENT_TASK_NAME,
                        reason_for_interaction="coordination_nurse_alerts_assigned_nurse_patient_ready",
                    )

            if isinstance(staff_member, Doctor):
                staff_member.clear_handoff()

            if patient.discharged:
                discharged_patients.append(patient)

            if not self._maybe_start_post_task_station_check(staff_member, patient, old_state):
                staff_member.clear_assignment()

        for patient in discharged_patients:
            self.active_patients = [
                active_patient for active_patient in self.active_patients if active_patient.gid != patient.gid
            ]
            self.completed_patients.append(patient)

    def _nearest_post_task_station(self, origin: tuple[float, float]) -> Optional[tuple[str, tuple[float, float]]]:
        candidates: list[tuple[float, str, tuple[float, float]]] = []
        for zone_id in getattr(config, "POST_TASK_STATION_CHECK_STATIONS", ("COCPIT", "NURSTA", "NUROPE")):
            try:
                point = self.condition_manager.station_attractor(str(zone_id))
            except Exception:
                continue
            if point is None:
                continue
            candidates.append((math.dist(origin, point), str(zone_id), point))
        if not candidates:
            return None
        _, zone_id, point = min(candidates, key=lambda item: item[0])
        return zone_id, point

    def _maybe_start_post_task_station_check(self, staff_member: StaffAgent, patient: Patient, completed_task_name: Optional[str]) -> bool:
        if not bool(getattr(config, "POST_TASK_STATION_CHECK_ENABLED", False)):
            return False
        if not isinstance(staff_member, (Doctor, Nurse)):
            return False
        if patient.discharged:
            return False
        if completed_task_name is None:
            return False

        probability = (
            float(config.POST_TASK_STATION_CHECK_PROBABILITY_DOCTOR)
            if isinstance(staff_member, Doctor)
            else float(config.POST_TASK_STATION_CHECK_PROBABILITY_NURSE)
        )
        if probability <= 0.0 or self.random.random() >= probability:
            return False

        origin = patient.bed_position or patient.position
        station = self._nearest_post_task_station(origin)
        if station is None:
            self.ablation_diagnostics["station_check_route_failures"] += 1
            return False
        station_zone, station_point = station
        if station_zone in self.station_interaction_zone_ids():
            station_point = self.sample_station_position(station_zone, avoid_point=staff_member.position)

        lower_bound = int(config.POST_TASK_STATION_CHECK_DURATION_SECONDS_MIN)
        upper_bound = int(config.POST_TASK_STATION_CHECK_DURATION_SECONDS_MAX)
        if upper_bound < lower_bound:
            upper_bound = lower_bound
        dwell_seconds = self.random.randint(lower_bound, upper_bound)

        staff_member.clear_assignment()
        staff_member.start_post_task_station_check(
            self,
            zone_id=station_zone,
            destination=station_point,
            duration_seconds=dwell_seconds,
        )
        self.ablation_diagnostics["post_task_station_check_count"] += 1
        self.ablation_diagnostics[f"post_task_station_check_role_{staff_member.role}"] += 1
        self.ablation_diagnostics[f"post_task_station_check_station_{station_zone}"] += 1
        self.log_workflow_event(
            agent=staff_member,
            patient=patient,
            event_type="post_task_station_check_started",
            task_name=completed_task_name,
            old_state=completed_task_name,
            new_state=f"post_task_station_check_{station_zone}",
            position=staff_member.position,
            notes=(
                f"{staff_member.role} routed to nearby {station_zone} after bedside task "
                f"for {dwell_seconds}s station/computer check."
            ),
        )
        return True

    def _maybe_start_intertask_coordination_gap(self, patient: Patient, completed_task_name: str) -> None:
        if patient.discharged:
            return
        probability = float(config.INTERTASK_COORDINATION_GAP_PROBABILITY.get(completed_task_name, 0.0))
        if probability <= 0.0 or self.random.random() >= probability:
            return
        lower_bound, upper_bound = config.INTERTASK_COORDINATION_GAP_SECONDS.get(completed_task_name, (30, 90))
        hold_seconds = self.random.randint(int(lower_bound), int(upper_bound))
        patient.coordination_hold_until = max(patient.coordination_hold_until, self.timestep + hold_seconds)
        patient.coordination_hold_after_task = completed_task_name
        patient.coordination_hold_reason = "awaiting_intertask_coordination"
        self.log_workflow_event(
            agent=None,
            patient=patient,
            event_type="coordination_gap_started",
            task_name=patient.current_task_name,
            old_state=completed_task_name,
            new_state=patient.current_task_name,
            position=patient.position,
            notes=(
                f"Patient {patient.gid} waits {hold_seconds}s after {completed_task_name} "
                "for status/update coordination before next task."
            ),
        )

    def _schedule_task_transition_update(self, staff_member, patient, task_name: str) -> None:
        probability = float(
            config.TASK_TRANSITION_UPDATE_PROBABILITY.get(staff_member.role, {}).get(task_name, 0.0)
        )
        if probability <= 0.0:
            return
        if getattr(patient, "esi_level", 5) in {1, 2}:
            probability = min(1.0, probability + 0.08)
        if self.random.random() >= probability:
            return
        if staff_member.role == "Doctor":
            preferred_roles = ["Doctor"]
        elif staff_member.role == "Nurse":
            preferred_roles = ["Nurse", "CoordinationNurse"]
        else:
            return
        episode_context_key = (
            int(staff_member.gid),
            int(patient.gid),
            str(task_name),
            "post_task_transition_update",
            tuple(preferred_roles),
        )
        recent_timestamp = self.recent_task_update_episode_by_context.get(episode_context_key)
        if (
            recent_timestamp is not None
            and (self.timestep - recent_timestamp) < config.TASK_TRANSITION_UPDATE_EXPIRY_SECONDS
        ):
            self.ablation_diagnostics["task_update_episode_suppressed_repeat_count"] += 1
            self.ablation_diagnostics["task_transition_update_episode_suppressed_count"] += 1
            return
        if any(update.get("episode_context_key") == episode_context_key for update in self.pending_task_transition_updates):
            self.ablation_diagnostics["task_update_episode_suppressed_repeat_count"] += 1
            self.ablation_diagnostics["task_transition_update_episode_suppressed_count"] += 1
            return
        self.pending_task_transition_updates.append(
            {
                "initiator_id": staff_member.gid,
                "patient_id": patient.gid,
                "task_name": task_name,
                "episode_key": f"{staff_member.gid}:{patient.gid}:{task_name}:{self.timestep}",
                "episode_context_key": episode_context_key,
                "preferred_roles": preferred_roles,
                "created_at": self.timestep,
                "expires_at": self.timestep + config.TASK_TRANSITION_UPDATE_EXPIRY_SECONDS,
            }
        )
        self.recent_task_update_episode_by_context[episode_context_key] = self.timestep
        self.ablation_diagnostics["task_update_episode_started_count"] += 1

    def _task_transition_update_episode_key(self, update: Mapping[str, object], partner) -> tuple[object, ...]:
        """A transition update is one communication need, not a repeatable proximity event."""

        return (
            update.get("episode_key"),
            int(update.get("initiator_id", -1)),
            int(update.get("patient_id", -1)),
            str(update.get("task_name", "")),
            getattr(partner, "role", None),
        )

    def _schedule_senior_doctor_review(self, staff_member, patient, task_name: str) -> None:
        if not self.enable_senior_doctor_oversight:
            return
        if not isinstance(staff_member, Doctor) or getattr(staff_member, "senior_oversight_only", False):
            return
        if patient.discharged:
            return
        if not self._senior_review_eligible(patient, task_name):
            return
        max_reviews = (
            config.SENIOR_DOCTOR_MAX_COMPLEX_REVIEWS_PER_PATIENT
            if self._senior_review_complex_case(patient, task_name)
            else config.SENIOR_DOCTOR_MAX_REVIEWS_PER_PATIENT
        )
        if getattr(patient, "senior_doctor_review_count", 0) >= max_reviews:
            return
        probability = float(config.SENIOR_DOCTOR_REVIEW_PROBABILITY.get(task_name, 0.0))
        if probability <= 0.0:
            return
        if getattr(patient, "esi_level", 5) in {1, 2}:
            probability = min(1.0, probability + 0.08)
        if self.random.random() >= probability:
            return
        review_zone = self._senior_review_zone(patient, task_name)
        hold_low, hold_high = config.SENIOR_DOCTOR_REVIEW_HOLD_SECONDS
        hold_seconds = self.random.randint(int(hold_low), int(hold_high))
        patient.coordination_hold_until = max(patient.coordination_hold_until, self.timestep + hold_seconds)
        patient.coordination_hold_after_task = task_name
        patient.coordination_hold_reason = "awaiting_senior_review"
        patient.senior_doctor_review_count += 1
        self.pending_senior_doctor_reviews.append(
            {
                "initiator_id": staff_member.gid,
                "patient_id": patient.gid,
                "task_name": task_name,
                "review_zone": review_zone,
                "esi_level": getattr(patient, "esi_level", None),
                "created_at": self.timestep,
                "expires_at": self.timestep + config.SENIOR_DOCTOR_REVIEW_EXPIRY_SECONDS,
            }
        )
        self.log_workflow_event(
            agent=staff_member,
            patient=patient,
            event_type="senior_review_gap_started",
            task_name=patient.current_task_name,
            old_state=task_name,
            new_state=patient.current_task_name,
            position=patient.position,
            notes=(
                f"Patient {patient.gid} waits up to {hold_seconds}s for SeniorDoctor "
                f"review after {task_name}."
            ),
        )

    def _senior_review_complex_case(self, patient: Patient, task_name: str) -> bool:
        esi_level = getattr(patient, "esi_level", 5)
        return (
            esi_level in {1, 2}
            or (
                esi_level == 3
                and task_name in {
                    config.DIAGNOSTICS_TASK_NAME,
                    config.REASSESSMENT_TASK_NAME,
                    config.DISPOSITION_DECISION_TASK_NAME,
                }
            )
        )

    def _senior_review_eligible(self, patient: Patient, task_name: str) -> bool:
        esi_level = getattr(patient, "esi_level", 5)
        if esi_level in {1, 2} and task_name in {
            config.MEDICAL_EVALUATION_TASK_NAME,
            config.DIAGNOSTICS_TASK_NAME,
            config.REASSESSMENT_TASK_NAME,
            config.DISPOSITION_DECISION_TASK_NAME,
        }:
            return True
        if esi_level == 3 and task_name in {
            config.DIAGNOSTICS_TASK_NAME,
            config.REASSESSMENT_TASK_NAME,
            config.DISPOSITION_DECISION_TASK_NAME,
        }:
            return True
        return False

    def _senior_review_zone(self, patient: Patient, task_name: str) -> str:
        if (
            getattr(patient, "esi_level", 5) in {1, 2}
            and task_name in {config.DIAGNOSTICS_TASK_NAME, config.DISPOSITION_DECISION_TASK_NAME}
            and self.random.random() < config.SENIOR_DOCTOR_OFFSPA_REVIEW_PROBABILITY
        ):
            return "OFFSPA"
        return "COCPIT"

    def maybe_log_senior_doctor_room_assist(self, staff_member, patient: Patient, task_name: str) -> None:
        if not self.enable_senior_doctor_oversight:
            return
        if not isinstance(staff_member, Doctor) or getattr(staff_member, "senior_oversight_only", False):
            return
        if not self._senior_room_assist_eligible(patient, task_name):
            return
        if getattr(patient, "senior_doctor_room_assist_count", 0) >= 1:
            return
        probability = float(config.SENIOR_DOCTOR_ROOM_ASSIST_PROBABILITY.get(task_name, 0.0))
        if probability <= 0.0 or self.random.random() >= probability:
            return
        senior_doctors = [
            doctor for doctor in self.doctors
            if getattr(doctor, "senior_oversight_only", False)
            and not doctor.is_task_busy()
        ]
        if not senior_doctors:
            return
        senior_doctor = senior_doctors[0]
        pair_key = tuple(sorted((staff_member.gid, senior_doctor.gid)))
        last_timestamp = self.last_senior_doctor_review_by_pair.get(pair_key)
        if (
            last_timestamp is not None
            and (self.timestep - last_timestamp) < config.SENIOR_DOCTOR_ROOM_ASSIST_COOLDOWN_SECONDS
        ):
            return
        position = self._embodied_contact_position(staff_member, senior_doctor)
        before_count = len(self.interaction_log)
        self.log_interaction(
            agent_1=staff_member,
            agent_2=senior_doctor,
            position=position,
            interaction_type="senior_doctor_room_assist",
            task_name=task_name,
            reason_for_interaction="acuity_or_complexity_gated_senior_doctor_room_assist",
        )
        if len(self.interaction_log) > before_count:
            self.currently_interacting_pairs.add(pair_key)
            self.last_senior_doctor_review_by_pair[pair_key] = self.timestep
            patient.senior_doctor_room_assist_count += 1

    def _senior_room_assist_eligible(self, patient: Patient, task_name: str) -> bool:
        esi_level = getattr(patient, "esi_level", 5)
        if task_name not in config.SENIOR_DOCTOR_ROOM_ASSIST_PROBABILITY:
            return False
        if esi_level in {1, 2}:
            return True
        return esi_level == 3 and task_name in {
            config.DIAGNOSTICS_TASK_NAME,
            config.TREATMENT_TASK_NAME,
            config.REASSESSMENT_TASK_NAME,
            config.DISPOSITION_DECISION_TASK_NAME,
        }

    def _doctor_assigned_to_patient(self, patient: Patient) -> bool:
        if patient.claiming_staff_id in self.doctor_by_id:
            doctor = self.doctor_by_id[patient.claiming_staff_id]
            if not getattr(doctor, "senior_oversight_only", False) and doctor.target_patient_id == patient.gid:
                return True

        return any(
            not getattr(doctor, "senior_oversight_only", False)
            and (doctor.target_patient_id == patient.gid or doctor.handoff_patient_id == patient.gid)
            for doctor in self.doctors
        )

    def _doctor_wait_metrics_by_task(
        self,
        task_started_events: List[Mapping[str, object]],
        task_completed_events: List[Mapping[str, object]],
    ) -> Dict[str, object]:
        ready_at_by_patient_task: dict[tuple[int, str], int] = {}
        for event in task_completed_events:
            patient_id = event.get("patient_id")
            next_task = event.get("new_state")
            if patient_id is None or next_task is None or next_task == "discharged":
                continue
            ready_at_by_patient_task[(int(patient_id), str(next_task))] = int(event.get("timestamp", 0))

        waits_by_task: dict[str, list[int]] = defaultdict(list)
        for event in task_started_events:
            if event.get("agent_role") != "Doctor":
                continue
            task_name = str(event.get("task_name"))
            if config.TASK_INDEX_BY_NAME.get(task_name) not in config.ROLE_PERMISSIONS["Doctor"]:
                continue
            patient_id = event.get("patient_id")
            if patient_id is None:
                continue
            ready_at = ready_at_by_patient_task.get((int(patient_id), task_name))
            if ready_at is None:
                continue
            waits_by_task[task_name].append(max(int(event.get("timestamp", 0)) - ready_at, 0))

        backlog_waits_by_task: dict[str, list[int]] = defaultdict(list)
        for patient in self.active_patients:
            if (
                patient.discharged
                or patient.task_index not in config.ROLE_PERMISSIONS["Doctor"]
                or self._doctor_assigned_to_patient(patient)
            ):
                continue
            task_name = patient.current_task_name or "unknown"
            backlog_waits_by_task[task_name].append(patient.seconds_at_current_task(self.timestep))

        def summarize(values: list[int]) -> dict[str, float]:
            if not values:
                return {"count": 0, "mean": 0.0, "median": 0.0, "q75": 0.0, "q90": 0.0}
            ordered = sorted(values)

            def percentile(fraction: float) -> float:
                index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
                return float(ordered[index])

            return {
                "count": len(ordered),
                "mean": float(mean(ordered)),
                "median": percentile(0.50),
                "q75": percentile(0.75),
                "q90": percentile(0.90),
            }

        return {
            "completed_doctor_task_wait_seconds_by_task": {
                task_name: summarize(values)
                for task_name, values in waits_by_task.items()
            },
            "current_doctor_backlog_wait_seconds_by_task": {
                task_name: summarize(values)
                for task_name, values in backlog_waits_by_task.items()
            },
        }

    def _doctor_utilization_summary(self) -> Dict[str, float]:
        ordinary_doctor_ids = [
            doctor.gid for doctor in self.doctors
            if not getattr(doctor, "senior_oversight_only", False)
        ]
        total_seconds = 0
        task_seconds = 0
        walking_seconds = 0
        station_or_available_seconds = 0
        walking_modes = {
            "moving_to_patient",
            "moving_to_doctor",
            "leading_doctor_to_patient",
            "returning_home",
            "moving_to_post_task_station_check",
            "patrolling",
            "awaiting_nurse",
        }
        available_modes = {"idle", "returning_home", "patrolling", "post_task_station_check"}
        for doctor_id in ordinary_doctor_ids:
            counter = self.mode_dwell_by_agent.get(doctor_id, Counter())
            total_seconds += sum(counter.values())
            task_seconds += counter.get("performing_task", 0)
            walking_seconds += sum(counter.get(mode, 0) for mode in walking_modes)
            station_or_available_seconds += sum(counter.get(mode, 0) for mode in available_modes)
        if total_seconds <= 0:
            return {
                "patient_task_fraction": 0.0,
                "walking_fraction": 0.0,
                "station_or_available_fraction": 0.0,
            }
        return {
            "patient_task_fraction": float(task_seconds / total_seconds),
            "walking_fraction": float(walking_seconds / total_seconds),
            "station_or_available_fraction": float(station_or_available_seconds / total_seconds),
        }

    @staticmethod
    def _esi_label(esi_level: object) -> str:
        return f"ESI {int(esi_level)}" if esi_level is not None else "ESI unknown"

    @staticmethod
    def _mean_waits_by_esi(values: Mapping[str, list[int]]) -> dict[str, float]:
        return {
            label: float(mean(waits)) if waits else 0.0
            for label, waits in sorted(values.items())
        }

    def _esi_workflow_diagnostics(self, warmup_seconds: int = 0) -> dict:
        warmup_seconds = max(0, int(warmup_seconds))
        arrival_time_by_patient: dict[int, int] = {}
        arrival_attempts_by_esi: Counter = Counter()
        admitted_arrivals_by_esi: Counter = Counter()
        deferred_arrivals_by_esi: Counter = Counter()
        triaged_by_esi: Counter = Counter()
        bed_assignment_by_esi: Counter = Counter()
        time_to_bed: dict[str, list[int]] = defaultdict(list)
        time_to_first_nurse: dict[str, list[int]] = defaultdict(list)
        time_to_first_doctor: dict[str, list[int]] = defaultdict(list)
        first_nurse_seen: set[int] = set()
        first_doctor_seen: set[int] = set()
        first_nurse_seen_by_esi: Counter = Counter()
        first_doctor_seen_by_esi: Counter = Counter()
        completed_by_esi_from_events: Counter = Counter()
        accepted_waiting_by_esi: Counter = Counter()
        rejected_external_by_esi: Counter = Counter()

        all_workflow_events = list(self.workflow_event_log)
        for event in all_workflow_events:
            if event.get("event_type") == "patient_arrival" and event.get("patient_id") is not None:
                arrival_time_by_patient[int(event["patient_id"])] = int(event.get("timestamp", 0))

        for event in all_workflow_events:
            if int(event.get("timestamp", event.get("timestep", 0))) < warmup_seconds:
                continue
            event_type = event.get("event_type")
            patient_id = event.get("patient_id")
            esi_label = self._esi_label(event.get("esi_level"))
            timestamp = int(event.get("timestamp", 0))
            if event_type in {"patient_arrival", "deferred_arrival"}:
                arrival_attempts_by_esi[esi_label] += 1
            if event_type == "patient_arrival":
                admitted_arrivals_by_esi[esi_label] += 1
                triaged_by_esi[esi_label] += 1
                if event.get("new_state") == "urgent_waiting_for_placement":
                    accepted_waiting_by_esi[esi_label] += 1
            if event_type == "deferred_arrival":
                deferred_arrivals_by_esi[esi_label] += 1
                rejected_external_by_esi[esi_label] += 1
            if event_type == "bed_assigned":
                bed_assignment_by_esi[esi_label] += 1
                if patient_id is not None and int(patient_id) in arrival_time_by_patient:
                    time_to_bed[esi_label].append(max(timestamp - arrival_time_by_patient[int(patient_id)], 0))
            if event_type == "task_started" and patient_id is not None and int(patient_id) in arrival_time_by_patient:
                patient_key = int(patient_id)
                if event.get("agent_role") == "Nurse" and patient_key not in first_nurse_seen:
                    time_to_first_nurse[esi_label].append(max(timestamp - arrival_time_by_patient[patient_key], 0))
                    first_nurse_seen.add(patient_key)
                    first_nurse_seen_by_esi[esi_label] += 1
                if event.get("agent_role") == "Doctor" and patient_key not in first_doctor_seen:
                    time_to_first_doctor[esi_label].append(max(timestamp - arrival_time_by_patient[patient_key], 0))
                    first_doctor_seen.add(patient_key)
                    first_doctor_seen_by_esi[esi_label] += 1
            if event_type == "task_completed" and event.get("new_state") == "discharged":
                completed_by_esi_from_events[esi_label] += 1

        waiting_for_bed_by_esi = Counter(
            self._esi_label(patient.esi_level)
            for patient in self.active_patients
            if patient.bed_index is None and not patient.discharged
        )
        waiting_in_bed_by_esi = Counter(
            self._esi_label(patient.esi_level)
            for patient in self.active_patients
            if patient.bed_index is not None
            and not patient.discharged
            and patient.claiming_staff_id is None
            and not patient.coordination_hold_active(self.timestep)
        )
        completed_patients_by_esi = Counter(
            self._esi_label(patient.esi_level) for patient in self.completed_patients
        )
        active_patients_by_esi = Counter(
            self._esi_label(patient.esi_level)
            for patient in self.active_patients
            if not patient.discharged
        )
        end_of_run_backlog_by_esi = Counter()
        oldest_waiting_patient_age_by_esi: dict[str, int] = {}
        active_without_first_nurse_by_esi = Counter()
        active_without_first_doctor_by_esi = Counter()
        for patient in self.active_patients:
            if patient.discharged:
                continue
            esi_label = self._esi_label(patient.esi_level)
            is_waiting = (
                patient.bed_index is None
                or (
                    patient.claiming_staff_id is None
                    and not patient.coordination_hold_active(self.timestep)
                )
            )
            if is_waiting:
                end_of_run_backlog_by_esi[esi_label] += 1
                oldest_waiting_patient_age_by_esi[esi_label] = max(
                    oldest_waiting_patient_age_by_esi.get(esi_label, 0),
                    max(self.timestep - patient.spawned_at, 0),
                )
            if patient.gid not in first_nurse_seen:
                active_without_first_nurse_by_esi[esi_label] += 1
            if patient.gid not in first_doctor_seen:
                active_without_first_doctor_by_esi[esi_label] += 1
        low_acuity_deprioritized_by_esi = Counter()
        for key, value in self.ablation_diagnostics.items():
            prefix = "low_acuity_deprioritized_ESI_"
            if str(key).startswith(prefix):
                low_acuity_deprioritized_by_esi[f"ESI {str(key).replace(prefix, '')}"] += int(value)

        return {
            "arrival_attempts_total": int(sum(arrival_attempts_by_esi.values())),
            "arrival_attempts_by_esi": dict(arrival_attempts_by_esi),
            "admitted_arrivals_total": int(sum(admitted_arrivals_by_esi.values())),
            "admitted_arrivals_by_esi": dict(admitted_arrivals_by_esi),
            "deferred_or_rejected_arrivals_total": int(sum(deferred_arrivals_by_esi.values())),
            "deferred_or_rejected_arrivals_by_esi": dict(deferred_arrivals_by_esi),
            "rejected_external_total": int(sum(rejected_external_by_esi.values())),
            "rejected_external_by_esi": dict(rejected_external_by_esi),
            "accepted_waiting_by_esi": dict(accepted_waiting_by_esi),
            "ESI1_rejected_external_count": int(rejected_external_by_esi.get("ESI 1", 0)),
            "ESI2_rejected_external_count": int(rejected_external_by_esi.get("ESI 2", 0)),
            "urgent_waiting_esi1_esi2_count": int(
                accepted_waiting_by_esi.get("ESI 1", 0) + accepted_waiting_by_esi.get("ESI 2", 0)
            ),
            "bedded_by_esi": dict(bed_assignment_by_esi),
            "completed_by_esi": dict(completed_patients_by_esi),
            "active_by_esi": dict(active_patients_by_esi),
            "backlog_by_esi": dict(end_of_run_backlog_by_esi),
            "arrivals_by_esi": dict(arrival_attempts_by_esi),
            "triaged_by_esi": dict(triaged_by_esi),
            "bed_assignment_by_esi": dict(bed_assignment_by_esi),
            "waiting_for_bed_by_esi": dict(waiting_for_bed_by_esi),
            "waiting_in_bed_by_esi": dict(waiting_in_bed_by_esi),
            "low_acuity_deprioritized_by_esi": dict(low_acuity_deprioritized_by_esi),
            "time_to_bed_by_esi": self._mean_waits_by_esi(time_to_bed),
            "time_to_first_nurse_by_esi": self._mean_waits_by_esi(time_to_first_nurse),
            "time_to_first_doctor_by_esi": self._mean_waits_by_esi(time_to_first_doctor),
            "max_time_to_bed_by_esi": {label: max(waits) for label, waits in sorted(time_to_bed.items()) if waits},
            "max_time_to_first_nurse_by_esi": {
                label: max(waits) for label, waits in sorted(time_to_first_nurse.items()) if waits
            },
            "max_time_to_first_doctor_by_esi": {
                label: max(waits) for label, waits in sorted(time_to_first_doctor.items()) if waits
            },
            "first_nurse_seen_by_esi": dict(first_nurse_seen_by_esi),
            "first_doctor_seen_by_esi": dict(first_doctor_seen_by_esi),
            "completed_patients_by_esi": dict(completed_patients_by_esi),
            "completed_events_by_esi": dict(completed_by_esi_from_events),
            "active_patients_by_esi": dict(active_patients_by_esi),
            "end_of_run_backlog_by_esi": dict(end_of_run_backlog_by_esi),
            "oldest_waiting_patient_age_by_esi": dict(sorted(oldest_waiting_patient_age_by_esi.items())),
            "active_without_first_nurse_by_esi": dict(active_without_first_nurse_by_esi),
            "active_without_first_doctor_by_esi": dict(active_without_first_doctor_by_esi),
        }

    def _repair_state_inconsistencies(self) -> None:
        staff_lookup = self.staff_by_id()

        for patient in self.active_patients:
            if patient.claiming_staff_id is not None:
                staff_member = staff_lookup.get(patient.claiming_staff_id)
                if staff_member is None or staff_member.target_patient_id != patient.gid:
                    patient.claiming_staff_id = None

            if (
                patient.current_task_name == config.MEDICAL_EVALUATION_TASK_NAME
                and patient.seconds_at_current_task(self.timestep)
                > config.MEDICAL_EVALUATION_STALE_WAIT_SECONDS
                and not self._doctor_assigned_to_patient(patient)
            ):
                patient.claiming_staff_id = None
                patient.requested_doctor_id = None
                patient.requesting_nurse_id = None

        for staff_member in self.staff_agents:
            if staff_member.stalled_seconds < config.STAFF_TRANSIT_STALL_TIMEOUT_SECONDS:
                continue

            target_patient = (
                self.patient_by_id.get(staff_member.target_patient_id)
                if staff_member.target_patient_id is not None
                else None
            )
            handoff_patient = (
                self.patient_by_id.get(getattr(staff_member, "handoff_patient_id", None))
                if getattr(staff_member, "handoff_patient_id", None) is not None
                else None
            )
            repair_patient = target_patient or handoff_patient
            handoff_wait_repair = getattr(staff_member, "mode", None) == "awaiting_nurse"
            self.assignment_repair_events += 1
            if handoff_wait_repair:
                self.handoff_wait_repair_events += 1
            else:
                self.stuck_agent_events += 1

            self.log_workflow_event(
                agent=staff_member,
                patient=repair_patient,
                event_type="handoff_wait_repaired" if handoff_wait_repair else "staff_assignment_repaired",
                task_name=staff_member.current_task_name,
                old_state=staff_member.mode,
                new_state="assignment_cleared_after_stall",
                position=staff_member.position,
                notes=(
                    f"{staff_member.role} exceeded "
                    f"{'handoff wait' if handoff_wait_repair else 'transit stall'} timeout "
                    f"({config.STAFF_TRANSIT_STALL_TIMEOUT_SECONDS}s); assignment was cleared."
                ),
                extra_fields={
                    "repair_category": "handoff_wait" if handoff_wait_repair else "transit_stall",
                    "stalled_seconds": int(staff_member.stalled_seconds),
                    "mode": getattr(staff_member, "mode", None),
                    "current_task": getattr(staff_member, "current_task_name", None),
                    "destination": getattr(staff_member, "destination", None),
                    "route_destination": getattr(staff_member, "route_destination", None),
                    "route_plan_length": len(getattr(staff_member, "route_plan", [])),
                    "target_patient_id": getattr(staff_member, "target_patient_id", None),
                    "handoff_patient_id": getattr(staff_member, "handoff_patient_id", None),
                    "handoff_nurse_id": getattr(staff_member, "handoff_nurse_id", None),
                    "reserved_doctor_id": getattr(staff_member, "reserved_doctor_id", None),
                    "doctor_recheck_patient_id": getattr(staff_member, "doctor_recheck_patient_id", None),
                    "recheck_timer": getattr(staff_member, "recheck_timer", None),
                    "patient_claiming_staff_id": getattr(repair_patient, "claiming_staff_id", None),
                    "patient_requested_doctor_id": getattr(repair_patient, "requested_doctor_id", None),
                    "patient_requesting_nurse_id": getattr(repair_patient, "requesting_nurse_id", None),
                    "patient_current_task": getattr(repair_patient, "current_task_name", None),
                    "patient_bed_index": getattr(repair_patient, "bed_index", None),
                },
            )
            if target_patient is not None and target_patient.claiming_staff_id == staff_member.gid:
                target_patient.claiming_staff_id = None

            if isinstance(staff_member, Nurse):
                if (
                    target_patient is not None
                    and target_patient.requesting_nurse_id == staff_member.gid
                ):
                    target_patient.requested_doctor_id = None
                    target_patient.requesting_nurse_id = None
                staff_member.clear_doctor_wait_state()

            if isinstance(staff_member, Doctor):
                handoff_patient = (
                    self.patient_by_id.get(staff_member.handoff_patient_id)
                    if staff_member.handoff_patient_id is not None
                    else None
                )
                if (
                    handoff_patient is not None
                    and handoff_patient.requested_doctor_id == staff_member.gid
                ):
                    handoff_patient.requested_doctor_id = None
                    handoff_patient.requesting_nurse_id = None
                staff_member.clear_handoff()

            staff_member.clear_assignment()

        for nurse in self.nurses:
            tracked_patient = (
                self.patient_by_id.get(nurse.doctor_recheck_patient_id)
                if nurse.doctor_recheck_patient_id is not None
                else None
            )
            if (
                nurse.mode == "waiting_for_doctor"
                and (
                    tracked_patient is None
                    or tracked_patient.discharged
                    or tracked_patient.current_task_name != config.MEDICAL_EVALUATION_TASK_NAME
                )
            ):
                nurse.clear_doctor_wait_state()
                nurse.clear_assignment()

        doctor_targets: Dict[int, List[Doctor]] = {}
        for doctor in self.doctors:
            if doctor.target_patient_id is None:
                continue
            doctor_targets.setdefault(doctor.target_patient_id, []).append(doctor)

        for patient_id, targeting_doctors in doctor_targets.items():
            if len(targeting_doctors) <= 1:
                continue

            patient = self.patient_by_id.get(patient_id)
            keeper_id = (
                patient.claiming_staff_id
                if patient is not None
                and patient.claiming_staff_id in {doctor.gid for doctor in targeting_doctors}
                else targeting_doctors[0].gid
            )
            for doctor in targeting_doctors:
                if doctor.gid == keeper_id:
                    continue
                doctor.clear_handoff()
                doctor.clear_assignment()

    def log_interaction(
        self,
        agent_1,
        agent_2,
        position: tuple[float, float],
        interaction_type: str,
        task_name: Optional[str] = None,
        run_id: Optional[int] = None,
        reason_for_interaction: Optional[str] = None,
        perception_gate_result: Optional[dict] = None,
    ) -> None:
        self.ablation_diagnostics["eligible_encounters_considered"] += 1
        close_threshold = float(config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS)
        current_distance = math.dist(agent_1.position, agent_2.position)
        is_staff_staff = getattr(agent_1, "role", None) != "Patient" and getattr(agent_2, "role", None) != "Patient"
        is_patient_facing = "Patient" in {getattr(agent_1, "role", None), getattr(agent_2, "role", None)}
        actual_contact_position = self._embodied_contact_position(agent_1, agent_2)

        if current_distance > close_threshold:
            self.ablation_diagnostics["missed_not_close_opportunity_count"] += 1
            if interaction_type == "task_transition_update":
                self.ablation_diagnostics["task_transition_update_not_close_count"] += 1
            self.log_missed_opportunity(
                agent=agent_1,
                partner=agent_2,
                reason="not_in_close_physical_contact",
                position=actual_contact_position,
                patient=agent_1 if getattr(agent_1, "role", None) == "Patient" else agent_2 if getattr(agent_2, "role", None) == "Patient" else None,
                perception_state={
                    "interaction_type": interaction_type,
                    "distance_m": current_distance,
                    "close_threshold_m": close_threshold,
                    "blocked_validation_counted_log": True,
                },
                task_state=getattr(agent_1, "mode", None),
            )
            return

        requested_position = position
        position = actual_contact_position
        gate_result = perception_gate_result
        if gate_result is None:
            gate_result = self._perception_gate(
                agent_1,
                agent_2,
                interaction_type=interaction_type,
                position=position,
                task_name=task_name,
            )
        if not bool(gate_result.get("perception_checked", False)):
            self.interactions_logged_without_perception_check += 1
            self.ablation_diagnostics["interactions_logged_without_perception_check"] += 1
        if self.variant_settings["perception_required"]:
            if not gate_result["accepted"]:
                self.ablation_diagnostics[str(gate_result["category"])] += 1
                self.log_missed_opportunity(
                    agent=agent_1,
                    partner=agent_2,
                    reason="interaction_partner_not_perceived",
                    position=position,
                    patient=agent_2 if getattr(agent_2, "role", None) == "Patient" else None,
                    perception_state={
                        "mutual_visibility": False,
                        "interaction_type": interaction_type,
                        "gate_category": gate_result["category"],
                        "gate_reason": gate_result["reason"],
                    },
                    task_state=getattr(agent_1, "mode", None),
                )
                return
        self.ablation_diagnostics["perception_approved_candidate_count"] += 1
        self.ablation_diagnostics[str(gate_result.get("category", "accepted_direct_fov"))] += 1

        decision = self.interaction_engine.decide(
            agent_1=agent_1,
            agent_2=agent_2,
            simulation=self,
            position=position,
            interaction_type=interaction_type,
            task_name=task_name,
        )
        if bool(gate_result.get("force_interaction_decision", False)) and not decision.interact:
            decision.interact = True
            decision.reason = "embodied_perception_intent_completed_at_contact"
            if isinstance(decision.parsed_response, dict):
                decision.parsed_response["completed_staff_interaction_intent"] = True
        if (
            gate_result.get("part3_decision_output_contract")
            == "categorical_causal_v1"
            and not bool(gate_result.get("part3_decision_fallback", False))
        ):
            decision.topic = str(gate_result["part3_topic_family"])
            decision.reason = str(gate_result["part3_selected_reason"])
            decision.persona_id = str(gate_result["part3_persona_id"])
            decision.backend_name = str(gate_result["part3_policy_name"])
            decision.raw_response = None
            decision.parsed_response = {
                "decision_id": gate_result["part3_decision_id"],
                "evidence_id": gate_result["part3_evidence_id"],
                "selected_action": gate_result["part3_selected_action"],
                "selected_reason": gate_result["part3_selected_reason"],
                "topic_family": gate_result["part3_topic_family"],
                "decision_output_contract": "categorical_causal_v1",
                "free_text_used_as_causal_input": False,
            }
        if not decision.interact:
            self.ablation_diagnostics["rejected_by_probability_or_rule_decision"] += 1
            self.log_missed_opportunity(
                agent=agent_1,
                partner=agent_2,
                reason="attended_but_no_interaction",
                position=position,
                patient=agent_2 if getattr(agent_2, "role", None) == "Patient" else None,
                perception_state={"decision_reason": decision.reason},
                task_state=getattr(agent_1, "mode", None),
            )
            return

        self.ablation_diagnostics["accepted_as_interactions"] += 1
        self.perception_checked_interaction_count += 1
        self.ablation_diagnostics["perception_checked_interaction_count"] += 1
        if interaction_type == "task_transition_update":
            self.ablation_diagnostics["task_transition_update_logged_count"] += 1
            self.ablation_diagnostics["task_transition_update_max_logged_distance"] = max(
                float(self.ablation_diagnostics.get("task_transition_update_max_logged_distance", 0.0)),
                current_distance,
            )
            if self.which_zone(*position) in self.corridor_or_threshold_interaction_zone_ids():
                self.ablation_diagnostics["walking_update_corridor_interaction_count"] += 1
        if interaction_type == "opportunistic_corridor":
            self.ablation_diagnostics["corridor_close_copresence_logged_count"] += 1
            if reason_for_interaction == "corridor_stop_and_talk_micro_interaction":
                self.ablation_diagnostics["corridor_micro_interaction_count"] += 1
        if is_staff_staff:
            self.ablation_diagnostics["staff_staff_logged_max_distance"] = max(
                float(self.ablation_diagnostics.get("staff_staff_logged_max_distance", 0.0)),
                current_distance,
            )
        if is_patient_facing:
            self.ablation_diagnostics["patient_facing_logged_max_distance"] = max(
                float(self.ablation_diagnostics.get("patient_facing_logged_max_distance", 0.0)),
                current_distance,
            )
        if bool(gate_result.get("staff_interaction_intent_id")):
            self.ablation_diagnostics["pending_intent_completed_interaction_count"] += 1
        else:
            self.ablation_diagnostics["immediate_close_interaction_count"] += 1
        event_id = f"comm_{self.random_seed}_{self.condition_spec.name}_{self.next_communicative_event_index}"
        self.next_communicative_event_index += 1
        zone_id = self.which_zone(*position) or "Outside named zones"
        visible_before_interaction = self.perception.can_agents_mutually_perceive(agent_1, agent_2, self)
        initiation_distance = current_distance
        nonproximate_staff = bool(is_staff_staff and initiation_distance > close_threshold)
        close_proximity_staff = bool(is_staff_staff and initiation_distance <= close_threshold)
        if is_staff_staff and gate_result.get("interaction_source") == "opportunistic_perception_interruption":
            self.ablation_diagnostics["final_logged_distance_distribution_sum"] += initiation_distance
            self.ablation_diagnostics["final_logged_distance_distribution_count"] += 1
            if nonproximate_staff:
                self.ablation_diagnostics["nonproximate_logged_interaction_count"] += 1
            else:
                self.ablation_diagnostics["interaction_locations_are_actual_contact_coordinates"] += 1
        isovist_polygon = self.perception.compute_isovist(
            position=agent_1.position,
            heading=getattr(agent_1, "heading", 0.0),
            fov_degrees=config.FOV_DEGREES,
        )
        patient = agent_1 if getattr(agent_1, "role", None) == "Patient" else agent_2 if getattr(agent_2, "role", None) == "Patient" else None
        if patient is None and task_name is not None:
            patient = self.patient_by_id.get(getattr(agent_1, "target_patient_id", None))

        event: Dict[str, object] = {
            "event_id": event_id,
            "timestep": self.timestep,
            "timestamp": self.timestep,
            "simulation_seed": self.random_seed,
            "model_variant": self.model_variant,
            "x": position[0],
            "y": position[1],
            "agent_1": agent_1.gid,
            "agent_1_id": agent_1.gid,
            "agent_1_name": getattr(agent_1, "name", f"{agent_1.role} {agent_1.gid}"),
            "agent_1_role": agent_1.role,
            "role_1": agent_1.role,
            "agent_2": agent_2.gid,
            "agent_2_id": agent_2.gid,
            "agent_2_name": getattr(agent_2, "name", f"{agent_2.role} {agent_2.gid}"),
            "agent_2_role": agent_2.role,
            "role_2": agent_2.role,
            "patient_id": getattr(patient, "gid", None),
            "interaction_type": decision.interaction_type,
            "topic": decision.topic,
            "duration_seconds": decision.duration_seconds,
            "urgency": getattr(patient, "urgency", None),
            "esi_level": getattr(patient, "esi_level", None),
            "complaint_type": getattr(patient, "complaint_type", None),
            "memory_summary": decision.memory_summary,
            "zone_id": zone_id,
            "condition_name": self.condition_spec.name,
            "perception_based_baseline_enabled": bool(config.PERCEPTION_BASED_BASELINE_ENABLED),
            "legacy_proximity_mode_enabled": bool(config.LEGACY_PROXIMITY_MODE_ENABLED),
            "perception_checked": bool(gate_result.get("perception_checked", False)),
            "co_presence_checked": True,
            "close_contact_threshold_m": close_threshold,
            "requested_interaction_x": requested_position[0],
            "requested_interaction_y": requested_position[1],
            "logged_coordinate_is_actual_contact": True,
            "synthetic_or_relocated_coordinate": False,
            "midpoint_logged_validation_interaction": False,
            "perception_required": bool(self.variant_settings["perception_required"]),
            "final_nonproximate_staff_interaction": nonproximate_staff,
            "close_proximity_staff_interaction": close_proximity_staff,
            "initiation_distance_m": initiation_distance,
            "initial_opportunity_distance_m": gate_result.get("initial_opportunity_distance_m", initiation_distance),
            "initial_nonproximate_staff_opportunity": bool(gate_result.get("initial_nonproximate", nonproximate_staff)),
            "staff_interaction_intent_id": gate_result.get("staff_interaction_intent_id"),
            "interaction_source": gate_result.get("interaction_source"),
            "actual_contact_x": position[0],
            "actual_contact_y": position[1],
            "final_logged_distance_m": initiation_distance,
            "approach_intent_created": bool(gate_result.get("approach_intent_created", False)),
            "stationary_initiated_to_moving": bool(gate_result.get("stationary_initiated_to_moving", False)),
            "perception_salience_score": gate_result.get("salience_score"),
            "interaction_reason_type": gate_result.get("reason_type"),
            "interaction_reason_patient_id": gate_result.get("reason_patient_id"),
            "perception_gate_category": gate_result.get("category"),
            "perception_gate_reason": gate_result.get("reason"),
            "interaction_backend": self.interaction_backend_name,
            "persona_source": self.persona_source,
            "initiated_by": getattr(agent_1, "gid", None),
            "reason_for_interaction": reason_for_interaction or decision.reason,
            "perception_context": {
                "mutual_visibility": visible_before_interaction,
                "approach_distance_meters": initiation_distance,
                "initial_opportunity_distance_m": gate_result.get("initial_opportunity_distance_m", initiation_distance),
                "final_contact_distance_m": initiation_distance,
                "fov_degrees": config.FOV_DEGREES,
                "nonproximate_staff": nonproximate_staff,
                "gate_category": gate_result.get("category"),
                "salience_score": gate_result.get("salience_score"),
            },
            "memory_context_ids": decision.retrieved_memory_ids or [],
            "memory_context_summaries": decision.retrieved_memory_summaries or [],
            "backend_name": decision.backend_name or self.interaction_backend_name,
            "raw_llm_response": decision.raw_response,
            "parsed_structured_response": decision.parsed_response,
            "used_fallback": decision.used_fallback,
            "fallback": decision.used_fallback,
            "backend_used": decision.backend_name or self.interaction_backend_name,
            "memory_effect_applied": decision.memory_effect_applied,
            "memory_effect_type": decision.memory_effect_type,
            "memory_ids_used": decision.memory_ids_used or [],
            "probability_before": decision.probability_before,
            "probability_after": decision.probability_after,
            "duration_before": decision.duration_before,
            "duration_after": decision.duration_after,
            "accepted_because_of_memory": decision.accepted_because_of_memory,
            "rejected_because_of_memory": decision.rejected_because_of_memory,
            "decision_flipped_by_memory": decision.decision_flipped_by_memory,
            "topic_before": decision.topic_before,
            "topic_after": decision.topic_after,
            "prompt_context_hash": decision.prompt_context_hash,
            "context_summary": decision.context_summary,
            "persona_id": decision.persona_id,
            "was_visible_before_interaction": visible_before_interaction,
            "isovist_area_at_initiation": self._polygon_area(isovist_polygon),
            "approach_distance_meters": math.dist(agent_1.position, agent_2.position),
            "trace_support_ids": decision.retrieved_memory_ids or [],
        }
        if gate_result.get("part3_decision_output_contract") == "categorical_causal_v1":
            event.update(
                {
                    "part3_decision_id": gate_result["part3_decision_id"],
                    "part3_evidence_id": gate_result["part3_evidence_id"],
                    "part3_persona_id": gate_result["part3_persona_id"],
                    "part3_selected_action": gate_result["part3_selected_action"],
                    "part3_selected_reason": gate_result["part3_selected_reason"],
                    "part3_topic_family": gate_result["part3_topic_family"],
                    "part3_policy_name": gate_result["part3_policy_name"],
                    "part3_decision_fallback": gate_result[
                        "part3_decision_fallback"
                    ],
                    "part3_decision_output_contract": "categorical_causal_v1",
                    "part3_free_text_used_as_causal_input": False,
                    "part3_decision_applied_to_interaction": not bool(
                        gate_result["part3_decision_fallback"]
                    ),
                }
            )
        if task_name is not None:
            event["task_name"] = task_name
        if run_id is not None:
            event["run_id"] = run_id
        self.communicative_interaction_log.append(event)
        self.interaction_engine.remember(
            decision,
            agent_1,
            agent_2,
            self,
            zone_id,
            source_event_id=event_id,
            reason_type=gate_result.get("reason_type"),
            # Match the source ledger's patient-context precedence: an explicit
            # reason-linked patient first, otherwise the patient resolved on
            # the realized interaction itself.
            patient_context_id=(
                gate_result.get("reason_patient_id")
                if gate_result.get("reason_patient_id") is not None
                else getattr(patient, "gid", None)
            ),
            force_part3_grounded=self.part3_cognitive_controller is not None,
        )
        if config.ENABLE_INTERRUPTION_METRICS and (
            getattr(agent_1, "is_task_busy", lambda: False)() or getattr(agent_2, "is_task_busy", lambda: False)()
        ):
            self.interruption_log.append(
                {
                    "event_id": event_id,
                    "timestep": self.timestep,
                    "agent_1_id": agent_1.gid,
                    "agent_2_id": agent_2.gid,
                    "interaction_type": decision.interaction_type,
                    "task_name": task_name,
                    "useful": decision.interaction_type in {"nurse_doctor_handoff", "doctor_task", "nursing_task"},
                }
            )

    def log_workflow_event(
        self,
        agent,
        patient: Optional[Patient],
        event_type: str,
        task_name: Optional[str],
        old_state: Optional[str],
        new_state: Optional[str],
        position: Optional[tuple[float, float]] = None,
        notes: str = "",
        extra_fields: Optional[Dict[str, object]] = None,
    ) -> None:
        event_id = f"flow_{self.random_seed}_{self.condition_spec.name}_{self.next_workflow_event_index}"
        self.next_workflow_event_index += 1
        if position is None:
            if agent is not None:
                position = agent.position
            elif patient is not None:
                position = patient.position
            else:
                position = (0.0, 0.0)
        if event_type == "task_started" and patient is not None:
            if getattr(agent, "role", None) == "Nurse":
                self.first_nurse_seen_patient_ids.add(patient.gid)
            if getattr(agent, "role", None) == "Doctor":
                self.first_doctor_seen_patient_ids.add(patient.gid)
        event = {
            "event_id": event_id,
            "timestep": self.timestep,
            "timestamp": self.timestep,
            "simulation_seed": self.random_seed,
            "condition_name": self.condition_spec.name,
            "model_variant": self.model_variant,
            "agent_id": getattr(agent, "gid", None),
            "agent_name": getattr(agent, "name", None),
            "agent_role": getattr(agent, "role", None),
            "patient_id": getattr(patient, "gid", None),
            "esi_level": getattr(patient, "esi_level", None),
            "urgency": getattr(patient, "urgency", None),
            "event_type": event_type,
            "task_name": task_name,
            "old_state": old_state,
            "new_state": new_state,
            "x": position[0],
            "y": position[1],
            "zone_id": self.which_zone(*position) or "Outside named zones",
            "notes": notes,
        }
        if extra_fields:
            event.update(extra_fields)
        self.workflow_event_log.append(event)

    def log_missed_opportunity(
        self,
        agent,
        partner,
        reason: str,
        position: tuple[float, float],
        patient: Optional[Patient] = None,
        perception_state: Optional[dict] = None,
        task_state: Optional[str] = None,
    ) -> None:
        if not config.ENABLE_MISSED_OPPORTUNITY_LOG:
            return
        key = (
            getattr(agent, "gid", None),
            getattr(partner, "gid", None),
            reason,
            self.which_zone(*position) or "Outside named zones",
        )
        self.missed_opportunity_debug_counts[key] += 1
        if config.MISSED_OPPORTUNITY_LOG_MODE == "episode":
            started_at = self.active_missed_opportunity_since.setdefault(key, self.timestep)
            duration = self.timestep - started_at
            immediate_episode_reasons = {
                "partner_busy_or_moving_away",
                "attended_but_no_interaction",
                "interaction_blocked_by_task",
            }
            if reason in immediate_episode_reasons:
                duration = max(duration, config.MISSED_OPPORTUNITY_MIN_DURATION_SECONDS)
            if duration < config.MISSED_OPPORTUNITY_MIN_DURATION_SECONDS:
                return
            last_logged = self.last_missed_opportunity_logged_at.get(key)
            if (
                last_logged is not None
                and self.timestep - last_logged < config.MISSED_OPPORTUNITY_COOLDOWN_SECONDS
            ):
                return
            self.last_missed_opportunity_logged_at[key] = self.timestep
        else:
            started_at = self.timestep
            duration = 0
        event_id = f"miss_{self.random_seed}_{self.condition_spec.name}_{self.next_missed_event_index}"
        self.next_missed_event_index += 1
        event = {
                "event_id": event_id,
                "timestep": self.timestep,
                "simulation_seed": self.random_seed,
                "condition_name": self.condition_spec.name,
                "model_variant": self.model_variant,
                "agent_id": getattr(agent, "gid", None),
                "agent_name": getattr(agent, "name", None),
                "agent_role": getattr(agent, "role", None),
                "potential_partner_id": getattr(partner, "gid", None),
                "potential_partner_name": getattr(partner, "name", None),
                "potential_partner_role": getattr(partner, "role", None),
                "patient_id": getattr(patient, "gid", None),
                "esi_level": getattr(patient, "esi_level", None),
                "x": position[0],
                "y": position[1],
                "zone_id": self.which_zone(*position) or "Outside named zones",
                "reason": reason,
                "episode_started_at": started_at,
                "episode_duration_seconds": duration,
                "debug_timestep_count_for_key": self.missed_opportunity_debug_counts[key],
                "log_mode": config.MISSED_OPPORTUNITY_LOG_MODE,
                "perception_state": perception_state or {},
                "task_state": task_state,
            }
        self.missed_opportunity_log.append(event)
        self.interaction_engine.remember_missed_opportunity(event, agent)

    def location_cluster(self, position: tuple[float, float]) -> str:
        zone_id = self.which_zone(*position)
        if zone_id in config.STATION_ZONE_IDS:
            return zone_id
        if zone_id is not None and zone_id.startswith("CORR"):
            return zone_id

        if position[0] >= 6.0 and -3.0 <= position[1] <= 6.0:
            return "RIGHT_ROOM_CLUSTER"
        if position[0] <= -6.0 and -3.0 <= position[1] <= 6.0:
            return "LEFT_ROOM_CLUSTER"
        return zone_id or "Outside named zones"

    def record_trip_completion(
        self,
        label: str,
        distance_meters: float,
        duration_seconds: float,
    ) -> None:
        self.trip_path_lengths_by_label[label].append(float(distance_meters))
        self.trip_durations_by_label[label].append(float(duration_seconds))

    def _record_step_metrics(self) -> None:
        occupied_beds = {
            patient.bed_index
            for patient in self.active_patients
            if patient.bed_index is not None and not patient.discharged
        }
        self.bed_occupancy_samples.append(len(occupied_beds))
        pressure = self.ed_pressure_snapshot()
        ordinary_bed_indices = self.ordinary_bed_indices()
        special_bed_indices = set(getattr(config, "HIGH_ACUITY_RESERVED_BED_INDICES", set()))
        active_esi1_count = sum(
            1 for patient in self.active_patients
            if not patient.discharged and int(getattr(patient, "esi_level", 5)) == 1
        )
        active_esi2_count = sum(
            1 for patient in self.active_patients
            if not patient.discharged and int(getattr(patient, "esi_level", 5)) == 2
        )
        self.ordinary_bed_occupancy_samples.append(int(pressure["ordinary_bed_occupancy"]))
        self.special_bed_occupancy_samples.append(len(special_bed_indices & occupied_beds))
        self.waiting_queue_samples.append(int(pressure["waiting_count"]))
        self.ed_pressure_index_samples.append(float(pressure["pressure_index"]))
        self.active_high_acuity_patient_samples.append(int(pressure["active_high_acuity_patients"]))
        self.active_esi1_patient_samples.append(int(active_esi1_count))
        self.active_esi2_patient_samples.append(int(active_esi2_count))
        self.active_esi1_esi2_patient_samples.append(int(active_esi1_count + active_esi2_count))
        if int(pressure["waiting_count"]) == 0:
            self.zero_waiting_seconds += config.TIMESTEP_SECONDS
        else:
            self.waiting_pressure_seconds += config.TIMESTEP_SECONDS
        if bool(pressure["all_ordinary_beds_full"]):
            self.ordinary_beds_full_seconds += config.TIMESTEP_SECONDS
        if int(pressure["open_ordinary_beds"]) > 0:
            self.open_ordinary_bed_seconds += config.TIMESTEP_SECONDS
        if int(pressure["active_high_acuity_patients"]) > 0:
            self.high_acuity_active_seconds += config.TIMESTEP_SECONDS
        for bed_index in ordinary_bed_indices:
            bed_label = f"bed_{bed_index + 1}"
            is_open = bed_index not in occupied_beds
            self.ordinary_bed_open_samples_by_bed[bed_label].append(1 if is_open else 0)
            if bed_index not in occupied_beds:
                self.ordinary_bed_open_seconds_by_bed[bed_label] += config.TIMESTEP_SECONDS

        current_zone_by_agent = {}
        for agent in self.staff_agents:
            current_zone = self.which_zone(*agent.position) or "Outside named zones"
            current_zone_by_agent[agent.gid] = current_zone
            self.zone_dwell_by_role[agent.role][current_zone] += config.TIMESTEP_SECONDS
            self.zone_dwell_by_agent[agent.gid][current_zone] += config.TIMESTEP_SECONDS
            self.mode_dwell_by_role[agent.role][agent.mode] += config.TIMESTEP_SECONDS
            self.mode_dwell_by_agent[agent.gid][agent.mode] += config.TIMESTEP_SECONDS

            previous_zone = self.last_zone_by_agent.get(agent.gid)
            if previous_zone is not None and previous_zone != current_zone:
                self.zone_transition_counts[f"{previous_zone}->{current_zone}"] += 1
                self.zone_entered_at_by_agent[agent.gid] = self.timestep
            self.last_zone_by_agent[agent.gid] = current_zone

            current_cluster = self.location_cluster(agent.position)
            previous_cluster = self.last_cluster_by_agent.get(agent.gid)
            if previous_cluster is not None and previous_cluster != current_cluster:
                self.trip_counts[f"{previous_cluster}->{current_cluster}"] += 1
            self.last_cluster_by_agent[agent.gid] = current_cluster

            previous_position = agent.trail[-2] if len(agent.trail) >= 2 else agent.position
            distance_moved = math.dist(previous_position, agent.position)
            self.movement_distance_by_role[agent.role] += distance_moved
            self.movement_distance_by_agent[agent.gid] += distance_moved

        mutually_visible_partner_ids: Dict[int, set[int]] = defaultdict(set)
        for left_index, left_agent in enumerate(self.staff_agents):
            for right_agent in self.staff_agents[left_index + 1:]:
                if not self.perception.can_agents_mutually_perceive(left_agent, right_agent, self):
                    continue
                mutually_visible_partner_ids[left_agent.gid].add(right_agent.gid)
                mutually_visible_partner_ids[right_agent.gid].add(left_agent.gid)
                for agent in (left_agent, right_agent):
                    self.mutual_visibility_exposure_seconds_by_agent[
                        agent.gid
                    ] += config.TIMESTEP_SECONDS
                    self.mutual_visibility_exposure_seconds_by_agent_zone[
                        agent.gid
                    ][current_zone_by_agent[agent.gid]] += config.TIMESTEP_SECONDS
                midpoint = (
                    (left_agent.position[0] + right_agent.position[0]) / 2.0,
                    (left_agent.position[1] + right_agent.position[1]) / 2.0,
                )
                zone_id = self.which_zone(*midpoint) or "Outside named zones"
                self.mutual_visibility_counts_by_zone[zone_id] += 1
        for agent_id, partners in mutually_visible_partner_ids.items():
            if partners:
                self.any_mutually_visible_staff_seconds_by_agent[
                    agent_id
                ] += config.TIMESTEP_SECONDS

    def _log_coordination_hub_interactions(self) -> None:
        station_modes = {
            "idle",
            "returning_home",
            "waiting_for_doctor",
            "using_secondary_station",
            "moving_to_post_task_station_check",
            "post_task_station_check",
            "awaiting_nurse",
            "patrolling",
        }
        coordinator = self.coordination_nurse
        if coordinator.is_task_busy() or coordinator.mode not in station_modes:
            return

        coordinator_zone = self.which_zone(*coordinator.position)
        hub_zones = self.station_interaction_zone_ids() | set(config.STATION_ADJACENT_ZONE_IDS)
        if coordinator_zone not in hub_zones:
            return

        for partner in self.staff_agents:
            if partner.gid == coordinator.gid:
                continue
            if getattr(partner, "senior_oversight_only", False):
                continue
            if partner.is_task_busy() or partner.mode not in station_modes:
                continue
            partner_zone = self.which_zone(*partner.position)
            if partner_zone not in hub_zones:
                continue
            satellite_station_context = (
                coordinator_zone in {"NURSTA", "NUROPE"}
                or partner_zone in {"NURSTA", "NUROPE"}
                or coordinator_zone in self.active_care_area_object_ids()
                or partner_zone in self.active_care_area_object_ids()
            )
            if coordinator.mode in {"patrolling", "returning_home"} and not satellite_station_context:
                continue
            distance_threshold = (
                config.SATELLITE_STATION_CHECKIN_DISTANCE_METERS
                if satellite_station_context
                else config.STATION_COORDINATION_DISTANCE_METERS
            )
            if coordinator.distance_to_agent(partner) > distance_threshold:
                self.ablation_diagnostics["rejected_by_distance"] += 1
                continue

            midpoint = (
                (coordinator.position[0] + partner.position[0]) / 2.0,
                (coordinator.position[1] + partner.position[1]) / 2.0,
            )
            midpoint_zone = self.which_zone(*midpoint)
            if midpoint_zone not in hub_zones:
                midpoint = coordinator.position
                midpoint_zone = coordinator_zone

            self.ablation_diagnostics["eligible_encounters_considered"] += 1
            gate_result = self._perception_gate(
                coordinator,
                partner,
                interaction_type="opportunistic_station",
                position=midpoint,
            )
            if not gate_result["accepted"]:
                self.ablation_diagnostics[str(gate_result["category"])] += 1
                self.log_missed_opportunity(
                    agent=coordinator,
                    partner=partner,
                    reason="needed_partner_not_visible",
                    position=midpoint,
                    perception_state={
                        "encounter": "coordination_hub",
                        "gate_category": gate_result["category"],
                        "gate_reason": gate_result["reason"],
                    },
                    task_state=coordinator.mode,
                )
                continue

            pair_key = tuple(sorted((coordinator.gid, partner.gid)))
            if pair_key in self.currently_interacting_pairs:
                self.ablation_diagnostics["rejected_by_cooldown"] += 1
                self.ablation_diagnostics["repeated_station_pair_suppressed_count"] += 1
                continue
            if self._station_same_dwell_repeat_suppressed(coordinator, partner, midpoint_zone):
                continue
            last_timestamp = self.last_coordination_hub_interaction_by_pair.get(pair_key)
            if (
                last_timestamp is not None
                and (self.timestep - last_timestamp) < config.COORDINATION_HUB_INTERACTION_COOLDOWN_SECONDS
            ):
                self.ablation_diagnostics["rejected_by_cooldown"] += 1
                self.ablation_diagnostics["station_same_pair_repeat_count"] += 1
                self.ablation_diagnostics["repeated_station_pair_suppressed_count"] += 1
                continue

            if midpoint_zone in {"NURSTA", "NUROPE"} or midpoint_zone in self.active_care_area_object_ids() or satellite_station_context:
                base_probability = max(
                    config.COORDINATION_HUB_INTERACTION_PROBABILITY,
                    config.SATELLITE_STATION_CHECKIN_PROBABILITY,
                )
                self.ablation_diagnostics["satellite_station_checkin_opportunities"] += 1
            else:
                base_probability = config.COORDINATION_HUB_INTERACTION_PROBABILITY

            probability = self._hcw_hcw_interaction_probability(
                coordinator,
                partner,
                midpoint_zone,
                base_probability,
            )
            if self.random.random() >= probability:
                self.ablation_diagnostics["rejected_by_probability_or_rule_decision"] += 1
                continue

            contact_position = self._embodied_contact_position(coordinator, partner)
            before_count = len(self.interaction_log)
            self.log_interaction(
                agent_1=coordinator,
                agent_2=partner,
                position=contact_position,
                interaction_type="opportunistic_station",
                reason_for_interaction="coordination_hub_check",
                perception_gate_result=gate_result,
            )
            if len(self.interaction_log) > before_count:
                self.currently_interacting_pairs.add(pair_key)
                self._mark_station_same_dwell_episode(coordinator, partner, midpoint_zone)
                self.last_coordination_hub_interaction_by_pair[pair_key] = self.timestep
                self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep

    def _staff_externally_interruptible(self, agent) -> bool:
        """Return whether unrelated opportunistic perception can divert this staff member."""

        if getattr(agent, "senior_oversight_only", False):
            return False
        if agent.is_task_busy():
            self.ablation_diagnostics["externally_interruptible_rejections_count"] += 1
            return False
        protected_modes = {
            "placing_patient",
            "escorting_patient",
            "performing_task",
            "leading_doctor_to_patient",
        }
        if getattr(agent, "mode", None) in protected_modes and getattr(agent, "target_patient_id", None) is not None:
            self.ablation_diagnostics["externally_interruptible_rejections_count"] += 1
            return False
        if getattr(agent, "mode", None) == "moving_to_patient" and getattr(agent, "target_patient_id", None) is not None:
            self.ablation_diagnostics["externally_interruptible_rejections_count"] += 1
            return False
        return getattr(agent, "mode", None) in {
            "idle",
            "returning_home",
            "moving_to_secondary_station",
            "using_secondary_station",
            "moving_to_post_task_station_check",
            "post_task_station_check",
            "patrolling",
            "waiting_for_doctor",
            "moving_to_doctor",
            "awaiting_nurse",
            "interaction_approach",
            "interaction_receive",
        }

    def _staff_interruptible_for_opportunistic_perception(self, agent) -> bool:
        """Compatibility wrapper for external opportunistic interruption checks."""

        return self._staff_externally_interruptible(agent)

    def _staff_locally_communicative(self, staff_agent, partner=None) -> bool:
        """Return whether staff can talk to someone already co-present in their task context."""

        if getattr(staff_agent, "senior_oversight_only", False):
            return False
        mode = getattr(staff_agent, "mode", None)
        if staff_agent.is_task_busy() and getattr(staff_agent, "target_patient_id", None) is not None:
            self.ablation_diagnostics["locally_communicative_allowed_count"] += 1
            return True
        if mode in {
            "performing_task",
            "leading_doctor_to_patient",
            "moving_to_doctor",
            "awaiting_nurse",
            "moving_to_patient",
            "returning_home",
            "moving_to_post_task_station_check",
            "post_task_station_check",
            "moving_to_secondary_station",
            "using_secondary_station",
            "patrolling",
            "idle",
            "waiting_for_doctor",
        }:
            self.ablation_diagnostics["locally_communicative_allowed_count"] += 1
            return True
        return False

    def _active_patient_context_ids(self, staff_agent) -> set[int]:
        patient_ids: set[int] = set()
        target_id = getattr(staff_agent, "target_patient_id", None)
        if target_id is not None:
            patient_ids.add(int(target_id))
        handoff_id = getattr(staff_agent, "handoff_patient_id", None)
        if handoff_id is not None:
            patient_ids.add(int(handoff_id))
        return patient_ids

    def _co_task_interaction_allowed(self, left_agent, right_agent) -> Optional[Patient]:
        if left_agent.distance_to_agent(right_agent) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            return None
        if not (self._staff_locally_communicative(left_agent, right_agent) and self._staff_locally_communicative(right_agent, left_agent)):
            return None
        shared_ids = self._active_patient_context_ids(left_agent) & self._active_patient_context_ids(right_agent)
        if not shared_ids:
            self.ablation_diagnostics["cotask_interaction_suppressed_not_same_patient"] += 1
            return None
        for patient_id in sorted(shared_ids):
            patient = self.patient_by_id.get(patient_id)
            if patient is None or patient.discharged:
                continue
            patient_position = patient.bed_position or patient.position
            if (
                left_agent.distance_to_point(patient_position) <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
                and right_agent.distance_to_point(patient_position) <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
            ):
                self.ablation_diagnostics["cotask_interaction_allowed_count"] += 1
                return patient
        self.ablation_diagnostics["cotask_interaction_suppressed_not_same_patient"] += 1
        return None

    def _save_staff_movement_state(self, agent) -> dict:
        return {
            "mode": getattr(agent, "mode", None),
            "current_task_name": getattr(agent, "current_task_name", None),
            "task_remaining": getattr(agent, "task_remaining", 0),
            "target_patient_id": getattr(agent, "target_patient_id", None),
            "destination": getattr(agent, "destination", None),
            "route_plan": list(getattr(agent, "route_plan", [])),
            "route_destination": getattr(agent, "route_destination", None),
            "patrol_destination": getattr(agent, "patrol_destination", None),
            "idle_zone_id": getattr(agent, "idle_zone_id", None),
            "last_idle_position": getattr(agent, "last_idle_position", None),
            "needs_new_idle_target": getattr(agent, "needs_new_idle_target", False),
            "idle_seconds": getattr(agent, "idle_seconds", 0),
            "trip_label": getattr(agent, "trip_label", None),
            "trip_distance_meters": getattr(agent, "trip_distance_meters", 0.0),
            "trip_started_at": getattr(agent, "trip_started_at", None),
            "post_task_station_check_zone_id": getattr(agent, "post_task_station_check_zone_id", None),
            "post_task_station_check_remaining": getattr(agent, "post_task_station_check_remaining", 0),
        }

    def _restore_staff_movement_state(self, agent, saved_state: dict) -> None:
        if not saved_state:
            return
        for attribute, value in {
            "mode": saved_state.get("mode", "idle"),
            "current_task_name": saved_state.get("current_task_name"),
            "task_remaining": saved_state.get("task_remaining", 0),
            "target_patient_id": saved_state.get("target_patient_id"),
            "destination": saved_state.get("destination"),
            "route_plan": list(saved_state.get("route_plan", [])),
            "route_destination": saved_state.get("route_destination"),
            "patrol_destination": saved_state.get("patrol_destination"),
            "idle_zone_id": saved_state.get("idle_zone_id", getattr(agent, "home_zone_id", None)),
            "last_idle_position": saved_state.get("last_idle_position"),
            "needs_new_idle_target": saved_state.get("needs_new_idle_target", False),
            "idle_seconds": saved_state.get("idle_seconds", 0),
            "trip_label": saved_state.get("trip_label"),
            "trip_distance_meters": saved_state.get("trip_distance_meters", 0.0),
            "trip_started_at": saved_state.get("trip_started_at"),
            "post_task_station_check_zone_id": saved_state.get("post_task_station_check_zone_id"),
            "post_task_station_check_remaining": saved_state.get("post_task_station_check_remaining", 0),
        }.items():
            setattr(agent, attribute, value)
            setattr(agent, f"next_{attribute}", value)
        agent.next_position = agent.position
        self.ablation_diagnostics["route_resume_count"] += 1

    def _active_staff_intent_for_agent(self, agent) -> Optional[StaffInteractionIntent]:
        for intent in self.pending_staff_interaction_intents.values():
            if intent.status not in {"pending", "approaching"}:
                continue
            if agent.gid in {intent.initiator_id, intent.target_id}:
                return intent
        return None

    def _agent_has_active_staff_interaction_intent(self, agent) -> bool:
        return self._active_staff_intent_for_agent(agent) is not None

    def _agent_has_pending_staff_interaction_role(self, agent, role: Optional[str] = None) -> bool:
        for intent in self.pending_staff_interaction_intents.values():
            if intent.status not in {"pending", "approaching"}:
                continue
            if role == "initiator" and intent.initiator_id == agent.gid:
                return True
            if role == "target" and intent.target_id == agent.gid:
                return True
            if role is None and agent.gid in {intent.initiator_id, intent.target_id}:
                return True
        return False

    def _route_supported(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        route = self.plan_route(start, end)
        if not route:
            return False
        if hasattr(self.condition_manager, "route_is_walkable"):
            return bool(self.condition_manager.route_is_walkable(start, route))
        previous = start
        for waypoint in route:
            if self.wall_collision(previous, waypoint):
                return False
            previous = waypoint
        return True

    def _staff_contact_position(self, initiator, target) -> tuple[float, float]:
        return tuple(initiator.position)

    def _embodied_contact_position(self, agent_1, agent_2) -> tuple[float, float]:
        if getattr(agent_1, "role", None) == "Patient":
            return tuple(agent_2.position)
        return tuple(agent_1.position)

    def _perception_related_zones(self, initiator, target, position: Optional[tuple[float, float]] = None) -> set[str]:
        zones = {
            self.which_zone(*initiator.position),
            self.which_zone(*target.position),
        }
        if position is not None:
            zones.add(self.which_zone(*position))
        return {str(zone) for zone in zones if zone is not None}

    def _increment_related_counter(
        self,
        base_counter: str,
        initiator,
        target,
        position: Optional[tuple[float, float]] = None,
    ) -> None:
        related_names = {
            "raw_visible_staff": ("raw_visible_staff_cocpit_related_count", "raw_visible_staff_nursta_related_count"),
            "eligible_visible_staff_opportunity": (
                "eligible_cocpit_related_opportunity_count",
                "eligible_nursta_related_opportunity_count",
            ),
            "approach_intent": ("approach_intent_cocpit_related_count", "approach_intent_nursta_related_count"),
            "completed": ("completed_cocpit_related_count", "completed_nursta_related_count"),
        }
        cockpit_counter, nursta_counter = related_names.get(
            base_counter,
            (f"{base_counter}_cocpit_related_count", f"{base_counter}_nursta_related_count"),
        )
        zones = self._perception_related_zones(initiator, target, position)
        if "COCPIT" in zones:
            self.ablation_diagnostics[cockpit_counter] += 1
        if "NURSTA" in zones:
            self.ablation_diagnostics[nursta_counter] += 1

    def initial_or_current_corridor_context(self, initiator, target) -> bool:
        corridor_zones = self.corridor_or_threshold_interaction_zone_ids()
        zones = {
            self.which_zone(*initiator.position),
            self.which_zone(*target.position),
        }
        return any(zone in corridor_zones for zone in zones)

    def _act_staff_interaction_intent(self, agent) -> None:
        intent = self._active_staff_intent_for_agent(agent)
        if intent is None:
            return
        staff_lookup = self.staff_by_id()
        initiator = staff_lookup.get(intent.initiator_id)
        target = staff_lookup.get(intent.target_id)
        if initiator is None or target is None:
            agent.next_position = agent.position
            return
        if self.timestep >= intent.target_hold_expires_at and initiator.distance_to_agent(target) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            self._fail_staff_interaction_intent(intent, "target_resumed_before_contact")
            agent.next_position = agent.position
            return

        if agent.gid == intent.target_id:
            target_saved_mode = str(intent.saved_target_state.get("mode", ""))
            target_moving_modes = {
                "returning_home",
                "moving_to_secondary_station",
                "moving_to_post_task_station_check",
                "patrolling",
                "moving_to_doctor",
                "moving_to_patient",
            }
            if target_saved_mode in target_moving_modes:
                destination = (
                    intent.saved_target_state.get("destination")
                    or intent.saved_target_state.get("patrol_destination")
                    or intent.saved_target_state.get("route_destination")
                    or getattr(agent, "home_position", agent.position)
                )
                intent.target_continued_route = True
                self.ablation_diagnostics["target_continued_route_count"] += 1
                agent.next_mode = target_saved_mode
                agent.next_destination = destination
                agent.move_toward(self, destination)
                return
            if (
                self._staff_interruptible_for_opportunistic_perception(agent)
                and self._staff_interruptible_for_opportunistic_perception(initiator)
                and self._route_supported(agent.position, initiator.position)
            ):
                if not intent.target_route_overridden:
                    intent.target_route_overridden = True
                    self.ablation_diagnostics["route_override_count"] += 1
                    self.ablation_diagnostics["target_route_override_count"] += 1
                intent.target_moved_toward_initiator = True
                agent.next_mode = "interaction_approach"
                agent.next_destination = initiator.position
                agent.next_current_task_name = getattr(agent, "current_task_name", None)
                agent.next_task_remaining = getattr(agent, "task_remaining", 0)
                agent.move_toward(self, initiator.position)
            else:
                intent.target_paused = True
                self.ablation_diagnostics["target_receive_pause_count"] += 1
                agent.next_position = agent.position
                agent.next_mode = "interaction_receive"
                agent.next_destination = agent.position
                agent.next_route_plan = []
                agent.next_route_destination = None
            return

        intent.status = "approaching"
        intent.current_target_position = target.position
        if not self._route_supported(agent.position, target.position):
            self._fail_staff_interaction_intent(intent, "path blocked")
            agent.next_position = agent.position
            return
        if not intent.route_overridden:
            intent.route_overridden = True
            self.ablation_diagnostics["route_override_count"] += 1
        agent.next_mode = "interaction_approach"
        agent.next_destination = target.position
        agent.next_current_task_name = getattr(agent, "current_task_name", None)
        agent.next_task_remaining = getattr(agent, "task_remaining", 0)
        agent.move_toward(self, target.position)

    def _create_staff_interaction_intent(
        self,
        initiator,
        target,
        *,
        distance: float,
        interaction_type: str,
        gate_result: dict,
        salience: float,
        stationary_initiated_to_moving: bool,
        interaction_source: str = "opportunistic_perception_interruption",
    ) -> StaffInteractionIntent:
        intent = StaffInteractionIntent(
            intent_id=f"staff_intent_{self.random_seed}_{self.condition_spec.name}_{self.next_staff_interaction_intent_index}",
            initiator_id=initiator.gid,
            target_id=target.gid,
            created_time=self.timestep,
            initial_distance=float(distance),
            initial_initiator_position=tuple(initiator.position),
            initial_target_position=tuple(target.position),
            initial_initiator_zone=self.which_zone(*initiator.position),
            initial_target_zone=self.which_zone(*target.position),
            current_target_position=tuple(target.position),
            interaction_type=interaction_type,
            interaction_source=interaction_source,
            perception_gate_result=dict(gate_result),
            salience_score=float(salience),
            initial_nonproximate=distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS,
            timeout_seconds=int(config.PERCEPTION_STAFF_INTERACTION_INTENT_TIMEOUT_SECONDS),
            saved_initiator_state=self._save_staff_movement_state(initiator),
            saved_target_state=self._save_staff_movement_state(target),
            stationary_initiated_to_moving=stationary_initiated_to_moving,
            last_distance=float(distance),
        )
        self.next_staff_interaction_intent_index += 1
        self.pending_staff_interaction_intents[intent.intent_id] = intent
        self.ablation_diagnostics["approach_intent_count"] += 1
        self._increment_related_counter("approach_intent", initiator, target)
        if (
            interaction_type == "opportunistic_corridor"
            or self.initial_or_current_corridor_context(initiator, target)
        ):
            self.ablation_diagnostics["corridor_pending_intent_count"] += 1
        if intent.initial_nonproximate:
            self.ablation_diagnostics["nonproximate_approach_intent_count"] += 1
        return intent

    def _fail_staff_interaction_intent(self, intent: StaffInteractionIntent, reason: str) -> None:
        if intent.intent_id not in self.pending_staff_interaction_intents:
            return
        intent.status = "failed"
        intent.failure_reason = reason
        intent.completed_time = self.timestep
        self.pending_staff_interaction_intents.pop(intent.intent_id, None)
        self.failed_staff_interaction_intents.append(intent)
        self.ablation_diagnostics["approach_failure_count"] += 1
        self.ablation_diagnostics[f"approach_failure_reason_{reason.replace(' ', '_')}"] += 1
        staff_lookup = self.staff_by_id()
        initiator = staff_lookup.get(intent.initiator_id)
        target = staff_lookup.get(intent.target_id)
        if initiator is not None:
            self._restore_staff_movement_state(initiator, intent.saved_initiator_state)
        if target is not None:
            self._restore_staff_movement_state(target, intent.saved_target_state)
        if initiator is not None:
            self.log_missed_opportunity(
                agent=initiator,
                partner=target,
                reason=reason,
                position=initiator.position,
                perception_state={
                    "interaction_source": intent.interaction_source,
                    "intent_id": intent.intent_id,
                    "distance_m": intent.last_distance,
                    "initial_distance_m": intent.initial_distance,
                    "salience_score": intent.salience_score,
                    "interaction_type": intent.interaction_type,
                    "failure_reason": reason,
                },
                task_state=getattr(initiator, "mode", None),
            )

    def _complete_staff_interaction_intent(self, intent: StaffInteractionIntent, initiator, target) -> None:
        contact_position = self._staff_contact_position(initiator, target)
        if initiator.distance_to_agent(target) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            self._fail_staff_interaction_intent(intent, "not_close_at_completion")
            return
        if (
            not self._staff_interruptible_for_opportunistic_perception(initiator)
            or not self._staff_interruptible_for_opportunistic_perception(target)
        ):
            self._fail_staff_interaction_intent(intent, "protected_at_completion")
            return
        contact_zone = self.which_zone(*contact_position)
        corridor_started = (
            intent.interaction_type == "opportunistic_corridor"
            or intent.initial_initiator_zone in self.corridor_or_threshold_interaction_zone_ids()
            or intent.initial_target_zone in self.corridor_or_threshold_interaction_zone_ids()
        )
        if (
            corridor_started
            and contact_zone in self.station_interaction_zone_ids()
            and intent.initial_target_zone not in self.station_interaction_zone_ids()
        ):
            self.ablation_diagnostics["corridor_rejection_reason_reached_station_before_contact"] += 1
            self._fail_staff_interaction_intent(intent, "corridor_intent_reached_station_before_contact")
            return
        gate_result = dict(intent.perception_gate_result)
        gate_result.update({
            "accepted": True,
            "category": "accepted_perceived_staff_contact",
            "reason": "visible_staff_intent_reached_close_physical_contact",
            "perception_checked": True,
            "approach_intent_created": bool(intent.initial_nonproximate),
            "stationary_initiated_to_moving": bool(intent.stationary_initiated_to_moving),
            "interaction_source": intent.interaction_source,
            "staff_interaction_intent_id": intent.intent_id,
            "initial_opportunity_distance_m": intent.initial_distance,
            "initial_nonproximate": intent.initial_nonproximate,
            "force_interaction_decision": True,
        })
        before_count = len(self.interaction_log)
        self.log_interaction(
            agent_1=initiator,
            agent_2=target,
            position=contact_position,
            interaction_type=intent.interaction_type,
            reason_for_interaction="perception_salience_attention_approached_to_contact",
            perception_gate_result=gate_result,
        )
        intent.status = "completed"
        intent.completed_time = self.timestep
        self.pending_staff_interaction_intents.pop(intent.intent_id, None)
        self.completed_staff_interaction_intents.append(intent)
        if len(self.interaction_log) > before_count:
            pair_key = tuple(sorted((initiator.gid, target.gid)))
            self.currently_interacting_pairs.add(pair_key)
            self.last_perceived_staff_interaction_by_pair[pair_key] = self.timestep
            if "station" in intent.interaction_type:
                self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep
                self._mark_station_same_dwell_episode(initiator, target, contact_zone)
            if "corridor" in intent.interaction_type:
                self.last_opportunistic_corridor_interaction_by_pair[pair_key] = self.timestep
            self.ablation_diagnostics["perceived_staff_interaction_count"] += 1
            self.ablation_diagnostics["approach_success_count"] += 1
            self._mark_reason_completed(gate_result)
            self._increment_related_counter("completed", initiator, target, contact_position)
            if (
                intent.interaction_type == "opportunistic_corridor"
                or intent.initial_initiator_zone in self.corridor_or_threshold_interaction_zone_ids()
                or intent.initial_target_zone in self.corridor_or_threshold_interaction_zone_ids()
            ):
                if contact_zone in self.corridor_or_threshold_interaction_zone_ids():
                    self.ablation_diagnostics["corridor_pending_intent_completed_in_corridor_count"] += 1
                elif contact_zone in self.station_interaction_zone_ids():
                    self.ablation_diagnostics["corridor_pending_intent_completed_in_station_count"] += 1
            if intent.interaction_type == "task_transition_update" and contact_zone in self.corridor_or_threshold_interaction_zone_ids():
                self.ablation_diagnostics["walking_update_corridor_interaction_count"] += 1
            if intent.interaction_type == "task_transition_update":
                self.ablation_diagnostics["task_update_episode_completed_count"] += 1
                role_pair = str(intent.perception_gate_result.get("task_update_role_pair", self._role_pair_key(initiator, target)))
                patient_id = int(intent.perception_gate_result.get("task_update_patient_id", -1))
                self.ablation_diagnostics[f"task_update_episode_role_pair_{role_pair}"] += 1
                self.ablation_diagnostics[f"task_update_episode_patient_context_{patient_id}"] += 1
            if intent.initial_nonproximate:
                self.ablation_diagnostics["nonproximate_approach_success_count"] += 1
            else:
                self.ablation_diagnostics["immediate_close_logged_count"] += 1
            if len(self.perception_event_examples) < config.PERCEPTION_MAX_EXAMPLES:
                self.perception_event_examples.append({
                    "timestep": self.timestep,
                    "intent_id": intent.intent_id,
                    "initiator": getattr(initiator, "name", initiator.gid),
                    "target": getattr(target, "name", target.gid),
                    "initiator_role": initiator.role,
                    "target_role": target.role,
                    "initiator_mode": intent.saved_initiator_state.get("mode"),
                    "target_mode": intent.saved_target_state.get("mode"),
                    "initial_initiator_zone": intent.initial_initiator_zone,
                    "initial_target_zone": intent.initial_target_zone,
                    "initial_initiator_position": list(intent.initial_initiator_position),
                    "initial_target_position": list(intent.initial_target_position),
                    "final_contact_position": list(contact_position),
                    "initial_distance_m": round(intent.initial_distance, 3),
                    "final_distance_m": round(intent.last_distance, 3),
                    "visible": True,
                    "mutual_visible": intent.perception_gate_result.get("mutual_visibility"),
                    "salience_score": round(float(intent.salience_score or 0.0), 3),
                    "reason_selected": "pending embodied intent completed at close physical contact",
                    "approach_or_pause_occurred": bool(intent.initial_nonproximate),
                    "initiator_destination_overridden": bool(intent.route_overridden),
                    "target_paused": bool(intent.target_paused),
                    "target_moved_toward_initiator": bool(intent.target_moved_toward_initiator),
                    "target_continued_route": bool(intent.target_continued_route),
                    "target_destination_overridden": bool(intent.target_route_overridden),
                    "initiator_resumed_mode": intent.saved_initiator_state.get("mode"),
                    "target_resumed_mode": intent.saved_target_state.get("mode"),
                    "interaction_type": intent.interaction_type,
                    "topic": self.interaction_log[-1].get("topic") if self.interaction_log else None,
                    "location": self.which_zone(*contact_position) or "Outside named zones",
                    "cooldown_state": "new_or_cooldown_elapsed",
                    "created_time": intent.created_time,
                    "completed_time": intent.completed_time,
                    "status": intent.status,
                })
        self._restore_staff_movement_state(initiator, intent.saved_initiator_state)
        self._restore_staff_movement_state(target, intent.saved_target_state)

    def _update_staff_interaction_intents(self) -> None:
        if not self.pending_staff_interaction_intents:
            return
        staff_lookup = self.staff_by_id()
        for intent in list(self.pending_staff_interaction_intents.values()):
            initiator = staff_lookup.get(intent.initiator_id)
            target = staff_lookup.get(intent.target_id)
            if initiator is None or target is None:
                self._fail_staff_interaction_intent(intent, "other")
                continue
            intent.current_target_position = tuple(target.position)
            intent.last_distance = initiator.distance_to_agent(target)
            if self.timestep - intent.created_time > intent.timeout_seconds:
                self._fail_staff_interaction_intent(intent, "timeout")
                continue
            if self.timestep >= intent.target_hold_expires_at and intent.last_distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
                self._fail_staff_interaction_intent(intent, "target_resumed_before_contact")
                continue
            if not self._staff_interruptible_for_opportunistic_perception(initiator):
                self._fail_staff_interaction_intent(intent, "initiator became protected")
                continue
            if not self._staff_interruptible_for_opportunistic_perception(target):
                self._fail_staff_interaction_intent(intent, "target became protected")
                continue
            if not self._route_supported(initiator.position, target.position):
                self._fail_staff_interaction_intent(intent, "path blocked")
                continue
            if intent.last_distance > max(intent.initial_distance + 2.0, config.PERCEPTION_INTERACTION_RADIUS_METERS):
                self._fail_staff_interaction_intent(intent, "target moved away")
                continue
            if intent.last_distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
                self._complete_staff_interaction_intent(intent, initiator, target)

    def _visibility_candidate_salience(self, initiator, target, percept) -> float:
        distance = max(float(percept.distance_m), 0.01)
        salience = float(percept.salience)
        role_pair = "|".join(sorted((initiator.role, target.role)))
        salience *= float(config.PERCEPTION_ROLE_PAIR_SALIENCE_WEIGHTS.get(role_pair, 1.0))
        if bool(percept.mutual_visibility):
            salience += 0.10
        initiator_zone = self.which_zone(*initiator.position)
        target_zone = self.which_zone(*target.position)
        midpoint = (
            (initiator.position[0] + target.position[0]) / 2.0,
            (initiator.position[1] + target.position[1]) / 2.0,
        )
        midpoint_zone = self.which_zone(*midpoint)
        for zone_id in {initiator_zone, target_zone, midpoint_zone}:
            if zone_id is not None:
                salience += float(config.PERCEPTION_ZONE_SALIENCE_BONUSES.get(zone_id, 0.0))
        if initiator_zone is not None and initiator_zone == target_zone:
            salience += 0.12
        elif midpoint_zone in set(config.STATION_ADJACENT_ZONE_IDS):
            salience += 0.08
        if initiator.role == "CoordinationNurse" and target.role in {"Nurse", "Doctor"}:
            salience += 0.14
        if getattr(initiator, "mode", None) in {
            "returning_home",
            "moving_to_secondary_station",
            "moving_to_post_task_station_check",
            "patrolling",
        }:
            salience += 0.04
        if getattr(target, "mode", None) in {"moving_to_patient", "performing_task"}:
            salience *= float(config.PERCEPTION_TARGET_BUSYNESS_PENALTY)
        if distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            salience += 0.30
        return salience

    def _perceived_staff_interaction_type(self, initiator, target, midpoint_zone: Optional[str], distance: float) -> str:
        initiator_zone = self.which_zone(*initiator.position)
        target_zone = self.which_zone(*target.position)
        transit_modes = {
            "returning_home",
            "moving_to_secondary_station",
            "moving_to_post_task_station_check",
            "patrolling",
            "moving_to_doctor",
            "moving_to_patient",
        }
        movement_context = (
            getattr(initiator, "mode", None) in transit_modes
            or getattr(target, "mode", None) in transit_modes
            or (initiator_zone is not None and target_zone is not None and initiator_zone != target_zone)
            or distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
        )
        corridor_context = (
            initiator_zone in self.corridor_or_threshold_interaction_zone_ids()
            or target_zone in self.corridor_or_threshold_interaction_zone_ids()
            or midpoint_zone in self.corridor_or_threshold_interaction_zone_ids()
        )
        if corridor_context and movement_context:
            return "opportunistic_corridor"
        if midpoint_zone in self.station_interaction_zone_ids() and not movement_context:
            return "opportunistic_station"
        if midpoint_zone in self.corridor_or_threshold_interaction_zone_ids() and movement_context:
            return "opportunistic_corridor"
        if midpoint_zone in self.station_interaction_zone_ids() or midpoint_zone in set(config.STATION_ADJACENT_ZONE_IDS):
            return "opportunistic_station"
        if midpoint_zone in self.corridor_or_threshold_interaction_zone_ids():
            return "opportunistic_corridor"
        if distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
            return "staff_status_update"
        return "staff_status_update"

    def _perceived_staff_interaction_probability(
        self,
        interaction_type: str,
        initiator,
        target,
        zone_id: Optional[str],
        reason_type: Optional[str] = None,
        urgency: float = 0.0,
    ) -> float:
        work_relevant_reasons = {
            "POST_TASK_UPDATE",
            "SAME_PATIENT_UPDATE",
            "HANDOFF_NEED",
            "BED_FLOW_COORDINATION",
            "HIGH_ACUITY_ESCALATION",
        }
        work_relevant = str(reason_type or "") in work_relevant_reasons
        if work_relevant:
            urgency_multiplier = 1.0 + min(max(float(urgency), 0.0), 1.0) * 0.20
            if interaction_type == "opportunistic_station":
                base = config.PERCEIVED_STAFF_INTERACTION_PROBABILITY * 0.65 * urgency_multiplier
            else:
                base = config.PERCEIVED_STAFF_INTERACTION_PROBABILITY * urgency_multiplier
        elif interaction_type == "opportunistic_station":
            base = config.OPPORTUNISTIC_STATION_INTERACTION_PROBABILITY * 0.11
        elif interaction_type == "opportunistic_corridor":
            base = config.OPPORTUNISTIC_CORRIDOR_INTERACTION_PROBABILITY * 0.23
        else:
            base = config.PERCEIVED_STAFF_INTERACTION_PROBABILITY
        pressure = self.current_ed_pressure_index()
        if (
            not work_relevant
            and pressure >= self.pressure_action_threshold()
            and interaction_type in {"opportunistic_station", "opportunistic_corridor"}
        ):
            self.ablation_diagnostics["pressure_suppression_probability_application_count"] += 1
            suppression = float(self.scenario_definition.get("pressure_station_social_suppression", 0.0))
            base *= max(0.20, 1.0 - suppression)
        return self._hcw_hcw_interaction_probability(initiator, target, zone_id, base)

    def _visible_staff_reason(self, initiator, target, interaction_type: str, distance: float) -> dict:
        initiator_patient_id = getattr(initiator, "target_patient_id", None)
        target_patient_id = getattr(target, "target_patient_id", None)
        pair_key = tuple(sorted((initiator.gid, target.gid)))
        initiator_zone = self.which_zone(*initiator.position)
        target_zone = self.which_zone(*target.position)
        close = distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS

        reason_type = "NO_ACTION"
        patient_id = None
        urgency = 0.0
        target_can_satisfy = True
        rejection_reason = "no_task_or_workflow_reason"
        pressure = self.ed_pressure_snapshot()
        pressure_index = float(pressure["pressure_index"])
        pressure_active = pressure_index >= self.pressure_action_threshold()

        pending_updates = [
            update for update in self.pending_task_transition_updates
            if int(update.get("initiator_id", -1)) == int(initiator.gid)
            and target.role in set(update.get("preferred_roles", []))
        ]
        if pending_updates:
            update = min(pending_updates, key=lambda item: int(item.get("created_at", self.timestep)))
            reason_type = "POST_TASK_UPDATE"
            patient_id = int(update.get("patient_id", -1))
            urgency = 0.70
        elif initiator_patient_id is not None and initiator_patient_id == target_patient_id:
            reason_type = "SAME_PATIENT_UPDATE"
            patient_id = int(initiator_patient_id)
            urgency = 0.85
        elif (
            (initiator.role == "Nurse" and target.role == "Doctor")
            or (initiator.role == "Doctor" and target.role == "Nurse")
        ) and (
            getattr(initiator, "mode", None) in {"waiting_for_doctor", "moving_to_doctor", "leading_doctor_to_patient"}
            or getattr(target, "mode", None) in {"waiting_for_doctor", "moving_to_doctor", "leading_doctor_to_patient"}
        ):
            reason_type = "HANDOFF_NEED"
            patient_id = int(initiator_patient_id or target_patient_id or -1)
            urgency = 0.80
        elif initiator.role == "CoordinationNurse" and target.role in {"Nurse", "Doctor"}:
            waiting_or_backlog = any(
                not patient.discharged
                and (
                    patient.bed_index is None
                    or patient.needs_doctor_request(self.timestep)
                    or patient.coordination_hold_active(self.timestep)
                )
                for patient in self.active_patients
            )
            if waiting_or_backlog:
                reason_type = "BED_FLOW_COORDINATION"
                patient_id = -1
                urgency = 0.60 + min(pressure_index * 0.20, 0.25)
                if pressure_active:
                    self.ablation_diagnostics["pressure_created_bed_flow_reasons"] += 1
            else:
                target_can_satisfy = False
                rejection_reason = "no_active_bed_flow_need"
        elif any(
            patient_id is not None
            and patient_id in self.patient_by_id
            and getattr(self.patient_by_id[patient_id], "esi_level", 5) in {1, 2}
            for patient_id in (initiator_patient_id, target_patient_id)
        ):
            reason_type = "HIGH_ACUITY_ESCALATION"
            patient_id = int(initiator_patient_id or target_patient_id or -1)
            urgency = 0.90 + min(pressure_index * 0.10, 0.10)
            self.ablation_diagnostics["pressure_created_high_acuity_reasons"] += 1
        elif (
            interaction_type == "opportunistic_station"
            and close
            and initiator_zone == target_zone
            and initiator_zone in self.station_interaction_zone_ids()
        ):
            self.ablation_diagnostics["pressure_suppression_candidate_count"] += 1
            self.ablation_diagnostics["pressure_gate_active_evaluation_count"] += 1
            if pressure_active:
                target_can_satisfy = False
                rejection_reason = "pressure_suppressed_unrelated"
                self.ablation_diagnostics["pressure_suppression_block_count"] += 1
                self.ablation_diagnostics["pressure_suppressed_unrelated_interactions"] += 1
            else:
                reason_type = "SAME_STATION_BRIEF_CHECKIN"
                patient_id = -1
                urgency = 0.25
        elif (
            interaction_type == "opportunistic_corridor"
            and close
            and (
                initiator_zone in self.corridor_or_threshold_interaction_zone_ids()
                or target_zone in self.corridor_or_threshold_interaction_zone_ids()
            )
        ):
            self.ablation_diagnostics["pressure_suppression_candidate_count"] += 1
            self.ablation_diagnostics["pressure_gate_active_evaluation_count"] += 1
            if pressure_active:
                target_can_satisfy = False
                rejection_reason = "pressure_suppressed_unrelated"
                self.ablation_diagnostics["pressure_suppression_block_count"] += 1
                self.ablation_diagnostics["pressure_suppressed_unrelated_interactions"] += 1
            else:
                reason_type = "CORRIDOR_PASSING_UPDATE"
                patient_id = -1
                urgency = 0.30

        if not target_can_satisfy:
            reason_type = "NO_ACTION"
        reason_key = (
            pair_key,
            reason_type,
            int(patient_id) if patient_id is not None else -1,
            target.role,
        )
        cooldown = (
            config.PERCEPTION_REASON_LOW_STAKES_COMPLETION_COOLDOWN_SECONDS
            if reason_type in {"SAME_STATION_BRIEF_CHECKIN", "CORRIDOR_PASSING_UPDATE"}
            else config.PERCEPTION_REASON_RECENT_COMPLETION_COOLDOWN_SECONDS
        )
        last_completed = self.last_reason_interaction_by_key.get(reason_key)
        recently_completed = (
            last_completed is not None
            and (self.timestep - last_completed) < int(cooldown)
        )
        if reason_type != "NO_ACTION" and recently_completed:
            rejection_reason = "reason_recently_completed"
        elif reason_type != "NO_ACTION":
            rejection_reason = "actionable_reason"
        return {
            "reason_type": reason_type,
            "patient_id": int(patient_id) if patient_id is not None else -1,
            "target_role": target.role,
            "target_agent_id": target.gid,
            "urgency": urgency,
            "recency_penalty": 1.0 if recently_completed else 0.0,
            "diversion_cost": min(distance / max(config.PERCEPTION_INTERACTION_RADIUS_METERS, 1e-9), 1.0),
            "ed_pressure_index": pressure_index,
            "workload_protected_state_penalty": 0.0 if self._staff_interruptible_for_opportunistic_perception(target) else 1.0,
            "target_can_satisfy": target_can_satisfy,
            "recently_completed": recently_completed,
            "reason_key": reason_key,
            "rejection_reason": rejection_reason,
        }

    def _mark_reason_completed(self, gate_result: Mapping[str, object]) -> None:
        reason_key = gate_result.get("reason_key")
        if isinstance(reason_key, tuple):
            self.last_reason_interaction_by_key[reason_key] = self.timestep
        reason_type = str(gate_result.get("reason_type", "NO_ACTION"))
        if reason_type != "NO_ACTION":
            self.ablation_diagnostics["reason_to_completed_interaction_count"] += 1
            self.ablation_diagnostics[f"reason_completed_{reason_type}"] += 1
            if reason_type == "HIGH_ACUITY_ESCALATION":
                self.ablation_diagnostics["high_acuity_reason_completed_count"] += 1

    def _record_part3_decision_episode(
        self,
        *,
        initiator,
        target,
        percept: Percept,
        reason_info: Mapping[str, object],
        interaction_type: str,
        salience: float,
        probability: float,
        rule_draw: float,
        route_supported: bool,
        visible_staff_count: int,
        eligible_opportunity_count: int,
        attention_rank: int,
        retain_for_export: bool = True,
    ) -> Optional[dict]:
        """Record one sparse, feasible decision point without changing behavior."""

        if not route_supported:
            return None
        if retain_for_export:
            if not self.part3_episode_logging_enabled:
                return None
            if self.timestep < self.part3_episode_logging_start_seconds:
                return None
            if self.part3_max_decision_episodes <= 0:
                return None

        reason_type = str(reason_info.get("reason_type", "NO_ACTION"))
        reason_aliases = {
            "POST_TASK_UPDATE": "coordination_need",
            "SAME_PATIENT_UPDATE": "same_patient_update",
            "HANDOFF_NEED": "coordination_need",
            "BED_FLOW_COORDINATION": "coordination_need",
            "HIGH_ACUITY_ESCALATION": "urgent_escalation",
            "SAME_STATION_BRIEF_CHECKIN": "relational_check_in",
            "CORRIDOR_PASSING_UPDATE": "coordination_need",
        }
        patient_id = int(reason_info.get("patient_id", -1) or -1)
        patient = self.patient_by_id.get(patient_id)
        evidence_id = (
            f"part3:{self.scenario_mode}:{self.condition_spec.name}:seed{self.random_seed}:"
            f"t{self.timestep}:a{initiator.gid}:b{target.gid}:"
            f"{interaction_type}:{reason_type}"
        )
        topic_candidates = self.interaction_engine.behavior_library.topics_for(
            initiator.role,
            self.which_zone(*initiator.position),
        )
        selected_action = "engage" if rule_draw < probability else "defer"
        memory_state_before = []
        if self.part3_cognitive_controller is not None:
            memory_state_before = self.interaction_engine.grounded_part3_memory_context(
                agent_id=int(initiator.gid),
                current_timestamp=int(self.timestep),
                partner_id=int(target.gid),
                patient_context_id=(patient_id if patient_id >= 0 else None),
                reason_type=reason_type,
                interaction_type=interaction_type,
                zone_id=self.which_zone(*initiator.position)
                or "Outside named zones",
                salience_modifiers=(
                    self.part3_cognitive_controller.memory_salience_modifiers_for(
                        int(initiator.gid)
                    )
                ),
                limit=config.MEMORY_RETRIEVAL_LIMIT,
            )
        episode = {
            "episode_schema_version": 2,
            "evidence_id": evidence_id,
            "timestep": self.timestep,
            "fixture_only_not_scientific_data": False,
            "logging_only_rule_behavior_unchanged": True,
            "sampling": {
                "method": "deterministic_sha256_reservoir",
                "window_start_seconds": self.part3_episode_logging_start_seconds,
                "maximum_retained_per_run": self.part3_max_decision_episodes,
            },
            "metadata": {
                "scenario_mode": self.scenario_mode,
                "condition": self.condition_spec.name,
                "seed": self.random_seed,
                "timestep": self.timestep,
                "staff_id": initiator.gid,
                "staff_name": getattr(
                    initiator, "name", f"{initiator.role} {initiator.gid}"
                ),
                "role": initiator.role,
                "partner_id": target.gid,
                "partner_role": target.role,
            },
            "spatial_context": {
                "position": tuple(float(value) for value in initiator.position),
                "target_position": tuple(float(value) for value in target.position),
                "zone": self.which_zone(*initiator.position) or "Outside named zones",
                "distance_m": float(percept.distance_m),
                "mutual_visibility": bool(percept.mutual_visibility),
                "route_supported": True,
            },
            "perception_context": {
                "visible_staff_count": int(visible_staff_count),
                "eligible_staff_opportunity_count": int(
                    eligible_opportunity_count
                ),
                "attention_rank": int(attention_rank),
                "attention_capacity": int(
                    config.PERCEPTION_STAFF_ATTENTION_CAPACITY
                ),
            },
            "workflow_context": {
                "initiator_mode": getattr(initiator, "mode", None),
                "initiator_task": getattr(
                    initiator, "current_task_name", None
                ),
                "initiator_is_task_busy": bool(initiator.is_task_busy()),
                "target_mode": getattr(target, "mode", None),
                "target_task": getattr(target, "current_task_name", None),
                "target_is_task_busy": bool(target.is_task_busy()),
                "external_interruption_allowed": True,
                "opportunity_boundary": "externally_interruptible_staff_contact",
                "patient_id": patient_id if patient_id >= 0 else None,
                "esi_level": getattr(patient, "esi_level", None),
                "patient_current_task": getattr(
                    patient, "current_task_name", None
                ),
                "patient_task_elapsed_seconds": (
                    int(patient.seconds_at_current_task(self.timestep))
                    if patient is not None
                    else None
                ),
                "patient_bed_assigned": (
                    bool(patient.bed_index is not None)
                    if patient is not None
                    else None
                ),
                "patient_zone": (
                    self.which_zone(*patient.position)
                    if patient is not None
                    else None
                ),
                "patient_high_acuity_escalation": (
                    bool(patient.high_acuity_escalation)
                    if patient is not None
                    else None
                ),
                "patient_waiting_for_placement": (
                    bool(patient.waiting_for_placement())
                    if patient is not None
                    else None
                ),
                "active_patient_count": len(self.active_patients),
                "ed_pressure_index": float(
                    reason_info.get("ed_pressure_index", 0.0) or 0.0
                ),
            },
            "decision_context": {
                "interaction_type": interaction_type,
                "abm_reason_type": reason_type,
                "urgency": float(reason_info.get("urgency", 0.0) or 0.0),
                "salience": float(salience),
                "diversion_cost": float(reason_info.get("diversion_cost", 0.0) or 0.0),
                "feasible_actions": ["engage", "defer", "decline"],
                "allowed_reasons": sorted(
                    {
                        reason_aliases.get(reason_type, "coordination_need"),
                        "protect_task_continuity",
                        "insufficient_relevance",
                    }
                ),
                "allowed_topic_families": sorted(set(topic_candidates)),
            },
            "rule_reference": {
                "engagement_probability": float(probability),
                "random_draw": float(rule_draw),
                "selected_action": selected_action,
            },
            "memory_state_before": memory_state_before,
            "event_summary": (
                f"{initiator.role} had a feasible {reason_type.lower()} opportunity with "
                f"{target.role} in {self.which_zone(*initiator.position) or 'an uncoded area'}."
            ),
        }

        if not retain_for_export:
            return episode

        # Retain an across-shift deterministic sample without consuming the
        # simulation RNG. Logging therefore cannot alter the rule trajectory.
        self.part3_decision_episode_candidates_seen += 1
        priority = int.from_bytes(
            hashlib.sha256(evidence_id.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        if len(self.part3_decision_episode_log) < self.part3_max_decision_episodes:
            self.part3_decision_episode_log.append(episode)
            self._part3_decision_episode_priorities[evidence_id] = priority
            return episode

        worst_index, worst_episode = max(
            enumerate(self.part3_decision_episode_log),
            key=lambda item: self._part3_decision_episode_priorities[
                item[1]["evidence_id"]
            ],
        )
        worst_id = str(worst_episode["evidence_id"])
        if priority < self._part3_decision_episode_priorities[worst_id]:
            self.part3_decision_episode_log[worst_index] = episode
            del self._part3_decision_episode_priorities[worst_id]
            self._part3_decision_episode_priorities[evidence_id] = priority
        return episode

    def _log_perception_staff_interactions(self) -> None:
        """Perception-to-action loop for opportunistic staff-staff encounters."""

        if not config.PERCEPTION_BASED_BASELINE_ENABLED:
            return
        if self.timestep % max(int(config.PERCEPTION_OPPORTUNITY_CHECK_INTERVAL_SECONDS), 1) != 0:
            return

        staff_lookup = self.staff_by_id()
        stationary_modes = {"idle", "waiting_for_doctor", "using_secondary_station", "post_task_station_check"}
        moving_modes = {
            "returning_home",
            "moving_to_secondary_station",
            "moving_to_post_task_station_check",
            "patrolling",
            "moving_to_doctor",
            "moving_to_patient",
        }

        for initiator in self.staff_agents:
            if not self._staff_interruptible_for_opportunistic_perception(initiator):
                self.ablation_diagnostics["protected_state_suppression_count"] += 1
                continue

            bundle = self.perception.build_bundle(initiator, self, [])
            visible_by_id = {int(percept.entity_id) for percept in bundle.visible_agents if isinstance(percept.entity_id, int)}
            if getattr(initiator, "mode", None) in stationary_modes:
                for target in self.staff_agents:
                    if target.gid == initiator.gid or target.gid in visible_by_id:
                        continue
                    if getattr(target, "mode", None) not in moving_modes:
                        continue
                    distance = initiator.distance_to_agent(target)
                    if distance > config.PERCEPTION_INTERACTION_RADIUS_METERS:
                        continue
                    if not self._staff_interruptible_for_opportunistic_perception(target):
                        continue
                    if not self.perception._has_visibility(initiator.position, target.position):
                        continue
                    salience = self.perception.score_salience(initiator, target, self)
                    if salience < config.SALIENCE_MIN_THRESHOLD:
                        continue
                    self.ablation_diagnostics["stationary_field_of_regard_candidate_count"] += 1
                    bundle.visible_agents.append(
                        Percept(
                            entity_id=target.gid,
                            entity_type="agent",
                            role=target.role,
                            zone_id=self.which_zone(*target.position),
                            distance_m=distance,
                            mutual_visibility=(
                                self.perception._within_fov(target, initiator.position)
                                and self.perception._has_visibility(target.position, initiator.position)
                            ),
                            line_of_sight_length_m=distance,
                            salience=salience,
                            position=target.position,
                        )
                    )
            raw_staff_percepts = []
            for percept in bundle.visible_agents:
                target = staff_lookup.get(int(percept.entity_id)) if isinstance(percept.entity_id, int) else None
                if target is None or target.gid == initiator.gid or getattr(target, "role", None) == "Patient":
                    continue
                raw_staff_percepts.append((percept, target))
                self._increment_related_counter("raw_visible_staff", initiator, target)
            self.ablation_diagnostics["raw_visible_staff_percepts_count"] += len(raw_staff_percepts)
            self.ablation_diagnostics["visible_staff_percepts_count"] += len(raw_staff_percepts)
            staff_percepts = []
            for percept, target in raw_staff_percepts:
                if percept.distance_m > config.PERCEPTION_INTERACTION_RADIUS_METERS:
                    self.ablation_diagnostics["rejected_by_visibility_radius"] += 1
                    continue
                if not self._staff_interruptible_for_opportunistic_perception(target):
                    self.ablation_diagnostics["rejected_by_protected_state"] += 1
                    continue
                salience = self._visibility_candidate_salience(initiator, target, percept)
                if salience < config.PERCEPTION_STAFF_SALIENCE_THRESHOLD:
                    self.ablation_diagnostics["rejected_by_salience_threshold"] += 1
                    continue
                midpoint_position = self._staff_contact_position(initiator, target)
                midpoint_zone = self.which_zone(*midpoint_position)
                interaction_type = self._perceived_staff_interaction_type(
                    initiator,
                    target,
                    midpoint_zone,
                    initiator.distance_to_agent(target),
                )
                reason_info = self._visible_staff_reason(
                    initiator,
                    target,
                    interaction_type,
                    initiator.distance_to_agent(target),
                )
                reason_type = str(reason_info.get("reason_type", "NO_ACTION"))
                self.ablation_diagnostics[f"reason_type_{reason_type}"] += 1
                if reason_type == "NO_ACTION":
                    self.ablation_diagnostics["visible_staff_no_reason_count"] += 1
                    self.ablation_diagnostics["reason_gate_rejected_count"] += 1
                    self.ablation_diagnostics[f"reason_gate_rejection_reason_{reason_info.get('rejection_reason', 'no_action')}"] += 1
                    continue
                self.ablation_diagnostics["visible_staff_with_reason_count"] += 1
                if bool(reason_info.get("recently_completed", False)):
                    self.ablation_diagnostics["reason_gate_rejected_count"] += 1
                    self.ablation_diagnostics["reason_gate_rejection_reason_reason_recently_completed"] += 1
                    continue
                self.ablation_diagnostics["actionable_reason_opportunity_count"] += 1
                staff_percepts.append((salience + (float(reason_info.get("urgency", 0.0)) * 0.20), percept, target, reason_info, interaction_type))

            self.ablation_diagnostics["eligible_visible_staff_opportunity_count"] += len(staff_percepts)
            for _salience, _percept, eligible_target, _reason_info, _interaction_type in staff_percepts:
                self._increment_related_counter("eligible_visible_staff_opportunity", initiator, eligible_target)
            staff_percepts.sort(key=lambda item: item[0], reverse=True)
            attended = staff_percepts[: max(int(config.PERCEPTION_STAFF_ATTENTION_CAPACITY), 0)]
            if len(staff_percepts) > len(attended):
                self.ablation_diagnostics["rejected_by_attention_capacity"] += len(staff_percepts) - len(attended)

            for attention_rank, (
                salience,
                percept,
                target,
                reason_info,
                interaction_type,
            ) in enumerate(attended, start=1):
                if self._agent_has_pending_staff_interaction_role(initiator) or self._agent_has_pending_staff_interaction_role(target):
                    self.ablation_diagnostics["rejected_by_cooldown"] += 1
                    continue
                pair_key = tuple(sorted((initiator.gid, target.gid)))
                if pair_key in self.currently_interacting_pairs:
                    last_timestamp = self.last_perceived_staff_interaction_by_pair.get(pair_key)
                    if last_timestamp is None or (self.timestep - last_timestamp) < max(
                        int(config.PERCEPTION_OPPORTUNITY_CHECK_INTERVAL_SECONDS), 1
                    ):
                        self.ablation_diagnostics["rejected_by_cooldown"] += 1
                        continue
                last_timestamp = self.last_perceived_staff_interaction_by_pair.get(pair_key)
                if (
                    last_timestamp is not None
                    and (self.timestep - last_timestamp) < config.PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS
                ):
                    self.ablation_diagnostics["rejected_by_cooldown"] += 1
                    continue

                distance = initiator.distance_to_agent(target)
                context_position = self._staff_contact_position(initiator, target)
                context_zone = self.which_zone(*context_position)
                probability = self._perceived_staff_interaction_probability(
                    interaction_type,
                    initiator,
                    target,
                    context_zone,
                    reason_type=str(reason_info.get("reason_type", "")),
                    urgency=float(reason_info.get("urgency", 0.0) or 0.0),
                )
                stationary_to_moving = (
                    getattr(initiator, "mode", None) in stationary_modes
                    and getattr(target, "mode", None) in moving_modes
                )
                if stationary_to_moving:
                    self.ablation_diagnostics["stationary_visible_moving_candidate_count"] += 1
                    probability = min(probability * 5.0, 0.45)

                gate_result = {
                    "accepted": True,
                    "category": "accepted_perceived_staff",
                    "reason": "visible_staff_candidate_passed_salience_attention",
                    "perception_checked": True,
                    "salience_score": salience,
                    "reason_type": reason_info.get("reason_type"),
                    "reason_key": reason_info.get("reason_key"),
                    "reason_patient_id": reason_info.get("patient_id"),
                    "reason_urgency": reason_info.get("urgency"),
                    "diversion_cost": reason_info.get("diversion_cost"),
                    "approach_intent_created": distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS,
                    "stationary_initiated_to_moving": stationary_to_moving,
                }
                rule_draw = self.random.random()
                selected_action = "engage" if rule_draw < probability else "defer"
                route_supported = None
                part3_decision = None
                part3_controller_active = bool(
                    self.part3_cognitive_controller is not None
                    and self.part3_cognitive_controller.active_at(self.timestep)
                )
                if part3_controller_active:
                    route_supported = (
                        distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
                        or self._route_supported(initiator.position, target.position)
                    )
                    if not route_supported:
                        self.ablation_diagnostics["rejected_by_unreachable"] += 1
                        continue
                    episode = self._record_part3_decision_episode(
                        initiator=initiator,
                        target=target,
                        percept=percept,
                        reason_info=reason_info,
                        interaction_type=interaction_type,
                        salience=salience,
                        probability=probability,
                        rule_draw=rule_draw,
                        route_supported=True,
                        visible_staff_count=len(raw_staff_percepts),
                        eligible_opportunity_count=len(staff_percepts),
                        attention_rank=attention_rank,
                        retain_for_export=self.part3_episode_logging_enabled,
                    )
                    if episode is None:
                        episode = self._record_part3_decision_episode(
                            initiator=initiator,
                            target=target,
                            percept=percept,
                            reason_info=reason_info,
                            interaction_type=interaction_type,
                            salience=salience,
                            probability=probability,
                            rule_draw=rule_draw,
                            route_supported=True,
                            visible_staff_count=len(raw_staff_percepts),
                            eligible_opportunity_count=len(staff_percepts),
                            attention_rank=attention_rank,
                            retain_for_export=False,
                        )
                    if episode is None:
                        raise RuntimeError("Unable to construct a route-supported Part 3 episode")
                    part3_decision = self.part3_cognitive_controller.decide_episode(episode)
                    selected_action = str(part3_decision.selected_action)
                    gate_result.update(
                        {
                            "part3_decision_id": part3_decision.decision_id,
                            "part3_evidence_id": part3_decision.evidence_id,
                            "part3_persona_id": part3_decision.persona_id,
                            "part3_selected_action": part3_decision.selected_action,
                            "part3_selected_reason": part3_decision.selected_reason,
                            "part3_topic_family": part3_decision.topic_family,
                            "part3_policy_name": part3_decision.policy_name,
                            "part3_decision_fallback": part3_decision.was_fallback,
                            "part3_decision_output_contract": "categorical_causal_v1",
                        }
                    )
                    self.ablation_diagnostics["part3_closed_loop_decision_count"] += 1
                    self.ablation_diagnostics[
                        f"part3_closed_loop_action_{selected_action}"
                    ] += 1

                if selected_action != "engage":
                    if (
                        not part3_controller_active
                        and self.part3_episode_logging_enabled
                    ):
                        route_supported = (
                            distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
                            or self._route_supported(initiator.position, target.position)
                        )
                        self._record_part3_decision_episode(
                            initiator=initiator,
                            target=target,
                            percept=percept,
                            reason_info=reason_info,
                            interaction_type=interaction_type,
                            salience=salience,
                            probability=probability,
                            rule_draw=rule_draw,
                            route_supported=route_supported,
                            visible_staff_count=len(raw_staff_percepts),
                            eligible_opportunity_count=len(staff_percepts),
                            attention_rank=attention_rank,
                        )
                    self.ablation_diagnostics["missed_perceived_staff_opportunity_count"] += 1
                    if str(reason_info.get("reason_type", "")) == "HIGH_ACUITY_ESCALATION":
                        self.ablation_diagnostics["high_acuity_reason_missed_count"] += 1
                    self.log_missed_opportunity(
                        agent=initiator,
                        partner=target,
                        reason=(
                            f"part3_cognitive_{selected_action}"
                            if part3_controller_active
                            else "visible_salient_candidate_not_initiated"
                        ),
                        position=initiator.position,
                        perception_state={
                            "distance_m": distance,
                            "salience_score": salience,
                            "mutual_visibility": percept.mutual_visibility,
                            "interaction_type": interaction_type,
                            "part3_decision_id": gate_result.get("part3_decision_id"),
                            "part3_persona_id": gate_result.get("part3_persona_id"),
                            "part3_selected_reason": gate_result.get(
                                "part3_selected_reason"
                            ),
                            "part3_topic_family": gate_result.get(
                                "part3_topic_family"
                            ),
                        },
                        task_state=getattr(initiator, "mode", None),
                    )
                    continue
                if interaction_type == "opportunistic_corridor":
                    self.ablation_diagnostics["accepted_corridor_opportunistic"] += 1
                self.ablation_diagnostics["selected_visible_staff_opportunity_count"] += 1
                if (
                    interaction_type == "opportunistic_station"
                    and context_zone in self.station_interaction_zone_ids()
                    and self._station_same_dwell_repeat_suppressed(initiator, target, context_zone)
                ):
                    continue

                if not self._staff_interruptible_for_opportunistic_perception(target):
                    self.ablation_diagnostics["rejected_by_protected_state"] += 1
                    continue
                if not part3_controller_active:
                    route_supported = (
                        distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
                        or self._route_supported(initiator.position, target.position)
                    )
                    if not route_supported:
                        self.ablation_diagnostics["rejected_by_unreachable"] += 1
                        continue
                    self._record_part3_decision_episode(
                        initiator=initiator,
                        target=target,
                        percept=percept,
                        reason_info=reason_info,
                        interaction_type=interaction_type,
                        salience=salience,
                        probability=probability,
                        rule_draw=rule_draw,
                        route_supported=route_supported,
                        visible_staff_count=len(raw_staff_percepts),
                        eligible_opportunity_count=len(staff_percepts),
                        attention_rank=attention_rank,
                    )

                if distance <= config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
                    immediate_gate = dict(gate_result)
                    immediate_gate.update({
                        "category": "accepted_perceived_staff_close_contact",
                        "reason": "visible_staff_candidate_already_in_close_physical_contact",
                        "interaction_source": "opportunistic_perception_interruption",
                        "initial_opportunity_distance_m": distance,
                        "initial_nonproximate": False,
                        "force_interaction_decision": True,
                    })
                    before_count = len(self.interaction_log)
                    self.log_interaction(
                        agent_1=initiator,
                        agent_2=target,
                        position=self._staff_contact_position(initiator, target),
                        interaction_type=interaction_type,
                        reason_for_interaction="perception_salience_attention_close_contact",
                        perception_gate_result=immediate_gate,
                    )
                    if len(self.interaction_log) > before_count:
                        self.currently_interacting_pairs.add(pair_key)
                        self.last_perceived_staff_interaction_by_pair[pair_key] = self.timestep
                        if "station" in interaction_type:
                            self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep
                            self._mark_station_same_dwell_episode(initiator, target, context_zone)
                        if "corridor" in interaction_type:
                            self.last_opportunistic_corridor_interaction_by_pair[pair_key] = self.timestep
                        self.ablation_diagnostics["perceived_staff_interaction_count"] += 1
                        self.ablation_diagnostics["immediate_close_logged_count"] += 1
                        self._mark_reason_completed(immediate_gate)
                        self._increment_related_counter(
                            "completed",
                            initiator,
                            target,
                            self._staff_contact_position(initiator, target),
                        )
                else:
                    self.ablation_diagnostics["actionable_reason_to_approach_count"] += 1
                    intent = self._create_staff_interaction_intent(
                        initiator,
                        target,
                        distance=distance,
                        interaction_type=interaction_type,
                        gate_result=gate_result,
                        salience=salience,
                        stationary_initiated_to_moving=stationary_to_moving,
                    )
                    if gate_result["stationary_initiated_to_moving"]:
                        self.ablation_diagnostics["stationary_initiated_to_moving_count"] += 1
                    if len(self.perception_event_examples) < config.PERCEPTION_MAX_EXAMPLES:
                        self.perception_event_examples.append({
                            "timestep": self.timestep,
                            "intent_id": intent.intent_id,
                            "initiator": getattr(initiator, "name", initiator.gid),
                            "target": getattr(target, "name", target.gid),
                            "initiator_role": initiator.role,
                            "target_role": target.role,
                            "initiator_mode": getattr(initiator, "mode", None),
                            "target_mode": getattr(target, "mode", None),
                            "initial_distance_m": round(distance, 3),
                            "visible": True,
                            "mutual_visible": bool(percept.mutual_visibility),
                            "salience_score": round(float(salience), 3),
                            "reason_selected": "top visible candidate became pending embodied interaction intent",
                            "approach_or_pause_occurred": True,
                            "interaction_type": interaction_type,
                            "topic": None,
                            "location": self.which_zone(*initiator.position) or "Outside named zones",
                            "cooldown_state": "new_or_cooldown_elapsed",
                            "status": intent.status,
                        })
                    break
                if gate_result["stationary_initiated_to_moving"]:
                    self.ablation_diagnostics["stationary_initiated_to_moving_count"] += 1
                if len(self.perception_event_examples) < config.PERCEPTION_MAX_EXAMPLES:
                    self.perception_event_examples.append({
                        "timestep": self.timestep,
                        "initiator": getattr(initiator, "name", initiator.gid),
                        "target": getattr(target, "name", target.gid),
                        "initiator_role": initiator.role,
                        "target_role": target.role,
                        "initiator_mode": getattr(initiator, "mode", None),
                        "target_mode": getattr(target, "mode", None),
                        "initial_distance_m": round(distance, 3),
                        "visible": True,
                        "mutual_visible": bool(percept.mutual_visibility),
                        "salience_score": round(float(salience), 3),
                        "reason_selected": "top visible candidate passed salience and attention",
                        "approach_or_pause_occurred": bool(gate_result["approach_intent_created"]),
                        "interaction_type": interaction_type,
                        "topic": self.interaction_log[-1].get("topic") if self.interaction_log else None,
                        "location": self.which_zone(*initiator.position) or "Outside named zones",
                        "cooldown_state": "new_or_cooldown_elapsed",
                    })
                break

    def _log_doctor_coreview_interactions(self) -> None:
        """Log rare doctor-doctor co-review moments without changing workflow."""

        eligible_modes = {"idle", "returning_home", "patrolling"}
        doctors = [
            doctor
            for doctor in self.doctors
            if not doctor.is_task_busy()
            and not getattr(doctor, "senior_oversight_only", False)
            and doctor.mode in eligible_modes
            and self.which_zone(*doctor.position) == "COCPIT"
            and self._station_interaction_ready(doctor, "COCPIT")
        ]
        for left_index, left_doctor in enumerate(doctors):
            for right_doctor in doctors[left_index + 1:]:
                if left_doctor.distance_to_agent(right_doctor) > config.STATION_COORDINATION_DISTANCE_METERS:
                    continue
                pair_key = tuple(sorted((left_doctor.gid, right_doctor.gid)))
                if pair_key in self.currently_interacting_pairs:
                    continue
                last_timestamp = self.last_doctor_coreview_by_pair.get(pair_key)
                if (
                    last_timestamp is not None
                    and (self.timestep - last_timestamp) < config.DOCTOR_COREVIEW_COOLDOWN_SECONDS
                ):
                    continue
                if self.random.random() >= config.DOCTOR_COREVIEW_PROBABILITY:
                    continue
                before_count = len(self.interaction_log)
                self.log_interaction(
                    agent_1=left_doctor,
                    agent_2=right_doctor,
                    position=self._embodied_contact_position(left_doctor, right_doctor),
                    interaction_type="doctor_doctor_coreview",
                    reason_for_interaction="brief_case_review_or_documentation_check",
                )
                if len(self.interaction_log) > before_count:
                    self.currently_interacting_pairs.add(pair_key)
                    self.last_doctor_coreview_by_pair[pair_key] = self.timestep
                    self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep

    def _log_opportunistic_corridor_interactions(self) -> None:
        transit_modes = {
            "moving_to_patient",
            "returning_home",
            "moving_to_doctor",
            "leading_doctor_to_patient",
            "moving_to_secondary_station",
            "moving_to_post_task_station_check",
            "patrolling",
        }
        corridor_available_modes = transit_modes | {
            "idle",
            "waiting_for_doctor",
            "using_secondary_station",
            "post_task_station_check",
        }

        for left_index, left_agent in enumerate(self.staff_agents):
            if getattr(left_agent, "senior_oversight_only", False):
                continue
            if left_agent.is_task_busy() and not self._staff_locally_communicative(left_agent):
                continue
            if left_agent.mode not in transit_modes and not self._staff_locally_communicative(left_agent):
                continue

            for right_agent in self.staff_agents[left_index + 1:]:
                if getattr(right_agent, "senior_oversight_only", False):
                    continue
                if right_agent.is_task_busy() and not self._staff_locally_communicative(right_agent):
                    continue
                if right_agent.mode not in corridor_available_modes and not self._staff_locally_communicative(right_agent):
                    continue
                self.ablation_diagnostics["eligible_encounters_considered"] += 1
                contact_position = self._embodied_contact_position(left_agent, right_agent)
                zone_id = self.which_zone(*contact_position)
                if zone_id is None or zone_id not in self.corridor_or_threshold_interaction_zone_ids():
                    self.ablation_diagnostics["rejected_by_distance"] += 1
                    continue
                self.ablation_diagnostics["corridor_close_copresence_opportunity_count"] += 1
                if left_agent.distance_to_agent(right_agent) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
                    self.ablation_diagnostics["corridor_close_copresence_rejected_count"] += 1
                    self.ablation_diagnostics["corridor_rejection_reason_not_close"] += 1
                    continue
                gate_result = self._perception_gate(
                    left_agent,
                    right_agent,
                    interaction_type="opportunistic_corridor",
                    position=contact_position,
                )
                if not gate_result["accepted"]:
                    self.ablation_diagnostics[str(gate_result["category"])] += 1
                    self.ablation_diagnostics["corridor_close_copresence_rejected_count"] += 1
                    self.ablation_diagnostics[f"corridor_rejection_reason_{gate_result['category']}"] += 1
                    self.log_missed_opportunity(
                        agent=left_agent,
                        partner=right_agent,
                        reason=(
                            "visible_but_not_attended"
                            if gate_result["category"] == "rejected_fov_soft_probability"
                            else "partner_busy_or_moving_away"
                        ),
                        position=contact_position,
                        perception_state={
                            "encounter": "corridor",
                            "gate_category": gate_result["category"],
                            "gate_reason": gate_result["reason"],
                        },
                        task_state=left_agent.mode,
                    )
                    continue

                pair_key = tuple(sorted((left_agent.gid, right_agent.gid)))
                if pair_key in self.currently_interacting_pairs:
                    self.ablation_diagnostics["rejected_by_cooldown"] += 1
                    self.ablation_diagnostics["corridor_close_copresence_rejected_count"] += 1
                    self.ablation_diagnostics["corridor_rejection_reason_cooldown"] += 1
                    continue
                last_timestamp = self.last_opportunistic_corridor_interaction_by_pair.get(pair_key)
                if (
                    last_timestamp is not None
                    and (self.timestep - last_timestamp)
                    < config.OPPORTUNISTIC_CORRIDOR_INTERACTION_COOLDOWN_SECONDS
                ):
                    self.ablation_diagnostics["rejected_by_cooldown"] += 1
                    self.ablation_diagnostics["corridor_close_copresence_rejected_count"] += 1
                    self.ablation_diagnostics["corridor_rejection_reason_cooldown"] += 1
                    continue

                probability = self._hcw_hcw_interaction_probability(
                    left_agent,
                    right_agent,
                    zone_id,
                    config.OPPORTUNISTIC_CORRIDOR_INTERACTION_PROBABILITY,
                )
                if self.random.random() >= probability:
                    self.ablation_diagnostics["rejected_by_probability_or_rule_decision"] += 1
                    self.ablation_diagnostics["corridor_close_copresence_rejected_count"] += 1
                    self.ablation_diagnostics["corridor_rejection_reason_probability"] += 1
                    self.log_missed_opportunity(
                        agent=left_agent,
                        partner=right_agent,
                        reason="visible_but_not_attended",
                        position=contact_position,
                        perception_state={"encounter": "corridor", "mutual_visibility": True},
                        task_state=left_agent.mode,
                    )
                    continue

                before_count = len(self.interaction_log)
                self.log_interaction(
                    agent_1=left_agent,
                    agent_2=right_agent,
                    position=contact_position,
                    interaction_type="opportunistic_corridor",
                    reason_for_interaction="corridor_stop_and_talk_micro_interaction",
                    perception_gate_result=gate_result,
                )
                if len(self.interaction_log) > before_count:
                    self.currently_interacting_pairs.add(pair_key)
                    self.last_opportunistic_corridor_interaction_by_pair[pair_key] = self.timestep

    def _corridor_encounter_midpoint(self, left_agent, right_agent) -> Optional[tuple[float, float]]:
        left_previous = left_agent.trail[-2] if len(left_agent.trail) >= 2 else left_agent.position
        right_previous = right_agent.trail[-2] if len(right_agent.trail) >= 2 else right_agent.position

        fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
        left_points = [
            (
                left_previous[0] + ((left_agent.position[0] - left_previous[0]) * fraction),
                left_previous[1] + ((left_agent.position[1] - left_previous[1]) * fraction),
            )
            for fraction in fractions
        ]
        right_points = [
            (
                right_previous[0] + ((right_agent.position[0] - right_previous[0]) * fraction),
                right_previous[1] + ((right_agent.position[1] - right_previous[1]) * fraction),
            )
            for fraction in fractions
        ]

        closest_pair = min(
            (
                (left_point, right_point)
                for left_point in left_points
                for right_point in right_points
            ),
            key=lambda pair: math.dist(pair[0], pair[1]),
        )
        if math.dist(closest_pair[0], closest_pair[1]) > config.CORRIDOR_ENCOUNTER_DISTANCE_METERS:
            return None

        return (
            (closest_pair[0][0] + closest_pair[1][0]) / 2.0,
            (closest_pair[0][1] + closest_pair[1][1]) / 2.0,
        )

    def _bedside_cotask_interaction_type(self, left_agent, right_agent) -> str:
        roles = {getattr(left_agent, "role", ""), getattr(right_agent, "role", "")}
        if roles == {"Doctor", "Nurse"}:
            return "nurse_doctor_bedside_coordination"
        if roles == {"Nurse"}:
            return "nurse_nurse_bedside_assist"
        if roles == {"Doctor"}:
            return "doctor_doctor_bedside_review"
        return "bedside_cotask_update"

    def _log_bedside_cotask_interactions(self) -> None:
        for left_index, left_agent in enumerate(self.staff_agents):
            if getattr(left_agent, "senior_oversight_only", False):
                continue
            for right_agent in self.staff_agents[left_index + 1:]:
                if getattr(right_agent, "senior_oversight_only", False):
                    continue
                patient = self._co_task_interaction_allowed(left_agent, right_agent)
                if patient is None:
                    continue
                pair_key = tuple(sorted((left_agent.gid, right_agent.gid)))
                cooldown_key = (pair_key[0], pair_key[1], patient.gid)
                last_timestamp = self.last_bedside_cotask_interaction_by_pair_patient.get(cooldown_key)
                if (
                    last_timestamp is not None
                    and (self.timestep - last_timestamp) < config.BEDSIDE_COTASK_INTERACTION_COOLDOWN_SECONDS
                ):
                    self.ablation_diagnostics["cotask_interaction_suppressed_by_cooldown"] += 1
                    continue
                task_label = "|".join(
                    sorted(
                        str(task)
                        for task in (
                            getattr(left_agent, "current_task_name", None),
                            getattr(right_agent, "current_task_name", None),
                        )
                        if task is not None
                    )
                ) or "shared_patient_context"
                logged_key = (pair_key[0], pair_key[1], patient.gid, task_label)
                if logged_key in self.bedside_cotask_logged_keys:
                    self.ablation_diagnostics["cotask_interaction_suppressed_by_cooldown"] += 1
                    continue
                if self.random.random() >= config.BEDSIDE_COTASK_INTERACTION_PROBABILITY:
                    continue
                interaction_type = self._bedside_cotask_interaction_type(left_agent, right_agent)
                contact_position = self._embodied_contact_position(left_agent, right_agent)
                gate_result = self._perception_gate(
                    left_agent,
                    right_agent,
                    interaction_type=interaction_type,
                    position=contact_position,
                    task_name=task_label,
                )
                if not gate_result["accepted"]:
                    self.ablation_diagnostics[str(gate_result["category"])] += 1
                    continue
                gate_result.update({
                    "interaction_source": "bedside_cotask_update",
                    "force_interaction_decision": True,
                })
                before_count = len(self.interaction_log)
                self.log_interaction(
                    agent_1=left_agent,
                    agent_2=right_agent,
                    position=contact_position,
                    interaction_type=interaction_type,
                    task_name=task_label,
                    reason_for_interaction="bedside_cotask_update",
                    perception_gate_result=gate_result,
                )
                if len(self.interaction_log) > before_count:
                    self.currently_interacting_pairs.add(pair_key)
                    self.last_bedside_cotask_interaction_by_pair_patient[cooldown_key] = self.timestep
                    self.bedside_cotask_logged_keys.add(logged_key)
                    self.ablation_diagnostics["bedside_cotask_staff_interaction_count"] += 1
                    self.ablation_diagnostics[f"bedside_cotask_role_pair_{self._role_pair_key(left_agent, right_agent)}"] += 1
                    zone_id = self.which_zone(*contact_position) or "Outside named zones"
                    self.ablation_diagnostics[f"bedside_cotask_zone_{zone_id}"] += 1

    def _log_task_transition_updates(self) -> None:
        """Convert recent task completions into co-present transition updates.

        The update is only logged if the completing clinician later becomes
        physically close to a relevant colleague in a station or corridor
        context. This keeps task-transition communication tied to movement and
        co-presence rather than adding synthetic bedside events.
        """

        if not self.pending_task_transition_updates:
            return
        staff_lookup = self.staff_by_id()
        retained_updates: List[Dict[str, object]] = []
        available_modes = {
            "idle",
            "returning_home",
            "waiting_for_doctor",
            "moving_to_doctor",
            "leading_doctor_to_patient",
            "moving_to_secondary_station",
            "using_secondary_station",
            "moving_to_post_task_station_check",
            "post_task_station_check",
            "patrolling",
        }
        valid_zones = (
            self.corridor_or_threshold_interaction_zone_ids()
            | self.station_interaction_zone_ids()
            | set(config.STATION_ADJACENT_ZONE_IDS)
        )

        for update in self.pending_task_transition_updates:
            if self.timestep > int(update.get("expires_at", self.timestep)):
                self.ablation_diagnostics["task_update_episode_expired_count"] += 1
                continue
            initiator = staff_lookup.get(int(update.get("initiator_id", -1)))
            if initiator is None:
                continue
            if (
                (initiator.is_task_busy() and not self._staff_locally_communicative(initiator))
                or (initiator.mode not in available_modes and not self._staff_locally_communicative(initiator))
            ):
                retained_updates.append(update)
                continue

            preferred_roles = list(update.get("preferred_roles", []))
            candidates = [
                partner for partner in self.staff_agents
                if partner.gid != initiator.gid
                and partner.role in preferred_roles
                and not getattr(partner, "senior_oversight_only", False)
                and (not partner.is_task_busy() or self._staff_locally_communicative(partner))
                and (partner.mode in available_modes or self._staff_locally_communicative(partner))
                and initiator.distance_to_agent(partner) <= config.TASK_TRANSITION_UPDATE_DISTANCE_METERS
            ]
            if not candidates:
                retained_updates.append(update)
                continue

            candidates.sort(
                key=lambda partner: (
                    preferred_roles.index(partner.role) if partner.role in preferred_roles else 99,
                    initiator.distance_to_agent(partner),
                )
            )
            partner = candidates[0]
            pair_key = tuple(sorted((initiator.gid, partner.gid)))
            episode_key = self._task_transition_update_episode_key(update, partner)
            if episode_key in self.logged_task_transition_update_episode_keys:
                self.ablation_diagnostics["task_transition_update_episode_suppressed_count"] += 1
                continue
            last_timestamp = self.last_task_transition_update_by_pair.get(pair_key)
            if (
                last_timestamp is not None
                and (self.timestep - last_timestamp) < config.TASK_TRANSITION_UPDATE_COOLDOWN_SECONDS
            ):
                retained_updates.append(update)
                continue
            if self.random.random() >= config.TASK_TRANSITION_UPDATE_INTERACTION_PROBABILITY:
                retained_updates.append(update)
                continue

            contact_position = self._embodied_contact_position(initiator, partner)
            zone_id = self.which_zone(*contact_position)
            if zone_id not in valid_zones:
                retained_updates.append(update)
                continue
            if initiator.distance_to_agent(partner) > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS:
                corridor_context = (
                    zone_id in self.corridor_or_threshold_interaction_zone_ids()
                    or self.which_zone(*partner.position) in self.corridor_or_threshold_interaction_zone_ids()
                    or getattr(initiator, "mode", None) in {
                        "returning_home",
                        "moving_to_doctor",
                        "leading_doctor_to_patient",
                        "moving_to_secondary_station",
                        "moving_to_post_task_station_check",
                        "patrolling",
                    }
                    or getattr(partner, "mode", None) in {
                        "returning_home",
                        "moving_to_doctor",
                        "leading_doctor_to_patient",
                        "moving_to_secondary_station",
                        "moving_to_post_task_station_check",
                        "patrolling",
                    }
                )
                if corridor_context and not (
                    self._agent_has_pending_staff_interaction_role(initiator)
                    or self._agent_has_pending_staff_interaction_role(partner)
                ):
                    walking_gate = self._perception_gate(
                        initiator,
                        partner,
                        interaction_type="task_transition_update",
                        position=contact_position,
                        task_name=str(update.get("task_name", "")),
                    )
                    if (
                        walking_gate["accepted"]
                        and self._staff_interruptible_for_opportunistic_perception(initiator)
                        and self._staff_interruptible_for_opportunistic_perception(partner)
                        and self._route_supported(initiator.position, partner.position)
                    ):
                        walking_gate.update({
                            "interaction_source": "walking_task_transition_update",
                            "approach_intent_created": True,
                            "initial_opportunity_distance_m": initiator.distance_to_agent(partner),
                            "initial_nonproximate": True,
                            "force_interaction_decision": True,
                            "task_update_episode_key": update.get("episode_key"),
                            "task_update_patient_id": int(update.get("patient_id", -1)),
                            "task_update_role_pair": self._role_pair_key(initiator, partner),
                        })
                        self._create_staff_interaction_intent(
                            initiator,
                            partner,
                            distance=initiator.distance_to_agent(partner),
                            interaction_type="task_transition_update",
                            gate_result=walking_gate,
                            salience=0.5,
                            stationary_initiated_to_moving=False,
                            interaction_source="walking_task_transition_update",
                        )
                        self.logged_task_transition_update_episode_keys.add(episode_key)
                        self.ablation_diagnostics["task_transition_update_new_episode_count"] += 1
                        self.ablation_diagnostics[
                            f"task_transition_update_role_pair_{self._role_pair_key(initiator, partner)}"
                        ] += 1
                        self.ablation_diagnostics[
                            f"task_transition_update_patient_context_{int(update.get('patient_id', -1))}"
                        ] += 1
                        continue
                self.ablation_diagnostics["task_transition_update_not_close_count"] += 1
                self.ablation_diagnostics["missed_not_close_opportunity_count"] += 1
                self.log_missed_opportunity(
                    agent=initiator,
                    partner=partner,
                    reason="task_transition_update_not_in_close_physical_contact",
                    position=contact_position,
                    patient=self.patient_by_id.get(int(update.get("patient_id", -1))),
                    perception_state={
                        "interaction_type": "task_transition_update",
                        "distance_m": initiator.distance_to_agent(partner),
                        "close_threshold_m": config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS,
                        "candidate_radius_m": config.TASK_TRANSITION_UPDATE_DISTANCE_METERS,
                    },
                    task_state=getattr(initiator, "mode", None),
                )
                retained_updates.append(update)
                continue

            gate_result = self._perception_gate(
                initiator,
                partner,
                interaction_type="task_transition_update",
                position=contact_position,
                task_name=str(update.get("task_name", "")),
            )
            if not gate_result["accepted"]:
                self.ablation_diagnostics[str(gate_result["category"])] += 1
                retained_updates.append(update)
                continue

            before_count = len(self.interaction_log)
            self.log_interaction(
                agent_1=initiator,
                agent_2=partner,
                position=contact_position,
                interaction_type="task_transition_update",
                task_name=str(update.get("task_name", "")),
                reason_for_interaction="post_task_transition_update",
                perception_gate_result=gate_result,
            )
            if len(self.interaction_log) > before_count:
                self.currently_interacting_pairs.add(pair_key)
                self.logged_task_transition_update_episode_keys.add(episode_key)
                self.ablation_diagnostics["task_transition_update_new_episode_count"] += 1
                self.ablation_diagnostics["task_update_episode_completed_count"] += 1
                self.ablation_diagnostics[
                    f"task_transition_update_role_pair_{self._role_pair_key(initiator, partner)}"
                ] += 1
                self.ablation_diagnostics[
                    f"task_update_episode_role_pair_{self._role_pair_key(initiator, partner)}"
                ] += 1
                self.ablation_diagnostics[
                    f"task_transition_update_patient_context_{int(update.get('patient_id', -1))}"
                ] += 1
                self.ablation_diagnostics[
                    f"task_update_episode_patient_context_{int(update.get('patient_id', -1))}"
                ] += 1
                self.last_task_transition_update_by_pair[pair_key] = self.timestep
                self.last_opportunistic_corridor_interaction_by_pair[pair_key] = self.timestep
                self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep
            else:
                retained_updates.append(update)

        self.pending_task_transition_updates = retained_updates

    def _log_senior_doctor_oversight_interactions(self) -> None:
        """Log explicit senior reviews after supported doctor task completions."""

        if not self.enable_senior_doctor_oversight or not self.pending_senior_doctor_reviews:
            return
        senior_doctors = [
            doctor for doctor in self.doctors
            if getattr(doctor, "senior_oversight_only", False)
            and not doctor.is_task_busy()
        ]
        if not senior_doctors:
            return
        senior_doctor = senior_doctors[0]
        staff_lookup = self.staff_by_id()
        retained_reviews: List[Dict[str, object]] = []
        available_modes = {"idle", "returning_home", "awaiting_nurse", "patrolling"}

        for review in self.pending_senior_doctor_reviews:
            if self.timestep > int(review.get("expires_at", self.timestep)):
                continue
            initiator = staff_lookup.get(int(review.get("initiator_id", -1)))
            if (
                initiator is None
                or initiator.is_task_busy()
                or initiator.mode not in available_modes
                or getattr(initiator, "senior_oversight_only", False)
            ):
                retained_reviews.append(review)
                continue
            review_zone = str(review.get("review_zone", config.SENIOR_DOCTOR_HOME_ZONE_ID))
            review_position = self._zone_centroid(review_zone)
            if math.dist(initiator.position, review_position) > config.SENIOR_DOCTOR_REVIEW_DISTANCE_METERS:
                retained_reviews.append(review)
                continue
            pair_key = tuple(sorted((initiator.gid, senior_doctor.gid)))
            last_timestamp = self.last_senior_doctor_review_by_pair.get(pair_key)
            if (
                last_timestamp is not None
                and (self.timestep - last_timestamp) < config.SENIOR_DOCTOR_REVIEW_COOLDOWN_SECONDS
            ):
                retained_reviews.append(review)
                continue
            before_count = len(self.interaction_log)
            self.log_interaction(
                agent_1=initiator,
                agent_2=senior_doctor,
                position=self._embodied_contact_position(initiator, senior_doctor),
                interaction_type="senior_doctor_oversight_review",
                task_name=str(review.get("task_name", "")),
                reason_for_interaction=f"task_grounded_senior_doctor_review_at_{review_zone}",
            )
            if len(self.interaction_log) > before_count:
                self.currently_interacting_pairs.add(pair_key)
                self.last_senior_doctor_review_by_pair[pair_key] = self.timestep
                self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep

        self.pending_senior_doctor_reviews = retained_reviews

    def _log_opportunistic_station_interactions(self) -> None:
        station_modes = {
            "idle",
            "returning_home",
            "waiting_for_doctor",
            "moving_to_secondary_station",
            "using_secondary_station",
            "moving_to_post_task_station_check",
            "post_task_station_check",
        }
        for left_index, left_agent in enumerate(self.staff_agents):
            if getattr(left_agent, "senior_oversight_only", False):
                continue
            if left_agent.is_task_busy() or left_agent.mode not in station_modes:
                continue

            for right_agent in self.staff_agents[left_index + 1:]:
                if getattr(right_agent, "senior_oversight_only", False):
                    continue
                if right_agent.is_task_busy() or right_agent.mode not in station_modes:
                    continue
                self.ablation_diagnostics["eligible_encounters_considered"] += 1
                midpoint = (
                    (left_agent.position[0] + right_agent.position[0]) / 2.0,
                    (left_agent.position[1] + right_agent.position[1]) / 2.0,
                )
                zone_id = self.which_zone(*midpoint)
                if left_agent.distance_to_agent(right_agent) > config.STATION_COORDINATION_DISTANCE_METERS:
                    self.ablation_diagnostics["rejected_by_distance"] += 1
                    continue

                if zone_id not in self.station_interaction_zone_ids():
                    continue
                if not (
                    self._station_interaction_ready(left_agent, zone_id)
                    and self._station_interaction_ready(right_agent, zone_id)
                ):
                    self.ablation_diagnostics["rejected_by_station_startup_or_dwell_gate"] += 1
                    continue
                gate_result = self._perception_gate(
                    left_agent,
                    right_agent,
                    interaction_type="opportunistic_station",
                    position=midpoint,
                )
                if not gate_result["accepted"]:
                    self.ablation_diagnostics[str(gate_result["category"])] += 1
                    self.log_missed_opportunity(
                        agent=left_agent,
                        partner=right_agent,
                        reason="needed_partner_not_visible",
                        position=midpoint,
                        perception_state={
                            "encounter": "station",
                            "mutual_visibility": False,
                            "gate_category": gate_result["category"],
                            "gate_reason": gate_result["reason"],
                        },
                        task_state=left_agent.mode,
                    )
                    continue

                pair_key = tuple(sorted((left_agent.gid, right_agent.gid)))
                if pair_key in self.currently_interacting_pairs:
                    self.ablation_diagnostics["rejected_by_cooldown"] += 1
                    self.ablation_diagnostics["repeated_station_pair_suppressed_count"] += 1
                    continue
                if self._station_same_dwell_repeat_suppressed(left_agent, right_agent, zone_id):
                    continue
                last_timestamp = self.last_opportunistic_station_interaction_by_pair.get(pair_key)
                if (
                    last_timestamp is not None
                    and (self.timestep - last_timestamp)
                    < config.OPPORTUNISTIC_STATION_INTERACTION_COOLDOWN_SECONDS
                ):
                    self.ablation_diagnostics["rejected_by_cooldown"] += 1
                    self.ablation_diagnostics["station_same_pair_repeat_count"] += 1
                    self.ablation_diagnostics["repeated_station_pair_suppressed_count"] += 1
                    continue

                probability = self._hcw_hcw_interaction_probability(
                    left_agent,
                    right_agent,
                    zone_id,
                    config.OPPORTUNISTIC_STATION_INTERACTION_PROBABILITY,
                )
                if self.current_ed_pressure_index() >= self.pressure_action_threshold():
                    self.ablation_diagnostics["pressure_suppressed_unrelated_interactions"] += 1
                    if self.random.random() < float(self.scenario_definition.get("pressure_station_social_suppression", 0.0)):
                        continue
                if (
                    zone_id == "COCPIT"
                    and left_agent.mode == "idle"
                    and right_agent.mode == "idle"
                ):
                    probability *= config.COCPIT_IDLE_COLLOCATION_INTERACTION_MULTIPLIER
                if self.random.random() >= probability:
                    self.ablation_diagnostics["rejected_by_probability_or_rule_decision"] += 1
                    self.log_missed_opportunity(
                        agent=left_agent,
                        partner=right_agent,
                        reason="visible_but_not_attended",
                        position=midpoint,
                        perception_state={"encounter": "station", "mutual_visibility": True},
                        task_state=left_agent.mode,
                    )
                    continue

                before_count = len(self.interaction_log)
                self.log_interaction(
                    agent_1=left_agent,
                    agent_2=right_agent,
                    position=self._embodied_contact_position(left_agent, right_agent),
                    interaction_type="opportunistic_station",
                    perception_gate_result=gate_result,
                )
                if len(self.interaction_log) > before_count:
                    self.currently_interacting_pairs.add(pair_key)
                    self._mark_station_same_dwell_episode(left_agent, right_agent, zone_id)
                    self.last_opportunistic_station_interaction_by_pair[pair_key] = self.timestep

    def _station_interaction_ready(self, agent, zone_id: Optional[str]) -> bool:
        """Avoid counting initial co-location as a fresh station encounter."""

        if zone_id not in self.station_interaction_zone_ids():
            return False
        entered_at = int(self.zone_entered_at_by_agent.get(agent.gid, self.timestep))
        dwell_seconds = self.timestep - entered_at
        if (
            self.timestep < config.INITIAL_STATION_INTERACTION_GRACE_SECONDS
            and getattr(agent, "mode", None) == "idle"
            and zone_id == getattr(agent, "home_zone_id", None)
        ):
            return False
        return dwell_seconds >= config.STATION_INTERACTION_MIN_DWELL_SECONDS

    def _station_same_dwell_episode_key(
        self,
        left_agent,
        right_agent,
        zone_id: Optional[str],
    ) -> Optional[tuple[object, ...]]:
        if zone_id not in self.station_interaction_zone_ids():
            return None
        left_episode = self.active_station_episode_by_agent.get(left_agent.gid)
        right_episode = self.active_station_episode_by_agent.get(right_agent.gid)
        if (
            left_episode is None
            or right_episode is None
            or str(left_episode.get("station_zone")) != str(zone_id)
            or str(right_episode.get("station_zone")) != str(zone_id)
        ):
            return None
        left_id, right_id = sorted((int(left_agent.gid), int(right_agent.gid)))
        return (
            left_id,
            right_id,
            str(zone_id),
            str(left_episode.get("episode_id")),
            str(right_episode.get("episode_id")),
        )

    def _station_same_dwell_repeat_suppressed(self, left_agent, right_agent, zone_id: Optional[str]) -> bool:
        episode_key = self._station_same_dwell_episode_key(left_agent, right_agent, zone_id)
        if episode_key is None:
            return False
        self.ablation_diagnostics["station_same_pair_same_episode_opportunity_count"] += 1
        if episode_key not in self.station_same_episode_interaction_keys:
            return False
        self.ablation_diagnostics["station_same_episode_repeat_suppressed_count"] += 1
        self.ablation_diagnostics["station_same_dwell_repeat_suppressed_count"] += 1
        self.ablation_diagnostics["station_same_pair_repeat_count"] += 1
        self.ablation_diagnostics["repeated_station_pair_suppressed_count"] += 1
        self.ablation_diagnostics["rejected_by_cooldown"] += 1
        return True

    def _mark_station_same_dwell_episode(self, left_agent, right_agent, zone_id: Optional[str]) -> None:
        episode_key = self._station_same_dwell_episode_key(left_agent, right_agent, zone_id)
        if episode_key is None:
            return
        self.station_same_episode_interaction_keys.add(episode_key)
        self.ablation_diagnostics["station_same_pair_episode_count"] += 1
        self.ablation_diagnostics["station_episode_new_encounter_count"] += 1

    def interaction_zone_counts(self, interactions: Optional[Iterable[Dict[str, object]]] = None) -> Counter:
        zone_counts: Counter = Counter()
        source = self.interaction_log if interactions is None else interactions

        for event in source:
            zone_id = self.which_zone(float(event["x"]), float(event["y"]))
            zone_counts[zone_id or "Outside named zones"] += 1

        return zone_counts

    def _draw_condition_overlays(self, axis) -> None:
        self.condition_manager.draw_zone_overlays(axis)

    def _draw_floorplan(self, axis) -> None:
        self.condition_manager.draw_floorplan(axis)

    def interaction_type_counts(self, interactions: Optional[Iterable[Dict[str, object]]] = None) -> Counter:
        source = self.interaction_log if interactions is None else interactions
        return Counter(str(event["interaction_type"]) for event in source)

    def interaction_role_pair_counts(self, interactions: Optional[Iterable[Dict[str, object]]] = None) -> Counter:
        source = self.interaction_log if interactions is None else interactions
        pair_counts: Counter = Counter()
        for event in source:
            pair = tuple(sorted((str(event["role_1"]), str(event["role_2"]))))
            pair_counts[pair] += 1
        return pair_counts

    def interaction_topic_counts(self, interactions: Optional[Iterable[Dict[str, object]]] = None) -> Counter:
        source = self.interaction_log if interactions is None else interactions
        return Counter(str(event.get("topic", "unknown_topic")) for event in source)

    def interaction_esi_counts(self, interactions: Optional[Iterable[Dict[str, object]]] = None) -> Counter:
        source = self.interaction_log if interactions is None else interactions
        counter: Counter = Counter()
        for event in source:
            esi_level = event.get("esi_level")
            if esi_level is not None:
                counter[f"ESI {esi_level}"] += 1
        return counter

    def workflow_event_type_counts(self) -> Counter:
        return Counter(str(event.get("event_type", "unknown")) for event in self.workflow_event_log)

    def workflow_task_counts(self) -> Counter:
        return Counter(str(event.get("task_name", "none")) for event in self.workflow_event_log)

    def workflow_role_counts(self) -> Counter:
        return Counter(str(event.get("agent_role", "system")) for event in self.workflow_event_log)

    def workflow_health_summary(self, warmup_seconds: int = 0) -> dict:
        """Summarize whether the ED workflow actually operated during a run."""

        warmup_seconds = max(0, int(warmup_seconds))
        workflow_events = [
            event for event in self.workflow_event_log
            if int(event.get("timestamp", event.get("timestep", 0))) >= warmup_seconds
        ]
        event_counts = Counter(str(event.get("event_type", "unknown")) for event in workflow_events)
        task_started_events = [
            event for event in workflow_events if event.get("event_type") == "task_started"
        ]
        task_completed_events = [
            event for event in workflow_events if event.get("event_type") == "task_completed"
        ]
        doctor_task_names = {
            config.MEDICAL_EVALUATION_TASK_NAME,
            config.DIAGNOSTICS_TASK_NAME,
            config.TREATMENT_TASK_NAME,
            config.REASSESSMENT_TASK_NAME,
            config.DISPOSITION_DECISION_TASK_NAME,
        }
        doctor_task_started_events = [
            event
            for event in task_started_events
            if event.get("agent_role") == "Doctor" or event.get("task_name") in doctor_task_names
        ]
        doctor_task_completed_events = [
            event
            for event in task_completed_events
            if event.get("agent_role") == "Doctor" or event.get("task_name") in doctor_task_names
        ]
        placement_started_events = [
            event for event in task_started_events if event.get("task_name") == config.PLACEMENT_TASK_NAME
        ]
        placement_completed_events = [
            event for event in task_completed_events if event.get("task_name") == config.PLACEMENT_TASK_NAME
        ]
        completed_placement_patient_ids = {
            event.get("patient_id") for event in placement_completed_events
        }
        transit_stuck_repair_events = [
            event
            for event in workflow_events
            if event.get("event_type") == "staff_assignment_repaired"
            and (event.get("patient_id") is not None or event.get("task_name") is not None)
        ]
        handoff_wait_repair_events = [
            event
            for event in workflow_events
            if event.get("event_type") == "handoff_wait_repaired"
        ]
        assignment_repair_events = [
            event
            for event in workflow_events
            if event.get("event_type") in {"staff_assignment_repaired", "handoff_wait_repaired"}
        ]
        overdue_placement_events = [
            event
            for event in placement_started_events
            if event.get("patient_id") not in completed_placement_patient_ids
            and self.timestep - int(event.get("timestep", self.timestep))
            > (max(config.TASK_DURATIONS_SECONDS[config.PLACEMENT_TASK_NAME]) + 300)
        ]
        arrivals = int(event_counts.get("patient_arrival", 0))
        deferred_arrivals = int(event_counts.get("deferred_arrival", 0))
        bed_assignments = int(event_counts.get("bed_assigned", 0))
        placement_started = len(placement_started_events)
        placement_completed = len(placement_completed_events)
        task_started = len(task_started_events)
        task_completed = len(task_completed_events)
        discharges = len(self.completed_patients)
        max_stalled_seconds = max(
            (int(getattr(agent, "stalled_seconds", 0)) for agent in self.staff_agents),
            default=0,
        )
        terminal_stuck_agent_events = sum(
            1
            for agent in self.staff_agents
            if int(getattr(agent, "stalled_seconds", 0)) >= config.STAFF_TRANSIT_STALL_TIMEOUT_SECONDS
        )
        oscillation_warnings = len(overdue_placement_events)
        occupied_beds_now = {
            patient.bed_index
            for patient in self.active_patients
            if patient.bed_index is not None and not patient.discharged
        }
        bed_assignment_events = [
            event for event in workflow_events if event.get("event_type") == "bed_assigned"
        ]
        bed_assignment_counts = Counter(str(event.get("new_state")) for event in bed_assignment_events)
        bed_assignment_esi_by_bed: dict[str, Counter] = defaultdict(Counter)
        for event in bed_assignment_events:
            bed_label = str(event.get("new_state", "unknown")).replace("_assigned", "")
            bed_assignment_esi_by_bed[bed_label][f"ESI {event.get('esi_level')}"] += 1
        reserved_bed_numbers = {
            int(index) + 1
            for index in getattr(config, "HIGH_ACUITY_RESERVED_BED_INDICES", set())
        }
        reserved_bed_violations = [
            event
            for event in bed_assignment_events
            if str(event.get("new_state")) in {f"bed_{bed_number}_assigned" for bed_number in reserved_bed_numbers}
            and event.get("esi_level") not in {1, 2}
        ]
        active_waiting_patients = [
            patient
            for patient in self.active_patients
            if patient.bed_index is None and not patient.discharged
        ]
        doctor_backlog_patients = [
            patient
            for patient in self.active_patients
            if not patient.discharged
            and patient.task_index in config.ROLE_PERMISSIONS["Doctor"]
            and not self._doctor_assigned_to_patient(patient)
        ]
        beds_used = sorted(
            int(bed_index) + 1
            for bed_index in (set(self.bed_assignment_history) | occupied_beds_now)
            if bed_index is not None
        )
        sample_start = min(warmup_seconds, len(self.bed_occupancy_samples))
        bed_occupancy_samples = self.bed_occupancy_samples[sample_start:]
        ordinary_bed_occupancy_samples = self.ordinary_bed_occupancy_samples[sample_start:]
        special_bed_occupancy_samples = self.special_bed_occupancy_samples[sample_start:]
        waiting_queue_samples = self.waiting_queue_samples[sample_start:]
        ed_pressure_index_samples = self.ed_pressure_index_samples[sample_start:]
        active_high_acuity_patient_samples = self.active_high_acuity_patient_samples[sample_start:]
        active_esi1_patient_samples = self.active_esi1_patient_samples[sample_start:]
        active_esi2_patient_samples = self.active_esi2_patient_samples[sample_start:]
        active_esi1_esi2_patient_samples = self.active_esi1_esi2_patient_samples[sample_start:]
        average_occupancy = mean(bed_occupancy_samples) if bed_occupancy_samples else 0.0
        max_occupancy = max(bed_occupancy_samples) if bed_occupancy_samples else 0
        ordinary_capacity = len(self.ordinary_bed_indices())
        special_capacity = len(set(getattr(config, "HIGH_ACUITY_RESERVED_BED_INDICES", set())))
        average_ordinary_occupancy = (
            mean(ordinary_bed_occupancy_samples) if ordinary_bed_occupancy_samples else 0.0
        )
        p95_ordinary_occupancy = self._percentile(ordinary_bed_occupancy_samples, 95)
        average_special_occupancy = (
            mean(special_bed_occupancy_samples) if special_bed_occupancy_samples else 0.0
        )
        p95_special_occupancy = self._percentile(special_bed_occupancy_samples, 95)
        mean_waiting_queue_length = mean(waiting_queue_samples) if waiting_queue_samples else 0.0
        p95_waiting_queue_length = self._percentile(waiting_queue_samples, 95)
        ed_pressure_index_mean = mean(ed_pressure_index_samples) if ed_pressure_index_samples else 0.0
        ed_pressure_index_p95 = self._percentile(ed_pressure_index_samples, 95)
        total_sample_seconds = max(len(ed_pressure_index_samples) * config.TIMESTEP_SECONDS, 1)
        zero_waiting_seconds = sum(1 for value in waiting_queue_samples if int(value) == 0) * config.TIMESTEP_SECONDS
        waiting_pressure_seconds = sum(1 for value in waiting_queue_samples if int(value) > 0) * config.TIMESTEP_SECONDS
        ordinary_beds_full_seconds = (
            sum(1 for value in ordinary_bed_occupancy_samples if int(value) >= ordinary_capacity)
            * config.TIMESTEP_SECONDS
        )
        open_ordinary_bed_seconds = (
            sum(1 for value in ordinary_bed_occupancy_samples if int(value) < ordinary_capacity)
            * config.TIMESTEP_SECONDS
        )
        high_acuity_active_seconds = (
            sum(1 for value in active_high_acuity_patient_samples if int(value) > 0)
            * config.TIMESTEP_SECONDS
        )
        active_esi1_patient_minutes = (
            sum(int(value) for value in active_esi1_patient_samples) * config.TIMESTEP_SECONDS / 60.0
        )
        active_esi2_patient_minutes = (
            sum(int(value) for value in active_esi2_patient_samples) * config.TIMESTEP_SECONDS / 60.0
        )
        mean_concurrent_esi1 = mean(active_esi1_patient_samples) if active_esi1_patient_samples else 0.0
        mean_concurrent_esi1_esi2 = (
            mean(active_esi1_esi2_patient_samples) if active_esi1_esi2_patient_samples else 0.0
        )
        max_concurrent_esi1 = max(active_esi1_patient_samples) if active_esi1_patient_samples else 0
        max_concurrent_esi1_esi2 = (
            max(active_esi1_esi2_patient_samples) if active_esi1_esi2_patient_samples else 0
        )
        ordinary_bed_open_seconds_by_bed = {
            bed_label: int(sum(samples[sample_start:]) * config.TIMESTEP_SECONDS)
            for bed_label, samples in self.ordinary_bed_open_samples_by_bed.items()
        }
        doctor_wait_metrics = self._doctor_wait_metrics_by_task(task_started_events, task_completed_events)
        doctor_utilization = self._doctor_utilization_summary()
        esi_diagnostics = self._esi_workflow_diagnostics(warmup_seconds=warmup_seconds)

        status = "PASS"
        reasons: list[str] = []
        if arrivals > 0 and bed_assignments == 0:
            status = "FAIL"
            reasons.append("patients arrived but no bed assignments occurred")
        if bed_assignments > 0 and placement_started == 0:
            status = "FAIL"
            reasons.append("beds were assigned but placement tasks never started")
        if placement_started > 0 and placement_completed == 0 and self.timestep >= 900:
            status = "FAIL"
            reasons.append("placement tasks started but none completed")
        if task_started > 0 and task_completed == 0 and self.timestep >= 1800:
            status = "FAIL"
            reasons.append("workflow tasks started but none completed")
        if oscillation_warnings > 0:
            status = "FAIL" if status == "FAIL" else "WARN"
            reasons.append(f"{oscillation_warnings} placement task(s) exceeded expected escort/task time")
        if transit_stuck_repair_events or terminal_stuck_agent_events:
            status = "FAIL" if status == "FAIL" else "WARN"
            reasons.append("one or more patient-flow assignments required stall repair")
        if self.unsupported_assignment_rejections:
            status = "FAIL" if status == "FAIL" else "WARN"
            reasons.append("one or more patient-flow assignments had unsupported route targets")
        if self.route_movement_issues:
            status = "FAIL" if status == "FAIL" else "WARN"
            reasons.append("one or more staff movements had unsupported or non-walkable routes")
        if reserved_bed_violations:
            status = "FAIL"
            reasons.append("reserved high-acuity bed accepted an ESI 3-5 patient")
        if arrivals >= 2 and placement_completed < max(1, bed_assignments // 2) and self.timestep >= 1800:
            status = "FAIL" if status == "FAIL" else "WARN"
            reasons.append("patient placement throughput is low relative to bed assignments")
        if not reasons:
            reasons.append("arrivals, assignments, placement, and task progression are internally consistent")

        return {
            "arrival_attempts": arrivals + deferred_arrivals,
            "scenario_mode": self.scenario_mode,
            "scenario_definition": dict(self.scenario_definition),
            "scenario_start_hour": int(self.scenario_start_hour),
            "patient_arrivals": arrivals,
            "admitted_arrivals": arrivals,
            "deferred_arrivals": deferred_arrivals,
            "deferred_arrivals_by_esi": dict(self.deferred_arrivals_by_esi),
            "max_active_system_load": int(config.MAX_ACTIVE_SYSTEM_LOAD),
            "bed_assignments": bed_assignments,
            "successful_escorts_to_bed": placement_completed,
            "placement_tasks_started": placement_started,
            "placement_tasks_completed": placement_completed,
            "workflow_tasks_started": task_started,
            "workflow_tasks_completed": task_completed,
            "doctor_tasks_started": len(doctor_task_started_events),
            "doctor_tasks_completed": len(doctor_task_completed_events),
            "doctor_task_backlog_patients": len(doctor_backlog_patients),
            "average_doctor_task_wait_seconds": float(
                mean([patient.seconds_at_current_task(self.timestep) for patient in doctor_backlog_patients])
            ) if doctor_backlog_patients else 0.0,
            "doctor_wait_metrics_by_task": doctor_wait_metrics,
            "doctor_utilization": doctor_utilization,
            "discharges": discharges,
            "beds_used": beds_used,
            "bed_assignment_counts": dict(bed_assignment_counts),
            "bed_assignment_esi_by_bed": {
                bed_label: dict(counter)
                for bed_label, counter in bed_assignment_esi_by_bed.items()
            },
            "reserved_bed_numbers": sorted(reserved_bed_numbers),
            "reserved_bed_restriction_violations": len(reserved_bed_violations),
            "active_waiting_patients": len(active_waiting_patients),
            "average_active_waiting_seconds": float(
                mean([self.timestep - patient.spawned_at for patient in active_waiting_patients])
            ) if active_waiting_patients else 0.0,
            "average_bed_occupancy": float(average_occupancy),
            "max_bed_occupancy": int(max_occupancy),
            "ordinary_bed_capacity": int(ordinary_capacity),
            "average_ordinary_bed_occupancy": float(average_ordinary_occupancy),
            "p95_ordinary_bed_occupancy": float(p95_ordinary_occupancy),
            "special_high_acuity_bed_capacity": int(special_capacity),
            "average_special_high_acuity_bed_occupancy": float(average_special_occupancy),
            "p95_special_high_acuity_bed_occupancy": float(p95_special_occupancy),
            "percent_time_all_ordinary_beds_full": float(ordinary_beds_full_seconds / total_sample_seconds),
            "percent_time_at_least_one_ordinary_bed_open": float(open_ordinary_bed_seconds / total_sample_seconds),
            "ordinary_bed_open_seconds_by_bed": ordinary_bed_open_seconds_by_bed,
            "ordinary_bed_open_minutes_by_bed": {
                bed_label: float(seconds) / 60.0
                for bed_label, seconds in ordinary_bed_open_seconds_by_bed.items()
            },
            "percent_time_waiting_queue_gt_zero": float(waiting_pressure_seconds / total_sample_seconds),
            "percent_time_waiting_queue_zero": float(zero_waiting_seconds / total_sample_seconds),
            "mean_waiting_queue_length": float(mean_waiting_queue_length),
            "p95_waiting_queue_length": float(p95_waiting_queue_length),
            "ed_pressure_index_mean": float(ed_pressure_index_mean),
            "ed_pressure_index_p95": float(ed_pressure_index_p95),
            "waiting_pressure_minutes": float(waiting_pressure_seconds) / 60.0,
            "ordinary_beds_full_minutes": float(ordinary_beds_full_seconds) / 60.0,
            "open_ordinary_bed_minutes": float(open_ordinary_bed_seconds) / 60.0,
            "zero_waiting_minutes": float(zero_waiting_seconds) / 60.0,
            "high_acuity_active_minutes": float(high_acuity_active_seconds) / 60.0,
            "active_esi1_patient_minutes": float(active_esi1_patient_minutes),
            "active_esi2_patient_minutes": float(active_esi2_patient_minutes),
            "max_concurrent_esi1": int(max_concurrent_esi1),
            "max_concurrent_esi1_esi2": int(max_concurrent_esi1_esi2),
            "mean_concurrent_esi1": float(mean_concurrent_esi1),
            "mean_concurrent_esi1_esi2": float(mean_concurrent_esi1_esi2),
            "assignment_repair_events": int(len(assignment_repair_events)),
            "handoff_wait_repair_events": int(len(handoff_wait_repair_events)),
            "transit_stuck_repair_events": int(len(transit_stuck_repair_events)),
            "terminal_stuck_agent_events": int(terminal_stuck_agent_events),
            "assignment_repair_event_details": assignment_repair_events,
            "unsupported_assignment_rejection_count": int(len(self.unsupported_assignment_rejections)),
            "unsupported_assignment_rejection_details": list(self.unsupported_assignment_rejections),
            "route_movement_issue_count": int(len(self.route_movement_issues)),
            "route_movement_issue_details": list(self.route_movement_issues),
            "stuck_agent_events": int(self.stuck_agent_events),
            "oscillation_warnings": int(oscillation_warnings),
            "max_staff_stalled_seconds": int(max_stalled_seconds),
            "workflow_health_status": status,
            "workflow_health_reasons": reasons,
            **esi_diagnostics,
        }

    def interaction_summary(self) -> dict:
        return {
            "interactions_by_type": dict(self.interaction_type_counts()),
            "interactions_by_zone": dict(self.interaction_zone_counts()),
            "interactions_by_role_pair": {
                "|".join(pair): count for pair, count in self.interaction_role_pair_counts().items()
            },
            "interactions_by_topic": dict(self.interaction_topic_counts()),
            "interactions_by_patient_esi": dict(self.interaction_esi_counts()),
            "workflow_events_by_type": dict(self.workflow_event_type_counts()),
            "workflow_events_by_task": dict(self.workflow_task_counts()),
            "workflow_events_by_role": dict(self.workflow_role_counts()),
            "workflow_health": self.workflow_health_summary(),
            "missed_opportunities_by_reason": dict(
                Counter(str(event["reason"]) for event in self.missed_opportunity_log)
            ),
            "interruptions_by_role": dict(
                Counter(str(event.get("agent_1_id")) for event in self.interruption_log)
            ),
        }

    def llm_metrics(self) -> Dict[str, float]:
        stats = dict(self.interaction_engine.stats)
        llm_call_count = int(stats.get("llm_calls", 0))
        stats["parse_success_rate"] = (
            stats["parse_successes"] / llm_call_count
            if llm_call_count > 0
            else 1.0
        )
        stats["fallback_rate"] = (
            stats["llm_fallbacks"] / llm_call_count
            if llm_call_count > 0
            else 0.0
        )
        stats["decision_calls_per_hour"] = (
            stats["decision_calls"] / max(self.timestep / 3600.0, 1e-9)
        )
        return stats

    def movement_metrics(self) -> Dict[str, object]:
        average_trip_lengths = {
            label: mean(lengths)
            for label, lengths in self.trip_path_lengths_by_label.items()
            if lengths
        }
        average_trip_durations = {
            label: mean(durations)
            for label, durations in self.trip_durations_by_label.items()
            if durations
        }
        station_occupancy = Counter()
        for role_counters in self.zone_dwell_by_role.values():
            for station_zone in config.STATION_ZONE_IDS:
                station_occupancy[station_zone] += role_counters.get(station_zone, 0)

        return {
            "movement_distance_by_role": dict(self.movement_distance_by_role),
            "zone_dwell_by_role": {role: dict(counter) for role, counter in self.zone_dwell_by_role.items()},
            "station_occupancy_seconds": dict(station_occupancy),
            "trip_counts": dict(self.trip_counts),
            "zone_transition_counts": dict(self.zone_transition_counts),
            "average_trip_lengths_by_label": average_trip_lengths,
            "average_trip_durations_by_label": average_trip_durations,
        }

    def save_trace_outputs(self) -> None:
        config.MEMORY_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        config.RETRIEVAL_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        config.REFLECTION_DIR.mkdir(parents=True, exist_ok=True)
        for agent in self.staff_agents:
            safe_name = getattr(agent, "name", f"{agent.role}_{agent.gid}").replace(" ", "_")
            stream = self.interaction_engine.stream_for(agent.gid)
            memory_payload = [event.__dict__ for event in stream.events[-200:]]
            retrieval_payload = stream.retrieval_traces[-100:]
            reflection_payload = [
                event.__dict__ for event in stream.events
                if event.event_type == "reflection"
            ][-50:]
            if config.SAVE_MEMORY_TRACES:
                (config.MEMORY_TRACE_DIR / f"{safe_name}_{self.condition_spec.name}_{self.random_seed}.json").write_text(
                    json.dumps(memory_payload, indent=2)
                )
            if config.SAVE_RETRIEVAL_TRACES:
                (config.RETRIEVAL_TRACE_DIR / f"{safe_name}_{self.condition_spec.name}_{self.random_seed}.json").write_text(
                    json.dumps(retrieval_payload, indent=2)
                )
            (config.REFLECTION_DIR / f"{safe_name}_{self.condition_spec.name}_{self.random_seed}.json").write_text(
                json.dumps(reflection_payload, indent=2)
            )

    def generate_agent_shift_narrative(self, agent_gid: int) -> str:
        agent = next((staff_member for staff_member in self.staff_agents if staff_member.gid == agent_gid), None)
        if agent is None:
            return f"Agent {agent_gid} was not present in this simulation."

        dwell_counter = self.zone_dwell_by_agent.get(agent_gid, Counter())
        top_zones = dwell_counter.most_common(3)
        zone_phrase = ", ".join(
            f"{zone_id} ({seconds / 3600.0:.1f} hours)"
            for zone_id, seconds in top_zones
        ) or "no named zones"

        agent_interactions = [
            event
            for event in self.interaction_log
            if event["agent_1"] == agent_gid or event["agent_2"] == agent_gid
        ]
        type_counts = Counter(str(event.get("interaction_type", "unknown")) for event in agent_interactions)
        topic_counts = Counter(str(event.get("topic", "unknown")) for event in agent_interactions)
        total_interactions = len(agent_interactions)
        topic_phrase = ", ".join(
            f"{topic} ({(count / max(total_interactions, 1)) * 100.0:.0f}%)"
            for topic, count in topic_counts.most_common(3)
        ) or "no recorded topics"
        type_phrase = ", ".join(
            f"{interaction_type} ({count})"
            for interaction_type, count in type_counts.most_common(3)
        ) or "no interaction categories"

        return (
            f"{agent.role} (gid={agent.gid}) spent {self.timestep / 3600.0:.0f} hours in the "
            f"{self.condition_spec.name} ED condition. They moved "
            f"{self.movement_distance_by_agent.get(agent_gid, 0.0):.0f} metres across the shift, "
            f"spending most time in {zone_phrase}. They participated in {total_interactions} interactions: "
            f"{type_phrase}. The most common discussion topics were {topic_phrase}."
        )

    def compute_kde_grid(self):
        points_xy = [
            (float(event["x"]), float(event["y"]))
            for event in self.interaction_log
        ]
        return compute_kde_grid(
            points_xy,
            self.environment.plot_bounds,
            resolution=1.0,
            bandwidth=1.5,
        )

    def visibility_metrics_snapshot(self) -> Dict[str, object]:
        key_points = {
            "COCPIT": self.condition_manager.station_attractor("COCPIT"),
            "NUROPE": self.condition_manager.station_attractor("NUROPE"),
            "NURSTA": self.condition_manager.station_attractor("NURSTA"),
            "ENTRY": config.PATIENT_ENTRY_POINT,
            "CORR07_DOOR": config.ROUTING_WAYPOINTS["entry_door"],
        }
        isovist_areas = {}
        through_vision = {}
        for label, point in key_points.items():
            polygon = self.perception.compute_isovist(point, heading=0.0, fov_degrees=360.0)
            isovist_areas[label] = self._polygon_area(polygon)

        labels = list(key_points)
        for left_index, left_label in enumerate(labels):
            for right_label in labels[left_index + 1:]:
                through_vision[f"{left_label}->{right_label}"] = self.perception._has_visibility(
                    key_points[left_label],
                    key_points[right_label],
                )

        return {
            "mean_isovist_area_by_key_point": isovist_areas,
            "through_vision": through_vision,
            "mutual_visibility_counts_by_zone": dict(self.mutual_visibility_counts_by_zone),
        }

    def _polygon_area(self, polygon) -> float:
        if len(polygon) < 3:
            return 0.0
        area = 0.0
        for index, (x1, y1) in enumerate(polygon):
            x2, y2 = polygon[(index + 1) % len(polygon)]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) * 0.5

    def _status_lines(self) -> List[str]:
        position_digits = config.STATUS_POSITION_DECIMALS
        lines = []

        for staff_member in self.staff_agents:
            lines.append(
                f"{staff_member.role} {staff_member.gid}: "
                f"pos=({staff_member.position[0]:.{position_digits}f}, "
                f"{staff_member.position[1]:.{position_digits}f}) "
                f"state={staff_member.state_summary()}"
            )

        for patient in self.active_patients:
            lines.append(
                f"Patient {patient.gid}: "
                f"pos=({patient.position[0]:.{position_digits}f}, "
                f"{patient.position[1]:.{position_digits}f}) "
                f"state={patient.state_summary()}"
            )

        return lines

    def _print_hourly_status(self) -> None:
        hour_start = max(self.timestep - config.HOURLY_REPORT_INTERVAL_SECONDS, 0)
        hourly_interactions = [
            event for event in self.interaction_log
            if hour_start < int(event["timestamp"]) <= self.timestep
        ]
        print(f"\nHour mark at timestep {self.timestep}")
        print(f"Active patients: {len(self.active_patients)}")
        print(f"Total interactions: {len(self.interaction_log)}")
        print(f"Interactions this hour: {len(hourly_interactions)}")
        for line in self._status_lines():
            print(line)

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()
            if self.timestep % config.HOURLY_REPORT_INTERVAL_SECONDS == 0:
                self._print_hourly_status()

    def _draw_interactions(self, axis, marker_size: int, alpha: float, z_order: int) -> None:
        if self.interaction_log:
            xs = [event["x"] for event in self.interaction_log]
            ys = [event["y"] for event in self.interaction_log]
            axis.scatter(
                xs,
                ys,
                c="royalblue",
                s=marker_size,
                alpha=alpha,
                zorder=z_order,
                label=f"Interactions (n={len(self.interaction_log)})",
            )

    def _draw_active_agents(self, axis, z_order: int) -> None:
        plotted_roles = set()
        for agent in [*self.staff_agents, *self.active_patients]:
            marker_size = (
                config.COORDINATION_NURSE_MARKER_SIZE
                if agent.role == "CoordinationNurse"
                else config.AGENT_MARKER_SIZE
            )
            label = agent.role if agent.role not in plotted_roles else None
            axis.scatter(
                agent.position[0],
                agent.position[1],
                color=config.ROLE_COLORS[agent.role],
                marker=config.ROLE_MARKERS[agent.role],
                s=marker_size,
                label=label,
                zorder=z_order,
            )
            if hasattr(agent, "heading") and agent.role != "Patient":
                arrow_length = 0.28
                axis.arrow(
                    agent.position[0],
                    agent.position[1],
                    math.cos(agent.heading) * arrow_length,
                    math.sin(agent.heading) * arrow_length,
                    color=config.ROLE_COLORS[agent.role],
                    width=0.015,
                    head_width=0.10,
                    length_includes_head=True,
                    alpha=0.85,
                    zorder=z_order + 1,
                )
            plotted_roles.add(agent.role)

    def visualize(self) -> None:
        figure, axis = plt.subplots(figsize=config.FIGURE_SIZE)

        self._draw_floorplan(axis)
        self._draw_condition_overlays(axis)

        self._draw_interactions(axis, marker_size=12, alpha=0.5, z_order=5)

        for agent in [*self.staff_agents, *self.completed_patients, *self.active_patients]:
            trail_xs = [point[0] for point in agent.trail]
            trail_ys = [point[1] for point in agent.trail]
            axis.plot(
                trail_xs,
                trail_ys,
                color=config.ROLE_COLORS[agent.role],
                alpha=config.TRAIL_ALPHA,
                linewidth=config.TRAIL_LINEWIDTH,
                zorder=2,
            )

        self._draw_active_agents(axis, z_order=6)

        min_x, max_x, min_y, max_y = self.environment.plot_bounds
        axis.set_xlim(min_x, max_x)
        axis.set_ylim(min_y, max_y)
        axis.set_aspect("equal")
        axis.set_xlabel("X (meters)")
        axis.set_ylabel("Y (meters)")
        axis.set_title("Phase 1 ED ABM")
        axis.legend(loc="upper left")

        figure.tight_layout()
        figure.savefig(config.OUTPUT_FIGURE_PATH, dpi=300)
        plt.close(figure)

    def animate(self) -> None:
        import matplotlib.animation as animation

        figure, axis = plt.subplots(figsize=config.FIGURE_SIZE)

        def init_frame():
            axis.clear()
            self._draw_floorplan(axis)
            self._draw_condition_overlays(axis)
            axis.set_xlim(*self.environment.plot_bounds[:2])
            axis.set_ylim(*self.environment.plot_bounds[2:])
            axis.set_aspect("equal")
            return []

        def update_frame(frame_number):
            del frame_number
            for _ in range(config.ANIMATION_STEPS_PER_FRAME):
                if self.timestep < config.ANIMATION_DURATION_SECONDS:
                    self.step()

            axis.clear()
            self._draw_floorplan(axis)
            self._draw_condition_overlays(axis)
            self._draw_interactions(axis, marker_size=8, alpha=0.3, z_order=5)
            self._draw_active_agents(axis, z_order=6)

            simulated_minutes = self.timestep // 60
            axis.set_title(
                f"ED Sim - t={simulated_minutes}min | "
                f"Patients: [W: {len([p for p in self.active_patients if p.bed_index is None])} | "
                f"B: {len([p for p in self.active_patients if p.bed_index is not None])}] | "
                f"Discharged: {len(self.completed_patients)} | "
                f"Interactions: {len(self.interaction_log)}"
            )
            axis.set_xlim(*self.environment.plot_bounds[:2])
            axis.set_ylim(*self.environment.plot_bounds[2:])
            axis.set_aspect("equal")
            return []

        total_frames = max(
            config.ANIMATION_DURATION_SECONDS // config.ANIMATION_STEPS_PER_FRAME,
            1,
        )
        anim = animation.FuncAnimation(
            figure,
            update_frame,
            frames=total_frames,
            init_func=init_frame,
            interval=config.ANIMATION_INTERVAL_MS,
            blit=False,
        )

        if config.SAVE_ANIMATION:
            writer = animation.FFMpegWriter(fps=20)
            anim.save(str(config.ANIMATION_OUTPUT_PATH), writer=writer)
            print(f"Animation saved to {config.ANIMATION_OUTPUT_PATH}")
        else:
            plt.show()


def check_simulation_health(simulation: Simulation) -> None:
    print(f"\n=== HEALTH CHECK @ {simulation.timestep}s ===")

    print("Patients:")
    for patient in simulation.active_patients:
        print(
            "  "
            f"gid={patient.gid} "
            f"task_index={patient.task_index} "
            f"bed_index={patient.bed_index} "
            f"claiming_staff_id={patient.claiming_staff_id} "
            f"requested_doctor_id={patient.requested_doctor_id} "
            f"seconds_at_current_task={patient.seconds_at_current_task(simulation.timestep)}"
        )

    print("Staff:")
    for staff_member in simulation.staff_agents:
        print(
            "  "
            f"gid={staff_member.gid} "
            f"role={staff_member.role} "
            f"mode={staff_member.mode} "
            f"target_patient_id={staff_member.target_patient_id} "
            f"task_remaining={staff_member.task_remaining}"
        )

    inconsistencies: List[str] = []
    staff_lookup = simulation.staff_by_id()

    for patient in simulation.active_patients:
        if patient.claiming_staff_id is None:
            continue
        staff_member = staff_lookup.get(patient.claiming_staff_id)
        if staff_member is None or staff_member.target_patient_id != patient.gid:
            inconsistencies.append(
                "stale_claim "
                f"patient={patient.gid} "
                f"claiming_staff_id={patient.claiming_staff_id} "
                f"staff_target={getattr(staff_member, 'target_patient_id', None)}"
            )

    for doctor in simulation.doctors:
        if doctor.mode != "awaiting_nurse":
            continue
        nurse = simulation.nurse_by_id.get(doctor.handoff_nurse_id)
        if nurse is None or nurse.reserved_doctor_id != doctor.gid:
            inconsistencies.append(
                "mismatched_handoff "
                f"doctor={doctor.gid} "
                f"handoff_patient_id={doctor.handoff_patient_id} "
                f"handoff_nurse_id={doctor.handoff_nurse_id}"
            )

    for nurse in simulation.nurses:
        if nurse.mode == "waiting_for_doctor" and nurse.recheck_timer <= 0:
            inconsistencies.append(
                "expired_recheck_block "
                f"nurse={nurse.gid} "
                f"patient={nurse.doctor_recheck_patient_id} "
                f"recheck_failures={nurse.doctor_recheck_failures}"
            )

    duplicate_doctor_targets: Dict[int, List[int]] = {}
    for doctor in simulation.doctors:
        if doctor.target_patient_id is None:
            continue
        duplicate_doctor_targets.setdefault(doctor.target_patient_id, []).append(doctor.gid)
    for patient_id, doctor_ids in duplicate_doctor_targets.items():
        if len(doctor_ids) > 1:
            inconsistencies.append(
                "duplicate_doctor_target "
                f"patient={patient_id} "
                f"doctor_ids={doctor_ids}"
            )

    print("Inconsistencies:")
    if inconsistencies:
        for issue in inconsistencies:
            print(f"  {issue}")
    else:
        print("  none")
