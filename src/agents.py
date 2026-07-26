"""Agent classes for the ED ABM.

The hierarchy keeps movement, perception, and state transitions encapsulated in
agents so later phases can add richer sensing, routing, logging, or roles
without rewriting the simulation loop.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import config


Point = Tuple[float, float]


class GenericAgent(ABC):
    """Shared interface and state used by patients and staff."""

    def __init__(self, gid: int, role: str, position: Point) -> None:
        self.gid = gid
        self.role = role
        self.position = position
        self.next_position = position
        self.heading = 0.0
        self.next_heading = self.heading
        self.trail = [position]

    def begin_step(self) -> None:
        self.next_position = self.position
        self.next_heading = self.heading

    def update(self) -> None:
        self.position = self.next_position
        self.heading = self.next_heading
        self.trail.append(self.position)

    def distance_to_point(self, point: Point) -> float:
        return math.dist(self.position, point)

    def distance_to_agent(self, other: "GenericAgent") -> float:
        return math.dist(self.position, other.position)

    def is_at_point(self, point: Point) -> bool:
        return self.distance_to_point(point) <= config.ARRIVAL_TOLERANCE_METERS

    def is_task_busy(self) -> bool:
        return False

    def set_heading_toward(self, target: Point) -> None:
        if math.dist(self.position, target) <= 1e-9:
            return
        self.next_heading = math.atan2(target[1] - self.position[1], target[0] - self.position[0])

    @abstractmethod
    def perceive(self, simulation: "Simulation") -> Any:
        raise NotImplementedError

    @abstractmethod
    def attend(self, perception: Any, simulation: "Simulation") -> Any:
        raise NotImplementedError

    @abstractmethod
    def act(self, attention: Any, simulation: "Simulation") -> None:
        raise NotImplementedError

    @abstractmethod
    def state_summary(self) -> str:
        raise NotImplementedError


class Patient(GenericAgent):
    """Passive patient that advances through the ED checklist."""

    def __init__(
        self,
        gid: int,
        position: Point,
        esi_level: int,
        spawned_at: int,
        complaint_type: str = "unspecified",
        urgency: str = "routine",
        expected_resources: Optional[list[str]] = None,
        likely_pathway: str = "standard_ed_care",
        required_staff_roles: Optional[list[str]] = None,
        high_acuity_escalation: bool = False,
    ) -> None:
        super().__init__(gid=gid, role="Patient", position=position)
        self.esi_level = esi_level
        self.complaint_type = complaint_type
        self.urgency = urgency
        self.expected_resources = [] if expected_resources is None else list(expected_resources)
        self.likely_pathway = likely_pathway
        self.required_staff_roles = [] if required_staff_roles is None else list(required_staff_roles)
        self.high_acuity_escalation = high_acuity_escalation
        self.spawned_at = spawned_at
        self.current_task_started_at = spawned_at
        self.task_index = config.PLACEMENT_TASK_INDEX
        self.bed_index: Optional[int] = None
        self.assigned_care_position: Optional[Point] = None
        self.assigned_nurse_index: Optional[int] = None
        self.claiming_staff_id: Optional[int] = None
        self.requested_doctor_id: Optional[int] = None
        self.requesting_nurse_id: Optional[int] = None
        self.coordination_hold_until = spawned_at
        self.coordination_hold_reason: Optional[str] = None
        self.coordination_hold_after_task: Optional[str] = None
        self.escort_corridor_interaction_logged = False
        self.discharged = False

    @property
    def bed_position(self) -> Optional[Point]:
        if self.assigned_care_position is not None:
            return self.assigned_care_position
        if self.bed_index is None:
            return None
        return config.BED_POSITIONS[self.bed_index]

    @property
    def current_task_name(self) -> Optional[str]:
        if self.task_index >= len(config.CHECKLIST):
            return None
        return config.CHECKLIST[self.task_index]

    def waiting_for_placement(self) -> bool:
        return self.task_index == config.PLACEMENT_TASK_INDEX and self.bed_index is None

    def coordination_hold_active(self, current_timestep: int) -> bool:
        return (
            not self.discharged
            and self.coordination_hold_until is not None
            and current_timestep < self.coordination_hold_until
        )

    def needs_doctor_request(self, current_timestep: Optional[int] = None) -> bool:
        if current_timestep is not None and self.coordination_hold_active(current_timestep):
            return False
        return (
            self.task_index == config.MEDICAL_EVALUATION_TASK_INDEX
            and self.requested_doctor_id is None
            and not self.discharged
        )

    def seconds_at_current_task(self, current_timestep: int) -> int:
        return max(current_timestep - self.current_task_started_at, 0)

    def complete_current_task(self, current_timestep: int) -> None:
        self.task_index += 1
        self.current_task_started_at = current_timestep
        self.claiming_staff_id = None
        self.requested_doctor_id = None
        self.requesting_nurse_id = None
        self.coordination_hold_until = current_timestep
        self.coordination_hold_reason = None
        self.coordination_hold_after_task = None
        if self.task_index >= len(config.CHECKLIST):
            self.discharged = True

    def perceive(self, simulation: "Simulation") -> None:
        return None

    def attend(self, perception: None, simulation: "Simulation") -> None:
        return None

    def act(self, attention: None, simulation: "Simulation") -> None:
        return None

    def state_summary(self) -> str:
        if self.discharged:
            return "discharged"
        if self.bed_index is None:
            return f"ESI {self.esi_level} waiting for {self.current_task_name}"
        return f"ESI {self.esi_level} bed {self.bed_index + 1}: {self.current_task_name}"


class StaffAgent(GenericAgent, ABC):
    """Base class for mobile staff with task timers, routing, and idle wander."""

    def __init__(
        self,
        gid: int,
        role: str,
        position: Point,
        home_position: Point,
        home_zone_id: str,
        idle_position_radius: float,
    ) -> None:
        super().__init__(gid=gid, role=role, position=position)
        self.home_position = home_position
        self.home_zone_id = home_zone_id
        self.idle_position_radius = idle_position_radius

        self.mode = "idle"
        self.next_mode = self.mode
        self.current_task_name: Optional[str] = None
        self.next_current_task_name = self.current_task_name
        self.task_remaining = 0
        self.next_task_remaining = self.task_remaining
        self.target_patient_id: Optional[int] = None
        self.next_target_patient_id = self.target_patient_id
        self.destination: Optional[Point] = position
        self.next_destination = self.destination
        self.task_completed_this_step = False
        self.next_task_completed_this_step = False

        self.idle_zone_id = home_zone_id
        self.next_idle_zone_id = self.idle_zone_id
        self.last_idle_position: Optional[Point] = position
        self.next_last_idle_position = self.last_idle_position
        self.needs_new_idle_target = False
        self.next_needs_new_idle_target = self.needs_new_idle_target
        self.idle_seconds = 0
        self.next_idle_seconds = self.idle_seconds
        self.patrol_destination: Optional[Point] = None
        self.next_patrol_destination = self.patrol_destination
        self.post_task_station_check_zone_id: Optional[str] = None
        self.next_post_task_station_check_zone_id = self.post_task_station_check_zone_id
        self.post_task_station_check_remaining = 0
        self.next_post_task_station_check_remaining = self.post_task_station_check_remaining

        # Agents keep a committed waypoint route until the destination changes.
        # This prevents the left-bed nurse from replanning each intermediate
        # corridor position and bouncing forever between corridor hubs.
        self.route_plan: List[Point] = []
        self.next_route_plan = list(self.route_plan)
        self.route_destination: Optional[Point] = None
        self.next_route_destination = self.route_destination
        self.stalled_seconds = 0
        self.next_stalled_seconds = self.stalled_seconds
        self.movement_progress_this_step = False
        self.next_movement_progress_this_step = False
        self.trip_label: Optional[str] = None
        self.next_trip_label = self.trip_label
        self.trip_distance_meters = 0.0
        self.next_trip_distance_meters = self.trip_distance_meters
        self.trip_started_at: Optional[int] = None
        self.next_trip_started_at = self.trip_started_at
        self.name = f"{role} {gid}"
        self.persona = None
        self.persona_source_trace = "unassigned"
        self.memory_stream = None
        self.reflection_summaries: list[dict] = []

    def begin_step(self) -> None:
        super().begin_step()
        self.next_mode = self.mode
        self.next_current_task_name = self.current_task_name
        self.next_task_remaining = self.task_remaining
        self.next_target_patient_id = self.target_patient_id
        self.next_destination = self.destination
        self.next_task_completed_this_step = False
        self.next_idle_zone_id = self.idle_zone_id
        self.next_last_idle_position = self.last_idle_position
        self.next_needs_new_idle_target = self.needs_new_idle_target
        self.next_idle_seconds = self.idle_seconds
        self.next_patrol_destination = self.patrol_destination
        self.next_post_task_station_check_zone_id = self.post_task_station_check_zone_id
        self.next_post_task_station_check_remaining = self.post_task_station_check_remaining
        self.next_route_plan = list(self.route_plan)
        self.next_route_destination = self.route_destination
        self.next_stalled_seconds = self.stalled_seconds
        self.next_movement_progress_this_step = False
        self.next_trip_label = self.trip_label
        self.next_trip_distance_meters = self.trip_distance_meters
        self.next_trip_started_at = self.trip_started_at

    def update(self) -> None:
        previous_position = self.position
        super().update()
        self.mode = self.next_mode
        self.current_task_name = self.next_current_task_name
        self.task_remaining = self.next_task_remaining
        self.target_patient_id = self.next_target_patient_id
        self.destination = self.next_destination
        self.task_completed_this_step = self.next_task_completed_this_step
        self.idle_zone_id = self.next_idle_zone_id
        self.last_idle_position = self.next_last_idle_position
        self.needs_new_idle_target = self.next_needs_new_idle_target
        self.idle_seconds = self.next_idle_seconds
        self.patrol_destination = self.next_patrol_destination
        self.post_task_station_check_zone_id = self.next_post_task_station_check_zone_id
        self.post_task_station_check_remaining = self.next_post_task_station_check_remaining
        self.route_plan = list(self.next_route_plan)
        self.route_destination = self.next_route_destination
        self.stalled_seconds = self.next_stalled_seconds
        self.movement_progress_this_step = self.next_movement_progress_this_step
        self.trip_label = self.next_trip_label
        self.trip_distance_meters = self.next_trip_distance_meters
        self.trip_started_at = self.next_trip_started_at

        monitored_modes = {
            "moving_to_patient",
            "moving_to_doctor",
            "leading_doctor_to_patient",
            "returning_home",
            "moving_to_secondary_station",
            "moving_to_post_task_station_check",
            "awaiting_nurse",
            "patrolling",
        }
        if (
            self.mode in monitored_modes
            and math.dist(previous_position, self.position) <= 1e-9
            and not self.movement_progress_this_step
        ):
            self.stalled_seconds += config.TIMESTEP_SECONDS
        else:
            self.stalled_seconds = 0
        self.next_stalled_seconds = self.stalled_seconds
        if self.trip_label is not None:
            self.trip_distance_meters += math.dist(previous_position, self.position)
        self.next_trip_distance_meters = self.trip_distance_meters
        if self.mode == "idle":
            self.idle_seconds += config.TIMESTEP_SECONDS
        else:
            self.idle_seconds = 0
        self.next_idle_seconds = self.idle_seconds

    def is_task_busy(self) -> bool:
        return self.current_task_name is not None and self.task_remaining > 0

    def is_available(self) -> bool:
        return not self.is_task_busy() and self.mode in {"idle", "returning_home"}

    def sample_task_duration(self, simulation: "Simulation", task_name: str, patient: Optional[Patient] = None) -> int:
        lower_bound, upper_bound = config.TASK_DURATIONS_SECONDS[task_name]
        sampled_duration = simulation.random.randint(lower_bound, upper_bound)
        if patient is not None:
            sampled_duration = int(
                round(
                    sampled_duration
                    * float(config.ESI_TASK_DURATION_MULTIPLIERS.get(getattr(patient, "esi_level", 3), 1.0))
                )
            )
        if self.role in {"Nurse", "Doctor"}:
            sampled_duration = int(round(sampled_duration * config.BEDSIDE_LINGER_MULTIPLIER))
        return max(sampled_duration, config.TIMESTEP_SECONDS)

    def _workflow_interaction_type(self, task_name: str) -> str:
        del task_name
        if self.role == "CoordinationNurse":
            return "placement_handoff"
        if self.role == "Nurse":
            return "nursing_task"
        if self.role == "Doctor":
            return "doctor_task"
        return "workflow_task"

    def begin_trip(self, label: str, simulation: "Simulation") -> None:
        if self.trip_label == label and self.trip_started_at is not None:
            return
        self.next_trip_label = label
        self.next_trip_distance_meters = 0.0
        self.next_trip_started_at = simulation.timestep

    def complete_trip(self, simulation: "Simulation") -> None:
        if self.trip_label is None or self.trip_started_at is None:
            self.next_trip_label = None
            self.next_trip_distance_meters = 0.0
            self.next_trip_started_at = None
            return

        simulation.record_trip_completion(
            label=self.trip_label,
            distance_meters=self.trip_distance_meters,
            duration_seconds=max(simulation.timestep - self.trip_started_at, 0),
        )
        self.next_trip_label = None
        self.next_trip_distance_meters = 0.0
        self.next_trip_started_at = None

    def start_task(self, simulation: "Simulation", patient: Patient) -> None:
        task_name = patient.current_task_name
        if task_name is None:
            self.clear_assignment()
            return

        self.complete_trip(simulation)
        simulation.log_workflow_event(
            agent=self,
            patient=patient,
            event_type="task_started",
            task_name=task_name,
            old_state=patient.current_task_name,
            new_state="task_in_progress",
            position=patient.position,
            notes=f"{self.role} started {task_name} for patient {patient.gid}",
        )
        communicative_type = self._workflow_interaction_type(task_name)
        task_probabilities = config.PATIENT_FACING_TASK_INTERACTION_PROBABILITY.get(self.role, {})
        communication_probability = float(task_probabilities.get(task_name, 0.0))
        if getattr(patient, "esi_level", None) in {1, 2}:
            communication_probability = min(
                1.0,
                communication_probability + config.HIGH_ACUITY_PATIENT_FACING_BONUS,
            )
        if (
            communicative_type in {"placement_handoff", "nursing_task", "doctor_task"}
            and simulation.should_log_patient_facing_task_interaction(
                self,
                patient,
                task_name,
                communicative_type,
                communication_probability,
            )
        ):
            interaction_position = simulation.patient_facing_task_interaction_position(
                self,
                patient,
                task_name,
            )
            simulation.log_interaction(
                agent_1=self,
                agent_2=patient,
                position=interaction_position,
                interaction_type=communicative_type,
                task_name=task_name,
                reason_for_interaction=f"patient_facing_{task_name}",
            )
        self.next_mode = "performing_task"
        self.next_current_task_name = task_name
        self.next_task_remaining = self.sample_task_duration(simulation, task_name, patient)
        self.next_target_patient_id = patient.gid
        self.next_destination = patient.position
        self.next_route_plan = []
        self.next_route_destination = None

    def continue_task(self, simulation: "Simulation") -> None:
        patient = simulation.patient_by_id.get(self.target_patient_id)
        if patient is not None:
            simulation.maybe_log_patient_facing_in_task_interaction(self, patient)
        remaining = max(self.task_remaining - config.TIMESTEP_SECONDS, 0)
        self.next_task_remaining = remaining
        self.next_position = self.position
        self.next_mode = "performing_task"
        self.next_current_task_name = self.current_task_name
        self.next_route_plan = []
        self.next_route_destination = None

        if remaining == 0:
            self.next_task_completed_this_step = True

    def mark_idle_target_dirty(self) -> None:
        self.destination = None
        self.next_destination = None
        self.needs_new_idle_target = True
        self.next_needs_new_idle_target = True
        self.patrol_destination = None
        self.next_patrol_destination = None
        self.route_plan = []
        self.next_route_plan = []
        self.route_destination = None
        self.next_route_destination = None

    def clear_assignment(self) -> None:
        self.mode = "returning_home"
        self.next_mode = "returning_home"
        self.current_task_name = None
        self.next_current_task_name = None
        self.task_remaining = 0
        self.next_task_remaining = 0
        self.target_patient_id = None
        self.next_target_patient_id = None
        self.task_completed_this_step = False
        self.next_task_completed_this_step = False
        self.stalled_seconds = 0
        self.next_stalled_seconds = 0
        self.trip_label = None
        self.next_trip_label = None
        self.trip_distance_meters = 0.0
        self.next_trip_distance_meters = 0.0
        self.trip_started_at = None
        self.next_trip_started_at = None
        self.idle_seconds = 0
        self.next_idle_seconds = 0
        self.patrol_destination = None
        self.next_patrol_destination = None
        self.post_task_station_check_zone_id = None
        self.next_post_task_station_check_zone_id = None
        self.post_task_station_check_remaining = 0
        self.next_post_task_station_check_remaining = 0
        self.mark_idle_target_dirty()

    def start_post_task_station_check(
        self,
        simulation: "Simulation",
        zone_id: str,
        destination: Point,
        duration_seconds: int,
    ) -> None:
        del simulation
        dwell_seconds = max(duration_seconds, config.TIMESTEP_SECONDS)
        self.mode = "moving_to_post_task_station_check"
        self.next_mode = "moving_to_post_task_station_check"
        self.destination = destination
        self.next_destination = destination
        self.post_task_station_check_zone_id = zone_id
        self.next_post_task_station_check_zone_id = zone_id
        self.post_task_station_check_remaining = dwell_seconds
        self.next_post_task_station_check_remaining = dwell_seconds
        self.route_plan = []
        self.next_route_plan = []
        self.route_destination = None
        self.next_route_destination = None

    def _clear_post_task_station_check(self) -> None:
        self.post_task_station_check_zone_id = None
        self.next_post_task_station_check_zone_id = None
        self.post_task_station_check_remaining = 0
        self.next_post_task_station_check_remaining = 0

    def continue_post_task_station_check(self, simulation: "Simulation") -> bool:
        if self.mode not in {"moving_to_post_task_station_check", "post_task_station_check"}:
            return False

        destination = self.destination
        if destination is None:
            self._clear_post_task_station_check()
            self.mark_idle_target_dirty()
            self.return_home(simulation)
            return True

        if self.mode == "moving_to_post_task_station_check":
            if self.is_at_point(destination):
                self.next_position = destination
                self.next_mode = "post_task_station_check"
                self.next_last_idle_position = destination
                self.next_route_plan = []
                self.next_route_destination = None
                return True
            self.next_mode = "moving_to_post_task_station_check"
            self.move_toward(simulation, destination)
            return True

        remaining = max(self.post_task_station_check_remaining - config.TIMESTEP_SECONDS, 0)
        self.next_post_task_station_check_remaining = remaining
        self.next_position = self.position
        self.next_route_plan = []
        self.next_route_destination = None
        if remaining == 0:
            self._clear_post_task_station_check()
            simulation.ablation_diagnostics["post_task_station_check_completed_count"] += 1
            self.mark_idle_target_dirty()
            self.return_home(simulation)
            return True
        self.next_mode = "post_task_station_check"
        return True

    def _idle_zone_and_radius(self, simulation: "Simulation") -> tuple[str, float]:
        del simulation
        return (self.home_zone_id, self.idle_position_radius)

    def _assign_idle_destination(
        self,
        simulation: "Simulation",
        zone_id: str,
        radius_meters: float,
        center_point: Optional[Point] = None,
    ) -> Point:
        if zone_id in simulation.station_interaction_zone_ids():
            idle_point = simulation.sample_station_position(
                zone_id=zone_id,
                avoid_point=self.last_idle_position,
            )
        else:
            idle_point = simulation.sample_zone_position(
                zone_id=zone_id,
                radius_meters=radius_meters,
                avoid_point=self.last_idle_position,
                center_point=center_point,
            )
        self.next_destination = idle_point
        self.next_idle_zone_id = zone_id
        self.next_needs_new_idle_target = False
        self.next_route_plan = []
        self.next_route_destination = None
        return idle_point

    def _has_valid_idle_destination(self, zone_id: str) -> bool:
        return (
            self.destination is not None
            and not self.needs_new_idle_target
            and self.idle_zone_id == zone_id
        )

    def _select_patrol_destination(self, simulation: "Simulation") -> Optional[Point]:
        del simulation
        return None

    def start_patrol(self, simulation: "Simulation") -> bool:
        patrol_destination = self._select_patrol_destination(simulation)
        if patrol_destination is None:
            return False
        self.next_patrol_destination = patrol_destination
        self.next_destination = patrol_destination
        self.next_mode = "patrolling"
        self.next_idle_seconds = 0
        self.next_route_plan = []
        self.next_route_destination = None
        self.begin_trip(f"{self.role} Patrol", simulation)
        return True

    def continue_patrol(self, simulation: "Simulation") -> bool:
        if self.mode != "patrolling":
            return False

        patrol_destination = self.patrol_destination or self.destination
        if patrol_destination is None:
            self.next_patrol_destination = None
            self.mark_idle_target_dirty()
            self.return_home(simulation)
            return True

        if self.is_at_point(patrol_destination):
            self.complete_trip(simulation)
            self.next_patrol_destination = None
            self.mark_idle_target_dirty()
            self.return_home(simulation)
            return True

        self.next_mode = "patrolling"
        self.next_destination = patrol_destination
        self.next_patrol_destination = patrol_destination
        self.move_toward(simulation, patrol_destination)
        return True

    def _route_matches_destination(self, destination: Point) -> bool:
        if self.route_destination is None:
            return False
        return math.dist(self.route_destination, destination) <= config.ARRIVAL_TOLERANCE_METERS

    def _semantic_destination_type(self) -> str:
        mode = self.next_mode or self.mode
        if mode in {"moving_to_patient", "leading_doctor_to_patient"}:
            return "patient_bed_access"
        if mode == "moving_to_doctor":
            return "handoff"
        if mode == "waiting_for_doctor":
            return "doctor_recheck_wait"
        if mode == "returning_home":
            return "station_home"
        if mode == "patrolling":
            return "patrol"
        if mode in {"moving_to_secondary_station", "using_secondary_station"}:
            return "secondary_station"
        if mode in {"moving_to_post_task_station_check", "post_task_station_check"}:
            return "post_task_station_check"
        return str(mode or "movement")

    def _route_valid_from_current(self, simulation: "Simulation", route: List[Point]) -> bool:
        if not route:
            return False
        if hasattr(simulation, "route_is_walkable"):
            return bool(simulation.route_is_walkable(tuple(self.position), route))
        previous = self.position
        for waypoint in route:
            if simulation.wall_collision(previous, waypoint):
                return False
            previous = waypoint
        return True

    def _record_movement_route_issue(
        self,
        simulation: "Simulation",
        destination: Point,
        *,
        reason: str,
        cached_route_used: bool,
        cached_route_failed_validation: bool,
        replan_attempted: bool,
        replan_succeeded: bool,
        immediate_step_collided: bool,
        route: Optional[List[Point]] = None,
        next_waypoint: Optional[Point] = None,
    ) -> None:
        if hasattr(simulation, "record_movement_route_issue"):
            simulation.record_movement_route_issue(
                self,
                destination,
                reason=reason,
                semantic_destination_type=self._semantic_destination_type(),
                cached_route_used=cached_route_used,
                cached_route_failed_validation=cached_route_failed_validation,
                replan_attempted=replan_attempted,
                replan_succeeded=replan_succeeded,
                immediate_step_collided=immediate_step_collided,
                route=route,
                next_waypoint=next_waypoint,
            )

    def _validated_route_to(
        self,
        simulation: "Simulation",
        destination: Point,
        *,
        use_cached: bool = True,
    ) -> tuple[List[Point], dict[str, bool]]:
        flags = {
            "cached_route_used": False,
            "cached_route_failed_validation": False,
            "replan_attempted": False,
            "replan_succeeded": False,
        }
        route: List[Point] = []
        if use_cached and self._route_matches_destination(destination) and self.route_plan:
            flags["cached_route_used"] = True
            route = list(self.route_plan)
            if route and self._route_valid_from_current(simulation, route):
                return route, flags
            flags["cached_route_failed_validation"] = True

        flags["replan_attempted"] = True
        route = simulation.plan_route(self.position, destination)
        if route and self._route_valid_from_current(simulation, route):
            flags["replan_succeeded"] = True
            self.next_movement_progress_this_step = True
            return route, flags

        direct_route = [destination]
        if self._route_valid_from_current(simulation, direct_route):
            flags["replan_succeeded"] = True
            self.next_movement_progress_this_step = True
            return direct_route, flags

        return [], flags

    def move_toward(self, simulation: "Simulation", destination: Point) -> None:
        route, flags = self._validated_route_to(simulation, destination)
        if not route:
            self._record_movement_route_issue(
                simulation,
                destination,
                reason="no_walkable_route",
                immediate_step_collided=False,
                route=route,
                next_waypoint=None,
                **flags,
            )
            self.next_position = self.position
            self.next_route_plan = []
            self.next_route_destination = None
            return

        next_waypoint = route[0]
        distance = self.distance_to_point(next_waypoint)

        if distance <= config.ARRIVAL_TOLERANCE_METERS:
            self.next_position = next_waypoint
            self.set_heading_toward(next_waypoint)
            route.pop(0)
            self.next_route_plan = route
            self.next_route_destination = destination
            self.next_movement_progress_this_step = True
            return

        step_length = min(config.MOVEMENT_SPEED_METERS_PER_SECOND, distance)
        dx = (next_waypoint[0] - self.position[0]) / distance
        dy = (next_waypoint[1] - self.position[1]) / distance
        candidate_position = (
            self.position[0] + (dx * step_length),
            self.position[1] + (dy * step_length),
        )

        if simulation.wall_collision(self.position, candidate_position):
            refreshed_route, refreshed_flags = self._validated_route_to(
                simulation,
                destination,
                use_cached=False,
            )
            combined_flags = {
                "cached_route_used": flags["cached_route_used"],
                "cached_route_failed_validation": flags["cached_route_failed_validation"],
                "replan_attempted": True,
                "replan_succeeded": refreshed_flags["replan_succeeded"],
            }
            if refreshed_route:
                next_waypoint = refreshed_route[0]
                distance = self.distance_to_point(next_waypoint)
                if distance <= config.ARRIVAL_TOLERANCE_METERS:
                    refreshed_route.pop(0)
                    self.next_position = next_waypoint
                    self.set_heading_toward(next_waypoint)
                    self.next_route_plan = refreshed_route
                    self.next_route_destination = destination
                    self.next_movement_progress_this_step = True
                    return
                step_length = min(config.MOVEMENT_SPEED_METERS_PER_SECOND, distance)
                dx = (next_waypoint[0] - self.position[0]) / distance
                dy = (next_waypoint[1] - self.position[1]) / distance
                refreshed_candidate = (
                    self.position[0] + (dx * step_length),
                    self.position[1] + (dy * step_length),
                )
                if not simulation.wall_collision(self.position, refreshed_candidate):
                    self.next_position = refreshed_candidate
                    self.set_heading_toward(next_waypoint)
                    if math.dist(refreshed_candidate, next_waypoint) <= config.ARRIVAL_TOLERANCE_METERS:
                        refreshed_route.pop(0)
                    self.next_route_plan = refreshed_route
                    self.next_route_destination = destination
                    self.next_movement_progress_this_step = True
                    return

            self._record_movement_route_issue(
                simulation,
                destination,
                reason="immediate_step_collision",
                immediate_step_collided=True,
                route=route,
                next_waypoint=next_waypoint,
                **combined_flags,
            )
            self.next_position = self.position
            self.next_route_plan = []
            self.next_route_destination = None
            return

        self.next_position = candidate_position
        self.set_heading_toward(next_waypoint)
        if math.dist(candidate_position, next_waypoint) <= config.ARRIVAL_TOLERANCE_METERS:
            route.pop(0)
            self.next_movement_progress_this_step = True

        self.next_route_plan = route
        self.next_route_destination = destination

    def return_home(self, simulation: "Simulation") -> None:
        self.next_mode = "returning_home"
        self.next_current_task_name = None
        self.next_task_remaining = 0
        self.next_target_patient_id = None

        zone_id, radius_meters = self._idle_zone_and_radius(simulation)
        if self._has_valid_idle_destination(zone_id):
            destination = self.destination
        else:
            center_point = self.home_position if zone_id == self.home_zone_id else None
            destination = self._assign_idle_destination(
                simulation,
                zone_id,
                radius_meters,
                center_point=center_point,
            )

        if destination is None:
            destination = self.home_position
            self.next_destination = destination

        if self.is_at_point(destination):
            self.next_position = destination
            self.next_mode = "idle"
            if self.mode != "idle":
                self.next_last_idle_position = destination
                self.next_idle_seconds = 0
            else:
                self.next_last_idle_position = self.last_idle_position
                self.next_idle_seconds = self.idle_seconds
            self.next_route_plan = []
            self.next_route_destination = None
            return

        current_zone_id = simulation.which_zone(*self.position)
        travel_target = (
            destination
            if zone_id in simulation.station_interaction_zone_ids()
            else self.home_position if current_zone_id != self.home_zone_id else destination
        )
        self.move_toward(simulation, travel_target)

    def state_summary(self) -> str:
        if self.is_task_busy():
            return f"{self.current_task_name} ({self.task_remaining}s)"
        if self.mode == "post_task_station_check" and self.post_task_station_check_zone_id is not None:
            return f"checking_{self.post_task_station_check_zone_id} ({self.post_task_station_check_remaining}s)"
        return self.mode


class CoordinationNurse(StaffAgent):
    """Single staff member that handles entry placement and bed escort."""

    def _select_patrol_destination(self, simulation: "Simulation") -> Optional[Point]:
        candidate_points = [
            config.ROUTING_WAYPOINTS[waypoint_key]
            for waypoint_key in config.COORDINATION_PATROL_ROUTE_KEYS
            if waypoint_key in config.ROUTING_WAYPOINTS
        ]
        viable_points = [
            point for point in candidate_points
            if math.dist(point, self.position) > config.ARRIVAL_TOLERANCE_METERS
        ]
        if not viable_points:
            return None
        return simulation.random.choice(viable_points)

    def perceive(self, simulation: "Simulation") -> Any:
        workflow_candidates = [
            patient for patient in simulation.active_patients if patient.waiting_for_placement()
        ]
        return simulation.perception.build_bundle(self, simulation, workflow_candidates)

    def attend(self, perception: Any, simulation: "Simulation") -> Optional[Patient]:
        if self.target_patient_id is not None:
            return simulation.patient_by_id.get(self.target_patient_id)

        eligible_patients = [
            patient
            for patient in perception.workflow_candidates
            if simulation.find_available_bed_index(patient) is not None
        ]

        selected = min(
            eligible_patients,
            key=lambda patient: (getattr(patient, "esi_level", 5), patient.spawned_at),
            default=None,
        )
        simulation.record_esi_deprioritization(selected, eligible_patients, "bed")
        return selected

    def act(self, attention: Optional[Patient], simulation: "Simulation") -> None:
        if self.current_task_name == config.PLACEMENT_TASK_NAME and self.target_patient_id is not None:
            patient = simulation.patient_by_id.get(self.target_patient_id)
            if patient is None:
                self.clear_assignment()
                return

            bed_position = patient.bed_position
            remaining = max(self.task_remaining - config.TIMESTEP_SECONDS, 0)
            self.next_task_remaining = remaining
            self.next_mode = "performing_task"
            self.next_current_task_name = self.current_task_name
            self.next_task_completed_this_step = False

            if bed_position is not None and not self.is_at_point(bed_position):
                self.move_toward(simulation, bed_position)
                patient.next_position = self.next_position
                simulation.maybe_log_escort_corridor_interaction(self, patient)
            elif bed_position is not None:
                self.next_position = bed_position
                patient.next_position = bed_position

            if (
                bed_position is not None
                and math.dist(self.next_position, bed_position) <= config.ARRIVAL_TOLERANCE_METERS
                and remaining == 0
            ):
                self.next_task_completed_this_step = True
            return

        if self.is_task_busy():
            self.continue_task(simulation)
            return

        if self.target_patient_id is not None and self.mode == "moving_to_patient":
            patient = simulation.patient_by_id.get(self.target_patient_id)
            if patient is None:
                self.clear_assignment()
                return

            if self.is_at_point(patient.position):
                self.start_task(simulation, patient)
                return

            self.next_mode = "moving_to_patient"
            self.next_destination = patient.position
            self.move_toward(simulation, patient.position)
            return

        if self.continue_patrol(simulation):
            return

        if attention is None:
            if simulation.random.random() < config.COORDINATION_PATROL_PROBABILITY and self.start_patrol(simulation):
                return
            self.return_home(simulation)
            return

        if attention.bed_index is None:
            assigned_bed_index = simulation.assign_bed_to_patient(attention)
            if assigned_bed_index is None:
                self.return_home(simulation)
                return

        attention.claiming_staff_id = self.gid
        self.next_mode = "moving_to_patient"
        self.next_target_patient_id = attention.gid
        self.next_destination = attention.position
        self.begin_trip(config.PLACEMENT_TASK_NAME, simulation)
        self.move_toward(simulation, attention.position)


class Nurse(StaffAgent):
    """Bed-oriented nurse with explicit doctor handoff and station drift."""

    def __init__(
        self,
        gid: int,
        position: Point,
        home_position: Point,
        home_zone_id: str,
        covered_bed_indices: List[int],
    ) -> None:
        super().__init__(
            gid=gid,
            role="Nurse",
            position=position,
            home_position=home_position,
            home_zone_id=home_zone_id,
            idle_position_radius=config.HOME_POSITION_WANDER_RADIUS,
        )
        self.covered_bed_indices = covered_bed_indices
        self.recheck_timer = 0
        self.next_recheck_timer = 0
        self.reserved_doctor_id: Optional[int] = None
        self.next_reserved_doctor_id = self.reserved_doctor_id
        self.doctor_recheck_patient_id: Optional[int] = None
        self.next_doctor_recheck_patient_id = self.doctor_recheck_patient_id
        self.doctor_recheck_failures = 0
        self.next_doctor_recheck_failures = self.doctor_recheck_failures
        self.secondary_station_zone_id: Optional[str] = None
        self.next_secondary_station_zone_id = self.secondary_station_zone_id
        self.secondary_station_remaining = 0
        self.next_secondary_station_remaining = self.secondary_station_remaining

    def begin_step(self) -> None:
        super().begin_step()
        self.next_recheck_timer = self.recheck_timer
        self.next_reserved_doctor_id = self.reserved_doctor_id
        self.next_doctor_recheck_patient_id = self.doctor_recheck_patient_id
        self.next_doctor_recheck_failures = self.doctor_recheck_failures
        self.next_secondary_station_zone_id = self.secondary_station_zone_id
        self.next_secondary_station_remaining = self.secondary_station_remaining

    def update(self) -> None:
        super().update()
        self.recheck_timer = self.next_recheck_timer
        self.reserved_doctor_id = self.next_reserved_doctor_id
        self.doctor_recheck_patient_id = self.next_doctor_recheck_patient_id
        self.doctor_recheck_failures = self.next_doctor_recheck_failures
        self.secondary_station_zone_id = self.next_secondary_station_zone_id
        self.secondary_station_remaining = self.next_secondary_station_remaining

    def clear_doctor_wait_state(self) -> None:
        self.recheck_timer = 0
        self.next_recheck_timer = 0
        self.reserved_doctor_id = None
        self.next_reserved_doctor_id = None
        self.doctor_recheck_patient_id = None
        self.next_doctor_recheck_patient_id = None
        self.doctor_recheck_failures = 0
        self.next_doctor_recheck_failures = 0

    def _covered_patients(self, simulation: "Simulation") -> List[Patient]:
        visible_patients: List[Patient] = []
        for patient in simulation.active_patients:
            if patient.bed_index not in self.covered_bed_indices:
                continue
            if patient.discharged:
                continue
            visible_patients.append(patient)
        return visible_patients

    def _random_secondary_station_zone(self, simulation: "Simulation") -> Optional[str]:
        alternative_zones = [
            zone_id for zone_id in simulation.station_dwell_zone_ids() if zone_id != self.home_zone_id
        ]
        if not alternative_zones:
            return None

        return min(
            alternative_zones,
            key=lambda zone_id: math.dist(
                simulation.condition_manager.station_attractor(zone_id),
                self.position,
            ),
        )

    def _select_patrol_destination(self, simulation: "Simulation") -> Optional[Point]:
        if self.covered_bed_indices and simulation.random.random() < config.NURSE_PATROL_TO_BED_PROBABILITY:
            care_positions = simulation.patient_care_positions()
            candidate_beds = [
                care_positions[bed_index]
                for bed_index in self.covered_bed_indices
                if bed_index < len(care_positions)
                and math.dist(care_positions[bed_index], self.position) > config.ARRIVAL_TOLERANCE_METERS
            ]
            if candidate_beds:
                return simulation.random.choice(candidate_beds)

        corridor_points = [
            config.ROUTING_WAYPOINTS[waypoint_key]
            for waypoint_key in config.NURSE_CIRCULATION_ROUTE_KEYS_BY_HOME.get(self.home_zone_id, [])
            if waypoint_key in config.ROUTING_WAYPOINTS
        ]
        viable_corridor_points = [
            point for point in corridor_points
            if math.dist(point, self.position) > config.ARRIVAL_TOLERANCE_METERS
        ]
        if viable_corridor_points:
            return simulation.random.choice(viable_corridor_points)

        candidate_points = [
            simulation.condition_manager.station_attractor(zone_id)
            for zone_id in simulation.station_dwell_zone_ids()
            if zone_id != self.home_zone_id
        ]
        viable_points = [
            point for point in candidate_points
            if math.dist(point, self.position) > config.ARRIVAL_TOLERANCE_METERS
        ]
        if not viable_points:
            return None
        return min(viable_points, key=lambda point: math.dist(point, self.position))

    def _clear_secondary_station_visit(self) -> None:
        self.secondary_station_zone_id = None
        self.next_secondary_station_zone_id = None
        self.secondary_station_remaining = 0
        self.next_secondary_station_remaining = 0

    def _maybe_start_secondary_station_visit(self, simulation: "Simulation") -> bool:
        if self.secondary_station_zone_id is not None:
            return False
        if simulation.random.random() >= config.NURSE_SECONDARY_STATION_PROBABILITY:
            return False

        secondary_zone_id = self._random_secondary_station_zone(simulation)
        if secondary_zone_id is None:
            return False

        self.next_secondary_station_zone_id = secondary_zone_id
        lower_bound, upper_bound = config.NURSE_SECONDARY_STATION_DURATION
        if config.SECONDARY_STATION_DURATION_MEAN > 0:
            half_window = max(int(config.SECONDARY_STATION_DURATION_MEAN * 0.5), 10)
            lower_bound = max(int(config.SECONDARY_STATION_DURATION_MEAN - half_window), 10)
            upper_bound = int(config.SECONDARY_STATION_DURATION_MEAN + half_window)
        self.next_secondary_station_remaining = simulation.random.randint(lower_bound, upper_bound)
        self._assign_idle_destination(
            simulation,
            zone_id=secondary_zone_id,
            radius_meters=config.HOME_POSITION_WANDER_RADIUS,
            center_point=simulation.condition_manager.station_attractor(secondary_zone_id),
        )
        self.next_mode = "moving_to_secondary_station"
        return True

    def perceive(self, simulation: "Simulation") -> Any:
        return simulation.perception.build_bundle(
            self,
            simulation,
            workflow_candidates=self._covered_patients(simulation),
        )

    def attend(self, perception: Any, simulation: "Simulation") -> Any:
        if self.target_patient_id is not None:
            return simulation.patient_by_id.get(self.target_patient_id)

        candidate_tasks = [
            patient
            for patient in perception.workflow_candidates
            if patient.task_index in config.ROLE_PERMISSIONS[self.role]
            and patient.claiming_staff_id is None
            and not patient.coordination_hold_active(simulation.timestep)
            and simulation.patient_assignment_route_supported(self, patient)
        ]
        candidate_tasks.sort(
            key=lambda patient: (
                getattr(patient, "esi_level", 5),
                -patient.task_index,
                patient.spawned_at,
            )
        )
        if candidate_tasks:
            selected = candidate_tasks[0]
            simulation.record_esi_deprioritization(selected, candidate_tasks, "nurse")
            return ("nurse_task", selected)

        doctor_requests = [
            patient
            for patient in perception.workflow_candidates
            if patient.needs_doctor_request(simulation.timestep)
            and (
                patient.gid != self.doctor_recheck_patient_id
                or self.recheck_timer <= 0
            )
        ]
        doctor_requests.sort(key=lambda patient: (getattr(patient, "esi_level", 5), patient.spawned_at))
        if doctor_requests:
            selected = doctor_requests[0]
            simulation.record_esi_deprioritization(selected, doctor_requests, "doctor_request")
            return ("doctor_request", selected)

        return None

    def act(self, attention: Any, simulation: "Simulation") -> None:
        if self.is_task_busy():
            self.continue_task(simulation)
            return

        if self.continue_post_task_station_check(simulation):
            return

        if attention is not None and self.mode in {"moving_to_secondary_station", "using_secondary_station"}:
            self._clear_secondary_station_visit()
            self.mark_idle_target_dirty()

        if self.mode == "moving_to_secondary_station":
            destination = self.destination
            if destination is None:
                self._clear_secondary_station_visit()
                self.mark_idle_target_dirty()
                self.return_home(simulation)
                return

            if self.is_at_point(destination):
                # Station arrival itself is workflow/movement context. Any F2F
                # episode at the station is now generated by the universal
                # perception-to-action loop in Simulation, not by co-presence.
                self.next_position = destination
                self.next_mode = "using_secondary_station"
                self.next_last_idle_position = destination
                self.next_route_plan = []
                self.next_route_destination = None
                return

            self.next_mode = "moving_to_secondary_station"
            self.move_toward(simulation, destination)
            return

        if self.mode == "using_secondary_station":
            remaining = max(self.secondary_station_remaining - config.TIMESTEP_SECONDS, 0)
            self.next_secondary_station_remaining = remaining
            self.next_position = self.position
            self.next_route_plan = []
            self.next_route_destination = None

            if remaining == 0:
                self._clear_secondary_station_visit()
                self.mark_idle_target_dirty()
                self.return_home(simulation)
                return

            self.next_mode = "using_secondary_station"
            return

        if self.mode == "moving_to_patient" and self.target_patient_id is not None:
            patient = simulation.patient_by_id.get(self.target_patient_id)
            if patient is None or patient.discharged:
                self.clear_assignment()
                return

            destination = simulation._patient_assignment_destination(patient)
            if self.is_at_point(destination):
                self.start_task(simulation, patient)
                return

            self.next_mode = "moving_to_patient"
            self.next_destination = destination
            self.move_toward(simulation, destination)
            return

        if self.mode == "moving_to_doctor" and self.target_patient_id is not None:
            patient = simulation.patient_by_id.get(self.target_patient_id)
            doctor = simulation.doctor_by_id.get(self.reserved_doctor_id)
            if patient is None or doctor is None:
                self.clear_assignment()
                return

            if self.distance_to_agent(doctor) <= config.INTERACTION_DISTANCE_METERS:
                handoff_probability = config.NURSE_DOCTOR_HANDOFF_INTERACTION_PROBABILITY
                if getattr(patient, "esi_level", 5) in {1, 2}:
                    handoff_probability = config.HIGH_ACUITY_NURSE_DOCTOR_HANDOFF_INTERACTION_PROBABILITY
                if simulation.random.random() < handoff_probability:
                    simulation.log_interaction(
                        agent_1=self,
                        agent_2=doctor,
                        position=tuple(self.position),
                        interaction_type="nurse_doctor_handoff",
                    )
                self.complete_trip(simulation)
                self.next_mode = "leading_doctor_to_patient"
                destination = simulation._patient_assignment_destination(patient)
                self.next_destination = destination
                self.move_toward(simulation, destination)
                return

            self.next_mode = "moving_to_doctor"
            self.next_destination = doctor.position
            self.move_toward(simulation, doctor.position)
            return

        if self.mode == "leading_doctor_to_patient" and self.target_patient_id is not None:
            patient = simulation.patient_by_id.get(self.target_patient_id)
            if patient is None or patient.bed_position is None:
                self.clear_assignment()
                return

            destination = simulation._patient_assignment_destination(patient)
            if self.is_at_point(destination):
                self.next_reserved_doctor_id = None
                self.return_home(simulation)
                return

            self.next_mode = "leading_doctor_to_patient"
            self.next_destination = destination
            self.move_toward(simulation, destination)
            return

        if self.mode == "waiting_for_doctor" and self.recheck_timer > 0:
            tracked_patient = (
                simulation.patient_by_id.get(self.doctor_recheck_patient_id)
                if self.doctor_recheck_patient_id is not None
                else None
            )
            if (
                tracked_patient is None
                or tracked_patient.discharged
                or tracked_patient.current_task_name != config.MEDICAL_EVALUATION_TASK_NAME
            ):
                self.clear_doctor_wait_state()
                self.mark_idle_target_dirty()
                self.return_home(simulation)
                return

            self.next_recheck_timer = max(self.recheck_timer - config.TIMESTEP_SECONDS, 0)

            if not self._has_valid_idle_destination(self.home_zone_id):
                self._assign_idle_destination(
                    simulation,
                    zone_id=self.home_zone_id,
                    radius_meters=self.idle_position_radius,
                )

            waiting_point = self.next_destination if self.next_destination is not None else self.destination
            if waiting_point is None:
                waiting_point = self.home_position
                self.next_destination = waiting_point

            if self.is_at_point(waiting_point):
                self.next_position = waiting_point
                self.next_mode = "waiting_for_doctor"
                self.next_last_idle_position = waiting_point
                self.next_route_plan = []
                self.next_route_destination = None
            else:
                self.next_mode = "waiting_for_doctor"
                self.move_toward(simulation, waiting_point)
            return

        if attention is None:
            if self.needs_new_idle_target and self._maybe_start_secondary_station_visit(simulation):
                destination = self.next_destination
                if destination is not None and self.is_at_point(destination):
                    self.next_position = destination
                    self.next_mode = "using_secondary_station"
                    self.next_last_idle_position = destination
                    self.next_route_plan = []
                    self.next_route_destination = None
                elif destination is not None:
                    self.move_toward(simulation, destination)
                return
            if (
                self.idle_seconds >= config.STAFF_CIRCULATION_MIN_IDLE_SECONDS
                and simulation.random.random() < config.NURSE_CIRCULATION_PROBABILITY
                and self.start_patrol(simulation)
            ):
                return

            self.return_home(simulation)
            return

        attention_type, patient = attention

        if attention_type == "nurse_task":
            self.clear_doctor_wait_state()
            self._clear_secondary_station_visit()
            destination = simulation._patient_assignment_destination(patient)
            if not simulation.patient_assignment_route_supported(self, patient):
                simulation.record_unsupported_assignment_rejection(
                    self,
                    patient,
                    destination,
                    context="nurse_task_claim",
                )
                self.return_home(simulation)
                return
            patient.claiming_staff_id = self.gid
            self.next_mode = "moving_to_patient"
            self.next_target_patient_id = patient.gid
            self.next_destination = destination
            if patient.current_task_name is not None:
                self.begin_trip(patient.current_task_name, simulation)
            self.move_toward(simulation, destination)
            return

        doctor = simulation.find_free_doctor()
        if doctor is None:
            consecutive_failures = (
                self.doctor_recheck_failures + 1
                if patient.gid == self.doctor_recheck_patient_id
                else 1
            )
            if consecutive_failures >= config.DOCTOR_RECHECK_MAX_ATTEMPTS:
                self.clear_doctor_wait_state()
                self.mark_idle_target_dirty()
                self.return_home(simulation)
                return

            self.next_recheck_timer = config.DOCTOR_RECHECK_INTERVAL_SECONDS
            self.next_doctor_recheck_patient_id = patient.gid
            self.next_doctor_recheck_failures = consecutive_failures
            self.next_reserved_doctor_id = None
            self.next_target_patient_id = None
            if self.mode != "waiting_for_doctor":
                self.mark_idle_target_dirty()
            self.next_mode = "waiting_for_doctor"
            if not self._has_valid_idle_destination(self.home_zone_id):
                self._assign_idle_destination(
                    simulation,
                    zone_id=self.home_zone_id,
                    radius_meters=self.idle_position_radius,
                )
            waiting_point = self.next_destination if self.next_destination is not None else self.home_position
            if self.is_at_point(waiting_point):
                self.next_position = waiting_point
                self.next_last_idle_position = waiting_point
                self.next_route_plan = []
                self.next_route_destination = None
            else:
                self.move_toward(simulation, waiting_point)
            return

        self.clear_doctor_wait_state()
        self._clear_secondary_station_visit()
        doctor.reserve_handoff(patient.gid, self.gid)
        simulation.reserved_doctor_ids_this_step.add(doctor.gid)
        patient.requested_doctor_id = doctor.gid
        patient.requesting_nurse_id = self.gid
        self.next_mode = "moving_to_doctor"
        self.next_target_patient_id = patient.gid
        self.next_reserved_doctor_id = doctor.gid
        self.next_doctor_recheck_patient_id = None
        self.next_destination = doctor.position
        self.begin_trip("Doctor Handoff", simulation)
        self.move_toward(simulation, doctor.position)

    def state_summary(self) -> str:
        if self.is_task_busy():
            return f"{self.current_task_name} ({self.task_remaining}s)"
        if self.mode == "using_secondary_station" and self.secondary_station_zone_id is not None:
            return f"using_{self.secondary_station_zone_id} ({self.secondary_station_remaining}s)"
        return self.mode


class Doctor(StaffAgent):
    """Doctor that can self-dispatch or respond to nurse handoffs."""

    def __init__(self, gid: int, position: Point, home_position: Point, home_zone_id: str) -> None:
        super().__init__(
            gid=gid,
            role="Doctor",
            position=position,
            home_position=home_position,
            home_zone_id=home_zone_id,
            idle_position_radius=config.STAFF_IDLE_POSITION_NOISE_METERS,
        )
        self.handoff_patient_id: Optional[int] = None
        self.next_handoff_patient_id = self.handoff_patient_id
        self.handoff_nurse_id: Optional[int] = None
        self.next_handoff_nurse_id = self.handoff_nurse_id

    def begin_step(self) -> None:
        super().begin_step()
        self.next_handoff_patient_id = self.handoff_patient_id
        self.next_handoff_nurse_id = self.handoff_nurse_id

    def update(self) -> None:
        super().update()
        self.handoff_patient_id = self.next_handoff_patient_id
        self.handoff_nurse_id = self.next_handoff_nurse_id

    def reserve_handoff(self, patient_id: int, nurse_id: int) -> None:
        self.next_handoff_patient_id = patient_id
        self.next_handoff_nurse_id = nurse_id

    def clear_handoff(self) -> None:
        self.handoff_patient_id = None
        self.next_handoff_patient_id = None
        self.handoff_nurse_id = None
        self.next_handoff_nurse_id = None

    def _select_patrol_destination(self, simulation: "Simulation") -> Optional[Point]:
        candidate_points = [
            config.ROUTING_WAYPOINTS[waypoint_key]
            for waypoint_key in config.DOCTOR_PATROL_ROUTE_KEYS
            if waypoint_key in config.ROUTING_WAYPOINTS
        ]
        viable_points = [
            point for point in candidate_points
            if math.dist(point, self.position) > config.ARRIVAL_TOLERANCE_METERS
        ]
        if not viable_points:
            return None
        return simulation.random.choice(viable_points)

    def perceive(self, simulation: "Simulation") -> Any:
        workflow_candidates = [
            patient for patient in simulation.active_patients if not patient.discharged
        ]
        return simulation.perception.build_bundle(self, simulation, workflow_candidates)

    def attend(self, perception: Any, simulation: "Simulation") -> Optional[Patient]:
        if self.target_patient_id is not None:
            return simulation.patient_by_id.get(self.target_patient_id)

        doctor_candidates = [
            patient
            for patient in perception.workflow_candidates
            if patient.task_index in config.ROLE_PERMISSIONS[self.role]
            and patient.claiming_staff_id is None
            and not patient.coordination_hold_active(simulation.timestep)
            and simulation.patient_assignment_route_supported(self, patient)
            and (
                patient.task_index != config.MEDICAL_EVALUATION_TASK_INDEX
                or patient.requested_doctor_id == self.gid
                or patient.requested_doctor_id is None
            )
        ]
        doctor_candidates.sort(
            key=lambda patient: (
                getattr(patient, "esi_level", 5),
                -patient.task_index,
                patient.spawned_at,
            )
        )
        selected = doctor_candidates[0] if doctor_candidates else None
        simulation.record_esi_deprioritization(selected, doctor_candidates, "doctor")
        return selected

    def act(self, attention: Optional[Patient], simulation: "Simulation") -> None:
        if self.is_task_busy():
            self.continue_task(simulation)
            return

        if self.continue_post_task_station_check(simulation):
            return

        if self.continue_patrol(simulation):
            return

        if self.mode == "moving_to_patient" and self.target_patient_id is not None:
            patient = simulation.patient_by_id.get(self.target_patient_id)
            if patient is None or patient.discharged:
                self.clear_assignment()
                self.clear_handoff()
                return

            destination = simulation._patient_assignment_destination(patient)
            if self.is_at_point(destination):
                self.start_task(simulation, patient)
                patient.claiming_staff_id = self.gid
                return

            self.next_mode = "moving_to_patient"
            self.next_destination = destination
            self.move_toward(simulation, destination)
            return

        if self.handoff_patient_id is not None and self.handoff_nurse_id is not None:
            patient = simulation.patient_by_id.get(self.handoff_patient_id)
            nurse = simulation.nurse_by_id.get(self.handoff_nurse_id)
            if patient is None or nurse is None or patient.bed_position is None:
                self.clear_handoff()
                self.return_home(simulation)
                return
            if self.mode == "awaiting_nurse" and nurse.reserved_doctor_id != self.gid:
                self.clear_handoff()
                self.return_home(simulation)
                return

            # Once a handoff has been committed to the doctor's current state,
            # the doctor can only wait for the nurse or move to the patient.
            # A later branch in the same tick must not be allowed to re-open
            # the handoff or revert this travel commitment.
            if self.distance_to_agent(nurse) <= config.INTERACTION_DISTANCE_METERS:
                patient.claiming_staff_id = self.gid
                self.clear_handoff()
                self.next_mode = "moving_to_patient"
                self.next_target_patient_id = patient.gid
                destination = simulation._patient_assignment_destination(patient)
                self.next_destination = destination
                if patient.current_task_name is not None:
                    self.begin_trip(patient.current_task_name, simulation)
                self.move_toward(simulation, destination)
                return

            self.next_mode = "awaiting_nurse"
            self.next_position = self.position
            self.next_destination = self.position
            self.next_route_plan = []
            self.next_route_destination = None
            return

        if attention is None:
            if (
                self.idle_seconds >= config.STAFF_CIRCULATION_MIN_IDLE_SECONDS
                and simulation.random.random() < config.DOCTOR_CIRCULATION_PROBABILITY
                and self.start_patrol(simulation)
            ):
                return
            self.return_home(simulation)
            return

        destination = simulation._patient_assignment_destination(attention)
        if not simulation.patient_assignment_route_supported(self, attention):
            simulation.record_unsupported_assignment_rejection(
                self,
                attention,
                destination,
                context="doctor_task_claim",
            )
            self.return_home(simulation)
            return
        attention.claiming_staff_id = self.gid
        self.next_mode = "moving_to_patient"
        self.next_target_patient_id = attention.gid
        self.next_destination = destination
        if attention.current_task_name is not None:
            self.begin_trip(attention.current_task_name, simulation)
        self.move_toward(simulation, destination)


if TYPE_CHECKING:
    from src.simulation import Simulation
