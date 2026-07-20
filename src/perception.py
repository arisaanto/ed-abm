"""Isovist-based perception, salience scoring, and attention selection.

This layer keeps movement rule-based while grounding interaction opportunity in
what an agent can actually see from its current position and heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import networkx as nx

import config


Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _normalize_angle(angle_radians: float) -> float:
    while angle_radians <= -math.pi:
        angle_radians += 2.0 * math.pi
    while angle_radians > math.pi:
        angle_radians -= 2.0 * math.pi
    return angle_radians


def _angle_between(origin: Point, target: Point) -> float:
    return math.atan2(target[1] - origin[1], target[0] - origin[0])


def _heading_delta(heading: float, target_angle: float) -> float:
    return abs(_normalize_angle(target_angle - heading))


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    previous_index = len(polygon) - 1

    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[previous_index]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < ((x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12)) + x1
        )
        if intersects:
            inside = not inside
        previous_index = index

    return inside


def _ray_segment_intersection(origin: Point, angle: float, segment: Segment) -> Optional[Point]:
    ray_dx = math.cos(angle)
    ray_dy = math.sin(angle)
    (x1, y1), (x2, y2) = segment
    seg_dx = x2 - x1
    seg_dy = y2 - y1

    denominator = (ray_dx * seg_dy) - (ray_dy * seg_dx)
    if abs(denominator) <= 1e-12:
        return None

    diff_x = x1 - origin[0]
    diff_y = y1 - origin[1]
    ray_t = ((diff_x * seg_dy) - (diff_y * seg_dx)) / denominator
    seg_u = ((diff_x * ray_dy) - (diff_y * ray_dx)) / denominator

    if ray_t < 0.0 or not 0.0 <= seg_u <= 1.0:
        return None

    return (origin[0] + (ray_t * ray_dx), origin[1] + (ray_t * ray_dy))


def _cast_ray(
    origin: Point,
    angle: float,
    segments: Sequence[Segment],
    max_distance: float,
) -> Point:
    nearest_point = (
        origin[0] + (math.cos(angle) * max_distance),
        origin[1] + (math.sin(angle) * max_distance),
    )
    nearest_distance = max_distance

    for segment in segments:
        intersection = _ray_segment_intersection(origin, angle, segment)
        if intersection is None:
            continue
        distance = math.dist(origin, intersection)
        if distance < nearest_distance:
            nearest_point = intersection
            nearest_distance = distance

    return nearest_point


@dataclass
class Percept:
    """Structured percept consumed by role-specific attend logic."""

    entity_id: int | str
    entity_type: str
    role: Optional[str]
    zone_id: Optional[str]
    distance_m: float
    mutual_visibility: bool
    line_of_sight_length_m: float
    salience: float
    position: Point


@dataclass
class PerceptionBundle:
    """Full perceptual state for one agent in one timestep."""

    visible_polygon: List[Point]
    visible_agents: List[Percept]
    visible_landmarks: List[Percept]
    attended_percepts: List[Percept]
    workflow_candidates: list = field(default_factory=list)


class PerceptionService:
    """Shared perception service consumed by staff agents and simulation logic."""

    def __init__(self, condition_manager) -> None:
        self.condition_manager = condition_manager
        self._isovist_cache: dict[tuple, List[Point]] = {}

    def compute_isovist(
        self,
        position: Point,
        heading: Optional[float] = None,
        fov_degrees: Optional[float] = None,
    ) -> List[Point]:
        """Return the visibility polygon from one point."""

        cache_key = (
            self.condition_manager.spec.name,
            round(position[0], 2),
            round(position[1], 2),
            None if heading is None else round(heading, 3),
            None if fov_degrees is None else round(fov_degrees, 1),
        )
        cached_polygon = self._isovist_cache.get(cache_key)
        if cached_polygon is not None:
            return list(cached_polygon)

        angles = []
        epsilon = config.ISOVIST_RAY_EPSILON_RADIANS
        visibility_segments = self.condition_manager.visibility_walls

        if heading is not None and fov_degrees is not None and fov_degrees < 360.0:
            half_fov = math.radians(fov_degrees / 2.0)
            sample_step = math.radians(max(config.ISOVIST_SAMPLE_ANGLE_DEGREES, 1.0))
            boundary_angles = (heading - half_fov, heading + half_fov)
            for boundary_angle in boundary_angles:
                angles.extend([boundary_angle - epsilon, boundary_angle, boundary_angle + epsilon])

            sample_count = max(int((2.0 * half_fov) / sample_step), 2)
            for sample_index in range(sample_count + 1):
                fraction = sample_index / max(sample_count, 1)
                angles.append((heading - half_fov) + ((2.0 * half_fov) * fraction))
        else:
            for sample_index in range(
                max(int(360.0 / max(config.ISOVIST_SAMPLE_ANGLE_DEGREES, 1.0)), 12)
            ):
                angles.append(math.radians(sample_index * config.ISOVIST_SAMPLE_ANGLE_DEGREES))

        for segment_start, segment_end in visibility_segments:
            for endpoint in (segment_start, segment_end):
                endpoint_angle = _angle_between(position, endpoint)
                if heading is not None and fov_degrees is not None and fov_degrees < 360.0:
                    if _heading_delta(heading, endpoint_angle) > math.radians(fov_degrees / 2.0) + epsilon:
                        continue
                angles.extend([endpoint_angle - epsilon, endpoint_angle, endpoint_angle + epsilon])

        unique_angles = sorted({_normalize_angle(angle) for angle in angles})
        polygon_with_angles = []
        for angle in unique_angles:
            point = _cast_ray(
                origin=position,
                angle=angle,
                segments=visibility_segments,
                max_distance=config.VISIBILITY_MAX_DISTANCE,
            )
            polygon_with_angles.append((angle, point))

        polygon_with_angles.sort(key=lambda item: item[0])
        polygon = [point for _, point in polygon_with_angles]
        if len(self._isovist_cache) > 4_096:
            self._isovist_cache.clear()
        self._isovist_cache[cache_key] = polygon
        return list(polygon)

    def _within_fov(self, agent, target_position: Point) -> bool:
        if config.FOV_DEGREES >= 360.0:
            return True
        target_angle = _angle_between(agent.position, target_position)
        return _heading_delta(agent.heading, target_angle) <= math.radians(config.FOV_DEGREES / 2.0)

    def _has_visibility(self, start: Point, end: Point) -> bool:
        if math.dist(start, end) > config.VISIBILITY_MAX_DISTANCE:
            return False
        start_in_cockpit = self.condition_manager.point_in_zone(start[0], start[1], "COCPIT")
        end_in_cockpit = self.condition_manager.point_in_zone(end[0], end[1], "COCPIT")
        cockpit_transparent = self.condition_manager.spec.name in {"cockpit_only", "both"}
        if start_in_cockpit != end_in_cockpit and not cockpit_transparent:
            cockpit_polygon = self.condition_manager.zone_polygon("COCPIT")
            cockpit_edges = [
                (cockpit_polygon[index], cockpit_polygon[(index + 1) % len(cockpit_polygon)])
                for index in range(len(cockpit_polygon))
            ]
            if any(
                self._segments_intersect(start, end, edge_start, edge_end)
                for edge_start, edge_end in cockpit_edges
            ):
                return False
        if start_in_cockpit and end_in_cockpit and not cockpit_transparent:
            cockpit_centroid_y = self.condition_manager.zone_centroid("COCPIT")[1]
            if (
                (start[1] - cockpit_centroid_y) * (end[1] - cockpit_centroid_y) < 0.0
                and abs(start[1] - end[1]) >= 1.5
            ):
                return False
        return not any(
            self._segments_intersect(start, end, wall_start, wall_end)
            for wall_start, wall_end in self.condition_manager.visibility_walls
        )

    def _segments_intersect(
        self,
        left_start: Point,
        left_end: Point,
        right_start: Point,
        right_end: Point,
    ) -> bool:
        def orientation(point_a: Point, point_b: Point, point_c: Point) -> int:
            value = (
                (point_b[1] - point_a[1]) * (point_c[0] - point_b[0])
                - (point_b[0] - point_a[0]) * (point_c[1] - point_b[1])
            )
            if abs(value) <= 1e-9:
                return 0
            return 1 if value > 0 else 2

        def on_segment(point_a: Point, point_b: Point, point_c: Point) -> bool:
            return (
                min(point_a[0], point_c[0]) - 1e-9 <= point_b[0] <= max(point_a[0], point_c[0]) + 1e-9
                and min(point_a[1], point_c[1]) - 1e-9 <= point_b[1] <= max(point_a[1], point_c[1]) + 1e-9
            )

        if left_start == right_start or left_start == right_end or left_end == right_start or left_end == right_end:
            return False

        o1 = orientation(left_start, left_end, right_start)
        o2 = orientation(left_start, left_end, right_end)
        o3 = orientation(right_start, right_end, left_start)
        o4 = orientation(right_start, right_end, left_end)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(left_start, right_start, left_end):
            return True
        if o2 == 0 and on_segment(left_start, right_end, left_end):
            return True
        if o3 == 0 and on_segment(right_start, left_start, right_end):
            return True
        if o4 == 0 and on_segment(right_start, left_end, right_end):
            return True
        return False

    def score_salience(self, agent, candidate, simulation) -> float:
        distance = max(agent.distance_to_agent(candidate), 0.01)
        distance_weight = math.exp(-distance / max(config.SALIENCE_DISTANCE_DECAY, 0.01))
        workflow_weight = 1.0

        if getattr(candidate, "role", None) == "Patient":
            esi_level = getattr(candidate, "esi_level", None)
            if esi_level in config.ESI_PRIORITY_WEIGHT:
                workflow_weight *= float(config.ESI_PRIORITY_WEIGHT[esi_level])
            if getattr(candidate, "claiming_staff_id", None) == agent.gid:
                workflow_weight += 0.6
            if getattr(candidate, "task_index", -1) in config.ROLE_PERMISSIONS.get(agent.role, []):
                workflow_weight += 0.3
            if (
                agent.role == "Doctor"
                and getattr(candidate, "current_task_name", None) == config.MEDICAL_EVALUATION_TASK_NAME
            ):
                workflow_weight *= config.DOCTOR_RESPONSE_BIAS
        elif agent.role == "Nurse" and getattr(candidate, "role", None) == "Doctor":
            workflow_weight += 0.2
        elif agent.role == "Doctor" and getattr(candidate, "role", None) == "Nurse":
            workflow_weight += 0.1

        return distance_weight * workflow_weight

    def collect_visible_entities(self, agent, simulation, perception_polygon: Sequence[Point]) -> tuple[List[Percept], List[Percept]]:
        visible_agents: List[Percept] = []
        visible_landmarks: List[Percept] = []

        for candidate in simulation.all_agents:
            if candidate.gid == agent.gid:
                continue
            if not self._within_fov(agent, candidate.position):
                continue
            if not self._has_visibility(agent.position, candidate.position):
                continue
            if perception_polygon and not _point_in_polygon(candidate.position, perception_polygon):
                continue

            salience = self.score_salience(agent, candidate, simulation)
            if salience < config.SALIENCE_MIN_THRESHOLD:
                continue

            mutual_visibility = (
                hasattr(candidate, "heading")
                and self._within_fov(candidate, agent.position)
                and self._has_visibility(candidate.position, agent.position)
            )
            visible_agents.append(
                Percept(
                    entity_id=candidate.gid,
                    entity_type="agent",
                    role=candidate.role,
                    zone_id=simulation.which_zone(*candidate.position),
                    distance_m=agent.distance_to_agent(candidate),
                    mutual_visibility=mutual_visibility,
                    line_of_sight_length_m=agent.distance_to_agent(candidate),
                    salience=salience,
                    position=candidate.position,
                )
            )

        for zone_id in config.STATION_ZONE_IDS:
            landmark_position = self.condition_manager.station_attractor(zone_id)
            if not self._within_fov(agent, landmark_position):
                continue
            if not self._has_visibility(agent.position, landmark_position):
                continue
            visible_landmarks.append(
                Percept(
                    entity_id=zone_id,
                    entity_type="station",
                    role=None,
                    zone_id=zone_id,
                    distance_m=math.dist(agent.position, landmark_position),
                    mutual_visibility=False,
                    line_of_sight_length_m=math.dist(agent.position, landmark_position),
                    salience=math.exp(
                        -math.dist(agent.position, landmark_position)
                        / max(config.SALIENCE_DISTANCE_DECAY, 0.01)
                    ),
                    position=landmark_position,
                )
            )

        return visible_agents, visible_landmarks

    def select_attention(self, percepts: Iterable[Percept], capacity: int) -> List[Percept]:
        ranked_percepts = sorted(percepts, key=lambda percept: percept.salience, reverse=True)
        return ranked_percepts[: max(capacity, 0)]

    def build_bundle(self, agent, simulation, workflow_candidates: Optional[list] = None) -> PerceptionBundle:
        polygon = self.compute_isovist(
            position=agent.position,
            heading=agent.heading,
            fov_degrees=config.FOV_DEGREES,
        )
        visible_agents, visible_landmarks = self.collect_visible_entities(agent, simulation, polygon)
        attended = self.select_attention(visible_agents, config.ATTENTION_CAPACITY)
        return PerceptionBundle(
            visible_polygon=polygon,
            visible_agents=visible_agents,
            visible_landmarks=visible_landmarks,
            attended_percepts=attended,
            workflow_candidates=[] if workflow_candidates is None else workflow_candidates,
        )

    def can_agents_mutually_perceive(self, left_agent, right_agent, simulation) -> bool:
        if left_agent.distance_to_agent(right_agent) > config.VISIBILITY_MAX_DISTANCE:
            return False
        if not self._within_fov(left_agent, right_agent.position):
            return False
        if not self._within_fov(right_agent, left_agent.position):
            return False
        return self._has_visibility(left_agent.position, right_agent.position)

    def build_visibility_graph(self, environment, resolution: float) -> nx.Graph:
        graph = nx.Graph()
        min_x, max_x, min_y, max_y = environment.plot_bounds
        x = min_x
        sample_points: List[Point] = []
        while x <= max_x:
            y = min_y
            while y <= max_y:
                if self.condition_manager.which_zone(x, y) is not None:
                    sample_points.append((round(x, 3), round(y, 3)))
                y += resolution
            x += resolution

        for point in sample_points:
            graph.add_node(point)

        for left_index, left_point in enumerate(sample_points):
            for right_point in sample_points[left_index + 1:]:
                if self._has_visibility(left_point, right_point):
                    graph.add_edge(left_point, right_point, weight=math.dist(left_point, right_point))

        return graph

    def compute_visual_integration(self, visibility_graph: nx.Graph) -> dict:
        if not visibility_graph.nodes:
            return {}
        return nx.closeness_centrality(visibility_graph, distance="weight")
