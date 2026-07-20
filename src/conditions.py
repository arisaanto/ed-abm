"""Condition management for baseline and design-intervention geometry views.

This module keeps intervention logic out of agent and environment code. The
baseline environment remains the single source of physical collision geometry;
conditions only derive alternate visibility segments and effective station
zones/attractors for perception and idle sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import config


Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _polygon_centroid(polygon: Sequence[Point]) -> Point:
    vertices = list(polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon)
    signed_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        cross = (x1 * y2) - (x2 * y1)
        signed_area += cross
        centroid_x += (x1 + x2) * cross
        centroid_y += (y1 + y2) * cross

    signed_area *= 0.5
    if math.isclose(signed_area, 0.0):
        return (
            sum(point[0] for point in vertices) / max(len(vertices), 1),
            sum(point[1] for point in vertices) / max(len(vertices), 1),
        )

    scale = 1.0 / (6.0 * signed_area)
    return (centroid_x * scale, centroid_y * scale)


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


def _segment_matches(left: Segment, right: Segment, tolerance: float = 1e-6) -> bool:
    return (
        math.dist(left[0], right[0]) <= tolerance
        and math.dist(left[1], right[1]) <= tolerance
    ) or (
        math.dist(left[0], right[1]) <= tolerance
        and math.dist(left[1], right[0]) <= tolerance
    )


@dataclass(frozen=True)
class ConditionSpec:
    """Declarative description of one experimental condition."""

    name: str
    visibility_transparent_segments: Tuple[Segment, ...]
    removed_wall_segments: Tuple[Segment, ...]
    added_wall_segments: Tuple[Segment, ...]
    zone_polygon_overrides: Mapping[str, Tuple[Point, ...]]
    station_attractor_overrides: Mapping[str, Point]
    routing_waypoint_overrides: Mapping[str, Point]
    notes: str


class ConditionManager:
    """Derived geometry/accessors for one baseline or intervention condition."""

    def __init__(self, environment, spec: ConditionSpec, active_affordances: Optional[Iterable[Mapping[str, object]]] = None) -> None:
        self.environment = environment
        self.spec = spec
        self.affordance_registry: Dict[str, Mapping[str, object]] = {
            str(affordance["id"]): affordance
            for affordance in (active_affordances or [])
        }
        self.affordance_zone_ids: Tuple[str, ...] = tuple(self.affordance_registry)
        self.zone_polygons: Dict[str, Tuple[Point, ...]] = {
            zone_id: tuple(polygon)
            for zone_id, polygon in environment.zones.items()
        }
        self.zone_polygons.update(spec.zone_polygon_overrides)
        self.zone_polygons.update(
            {
                affordance_id: tuple((float(point[0]), float(point[1])) for point in affordance["polygon"])
                for affordance_id, affordance in self.affordance_registry.items()
            }
        )
        self.zone_centroids: Dict[str, Point] = {
            zone_id: _polygon_centroid(polygon)
            for zone_id, polygon in self.zone_polygons.items()
        }
        self.station_attractors: Dict[str, Point] = {
            zone_id: self.zone_centroids[zone_id]
            for zone_id in config.STATION_ZONE_IDS
            if zone_id in self.zone_centroids
        }
        self.station_attractors.update(
            {
                str(zone_id): (float(point[0]), float(point[1]))
                for zone_id, point in getattr(config, "DEFAULT_STATION_ATTRACTOR_OVERRIDES", {}).items()
            }
        )
        self.station_attractors.update(spec.station_attractor_overrides)
        self.transparent_wall_indices = self._resolve_transparent_wall_indices(
            environment.walls,
            spec.visibility_transparent_segments,
        )
        self.removed_wall_indices = self._resolve_transparent_wall_indices(
            environment.walls,
            spec.removed_wall_segments,
        )
        self.condition_removed_wall_indices = frozenset(
            set(self.removed_wall_indices) | set(self._nursta_baseline_box_wall_indices())
        )
        self.effective_walls: Tuple[Segment, ...] = tuple(
            wall
            for index, wall in enumerate(environment.walls)
            if index not in self.condition_removed_wall_indices
        ) + tuple(spec.added_wall_segments)
        self.visibility_walls: Tuple[Segment, ...] = tuple(
            wall
            for index, wall in enumerate(environment.walls)
            if index not in self.transparent_wall_indices
            and index not in self.condition_removed_wall_indices
        ) + tuple(spec.added_wall_segments)
        self.routing_waypoints = self.adjusted_routing_waypoints()
        self.routing_graph = self._build_routing_graph()

    def _nursta_baseline_box_wall_indices(self) -> frozenset[int]:
        """Old NURSTA desk/connector wall indices replaced after NURSTA relocation."""

        if "NURSTA" not in self.spec.zone_polygon_overrides:
            return frozenset()

        baseline_polygon = self.environment.zones.get("NURSTA")
        if not baseline_polygon:
            return frozenset()

        vertices = baseline_polygon[:-1] if baseline_polygon[0] == baseline_polygon[-1] else baseline_polygon
        x_values = [point[0] for point in vertices]
        y_values = [point[1] for point in vertices]
        x_min = min(x_values) - 0.10
        x_max = max(x_values) + 0.10
        station_top_y = max(y_values)
        upper_y = station_top_y + 1.25
        lower_y = station_top_y - 0.10

        indices = set()
        for index, (wall_start, wall_end) in enumerate(self.environment.walls):
            wall_xs = (wall_start[0], wall_end[0])
            wall_ys = (wall_start[1], wall_end[1])
            if (
                min(wall_xs) >= x_min
                and max(wall_xs) <= x_max
                and min(wall_ys) >= lower_y
                and max(wall_ys) <= upper_y
            ):
                indices.add(index)
        return frozenset(indices)

    def render_suppressed_wall_indices(self) -> frozenset[int]:
        """Wall indices suppressed in figures/animations for this condition."""

        return self.condition_removed_wall_indices

    def render_added_wall_segments(self) -> Tuple[Segment, ...]:
        """Condition-specific solid wall segments added for relocated station walls."""

        return tuple(self.spec.added_wall_segments)

    def _resolve_transparent_wall_indices(
        self,
        walls: Sequence[Segment],
        target_segments: Iterable[Segment],
    ) -> frozenset[int]:
        indices = set()
        for target_segment in target_segments:
            for index, wall_segment in enumerate(walls):
                if _segment_matches(wall_segment, target_segment):
                    indices.add(index)
                    break
        return frozenset(indices)

    def zone_polygon(self, zone_id: str) -> Tuple[Point, ...]:
        return self.zone_polygons[zone_id]

    def zone_centroid(self, zone_id: str) -> Point:
        return self.zone_centroids[zone_id]

    def station_attractor(self, zone_id: str) -> Point:
        return self.station_attractors.get(zone_id, self.zone_centroid(zone_id))

    def adjusted_routing_waypoints(self) -> Dict[str, Point]:
        waypoints = {name: tuple(point) for name, point in config.ROUTING_WAYPOINTS.items()}
        waypoints.update(self.spec.routing_waypoint_overrides)
        for affordance_id, affordance in self.affordance_registry.items():
            if bool(affordance.get("supports_routing", False)):
                waypoints[f"affordance_{affordance_id}"] = self.zone_centroid(affordance_id)
        return waypoints

    def point_in_zone(self, x: float, y: float, zone_id: str) -> bool:
        return _point_in_polygon((x, y), self.zone_polygons[zone_id])

    def which_zone(self, x: float, y: float) -> Optional[str]:
        for zone_id in self.affordance_zone_ids:
            if self.point_in_zone(x, y, zone_id):
                return zone_id
        for zone_id in self.zone_polygons:
            if self.point_in_zone(x, y, zone_id):
                return zone_id
        return None

    def draw_zone_overlays(self, axis) -> None:
        """Draw only overridden zones so interventions remain visible on plots."""

        for zone_id, polygon in self.spec.zone_polygon_overrides.items():
            if zone_id == "NURSTA":
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            axis.plot(xs, ys, color="steelblue", linewidth=1.5, linestyle="--", zorder=2)
            centroid_x, centroid_y = self.zone_centroid(zone_id)
            axis.text(
                centroid_x,
                centroid_y,
                f"{zone_id}*",
                color="steelblue",
                fontsize=config.ZONE_LABEL_FONTSIZE,
                ha="center",
                va="center",
                zorder=3,
            )

    def draw_floorplan(self, axis) -> None:
        """Draw the floorplan with condition-specific render-only styling."""

        for zone_id in self.environment.zones:
            polygon = self.zone_polygons.get(zone_id, tuple(self.environment.zones[zone_id]))
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            axis.plot(
                xs,
                ys,
                color=config.ZONE_COLOR,
                linewidth=config.ZONE_LINEWIDTH,
                zorder=1,
            )

            centroid_x, centroid_y = self.zone_centroids.get(zone_id, self.environment.zone_centroids[zone_id])
            axis.text(
                centroid_x,
                centroid_y,
                zone_id,
                color=config.ZONE_LABEL_COLOR,
                fontsize=config.ZONE_LABEL_FONTSIZE,
                ha="center",
                va="center",
                zorder=2,
            )

        suppressed_indices = self.render_suppressed_wall_indices()
        transparent_condition = self.spec.name in {"cockpit_only", "both"}
        for index, (wall_start, wall_end) in enumerate(self.environment.walls):
            if index in suppressed_indices:
                continue
            is_transparent = transparent_condition and index in self.transparent_wall_indices
            axis.plot(
                [wall_start[0], wall_end[0]],
                [wall_start[1], wall_end[1]],
                color="#8FA7B3" if is_transparent else config.WALL_COLOR,
                linewidth=0.9 if is_transparent else config.WALL_LINEWIDTH,
                alpha=0.50 if is_transparent else 1.0,
                zorder=3,
            )

        for wall_start, wall_end in self.render_added_wall_segments():
            axis.plot(
                [wall_start[0], wall_end[0]],
                [wall_start[1], wall_end[1]],
                color=config.WALL_COLOR,
                linewidth=config.WALL_LINEWIDTH,
                zorder=4,
            )

    def wall_collision(self, position: Point, next_position: Point) -> bool:
        if position == next_position:
            return False
        return any(
            self._segments_intersect(position, next_position, wall_start, wall_end)
            for wall_start, wall_end in self.effective_walls
        )

    def has_line_of_sight(self, start: Point, end: Point) -> bool:
        return not self.wall_collision(start, end)

    def point_has_wall_clearance(self, point: Point, clearance_meters: float) -> bool:
        for segment_start, segment_end in self.effective_walls:
            if self._point_segment_distance(point, segment_start, segment_end) < clearance_meters:
                return False
        return True

    def plan_route(self, start: Point, end: Point) -> list[Point]:
        route = self._graph_route(start, end)
        if route and self.route_is_walkable(start, route):
            return route
        recovery_route = self._recovery_route(start, end)
        if recovery_route and self.route_is_walkable(start, recovery_route):
            return recovery_route
        direct_route = [end]
        if self.route_is_walkable(start, direct_route):
            return direct_route
        return []

    def route_is_walkable(self, start: Point, route: Sequence[Point]) -> bool:
        """Return true only if each route leg is collision-free under movement steps."""

        previous = start
        for waypoint in route:
            if self.wall_collision(previous, waypoint):
                return False
            if not self._stepped_leg_is_walkable(previous, waypoint):
                return False
            previous = waypoint
        return bool(route)

    def _stepped_leg_is_walkable(self, start: Point, end: Point) -> bool:
        distance = math.dist(start, end)
        if distance <= 1e-9:
            return True
        step_length = float(getattr(config, "MOVEMENT_SPEED_METERS_PER_SECOND", 1.0))
        if step_length <= 0:
            step_length = 1.0
        steps = max(1, int(math.ceil(distance / step_length)))
        previous = start
        for step_index in range(1, steps + 1):
            fraction = min((step_index * step_length) / distance, 1.0)
            candidate = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            if self.wall_collision(previous, candidate):
                return False
            previous = candidate
        return True

    def _graph_route(self, start: Point, end: Point) -> list[Point]:
        if self.has_line_of_sight(start, end):
            return [end]

        graph = {name: neighbors.copy() for name, neighbors in self.routing_graph.items()}
        waypoint_positions = dict(self.routing_waypoints)
        start_name = "__start__"
        end_name = "__end__"

        graph[start_name] = {}
        graph[end_name] = {}
        waypoint_positions[start_name] = start
        waypoint_positions[end_name] = end

        start_connectors = self._connector_waypoints_for_point(start)
        end_connectors = self._connector_waypoints_for_point(end)

        for waypoint_name, waypoint_point in self.routing_waypoints.items():
            if start_connectors is not None and waypoint_name not in start_connectors:
                continue
            if self.has_line_of_sight(start, waypoint_point):
                distance = math.dist(start, waypoint_point)
                graph[start_name][waypoint_name] = distance
                graph[waypoint_name][start_name] = distance

        for waypoint_name, waypoint_point in self.routing_waypoints.items():
            if end_connectors is not None and waypoint_name not in end_connectors:
                continue
            if self.has_line_of_sight(end, waypoint_point):
                distance = math.dist(end, waypoint_point)
                graph[end_name][waypoint_name] = distance
                graph[waypoint_name][end_name] = distance

        if self.has_line_of_sight(start, end):
            direct_distance = math.dist(start, end)
            graph[start_name][end_name] = direct_distance
            graph[end_name][start_name] = direct_distance

        path_names = self._shortest_path(graph, start_name, end_name)
        if not path_names:
            return []
        return [waypoint_positions[name] for name in path_names[1:]]

    def _recovery_route(self, start: Point, end: Point) -> list[Point]:
        """Find a deterministic escape step for valid points orphaned near walls."""

        best_route: Optional[list[Point]] = None
        best_score = math.inf
        angles = (
            0.0,
            -math.pi / 4.0,
            math.pi / 4.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -3.0 * math.pi / 4.0,
            3.0 * math.pi / 4.0,
            math.pi,
            -math.pi / 8.0,
            math.pi / 8.0,
            -3.0 * math.pi / 8.0,
            3.0 * math.pi / 8.0,
            -5.0 * math.pi / 8.0,
            5.0 * math.pi / 8.0,
            -7.0 * math.pi / 8.0,
            7.0 * math.pi / 8.0,
        )
        for radius in (0.5, 0.75, 1.0, 1.5, 2.0):
            for angle in angles:
                candidate = (
                    start[0] + math.cos(angle) * radius,
                    start[1] + math.sin(angle) * radius,
                )
                if self.wall_collision(start, candidate):
                    continue
                if not self.point_has_wall_clearance(candidate, config.WALL_CLEARANCE_METERS):
                    continue
                onward_route = self._graph_route(candidate, end)
                if not onward_route:
                    continue
                route = [candidate, *onward_route]
                if not self.route_is_walkable(start, route):
                    continue
                previous = start
                score = 0.0
                for waypoint in route:
                    score += math.dist(previous, waypoint)
                    previous = waypoint
                if score < best_score:
                    best_route = route
                    best_score = score
        return best_route or []

    def _build_routing_graph(self) -> Dict[str, Dict[str, float]]:
        graph = {name: {} for name in self.routing_waypoints}
        for left_name, right_name in config.ROUTING_EDGES:
            if left_name not in self.routing_waypoints or right_name not in self.routing_waypoints:
                continue
            left_point = self.routing_waypoints[left_name]
            right_point = self.routing_waypoints[right_name]
            graph[left_name][right_name] = math.dist(left_point, right_point)
            graph[right_name][left_name] = math.dist(right_point, left_point)
        return graph

    def _shortest_path(
        self,
        graph: Dict[str, Dict[str, float]],
        start_name: str,
        end_name: str,
    ) -> list[str]:
        queue: list[tuple[float, str]] = [(0.0, start_name)]
        distances = {start_name: 0.0}
        previous: Dict[str, str] = {}
        visited = set()

        while queue:
            current_distance, current_name = heapq.heappop(queue)
            if current_name in visited:
                continue
            visited.add(current_name)
            if current_name == end_name:
                break
            for neighbor_name, edge_distance in graph[current_name].items():
                tentative_distance = current_distance + edge_distance
                if tentative_distance < distances.get(neighbor_name, math.inf):
                    distances[neighbor_name] = tentative_distance
                    previous[neighbor_name] = current_name
                    heapq.heappush(queue, (tentative_distance, neighbor_name))

        if end_name not in distances:
            return []

        path = [end_name]
        current_name = end_name
        while current_name != start_name:
            current_name = previous[current_name]
            path.append(current_name)
        return list(reversed(path))

    def _connector_waypoints_for_point(self, point: Point) -> Optional[set[str]]:
        if self.spec.name not in {"nursta_only", "both"}:
            return None
        if self.which_zone(point[0], point[1]) != "NURSTA":
            return None
        return {"nursta_west", "nursta_east"}

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

    def _point_segment_distance(self, point: Point, segment_start: Point, segment_end: Point) -> float:
        line_dx = segment_end[0] - segment_start[0]
        line_dy = segment_end[1] - segment_start[1]
        if abs(line_dx) <= 1e-12 and abs(line_dy) <= 1e-12:
            return math.dist(point, segment_start)

        projection = (
            ((point[0] - segment_start[0]) * line_dx) + ((point[1] - segment_start[1]) * line_dy)
        ) / ((line_dx * line_dx) + (line_dy * line_dy))
        projection = max(0.0, min(1.0, projection))
        projected_point = (
            segment_start[0] + (projection * line_dx),
            segment_start[1] + (projection * line_dy),
        )
        return math.dist(point, projected_point)


def _normalize_segment_specs(raw_specs: Sequence[Mapping[str, Sequence[float]]]) -> Tuple[Segment, ...]:
    normalized = []
    for segment in raw_specs:
        normalized.append(
            (
                (float(segment["start"][0]), float(segment["start"][1])),
                (float(segment["end"][0]), float(segment["end"][1])),
            )
        )
    return tuple(normalized)


def _normalize_zone_overrides(
    raw_overrides: Mapping[str, Sequence[Sequence[float]]]
) -> Dict[str, Tuple[Point, ...]]:
    normalized: Dict[str, Tuple[Point, ...]] = {}
    for zone_id, polygon in raw_overrides.items():
        normalized[zone_id] = tuple((float(point[0]), float(point[1])) for point in polygon)
    return normalized


def _normalize_routing_overrides(
    raw_overrides: Mapping[str, Sequence[float]]
) -> Dict[str, Point]:
    return {
        waypoint_name: (float(point[0]), float(point[1]))
        for waypoint_name, point in raw_overrides.items()
    }


def _normalize_attractor_overrides(
    raw_overrides: Mapping[str, Sequence[float]]
) -> Dict[str, Point]:
    return {
        zone_id: (float(point[0]), float(point[1]))
        for zone_id, point in raw_overrides.items()
    }


def get_condition_spec(condition_name: Optional[str] = None) -> ConditionSpec:
    """Return a normalized condition specification from config."""

    resolved_name = config.CONDITION_NAME if condition_name is None else condition_name
    raw_spec = config.CONDITIONS[resolved_name]
    return ConditionSpec(
        name=resolved_name,
        visibility_transparent_segments=_normalize_segment_specs(
            raw_spec.get("visibility_transparent_segments", [])
        ),
        removed_wall_segments=_normalize_segment_specs(
            raw_spec.get("removed_wall_segments", [])
        ),
        added_wall_segments=_normalize_segment_specs(
            raw_spec.get("added_wall_segments", [])
        ),
        zone_polygon_overrides=_normalize_zone_overrides(
            raw_spec.get("zone_polygon_overrides", {})
        ),
        station_attractor_overrides=_normalize_attractor_overrides(
            raw_spec.get("station_attractor_overrides", {})
        ),
        routing_waypoint_overrides=_normalize_routing_overrides(
            raw_spec.get("routing_waypoint_overrides", {})
        ),
        notes=str(raw_spec.get("notes", "")),
    )
