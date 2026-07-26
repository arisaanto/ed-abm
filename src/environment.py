"""Environment, geometry, and waypoint-routing helpers for the ED ABM."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


Point = Tuple[float, float]


class Environment:
    """Container for walls, zones, centroids, and lightweight routing."""

    def __init__(self, wall_path: Path, zone_path: Path) -> None:
        self.wall_path = wall_path
        self.zone_path = zone_path
        self.walls = self._load_walls(wall_path)
        self.zones = self._load_zones(zone_path)
        self.zone_centroids = {
            zone_id: self._polygon_centroid(polygon)
            for zone_id, polygon in self.zones.items()
        }
        self.plot_bounds = self._compute_plot_bounds()
        self.routing_waypoints = dict(config.ROUTING_WAYPOINTS)
        self.routing_graph = self._build_routing_graph()

    def _load_walls(self, path: Path) -> List[Tuple[Point, Point]]:
        raw_segments = json.loads(path.read_text())
        return [
            ((segment["start"][0], segment["start"][1]), (segment["end"][0], segment["end"][1]))
            for segment in raw_segments
        ]

    def _load_zones(self, path: Path) -> Dict[str, List[Point]]:
        raw_zones = json.loads(path.read_text())
        return {
            zone_id: [(vertex[0], vertex[1]) for vertex in polygon]
            for zone_id, polygon in raw_zones.items()
        }

    def _polygon_centroid(self, polygon: List[Point]) -> Point:
        vertices = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
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
            average_x = sum(point[0] for point in vertices) / len(vertices)
            average_y = sum(point[1] for point in vertices) / len(vertices)
            return (average_x, average_y)

        scale = 1.0 / (6.0 * signed_area)
        return (centroid_x * scale, centroid_y * scale)

    def _compute_plot_bounds(self) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []

        for start, end in self.walls:
            xs.extend([start[0], end[0]])
            ys.extend([start[1], end[1]])

        for polygon in self.zones.values():
            xs.extend(point[0] for point in polygon)
            ys.extend(point[1] for point in polygon)

        padding = config.PLOT_PADDING_METERS
        return (
            min(xs) - padding,
            max(xs) + padding,
            min(ys) - padding,
            max(ys) + padding,
        )

    def _build_routing_graph(self) -> Dict[str, Dict[str, float]]:
        graph = {name: {} for name in self.routing_waypoints}

        for left_name, right_name in config.ROUTING_EDGES:
            left_point = self.routing_waypoints[left_name]
            right_point = self.routing_waypoints[right_name]

            graph[left_name][right_name] = self.distance(left_point, right_point)
            graph[right_name][left_name] = self.distance(right_point, left_point)

        return graph

    def point_in_zone(self, x: float, y: float, zone_id: str) -> bool:
        polygon = self.zones[zone_id]

        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            if self._point_on_segment((x, y), point, next_point):
                return True

        inside = False
        previous_index = len(polygon) - 1

        for index, (x1, y1) in enumerate(polygon):
            x2, y2 = polygon[previous_index]
            intersects = ((y1 > y) != (y2 > y)) and (
                x < ((x2 - x1) * (y - y1) / (y2 - y1 + 1e-12)) + x1
            )
            if intersects:
                inside = not inside
            previous_index = index

        return inside

    def which_zone(self, x: float, y: float) -> Optional[str]:
        for zone_id in self.zones:
            if self.point_in_zone(x, y, zone_id):
                return zone_id
        return None

    def wall_collision(self, position: Point, next_position: Point) -> bool:
        if position == next_position:
            return False

        for wall_start, wall_end in self.walls:
            if self._segments_intersect(position, next_position, wall_start, wall_end):
                return True
        return False

    def has_line_of_sight(self, start: Point, end: Point) -> bool:
        return not self.wall_collision(start, end)

    def distance(self, left: Point, right: Point) -> float:
        return math.dist(left, right)

    def plan_route(self, start: Point, end: Point) -> List[Point]:
        """Return the remaining route from start to end as waypoint positions."""

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

        for waypoint_name, waypoint_point in self.routing_waypoints.items():
            if self.has_line_of_sight(start, waypoint_point):
                distance = self.distance(start, waypoint_point)
                graph[start_name][waypoint_name] = distance
                graph[waypoint_name][start_name] = distance

            if self.has_line_of_sight(end, waypoint_point):
                distance = self.distance(end, waypoint_point)
                graph[end_name][waypoint_name] = distance
                graph[waypoint_name][end_name] = distance

        if self.has_line_of_sight(start, end):
            direct_distance = self.distance(start, end)
            graph[start_name][end_name] = direct_distance
            graph[end_name][start_name] = direct_distance

        path_names = self._shortest_path(graph, start_name, end_name)

        if not path_names:
            return [end]

        return [waypoint_positions[name] for name in path_names[1:]]

    def _shortest_path(
        self,
        graph: Dict[str, Dict[str, float]],
        start_name: str,
        end_name: str,
    ) -> List[str]:
        queue: List[Tuple[float, str]] = [(0.0, start_name)]
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

    def _orientation(self, left: Point, middle: Point, right: Point) -> float:
        return ((middle[0] - left[0]) * (right[1] - left[1])) - (
            (middle[1] - left[1]) * (right[0] - left[0])
        )

    def _point_on_segment(self, point: Point, start: Point, end: Point) -> bool:
        epsilon = 1e-9
        cross = self._orientation(start, end, point)
        if not math.isclose(cross, 0.0, abs_tol=epsilon):
            return False

        return (
            min(start[0], end[0]) - epsilon
            <= point[0]
            <= max(start[0], end[0]) + epsilon
            and min(start[1], end[1]) - epsilon
            <= point[1]
            <= max(start[1], end[1]) + epsilon
        )

    def _segments_intersect(
        self,
        segment_start: Point,
        segment_end: Point,
        wall_start: Point,
        wall_end: Point,
    ) -> bool:
        epsilon = 1e-9
        orientation_1 = self._orientation(segment_start, segment_end, wall_start)
        orientation_2 = self._orientation(segment_start, segment_end, wall_end)
        orientation_3 = self._orientation(wall_start, wall_end, segment_start)
        orientation_4 = self._orientation(wall_start, wall_end, segment_end)

        if math.isclose(orientation_1, 0.0, abs_tol=epsilon) and self._point_on_segment(
            wall_start,
            segment_start,
            segment_end,
        ):
            return True
        if math.isclose(orientation_2, 0.0, abs_tol=epsilon) and self._point_on_segment(
            wall_end,
            segment_start,
            segment_end,
        ):
            return True
        if math.isclose(orientation_3, 0.0, abs_tol=epsilon) and self._point_on_segment(
            segment_start,
            wall_start,
            wall_end,
        ):
            return True
        if math.isclose(orientation_4, 0.0, abs_tol=epsilon) and self._point_on_segment(
            segment_end,
            wall_start,
            wall_end,
        ):
            return True

        return ((orientation_1 > 0.0) != (orientation_2 > 0.0)) and (
            (orientation_3 > 0.0) != (orientation_4 > 0.0)
        )
