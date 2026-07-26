#!/usr/bin/env python3
"""Build the final Part 2 figures and tables from completed outputs only.

The script reads the canonical n=100 intervention results and the completed
four-factor sensitivity table. It never imports or runs the simulation engine.
Geometry classes are used only to draw the four already-defined conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from statistics import NormalDist
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/abm-part2-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/abm-part2-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.conditions import ConditionManager, get_condition_spec
from src.environment import Environment
from src.perception import PerceptionService


SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
INTERVENTIONS = ("cockpit_only", "nursta_only", "both")
ROLE_COUNTS = {
    "CoordinationNurse": 1,
    "Doctor": 3,
    "Nurse": 5,
}
ROLE_LABELS = {
    "CoordinationNurse": "Coordination nurse",
    "Doctor": "Doctor",
    "Nurse": "Nurse",
}

SCENARIO_LABELS = {
    "normal_load": "Normal load",
    "high_load_high_acuity": "High load",
}
CONDITION_LABELS = {
    "baseline": "Baseline",
    "cockpit_only": "COCPIT only",
    "nursta_only": "NURSTA only",
    "both": "Both",
}
CONTRAST_LABELS = {
    "cockpit_only": "COCPIT only − Baseline",
    "nursta_only": "NURSTA only − Baseline",
    "both": "Both − Baseline",
}
COLORS = {
    "observed": "#6FA8A2",
    "baseline": "#AAA9A4",
    "cockpit_only": "#6FA8A2",
    "nursta_only": "#6B7B8E",
    "both": "#74698C",
    "ink": "#24252A",
    "muted": "#737B87",
    "line": "#D9D7D2",
    "paper": "#FFFFFF",
    "zone": "#C9CED5",
    "standing": "#E9E5EF",
    "transparent": "#A7AAAE",
    "patient": "#6FA8A2",
}

SUMMARY_METRICS = {
    "F2F interactions per hour": ("validation_metrics", "f2f_per_hour"),
    "Interaction count": ("validation_metrics", "interaction_count"),
    "Staff–staff interaction share": ("validation_metrics", "hcw_hcw_share"),
    "Patient-facing interaction share": ("validation_metrics", "patient_facing_share"),
    "Corridor interaction share": ("validation_metrics", "corridor_share"),
    "Bedside / patient-room interaction share": (
        "validation_metrics",
        "bedside_or_patient_room_share",
    ),
    "Visible staff encounters": (
        "perception_reason_funnel_metrics",
        "raw_visible_staff_percepts_count",
    ),
    "Actionable visibility opportunities": (
        "perception_reason_funnel_metrics",
        "eligible_visible_staff_opportunity_count",
    ),
    "Selected coordination opportunities": (
        "perception_reason_funnel_metrics",
        "selected_visible_staff_opportunity_count",
    ),
    "Approach attempts": ("perception_reason_funnel_metrics", "approach_intent_count"),
    "Successful approaches": (
        "perception_reason_funnel_metrics",
        "approach_success_count",
    ),
}

MOVEMENT_METRICS = (
    "Total staff movement distance (km)",
    "Zone transitions",
    "Station occupancy (staff-hours)",
)

FRICTION_METRICS = (
    "Staff–staff contacts per run",
    "Metres travelled per staff–staff contact",
    "Staff–staff contacts per movement kilometre",
)

SENSITIVITY_METRIC_LABELS = {
    "f2f_per_hour": "F2F interactions per hour",
    "raw_visible_staff_percepts": "Visible staff encounters",
    "eligible_visible_opportunities": "Actionable visibility opportunities",
    "selected_visible_opportunities": "Selected coordination opportunities",
    "approach_intents": "Approach attempts",
    "approach_successes": "Successful approaches",
    "hcw_hcw_share": "Staff–staff interaction share",
    "patient_facing_share": "Patient-facing interaction share",
    "corridor_share": "Corridor interaction share",
    "bedside_share": "Bedside / patient-room interaction share",
    "ed_pressure_index_mean": "Mean ED pressure index",
}

FACTOR_ORDER = (
    "Visibility response",
    "Pressure activation threshold",
    "Repeat-contact delay",
    "Station-return tendency",
)

FACTOR_DISPLAY = {
    "Visibility response": "Visibility response",
    "Pressure activation threshold": "Crowding threshold",
    "Repeat-contact delay": "Repeat-contact delay",
    "Station-return tendency": "Station-return tendency",
}


def _configure_style() -> str:
    preferred = ("Helvetica", "Arial", "DejaVu Sans")
    family = "DejaVu Sans"
    for candidate in preferred:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family, "Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": COLORS["paper"],
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
        }
    )
    return family


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def _despine(axis: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(left)
    axis.spines["bottom"].set_visible(bottom)
    if not left:
        axis.tick_params(axis="y", left=False)
    if not bottom:
        axis.tick_params(axis="x", bottom=False)


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    mean = float(np.mean(array))
    sem = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    half = float(_student_t_critical_975(len(array) - 1) * sem)
    return mean - half, mean + half


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value without SciPy."""

    if degrees_of_freedom <= 0:
        return math.nan
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
    )


def _read_nested(mapping: dict[str, Any], path: tuple[str, str]) -> float:
    try:
        return float(mapping[path[0]][path[1]])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _load_runs(canonical_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(canonical_dir.rglob("summary.json")):
        summary = json.loads(path.read_text())
        metadata = summary.get("metadata", {})
        row: dict[str, Any] = {
            "scenario": str(metadata.get("scenario_mode", "")),
            "condition": str(metadata.get("condition", "")),
            "seed": int(metadata.get("seed", -1)),
            "workflow_status": str(
                summary.get("hard_gate_diagnostics", {}).get(
                    "workflow_health_status", "UNKNOWN"
                )
            ),
            "summary_path": str(path),
        }
        for label, metric_path in SUMMARY_METRICS.items():
            row[label] = _read_nested(summary, metric_path)
        movement = summary.get("movement_dwell_metrics", {})
        row["Total staff movement distance (km)"] = (
            sum(float(value) for value in movement.get("movement_distance_by_role", {}).values())
            / 1000.0
        )
        for role, count in ROLE_COUNTS.items():
            role_distance = float(
                movement.get("movement_distance_by_role", {}).get(role, 0.0)
            )
            row[f"{role} movement (km/person-run)"] = (
                role_distance / 1000.0 / count
            )
        row["Zone transitions"] = sum(
            float(value) for value in movement.get("zone_transition_counts", {}).values()
        )
        row["Station occupancy (staff-hours)"] = (
            sum(float(value) for value in movement.get("station_occupancy_seconds", {}).values())
            / 3600.0
        )
        interaction_count = float(row["Interaction count"])
        staff_contact_share = float(row["Staff–staff interaction share"])
        staff_contacts = interaction_count * staff_contact_share
        movement_km = float(row["Total staff movement distance (km)"])
        row["Staff–staff contacts per run"] = staff_contacts
        row["Metres travelled per staff–staff contact"] = (
            1000.0 * movement_km / staff_contacts if staff_contacts > 0 else math.nan
        )
        row["Staff–staff contacts per movement kilometre"] = (
            staff_contacts / movement_km if movement_km > 0 else math.nan
        )
        visible_pair_samples = sum(
            float(value)
            for value in summary.get("experienced_visibility_metrics", {})
            .get("mutual_visibility_counts_by_zone", {})
            .values()
        )
        staff_count = int(metadata.get("staff_count", 0))
        duration_seconds = float(metadata.get("duration_seconds", 0.0))
        possible_pair_seconds = math.comb(staff_count, 2) * duration_seconds if staff_count > 1 else 0.0
        row["Mutual staff visibility exposure (%)"] = (
            100.0 * visible_pair_samples * float(config.TIMESTEP_SECONDS) / possible_pair_seconds
            if possible_pair_seconds > 0
            else math.nan
        )
        hard = summary.get("hard_gate_diagnostics", {})
        for key, value in hard.items():
            if isinstance(value, (int, float, bool)):
                row[f"hard::{key}"] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No summary.json files found below {canonical_dir}")
    return frame


def _event_file_metadata(path: Path) -> tuple[str, str, int]:
    parts = path.parts
    seed_index = next(i for i, value in enumerate(parts) if value.startswith("seed_"))
    return parts[seed_index - 2], parts[seed_index - 1], int(parts[seed_index].split("_")[-1])


def _load_events(canonical_dir: Path) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    paths = sorted(canonical_dir.rglob("interaction_events.csv"))
    for path in paths:
        frame = pd.read_csv(path)
        scenario, condition, seed = _event_file_metadata(path)
        if "counted_for_validation" in frame:
            counted = frame["counted_for_validation"]
            if counted.dtype != bool:
                counted = counted.astype(str).str.lower().isin({"true", "1", "yes"})
            frame = frame[counted]
        frame = frame[pd.to_numeric(frame["x"], errors="coerce").notna()]
        frame = frame[pd.to_numeric(frame["y"], errors="coerce").notna()]
        frames.append(
            frame.assign(
                scenario=scenario,
                condition=condition,
                seed=seed,
            )
        )
    return pd.concat(frames, ignore_index=True), len(paths)


def _spatial_ecology_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Measure interaction compactness and role–zone segregation per run.

    Mean separation is the average Euclidean distance between all interaction
    pairs in a run. Role–zone segregation is normalized mutual information
    between role pair and broad zone group: zero means their spatial
    distributions are independent; one means perfect association. To prevent
    larger intervention event counts from biasing that estimate, every
    scenario/seed quartet is repeatedly downsampled to its smallest condition
    count before the metric is averaged.
    """

    def normalized_mutual_information(frame: pd.DataFrame) -> float:
        contingency = pd.crosstab(
            frame["role_pair"].fillna("Other"),
            frame["zone_group"].fillna("other"),
        ).to_numpy(dtype=float)
        joint = contingency / contingency.sum()
        role_probability = joint.sum(axis=1, keepdims=True)
        zone_probability = joint.sum(axis=0, keepdims=True)
        expected = role_probability @ zone_probability
        nonzero = joint > 0
        mutual_information = float(
            np.sum(joint[nonzero] * np.log(joint[nonzero] / expected[nonzero]))
        )
        role_entropy = float(
            -np.sum(role_probability[role_probability > 0] * np.log(role_probability[role_probability > 0]))
        )
        zone_entropy = float(
            -np.sum(zone_probability[zone_probability > 0] * np.log(zone_probability[zone_probability > 0]))
        )
        denominator = min(role_entropy, zone_entropy)
        return mutual_information / denominator if denominator > 0 else math.nan

    reference_manager = _condition_manager("baseline")
    xmin, xmax, ymin, ymax = reference_manager.environment.plot_bounds
    cell_size_m = 0.75
    x_edges = np.arange(xmin, xmax + cell_size_m, cell_size_m)
    y_edges = np.arange(ymin, ymax + cell_size_m, cell_size_m)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0

    def largest_connected_mask(manager: ConditionManager) -> np.ndarray:
        candidate = np.asarray(
            [
                [
                    manager.which_zone(float(x), float(y)) is not None
                    and manager.point_has_wall_clearance((float(x), float(y)), 0.18)
                    for y in y_centres
                ]
                for x in x_centres
            ],
            dtype=bool,
        )
        visited = np.zeros_like(candidate, dtype=bool)
        components: list[list[tuple[int, int]]] = []
        for x_index, y_index in zip(*np.where(candidate)):
            if visited[x_index, y_index]:
                continue
            component: list[tuple[int, int]] = []
            stack = [(int(x_index), int(y_index))]
            visited[x_index, y_index] = True
            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))
                current_point = (float(x_centres[current_x]), float(y_centres[current_y]))
                for x_step, y_step in (
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1),
                ):
                    next_x, next_y = current_x + x_step, current_y + y_step
                    if not (0 <= next_x < candidate.shape[0] and 0 <= next_y < candidate.shape[1]):
                        continue
                    if visited[next_x, next_y] or not candidate[next_x, next_y]:
                        continue
                    next_point = (float(x_centres[next_x]), float(y_centres[next_y]))
                    if manager.wall_collision(current_point, next_point):
                        continue
                    visited[next_x, next_y] = True
                    stack.append((next_x, next_y))
            components.append(component)
        largest = max(components, key=len)
        result = np.zeros_like(candidate, dtype=bool)
        for x_index, y_index in largest:
            result[x_index, y_index] = True
        return result

    walkable_masks: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        manager = _condition_manager(condition)
        walkable_masks[condition] = largest_connected_mask(manager)

    event_counts = events.groupby(["scenario", "seed", "condition"]).size().unstack("condition")
    balanced_counts = event_counts.loc[:, list(CONDITIONS)].min(axis=1).astype(int)
    rows: list[dict[str, Any]] = []
    grouped = events.groupby(["scenario", "condition", "seed"], sort=False)
    for (scenario, condition, seed), frame in grouped:
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        if len(coordinates) > 1:
            differences = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            distances = np.sqrt(np.sum(differences * differences, axis=2))
            upper = np.triu_indices(len(coordinates), k=1)
            mean_separation = float(np.mean(distances[upper]))
        else:
            mean_separation = math.nan

        target_count = int(balanced_counts.loc[(scenario, int(seed))])
        condition_index = CONDITIONS.index(str(condition))
        random_generator = np.random.default_rng(
            830_003 + int(seed) * 101 + SCENARIOS.index(str(scenario)) * 10_007 + condition_index
        )
        segregation_samples = []
        for _ in range(40):
            if len(frame) == target_count:
                sample = frame
            else:
                selected = random_generator.choice(len(frame), size=target_count, replace=False)
                sample = frame.iloc[selected]
            segregation_samples.append(normalized_mutual_information(sample))
        segregation = float(np.mean(segregation_samples))

        density = _density_grid(frame, x_edges, y_edges, sigma=1.5)
        walkable_mask = walkable_masks[str(condition)]
        density = np.where(walkable_mask, density, 0.0)
        density /= density.sum()
        ordered_density = np.sort(density[walkable_mask])[::-1]
        cumulative_density = np.cumsum(ordered_density)
        footprint_cells = int(np.searchsorted(cumulative_density, 0.90) + 1)
        footprint_share = 100.0 * footprint_cells / int(np.sum(walkable_mask))
        staff_contact_flag = frame["is_hcw_hcw"]
        if staff_contact_flag.dtype != bool:
            staff_contact_flag = staff_contact_flag.astype(str).str.lower().isin({"true", "1", "yes"})
        staff_contacts = frame[staff_contact_flag]
        coordination_nurse_contacts = staff_contacts[
            staff_contacts["role_pair"].fillna("").str.contains("CoordinationNurse", regex=False)
        ]
        coordination_nurse_share = (
            100.0 * len(coordination_nurse_contacts) / len(staff_contacts)
            if len(staff_contacts)
            else math.nan
        )
        rows.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": int(seed),
                "Mean separation between interactions (m)": mean_separation,
                "Role–zone segregation (0–1)": segregation,
                "90% interaction footprint (% of walkable ED)": footprint_share,
                "Coordination-nurse participation in staff contacts (%)": coordination_nurse_share,
            }
        )
    return pd.DataFrame(rows)


def _paired_effects(runs: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_frame = runs[runs["scenario"] == scenario]
        for intervention in INTERVENTIONS:
            for metric in metrics:
                pivot = scenario_frame.pivot_table(
                    index="seed", columns="condition", values=metric, aggfunc="first"
                )
                complete = pivot[["baseline", intervention]].dropna()
                delta = (complete[intervention] - complete["baseline"]).to_numpy(dtype=float)
                low, high = _ci95(delta)
                rows.append(
                    {
                        "scenario": scenario,
                        "intervention": intervention,
                        "metric": metric,
                        "paired_n": len(delta),
                        "baseline_mean": float(complete["baseline"].mean()),
                        "condition_mean": float(complete[intervention].mean()),
                        "mean_delta": float(np.mean(delta)),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def _condition_manager(condition: str) -> ConditionManager:
    environment = Environment(config.WALL_POSITIONS_PATH, config.ZONE_BOUNDARIES_PATH)
    return ConditionManager(environment, get_condition_spec(condition), [])


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _isovist_area(manager: ConditionManager, origin: tuple[float, float]) -> float:
    endpoints = PerceptionService(manager).compute_isovist(
        origin, heading=0.0, fov_degrees=360.0
    )
    return _polygon_area(endpoints)


def _walkable_grid(
    manager: ConditionManager,
    resolution_m: float = 1.0,
) -> list[tuple[float, float]]:
    """Return the largest physically connected grid component in the ED."""

    xmin, xmax, ymin, ymax = manager.environment.plot_bounds
    candidates: list[tuple[float, float]] = []
    for x in np.arange(xmin, xmax + 1e-9, resolution_m):
        for y in np.arange(ymin, ymax + 1e-9, resolution_m):
            point = (float(x), float(y))
            if manager.which_zone(*point) is None:
                continue
            if manager.point_has_wall_clearance(point, 0.18):
                candidates.append(point)

    walk_graph = nx.Graph()
    walk_graph.add_nodes_from(range(len(candidates)))
    neighbor_limit = resolution_m * math.sqrt(2.0) + 1e-6
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if math.dist(left, right) <= neighbor_limit and not manager.wall_collision(left, right):
                walk_graph.add_edge(left_index, right_index)
    if not walk_graph.number_of_nodes():
        return []
    largest = max(nx.connected_components(walk_graph), key=len)
    return [candidates[index] for index in sorted(largest)]


def _whole_space_vga_metrics(resolution_m: float = 1.0) -> pd.DataFrame:
    """Compute comparable whole-space VGA-style metrics for all conditions."""

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        manager = _condition_manager(condition)
        perception = PerceptionService(manager)
        points = _walkable_grid(manager, resolution_m=resolution_m)
        visibility_graph = nx.Graph()
        visibility_graph.add_nodes_from(range(len(points)))
        through_lengths = np.zeros(len(points), dtype=float)
        for left_index, left in enumerate(points):
            for right_index in range(left_index + 1, len(points)):
                right = points[right_index]
                if perception._has_visibility(left, right):
                    distance = math.dist(left, right)
                    visibility_graph.add_edge(left_index, right_index)
                    through_lengths[left_index] += distance
                    through_lengths[right_index] += distance

        connectivity = [degree for _, degree in visibility_graph.degree()]
        integration = nx.closeness_centrality(visibility_graph)
        integration_values = np.asarray(list(integration.values()), dtype=float)
        connectivity_values = np.asarray(connectivity, dtype=float)
        intelligibility = float(np.corrcoef(connectivity_values, integration_values)[0, 1])
        isovist_areas = [_isovist_area(manager, point) for point in points]
        rows.append(
            {
                "condition": condition,
                "sample_points": len(points),
                "Connectivity": float(np.mean(connectivity)),
                "Isovist area": float(np.mean(isovist_areas)),
                "Through vision": float(np.mean(through_lengths)),
                "Visual integration": float(np.mean(integration_values)),
                "Visual intelligibility": intelligibility,
                "grid_resolution_m": resolution_m,
            }
        )
    return pd.DataFrame(rows)


def _draw_floorplan_walls(
    axis: plt.Axes,
    manager: ConditionManager,
    *,
    condition_design: bool = False,
    quiet: bool = False,
    linewidth: float = 0.55,
) -> None:
    suppressed = manager.render_suppressed_wall_indices()
    for index, (start, end) in enumerate(manager.environment.walls):
        if index in suppressed:
            continue
        transparent = index in manager.transparent_wall_indices
        base_color = "#85898E" if quiet else COLORS["ink"]
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=(COLORS["transparent"] if transparent and condition_design else base_color),
            linewidth=(0.85 if transparent and condition_design else linewidth),
            alpha=(0.34 if transparent and condition_design else (0.52 if quiet else 0.78)),
            solid_capstyle="round",
            zorder=5,
        )
    for start, end in manager.render_added_wall_segments():
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#55585D" if quiet else COLORS["ink"],
            linewidth=max(1.2, linewidth),
            solid_capstyle="round",
            zorder=6,
        )


def _set_map_axis(
    axis: plt.Axes,
    manager: ConditionManager,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    xmin, xmax, ymin, ymax = bounds or manager.environment.plot_bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_intervention_atlas(figures_dir: Path) -> None:
    fig = plt.figure(figsize=(8.15, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=(1.62, 1.0, 1.0, 1.0), wspace=0.045)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    central_bounds = (-4.45, 4.55, -9.15, 1.75)

    baseline_axis = axes[0]
    baseline_manager = _condition_manager("baseline")
    _draw_floorplan_walls(baseline_axis, baseline_manager, quiet=True, linewidth=0.64)
    baseline_axis.scatter(
        [point[0] for point in config.BED_POSITIONS],
        [point[1] for point in config.BED_POSITIONS],
        s=10,
        color=COLORS["patient"],
        edgecolor="white",
        linewidth=0.35,
        zorder=8,
    )
    for label in ("COCPIT", "NURSTA"):
        x, y = baseline_manager.station_attractor(label)
        baseline_axis.text(x, y, label, ha="center", va="center", fontsize=6.5, zorder=9)
    baseline_axis.add_patch(
        Rectangle(
            (central_bounds[0], central_bounds[2]),
            central_bounds[1] - central_bounds[0],
            central_bounds[3] - central_bounds[2],
            facecolor="#E9E5EF",
            fill=True,
            alpha=0.36,
            edgecolor=COLORS["both"],
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            zorder=0,
        )
    )
    baseline_axis.set_title("Baseline", loc="left", pad=5)
    _set_map_axis(baseline_axis, baseline_manager)

    for axis, condition in zip(axes[1:], INTERVENTIONS):
        manager = _condition_manager(condition)
        axis.set_facecolor("#FAFAF8")
        axis.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=axis.transAxes,
                fill=False,
                edgecolor="#ECECE8",
                linewidth=0.6,
                zorder=0,
            )
        )
        polygon = manager.zone_polygon("NURSTA")
        axis.fill(
            [point[0] for point in polygon],
            [point[1] for point in polygon],
            color=COLORS["standing"],
            alpha=0.68,
            zorder=1,
        )
        _draw_floorplan_walls(axis, manager, condition_design=True, linewidth=0.82)
        axis.scatter(
            [point[0] for point in config.BED_POSITIONS],
            [point[1] for point in config.BED_POSITIONS],
            s=13,
            color=COLORS["patient"],
            edgecolor="white",
            linewidth=0.4,
            zorder=8,
        )
        for label in ("COCPIT", "NURSTA"):
            x, y = manager.station_attractor(label)
            axis.text(x, y, label, ha="center", va="center", fontsize=6.4, zorder=9)
        axis.set_title(CONDITION_LABELS[condition], loc="center", pad=5)
        _set_map_axis(axis, manager, central_bounds)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["patient"], markeredgecolor="white", markersize=5, label="Patient rooms"),
        Line2D([0], [0], color=COLORS["transparent"], alpha=0.55, lw=1.1, label="Transparent partition"),
        Patch(facecolor=COLORS["standing"], edgecolor="none", label="NURSTA staff area"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.65,
        bbox_to_anchor=(0.5, -0.025),
    )
    _save_figure(fig, figures_dir / "figA_intervention_layout")


def _density_grid(
    points: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma: float,
) -> np.ndarray:
    counts, _, _ = np.histogram2d(points["x"], points["y"], bins=(x_edges, y_edges))
    radius = max(1, int(math.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    density = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 0, counts)
    density = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 1, density)
    total = density.sum()
    return density / total if total > 0 else density


def _plot_spatial_redistribution(events: pd.DataFrame, figures_dir: Path) -> None:
    manager = _condition_manager("baseline")
    xmin, xmax, ymin, ymax = manager.environment.plot_bounds
    x_edges = np.linspace(xmin, xmax, 105)
    y_edges = np.linspace(ymin, ymax, 78)
    contrasts: dict[tuple[str, str], tuple[np.ndarray, int, int]] = {}
    absolute_values: list[float] = []
    for scenario in SCENARIOS:
        baseline = events[(events["scenario"] == scenario) & (events["condition"] == "baseline")]
        baseline_density = _density_grid(baseline, x_edges, y_edges, sigma=1.55)
        for intervention in INTERVENTIONS:
            condition_events = events[
                (events["scenario"] == scenario) & (events["condition"] == intervention)
            ]
            difference = (
                _density_grid(condition_events, x_edges, y_edges, sigma=1.55)
                - baseline_density
            ) * 100.0
            contrasts[(scenario, intervention)] = (
                difference,
                len(baseline),
                len(condition_events),
            )
            absolute_values.extend(np.abs(difference).ravel().tolist())

    nonzero = np.asarray([value for value in absolute_values if value > 1e-9], dtype=float)
    absolute_max = float(np.quantile(nonzero, 0.99)) if len(nonzero) else 1.0

    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-absolute_max, vcenter=0.0, vmax=absolute_max)
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for row, scenario in enumerate(SCENARIOS):
        for column, intervention in enumerate(INTERVENTIONS):
            axis = axes[row, column]
            density, _, _ = contrasts[(scenario, intervention)]
            image = axis.imshow(
                density.T,
                origin="lower",
                extent=(xmin, xmax, ymin, ymax),
                cmap=cmap,
                norm=norm,
                interpolation="bilinear",
                zorder=1,
            )
            panel_manager = _condition_manager(intervention)
            _draw_floorplan_walls(axis, panel_manager, linewidth=0.42)
            _set_map_axis(axis, panel_manager)
            if row == 0:
                axis.set_title(CONDITION_LABELS[intervention], pad=2)
            if column == 0:
                axis.text(
                    -0.04,
                    0.5,
                    SCENARIO_LABELS[scenario],
                    rotation=90,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold",
                )
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.04,
        aspect=38,
        shrink=0.68,
        extend="both",
    )
    colorbar.set_label("Change from baseline in F2F interaction-density share (percentage points per cell)")
    colorbar.outline.set_linewidth(0.5)
    _save_figure(fig, figures_dir / "figB_interaction_redistribution")


def _density_mass_level(density: np.ndarray, mass: float) -> float:
    """Return the density threshold enclosing the requested probability mass."""

    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    return float(ordered[min(int(np.searchsorted(cumulative, mass)), len(ordered) - 1)])


def _plot_interaction_footprint_candidate(events: pd.DataFrame, figures_dir: Path) -> None:
    """Overlay the 50% cores and 90% footprints of Baseline and Both."""

    manager = _condition_manager("baseline")
    xmin, xmax, ymin, ymax = manager.environment.plot_bounds
    x_edges = np.linspace(xmin, xmax, 105)
    y_edges = np.linspace(ymin, ymax, 78)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True, constrained_layout=True)
    styles = {
        "baseline": ("#AAA9A4", "Baseline"),
        "both": (COLORS["both"], "Both"),
    }
    for axis, scenario in zip(axes, SCENARIOS):
        for condition in ("baseline", "both"):
            points = events[(events["scenario"] == scenario) & (events["condition"] == condition)]
            density = _density_grid(points, x_edges, y_edges, sigma=1.55)
            color, _ = styles[condition]
            level90 = _density_mass_level(density, 0.90)
            level50 = _density_mass_level(density, 0.50)
            maximum = float(np.max(density))
            axis.contourf(
                x_centres,
                y_centres,
                density.T,
                levels=[level90, maximum],
                colors=[color],
                alpha=0.075 if condition == "baseline" else 0.095,
                zorder=1,
            )
            axis.contour(
                x_centres,
                y_centres,
                density.T,
                levels=[level90],
                colors=[color],
                linewidths=0.9,
                linestyles="--",
                alpha=0.88,
                zorder=3,
            )
            axis.contour(
                x_centres,
                y_centres,
                density.T,
                levels=[level50],
                colors=[color],
                linewidths=1.35,
                linestyles="-",
                alpha=0.95,
                zorder=4,
            )
        _draw_floorplan_walls(axis, manager, quiet=True, linewidth=0.45)
        _set_map_axis(axis, manager)
        axis.set_title(SCENARIO_LABELS[scenario], pad=4)

    handles = [
        Line2D([0], [0], color=styles[condition][0], lw=1.3, label=styles[condition][1])
        for condition in ("baseline", "both")
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#808387", lw=1.25, label="50% core"),
            Line2D([0], [0], color="#808387", lw=0.9, ls="--", label="90% footprint"),
        ]
    )
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    _save_figure(fig, figures_dir / "figC_candidate_interaction_footprints")


def _spatial_fabric_arrays(
    points: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    role_pair_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_index = np.digitize(points["x"].to_numpy(dtype=float), x_edges) - 1
    y_index = np.digitize(points["y"].to_numpy(dtype=float), y_edges) - 1
    valid = (
        (x_index >= 0)
        & (x_index < len(x_edges) - 1)
        & (y_index >= 0)
        & (y_index < len(y_edges) - 1)
    )
    frame = pd.DataFrame(
        {
            "x_index": x_index[valid],
            "y_index": y_index[valid],
            "role_pair": points.loc[valid, "role_pair"].fillna("Other").astype(str).to_numpy(),
        }
    )
    counts = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
    diversity = np.zeros_like(counts)
    for (x_bin, y_bin), group in frame.groupby(["x_index", "y_index"]):
        counts[int(x_bin), int(y_bin)] = len(group)
        probabilities = group["role_pair"].value_counts(normalize=True).to_numpy(dtype=float)
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        diversity[int(x_bin), int(y_bin)] = entropy / math.log(role_pair_count)
    return counts, diversity


def _plot_spatial_fabric_candidate(events: pd.DataFrame, figures_dir: Path) -> None:
    """Map interaction intensity and role-pair diversity as a spatial fabric."""

    manager = _condition_manager("baseline")
    xmin, xmax, ymin, ymax = manager.environment.plot_bounds
    cell_size = 0.85
    x_edges = np.arange(xmin, xmax + cell_size, cell_size)
    y_edges = np.arange(ymin, ymax + cell_size, cell_size)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    role_pair_count = max(2, int(events["role_pair"].nunique()))
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    nonzero_counts: list[float] = []
    for scenario in SCENARIOS:
        for condition in ("baseline", "both"):
            points = events[(events["scenario"] == scenario) & (events["condition"] == condition)]
            counts, diversity = _spatial_fabric_arrays(points, x_edges, y_edges, role_pair_count)
            arrays[(scenario, condition)] = (counts, diversity)
            nonzero_counts.extend(counts[counts > 0].tolist())
    intensity_reference = float(np.quantile(nonzero_counts, 0.97))
    color_map = LinearSegmentedColormap.from_list(
        "spatial_fabric",
        ("#E7E6E1", "#8FBBAF", "#74698C"),
    )
    normalization = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.25), sharex=True, sharey=True, constrained_layout=True)
    for row, scenario in enumerate(SCENARIOS):
        for column, condition in enumerate(("baseline", "both")):
            axis = axes[row, column]
            counts, diversity = arrays[(scenario, condition)]
            for x_bin, y_bin in zip(*np.where(counts > 0)):
                relative_intensity = min(1.0, counts[x_bin, y_bin] / intensity_reference)
                tile_size = cell_size * (0.18 + 0.76 * math.sqrt(relative_intensity))
                axis.add_patch(
                    Rectangle(
                        (x_centres[x_bin] - tile_size / 2.0, y_centres[y_bin] - tile_size / 2.0),
                        tile_size,
                        tile_size,
                        facecolor=color_map(normalization(diversity[x_bin, y_bin])),
                        edgecolor="none",
                        alpha=0.9,
                        zorder=2,
                    )
                )
            _draw_floorplan_walls(axis, manager, quiet=True, linewidth=0.42)
            _set_map_axis(axis, manager)
            if row == 0:
                axis.set_title(CONDITION_LABELS[condition], pad=3)
            if column == 0:
                axis.text(
                    -0.035,
                    0.5,
                    SCENARIO_LABELS[scenario],
                    rotation=90,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                )
    scalar = matplotlib.cm.ScalarMappable(norm=normalization, cmap=color_map)
    colorbar = fig.colorbar(scalar, ax=axes, orientation="horizontal", fraction=0.04, pad=0.035, aspect=45)
    colorbar.set_label("Role-pair diversity within each spatial cell")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["One role pair", "Mixed", "Highly mixed"])
    colorbar.outline.set_linewidth(0.45)
    fig.text(
        0.5,
        -0.015,
        "Tile area shows interaction intensity",
        ha="center",
        va="top",
        fontsize=7.0,
        color=COLORS["muted"],
    )
    _save_figure(fig, figures_dir / "figC_candidate_spatial_fabric")


def _plot_spatial_coordination_ecology(runs: pd.DataFrame, figures_dir: Path) -> None:
    """Show the baseline-to-combined state transition in spatial ecology."""

    before_color = "#D9D5E3"
    after_color = COLORS["both"]
    connector_color = "#8D9196"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.85),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, scenario in zip(axes, SCENARIOS):
        scenario_runs = runs[runs["scenario"] == scenario]
        condition_means: dict[str, tuple[float, float]] = {}
        for condition in ("baseline", "both"):
            subset = scenario_runs[scenario_runs["condition"] == condition]
            x_values = subset["Mean separation between interactions (m)"].to_numpy(dtype=float)
            y_values = subset["Role–zone segregation (0–1)"].to_numpy(dtype=float)
            x_mean, y_mean = float(np.mean(x_values)), float(np.mean(y_values))
            condition_means[condition] = (x_mean, y_mean)
            axis.scatter(
                x_mean,
                y_mean,
                s=54,
                facecolor=(before_color if condition == "baseline" else after_color),
                edgecolor="#38393D",
                linewidth=0.72,
                zorder=4,
            )
            axis.annotate(
                CONDITION_LABELS[condition],
                (x_mean, y_mean),
                xytext=(7, 6) if condition == "baseline" else (7, -13),
                textcoords="offset points",
                fontsize=7.1,
                color=COLORS["ink"],
                zorder=5,
            )

        baseline = condition_means["baseline"]
        both = condition_means["both"]
        axis.plot(
            [baseline[0], baseline[0], both[0]],
            [baseline[1], both[1], both[1]],
            color=connector_color,
            linewidth=0.9,
            linestyle=(0, (1.8, 2.0)),
            solid_capstyle="round",
            zorder=1,
        )
        mean_distance_change = both[0] - baseline[0]
        segregation_change = 100.0 * (both[1] / baseline[1] - 1.0)
        arrow_x = both[0]
        axis.add_patch(
            FancyArrowPatch(
                (arrow_x, baseline[1] - 0.012),
                (arrow_x, both[1] + 0.012),
                arrowstyle="->",
                mutation_scale=9,
                linewidth=0.95,
                color=COLORS["ink"],
                zorder=3,
            )
        )
        axis.text(
            both[0] + 0.10,
            (baseline[1] + both[1]) / 2.0,
            f"{abs(segregation_change):.0f}% less\nsegregated",
            ha="right",
            va="center",
            fontsize=6.5,
            linespacing=1.35,
            color="#666A70",
            zorder=4,
        )
        axis.text(
            (baseline[0] + both[0]) / 2.0,
            both[1] - 0.014,
            f"{abs(mean_distance_change):.1f}m closer",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#666A70",
            zorder=4,
        )
        axis.set_title(
            SCENARIO_LABELS[scenario],
            loc="left",
            pad=12,
            fontsize=8.0,
            fontweight="bold",
        )
        axis.set_xlabel("Mean interaction separation (m)")
        _despine(axis)
    axes[0].set_ylabel("Role–location segregation\n(lower = more mixed)")
    axes[0].set_xlim(10.45, 7.65)
    axes[0].set_ylim(0.44, 0.78)
    _save_figure(fig, figures_dir / "figC_spatial_coordination_ecology")


def _format_estimate(
    mean: float,
    low: float,
    high: float,
    *,
    scale: float = 1.0,
    digits: int = 2,
) -> str:
    threshold = 0.5 * 10 ** (-digits)

    def signed(value: float) -> str:
        scaled = value * scale
        if abs(scaled) < threshold:
            scaled = 0.0
        return f"{scaled:+.{digits}f}"

    return f"{signed(mean)} [{signed(low)}, {signed(high)}]"


def _vga_spatial_table(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index("condition")
    rows = []
    specifications = (
        ("Connectivity (visible points)", "Connectivity", 1),
        ("Isovist area (m²)", "Isovist area", 1),
        ("Visual integration (0–1)", "Visual integration", 3),
        ("Visual intelligibility (r)", "Visual intelligibility", 3),
    )
    for label, source, digits in specifications:
        rows.append(
            {
                "Metric": label,
                **{
                    CONDITION_LABELS[condition]: f"{float(indexed.loc[condition, source]):.{digits}f}"
                    for condition in CONDITIONS
                },
            }
        )
    return pd.DataFrame(rows)


def _paired_effect_table(effects: pd.DataFrame) -> pd.DataFrame:
    metric_specs = (
        ("F2F interactions per hour", 1.0, 2, "Δ F2F/h [95% CI]"),
        ("Actionable visibility opportunities", 1.0, 1, "Δ opportunities [95% CI]"),
        ("Staff–staff interaction share", 100.0, 1, "Δ staff–staff (pp) [95% CI]"),
        ("Patient-facing interaction share", 100.0, 1, "Δ patient-facing (pp) [95% CI]"),
        ("Corridor interaction share", 100.0, 1, "Δ corridor (pp) [95% CI]"),
    )
    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        for intervention in INTERVENTIONS:
            record: dict[str, str] = {
                "Scenario": SCENARIO_LABELS[scenario],
                "Contrast": CONTRAST_LABELS[intervention],
            }
            for metric, scale, digits, label in metric_specs:
                row = effects[
                    (effects["scenario"] == scenario)
                    & (effects["intervention"] == intervention)
                    & (effects["metric"] == metric)
                ].iloc[0]
                record[label] = _format_estimate(
                    float(row["mean_delta"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                    scale=scale,
                    digits=digits,
                )
            rows.append(record)
    return pd.DataFrame(rows)


def _movement_effect_table(effects: pd.DataFrame) -> pd.DataFrame:
    metric_specs = (
        (
            "Total staff movement distance (km)",
            1,
            "Δ movement (km/run) [95% CI]",
        ),
        ("Zone transitions", 1, "Δ zone transitions [95% CI]"),
    )
    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        for intervention in INTERVENTIONS:
            record: dict[str, str] = {
                "Scenario": SCENARIO_LABELS[scenario],
                "Contrast": CONTRAST_LABELS[intervention],
            }
            for metric, digits, label in metric_specs:
                row = effects[
                    (effects["scenario"] == scenario)
                    & (effects["intervention"] == intervention)
                    & (effects["metric"] == metric)
                ].iloc[0]
                record[label] = _format_estimate(
                    float(row["mean_delta"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                    digits=digits,
                )
                if metric == "Total staff movement distance (km)":
                    percent = 100.0 * float(row["mean_delta"]) / float(row["baseline_mean"])
                    record["Δ movement (%)"] = f"{percent:+.1f}"
            rows.append(record)
    column_order = [
        "Scenario",
        "Contrast",
        "Δ movement (km/run) [95% CI]",
        "Δ movement (%)",
        "Δ zone transitions [95% CI]",
    ]
    return pd.DataFrame(rows)[column_order]


def _coordination_friction_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize complementary coordination-cost measures without compositing them."""

    metric_specs = (
        ("Metres travelled per staff–staff contact", "Coordination cost (m/contact)", 1),
        ("Mean separation between interactions (m)", "Separation (m) [95% CI]", 2),
        ("Role–zone segregation (0–1)", "Segregation [95% CI]", 3),
        (
            "Mutual staff visibility exposure (%)",
            "Mutual visibility (%)",
            2,
        ),
        (
            "Coordination-nurse participation in staff contacts (%)",
            "Coordination-nurse share (%)",
            1,
        ),
    )
    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        scenario_runs = runs[runs["scenario"] == scenario]
        footprint_pivot = scenario_runs.pivot(
            index="seed",
            columns="condition",
            values="90% interaction footprint (% of walkable ED)",
        )
        for condition in CONDITIONS:
            subset = scenario_runs[scenario_runs["condition"] == condition]
            record = {
                "Scenario": SCENARIO_LABELS[scenario],
                "Condition": CONDITION_LABELS[condition],
            }
            for metric, label, digits in metric_specs:
                values = subset[metric].dropna().to_numpy(dtype=float)
                if metric in {
                    "Mean separation between interactions (m)",
                    "Role–zone segregation (0–1)",
                }:
                    low, high = _ci95(values)
                    record[label] = (
                        f"{float(np.mean(values)):.{digits}f} "
                        f"[{low:.{digits}f}, {high:.{digits}f}]"
                    )
                else:
                    record[label] = f"{float(np.mean(values)):.{digits}f}"
            if condition == "baseline":
                record["Δ footprint (pp) [95% CI]"] = "—"
            else:
                footprint_change = (
                    footprint_pivot[condition] - footprint_pivot["baseline"]
                ).dropna().to_numpy(dtype=float)
                low, high = _ci95(footprint_change)
                record["Δ footprint (pp) [95% CI]"] = _format_estimate(
                    float(np.mean(footprint_change)), low, high, digits=1
                )
            rows.append(record)
    return pd.DataFrame(rows)


def _role_specific_spatial_effect_table(
    runs: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Decompose the combined intervention's movement and staff contact effects."""

    staff_events = events[events["is_hcw_hcw"].astype(bool)].copy()
    participation_records: list[dict[str, Any]] = []
    for role, count in ROLE_COUNTS.items():
        participation = (
            (staff_events["role_a"] == role).astype(int)
            + (staff_events["role_b"] == role).astype(int)
        )
        grouped = (
            staff_events.assign(participation=participation)
            .groupby(["scenario", "condition", "seed"], as_index=False)[
                "participation"
            ]
            .sum()
        )
        grouped["role"] = role
        grouped["participations_per_person_hour"] = (
            grouped["participation"] / count / 10.0
        )
        participation_records.extend(grouped.to_dict("records"))
    participations = pd.DataFrame(participation_records)

    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        scenario_runs = runs[runs["scenario"] == scenario]
        total_pivot = scenario_runs.pivot(
            index="seed",
            columns="condition",
            values="Total staff movement distance (km)",
        )
        total_delta = total_pivot["both"] - total_pivot["baseline"]
        for role, count in ROLE_COUNTS.items():
            movement_pivot = scenario_runs.pivot(
                index="seed",
                columns="condition",
                values=f"{role} movement (km/person-run)",
            )
            movement_delta = (
                movement_pivot["both"] - movement_pivot["baseline"]
            ).dropna()
            movement_low, movement_high = _ci95(
                movement_delta.to_numpy(dtype=float)
            )
            role_total_delta = movement_delta * count
            movement_share = (
                100.0
                * float(np.mean(role_total_delta))
                / float(np.mean(total_delta))
            )

            role_participations = participations[
                (participations["scenario"] == scenario)
                & (participations["role"] == role)
            ].pivot(
                index="seed",
                columns="condition",
                values="participations_per_person_hour",
            )
            contact_delta = (
                role_participations["both"] - role_participations["baseline"]
            ).dropna()
            contact_low, contact_high = _ci95(contact_delta.to_numpy(dtype=float))
            rows.append(
                {
                    "Scenario": SCENARIO_LABELS[scenario],
                    "Staff role": ROLE_LABELS[role],
                    "Δ movement/person (km/run) [95% CI]": _format_estimate(
                        float(np.mean(movement_delta)),
                        movement_low,
                        movement_high,
                        digits=2,
                    ),
                    "Added movement share (%)": f"{movement_share:.1f}",
                    "Δ staff contacts/person-hour [95% CI]": _format_estimate(
                        float(np.mean(contact_delta)),
                        contact_low,
                        contact_high,
                        digits=2,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_sensitivity_source(sensitivity_dir: Path) -> pd.DataFrame:
    candidates = (
        sensitivity_dir / "tables" / "tableB_sensitivity_detailed.csv",
    )
    for path in candidates:
        if path.exists():
            return _normalize_sensitivity_source(pd.read_csv(path))
    archive = (
        PROJECT_DIR
        / "outputs"
        / "source_results"
        / "part2_parameter_sensitivity_n20_outputs.zip"
    )
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            members = [
                name
                for name in bundle.namelist()
                if name.endswith("/analysis/parameter_sensitivity_detailed.csv")
            ]
            if len(members) == 1:
                with bundle.open(members[0]) as source:
                    return _normalize_sensitivity_source(pd.read_csv(source))
    raise SystemExit("No completed sensitivity detail table was found")


def _normalize_sensitivity_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the legacy analysis schema or the final readable schema."""

    if "metric" in frame.columns:
        return frame
    required = {"Factor", "Factor level", "Scenario", "Intervention", "Outcome"}
    if not required.issubset(frame.columns):
        raise SystemExit("The completed sensitivity detail table has an unsupported schema")
    reverse_scenarios = {
        **{value: key for key, value in SCENARIO_LABELS.items()},
        "High load / high acuity": "high_load_high_acuity",
    }
    reverse_conditions = {
        **{value: key for key, value in CONDITION_LABELS.items()},
        "Both interventions": "both",
    }
    reverse_metrics = {
        **{value: key for key, value in SENSITIVITY_METRIC_LABELS.items()},
        "Staff-staff interaction share": "hcw_hcw_share",
    }
    normalized = frame.rename(
        columns={
            "Factor": "parameter_label",
            "Factor level": "level_label",
            "Scenario": "scenario",
            "Intervention": "condition",
            "Outcome": "metric",
            "Default value": "sensitivity_default",
            "Test value": "sensitivity_value",
            "Paired n": "paired_n",
            "Default paired delta": "default_paired_delta",
            "Sensitivity paired delta": "sensitivity_paired_delta",
            "95% CI low": "ci95_low",
            "95% CI high": "ci95_high",
            "Effect retention": "effect_retention",
            "Direction consistency": "sign_consistency",
            "Standardized paired effect": "standardized_paired_effect",
            "Robustness verdict": "verdict",
        }
    )
    normalized["scenario"] = normalized["scenario"].map(reverse_scenarios)
    normalized["condition"] = normalized["condition"].map(reverse_conditions)
    normalized["metric"] = normalized["metric"].map(reverse_metrics)
    normalized["parameter_label"] = normalized["parameter_label"].replace(
        {"Crowding threshold": "Pressure activation threshold"}
    )
    normalized["level"] = normalized["level_label"].replace({"short": "low", "long": "high"})
    return normalized


def _clean_sensitivity_detail(source: pd.DataFrame) -> pd.DataFrame:
    result = source.copy()
    result = result[result["parameter_label"].isin(FACTOR_ORDER)].copy()
    result["parameter_label"] = result["parameter_label"].map(FACTOR_DISPLAY)
    result["scenario"] = result["scenario"].map(SCENARIO_LABELS)
    result["condition"] = result["condition"].map(CONDITION_LABELS)
    result["metric"] = result["metric"].map(SENSITIVITY_METRIC_LABELS)
    result = result.rename(
        columns={
            "parameter_label": "Factor",
            "level_label": "Factor level",
            "scenario": "Scenario",
            "condition": "Intervention",
            "metric": "Outcome",
            "sensitivity_default": "Default value",
            "sensitivity_value": "Test value",
            "paired_n": "Paired n",
            "default_paired_delta": "Default paired delta",
            "sensitivity_paired_delta": "Sensitivity paired delta",
            "effect_retention": "Effect retention",
            "ci95_low": "95% CI low",
            "ci95_high": "95% CI high",
            "sign_consistency": "Direction consistency",
            "standardized_paired_effect": "Standardized paired effect",
            "verdict": "Robustness verdict",
        }
    )
    columns = [
        "Factor",
        "Factor level",
        "Scenario",
        "Intervention",
        "Outcome",
        "Default value",
        "Test value",
        "Paired n",
        "Default paired delta",
        "Sensitivity paired delta",
        "95% CI low",
        "95% CI high",
        "Effect retention",
        "Direction consistency",
        "Standardized paired effect",
        "Robustness verdict",
    ]
    return result[columns].sort_values(
        ["Factor", "Scenario", "Intervention", "Outcome", "Factor level"]
    )


def _sensitivity_values_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Factor": "Visibility response",
                "Low": "0.07",
                "Default": "0.14",
                "High": "0.21",
                "Plain-language meaning": "Response probability after a staff member becomes visibly salient.",
            },
            {
                "Factor": "Crowding threshold",
                "Low": "Normal 0.50; High load 0.40",
                "Default": "Normal 0.62; High load 0.50",
                "High": "Normal 0.74; High load 0.62",
                "Plain-language meaning": "When ED pressure begins suppressing low-priority interactions.",
            },
            {
                "Factor": "Repeat-contact delay",
                "Low": "90 s",
                "Default": "180 s",
                "High": "360 s",
                "Plain-language meaning": "Minimum delay before another visibility-driven contact.",
            },
            {
                "Factor": "Station-return tendency",
                "Low": "0.20",
                "Default": "0.42",
                "High": "0.70",
                "Plain-language meaning": "Probability of a nurse station check after completing a task.",
            },
        ]
    )


def _default_effect_lookup(effects: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    lookup = {}
    subset = effects[effects["metric"] == "F2F interactions per hour"]
    for row in subset.itertuples(index=False):
        lookup[(row.scenario, row.intervention)] = (
            float(row.mean_delta),
            float(row.ci95_low),
            float(row.ci95_high),
        )
    return lookup


def _plot_sensitivity(
    source: pd.DataFrame,
    default_effects: pd.DataFrame,
    figures_dir: Path,
) -> None:
    f2f = source[source["metric"] == "f2f_per_hour"].copy()
    lookup = _default_effect_lookup(default_effects)
    fig, axes = plt.subplots(2, 4, figsize=(7.85, 4.55), sharex=True, sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.065, h_pad=0.045, wspace=0.035, hspace=0.055)
    x_values = np.array([0.0, 1.0, 2.0])
    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, factor in enumerate(FACTOR_ORDER):
            axis = axes[row_index, column_index]
            factor_rows = f2f[
                (f2f["scenario"] == scenario) & (f2f["parameter_label"] == factor)
            ]
            for intervention in INTERVENTIONS:
                line_rows = factor_rows[factor_rows["condition"] == intervention].set_index("level")
                low = line_rows.loc["low"]
                high = line_rows.loc["high"]
                default_mean, default_low, default_high = lookup[(scenario, intervention)]
                means = np.array(
                    [low["sensitivity_paired_delta"], default_mean, high["sensitivity_paired_delta"]],
                    dtype=float,
                )
                lows = np.array([low["ci95_low"], default_low, high["ci95_low"]], dtype=float)
                highs = np.array([low["ci95_high"], default_high, high["ci95_high"]], dtype=float)
                axis.plot(
                    x_values,
                    means,
                    color=COLORS[intervention],
                    linewidth=1.35,
                    marker="o",
                    markersize=3.3,
                    markeredgecolor="#3C3D42",
                    markeredgewidth=0.48,
                    zorder=3,
                )
                axis.errorbar(
                    x_values,
                    means,
                    yerr=np.vstack([means - lows, highs - means]),
                    color=COLORS[intervention],
                    linewidth=0,
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.72,
                    zorder=2,
                )
            axis.axhline(0, color=COLORS["line"], linewidth=0.75, zorder=0)
            axis.axvline(1, color=COLORS["line"], linewidth=0.75, linestyle="--", zorder=0)
            axis.set_xticks(x_values, ["Low", "Default", "High"])
            if row_index == 0:
                axis.set_title(FACTOR_DISPLAY[factor], pad=7, fontweight="bold")
            _despine(axis)
    y_min, y_max = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(y_min, y_max + 0.14 * (y_max - y_min))
    axes[0, 0].text(
        0.02, 0.965, "Normal load", transform=axes[0, 0].transAxes,
        ha="left", va="top", fontsize=7.8, fontweight="semibold", color=COLORS["ink"],
    )
    axes[1, 0].text(
        0.02, 0.965, "High load", transform=axes[1, 0].transAxes,
        ha="left", va="top", fontsize=7.8, fontweight="semibold", color=COLORS["ink"],
    )
    fig.supylabel("Δ F2F interactions per hour", x=-0.012, fontsize=8.2, fontweight="bold")
    handles = [
        Line2D([0], [0], color=COLORS[value], marker="o", lw=1.4, label=CONDITION_LABELS[value])
        for value in INTERVENTIONS
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    _save_figure(fig, figures_dir / "figA_sensitivity_response")
def _markdown_table(
    frame: pd.DataFrame,
    *,
    bold_columns: Sequence[str] = (),
    bold_cells: set[tuple[int, str]] | None = None,
) -> str:
    columns = [str(column) for column in frame.columns]
    emphasized = set(bold_columns)
    cells = bold_cells or set()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        values = [
            f"**{value}**"
            if column in emphasized or (row_index, column) in cells
            else value
            for column, value in zip(columns, values)
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    ascii_replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in ascii_replacements.items():
        text = text.replace(old, new)
    unicode_replacements = {
        "Δ": r"\ensuremath{\Delta}",
        "→": r"\ensuremath{\rightarrow}",
        "−": r"\ensuremath{-}",
        "×": r"\ensuremath{\times}",
        "²": r"\textsuperscript{2}",
        "–": "--",
        "—": "---",
        "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}",
    }
    for old, new in unicode_replacements.items():
        text = text.replace(old, new)
    return text


def _latex_table(
    frame: pd.DataFrame,
    *,
    bold_columns: Sequence[str] = (),
    bold_cells: set[tuple[int, str]] | None = None,
) -> str:
    alignment = "l" * len(frame.columns)
    columns = [str(column) for column in frame.columns]
    emphasized = set(bold_columns)
    cells = bold_cells or set()
    rows = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(r"\textbf{" + _latex_escape(column) + "}" for column in columns) + r" \\",
        r"\midrule",
    ]
    previous_scenario: str | None = None
    scenario_index = columns.index("Scenario") if "Scenario" in columns else None
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        if scenario_index is not None:
            scenario = str(row[scenario_index])
            if previous_scenario is not None and scenario != previous_scenario:
                rows.append(r"\addlinespace[2pt]")
            previous_scenario = scenario
        values = []
        for column, value in zip(columns, row):
            escaped = _latex_escape(value)
            values.append(
                r"\textbf{" + escaped + "}"
                if column in emphasized or (row_index, column) in cells
                else escaped
            )
        rows.append(" & ".join(values) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows)


def _write_table(
    frame: pd.DataFrame,
    base_path: Path,
    *,
    tex_and_markdown: bool = True,
    bold_columns: Sequence[str] = (),
    bold_cells: set[tuple[int, str]] | None = None,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base_path.with_suffix(".csv"), index=False)
    if tex_and_markdown:
        base_path.with_suffix(".md").write_text(
            _markdown_table(
                frame,
                bold_columns=bold_columns,
                bold_cells=bold_cells,
            )
        )
        base_path.with_suffix(".tex").write_text(
            _latex_table(
                frame,
                bold_columns=bold_columns,
                bold_cells=bold_cells,
            )
        )


def _estimate_excludes_zero(value: Any) -> bool:
    match = re.search(r"\[([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\]", str(value))
    if not match:
        return False
    low, high = (float(match.group(1)), float(match.group(2)))
    return low > 0.0 or high < 0.0


def _significant_cells(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[int, str]]:
    return {
        (row_index, column)
        for row_index, row in frame.iterrows()
        for column in columns
        if _estimate_excludes_zero(row[column])
    }


def _write_vga_table(frame: pd.DataFrame, base_path: Path, metrics: pd.DataFrame) -> None:
    note = (
        "VGA-style approximations on a 1.0 m regular grid over the largest connected walkable "
        "component. Connectivity is the mean count of directly visible grid points. Isovist area "
        "is the mean 360-degree ray-cast visible area. Visual integration is mean "
        "normalized closeness centrality on the visibility graph. Visual intelligibility is the "
        "Pearson correlation between local connectivity and global visual integration across grid "
        "points. Condition-specific visibility walls are used throughout; these are not formal "
        "DepthmapX outputs."
    )
    _write_table(frame, base_path)
    point_counts = ", ".join(
        f"{CONDITION_LABELS[row.condition]} n={int(row.sample_points)}"
        for row in metrics.itertuples(index=False)
    )
    markdown = base_path.with_suffix(".md")
    markdown.write_text(markdown.read_text() + f"\n*Note.* {note} Sample points: {point_counts}.\n")
    latex = base_path.with_suffix(".tex")
    latex.write_text(
        latex.read_text()
        + "\n\\par\\footnotesize\\textit{Note.} "
        + _latex_escape(note + " Sample points: " + point_counts + ".")
        + "\n"
    )


def _write_effect_table(frame: pd.DataFrame, base_path: Path) -> None:
    note = (
        "All effects use paired seeds (n=100 per scenario and contrast). Values in brackets are "
        "95% confidence intervals for paired mean differences. Effect magnitudes and uncertainty "
        "are reported in their original units; standardized effects and p-values are intentionally "
        "not foregrounded. Bold values have 95% confidence intervals that exclude zero."
    )
    effect_columns = tuple(
        column for column in frame.columns if str(column).endswith("[95% CI]")
    )
    _write_table(
        frame,
        base_path,
        bold_cells=_significant_cells(frame, effect_columns),
    )
    markdown = base_path.with_suffix(".md")
    markdown.write_text(markdown.read_text() + f"\n*Note.* {note}\n")
    latex = base_path.with_suffix(".tex")
    latex.write_text(
        latex.read_text()
        + "\n\\par\\footnotesize\\textit{Note.} "
        + _latex_escape(note)
        + "\n"
    )


def _write_ecology_table(frame: pd.DataFrame, base_path: Path) -> None:
    note = (
        "Coordination cost uses validation-counted staff–staff contacts. Interaction "
        "separation is the mean pairwise Euclidean distance between interaction locations. "
        "Role–location segregation is equal-count normalized mutual information between role pair "
        "and broad zone. Mutual staff visibility is the share of all possible staff-pair seconds "
        "during the complete 12-hour run in which the two staff members could mutually perceive "
        "one another. The 90% footprint is the share of the largest connected walkable ED grid "
        "component required to contain 90% of normalized interaction density (0.75 m grid; "
        "1.125 m Gaussian bandwidth); its change is reported in percentage points relative to the "
        "same-scenario baseline. Separation and segregation intervals summarize condition-level "
        "seed variation; footprint change intervals use paired "
        "seeds. Coordination-nurse participation is the share of validation-counted "
        "staff–staff contacts involving the coordination nurse."
    )
    _write_table(frame, base_path)
    markdown = base_path.with_suffix(".md")
    markdown.write_text(markdown.read_text() + f"\n*Note.* {note}\n")
    latex = base_path.with_suffix(".tex")
    latex.write_text(
        latex.read_text()
        + "\n\\par\\footnotesize\\textit{Note.} "
        + _latex_escape(note)
        + "\n"
    )


def _write_role_effect_table(frame: pd.DataFrame, base_path: Path) -> None:
    note = (
        "Both minus Baseline, paired by seed (n=100 per scenario). Movement is "
        "normalized by the fixed role composition: one coordination nurse, three "
        "doctors, and five nurses. Staff-contact participation counts each staff "
        "member involved in a validation-counted staff–staff interaction and is "
        "reported per person across the 10-hour evaluation window. Movement share "
        "decomposes the total added staff movement; rounding may prevent exact summation "
        "to 100%. Bold identifies the role contributing the majority of added movement."
    )
    coordination_rows = frame.index[
        frame["Staff role"].eq("Coordination nurse")
    ].tolist()
    bold_cells = {
        (row_index, column)
        for row_index in coordination_rows
        for column in ("Staff role", "Added movement share (%)")
    }
    _write_table(frame, base_path, bold_cells=bold_cells)
    markdown = base_path.with_suffix(".md")
    markdown.write_text(markdown.read_text() + f"\n*Note.* {note}\n")
    latex = base_path.with_suffix(".tex")
    latex.write_text(
        latex.read_text()
        + "\n\\par\\footnotesize\\textit{Note.} "
        + _latex_escape(note)
        + "\n"
    )


def _prepare_output_dirs(spatial_dir: Path, sensitivity_dir: Path) -> tuple[Path, Path, Path, Path]:
    spatial_figures = spatial_dir / "figures"
    spatial_tables = spatial_dir / "tables"
    sensitivity_figures = sensitivity_dir / "figures"
    sensitivity_tables = sensitivity_dir / "tables"
    for path in (spatial_figures, spatial_tables, sensitivity_figures, sensitivity_tables):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return spatial_figures, spatial_tables, sensitivity_figures, sensitivity_tables


def _finalize_sensitivity_layout(sensitivity_dir: Path) -> None:
    for name in ("analysis",):
        path = sensitivity_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("README.md", "integrity.json"):
        path = sensitivity_dir / name
        if path.exists():
            path.unlink()


def build(args: argparse.Namespace) -> None:
    font_family = _configure_style()
    canonical_dir = args.canonical_dir.resolve()
    spatial_dir = args.spatial_out.resolve()
    sensitivity_dir = args.sensitivity_out.resolve()
    sensitivity_source = _load_sensitivity_source(sensitivity_dir)
    runs = _load_runs(canonical_dir)
    events, event_file_count = _load_events(canonical_dir)
    runs = runs.merge(
        _spatial_ecology_metrics(events),
        on=["scenario", "condition", "seed"],
        how="left",
        validate="one_to_one",
    )
    effects = _paired_effects(runs, (*SUMMARY_METRICS, *MOVEMENT_METRICS, *FRICTION_METRICS))
    vga_metrics = _whole_space_vga_metrics(resolution_m=1.0)

    spatial_figures, spatial_tables, sensitivity_figures, sensitivity_tables = _prepare_output_dirs(
        spatial_dir, sensitivity_dir
    )

    _plot_intervention_atlas(spatial_figures)
    _plot_spatial_redistribution(events, spatial_figures)
    _plot_spatial_coordination_ecology(runs, spatial_figures)
    _plot_sensitivity(sensitivity_source, effects, sensitivity_figures)

    movement_table = _movement_effect_table(effects)
    _write_table(
        movement_table,
        spatial_tables / "tableA_movement_effects",
        bold_cells=_significant_cells(
            movement_table,
            (
                "Δ movement (km/run) [95% CI]",
                "Δ zone transitions [95% CI]",
            ),
        ),
    )
    movement_note = (
        "Both scenarios and contrasts use paired seeds (n=100). Values in brackets are "
        "95% confidence intervals for paired mean differences. Percent change is relative "
        "to the same-scenario Baseline. Bold values have 95% confidence intervals that "
        "exclude zero."
    )
    movement_base = spatial_tables / "tableA_movement_effects"
    movement_base.with_suffix(".md").write_text(
        movement_base.with_suffix(".md").read_text()
        + f"\n*Note.* {movement_note}\n"
    )
    movement_base.with_suffix(".tex").write_text(
        movement_base.with_suffix(".tex").read_text()
        + "\n\\par\\footnotesize\\textit{Note.} "
        + _latex_escape(movement_note)
        + "\n"
    )
    _write_vga_table(
        _vga_spatial_table(vga_metrics),
        spatial_tables / "tableB_vga_spatial_metrics",
        vga_metrics,
    )
    _write_effect_table(
        _paired_effect_table(effects),
        spatial_tables / "tableC_paired_intervention_effects",
    )
    _write_ecology_table(
        _coordination_friction_table(runs),
        spatial_tables / "tableD_coordination_friction_profile",
    )
    _write_role_effect_table(
        _role_specific_spatial_effect_table(runs, events),
        spatial_tables / "tableE_role_specific_spatial_effects",
    )
    _write_table(
        _sensitivity_values_table(),
        sensitivity_tables / "tableA_sensitivity_values",
        bold_columns=("Default",),
    )
    _write_table(
        _clean_sensitivity_detail(sensitivity_source),
        sensitivity_tables / "tableB_sensitivity_detailed",
        tex_and_markdown=False,
    )
    _finalize_sensitivity_layout(sensitivity_dir)

    print(f"Figure font: {font_family}")
    print(f"Intervention summaries: {len(runs)}")
    print(f"Interaction-event files: {event_file_count}")
    print(f"Validation-counted coordinate rows: {len(events)}")
    print(f"Spatial outputs: {spatial_dir}")
    print(f"Sensitivity outputs: {sensitivity_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "source_results" / "part2_spatial_interventions_n100",
    )
    parser.add_argument(
        "--spatial-out",
        type=Path,
        default=PROJECT_DIR / "outputs" / "findings" / "part2_spatial_interventions_n100",
    )
    parser.add_argument(
        "--sensitivity-out",
        type=Path,
        default=PROJECT_DIR / "outputs" / "findings" / "part2_parameter_sensitivity_n20",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
