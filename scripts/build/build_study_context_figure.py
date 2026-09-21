#!/usr/bin/env python3
"""Build the floor-plan and intervention context figure used as Figure 2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/abm-context-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/abm-context-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.conditions import ConditionManager, get_condition_spec
from src.environment import Environment
from src.simulation import Simulation


COLORS = {
    "ink": "#24252A",
    "muted": "#737B87",
    "line": "#D9D7D2",
    "patient": "#386CB0",
    "CoordinationNurse": "#E68613",
    "Nurse": "#3A923A",
    "Doctor": "#C23B33",
    "transparent": "#A7AAAE",
    "standing": "#E9E5EF",
    "box": "#74698C",
}
MARKERS = {
    "Patient": "o",
    "CoordinationNurse": "D",
    "Nurse": "s",
    "Doctor": "^",
}
ZONE_LABELS = {
    "COCPIT": "COCPIT\ncentral station",
    "NURSTA": "NURSTA\nsatellite station",
    "NUROPE": "NUROPE\nnursing work area",
    "OFFSPA": "OFFSPA\noffice space",
    "NEUOFF": "NEUOFF\nadjacent office area",
}


def _manager(condition: str) -> ConditionManager:
    environment = Environment(config.WALL_POSITIONS_PATH, config.ZONE_BOUNDARIES_PATH)
    return ConditionManager(environment, get_condition_spec(condition))


def _draw_plan(axis, manager: ConditionManager, *, labels: bool, quiet: bool = False) -> None:
    for zone_id, polygon in manager.zone_polygons.items():
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        axis.plot(xs, ys, color="#CCD1D6", linewidth=0.42, zorder=1)
        if labels:
            x, y = manager.zone_centroid(zone_id)
            label = ZONE_LABELS.get(zone_id, zone_id)
            label_size = 5.4 if zone_id in ZONE_LABELS else 6.1
            axis.text(
                x, y, label, ha="center", va="center", fontsize=label_size,
                color=COLORS["muted"], linespacing=0.92, zorder=2,
            )
    suppressed = manager.render_suppressed_wall_indices()
    for index, (start, end) in enumerate(manager.environment.walls):
        if index in suppressed:
            continue
        transparent = index in manager.transparent_wall_indices
        axis.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=COLORS["transparent"] if transparent else COLORS["ink"],
            linewidth=0.78 if transparent else (0.55 if quiet else 0.75),
            alpha=0.42 if transparent else (0.58 if quiet else 0.82),
            solid_capstyle="round", zorder=4,
        )
    for start, end in manager.render_added_wall_segments():
        axis.plot(
            [start[0], end[0]], [start[1], end[1]], color=COLORS["ink"],
            linewidth=1.05, solid_capstyle="round", zorder=5,
        )


def _map_axis(axis, manager: ConditionManager, bounds=None) -> None:
    xmin, xmax, ymin, ymax = bounds or manager.environment.plot_bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_snapshot(axis, simulation: Simulation) -> None:
    agents = [*simulation.active_patients, *simulation.staff_agents]
    drawn = set()
    for agent in agents:
        role = str(agent.role)
        color = COLORS.get(role, COLORS["patient"])
        marker = MARKERS.get(role, "o")
        size = 22 if role == "CoordinationNurse" else (15 if role == "Patient" else 19)
        axis.scatter(
            agent.position[0], agent.position[1], marker=marker, s=size,
            facecolor=color, edgecolor="white", linewidth=0.42,
            alpha=0.95, zorder=9,
            label=role if role not in drawn else None,
        )
        drawn.add(role)


def _draw_change_panel(axis, *, condition: str, bounds, title: str) -> None:
    baseline = _manager("baseline")
    changed = _manager(condition)
    inset_left = axis.inset_axes([0.00, 0.12, 0.47, 0.75])
    inset_right = axis.inset_axes([0.53, 0.12, 0.47, 0.75])
    for inset, manager, heading in (
        (inset_left, baseline, "Before"),
        (inset_right, changed, "After"),
    ):
        inset.set_facecolor("#FAFAF8")
        if condition == "nursta_only":
            polygon = manager.zone_polygon("NURSTA")
            inset.fill(
                [p[0] for p in polygon], [p[1] for p in polygon],
                color=COLORS["standing"], alpha=0.72, zorder=0,
            )
        _draw_plan(inset, manager, labels=False, quiet=True)
        _map_axis(inset, manager, bounds)
        inset.set_title(heading, fontsize=6.5, fontweight="bold", pad=2)
    axis.axis("off")
    axis.text(0.0, 1.01, title, transform=axis.transAxes, fontsize=8.0,
              fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])


def _draw_heatmap_comparison(axis) -> None:
    """Reuse the two heatmaps from the established Part 1 spatial figure."""
    source = (
        PROJECT_DIR / "outputs" / "findings" / "part1_validation_n100"
        / "figures" / "figC_spatial_similarity.png"
    )
    image = Image.open(source).convert("RGB")
    width, height = image.size
    # The original figure places the empirical and simulated heatmaps in the
    # rightmost third. Crop each plot, excluding its old title and axis labels,
    # then place the two maps side by side in the context figure.
    empirical = image.crop((int(0.662 * width), int(0.050 * height),
                            int(0.995 * width), int(0.485 * height)))
    simulated = image.crop((int(0.662 * width), int(0.555 * height),
                            int(0.995 * width), int(0.965 * height)))
    left = axis.inset_axes([0.00, 0.04, 0.48, 0.84])
    right = axis.inset_axes([0.52, 0.04, 0.48, 0.84])
    for inset, crop, heading in (
        (left, empirical, "Observed"),
        (right, simulated, "Simulated"),
    ):
        inset.imshow(crop)
        inset.set_title(heading, fontsize=6.5, fontweight="bold", pad=1.5)
        inset.axis("off")
    axis.axis("off")
    axis.text(0.0, 1.01, "Interaction density", transform=axis.transAxes,
              fontsize=8.0, fontweight="bold", ha="left", va="bottom",
              color=COLORS["ink"])


def build(output_dir: Path, *, seed: int, snapshot_seconds: int) -> None:
    family = "DejaVu Sans"
    for candidate in ("Helvetica", "Arial", "DejaVu Sans"):
        try:
            matplotlib.font_manager.findfont(candidate, fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update({
        "font.family": family, "font.sans-serif": [family, "Arial", "DejaVu Sans"],
        "font.size": 8.0, "axes.titlesize": 9.0, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    simulation = Simulation(
        random_seed=seed, condition_name="baseline", scenario_mode="normal_load",
        scenario_start_hour=7,
    )
    simulation.run(snapshot_seconds)
    manager = simulation.condition_manager

    figure = plt.figure(figsize=(8.15, 4.8), constrained_layout=True)
    grid = figure.add_gridspec(
        3, 2, width_ratios=(1.62, 1.0), height_ratios=(1.18, 1.0, 1.0),
        hspace=0.08, wspace=0.07,
    )
    left = figure.add_subplot(grid[:, 0])
    top = figure.add_subplot(grid[0, 1])
    middle = figure.add_subplot(grid[1, 1])
    bottom = figure.add_subplot(grid[2, 1])

    _draw_plan(left, manager, labels=True)
    _draw_snapshot(left, simulation)
    focus_bounds = (-4.45, 4.55, -11.55, 1.75)
    left.add_patch(Rectangle(
        (focus_bounds[0], focus_bounds[2]),
        focus_bounds[1] - focus_bounds[0], focus_bounds[3] - focus_bounds[2],
        facecolor="#E9E5EF", fill=True, alpha=0.36,
        edgecolor=COLORS["box"], linewidth=0.9,
        linestyle=(0, (2, 2)), zorder=3,
    ))
    _map_axis(left, manager)
    left.set_title(
        "ED layout",
        loc="left", fontsize=8.4, fontweight="bold", pad=4,
    )
    handles = [
        Line2D([0], [0], marker=MARKERS[role], color="none",
               markerfacecolor=COLORS[role], markeredgecolor="white",
               markersize=5.5, label=label)
        for role, label in (
            ("CoordinationNurse", "Coordination nurse"),
            ("Nurse", "Nurse"), ("Doctor", "Doctor")
        )
    ]
    handles.append(Line2D([0], [0], marker=MARKERS["Patient"], color="none",
                          markerfacecolor=COLORS["patient"], markeredgecolor="white",
                          markersize=5.0, label="Patient"))
    left.legend(handles=handles, loc="upper left", frameon=False, ncol=2,
                fontsize=6.1, columnspacing=0.8, handletextpad=0.35)

    _draw_heatmap_comparison(top)
    _draw_change_panel(
        middle, condition="cockpit_only", bounds=(-4.2, 4.2, -2.8, 1.2),
        title="Transparent central station",
    )
    _draw_change_panel(
        bottom, condition="nursta_only", bounds=(-4.2, 4.2, -12.0, -5.4),
        title="Relocated satellite nurse station",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "study-context-floorplan"
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_DIR / "outputs" / "findings" / "study_overview" / "figures",
    )
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--snapshot-seconds", type=int, default=10_800)
    args = parser.parse_args()
    build(args.output_dir, seed=args.seed, snapshot_seconds=args.snapshot_seconds)


if __name__ == "__main__":
    main()
