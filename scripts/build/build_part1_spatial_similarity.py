#!/usr/bin/env python3
"""Rebuild the Part 1 point-overlay and heatmap figure from frozen outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = PROJECT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.analysis import compute_kde_grid
from src.environment import Environment


OBSERVED = "#65AAA6"
SIMULATED = "#62577E"
WALL_LIGHT = "#B8BEC7"
WALL_DARK = "#2E2E2E"
INK = "#2F2F37"
PAPER = "#FFFFFF"


def _points() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(
        WORKSPACE_DIR / "source-materials" / "shadowing_with-participant-info.csv"
    )
    observed = observed.loc[
        observed["interfaceType"].fillna("").astype(str).str.contains("Person", case=False),
        ["x_shadowing", "y_shadowing"],
    ].copy()
    simulated_frames: list[pd.DataFrame] = []
    source = (
        PROJECT_DIR
        / "outputs"
        / "source_results"
        / "part1_validation_n100"
        / "normal_load"
        / "baseline"
    )
    for path in sorted(source.glob("seed_*/interaction_events.csv")):
        frame = pd.read_csv(path)
        simulated_frames.append(
            frame.loc[frame["counted_for_validation"].astype(bool), ["x", "y"]]
        )
    simulated = pd.concat(simulated_frames, ignore_index=True)
    if len(observed) != 359 or len(simulated) != 22_142:
        raise ValueError(
            f"Unexpected point counts: observed={len(observed)}, simulated={len(simulated)}"
        )
    return observed, simulated


def _draw_walls(axis: plt.Axes, environment: Environment, color: str, width: float) -> None:
    for start, end in environment.walls:
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            solid_capstyle="projecting",
            zorder=4,
        )


def _format_map(
    axis: plt.Axes,
    environment: Environment,
    *,
    show_ylabel: bool,
    expanded_top: bool,
) -> None:
    xmin, xmax, ymin, ymax = environment.plot_bounds
    axis.set_xlim(xmin, xmax)
    # The extra metre prevents the northern wall from sitting on the clipping
    # boundary in the compact heatmap panels.
    axis.set_ylim(ymin, ymax + (1.1 if expanded_top else 0.0))
    axis.set_aspect("equal", adjustable="box")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(labelsize=7)
    axis.set_xlabel("x (m)", fontsize=7.5)
    if show_ylabel:
        axis.set_ylabel("y (m)", fontsize=7.5)
    else:
        axis.set_yticklabels([])


def _heatmap(
    axis: plt.Axes,
    frame: pd.DataFrame,
    columns: tuple[str, str],
    environment: Environment,
) -> None:
    points = [tuple(item) for item in frame.loc[:, list(columns)].to_numpy(dtype=float)]
    density = compute_kde_grid(
        points,
        environment.plot_bounds,
        resolution=0.5,
        bandwidth=1.5,
    )
    positive = density[density > 0]
    maximum = float(np.quantile(positive, 0.992))
    colors = [
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 0.94, 0.62, 0.52),
        (0.99, 0.66, 0.22, 0.78),
        (0.86, 0.10, 0.08, 0.95),
    ]
    cmap = LinearSegmentedColormap.from_list("ed_hotspots", colors, N=256)
    xmin, xmax, ymin, ymax = environment.plot_bounds
    axis.imshow(
        density,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        norm=PowerNorm(gamma=0.62, vmin=0.0, vmax=maximum),
        interpolation="bilinear",
        zorder=1,
    )
    _draw_walls(axis, environment, WALL_DARK, 0.68)
    _format_map(axis, environment, show_ylabel=False, expanded_top=True)


def main() -> None:
    observed, simulated = _points()
    environment = Environment(config.WALL_POSITIONS_PATH, config.ZONE_BOUNDARIES_PATH)
    rng = np.random.default_rng(49)
    sampled = simulated.iloc[rng.choice(len(simulated), size=len(observed), replace=False)].copy()
    # Display-only jitter separates coincident points. It never enters analysis.
    observed_jitter = observed.to_numpy(dtype=float) + rng.normal(0.0, 0.08, (len(observed), 2))
    simulated_jitter = sampled.to_numpy(dtype=float) + rng.normal(0.0, 0.08, (len(sampled), 2))

    fig = plt.figure(figsize=(9.2, 5.32), facecolor=PAPER)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(2.03, 1.0),
        left=0.055,
        right=0.985,
        top=0.95,
        bottom=0.095,
        wspace=0.14,
    )
    left = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, hspace=0.36)
    empirical_axis = fig.add_subplot(right[0, 0])
    simulated_axis = fig.add_subplot(right[1, 0])

    _draw_walls(left, environment, WALL_LIGHT, 0.55)
    left.scatter(
        observed_jitter[:, 0],
        observed_jitter[:, 1],
        s=12,
        color=OBSERVED,
        alpha=0.43,
        linewidths=0,
        zorder=2,
    )
    left.scatter(
        simulated_jitter[:, 0],
        simulated_jitter[:, 1],
        s=8,
        color=SIMULATED,
        alpha=0.58,
        linewidths=0,
        zorder=3,
    )
    _format_map(left, environment, show_ylabel=True, expanded_top=False)
    left.set_title("Point overlay", loc="left", fontsize=10.5, fontweight="normal", pad=6)
    left.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="", markersize=5.5, color=OBSERVED, label="Observed"),
            Line2D([], [], marker="o", linestyle="", markersize=5.5, color=SIMULATED, label="Simulated sample"),
        ),
        frameon=False,
        loc="lower left",
        fontsize=7,
        borderpad=0.2,
        handletextpad=0.5,
    )

    _heatmap(empirical_axis, observed, ("x_shadowing", "y_shadowing"), environment)
    _heatmap(simulated_axis, simulated, ("x", "y"), environment)
    empirical_axis.set_title("Empirical heatmap", loc="left", fontsize=10.5, fontweight="normal", pad=6)
    simulated_axis.set_title("Simulated heatmap", loc="left", fontsize=10.5, fontweight="normal", pad=6)

    findings_output = (
        PROJECT_DIR / "outputs" / "findings" / "part1_validation_n100" / "figures"
    )
    manuscript_output = WORKSPACE_DIR / "manuscript" / "figures"
    findings_output.mkdir(parents=True, exist_ok=True)
    manuscript_output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "facecolor": PAPER}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(findings_output / f"figC_spatial_similarity.{extension}", **kwargs)
        if extension == "pdf":
            fig.savefig(
                manuscript_output / "part1-spatial-similarity.pdf",
                **kwargs,
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
