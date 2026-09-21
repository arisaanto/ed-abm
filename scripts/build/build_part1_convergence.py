#!/usr/bin/env python3
"""Build the Part 1 replication-convergence figure from frozen run summaries."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNS = PROJECT_DIR / "outputs/source_results/part1_validation_n100/normal_load/baseline"
OUTPUTS = (
    PROJECT_DIR / "outputs/findings/part1_validation_n100/figures",
    PROJECT_DIR.parent / "manuscript/figures",
)
PURPLE = "#746a8d"


def load_rates() -> list[float]:
    """Return the 100 frozen ten-hour interaction rates in seed order."""
    rates: list[tuple[int, float]] = []
    for path in RUNS.glob("seed_*/summary.json"):
        with path.open() as handle:
            record = json.load(handle)
        rates.append(
            (
                int(record["metadata"]["seed"]),
                float(record["validation_metrics"]["f2f_per_hour"]),
            )
        )
    rates.sort()
    if [seed for seed, _ in rates] != list(range(1, 101)):
        raise ValueError("Expected exactly the frozen Part 1 seeds 1-100")
    return [rate for _, rate in rates]


def cumulative_statistics(rates: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = []
    errors = []
    for n in range(1, len(rates) + 1):
        sample = rates[:n]
        means.append(statistics.mean(sample))
        errors.append(statistics.stdev(sample) / math.sqrt(n) if n > 1 else np.nan)
    means_array = np.asarray(means)
    errors_array = np.asarray(errors)
    return means_array, means_array - 1.96 * errors_array, means_array + 1.96 * errors_array


def main() -> None:
    rates = load_rates()
    x = np.arange(1, len(rates) + 1)
    means, lower, upper = cumulative_statistics(rates)
    standard_errors = np.asarray(
        [
            statistics.stdev(rates[:n]) / math.sqrt(n) if n > 1 else np.nan
            for n in x
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": "#25272b",
            "axes.linewidth": 1.0,
        }
    )
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.35, 1], "hspace": 0.08},
    )

    top.fill_between(x[1:], lower[1:], upper[1:], color=PURPLE, alpha=0.13, linewidth=0)
    top.plot(x, means, color=PURPLE, linewidth=2.2)
    for n in (20, 50):
        top.scatter(n, means[n - 1], s=28, color="#303136", zorder=4)
    top.scatter(100, means[-1], s=58, color=PURPLE, zorder=4)
    top.annotate(
        f"n=100\n{means[-1]:.2f}/h",
        (100, means[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        color=PURPLE,
        fontsize=9,
    )
    top.set_ylabel("Cumulative F2F/hour")
    top.set_ylim(20.8, max(23.65, float(np.nanmax(upper)) + 0.08))
    top.set_yticks(np.arange(21.0, 23.6, 0.5))

    bottom.plot(x[1:], standard_errors[1:], color="#737b86", linewidth=1.8)
    bottom.fill_between(x[1:], 0, standard_errors[1:], color="#737b86", alpha=0.09, linewidth=0)
    bottom.scatter(100, standard_errors[-1], s=42, color="#737b86", zorder=4)
    bottom.annotate(
        f"{standard_errors[-1]:.2f}",
        (100, standard_errors[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        color="#737b86",
        fontsize=9,
    )
    bottom.set_ylabel("SE of running mean")
    bottom.set_xlabel("Runs")
    bottom.set_ylim(0, 0.62)
    bottom.set_yticks((0.0, 0.2, 0.4, 0.6))
    bottom.set_xticks((20, 40, 60, 80, 100))

    for axis in (top, bottom):
        axis.grid(axis="y", color="#dedbd4", linewidth=0.8, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(1, 104)

    figure.subplots_adjust(left=0.10, right=0.92, bottom=0.11, top=0.98)
    for directory in OUTPUTS:
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / "part1-replication-convergence"
        figure.savefig(stem.with_suffix(".png"), dpi=300)
        figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
