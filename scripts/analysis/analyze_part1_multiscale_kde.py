#!/usr/bin/env python3
"""Cross-validated, multiscale comparison of observed and simulated hotspots.

The six observed dates are divided into the ten unique three-day versus
three-day partitions. For each partition, the observed halves define the
empirical repeatability benchmark. The pooled ABM cloud and size-matched ABM
samples are compared with each half in turn and the two directions are
averaged, so the result does not depend on which half is called held out.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = PROJECT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.analysis import compute_kde_grid
from src.empirical import classify_zone_supergroup
from src.environment import Environment


BANDWIDTHS_M = (0.5, 1.0, 2.0, 4.0)
COLORS = {
    "ink": "#27272A",
    "observed": "#65AAA6",
    "simulated": "#62577E",
    "line": "#B8B8B3",
    "paper": "#FCFCFA",
}


def _density(
    frame: pd.DataFrame,
    columns: tuple[str, str],
    bandwidth: float,
    environment: Environment,
) -> np.ndarray:
    points = [tuple(row) for row in frame.loc[:, list(columns)].astype(float).to_numpy()]
    density = compute_kde_grid(
        points,
        environment.plot_bounds,
        resolution=1.0,
        bandwidth=bandwidth,
    ).ravel()
    density = np.clip(np.asarray(density, dtype=float), 0.0, None)
    # A very narrow KDE can underflow to exact zero in one map while retaining
    # a subnormal positive value in the other.  Flooring both distributions at
    # machine tiny preserves the limit and avoids an artificial infinite JSD.
    density = np.maximum(density, np.finfo(float).tiny)
    total = float(density.sum())
    if total <= 0.0:
        raise ValueError("KDE contained no positive density")
    return density / total


def _jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the square root of Jensen-Shannon divergence using natural logs."""

    midpoint = 0.5 * (left + right)
    divergence = 0.5 * float(rel_entr(left, midpoint).sum())
    divergence += 0.5 * float(rel_entr(right, midpoint).sum())
    return float(np.sqrt(max(divergence, 0.0)))


def _annotate_observed_zones(
    observed: pd.DataFrame,
    environment: Environment,
) -> pd.DataFrame:
    annotated = observed.copy()
    groups: list[str] = []
    for x_coord, y_coord in annotated[["x_shadowing", "y_shadowing"]].to_numpy(
        dtype=float
    ):
        zone_id = environment.which_zone(float(x_coord), float(y_coord))
        groups.append(
            classify_zone_supergroup(
                zone_id or "Outside named zones",
                float(x_coord),
                float(y_coord),
            )
        )
    annotated["zone_supergroup"] = groups
    return annotated


def _random_candidate_pools(
    environment: Environment,
    resolution: float = 0.25,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return uniformly spaced feasible care-area points and broad-zone pools."""

    xmin, xmax, ymin, ymax = environment.plot_bounds
    candidates: list[tuple[float, float, str]] = []
    for x_coord in np.arange(xmin + resolution / 2, xmax, resolution):
        for y_coord in np.arange(ymin + resolution / 2, ymax, resolution):
            zone_id = environment.which_zone(float(x_coord), float(y_coord))
            group = classify_zone_supergroup(
                zone_id or "Outside named zones",
                float(x_coord),
                float(y_coord),
            )
            if zone_id is not None or group == "bedside_or_patient_room":
                candidates.append((float(x_coord), float(y_coord), group))
    if not candidates:
        raise ValueError("No feasible random-placement candidates were found")
    all_points = np.asarray([(x, y) for x, y, _ in candidates], dtype=float)
    pools = {
        group: np.asarray(
            [(x, y) for x, y, candidate_group in candidates if candidate_group == group],
            dtype=float,
        )
        for group in {group for _, _, group in candidates}
    }
    return all_points, pools


def _draw_global_random(
    candidates: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    indices = rng.choice(len(candidates), size=size, replace=True)
    return pd.DataFrame(candidates[indices], columns=["x", "y"])


def _draw_zone_preserving_random(
    reference: pd.DataFrame,
    all_candidates: np.ndarray,
    candidate_pools: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled: list[np.ndarray] = []
    counts = reference["zone_supergroup"].value_counts()
    for group, count in counts.items():
        # Uncoded empirical points have no defensible geometric zone. Their
        # count is retained, but their null locations are drawn globally.
        pool = candidate_pools.get(str(group), all_candidates)
        indices = rng.choice(len(pool), size=int(count), replace=True)
        sampled.append(pool[indices])
    points = np.concatenate(sampled, axis=0)
    rng.shuffle(points)
    return pd.DataFrame(points, columns=["x", "y"])


def _load_simulated_points(source_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(source_dir.glob("seed_*/interaction_events.csv")):
        frame = pd.read_csv(path)
        selected = frame.loc[frame["counted_for_validation"].astype(bool), ["x", "y"]]
        frames.append(selected)
    if len(frames) != 100:
        raise ValueError(f"Expected 100 baseline seeds; found {len(frames)}")
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 22_142:
        raise ValueError(f"Expected 22,142 simulated points; found {len(result)}")
    return result


def _analyse(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    environment: Environment,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = _annotate_observed_zones(observed, environment)
    all_random_candidates, zone_candidate_pools = _random_candidate_pools(environment)
    dates = sorted(observed["date"].astype(str).unique())
    if len(dates) != 6:
        raise ValueError(f"Expected six observation dates; found {len(dates)}")
    split_definitions = [
        split for split in combinations(dates, 3) if dates[0] in split
    ]
    model_rng = np.random.default_rng(random_seed)
    global_rng = np.random.default_rng(random_seed + 1)
    zone_rng = np.random.default_rng(random_seed + 2)
    split_rows: list[dict[str, object]] = []

    for bandwidth in BANDWIDTHS_M:
        pooled_simulated = _density(simulated, ("x", "y"), bandwidth, environment)
        for split_index, training_dates in enumerate(split_definitions, start=1):
            training = observed.loc[observed["date"].astype(str).isin(training_dates)]
            held_out = observed.loc[~observed["date"].astype(str).isin(training_dates)]
            training_density = _density(
                training, ("x_shadowing", "y_shadowing"), bandwidth, environment
            )
            held_out_density = _density(
                held_out, ("x_shadowing", "y_shadowing"), bandwidth, environment
            )
            observed_distance = _jensen_shannon_distance(
                training_density, held_out_density
            )
            pooled_distance = 0.5 * (
                _jensen_shannon_distance(pooled_simulated, held_out_density)
                + _jensen_shannon_distance(pooled_simulated, training_density)
            )
            matched_distances: list[float] = []
            global_random_distances: list[float] = []
            zone_random_distances: list[float] = []
            for _ in range(bootstrap_samples):
                training_indices = model_rng.choice(
                    len(simulated), size=len(training), replace=False
                )
                held_out_indices = model_rng.choice(
                    len(simulated), size=len(held_out), replace=False
                )
                matched_to_held_out = _density(
                    simulated.iloc[training_indices],
                    ("x", "y"),
                    bandwidth,
                    environment,
                )
                matched_to_training = _density(
                    simulated.iloc[held_out_indices],
                    ("x", "y"),
                    bandwidth,
                    environment,
                )
                matched_distances.append(
                    0.5
                    * (
                        _jensen_shannon_distance(
                            matched_to_held_out, held_out_density
                        )
                        + _jensen_shannon_distance(
                            matched_to_training, training_density
                        )
                    )
                )
                global_to_held_out = _draw_global_random(
                    all_random_candidates, len(training), global_rng
                )
                global_to_training = _draw_global_random(
                    all_random_candidates, len(held_out), global_rng
                )
                global_random_distances.append(
                    0.5
                    * (
                        _jensen_shannon_distance(
                            _density(
                                global_to_held_out,
                                ("x", "y"),
                                bandwidth,
                                environment,
                            ),
                            held_out_density,
                        )
                        + _jensen_shannon_distance(
                            _density(
                                global_to_training,
                                ("x", "y"),
                                bandwidth,
                                environment,
                            ),
                            training_density,
                        )
                    )
                )
                zone_to_held_out = _draw_zone_preserving_random(
                    training,
                    all_random_candidates,
                    zone_candidate_pools,
                    zone_rng,
                )
                zone_to_training = _draw_zone_preserving_random(
                    held_out,
                    all_random_candidates,
                    zone_candidate_pools,
                    zone_rng,
                )
                zone_random_distances.append(
                    0.5
                    * (
                        _jensen_shannon_distance(
                            _density(
                                zone_to_held_out,
                                ("x", "y"),
                                bandwidth,
                                environment,
                            ),
                            held_out_density,
                        )
                        + _jensen_shannon_distance(
                            _density(
                                zone_to_training,
                                ("x", "y"),
                                bandwidth,
                                environment,
                            ),
                            training_density,
                        )
                    )
                )
            split_rows.append(
                {
                    "bandwidth_m": bandwidth,
                    "split": split_index,
                    "training_dates": ";".join(training_dates),
                    "comparison_direction": "average_of_both_held_out_directions",
                    "training_events": len(training),
                    "held_out_events": len(held_out),
                    "observed_half_distance": observed_distance,
                    "abm_pooled_distance": pooled_distance,
                    "abm_matched_median_distance": float(np.median(matched_distances)),
                    "abm_matched_low": float(np.quantile(matched_distances, 0.025)),
                    "abm_matched_high": float(np.quantile(matched_distances, 0.975)),
                    "global_random_median_distance": float(
                        np.median(global_random_distances)
                    ),
                    "global_random_low": float(
                        np.quantile(global_random_distances, 0.025)
                    ),
                    "global_random_high": float(
                        np.quantile(global_random_distances, 0.975)
                    ),
                    "zone_random_median_distance": float(
                        np.median(zone_random_distances)
                    ),
                    "zone_random_low": float(
                        np.quantile(zone_random_distances, 0.025)
                    ),
                    "zone_random_high": float(
                        np.quantile(zone_random_distances, 0.975)
                    ),
                }
            )

    splits = pd.DataFrame(split_rows)
    summary_rows: list[dict[str, object]] = []
    for bandwidth, group in splits.groupby("bandwidth_m", sort=True):
        differences = (
            group["abm_matched_median_distance"] - group["observed_half_distance"]
        )
        summary_rows.append(
            {
                "bandwidth_m": bandwidth,
                "observed_median_distance": group["observed_half_distance"].median(),
                "observed_min_distance": group["observed_half_distance"].min(),
                "observed_max_distance": group["observed_half_distance"].max(),
                "abm_matched_median_distance": group[
                    "abm_matched_median_distance"
                ].median(),
                "abm_matched_min_distance": group[
                    "abm_matched_median_distance"
                ].min(),
                "abm_matched_max_distance": group[
                    "abm_matched_median_distance"
                ].max(),
                "median_paired_difference": differences.median(),
                "splits_abm_as_close_or_closer": int((differences <= 0).sum()),
                "split_count": len(group),
                "global_random_median_distance": group[
                    "global_random_median_distance"
                ].median(),
                "zone_random_median_distance": group[
                    "zone_random_median_distance"
                ].median(),
            }
        )
    return splits, pd.DataFrame(summary_rows)


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.6, 3.9))
    fig.patch.set_facecolor(COLORS["paper"])
    axis.set_facecolor(COLORS["paper"])
    x = summary["bandwidth_m"].to_numpy(dtype=float)
    series = (
        (
            "Empirical to empirical",
            "observed",
            "observed_median_distance",
            "observed_min_distance",
            "observed_max_distance",
        ),
        (
            "Simulation to empirical",
            "simulated",
            "abm_matched_median_distance",
            "abm_matched_min_distance",
            "abm_matched_max_distance",
        ),
    )
    for label, color_key, median_key, low_key, high_key in series:
        median = summary[median_key].to_numpy(dtype=float)
        low = summary[low_key].to_numpy(dtype=float)
        high = summary[high_key].to_numpy(dtype=float)
        axis.fill_between(x, low, high, color=COLORS[color_key], alpha=0.14, linewidth=0)
        axis.plot(
            x,
            median,
            color=COLORS[color_key],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(BANDWIDTHS_M, ["0.5", "1", "2", "4"])
    axis.set_xlabel("KDE bandwidth (m)")
    axis.set_ylabel("Jensen–Shannon distance\n(lower is closer)")
    axis.grid(axis="y", color="#E7E7E3", linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(COLORS["line"])
    axis.spines["bottom"].set_color(COLORS["line"])
    axis.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(output_dir / f"part1_multiscale_kde.{extension}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observed",
        type=Path,
        default=(
            WORKSPACE_DIR
            / "source-materials"
            / "shadowing_with-participant-info.csv"
        ),
    )
    parser.add_argument(
        "--simulated",
        type=Path,
        default=PROJECT_DIR
        / "outputs"
        / "source_results"
        / "part1_validation_n100"
        / "normal_load"
        / "baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "findings" / "part1_multiscale_kde",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write the optional multiscale diagnostic plot.",
    )
    args = parser.parse_args()

    observed = pd.read_csv(args.observed)
    observed = observed.loc[
        observed["interfaceType"].fillna("").astype(str).str.contains("Person", case=False)
    ].copy()
    observed["date"] = observed["date"].astype(str)
    if len(observed) != 359:
        raise ValueError(f"Expected 359 observed events; found {len(observed)}")
    simulated = _load_simulated_points(args.simulated)
    environment = Environment(config.WALL_POSITIONS_PATH, config.ZONE_BOUNDARIES_PATH)
    splits, summary = _analyse(
        observed,
        simulated,
        environment,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output_dir / "multiscale_split_results.csv", index=False)
    summary.to_csv(args.output_dir / "multiscale_summary.csv", index=False)
    if args.plot:
        _plot(summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
