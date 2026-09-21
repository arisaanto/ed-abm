#!/usr/bin/env python3
"""Audit whether the all-pairs visibility percentage conceals local co-presence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from statistics import NormalDist, mean, stdev
import sys
from itertools import combinations
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


SETTINGS = {
    "restricted": {"max_distance": 12.0, "fov": 180.0, "interaction_radius": 8.0},
    "default": {"max_distance": 18.0, "fov": 210.0, "interaction_radius": 10.0},
    "expanded": {"max_distance": 24.0, "fov": 270.0, "interaction_radius": 12.0},
}


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    """Approximate the two-sided 95% Student-t critical value without SciPy."""

    if degrees_of_freedom <= 0:
        return math.nan
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def _paired_summary(values: list[float]) -> tuple[float, float, float]:
    estimate = mean(values)
    if len(values) < 2:
        return estimate, estimate, estimate
    half_width = (
        _student_t_critical_975(len(values) - 1)
        * stdev(values)
        / math.sqrt(len(values))
    )
    return estimate, estimate - half_width, estimate + half_width


def _write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Summarize co-presence, co-awareness, and contact by visibility setting."""

    by_cell: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
    for raw in rows:
        row = {key: float(value) if key not in {"setting", "condition"} else value for key, value in raw.items()}
        all_pair_samples = float(row["all_pair_samples"])
        row["near_pair_share"] = float(row["near_pair_samples"]) / max(all_pair_samples, 1.0)
        by_cell[(str(row["setting"]), int(row["seed"]))][str(row["condition"])] = row

    summary_rows: list[dict[str, object]] = []
    metrics = {
        "near_pair_share": "nearby_pair_time_share",
        "near_pairs_visible_share": "nearby_pairs_mutually_visible_share",
        "staff_time_with_any_visible_colleague_share": "staff_time_any_visible_colleague_share",
        "f2f_per_hour": "f2f_per_hour",
    }
    for setting in SETTINGS:
        cells = [
            value for (cell_setting, _), value in sorted(by_cell.items())
            if cell_setting == setting and {"baseline", "cockpit_only"} <= set(value)
        ]
        for source_metric, output_metric in metrics.items():
            baseline = [float(cell["baseline"][source_metric]) for cell in cells]
            cockpit = [float(cell["cockpit_only"][source_metric]) for cell in cells]
            deltas = [right - left for left, right in zip(baseline, cockpit)]
            estimate, low, high = _paired_summary(deltas)
            summary_rows.append(
                {
                    "setting": setting,
                    "metric": output_metric,
                    "seed_count": len(cells),
                    "baseline_mean": mean(baseline),
                    "cockpit_only_mean": mean(cockpit),
                    "paired_mean_difference": estimate,
                    "paired_ci95_low": low,
                    "paired_ci95_high": high,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "visibility_copresence_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "visibility_copresence_summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n"
    )


def _run_one(task: tuple[str, str, int, int, int, int]) -> dict[str, float | int | str]:
    setting_name, condition, seed, duration, warmup, sample_interval = task
    import config
    from src.simulation import Simulation

    setting = SETTINGS[setting_name]
    config.VISIBILITY_MAX_DISTANCE = setting["max_distance"]
    config.FOV_DEGREES = setting["fov"]
    config.PERCEPTION_INTERACTION_RADIUS_METERS = setting["interaction_radius"]
    simulation = Simulation(
        random_seed=seed,
        condition_name=condition,
        scenario_mode="normal_load",
        scenario_start_hour=7,
        part3_isolate_exogenous_arrival_stream=True,
    )

    totals = {
        "all_pair_samples": 0,
        "visible_pair_samples": 0,
        "near_pair_samples": 0,
        "visible_near_pair_samples": 0,
        "same_zone_pair_samples": 0,
        "visible_same_zone_pair_samples": 0,
        "central_pair_samples": 0,
        "visible_central_pair_samples": 0,
        "staff_samples": 0,
        "staff_samples_with_any_visible": 0,
    }
    for _ in range(duration):
        simulation.step()
        if simulation.timestep < warmup or simulation.timestep % sample_interval:
            continue
        visible_by_agent = {agent.gid: False for agent in simulation.staff_agents}
        for left, right in combinations(simulation.staff_agents, 2):
            visible = simulation.perception.can_agents_mutually_perceive(left, right, simulation)
            distance = math.dist(left.position, right.position)
            left_zone = simulation.which_zone(*left.position)
            right_zone = simulation.which_zone(*right.position)
            same_zone = left_zone is not None and left_zone == right_zone
            central = same_zone and left_zone in {"COCPIT", "CORR03", "NUROPE"}
            totals["all_pair_samples"] += 1
            totals["visible_pair_samples"] += int(visible)
            if distance <= setting["interaction_radius"]:
                totals["near_pair_samples"] += 1
                totals["visible_near_pair_samples"] += int(visible)
            if same_zone:
                totals["same_zone_pair_samples"] += 1
                totals["visible_same_zone_pair_samples"] += int(visible)
            if central:
                totals["central_pair_samples"] += 1
                totals["visible_central_pair_samples"] += int(visible)
            if visible:
                visible_by_agent[left.gid] = True
                visible_by_agent[right.gid] = True
        totals["staff_samples"] += len(visible_by_agent)
        totals["staff_samples_with_any_visible"] += sum(visible_by_agent.values())

    interactions = [
        event for event in simulation.interaction_log
        if int(event.get("timestamp", 0)) >= warmup
    ]
    hours = max((duration - warmup) / 3600.0, 1e-9)

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(totals[denominator], 1)

    return {
        "setting": setting_name,
        "condition": condition,
        "seed": seed,
        "max_distance_m": setting["max_distance"],
        "fov_degrees": setting["fov"],
        "interaction_radius_m": setting["interaction_radius"],
        "f2f_per_hour": len(interactions) / hours,
        "all_pairs_visible_share": ratio("visible_pair_samples", "all_pair_samples"),
        "near_pairs_visible_share": ratio("visible_near_pair_samples", "near_pair_samples"),
        "same_zone_pairs_visible_share": ratio("visible_same_zone_pair_samples", "same_zone_pair_samples"),
        "central_area_pairs_visible_share": ratio("visible_central_pair_samples", "central_pair_samples"),
        "staff_time_with_any_visible_colleague_share": ratio(
            "staff_samples_with_any_visible", "staff_samples"
        ),
        **totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration", type=int, default=43_200)
    parser.add_argument("--warmup", type=int, default=7_200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--summarize-csv",
        type=Path,
        help="Summarize an existing run-level CSV without rerunning simulations.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_DIR / "outputs" / "findings" / "visibility_face_validity",
    )
    args = parser.parse_args()
    if args.summarize_csv:
        with args.summarize_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        _write_summary(rows, args.output_dir)
        print(args.output_dir / "visibility_copresence_summary.csv")
        return
    tasks = [
        (setting, condition, seed, args.duration, args.warmup, args.sample_interval)
        for setting in SETTINGS
        for seed in range(1, args.seeds + 1)
        for condition in ("baseline", "cockpit_only")
    ]
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        rows = pool.map(_run_one, tasks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "visibility_face_validity_runs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(rows, args.output_dir)

    print(csv_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
