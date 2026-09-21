#!/usr/bin/env python3
"""Summarize the paired Part 2 analysis-window sensitivity audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCENARIO_LABELS = {
    "normal_load": "Normal load",
    "high_load_high_acuity": "High load",
}
CONDITIONS = ("baseline", "cockpit_only")
WARMUPS = (0, 3600, 7200, 10800)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return percentile_interval(draws)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/part2_warmup_audit_n30"))
    args = parser.parse_args()

    records: dict[tuple[str, int, str, int], dict] = {}
    record_paths: dict[tuple[str, int, str, int], Path] = {}
    for path in sorted(args.results.rglob("summary.json")):
        payload = json.loads(path.read_text())
        meta = payload["metadata"]
        key = (
            str(meta["scenario_mode"]),
            int(meta["warmup_seconds"]),
            str(meta["condition"]),
            int(meta["seed"]),
        )
        if key in records:
            raise SystemExit(f"Duplicate run key: {key}")
        records[key] = payload
        record_paths[key] = path

    expected = {
        (scenario, warmup, condition, seed)
        for scenario in SCENARIO_LABELS
        for warmup in WARMUPS
        for condition in CONDITIONS
        for seed in range(1, 31)
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise SystemExit(f"Incomplete audit: {len(records)} runs; missing={missing[:3]}; extra={extra[:3]}")

    for key, payload in records.items():
        meta = payload["metadata"]
        gates = payload["hard_gate_diagnostics"]
        if meta.get("part3_exogenous_arrival_stream_isolated") is not True:
            raise SystemExit(f"Arrival stream was not isolated for {key}")
        if gates.get("workflow_health_status") != "PASS":
            raise SystemExit(f"Workflow gate failed for {key}")
        for field in (
            "route_failure_count",
            "nonproximate_logged_interaction_count",
            "validation_counted_interactions_beyond_close_threshold_count",
        ):
            if int(gates.get(field, 0)) != 0:
                raise SystemExit(f"Hard gate {field} failed for {key}")

    # Within a cut-off, Baseline and COCPIT must receive the same exogenous
    # arrival attempts. Counts legitimately fall at later cut-offs because the
    # summary reports the evaluated window only.
    for scenario in SCENARIO_LABELS:
        for warmup in WARMUPS:
            for seed in range(1, 31):
                attempts = {
                    int(records[(scenario, warmup, condition, seed)]["patient_flow_metrics"]["arrival_attempts"])
                    for condition in CONDITIONS
                }
                if len(attempts) != 1:
                    raise SystemExit(
                        "Arrival attempts differ within paired "
                        f"scenario={scenario}, warmup={warmup}, seed={seed}: {attempts}"
                    )

    # Changing only the analysis cut-off must not alter the simulated trace.
    # Each later interaction file should exactly equal the 0-hour trace after
    # filtering on event time (apart from the derived time-after-warm-up field).
    for scenario in SCENARIO_LABELS:
        for condition in CONDITIONS:
            for seed in range(1, 31):
                reference_key = (scenario, 0, condition, seed)
                with (record_paths[reference_key].parent / "interaction_events.csv").open(newline="") as handle:
                    reference = list(csv.DictReader(handle))
                for warmup in WARMUPS[1:]:
                    key = (scenario, warmup, condition, seed)
                    with (record_paths[key].parent / "interaction_events.csv").open(newline="") as handle:
                        observed = list(csv.DictReader(handle))
                    expected_rows = [row for row in reference if float(row["time_seconds"]) >= warmup]
                    for rows_to_clean in (expected_rows, observed):
                        for row in rows_to_clean:
                            row.pop("time_after_warmup_seconds", None)
                    if observed != expected_rows:
                        raise SystemExit(f"Trace changed when only warm-up changed for {key}")

    rng = np.random.default_rng(26000639)
    rows: list[dict[str, object]] = []
    occupancy_rows: list[dict[str, object]] = []
    for scenario in SCENARIO_LABELS:
        for warmup in WARMUPS:
            baseline_rates = []
            cockpit_rates = []
            baseline_counts = []
            cockpit_counts = []
            baseline_occupancy = []
            baseline_queue = []
            for seed in range(1, 31):
                baseline = records[(scenario, warmup, "baseline", seed)]
                cockpit = records[(scenario, warmup, "cockpit_only", seed)]
                baseline_rates.append(float(baseline["validation_metrics"]["f2f_per_hour"]))
                cockpit_rates.append(float(cockpit["validation_metrics"]["f2f_per_hour"]))
                baseline_counts.append(float(baseline["validation_metrics"]["interaction_count"]))
                cockpit_counts.append(float(cockpit["validation_metrics"]["interaction_count"]))
                baseline_occupancy.append(
                    float(baseline["patient_flow_metrics"]["average_ordinary_bed_occupancy"])
                )
                baseline_queue.append(float(baseline["patient_flow_metrics"]["mean_waiting_queue_length"]))

            baseline_array = np.asarray(baseline_rates)
            cockpit_array = np.asarray(cockpit_rates)
            effects = cockpit_array - baseline_array
            lo, hi = paired_bootstrap(effects, rng)
            baseline_mean = float(baseline_array.mean())
            effect_mean = float(effects.mean())
            rows.append(
                {
                    "scenario": SCENARIO_LABELS[scenario],
                    "warmup_hours": warmup // 3600,
                    "evaluation_hours": (43200 - warmup) / 3600,
                    "baseline_f2f_per_hour": baseline_mean,
                    "cockpit_f2f_per_hour": float(cockpit_array.mean()),
                    "paired_effect_f2f_per_hour": effect_mean,
                    "paired_effect_ci_low": lo,
                    "paired_effect_ci_high": hi,
                    "relative_effect_percent": 100.0 * effect_mean / baseline_mean,
                    "mean_baseline_count": float(np.mean(baseline_counts)),
                    "mean_cockpit_count": float(np.mean(cockpit_counts)),
                }
            )
            occupancy_rows.append(
                {
                    "scenario": SCENARIO_LABELS[scenario],
                    "warmup_hours": warmup // 3600,
                    "baseline_mean_ordinary_bed_occupancy": float(np.mean(baseline_occupancy)),
                    "baseline_mean_waiting_queue": float(np.mean(baseline_queue)),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "warmup_effect_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "warmup_state_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(occupancy_rows[0]))
        writer.writeheader()
        writer.writerows(occupancy_rows)

    print(f"Verified and summarized {len(records)} runs in {args.output}")


if __name__ == "__main__":
    main()
