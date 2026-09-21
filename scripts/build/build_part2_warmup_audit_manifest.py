#!/usr/bin/env python3
"""Build a paired Part 2 manifest for nearby warm-up cut-offs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only")
WARMUPS = (0, 3_600, 7_200, 10_800)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="manifests/part2_warmup_audit_n30.csv"
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scenario_mode",
        "condition",
        "seed",
        "duration",
        "warmup_seconds",
        "scenario_start_hour",
        "batch_name",
        "run_id",
        "output_dir",
        "validation_target",
        "isolate_exogenous_arrival_stream",
    )
    rows = []
    for scenario in SCENARIOS:
        scenario_label = "normal_load" if scenario == "normal_load" else "high_load"
        for warmup in WARMUPS:
            warmup_label = f"warmup_{warmup // 3600}h"
            for condition in CONDITIONS:
                for seed in range(1, 31):
                    rows.append(
                        {
                            "scenario_mode": scenario,
                            "condition": condition,
                            "seed": seed,
                            "duration": 43_200,
                            "warmup_seconds": warmup,
                            "scenario_start_hour": 10,
                            "batch_name": "part2_warmup_audit_n30",
                            "run_id": f"{scenario_label}_{warmup_label}_{condition}_seed_{seed}",
                            "output_dir": (
                                "outputs/batch/part2_warmup_audit_n30/"
                                f"{scenario_label}/{warmup_label}/{condition}/seed_{seed}"
                            ),
                            "validation_target": "care_area",
                            "isolate_exogenous_arrival_stream": True,
                        }
                    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} runs")


if __name__ == "__main__":
    main()
