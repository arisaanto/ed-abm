#!/usr/bin/env python3
"""Build the preregistered paired Part 3 closed-loop main manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
ASSIGNMENT_ROUNDS = range(1, 6)
FIELDNAMES = (
    "run_index",
    "scenario",
    "condition",
    "seed",
    "assignment_round",
    "duration_seconds",
    "warmup_seconds",
    "max_model_decisions",
    "max_model_decisions_per_agent",
    "decision_window_seconds",
    "max_model_decisions_per_window",
    "model_decision_sample_rate",
)


def build_rows(seed_count: int) -> list[dict[str, int | float | str]]:
    if seed_count < 2:
        raise ValueError("The paired main study requires at least two seeds")
    rows: list[dict[str, int | float | str]] = []
    # Interleave all conditions within each seed/round/scenario. Independent
    # workers may finish in any order, but every scheduling wave receives a
    # mixture of spatial conditions rather than one condition at a time.
    for seed in range(1, seed_count + 1):
        for assignment_round in ASSIGNMENT_ROUNDS:
            for scenario in SCENARIOS:
                for condition in CONDITIONS:
                    rows.append(
                        {
                            "run_index": len(rows) + 1,
                            "scenario": scenario,
                            "condition": condition,
                            "seed": seed,
                            "assignment_round": assignment_round,
                            "duration_seconds": 43200,
                            "warmup_seconds": 7200,
                            "max_model_decisions": 300,
                            "max_model_decisions_per_agent": 100,
                            "decision_window_seconds": 7200,
                            "max_model_decisions_per_window": 100,
                            "model_decision_sample_rate": 0.10,
                        }
                    )
    expected = len(SCENARIOS) * len(CONDITIONS) * seed_count * len(ASSIGNMENT_ROUNDS)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} rows, built {len(rows)}")
    identities = {
        (row["scenario"], row["condition"], row["seed"], row["assignment_round"])
        for row in rows
    }
    if len(identities) != expected:
        raise AssertionError("Manifest contains duplicate run identities")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument(
        "--output",
        default="manifests/part3_closed_loop_main_n10.csv",
    )
    args = parser.parse_args()
    output = Path(args.output)
    rows = build_rows(args.seed_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} runs: {len(SCENARIOS)} scenarios x "
        f"{len(CONDITIONS)} conditions x {args.seed_count} paired seeds x "
        f"{len(ASSIGNMENT_ROUNDS)} persona rotations -> {output}"
    )


if __name__ == "__main__":
    main()
