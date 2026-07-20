#!/usr/bin/env python3
"""Verify integrity and paired completeness of Part 2 spatial-intervention runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


EXPECTED_SCENARIOS = ("normal_load", "high_load_high_acuity")
EXPECTED_CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
HARD_ZERO_KEYS = (
    "route_failures",
    "route_failure_count",
    "stuck_agent_events",
    "transit_stuck_repair_events",
    "terminal_stuck_agent_events",
    "unsupported_assignment_rejection_count",
    "route_movement_issue_count",
    "oscillation_warnings",
    "station_check_route_failures",
    "nonproximate_logged_interaction_count",
    "synthetic_or_relocated_interaction_coordinate_count",
    "midpoint_logged_validation_interaction_count",
    "validation_counted_interactions_beyond_close_threshold_count",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary_files(batch_dir: Path) -> list[Path]:
    return sorted(batch_dir.rglob("summary.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metadata(summary: dict[str, Any]) -> tuple[str, str, int]:
    metadata = summary.get("metadata", {})
    return (
        str(metadata.get("scenario_mode", "")),
        str(metadata.get("condition", "")),
        _safe_int(metadata.get("seed", -1), -1),
    )


def _hard_gate_value(summary: dict[str, Any], key: str) -> float:
    hard = summary.get("hard_gate_diagnostics", {})
    workflow = summary.get("workflow_health", {})
    if key in hard:
        return _safe_float(hard.get(key))
    return _safe_float(workflow.get(key))


def _expectation(mode: str) -> tuple[tuple[int, ...], int]:
    if mode == "smoke":
        seeds = tuple(range(1, 4))
    elif mode == "full":
        seeds = tuple(range(1, 101))
    else:
        raise ValueError(f"Unknown expectation mode: {mode}")
    return seeds, len(EXPECTED_SCENARIOS) * len(EXPECTED_CONDITIONS) * len(seeds)


def verify(batch_dir: Path, expected: str) -> dict[str, Any]:
    seeds, expected_total = _expectation(expected)
    paths = _summary_files(batch_dir)
    summaries = [_load(path) for path in paths]
    observed_keys = [_metadata(summary) for summary in summaries]
    observed_set = set(observed_keys)

    expected_set = {
        (scenario, condition, seed)
        for scenario in EXPECTED_SCENARIOS
        for condition in EXPECTED_CONDITIONS
        for seed in seeds
    }
    duplicate_counts = Counter(observed_keys)
    duplicates = [
        {"scenario": key[0], "condition": key[1], "seed": key[2], "count": count}
        for key, count in sorted(duplicate_counts.items())
        if count > 1
    ]
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)

    paired_missing = []
    for scenario in EXPECTED_SCENARIOS:
        for seed in seeds:
            conditions = {
                condition
                for observed_scenario, condition, observed_seed in observed_set
                if observed_scenario == scenario and observed_seed == seed
            }
            missing_conditions = sorted(set(EXPECTED_CONDITIONS) - conditions)
            if missing_conditions:
                paired_missing.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "missing_conditions": missing_conditions,
                    }
                )

    workflow_counts = Counter(
        str(summary.get("hard_gate_diagnostics", {}).get(
            "workflow_health_status",
            summary.get("workflow_health", {}).get("workflow_health_status", "UNKNOWN"),
        ))
        for summary in summaries
    )
    hard_gate_sums = {
        key: sum(_hard_gate_value(summary, key) for summary in summaries)
        for key in HARD_ZERO_KEYS
    }
    repair_sums = {
        key: sum(_hard_gate_value(summary, key) for summary in summaries)
        for key in (
            "assignment_repair_events",
            "handoff_wait_repair_events",
            "transit_stuck_repair_events",
            "terminal_stuck_agent_events",
            "unsupported_assignment_rejection_count",
            "route_movement_issue_count",
            "stuck_agent_events",
        )
    }
    missing_key_metrics = defaultdict(int)
    for summary in summaries:
        for key in ("f2f_per_hour", "interaction_count", "composite_distance"):
            if key not in summary.get("validation_metrics", {}):
                missing_key_metrics[key] += 1

    failed = bool(
        len(paths) != expected_total
        or missing
        or unexpected
        or duplicates
        or paired_missing
        or workflow_counts.get("PASS", 0) != len(summaries)
        or any(value != 0 for value in hard_gate_sums.values())
        or missing_key_metrics
    )
    return {
        "batch_dir": str(batch_dir),
        "expected_mode": expected,
        "expected_summary_count": expected_total,
        "observed_summary_count": len(paths),
        "expected_scenarios": list(EXPECTED_SCENARIOS),
        "observed_scenarios": sorted({key[0] for key in observed_set}),
        "expected_conditions": list(EXPECTED_CONDITIONS),
        "observed_conditions": sorted({key[1] for key in observed_set}),
        "expected_seed_min": min(seeds),
        "expected_seed_max": max(seeds),
        "observed_seed_min": min((key[2] for key in observed_set), default=None),
        "observed_seed_max": max((key[2] for key in observed_set), default=None),
        "missing_runs": [
            {"scenario": scenario, "condition": condition, "seed": seed}
            for scenario, condition, seed in missing
        ],
        "unexpected_runs": [
            {"scenario": scenario, "condition": condition, "seed": seed}
            for scenario, condition, seed in unexpected
        ],
        "duplicate_runs": duplicates,
        "paired_missing": paired_missing,
        "workflow_status_counts": dict(workflow_counts),
        "hard_gate_sums": hard_gate_sums,
        "repair_event_sums": repair_sums,
        "missing_key_metrics": dict(missing_key_metrics),
        "integrity_pass": not failed,
        "message": (
            "Part 2 batch integrity checks passed."
            if not failed
            else "Do not interpret Part 2 effects until integrity failures are resolved."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--expected", choices=("smoke", "full"), default="full")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    result = verify(Path(args.batch_dir), args.expected)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)
    raise SystemExit(0 if result["integrity_pass"] else 1)


if __name__ == "__main__":
    main()
