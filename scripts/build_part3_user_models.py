#!/usr/bin/env python3
"""Validate and export the preregistered Part 3 cognitive personas.

This script does not read run summaries, infer personality, cluster agents, or
call an LLM. It defines the experimental factors and a balanced role-crossed
assignment schedule before outcomes are observed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.personas import (
    OCEAN_DIMENSIONS,
    WORKPLACE_PRIOR_DIMENSIONS,
    balanced_cognitive_persona_assignments,
    default_cognitive_personas,
    write_cognitive_persona_outputs,
)


DEFAULT_STAFF_ROSTER = {
    1: "CoordinationNurse",
    10: "Nurse",
    11: "Nurse",
    12: "Nurse",
    13: "Nurse",
    14: "Nurse",
    20: "Doctor",
    21: "Doctor",
    22: "Doctor",
}


def _parse_staff_ids(value: str) -> list[int]:
    staff_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not staff_ids:
        raise argparse.ArgumentTypeError("at least one comma-separated staff id is required")
    return staff_ids


def _assignment_rows(staff_ids: list[int], blocks: int) -> list[dict[str, int | str]]:
    rows = balanced_cognitive_persona_assignments(staff_ids, blocks=blocks)
    for row in rows:
        row["role"] = DEFAULT_STAFF_ROSTER.get(int(row["staff_id"]), "Unknown")
    return rows


def validate_protocol(staff_ids: list[int], blocks: int) -> dict[str, int]:
    personas = default_cognitive_personas()
    persona_ids = {persona.persona_id for persona in personas}
    if len(personas) != 5 or len(persona_ids) != 5:
        raise AssertionError("Part 3 requires five unique cognitive personas")
    for persona in personas:
        if set(persona.ocean_display) != set(OCEAN_DIMENSIONS):
            raise AssertionError(f"Incomplete OCEAN display crosswalk: {persona.persona_id}")
        if set(persona.workplace_priors) != set(WORKPLACE_PRIOR_DIMENSIONS):
            raise AssertionError(f"Incomplete workplace profile: {persona.persona_id}")

    rows = _assignment_rows(staff_ids, blocks)
    by_staff: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_staff[int(row["staff_id"])][str(row["persona_id"])] += 1
    expected = Counter({persona_id: blocks for persona_id in persona_ids})
    unbalanced = {staff_id: counts for staff_id, counts in by_staff.items() if counts != expected}
    if unbalanced:
        raise AssertionError(f"Persona rotation is not balanced within staff id: {unbalanced}")
    return {
        "persona_count": len(personas),
        "staff_count": len(staff_ids),
        "blocks": blocks,
        "assignment_rounds": blocks * len(personas),
        "assignment_rows": len(rows),
    }


def write_protocol(out_dir: Path, staff_ids: list[int], blocks: int) -> list[Path]:
    summary = validate_protocol(staff_ids, blocks)
    out_dir.mkdir(parents=True, exist_ok=True)
    persona_path = out_dir / "cognitive_personas.json"
    cards_path = out_dir / "cognitive_persona_cards.md"
    assignments_path = out_dir / "balanced_persona_assignments.csv"
    write_cognitive_persona_outputs(persona_path, cards_path)

    rows = _assignment_rows(staff_ids, blocks)
    fields = ["block", "assignment_round", "staff_id", "role", "persona_id"]
    with assignments_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    return [persona_path, cards_path, assignments_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/part3/protocol")
    parser.add_argument(
        "--staff-ids",
        type=_parse_staff_ids,
        default=list(DEFAULT_STAFF_ROSTER),
        help="Comma-separated ABM staff ids; defaults to the validated nine-person roster",
    )
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--check-only", action="store_true", help="Validate without writing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_protocol(args.staff_ids, args.blocks)
    if args.check_only:
        print(json.dumps(summary, indent=2))
        return
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    for path in write_protocol(out_dir, args.staff_ids, args.blocks):
        print(path)


if __name__ == "__main__":
    main()
