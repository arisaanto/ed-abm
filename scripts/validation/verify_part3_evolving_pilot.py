#!/usr/bin/env python3
"""Verify a full-factorial evolving-persona pilot or main study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run.run_part3_closed_loop_batch import _read_manifest, _run_dir
from scripts.validation.verify_part3_closed_loop_pilot import verify as verify_batch


DIMENSIONS = (
    "coordination_need",
    "interruption_strain",
    "task_continuity",
    "team_support",
)
EXPECTED_CHECKPOINT_COUNT = 5
EXPECTED_STATE_INTERVAL_SECONDS = 7200
EXPECTED_STATE_START_SECONDS = 7200
EXPECTED_AGENT_COUNT = 9


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def verify(
    output_root: Path,
    manifest_path: Path,
    *,
    study_design: str = "evolving_pilot",
    expected_seed_count: int = 1,
) -> dict[str, Any]:
    if study_design not in {"evolving_pilot", "main"}:
        raise ValueError(f"Unsupported study design: {study_design}")
    is_main = study_design == "main"
    base = verify_batch(
        output_root,
        manifest_path,
        study_design=study_design,
        expected_seed_count=expected_seed_count,
    )
    rows = _read_manifest(manifest_path)
    errors = list(base.get("errors", []))
    warnings = list(base.get("warnings", []))
    state_rows: list[dict[str, Any]] = []
    final_state_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    expected_state_rows = 0
    for manifest_row in rows:
        run_dir = _run_dir(output_root, manifest_row)
        summary = json.loads((run_dir / "summary.json").read_text())
        closed_loop = summary.get("part3_closed_loop", {})
        if (
            closed_loop.get("evolving_state_enabled") is not True
            or closed_loop.get("evolving_state_contract")
            != "evidence_linked_bounded_state_v1"
            or int(closed_loop.get("evolving_state_checkpoint_count", -1))
            != EXPECTED_CHECKPOINT_COUNT
            or int(closed_loop.get("evolving_state_interval_seconds", -1))
            != EXPECTED_STATE_INTERVAL_SECONDS
            or int(closed_loop.get("start_seconds", -1))
            != EXPECTED_STATE_START_SECONDS
        ):
            errors.append(f"Invalid evolving-state configuration in {run_dir}")
        run_state_rows = _jsonl(run_dir / "part3_evolving_state.jsonl")
        state_rows.extend(run_state_rows)
        expected_state_rows += EXPECTED_CHECKPOINT_COUNT * EXPECTED_AGENT_COUNT
        if run_state_rows:
            final_timestep = max(int(row["timestep"]) for row in run_state_rows)
            final_state_rows.extend(
                row
                for row in run_state_rows
                if int(row["timestep"]) == final_timestep
            )
        decision_rows.extend(_jsonl(run_dir / "part3_cognitive_decisions.jsonl"))

    if len(state_rows) != expected_state_rows:
        errors.append(
            f"Expected {expected_state_rows} agent-checkpoint states, found {len(state_rows)}"
        )
    model_state_rows = [row for row in state_rows if row.get("model_called") is True]
    state_model_coverage = (
        len(model_state_rows) / len(state_rows) if state_rows else 0.0
    )
    if state_model_coverage < 0.75:
        errors.append("Fewer than 75% of agent-checkpoints contained update evidence")

    values: dict[str, Counter[int]] = {name: Counter() for name in DIMENSIONS}
    final_values: dict[str, Counter[int]] = {name: Counter() for name in DIMENSIONS}
    deltas_by_persona: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for row in state_rows:
        state = row["state"]
        prior = row["prior_state"]
        for name in DIMENSIONS:
            values[name][int(state[name])] += 1
        if row.get("model_called") is True:
            deltas_by_persona[str(row["persona_id"])].add(
                tuple(int(state[name]) - int(prior[name]) for name in DIMENSIONS)
            )
    for row in final_state_rows:
        for name in DIMENSIONS:
            final_values[name][int(row["state"][name])] += 1
    collapsed_dimensions = [
        name for name in DIMENSIONS if len(values[name]) < 2
    ]
    if collapsed_dimensions:
        message = "State dimensions never changed: " + ", ".join(collapsed_dimensions)
        (warnings if is_main else errors).append(message)
    saturated_dimensions = {}
    for name, counts in final_values.items():
        total = sum(counts.values())
        boundary = counts[-2] + counts[2]
        share = boundary / total if total else 0.0
        if share > 0.75:
            saturated_dimensions[name] = share
    if saturated_dimensions:
        message = f"Final transient states saturated at boundaries: {saturated_dimensions}"
        (warnings if is_main else errors).append(message)
    prompt_locked_personas = [
        persona_id
        for persona_id, deltas in deltas_by_persona.items()
        if len(deltas) < 2
    ]
    if prompt_locked_personas:
        message = (
            "State transitions did not vary with evidence for personas: "
            + ", ".join(prompt_locked_personas)
        )
        (warnings if is_main else errors).append(message)

    model_decisions = [
        row for row in decision_rows if row.get("was_fallback") is not True
    ]
    action_counts = Counter(str(row["selected_action"]) for row in model_decisions)
    if len(action_counts) < 3:
        message = "Model decisions used fewer than three actions"
        (warnings if is_main else errors).append(message)
    if model_decisions and max(action_counts.values()) / len(model_decisions) > 0.95:
        message = "One decision action exceeded 95% of model calls"
        (warnings if is_main else errors).append(message)
    actions_by_persona: dict[str, Counter[str]] = defaultdict(Counter)
    for row in model_decisions:
        actions_by_persona[str(row["persona_id"])][str(row["selected_action"])] += 1
    single_action_personas = [
        persona_id for persona_id, counts in actions_by_persona.items() if len(counts) < 2
    ]
    if single_action_personas:
        message = (
            "Persona decisions used one action for: "
            + ", ".join(single_action_personas)
        )
        (warnings if is_main else errors).append(message)

    ablation_path = output_root / "state_ablation_summary.json"
    if not ablation_path.is_file():
        errors.append("Missing matched state-versus-neutral ablation")
        ablation = {}
    else:
        ablation = json.loads(ablation_path.read_text())
        sample_count = int(ablation.get("sample_count", 0))
        action_change_rate = float(ablation.get("action_change_rate", 0.0))
        if sample_count < 100:
            errors.append("State ablation contains fewer than 100 matched decisions")
        if not is_main and not 0.02 <= action_change_rate <= 0.50:
            errors.append(
                "State ablation action-change rate is outside the 2%-50% architecture gate"
            )
        incomplete_personas = [
            persona_id
            for persona_id, result in ablation.get("by_persona", {}).items()
            if int(result.get("sample_count", 0)) < 20
        ]
        if incomplete_personas:
            errors.append(
                "State ablation has fewer than 20 decisions for: "
                + ", ".join(incomplete_personas)
            )

    technical_pass = base.get("technical_integrity_pass") is True and not errors
    return {
        "verification_pass": technical_pass,
        "technical_integrity_pass": technical_pass,
        "scientific_result": is_main and technical_pass,
        "study_design": study_design,
        "expected_seed_count": expected_seed_count,
        "pilot_scope": None if is_main else "architecture_and_cost_gate_only",
        "base_integrity_pass": base.get("verification_pass") is True,
        "run_count": len(rows),
        "state_row_count": len(state_rows),
        "expected_state_row_count": expected_state_rows,
        "model_state_update_count": len(model_state_rows),
        "state_model_coverage": state_model_coverage,
        "state_value_counts": {
            name: {str(value): count for value, count in sorted(counts.items())}
            for name, counts in values.items()
        },
        "final_state_value_counts": {
            name: {str(value): count for value, count in sorted(counts.items())}
            for name, counts in final_values.items()
        },
        "state_delta_pattern_count_by_persona": {
            persona_id: len(deltas)
            for persona_id, deltas in sorted(deltas_by_persona.items())
        },
        "model_action_counts": dict(sorted(action_counts.items())),
        "model_action_counts_by_persona": {
            persona_id: dict(sorted(counts.items()))
            for persona_id, counts in sorted(actions_by_persona.items())
        },
        "state_ablation": ablation,
        "errors": errors,
        "warnings": warnings,
        "base_verification": base,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--study-design",
        choices=("evolving_pilot", "main"),
        default="evolving_pilot",
    )
    parser.add_argument("--expected-seed-count", type=int, default=1)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = verify(
        Path(args.output_root),
        Path(args.manifest),
        study_design=args.study_design,
        expected_seed_count=args.expected_seed_count,
    )
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verification_pass"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
