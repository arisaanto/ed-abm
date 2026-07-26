#!/usr/bin/env python3
"""Build a matched 2 x 2 Part 3 architecture-ablation packet set.

This is a CPU-only design/preflight step. It reuses model-called opportunities
from the accepted closed-loop main study and never runs the ABM or an LLM. All
four factorial cells are prepared for one contemporaneous inference run. The
already observed full-persona/full-memory response is retained only as an
external reproducibility reference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.build.plan_part3_synthetic_study import scientific_decision_packet
from src.personas import CognitivePersona, cognitive_persona_by_id
from src.vllm_backend import scientific_packet_prompt_leakage


SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
PERSONAS = (
    "adaptive_generalist",
    "focus_protector",
    "patient_advocate",
    "team_connector",
    "vigilant_monitor",
)
ROLES = ("CoordinationNurse", "Nurse", "Doctor")
VARIANTS = (
    "full_persona_full_memory",
    "neutral_orientation",
    "persona_no_memory",
    "neutral_no_memory",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _digest(*values: Any) -> str:
    payload = "|".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _neutral_orientation() -> CognitivePersona:
    return CognitivePersona(
        persona_id="neutral_orientation",
        display_name="Neutral orientation",
        archetype_title="Neutral orientation",
        summary=(
            "Uses the supplied operational evidence without a dominant workplace "
            "priority or persona-specific interaction preference."
        ),
        workplace_priors={
            "coordination_orientation": 0.5,
            "focus_protection": 0.5,
            "patient_advocacy": 0.5,
            "vigilance": 0.5,
            "adaptability": 0.5,
            "privacy_control_preference": 0.5,
            "interaction_initiative": 0.5,
        },
        decision_principles=[
            "Use urgency, relevance, patient linkage, visibility, pressure, and diversion cost without privileging one orientation.",
            "Engage, defer, or decline only when the supplied evidence supports that choice.",
        ],
        topic_family_preferences=[],
        memory_salience_modifiers={},
        reflection_voice="Neutral and evidence-bounded.",
        source_basis=["Matched generic-LLM architecture ablation"],
    )


def _source_records(results_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(results_root.rglob("part3_provider_inference.jsonl")):
        for row in _read_jsonl(path):
            packet = row.get("packet", {})
            trace = packet.get("trace_evidence", {})
            memory = trace.get("memory_state_before") or []
            if (
                packet.get("packet_type") == "in_simulation_decision"
                and packet.get("fixture_only_not_scientific_data") is False
                and memory
            ):
                records.append({**row, "source_path": str(path), "memory_count": len(memory)})
    if not records:
        raise ValueError("No memory-bearing main-study provider packets found")
    return records


def _select_balanced(
    records: list[dict[str, Any]], per_stratum: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if per_stratum != 1:
        raise ValueError("The preregistered architecture ablation uses one source per stratum")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        packet = record["packet"]
        metadata = packet["trace_evidence"]["metadata"]
        key = (
            str(metadata["scenario_mode"]),
            str(metadata["condition"]),
            str(packet["user_model_profile"]["persona_id"]),
            str(metadata["role"]),
        )
        grouped[key].append(record)

    expected = {
        (scenario, condition, persona, role)
        for scenario in SCENARIOS
        for condition in CONDITIONS
        for persona in PERSONAS
        for role in ROLES
    }
    missing = sorted(expected - set(grouped))
    short = [
        {"stratum": list(key), "available": len(grouped[key])}
        for key in sorted(expected & set(grouped))
        if len(grouped[key]) < per_stratum
    ]
    if missing or short:
        raise ValueError(
            f"Ablation source coverage incomplete: missing={missing}, short={short}"
        )

    selected = []
    seed_load: Counter[str] = Counter()
    ordered_strata = sorted(
        expected,
        key=lambda key: _digest(*key, "ablation_stratum_order_v2"),
    )
    for key in ordered_strata:
        candidates_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in grouped[key]:
            seed = str(row["packet"]["trace_evidence"]["metadata"]["seed"])
            candidates_by_seed[seed].append(row)
        seed = min(
            candidates_by_seed,
            key=lambda value: (
                seed_load[value],
                _digest(*key, value, "ablation_seed_balance_v2"),
            ),
        )
        candidates = sorted(
            candidates_by_seed[seed],
            key=lambda row: _digest(
                row["packet"]["prompt_id"], row["packet_sha256"], "ablation_v2"
            ),
        )
        selected.append(candidates[0])
        seed_load[seed] += 1
    return selected, [
        {"stratum": list(key), "available": len(grouped[key])}
        for key in sorted(expected)
    ]


def _variant_packet(
    source: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    source_packet = source["packet"]
    source_trace = deepcopy(source_packet["trace_evidence"])
    source_persona_id = str(source_packet["user_model_profile"]["persona_id"])
    original_memory_count = len(source_trace.get("memory_state_before") or [])
    if variant in {"neutral_orientation", "neutral_no_memory"}:
        persona = _neutral_orientation()
    elif variant in {"full_persona_full_memory", "persona_no_memory"}:
        persona = cognitive_persona_by_id()[source_persona_id]
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    if variant in {"persona_no_memory", "neutral_no_memory"}:
        source_trace["memory_state_before"] = []

    packet = scientific_decision_packet(persona, source_trace)
    pair_id = _digest(source_packet["prompt_id"], source["packet_sha256"])[:24]
    packet["prompt_id"] = f"ablation-{pair_id}-{variant.replace('_', '-')}"
    packet["ablation_variant"] = variant
    packet["ablation_pair_id"] = pair_id
    packet["ablation_not_main_result"] = True
    packet["ablation_reference"] = {
        "source_prompt_id": source_packet["prompt_id"],
        "source_packet_sha256": source["packet_sha256"],
        "source_persona_id": source_persona_id,
        "source_memory_count": original_memory_count,
        "observed_full_persona_memory_response": source["result"]["response"],
        "model": source.get("model"),
        "model_revision": source.get("model_revision"),
        "thinking_enabled": source.get("thinking_enabled"),
    }
    leakage = scientific_packet_prompt_leakage(packet)
    if leakage:
        raise ValueError(f"Evaluator-only information leaked into ablation prompt: {leakage}")
    return packet


def build(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_records = _source_records(results_root)
    selected, source_coverage = _select_balanced(
        source_records, args.replicates_per_stratum
    )
    packets = [
        _variant_packet(source, variant)
        for source in selected
        for variant in VARIANTS
    ]

    prompt_path = output_dir / "prompt_packets" / "part3_architecture_ablation_packets.jsonl"
    inventory_path = output_dir / "part3_architecture_ablation_inventory.csv"
    preflight_path = output_dir / "part3_architecture_ablation_preflight.json"
    _write_jsonl(prompt_path, packets)

    inventory = []
    for packet in packets:
        metadata = packet["trace_evidence"]["metadata"]
        reference = packet["ablation_reference"]
        inventory.append(
            {
                "prompt_id": packet["prompt_id"],
                "ablation_pair_id": packet["ablation_pair_id"],
                "variant": packet["ablation_variant"],
                "scenario": metadata["scenario_mode"],
                "condition": metadata["condition"],
                "source_persona_id": reference["source_persona_id"],
                "prompt_persona_id": packet["user_model_profile"]["persona_id"],
                "role": metadata["role"],
                "seed": metadata["seed"],
                "source_memory_count": reference["source_memory_count"],
                "visible_memory_count": len(
                    packet["llm_visible_evidence"].get("memory_state_before") or []
                ),
                "reference_action": reference[
                    "observed_full_persona_memory_response"
                ]["selected_action"],
                "source_packet_sha256": reference["source_packet_sha256"],
            }
        )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    pair_counts = Counter(row["ablation_pair_id"] for row in inventory)
    variant_counts = Counter(row["variant"] for row in inventory)
    expected_sources = (
        len(SCENARIOS)
        * len(CONDITIONS)
        * len(PERSONAS)
        * len(ROLES)
        * args.replicates_per_stratum
    )
    errors = []
    source_models = Counter(str(row.get("model")) for row in selected)
    source_revisions = Counter(str(row.get("model_revision")) for row in selected)
    source_thinking_modes = Counter(bool(row.get("thinking_enabled")) for row in selected)
    source_seed_counts = Counter(
        str(row["packet"]["trace_evidence"]["metadata"]["seed"])
        for row in selected
    )
    source_reference_action_counts = Counter(
        str(row["result"]["response"]["selected_action"])
        for row in selected
    )
    if len(selected) != expected_sources:
        errors.append(f"Expected {expected_sources} source packets, found {len(selected)}")
    if any(count != len(VARIANTS) for count in pair_counts.values()):
        errors.append("At least one ablation set does not contain all four generated variants")
    if variant_counts != Counter({variant: expected_sources for variant in VARIANTS}):
        errors.append(f"Variant counts are unbalanced: {dict(variant_counts)}")
    if any(
        row["variant"] in {"persona_no_memory", "neutral_no_memory"}
        and row["visible_memory_count"] != 0
        for row in inventory
    ):
        errors.append("A no-memory packet still exposes retrieved memory")
    if any(
        row["variant"] in {"full_persona_full_memory", "neutral_orientation"}
        and row["visible_memory_count"] != row["source_memory_count"]
        for row in inventory
    ):
        errors.append("A full-memory packet does not expose the complete retrieved memory")
    if any(
        row["variant"] in {"neutral_orientation", "neutral_no_memory"}
        and row["prompt_persona_id"] != "neutral_orientation"
        for row in inventory
    ):
        errors.append("A neutral-orientation packet retains a persona prompt")
    if len(source_models) != 1 or "Qwen/Qwen3.6-35B-A3B" not in source_models:
        errors.append(f"Selected sources do not share the expected model: {dict(source_models)}")
    if len(source_revisions) != 1 or "None" in source_revisions:
        errors.append(
            f"Selected sources do not share one pinned model revision: {dict(source_revisions)}"
        )
    if source_thinking_modes != Counter({False: expected_sources}):
        errors.append(
            f"Thinking must be disabled in every selected source: {dict(source_thinking_modes)}"
        )
    if any(packet.get("estimated_tokens", 0) <= 0 for packet in packets):
        errors.append("At least one ablation packet has no valid token estimate")
    if len(source_seed_counts) != 10:
        errors.append(f"Selected sources do not cover all ten seeds: {dict(source_seed_counts)}")
    expected_per_seed = expected_sources // 10
    if expected_sources % 10 or set(source_seed_counts.values()) != {expected_per_seed}:
        errors.append(
            "Selected sources are not exactly balanced across seeds: "
            f"{dict(source_seed_counts)}"
        )
    if set(source_reference_action_counts) != {"engage", "defer", "decline"}:
        errors.append(
            "Selected sources do not cover all three observed action categories: "
            f"{dict(source_reference_action_counts)}"
        )

    preflight = {
        "preflight_pass": not errors,
        "abm_rerun": False,
        "llm_run": False,
        "scientific_result": False,
        "design_status": "prepared_not_executed",
        "source_results_root": str(results_root),
        "source_record_count": len(source_records),
        "selected_source_count": len(selected),
        "packet_count": len(packets),
        "replicates_per_stratum": args.replicates_per_stratum,
        "variant_counts": dict(sorted(variant_counts.items())),
        "expected_source_stratum_count": len(source_coverage),
        "source_coverage": source_coverage,
        "source_memory_minimum": min(row["memory_count"] for row in selected),
        "source_memory_maximum": max(row["memory_count"] for row in selected),
        "source_model_counts": dict(source_models),
        "source_model_revision_counts": dict(source_revisions),
        "source_thinking_enabled_counts": {
            str(key).lower(): value for key, value in source_thinking_modes.items()
        },
        "source_seed_counts": dict(sorted(source_seed_counts.items())),
        "expected_source_count_per_seed": expected_per_seed,
        "source_reference_action_counts": dict(
            sorted(source_reference_action_counts.items())
        ),
        "maximum_estimated_tokens": max(packet["estimated_tokens"] for packet in packets),
        "reference_policy": (
            "all four factorial cells generated contemporaneously; the observed "
            "main-study full-persona/full-memory response is a reproducibility reference"
        ),
        "factorial_design": {
            "orientation_factor": ["full_persona", "neutral_orientation"],
            "memory_factor": ["full_memory", "no_memory"],
            "historical_reference_cell": "full_persona|full_memory",
            "generated_cells": [
                "full_persona|full_memory",
                "neutral_orientation|full_memory",
                "full_persona|no_memory",
                "neutral_orientation|no_memory",
            ],
            "complete_two_by_two_design": True,
        },
        "neutral_orientation_definition": (
            "All workplace priors fixed at 0.5; no dominant topic preference; "
            "grounded memory retained."
        ),
        "no_memory_definition": (
            "Original persona retained; model-visible retrieved memory removed; "
            "all other source evidence retained."
        ),
        "neutral_no_memory_definition": (
            "Neutral workplace orientation and no model-visible retrieved memory; "
            "all other source evidence retained."
        ),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "errors": errors,
    }
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
    return preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates-per-stratum", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(json.dumps(result, indent=2))
    if not result["preflight_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
