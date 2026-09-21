#!/usr/bin/env python3
"""Build a crossed common-situation audit from accepted Part 3 decisions.

The accepted closed-loop experiment is left unchanged. This CPU-only builder
selects one source decision context for every workload x layout x source
persona x staff-role stratum, then presents each frozen context to all five
personas. Within a context, the opportunity, retrieved contacts, and evolving
experience values are identical; only the persona policy changes.
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
from src.personas import cognitive_persona_by_id
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
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    return packet["trace_evidence"]["metadata"]


def _context_features(record: Mapping[str, Any]) -> tuple[str, ...]:
    packet = record["packet"]
    visible = packet["llm_visible_evidence"]
    contact = visible["contact_opportunity"]
    trace = packet["trace_evidence"]
    return (
        f"seed={_metadata(packet)['seed']}",
        f"reason={trace['decision_context'].get('abm_reason_type', 'unknown')}",
        f"diversion={contact.get('diversion_cost_band', 'unknown')}",
        f"urgency={contact.get('urgency_band', 'unknown')}",
        f"patient={bool(contact.get('patient_context_present'))}",
        f"memory={bool(visible.get('memory_state_before'))}",
        f"task={bool(contact.get('active_task_present'))}",
    )


def _source_records(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("part3_provider_inference.jsonl")):
        for row in _read_jsonl(path):
            packet = row.get("packet", {})
            response = row.get("result", {}).get("response", {})
            if (
                packet.get("packet_type") == "in_simulation_decision"
                and packet.get("fixture_only_not_scientific_data") is False
                and packet.get("decision_output_contract") == "categorical_causal_v1"
                and response.get("selected_action") in {"engage", "defer", "decline"}
                and row.get("model") == MODEL_ID
                and row.get("model_revision") == MODEL_REVISION
                and row.get("thinking_enabled") is False
            ):
                records.append({**row, "source_path": str(path)})
    if not records:
        raise ValueError("No accepted scientific decision packets found")
    return records


def _select_crossed_contexts(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        packet = record["packet"]
        metadata = _metadata(packet)
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
    if missing:
        raise ValueError(f"Missing source strata: {missing}")

    feature_load: Counter[str] = Counter()
    seed_load: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for key in sorted(expected, key=lambda value: _digest(*value, "common-audit-order-v1")):
        candidates = grouped[key]
        ranked = sorted(
            candidates,
            key=lambda row: (
                sum(feature_load[value] for value in _context_features(row)),
                seed_load[_context_features(row)[0]],
                _digest(row["packet"]["prompt_id"], "common-audit-choice-v1"),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        for feature in _context_features(chosen):
            feature_load[feature] += 1
        seed_load[_context_features(chosen)[0]] += 1
        coverage.append({"stratum": list(key), "available": len(candidates)})
    return selected, coverage


def _crossed_packet(source: Mapping[str, Any], target_persona_id: str) -> dict[str, Any]:
    source_packet = source["packet"]
    source_trace = deepcopy(source_packet["trace_evidence"])
    target_persona = cognitive_persona_by_id()[target_persona_id]
    packet = scientific_decision_packet(target_persona, source_trace)
    source_hash = str(source["packet_sha256"])
    pair_id = _digest(source_packet["prompt_id"], source_hash)[:24]
    packet["prompt_id"] = f"common-situation-{pair_id}-{target_persona_id}"
    packet["common_situation_pair_id"] = pair_id
    packet["common_situation_audit"] = True
    packet["common_situation_reference"] = {
        "source_prompt_id": source_packet["prompt_id"],
        "source_packet_sha256": source_hash,
        "source_persona_id": source_packet["user_model_profile"]["persona_id"],
        "source_action": source["result"]["response"]["selected_action"],
        "source_path": source["source_path"],
    }
    leakage = scientific_packet_prompt_leakage(packet)
    if leakage:
        raise ValueError(f"Evaluator-only information leaked into prompt: {leakage}")
    return packet


def build(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    source_records = _source_records(results_root)
    selected, source_coverage = _select_crossed_contexts(source_records)
    packets = [
        _crossed_packet(source, target_persona)
        for source in selected
        for target_persona in PERSONAS
    ]

    packet_path = output_dir / "prompt_packets" / "part3_common_situation_audit_packets.jsonl"
    inventory_path = output_dir / "part3_common_situation_audit_inventory.csv"
    preflight_path = output_dir / "part3_common_situation_audit_preflight.json"
    spec_path = output_dir / "part3_common_situation_audit_spec.json"
    _write_jsonl(packet_path, packets)

    inventory: list[dict[str, Any]] = []
    for packet in packets:
        contact = packet["llm_visible_evidence"]["contact_opportunity"]
        metadata = _metadata(packet)
        reference = packet["common_situation_reference"]
        inventory.append(
            {
                "prompt_id": packet["prompt_id"],
                "pair_id": packet["common_situation_pair_id"],
                "target_persona_id": packet["user_model_profile"]["persona_id"],
                "source_persona_id": reference["source_persona_id"],
                "scenario": metadata["scenario_mode"],
                "condition": metadata["condition"],
                "role": metadata["role"],
                "seed": metadata["seed"],
                "abm_reason": packet["trace_evidence"]["decision_context"].get(
                    "abm_reason_type"
                ),
                "diversion_band": contact.get("diversion_cost_band"),
                "diversion_cost": contact.get("diversion_cost"),
                "urgency_band": contact.get("urgency_band"),
                "urgency": contact.get("urgency"),
                "patient_context_present": bool(contact.get("patient_context_present")),
                "memory_count": len(packet["llm_visible_evidence"].get("memory_state_before") or []),
                "evolving_state_sha256": _canonical_sha256(
                    packet["llm_visible_evidence"]["evolving_state_before"]
                ),
                "visible_context_sha256": _canonical_sha256(packet["llm_visible_evidence"]),
                "source_packet_sha256": reference["source_packet_sha256"],
            }
        )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    pair_counts = Counter(row["pair_id"] for row in inventory)
    target_counts = Counter(row["target_persona_id"] for row in inventory)
    errors: list[str] = []
    expected_context_count = len(SCENARIOS) * len(CONDITIONS) * len(PERSONAS) * len(ROLES)
    if len(selected) != expected_context_count:
        errors.append(f"Expected {expected_context_count} contexts, found {len(selected)}")
    if any(count != len(PERSONAS) for count in pair_counts.values()):
        errors.append("At least one context is not crossed over all five personas")
    if target_counts != Counter({persona: expected_context_count for persona in PERSONAS}):
        errors.append(f"Target persona counts are unbalanced: {dict(target_counts)}")
    for pair_id in pair_counts:
        hashes = {
            row["visible_context_sha256"]
            for row in inventory
            if row["pair_id"] == pair_id
        }
        if len(hashes) != 1:
            errors.append(f"Visible evidence differs within pair {pair_id}")
    coverage_counts = {
        "scenario": dict(Counter(row["scenario"] for row in inventory[::len(PERSONAS)])),
        "condition": dict(Counter(row["condition"] for row in inventory[::len(PERSONAS)])),
        "role": dict(Counter(row["role"] for row in inventory[::len(PERSONAS)])),
        "source_persona": dict(Counter(row["source_persona_id"] for row in inventory[::len(PERSONAS)])),
        "seed": dict(Counter(str(row["seed"]) for row in inventory[::len(PERSONAS)])),
        "abm_reason": dict(Counter(str(row["abm_reason"]) for row in inventory[::len(PERSONAS)])),
        "diversion_band": dict(Counter(str(row["diversion_band"]) for row in inventory[::len(PERSONAS)])),
        "urgency_band": dict(Counter(str(row["urgency_band"]) for row in inventory[::len(PERSONAS)])),
        "patient_context_present": dict(Counter(str(row["patient_context_present"]) for row in inventory[::len(PERSONAS)])),
    }
    required_levels = {
        "diversion_band": {"low", "medium", "high"},
        "urgency_band": {"low", "medium", "high"},
        "patient_context_present": {"True", "False"},
    }
    for field, expected_levels in required_levels.items():
        observed_levels = set(coverage_counts[field])
        if not expected_levels.issubset(observed_levels):
            errors.append(
                f"Coverage missing for {field}: expected {sorted(expected_levels)}, "
                f"observed {sorted(observed_levels)}"
            )

    specification = {
        "audit_name": "Part 3 common-situation persona audit",
        "status": "specified_before_inference",
        "primary_question": (
            "Do the five persona policies make different choices when the complete "
            "observable decision situation is held identical?"
        ),
        "context_count": expected_context_count,
        "target_persona_count": len(PERSONAS),
        "expected_packet_count": expected_context_count * len(PERSONAS),
        "primary_outcome": "engage versus defer_or_decline",
        "secondary_outcome": "three-category action distribution",
        "prespecified_checks": [
            "Team Connector minus Focus Protector engagement on operational coordination opportunities is positive.",
            "Patient Advocate shows a larger patient-linked engagement contrast than the mean of the other personas.",
            "Vigilant Monitor shows a larger high-versus-low urgency engagement contrast than the mean of the other personas.",
            "Focus Protector engages less under high than low diversion cost.",
            "Adaptive Generalist engages less under high than low diversion cost.",
        ],
        "inference_note": (
            "Intervals describe variation across sampled model situations, not uncertainty "
            "about human personality or staff behaviour."
        ),
    }
    spec_path.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n")

    preflight = {
        "preflight_pass": not errors,
        "errors": errors,
        "source_record_count": len(source_records),
        "selected_context_count": len(selected),
        "packet_count": len(packets),
        "pair_count": len(pair_counts),
        "target_persona_counts": dict(target_counts),
        "coverage_counts": coverage_counts,
        "source_coverage": source_coverage,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    if errors:
        raise ValueError("Common-situation preflight failed: " + "; ".join(errors))
    print(json.dumps(preflight, indent=2, sort_keys=True))
    return preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
