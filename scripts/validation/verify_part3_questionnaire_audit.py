#!/usr/bin/env python3
"""Verify the Part 3 empirical-questionnaire audit and its negative controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.questionnaire_audit import (
    QUESTION_IDS,
    RATEABLE_QUESTION_IDS,
    normalize_questionnaire_response,
)
from src.vllm_backend import packet_json_schema


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _schema_digest(packet: dict[str, Any]) -> str:
    schema = packet_json_schema(str(packet["packet_type"]), packet)
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def _numeric(row: dict[str, Any]) -> float | None:
    value = row["share_percent"] if row["question_id"] == "q2" else row["score_1_to_7"]
    return float(value) if value is not None else None


def verify(packets_path: Path, responses_path: Path) -> dict[str, Any]:
    packets = _read_jsonl(packets_path)
    results = _read_jsonl(responses_path)
    packet_by_id = {str(row.get("prompt_id", "")): row for row in packets}
    result_by_id = {str(row.get("prompt_id", "")): row for row in results}
    errors = []
    if len(packet_by_id) != len(packets):
        errors.append("duplicate packet prompt IDs")
    if len(result_by_id) != len(results):
        errors.append("duplicate response prompt IDs")
    missing = sorted(set(packet_by_id) - set(result_by_id))
    unexpected = sorted(set(result_by_id) - set(packet_by_id))
    if missing:
        errors.append(f"missing responses: {missing[:10]}")
    if unexpected:
        errors.append(f"unexpected responses: {unexpected[:10]}")

    long_rows = []
    retry_count = 0
    exhausted = 0
    for prompt_id, packet in packet_by_id.items():
        result = result_by_id.get(prompt_id)
        if result is None:
            continue
        retry_count += int(result.get("semantic_retry_count", 0) or 0)
        if result.get("generation_error"):
            exhausted += 1
            errors.append(f"{prompt_id}: semantic correction exhausted")
            continue
        if result.get("packet_type") != "empirical_questionnaire_bundle":
            errors.append(f"{prompt_id}: packet type mismatch")
        if result.get("response_schema_sha256") != _schema_digest(packet):
            errors.append(f"{prompt_id}: schema digest mismatch")
        try:
            raw = json.loads(str(result.get("raw_response", "")))
            normalized = normalize_questionnaire_response(packet, raw)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"{prompt_id}: {type(error).__name__}: {error}")
            continue
        if normalized != result.get("response"):
            errors.append(f"{prompt_id}: normalized response differs from stored response")
        metadata = packet["trace_evidence"]["metadata"]
        for row in normalized["responses"]:
            long_rows.append(
                {
                    **metadata,
                    "prompt_id": prompt_id,
                    "control_variant": packet["control_variant"],
                    "assessment_variant": packet.get("assessment_variant", "balanced"),
                    **row,
                }
            )

    expected_counts = {
        "full_trace_persona": 300,
        "full_trace_neutral": 30,
        "evidence_withheld_persona": 30,
        "evidence_withheld_neutral": 30,
    }
    observed_counts = Counter(row.get("control_variant") for row in packets)
    if dict(observed_counts) != expected_counts:
        errors.append(f"control inventory differs: {dict(observed_counts)}")
    expected_assessments = {
        "balanced": 390,
    }
    observed_assessments = Counter(
        row.get("assessment_variant", "balanced") for row in packets
    )
    if dict(observed_assessments) != expected_assessments:
        errors.append(
            f"assessment inventory differs: {dict(observed_assessments)}"
        )

    def rate(rows: list[dict[str, Any]], predicate) -> float | None:
        selected = [row for row in rows if predicate(row)]
        return (
            sum(row["rateability"] == "insufficient_evidence" for row in selected) / len(selected)
            if selected else None
        )

    q8_rows = [
        row for row in long_rows
        if row["question_id"] == "q8" and row["control_variant"].startswith("full_trace")
    ]
    withheld_rows = [
        row for row in long_rows if row["control_variant"].startswith("evidence_withheld")
    ]
    supported_rows = [
        row for row in long_rows
        if row["control_variant"].startswith("full_trace") and row["question_id"] != "q8"
    ]
    q8_refusal = rate(q8_rows, lambda _row: True)
    withheld_refusal = rate(withheld_rows, lambda _row: True)
    supported_rateability = (
        sum(row["rateability"] == "rateable" for row in supported_rows) / len(supported_rows)
        if supported_rows else None
    )
    per_item_rateability = {}
    for question_id in RATEABLE_QUESTION_IDS:
        members = [row for row in supported_rows if row["question_id"] == question_id]
        per_item_rateability[question_id] = (
            sum(row["rateability"] == "rateable" for row in members) / len(members)
            if members else None
        )

    scientific_rows = []
    ensemble: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        if row["control_variant"] == "full_trace_persona":
            ensemble[
                (str(row["run_id"]), str(row["agent_id"]), str(row["question_id"]))
            ].append(row)
    for key, members in ensemble.items():
        variants = {str(row["assessment_variant"]) for row in members}
        if variants != {"balanced"}:
            errors.append(f"incomplete assessment ensemble for {key}")
            continue
        scientific_rows.append(dict(members[0]))

    scenario_means: dict[str, dict[str, float | None]] = defaultdict(dict)
    for scenario in ("normal_load", "high_load_high_acuity"):
        for question_id in QUESTION_IDS:
            values = [
                _numeric(row)
                for row in scientific_rows
                if row["scenario_mode"] == scenario
                and row["question_id"] == question_id
                and _numeric(row) is not None
            ]
            scenario_means[scenario][question_id] = statistics.mean(values) if values else None
    directional = {}
    for question_id in ("q1", "q2"):
        normal = scenario_means["normal_load"][question_id]
        high = scenario_means["high_load_high_acuity"][question_id]
        directional[question_id] = bool(normal is not None and high is not None and high > normal)

    matched_differences: dict[str, list[float]] = defaultdict(list)
    keyed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in long_rows:
        if (
            row["control_variant"] == "full_trace_persona"
            and row["assessment_variant"] != "balanced"
        ):
            continue
        key = (str(row["run_id"]), str(row["agent_id"]), str(row["question_id"]))
        keyed[key][str(row["control_variant"])] = row
    for variants in keyed.values():
        left = variants.get("full_trace_persona")
        right = variants.get("full_trace_neutral")
        if left and right and _numeric(left) is not None and _numeric(right) is not None:
            matched_differences[str(left["question_id"])].append(
                abs(float(_numeric(left)) - float(_numeric(right)))
            )

    scientific_gate = bool(
        not errors
        and q8_refusal is not None and q8_refusal >= 0.95
        and withheld_refusal is not None and withheld_refusal >= 0.95
        and all(
            value is not None and value >= 0.90
            for value in per_item_rateability.values()
        )
        and all(directional.values())
    )
    return {
        "technical_verification_pass": not errors,
        "scientific_gate_pass": scientific_gate,
        "scientific_result": False,
        "independent_output_review_required": True,
        "human_review_completed": False,
        "packet_count": len(packets),
        "response_count": len(results),
        "normalized_question_response_count": len(long_rows),
        "scientific_question_response_count_after_within_shift_aggregation": len(scientific_rows),
        "control_variant_counts": dict(sorted(observed_counts.items())),
        "assessment_variant_counts": dict(sorted(observed_assessments.items())),
        "missing_response_count": len(missing),
        "unexpected_response_count": len(unexpected),
        "semantic_retry_total": retry_count,
        "semantic_correction_exhausted_count": exhausted,
        "q8_acoustic_negative_control_refusal_rate": q8_refusal,
        "evidence_withheld_refusal_rate": withheld_refusal,
        "supported_q1_q7_rateability_rate": supported_rateability,
        "supported_rateability_rate_by_item": per_item_rateability,
        "q1_q2_high_load_direction_pass": directional,
        "scenario_question_means": dict(scenario_means),
        "matched_persona_neutral_mean_absolute_difference": {
            key: statistics.mean(values) if values else None
            for key, values in sorted(matched_differences.items())
        },
        "empirical_comparison_completed": False,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(Path(args.packets), Path(args.responses))
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["technical_verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
