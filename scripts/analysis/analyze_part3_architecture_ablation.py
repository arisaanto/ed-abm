#!/usr/bin/env python3
"""Analyze the matched Part 3 persona x memory architecture ablation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


CELL_BY_VARIANT = {
    "full_persona_full_memory": (1, 1),
    "neutral_orientation": (0, 1),
    "persona_no_memory": (1, 0),
    "neutral_no_memory": (0, 0),
}
T_975 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
    12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
    17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
    22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
    27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _response_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("response")
    if isinstance(payload, dict):
        return payload
    result = row.get("result")
    if isinstance(result, dict) and isinstance(result.get("response"), dict):
        return result["response"]
    raise ValueError(f"Missing parsed response for {row.get('prompt_id')}")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def factorial_effects(cells: dict[str, dict[str, Any]]) -> dict[str, float]:
    engage = {
        key: float(value["selected_action"] == "engage")
        for key, value in cells.items()
    }
    ff = engage["full_persona_full_memory"]
    nf = engage["neutral_orientation"]
    fn = engage["persona_no_memory"]
    nn = engage["neutral_no_memory"]
    return {
        "orientation_effect_with_memory": ff - nf,
        "orientation_effect_without_memory": fn - nn,
        "memory_effect_with_persona": ff - fn,
        "memory_effect_neutral": nf - nn,
        "orientation_main_effect": ((ff - nf) + (fn - nn)) / 2.0,
        "memory_main_effect": ((ff - fn) + (nf - nn)) / 2.0,
        "orientation_memory_interaction": ff - nf - fn + nn,
    }


def _aggregate(
    values: list[float],
    *,
    clusters: list[str] | None = None,
) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    cluster_count = len(set(clusters or []))
    if clusters is not None and len(clusters) != n:
        raise ValueError("Cluster labels must align with effect values")
    if clusters is not None and cluster_count > 1:
        residual_sums: dict[str, float] = defaultdict(float)
        for value, cluster in zip(values, clusters):
            residual_sums[str(cluster)] += value - mean
        variance = (
            cluster_count
            / (cluster_count - 1)
            * sum(value ** 2 for value in residual_sums.values())
            / (n ** 2)
        )
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
        se = math.sqrt(variance)
        critical = T_975.get(cluster_count, 1.96)
        low, high = mean - critical * se, mean + critical * se
        ci_method = "seed_cluster_robust_t"
    elif n > 1:
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
        se = sd / math.sqrt(n)
        critical = T_975.get(n, 1.96)
        low, high = mean - critical * se, mean + critical * se
        ci_method = "pair_level_t"
    else:
        sd = se = 0.0
        low = high = mean
        ci_method = "descriptive_single_observation"
    return {
        "n_pairs": n,
        "cluster_count": cluster_count if clusters is not None else 0,
        "ci_method": ci_method,
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "positive_share": round(sum(value > 0 for value in values) / n, 6),
        "negative_share": round(sum(value < 0 for value in values) / n, 6),
        "zero_share": round(sum(value == 0 for value in values) / n, 6),
    }


def analyze(packets_path: Path, responses_path: Path, output_dir: Path) -> dict[str, Any]:
    packets = _read_jsonl(packets_path)
    responses = _read_jsonl(responses_path)
    response_by_id = {row["prompt_id"]: row for row in responses}
    errors = []
    if len(response_by_id) != len(responses):
        errors.append("Duplicate response prompt ids")
    if set(response_by_id) != {packet["prompt_id"] for packet in packets}:
        errors.append("Packet and response prompt-id inventories differ")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    metadata_by_pair: dict[str, dict[str, Any]] = {}
    historical_reference_by_pair: dict[str, dict[str, Any]] = {}
    cell_rows = []
    for packet in packets:
        pair_id = packet["ablation_pair_id"]
        variant = packet["ablation_variant"]
        reference = packet["ablation_reference"]
        trace = packet["trace_evidence"]
        metadata = trace["metadata"]
        contact = packet["llm_visible_evidence"]["contact_opportunity"]
        metadata_by_pair[pair_id] = {
            "scenario": metadata["scenario_mode"],
            "condition": metadata["condition"],
            "source_persona_id": reference["source_persona_id"],
            "role": metadata["role"],
            "seed": metadata["seed"],
            "source_prompt_id": reference["source_prompt_id"],
            "abm_reason_type": trace["decision_context"]["abm_reason_type"],
            "urgency_band": contact["urgency_band"],
            "mutual_visibility": contact["mutual_visibility"],
            "patient_context_present": contact["patient_context_present"],
            "source_memory_count": reference["source_memory_count"],
        }
        generated = _response_payload(response_by_id[packet["prompt_id"]])
        grouped[pair_id][variant] = generated
        historical_reference_by_pair[pair_id] = reference[
            "observed_full_persona_memory_response"
        ]

    expected_cells = set(CELL_BY_VARIANT)
    pair_rows = []
    for pair_id, cells in sorted(grouped.items()):
        if set(cells) != expected_cells:
            errors.append(f"Pair {pair_id} has cells {sorted(cells)}")
            continue
        meta = metadata_by_pair[pair_id]
        historical = historical_reference_by_pair[pair_id]
        for cell, response in cells.items():
            orientation, memory = CELL_BY_VARIANT[cell]
            cell_rows.append(
                {
                    "ablation_pair_id": pair_id,
                    **meta,
                    "cell": cell,
                    "persona_orientation_present": orientation,
                    "grounded_memory_present": memory,
                    "selected_action": response["selected_action"],
                    "selected_reason": response["selected_reason"],
                    "topic_family": response["topic_family"],
                    "engage_indicator": int(response["selected_action"] == "engage"),
                }
            )
        effects = factorial_effects(cells)
        pair_rows.append(
            {
                "ablation_pair_id": pair_id,
                **meta,
                **effects,
                "persona_memory_action_agreement": int(
                    cells["full_persona_full_memory"]["selected_action"]
                    == cells["persona_no_memory"]["selected_action"]
                ),
                "full_memory_orientation_action_agreement": int(
                    cells["full_persona_full_memory"]["selected_action"]
                    == cells["neutral_orientation"]["selected_action"]
                ),
                "all_four_action_agreement": int(
                    len({response["selected_action"] for response in cells.values()}) == 1
                ),
                "all_four_reason_agreement": int(
                    len({response["selected_reason"] for response in cells.values()}) == 1
                ),
                "all_four_topic_agreement": int(
                    len({response["topic_family"] for response in cells.values()}) == 1
                ),
                "historical_reference_action_agreement": int(
                    cells["full_persona_full_memory"]["selected_action"]
                    == historical["selected_action"]
                ),
                "historical_reference_reason_agreement": int(
                    cells["full_persona_full_memory"]["selected_reason"]
                    == historical["selected_reason"]
                ),
                "historical_reference_topic_agreement": int(
                    cells["full_persona_full_memory"]["topic_family"]
                    == historical["topic_family"]
                ),
            }
        )

    effect_names = list(factorial_effects({
        key: {"selected_action": "decline"} for key in CELL_BY_VARIANT
    }))
    aggregate_rows = []
    scopes = [("overall", "all", pair_rows)]
    for field in (
        "scenario", "condition", "source_persona_id", "role",
        "abm_reason_type", "urgency_band", "source_memory_count",
    ):
        values = sorted({row[field] for row in pair_rows})
        scopes.extend(
            (field, value, [row for row in pair_rows if row[field] == value])
            for value in values
        )
    for scope, value, members in scopes:
        for effect in effect_names:
            aggregate_rows.append(
                {
                    "scope": scope,
                    "scope_value": value,
                    "effect": effect,
                    **_aggregate(
                        [float(row[effect]) for row in members],
                        clusters=[str(row["seed"]) for row in members],
                    ),
                    "claim_boundary": (
                        "Matched prompt-level component substitution on fixed observed "
                        "opportunities; not a recursive closed-loop trajectory effect."
                    ),
                }
            )

    cell_fields = [
        "ablation_pair_id", "scenario", "condition", "source_persona_id", "role",
        "seed", "source_prompt_id", "abm_reason_type", "urgency_band",
        "mutual_visibility", "patient_context_present", "source_memory_count",
        "cell", "persona_orientation_present",
        "grounded_memory_present", "selected_action", "selected_reason", "topic_family",
        "engage_indicator",
    ]
    pair_fields = [
        "ablation_pair_id", "scenario", "condition", "source_persona_id", "role",
        "seed", "source_prompt_id", "abm_reason_type", "urgency_band",
        "mutual_visibility", "patient_context_present", "source_memory_count",
        *effect_names, "persona_memory_action_agreement",
        "full_memory_orientation_action_agreement", "all_four_action_agreement",
        "all_four_reason_agreement", "all_four_topic_agreement",
        "historical_reference_action_agreement",
        "historical_reference_reason_agreement",
        "historical_reference_topic_agreement",
    ]
    aggregate_fields = [
        "scope", "scope_value", "effect", "n_pairs", "cluster_count", "ci_method",
        "mean", "sd", "se",
        "ci95_low", "ci95_high", "positive_share", "negative_share", "zero_share",
        "claim_boundary",
    ]
    _write_csv(output_dir / "architecture_ablation_cells.csv", cell_rows, cell_fields)
    _write_csv(output_dir / "architecture_ablation_pair_effects.csv", pair_rows, pair_fields)
    _write_csv(output_dir / "architecture_ablation_effects.csv", aggregate_rows, aggregate_fields)

    overall_effects = {
        row["effect"]: {
            key: row[key]
            for key in (
                "n_pairs", "cluster_count", "ci_method", "mean", "se",
                "ci95_low", "ci95_high",
            )
        }
        for row in aggregate_rows
        if row["scope"] == "overall"
    }
    historical_action_confusion = Counter()
    historical_action_agreement_by_persona: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        pair_id = row["ablation_pair_id"]
        historical_action = historical_reference_by_pair[pair_id]["selected_action"]
        replay_action = grouped[pair_id]["full_persona_full_memory"]["selected_action"]
        historical_action_confusion[f"{historical_action}_to_{replay_action}"] += 1
    for persona in sorted({row["source_persona_id"] for row in pair_rows}):
        members = [row for row in pair_rows if row["source_persona_id"] == persona]
        historical_action_agreement_by_persona[persona] = {
            "matched_pair_count": len(members),
            "action_agreement_rate": round(
                sum(row["historical_reference_action_agreement"] for row in members)
                / len(members),
                6,
            ),
        }

    summary = {
        "analysis_pass": not errors and len(pair_rows) == 120 and len(cell_rows) == 480,
        "packet_count": len(packets),
        "response_count": len(responses),
        "matched_pair_count": len(pair_rows),
        "reconstructed_cell_count": len(cell_rows),
        "action_counts_by_cell": {
            cell: dict(Counter(row["selected_action"] for row in cell_rows if row["cell"] == cell))
            for cell in CELL_BY_VARIANT
        },
        "all_four_action_agreement_rate": (
            sum(row["all_four_action_agreement"] for row in pair_rows) / len(pair_rows)
            if pair_rows else None
        ),
        "historical_reference_action_agreement_rate": (
            sum(row["historical_reference_action_agreement"] for row in pair_rows)
            / len(pair_rows) if pair_rows else None
        ),
        "historical_reference_reason_agreement_rate": (
            sum(row["historical_reference_reason_agreement"] for row in pair_rows)
            / len(pair_rows) if pair_rows else None
        ),
        "historical_reference_topic_agreement_rate": (
            sum(row["historical_reference_topic_agreement"] for row in pair_rows)
            / len(pair_rows) if pair_rows else None
        ),
        "historical_reference_action_confusion": dict(
            sorted(historical_action_confusion.items())
        ),
        "historical_reference_action_agreement_by_persona": (
            historical_action_agreement_by_persona
        ),
        "primary_effects_seed_clustered": overall_effects,
        "primary_inference_unit": "simulation seed",
        "replay_sensitivity_note": (
            "The contemporaneous full-persona/full-memory replay is model-visible "
            "equivalent to the historical closed-loop packet, but categorical replay "
            "agreement is imperfect. Treat orientation contrasts as prompt-level "
            "component evidence and memory contrasts as sensitivity evidence; neither "
            "is a recursive trajectory estimate."
        ),
        "scientific_result_scope": "matched_prompt_level_architecture_ablation",
        "closed_loop_trajectory_result": False,
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "architecture_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(Path(args.packets), Path(args.responses), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    if not summary["analysis_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
