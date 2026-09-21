#!/usr/bin/env python3
"""Analyse the crossed Part 3 common-situation persona audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Callable, Iterable


PERSONAS = (
    "adaptive_generalist",
    "focus_protector",
    "patient_advocate",
    "team_connector",
    "vigilant_monitor",
)
PERSONA_LABELS = {
    "adaptive_generalist": "Adaptive Generalist",
    "focus_protector": "Focus Protector",
    "patient_advocate": "Patient Advocate",
    "team_connector": "Team Connector",
    "vigilant_monitor": "Vigilant Monitor",
}
OPERATIONAL_REASONS = {
    "BED_FLOW_COORDINATION",
    "HANDOFF_NEED",
    "POST_TASK_UPDATE",
    "CORRIDOR_PASSING_UPDATE",
    "HIGH_ACUITY_ESCALATION",
}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _ci(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]


def _paired_bootstrap(
    contexts: list[dict[str, Any]],
    estimator: Callable[[list[dict[str, Any]]], float],
    *,
    draws: int = 10000,
    seed: int = 20260920,
) -> tuple[float, float, float]:
    observed = estimator(contexts)
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        sample = [contexts[rng.randrange(len(contexts))] for _ in contexts]
        boot.append(estimator(sample))
    low, high = _ci(boot)
    return observed, low, high


def _engage(row: dict[str, Any], persona: str) -> float:
    return float(row["actions"][persona] == "engage")


def _persona_rate(rows: list[dict[str, Any]], persona: str) -> float:
    return mean(_engage(row, persona) for row in rows)


def _subset(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def _difference(rows: list[dict[str, Any]], left: str, right: str) -> float:
    return mean(_engage(row, left) - _engage(row, right) for row in rows)


def _within_contrast(
    rows: list[dict[str, Any]], persona: str, field: str, high: Any, low: Any
) -> float:
    high_rows = _subset(rows, lambda row: row[field] == high)
    low_rows = _subset(rows, lambda row: row[field] == low)
    if not high_rows or not low_rows:
        raise ValueError(f"No coverage for {field}: high={high}, low={low}")
    return _persona_rate(high_rows, persona) - _persona_rate(low_rows, persona)


def _did_vs_others(
    rows: list[dict[str, Any]], persona: str, field: str, high: Any, low: Any
) -> float:
    focal = _within_contrast(rows, persona, field, high, low)
    others = [
        _within_contrast(rows, other, field, high, low)
        for other in PERSONAS
        if other != persona
    ]
    return focal - mean(others)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    packet_rows = list(_read_jsonl(Path(args.packets)))
    response_rows = list(_read_jsonl(Path(args.responses)))
    responses = {str(row["prompt_id"]): row for row in response_rows}
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    metadata: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for packet in packet_rows:
        prompt_id = str(packet["prompt_id"])
        response = responses.get(prompt_id)
        if response is None:
            errors.append(f"Missing response: {prompt_id}")
            continue
        pair_id = str(packet["common_situation_pair_id"])
        persona = str(packet["user_model_profile"]["persona_id"])
        action = str(response["response"]["selected_action"])
        grouped[pair_id][persona] = action
        contact = packet["llm_visible_evidence"]["contact_opportunity"]
        trace = packet["trace_evidence"]
        metadata.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "scenario": trace["metadata"]["scenario_mode"],
                "condition": trace["metadata"]["condition"],
                "role": trace["metadata"]["role"],
                "seed": trace["metadata"]["seed"],
                "source_persona": packet["common_situation_reference"]["source_persona_id"],
                "abm_reason": trace["decision_context"].get("abm_reason_type"),
                "diversion_band": contact.get("diversion_cost_band"),
                "urgency_band": contact.get("urgency_band"),
                "patient_context_present": bool(contact.get("patient_context_present")),
            },
        )
    contexts: list[dict[str, Any]] = []
    for pair_id, actions in grouped.items():
        if set(actions) != set(PERSONAS):
            errors.append(f"Incomplete persona crossing for {pair_id}: {sorted(actions)}")
            continue
        contexts.append({**metadata[pair_id], "actions": dict(actions)})
    if errors:
        raise ValueError("; ".join(errors[:20]))

    persona_rows: list[dict[str, Any]] = []
    for persona in PERSONAS:
        value, low, high = _paired_bootstrap(
            contexts, lambda rows, p=persona: _persona_rate(rows, p)
        )
        persona_rows.append(
            {
                "persona": PERSONA_LABELS[persona],
                "n_shared_situations": len(contexts),
                "engagement_rate": round(value, 6),
                "ci95_low": round(low, 6),
                "ci95_high": round(high, 6),
                "engage_count": sum(row["actions"][persona] == "engage" for row in contexts),
                "defer_count": sum(row["actions"][persona] == "defer" for row in contexts),
                "decline_count": sum(row["actions"][persona] == "decline" for row in contexts),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for index, left in enumerate(PERSONAS):
        for right in PERSONAS[index + 1 :]:
            value, low, high = _paired_bootstrap(
                contexts,
                lambda rows, l=left, r=right: _difference(rows, l, r),
                seed=20260921 + index,
            )
            pairwise_rows.append(
                {
                    "left_persona": PERSONA_LABELS[left],
                    "right_persona": PERSONA_LABELS[right],
                    "n_shared_situations": len(contexts),
                    "engagement_rate_difference": round(value, 6),
                    "ci95_low": round(low, 6),
                    "ci95_high": round(high, 6),
                    "action_disagreement_rate": round(
                        mean(row["actions"][left] != row["actions"][right] for row in contexts),
                        6,
                    ),
                }
            )

    operational = _subset(contexts, lambda row: row["abm_reason"] in OPERATIONAL_REASONS)
    checks = [
        (
            "Team Connector vs Focus Protector on operational coordination",
            "positive",
            operational,
            lambda rows: _difference(rows, "team_connector", "focus_protector"),
        ),
        (
            "Patient Advocate patient-linked differential vs other personas",
            "positive",
            contexts,
            lambda rows: _did_vs_others(
                rows, "patient_advocate", "patient_context_present", True, False
            ),
        ),
        (
            "Vigilant Monitor high-vs-low urgency differential vs other personas",
            "positive",
            contexts,
            lambda rows: _did_vs_others(
                rows, "vigilant_monitor", "urgency_band", "high", "low"
            ),
        ),
        (
            "Focus Protector high-vs-low diversion contrast",
            "negative",
            contexts,
            lambda rows: _within_contrast(
                rows, "focus_protector", "diversion_band", "high", "low"
            ),
        ),
        (
            "Adaptive Generalist high-vs-low diversion contrast",
            "negative",
            contexts,
            lambda rows: _within_contrast(
                rows, "adaptive_generalist", "diversion_band", "high", "low"
            ),
        ),
    ]
    check_rows: list[dict[str, Any]] = []
    for index, (name, expected, rows, estimator) in enumerate(checks):
        value, low, high = _paired_bootstrap(
            rows, estimator, seed=20261000 + index
        )
        direction_pass = value > 0 if expected == "positive" else value < 0
        check_rows.append(
            {
                "construct_check": name,
                "expected_direction": expected,
                "n_shared_situations": len(rows),
                "estimate": round(value, 6),
                "ci95_low": round(low, 6),
                "ci95_high": round(high, 6),
                "direction_pass": direction_pass,
            }
        )

    context_rows: list[dict[str, Any]] = []
    for row in sorted(contexts, key=lambda value: value["pair_id"]):
        context_rows.append(
            {
                **{key: value for key, value in row.items() if key != "actions"},
                **{f"{persona}_action": row["actions"][persona] for persona in PERSONAS},
                "distinct_action_count": len(set(row["actions"].values())),
            }
        )
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "common_situation_persona_rates.csv", persona_rows)
    _write_csv(output_dir / "common_situation_pairwise_differences.csv", pairwise_rows)
    _write_csv(output_dir / "common_situation_construct_checks.csv", check_rows)
    _write_csv(output_dir / "common_situation_context_decisions.csv", context_rows)

    summary = {
        "analysis_pass": True,
        "context_count": len(contexts),
        "response_count": len(response_rows),
        "contexts_with_action_divergence": sum(
            len(set(row["actions"].values())) > 1 for row in contexts
        ),
        "action_divergence_rate": mean(
            len(set(row["actions"].values())) > 1 for row in contexts
        ),
        "all_prespecified_directions_pass": all(
            row["direction_pass"] for row in check_rows
        ),
        "persona_rates": persona_rows,
        "prespecified_checks": check_rows,
        "interpretation_boundary": (
            "This is an implementation audit of five designed model policies. "
            "It does not estimate personality effects in human staff."
        ),
    }
    (output_dir / "common_situation_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
