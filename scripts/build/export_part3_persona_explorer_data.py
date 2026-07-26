#!/usr/bin/env python3
"""Export verified, manually reviewed Part 3 appraisals to the persona explorer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_DIR / "web" / "persona-explorer" / "data" / "persona-results.json"

SCENARIO_MAP = {
    "normal_load": "normal_load",
    "high_load_high_acuity": "high_load",
}

# Each explorer score is one direct survey dimension. No composite is created.
SCORE_DIMENSIONS = {
    "overall_experience": "overall_person_space_fit",
    "coordination_support": "team_awareness",
    "focus_support": "task_continuity",
    "spatial_legibility": "spatial_legibility",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _mean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values]
    if not selected:
        raise ValueError("Cannot average an empty score group")
    return statistics.fmean(selected)


def _approved_quotes(
    review_rows: list[Mapping[str, str]],
) -> dict[tuple[str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, str, str]]] = defaultdict(list)
    for row in review_rows:
        if not _truthy(row.get("include_in_explorer")):
            continue
        review_passes = (
            _truthy(row.get("review_grounding_pass")),
            _truthy(row.get("review_direct_answer_pass")),
            _truthy(row.get("review_context_labeling_pass")),
            _truthy(row.get("review_claim_layer_pass")),
            _truthy(row.get("review_tradeoff_pass")),
        )
        naturalness = int(str(row.get("review_naturalness_1_to_3") or "0"))
        if not all(review_passes) or naturalness < 2:
            raise ValueError(
                f"Selected explorer quote has not passed review: {row.get('prompt_id')}"
            )
        scenario = SCENARIO_MAP[str(row["scenario"])]
        key = (str(row["persona_id"]), scenario, str(row["condition"]))
        order = int(str(row.get("explorer_display_order") or "999"))
        answer = str(row.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"Selected explorer quote is empty: {row.get('prompt_id')}")
        grouped[key].append((order, str(row.get("question_id", "")), answer))

    result = {}
    for key, rows in grouped.items():
        quotes = [answer for _, _, answer in sorted(rows)]
        if len(quotes) != 3:
            raise ValueError(
                f"Expected exactly three approved explorer quotes for {key}; got {len(quotes)}"
            )
        result[key] = quotes
    return result


def export(args: argparse.Namespace) -> dict[str, Any]:
    analysis_dir = Path(args.analysis_dir).resolve()
    template_path = Path(args.template).resolve()
    output_path = Path(args.output).resolve()
    summary = json.loads((analysis_dir / "appraisal_analysis_summary.json").read_text())
    if summary.get("analysis_pass") is not True or summary.get("paired_analysis_ready") is not True:
        raise ValueError("The verified appraisal analysis is not ready for explorer export")
    if summary.get("manual_review_required") is not True:
        raise ValueError("The appraisal analysis does not require the expected manual review")

    template = json.loads(template_path.read_text())
    survey_rows = _read_csv(analysis_dir / "survey_responses.csv")
    review_rows = _read_csv(analysis_dir / "interview_manual_review_sample.csv")
    quotes = _approved_quotes(review_rows)

    score_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    paired_units: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in survey_rows:
        if row.get("rateability") != "rateable" or not row.get("score_1_to_7"):
            continue
        scenario = SCENARIO_MAP[str(row["scenario"])]
        base = (str(row["persona_id"]), scenario, str(row["condition"]))
        score_groups[(*base, str(row["dimension"]))].append(float(row["score_1_to_7"]))
        paired_units[base].add(str(row["pair_id"]))

    expected_keys = {
        (persona["id"], scenario["id"], condition["id"])
        for persona in template["personas"]
        for scenario in template["scenarios"]
        for condition in template["conditions"]
    }
    if set(quotes) != expected_keys:
        missing = sorted(expected_keys - set(quotes))
        unexpected = sorted(set(quotes) - expected_keys)
        raise ValueError(
            f"Explorer quote coverage is incomplete; missing={missing}, unexpected={unexpected}"
        )

    results = []
    for persona_id, scenario, condition in sorted(expected_keys):
        base = (persona_id, scenario, condition)
        scores = {
            display_key: round(_mean(score_groups[(*base, dimension)]), 1)
            for display_key, dimension in SCORE_DIMENSIONS.items()
        }
        results.append(
            {
                "persona": persona_id,
                "scenario": scenario,
                "condition": condition,
                "source": "verified_part3_appraisal_analysis",
                "sample_size": len(paired_units[base]),
                "scores": scores,
                "quotes": quotes[base],
            }
        )

    template["meta"] = {
        "status": "observed",
        "scientific_result": True,
        "not_human_data": True,
        "claim_boundary": summary["claim_boundary"],
        "score_dimension_map": SCORE_DIMENSIONS,
        "note": (
            "Scores are direct means of verified synthetic appraisal dimensions; "
            "quotes passed the prespecified manual review and selection fields."
        ),
    }
    template["results"] = results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2) + "\n")
    return {
        "export_pass": True,
        "result_count": len(results),
        "quote_count": sum(len(row["quotes"]) for row in results),
        "score_dimension_map": SCORE_DIMENSIONS,
        "output": str(output_path),
        "human_testimony": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    return parser.parse_args()


def main() -> None:
    result = export(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
