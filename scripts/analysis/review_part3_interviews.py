#!/usr/bin/env python3
"""Apply the independent Codex audit to the prespecified Part 3 sample.

This script does not perform human review or discover themes. It merges a
frozen, investigator-requested Codex audit with the 40-bundle review sample,
then measures how well the existing dictionary reproduces the independently
assigned design-priority labels.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import tarfile
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_DIR
    / "outputs"
    / "source_results"
    / "part3_appraisals_final_24789077.tar.gz"
)
DEFAULT_MANIFEST = (
    PROJECT_DIR / "manifests" / "part3_independent_interview_review.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "findings"
    / "part3_cognitive_personas_n10"
    / "review"
)

QUESTION_IDS = (
    "overall_experience",
    "critical_moments",
    "coordination_and_focus",
    "supportive_feature",
    "difficult_feature",
    "counterfactual_change",
)

DICTIONARY_PATTERNS = {
    "visual": (
        r"transparen|visual|visibility|intervisib|status board|sightline|"
        r"line of sight|blind spot"
    ),
    "semi_private": r"semi-private|private|privacy|huddle|alcove|pod|nook|screen",
    "distributed": (
        r"near patient|near the patient|decentral|distributed|closer to patient|"
        r"at bedside|bedside|without.*return|without.*travel|reduce travel"
    ),
    "acoustic": r"acoustic|sound|noise|quiet",
    "availability": (
        r"availab|focus mode|do not disturb|status indicator|"
        r"indicator of.*status|signal"
    ),
}


def _read_csv(source: Path, suffix: str) -> pd.DataFrame:
    if source.is_dir():
        matches = [p for p in source.rglob(Path(suffix).name) if str(p).endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {suffix!r}; found {len(matches)}")
        return pd.read_csv(matches[0])
    with tarfile.open(source, "r:gz") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {suffix!r}; found {len(matches)}")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise ValueError(f"Could not read {matches[0].name}")
        return pd.read_csv(io.BytesIO(handle.read()))


def _bundle_id(prompt_id: str) -> str:
    match = re.search(r"_([0-9a-f]{20})$", prompt_id)
    if not match:
        raise ValueError(f"Unexpected prompt id: {prompt_id}")
    return match.group(1)


def _validate_manifest(manifest: dict[str, Any], sample: pd.DataFrame) -> None:
    if manifest.get("human_review_completed") is not False:
        raise ValueError("Independent audit must not be labeled as human review")
    bundles = manifest["bundles"]
    observed = {_bundle_id(value) for value in sample["prompt_id"].unique()}
    if set(bundles) != observed:
        raise ValueError(
            "Audit manifest does not exactly match the prespecified sample: "
            f"missing={sorted(observed - set(bundles))}, "
            f"extra={sorted(set(bundles) - observed)}"
        )
    counts = sample.groupby("prompt_id")["question_id"].agg(list)
    for prompt_id, questions in counts.items():
        if set(questions) != set(QUESTION_IDS) or len(questions) != len(QUESTION_IDS):
            raise ValueError(f"Incomplete question set for {prompt_id}")
        bundle = bundles[_bundle_id(prompt_id)]
        for key in (
            "grounding_fail",
            "claim_layer_fail",
            "naturalness_1",
            "naturalness_3",
            "selected_for_display",
        ):
            unknown = set(bundle[key]) - set(QUESTION_IDS)
            if unknown:
                raise ValueError(f"Unknown questions in {prompt_id}/{key}: {unknown}")
        selected = bundle["selected_for_display"]
        failed = set(bundle["grounding_fail"]) | set(bundle["claim_layer_fail"])
        if set(selected) & failed:
            raise ValueError(f"Failed answer selected for display in {prompt_id}")
        if len(selected) > 3 or len(selected) != len(set(selected)):
            raise ValueError(f"Invalid display selection in {prompt_id}")
        unknown_codes = set(bundle["design_priorities"]) - set(
            manifest["design_priority_codebook"]
        )
        if unknown_codes:
            raise ValueError(f"Unknown design-priority codes in {prompt_id}")


def _annotate(sample: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    defaults = manifest["defaults"]
    records: list[dict[str, Any]] = []
    for source_row in sample.to_dict(orient="records"):
        row = dict(source_row)
        bundle = manifest["bundles"][_bundle_id(str(row["prompt_id"]))]
        question = str(row["question_id"])
        row["review_grounding_pass"] = question not in bundle["grounding_fail"]
        if question in bundle["naturalness_1"]:
            naturalness = 1
        elif question in bundle["naturalness_3"]:
            naturalness = 3
        else:
            naturalness = int(defaults["naturalness_1_to_3"])
        row["review_naturalness_1_to_3"] = naturalness
        row["review_direct_answer_pass"] = bool(defaults["direct_answer_pass"])
        row["review_context_labeling_pass"] = bool(
            defaults["context_labeling_pass"]
        )
        row["review_claim_layer_pass"] = (
            question not in bundle["claim_layer_fail"]
        )
        row["review_tradeoff_pass"] = (
            bool(defaults["tradeoff_pass_for_counterfactual"])
            if question == "counterfactual_change"
            else pd.NA
        )
        row["review_notes"] = bundle["notes"].get(question, "")
        selected = bundle["selected_for_display"]
        row["include_in_explorer"] = question in selected
        row["explorer_display_order"] = (
            selected.index(question) + 1 if question in selected else pd.NA
        )
        row["reviewer_type"] = manifest["reviewer_type"]
        row["reviewer_system"] = manifest["reviewer_system"]
        row["review_date"] = manifest["review_date"]
        row["human_review_completed"] = False
        records.append(row)
    return pd.DataFrame(records)


def _dictionary_validation(
    reviewed: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counterfactuals = reviewed[
        reviewed["question_id"] == "counterfactual_change"
    ].copy()
    manual_rows = []
    validation_rows = []
    for row in counterfactuals.itertuples(index=False):
        bundle_id = _bundle_id(row.prompt_id)
        assigned = set(manifest["bundles"][bundle_id]["design_priorities"])
        answer = str(row.answer).lower()
        detected = {
            code
            for code, pattern in DICTIONARY_PATTERNS.items()
            if re.search(pattern, answer)
        }
        manual_rows.append(
            {
                "prompt_id": row.prompt_id,
                "bundle_id": bundle_id,
                "scenario": row.scenario,
                "condition": row.condition,
                "persona_id": row.persona_id,
                "answer": row.answer,
                **{
                    f"manual_{code}": code in assigned
                    for code in DICTIONARY_PATTERNS
                },
                **{
                    f"dictionary_{code}": code in detected
                    for code in DICTIONARY_PATTERNS
                },
            }
        )
    coded = pd.DataFrame(manual_rows)
    for code in DICTIONARY_PATTERNS:
        manual = coded[f"manual_{code}"]
        predicted = coded[f"dictionary_{code}"]
        tp = int((manual & predicted).sum())
        fp = int((~manual & predicted).sum())
        fn = int((manual & ~predicted).sum())
        tn = int((~manual & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        validation_rows.append(
            {
                "design_priority_code": code,
                "design_priority": manifest["design_priority_codebook"][code],
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return coded, pd.DataFrame(validation_rows)


def _summary(
    reviewed: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    scientific_use = (
        reviewed["review_grounding_pass"].astype(bool)
        & reviewed["review_claim_layer_pass"].astype(bool)
        & reviewed["review_direct_answer_pass"].astype(bool)
        & reviewed["review_context_labeling_pass"].astype(bool)
    )
    bundle_use = reviewed.assign(scientific_use=scientific_use).groupby(
        "prompt_id"
    )["scientific_use"].agg(["sum", "count"])
    return {
        "review_scope": manifest["review_scope"],
        "reviewer_type": manifest["reviewer_type"],
        "reviewer_system": manifest["reviewer_system"],
        "review_date": manifest["review_date"],
        "human_review_completed": False,
        "provenance_note": manifest["provenance_note"],
        "bundle_count": int(reviewed["prompt_id"].nunique()),
        "answer_count": int(len(reviewed)),
        "answers_passing_scientific_use_gate": int(scientific_use.sum()),
        "answers_excluded_from_scientific_use": int((~scientific_use).sum()),
        "bundles_with_at_least_three_usable_answers": int(
            (bundle_use["sum"] >= 3).sum()
        ),
        "bundles_with_no_display_quote": int(
            reviewed.groupby("prompt_id")["include_in_explorer"]
            .sum()
            .eq(0)
            .sum()
        ),
        "selected_display_answer_count": int(
            reviewed["include_in_explorer"].astype(bool).sum()
        ),
        "dictionary_validation_minimum_f1": float(validation["f1"].min()),
        "dictionary_validation_macro_f1": float(validation["f1"].mean()),
        "interpretation": (
            "The audit supports selective use of grounded synthetic responses. "
            "It does not satisfy a human-review requirement and does not establish "
            "themes as human-coded findings."
        ),
    }


def _write_markdown(
    summary: dict[str, Any], validation: pd.DataFrame, output: Path
) -> None:
    rows = "\n".join(
        "| {design_priority} | {precision:.2f} | {recall:.2f} | {f1:.2f} | "
        "{true_positive} | {false_positive} | {false_negative} |".format(**row)
        for row in validation.to_dict(orient="records")
    )
    output.write_text(
        "# Independent Codex Audit of Part 3 Interviews\n\n"
        "## Provenance\n\n"
        f"- Reviewer type: `{summary['reviewer_type']}`\n"
        f"- Reviewer system: {summary['reviewer_system']}\n"
        f"- Review date: {summary['review_date']}\n"
        "- Human review completed: **no**\n\n"
        f"{summary['provenance_note']}\n\n"
        "## Disposition\n\n"
        f"- Bundles audited: {summary['bundle_count']}\n"
        f"- Answers audited: {summary['answer_count']}\n"
        "- Answers passing the scientific-use gate: "
        f"{summary['answers_passing_scientific_use_gate']}\n"
        "- Answers excluded from scientific use: "
        f"{summary['answers_excluded_from_scientific_use']}\n"
        "- Answers selected for supplementary display: "
        f"{summary['selected_display_answer_count']}\n"
        "- Bundles with no acceptable display quotation: "
        f"{summary['bundles_with_no_display_quote']}\n\n"
        "An answer was excluded when it introduced unsupported events or turned "
        "a modeled movement, visibility, or contact pattern into an unmeasured "
        "claim about privacy, dignity, safety, errors, care quality, or patient "
        "outcomes. Fluent prose was not treated as evidence.\n\n"
        "## Dictionary validation\n\n"
        "| Design priority | Precision | Recall | F1 | TP | FP | FN |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "These statistics compare the pre-existing keyword dictionary with "
        "independently assigned codes on the 40 prespecified counterfactual "
        "answers. They measure dictionary agreement with this Codex audit, not "
        "agreement with a human coder.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    sample = _read_csv(
        args.source, "analysis/interview_manual_review_sample.csv"
    )
    _validate_manifest(manifest, sample)
    reviewed = _annotate(sample, manifest)
    coded, validation = _dictionary_validation(reviewed, manifest)
    summary = _summary(reviewed, validation, manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(args.output_dir / "reviewed_interview_sample.csv", index=False)
    coded.to_csv(
        args.output_dir / "reviewed_counterfactual_codes.csv", index=False
    )
    validation.to_csv(
        args.output_dir / "dictionary_validation.csv", index=False
    )
    (args.output_dir / "review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _write_markdown(
        summary, validation, args.output_dir / "independent_review_report.md"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
