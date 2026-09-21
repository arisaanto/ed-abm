#!/usr/bin/env python3
"""Standardize Part 3 engagement rates to a common opportunity mix.

The closed-loop intervention changes both the number and composition of
optional encounters. This audit asks whether the reported Baseline-to-Both
engagement contrasts remain after each condition is reweighted to the same
mix of focal staff role, ABM interaction reason, and partner role. It uses
only sampled LLM decisions and retains strata observed in both conditions.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


PERSONA_LABELS = {
    "team_connector": "Team Connector",
    "focus_protector": "Focus Protector",
    "patient_advocate": "Patient Advocate",
    "vigilant_monitor": "Vigilant Monitor",
    "adaptive_generalist": "Adaptive Generalist",
}

CONDITION_LABELS = {"baseline": "Baseline", "both": "Both"}


def read_decisions(source: Path, strata_fields: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if source.is_dir():
        paths = sorted(source.rglob("part3_cognitive_decisions.jsonl"))
        if not paths:
            raise FileNotFoundError("No Part 3 decision logs in result directory")
        for path in paths:
            for line in path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with tarfile.open(source, "r:gz") as bundle:
            members = [
                member
                for member in bundle.getmembers()
                if member.name.endswith("/part3_cognitive_decisions.jsonl")
            ]
            if not members:
                raise FileNotFoundError("No Part 3 decision logs in archive")
            for member in members:
                stream = bundle.extractfile(member)
                if stream is None:
                    continue
                for raw_line in stream:
                    line = raw_line.decode("utf-8").strip()
                    if line:
                        rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    required = {
        "scenario",
        "condition",
        "seed",
        "persona_id",
        "role",
        "abm_reason_type",
        "sampled_for_model",
        "selected_action",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Decision logs lack columns: {sorted(missing)}")
    frame = frame.loc[
        frame["sampled_for_model"].astype(bool)
        & frame["condition"].isin(CONDITION_LABELS)
    ].copy()
    frame["engaged"] = frame["selected_action"].eq("engage").astype(float)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    missing_strata = set(strata_fields) - set(frame.columns)
    if missing_strata:
        raise ValueError(f"Decision logs lack strata columns: {sorted(missing_strata)}")
    frame["stratum"] = frame[strata_fields].fillna("Unknown").astype(str).agg(" | ".join, axis=1)
    return frame


def standardized_contrast(frame: pd.DataFrame) -> dict:
    counts = frame.groupby(["condition", "stratum"]).size().unstack(fill_value=0)
    for condition in CONDITION_LABELS:
        if condition not in counts.index:
            counts.loc[condition] = 0
    common = counts.columns[
        (counts.loc["baseline"] > 0) & (counts.loc["both"] > 0)
    ]
    retained = frame.loc[frame["stratum"].isin(common)].copy()
    total_counts = frame.groupby("condition").size()
    retained_counts = retained.groupby("condition").size()

    stratum_counts = (
        retained.groupby(["condition", "stratum"]).size().unstack(fill_value=0)
    )
    condition_mix = stratum_counts.div(stratum_counts.sum(axis=1), axis=0)
    target_mix = condition_mix.loc[["baseline", "both"]].mean(axis=0)
    cell_rates = retained.groupby(["condition", "stratum"])["engaged"].mean()
    rates = {}
    for condition in CONDITION_LABELS:
        rates[condition] = float(
            sum(
                target_mix[stratum] * cell_rates.loc[(condition, stratum)]
                for stratum in common
            )
        )
    return {
        "baseline_rate": rates["baseline"],
        "both_rate": rates["both"],
        "contrast": rates["both"] - rates["baseline"],
        "common_strata": int(len(common)),
        "baseline_n": int(total_counts.get("baseline", 0)),
        "both_n": int(total_counts.get("both", 0)),
        "baseline_common_support_fraction": float(
            retained_counts.get("baseline", 0) / total_counts.get("baseline", 1)
        ),
        "both_common_support_fraction": float(
            retained_counts.get("both", 0) / total_counts.get("both", 1)
        ),
    }


def bootstrap_interval(frame: pd.DataFrame, repetitions: int, rng: np.random.Generator) -> tuple[float, float]:
    seeds = np.array(sorted(frame["seed"].unique()))
    strata = np.array(sorted(frame["stratum"].unique()))
    conditions = np.array(["baseline", "both"])
    seed_index = {value: index for index, value in enumerate(seeds)}
    condition_index = {value: index for index, value in enumerate(conditions)}
    stratum_index = {value: index for index, value in enumerate(strata)}
    totals = np.zeros((len(seeds), 2, len(strata)), dtype=float)
    engaged = np.zeros_like(totals)
    grouped = frame.groupby(["seed", "condition", "stratum"])["engaged"].agg(["size", "sum"])
    for (seed, condition, stratum), row in grouped.iterrows():
        index = (
            seed_index[int(seed)],
            condition_index[str(condition)],
            stratum_index[str(stratum)],
        )
        totals[index] = float(row["size"])
        engaged[index] = float(row["sum"])
    values: list[float] = []
    for _ in range(repetitions):
        weights = np.bincount(
            rng.integers(0, len(seeds), size=len(seeds)), minlength=len(seeds)
        )
        boot_totals = np.tensordot(weights, totals, axes=(0, 0))
        boot_engaged = np.tensordot(weights, engaged, axes=(0, 0))
        common = (boot_totals[0] > 0) & (boot_totals[1] > 0)
        if not common.any():
            continue
        supported_totals = boot_totals[:, common]
        supported_engaged = boot_engaged[:, common]
        mix = supported_totals / supported_totals.sum(axis=1, keepdims=True)
        target_mix = mix.mean(axis=0)
        rates = supported_engaged / supported_totals
        standardized = (rates * target_mix).sum(axis=1)
        values.append(float(standardized[1] - standardized[0]))
    if not values:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        "--archive",
        dest="source",
        type=Path,
        required=True,
        help="Extracted result directory or .tar.gz result archive.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument(
        "--strata-fields",
        nargs="+",
        default=["role", "abm_reason_type", "partner_role"],
        help="Opportunity fields to match exactly before comparing engagement.",
    )
    args = parser.parse_args()

    decisions = read_decisions(args.source, args.strata_fields)
    rng = np.random.default_rng(20260824)
    rows = []
    for (scenario, persona_id), group in decisions.groupby(
        ["scenario", "persona_id"], sort=True
    ):
        result = standardized_contrast(group)
        low, high = bootstrap_interval(group, args.bootstrap_repetitions, rng)
        rows.append(
            {
                "scenario": scenario,
                "persona": PERSONA_LABELS.get(persona_id, persona_id),
                "baseline_opportunity_matched_engagement_percent": 100 * result["baseline_rate"],
                "both_opportunity_matched_engagement_percent": 100 * result["both_rate"],
                "matched_change_percentage_points": 100 * result["contrast"],
                "bootstrap_95_ci_low": 100 * low,
                "bootstrap_95_ci_high": 100 * high,
                "common_opportunity_strata": result["common_strata"],
                "baseline_sampled_opportunities": result["baseline_n"],
                "both_sampled_opportunities": result["both_n"],
                "baseline_common_support_percent": 100 * result["baseline_common_support_fraction"],
                "both_common_support_percent": 100 * result["both_common_support_fraction"],
            }
        )

    output = pd.DataFrame(rows).sort_values(["scenario", "persona"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "opportunity_matched_engagement.csv", index=False)
    summary = {
        "definition": (
            "Within each persona and workload, Baseline and Both were standardized "
            "to the equally weighted average of their common opportunity "
            f"distributions, with exact strata defined by {', '.join(args.strata_fields)}. "
            "Strata absent from either condition were excluded."
        ),
        "strata_fields": args.strata_fields,
        "bootstrap": (
            f"Paired seed bootstrap with {args.bootstrap_repetitions} repetitions; "
            "the ten seed identifiers were resampled with replacement."
        ),
        "rows": output.to_dict(orient="records"),
    }
    (args.output_dir / "opportunity_matched_engagement.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
