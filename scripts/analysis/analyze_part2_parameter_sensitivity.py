#!/usr/bin/env python3
"""Analyze Part 2 n20 parameter sensitivity, with optional parameter replacement."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / "outputs" / ".xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


EXPECTED_RUN_COUNT = 1_280
PRESSURE_THRESHOLD_RERUN_COUNT = 320
INVALID_CROWDING_PARAMETER = "SCENARIO_PRESSURE_STATION_SOCIAL_SUPPRESSION"
PRESSURE_THRESHOLD_PARAMETER = "SCENARIO_PRESSURE_ACTION_THRESHOLD"
SCENARIOS = ("normal_load", "high_load_high_acuity")
INTERVENTIONS = ("cockpit_only", "nursta_only", "both")
PARAMETER_LABELS = {
    "PERCEIVED_STAFF_INTERACTION_PROBABILITY": "Visibility response",
    INVALID_CROWDING_PARAMETER: "Invalid crowding suppression (excluded)",
    PRESSURE_THRESHOLD_PARAMETER: "Pressure activation threshold",
    "PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS": "Repeat-contact delay",
    "POST_TASK_STATION_CHECK_PROBABILITY_NURSE": "Station-return tendency",
}
PARAMETER_ORDER = (
    "PERCEIVED_STAFF_INTERACTION_PROBABILITY",
    PRESSURE_THRESHOLD_PARAMETER,
    "PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS",
    "POST_TASK_STATION_CHECK_PROBABILITY_NURSE",
)
LEVEL_LABELS = {
    ("PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS", "low"): "short",
    ("PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS", "high"): "long",
}
METRICS = {
    "f2f_per_hour": ("validation_metrics", "f2f_per_hour"),
    "raw_visible_staff_percepts": (
        "perception_reason_funnel_metrics",
        "raw_visible_staff_percepts_count",
    ),
    "eligible_visible_opportunities": (
        "perception_reason_funnel_metrics",
        "eligible_visible_staff_opportunity_count",
    ),
    "selected_visible_opportunities": (
        "perception_reason_funnel_metrics",
        "selected_visible_staff_opportunity_count",
    ),
    "approach_intents": ("perception_reason_funnel_metrics", "approach_intent_count"),
    "approach_successes": ("perception_reason_funnel_metrics", "approach_success_count"),
    "hcw_hcw_share": ("validation_metrics", "hcw_hcw_share"),
    "patient_facing_share": ("validation_metrics", "patient_facing_share"),
    "corridor_share": ("validation_metrics", "corridor_share"),
    "bedside_share": ("validation_metrics", "bedside_or_patient_room_share"),
    "ed_pressure_index_mean": ("operational_pressure_metrics", "ed_pressure_index_mean"),
}
REFERENCE_METRICS = {
    metric: f"{section}.{key}" for metric, (section, key) in METRICS.items()
}
CONCLUSIONS = (
    "COCPIT increases F2F/hour",
    "COCPIT increases visibility-driven contact",
    "COCPIT shifts toward HCW-HCW/corridor coordination",
    "Patient-facing/bedside share decreases under COCPIT",
    "NURSTA remains smaller/local",
    "Operational pressure remains modest",
)
VERDICT_ORDER = {"inconclusive": 0, "flipped": 1, "mixed": 2, "weakened": 3, "stable": 4}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _level_label(parameter: str, level: str) -> str:
    return LEVEL_LABELS.get((parameter, level), level)


def _load_runs(batch_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.rglob("summary.json")):
        summary = json.loads(path.read_text())
        metadata = summary.get("metadata", {})
        sensitivity = summary.get("sensitivity", {})
        parameter = str(
            metadata.get("sensitivity_parameter") or sensitivity.get("parameter") or ""
        )
        level = str(metadata.get("sensitivity_level") or sensitivity.get("level") or "")
        row = {
            "summary_path": str(path),
            "parameter_code": parameter,
            "parameter_label": PARAMETER_LABELS.get(parameter, parameter),
            "level": level,
            "level_label": _level_label(parameter, level),
            "sensitivity_default": _safe_float(
                metadata.get("sensitivity_default_value", sensitivity.get("default_value"))
            ),
            "sensitivity_value": _safe_float(
                metadata.get("sensitivity_applied_value", sensitivity.get("applied_value"))
            ),
            "scenario": metadata.get("scenario_mode"),
            "condition": metadata.get("condition"),
            "seed": int(metadata.get("seed", -1)),
        }
        for metric, (section, key) in METRICS.items():
            row[metric] = _safe_float(summary.get(section, {}).get(key))
        rows.append(row)
    return pd.DataFrame(rows)


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        value = float(values[0]) if len(values) else math.nan
        return value, value
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    multiplier = float(stats.t.ppf(0.975, len(values) - 1)) if stats is not None else 1.96
    mean = float(np.mean(values))
    return mean - multiplier * se, mean + multiplier * se


def _paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["parameter_code", "parameter_label", "level", "level_label", "scenario"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        parameter, parameter_label, level, level_label, scenario = keys
        default_value = float(group["sensitivity_default"].iloc[0])
        applied_value = float(group["sensitivity_value"].iloc[0])
        for intervention in INTERVENTIONS:
            for metric in METRICS:
                pivot = group[group["condition"].isin(["baseline", intervention])].pivot_table(
                    index="seed", columns="condition", values=metric, aggfunc="mean"
                )
                if "baseline" not in pivot or intervention not in pivot:
                    continue
                complete = pivot[["baseline", intervention]].dropna()
                if complete.empty:
                    continue
                baseline = complete["baseline"].to_numpy(dtype=float)
                alternative = complete[intervention].to_numpy(dtype=float)
                delta = alternative - baseline
                n = len(delta)
                sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
                se = sd / math.sqrt(n) if n > 1 else 0.0
                low, high = _ci95(delta)
                dz = float(np.mean(delta) / sd) if sd > 1e-12 else math.nan
                df = n - 1
                correction = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else math.nan
                gz = correction * dz if math.isfinite(correction) and math.isfinite(dz) else math.nan
                baseline_mean = float(np.mean(baseline))
                rows.append(
                    {
                        "parameter_code": parameter,
                        "parameter_label": parameter_label,
                        "level": level,
                        "level_label": level_label,
                        "scenario": scenario,
                        "condition": intervention,
                        "metric": metric,
                        "sensitivity_default": default_value,
                        "sensitivity_value": applied_value,
                        "paired_n": n,
                        "baseline_mean": baseline_mean,
                        "condition_mean": float(np.mean(alternative)),
                        "sensitivity_paired_delta": float(np.mean(delta)),
                        "sd_paired_delta": sd,
                        "se_paired_delta": se,
                        "ci95_low": low,
                        "ci95_high": high,
                        "positive_sign_consistency": float(np.mean(delta > 0)),
                        "negative_sign_consistency": float(np.mean(delta < 0)),
                        "percent_change_vs_baseline": (
                            float(np.mean(delta) / baseline_mean * 100.0)
                            if abs(baseline_mean) > 1e-12
                            else math.nan
                        ),
                        "cohens_dz": dz,
                        "bias_corrected_paired_gz": gz,
                    }
                )
    return pd.DataFrame(rows)


def _attach_default_reference(effects: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    lookup = {
        (str(row.scenario), str(row.intervention), str(row.metric)): float(row.mean_delta)
        for row in reference.itertuples(index=False)
    }
    result = effects.copy()
    result["default_paired_delta"] = math.nan
    result["effect_retention"] = math.nan
    result["sign_consistency"] = math.nan
    result["standardized_paired_effect"] = result["bias_corrected_paired_gz"]
    for index, row in result.iterrows():
        default_delta = lookup.get(
            (
                str(row["scenario"]),
                str(row["condition"]),
                REFERENCE_METRICS.get(str(row["metric"]), ""),
            ),
            math.nan,
        )
        result.at[index, "default_paired_delta"] = default_delta
        if math.isfinite(default_delta) and abs(default_delta) > 1e-12:
            result.at[index, "effect_retention"] = (
                float(row["sensitivity_paired_delta"]) / default_delta
            )
            result.at[index, "sign_consistency"] = (
                float(row["positive_sign_consistency"])
                if default_delta > 0
                else float(row["negative_sign_consistency"])
            )
    result["verdict"] = result.apply(_row_verdict, axis=1)
    return result


def _row_verdict(row: pd.Series) -> str:
    if row["metric"] == "ed_pressure_index_mean":
        percent = abs(float(row["percent_change_vs_baseline"]))
        overlaps_zero = float(row["ci95_low"]) <= 0 <= float(row["ci95_high"])
        if percent <= 5.0 and overlaps_zero:
            return "stable"
        return "weakened"
    default_delta = float(row["default_paired_delta"])
    delta = float(row["sensitivity_paired_delta"])
    retention = float(row["effect_retention"])
    consistency = float(row["sign_consistency"])
    if not all(math.isfinite(value) for value in (default_delta, delta, retention, consistency)):
        return "inconclusive"
    expected_sign = float(np.sign(default_delta))
    if expected_sign == 0:
        return "inconclusive"
    if delta * expected_sign < 0:
        return "flipped"
    ci_directional = (
        float(row["ci95_low"]) > 0 if expected_sign > 0 else float(row["ci95_high"]) < 0
    )
    if retention >= 0.5 and consistency >= 0.70 and ci_directional:
        return "stable"
    return "weakened"


def _combine_measure_verdicts(values: list[str]) -> str:
    if not values or "inconclusive" in values:
        return "inconclusive"
    if "flipped" in values:
        return "flipped"
    if all(value == "stable" for value in values):
        return "stable"
    return "weakened"


def _combine_scenarios(values: dict[str, str]) -> str:
    if set(values) != set(SCENARIOS) or "inconclusive" in values.values():
        return "inconclusive"
    unique = set(values.values())
    if "flipped" in unique:
        return "flipped"
    if unique == {"stable"}:
        return "stable"
    if unique == {"weakened"}:
        return "weakened"
    return "mixed"


def _metric_conclusion(
    group: pd.DataFrame,
    condition: str,
    metrics: tuple[str, ...],
) -> str:
    by_scenario: dict[str, str] = {}
    for scenario in SCENARIOS:
        subset = group[
            (group["scenario"] == scenario)
            & (group["condition"] == condition)
            & (group["metric"].isin(metrics))
        ]
        verdict_lookup = {str(row.metric): str(row.verdict) for row in subset.itertuples()}
        by_scenario[scenario] = _combine_measure_verdicts(
            [verdict_lookup.get(metric, "inconclusive") for metric in metrics]
        )
    return _combine_scenarios(by_scenario)


def _nursta_conclusion(group: pd.DataFrame) -> str:
    by_scenario: dict[str, str] = {}
    for scenario in SCENARIOS:
        subset = group[(group["scenario"] == scenario) & (group["metric"] == "f2f_per_hour")]
        lookup = {str(row.condition): float(row.sensitivity_paired_delta) for row in subset.itertuples()}
        if not {"cockpit_only", "nursta_only"}.issubset(lookup) or abs(lookup["cockpit_only"]) < 1e-12:
            by_scenario[scenario] = "inconclusive"
            continue
        ratio = abs(lookup["nursta_only"]) / abs(lookup["cockpit_only"])
        by_scenario[scenario] = "stable" if ratio <= 0.5 else "weakened" if ratio < 1.0 else "flipped"
    return _combine_scenarios(by_scenario)


def _operational_conclusion(group: pd.DataFrame) -> str:
    by_scenario: dict[str, str] = {}
    for scenario in SCENARIOS:
        subset = group[
            (group["scenario"] == scenario)
            & (group["metric"] == "ed_pressure_index_mean")
            & (group["condition"].isin(INTERVENTIONS))
        ]
        by_scenario[scenario] = _combine_measure_verdicts(
            [str(value) for value in subset["verdict"].tolist()]
            if len(subset) == len(INTERVENTIONS)
            else ["inconclusive"]
        )
    return _combine_scenarios(by_scenario)


def _verdict_matrix(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for parameter in PARAMETER_ORDER:
        for level in ("low", "high"):
            group = effects[
                (effects["parameter_code"] == parameter) & (effects["level"] == level)
            ]
            rows.append(
                {
                    "parameter_code": parameter,
                    "parameter_label": PARAMETER_LABELS[parameter],
                    "level": level,
                    "level_label": _level_label(parameter, level),
                    CONCLUSIONS[0]: _metric_conclusion(group, "cockpit_only", ("f2f_per_hour",)),
                    CONCLUSIONS[1]: _metric_conclusion(
                        group,
                        "cockpit_only",
                        ("eligible_visible_opportunities", "approach_successes"),
                    ),
                    CONCLUSIONS[2]: _metric_conclusion(
                        group, "cockpit_only", ("hcw_hcw_share", "corridor_share")
                    ),
                    CONCLUSIONS[3]: _metric_conclusion(
                        group, "cockpit_only", ("patient_facing_share", "bedside_share")
                    ),
                    CONCLUSIONS[4]: _nursta_conclusion(group),
                    CONCLUSIONS[5]: _operational_conclusion(group),
                }
            )
    return pd.DataFrame(rows)


def _plot_verdict_matrix(verdicts: pd.DataFrame, out_dir: Path) -> None:
    labels = [f"{row.parameter_label}\n{row.level_label}" for row in verdicts.itertuples()]
    matrix = np.array(
        [[VERDICT_ORDER.get(str(row[column]), 0) for column in CONCLUSIONS] for _, row in verdicts.iterrows()]
    )
    colors = ["#D9D7D2", "#B76E79", "#8B7895", "#C9B77D", "#6FA8A2"]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    ax.set_xticks(np.arange(len(CONCLUSIONS)), labels=CONCLUSIONS, rotation=30, ha="right", fontsize=8)
    for i, (_, row) in enumerate(verdicts.iterrows()):
        for j, column in enumerate(CONCLUSIONS):
            ax.text(j, i, str(row[column]), ha="center", va="center", fontsize=6)
    ax.tick_params(length=0)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 320} if extension == "png" else {}
        fig.savefig(out_dir / f"parameter_sensitivity_verdict_matrix.{extension}", **kwargs)
    plt.close(fig)


def _plot_f2f_supplement(effects: pd.DataFrame, out_dir: Path) -> None:
    colors = {"cockpit_only": "#6B7B8E", "nursta_only": "#8FBBAF", "both": "#74698C"}
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True)
    for column, parameter in enumerate(PARAMETER_ORDER):
        for row_index, scenario in enumerate(SCENARIOS):
            ax = axes[row_index, column]
            subset = effects[
                (effects["parameter_code"] == parameter)
                & (effects["scenario"] == scenario)
                & (effects["metric"] == "f2f_per_hour")
            ]
            for intervention in INTERVENTIONS:
                intervention_rows = subset[subset["condition"] == intervention]
                values = {
                    str(item.level): float(item.sensitivity_paired_delta)
                    for item in intervention_rows.itertuples()
                }
                default_values = intervention_rows["default_paired_delta"].dropna()
                if not default_values.empty:
                    values["default"] = float(default_values.iloc[0])
                if set(values) == {"low", "default", "high"}:
                    ax.plot(
                        [0, 1, 2],
                        [values["low"], values["default"], values["high"]],
                        marker="o",
                        linewidth=1.3,
                        markersize=3.5,
                        color=colors[intervention],
                        label=intervention,
                    )
            ax.axhline(0, color="#CFCFC8", linewidth=0.8)
            if row_index == 0:
                ax.set_title(PARAMETER_LABELS[parameter], fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{scenario}\nPaired delta F2F/hour", fontsize=8)
            ax.set_xticks([0, 1, 2], ["low", "default", "high"], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 320} if extension == "png" else {}
        fig.savefig(out_dir / f"supplement_delta_f2f_response.{extension}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--reference-effects",
        default="outputs/findings/part2_spatial_interventions_n100/tables/part2_headline_effects.csv",
    )
    parser.add_argument(
        "--pressure-threshold-rerun-dir",
        required=True,
        help=(
            "Verified 320-run pressure-threshold replacement. Invalid original "
            "crowding-suppression rows are removed before combining."
        ),
    )
    parser.add_argument("--with-f2f-supplement", action="store_true")
    args = parser.parse_args()
    reference_path = Path(args.reference_effects)
    if not reference_path.exists():
        raise SystemExit(f"Default n100 paired-effects reference not found: {reference_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_runs = _load_runs(Path(args.batch_dir))
    if len(original_runs) != EXPECTED_RUN_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_RUN_COUNT} original summaries, found {len(original_runs)}. "
            "Verify before analysis."
        )
    rerun_dir = Path(args.pressure_threshold_rerun_dir)
    replacement = _load_runs(rerun_dir)
    if len(replacement) != PRESSURE_THRESHOLD_RERUN_COUNT:
        raise SystemExit(
            f"Expected {PRESSURE_THRESHOLD_RERUN_COUNT} pressure-threshold summaries, "
            f"found {len(replacement)}."
        )
    parameters = set(replacement["parameter_code"].astype(str))
    if parameters != {PRESSURE_THRESHOLD_PARAMETER}:
        raise SystemExit(
            "Pressure-threshold rerun contains unexpected sensitivity parameters: "
            f"{sorted(parameters)}"
        )
    retained = original_runs[original_runs["parameter_code"] != INVALID_CROWDING_PARAMETER]
    runs = pd.concat([retained, replacement], ignore_index=True)
    if len(runs) != EXPECTED_RUN_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_RUN_COUNT} combined summaries, found {len(runs)}."
        )
    effects = _attach_default_reference(_paired_effects(runs), reference_path)
    verdicts = _verdict_matrix(effects)
    detailed_columns = [
        "parameter_code",
        "parameter_label",
        "level",
        "level_label",
        "scenario",
        "condition",
        "metric",
        "sensitivity_default",
        "sensitivity_value",
        "paired_n",
        "default_paired_delta",
        "sensitivity_paired_delta",
        "effect_retention",
        "ci95_low",
        "ci95_high",
        "sign_consistency",
        "standardized_paired_effect",
        "verdict",
    ]
    effects[detailed_columns].to_csv(out_dir / "parameter_sensitivity_detailed.csv", index=False)
    verdicts.to_csv(out_dir / "parameter_sensitivity_verdict_matrix.csv", index=False)
    _plot_verdict_matrix(verdicts, out_dir)
    if args.with_f2f_supplement:
        _plot_f2f_supplement(effects, out_dir)
    report = [
        "# Part 2 Parameter-Sensitivity Results",
        "",
        "Raw paired deltas are primary. Effect retention compares each sensitivity paired delta with the completed default-parameter n100 paired delta. Cohen's d_z and bias-corrected paired g_z are secondary.",
        "",
        "Operational pressure is judged separately: the absolute effect should remain small and its 95% CI should overlap zero; a ratio to a near-zero default effect is not meaningful.",
        "",
        f"- Runs analyzed: {len(runs)}",
        f"- Detailed effect rows: {len(effects)}",
        f"- Verdict rows: {len(verdicts)}",
        "- Verdicts: stable, weakened, flipped, mixed, inconclusive.",
    ]
    (out_dir / "part2_parameter_sensitivity_report.md").write_text("\n".join(report) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
