#!/usr/bin/env python3
"""Build the final Part 3 figures, tables, and persona-explorer data.

The script reads completed Part 3 archives or extracted result directories.
It never imports the simulation engine, runs an ABM, or performs LLM inference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import itertools
import json
import math
import os
import re
from statistics import NormalDist
import tarfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_RESULTS_DIR = PROJECT_DIR / "outputs" / "source_results"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/abm-part3-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/abm-part3-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Rectangle,
)
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_MAIN_SOURCE = SOURCE_RESULTS_DIR / "part3_closed_loop_main_n10_24719095.tar.gz"
DEFAULT_APPRAISAL_SOURCE = SOURCE_RESULTS_DIR / "part3_appraisals_final_24789077.tar.gz"
DEFAULT_ABLATION_SOURCE = SOURCE_RESULTS_DIR / "part3_architecture_ablation_v2_24817925.tar.gz"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "findings" / "part3_cognitive_personas_n10"
DEFAULT_STUDY_FIGURES_DIR = (
    PROJECT_DIR / "outputs" / "findings" / "study_overview" / "figures"
)
DEFAULT_EXPLORER_DATA = (
    PROJECT_DIR / "web" / "persona-explorer" / "data" / "persona-results.json"
)
DEFAULT_REVIEW_DIR = (
    PROJECT_DIR
    / "outputs"
    / "findings"
    / "part3_cognitive_personas_n10"
    / "review"
)
DEFAULT_REVIEWED_INTERVIEWS = DEFAULT_REVIEW_DIR / "reviewed_interview_sample.csv"
DEFAULT_REVIEWED_COUNTERFACTUALS = (
    DEFAULT_REVIEW_DIR / "reviewed_counterfactual_codes.csv"
)

SCENARIOS = ("normal_load", "high_load_high_acuity")
CONDITIONS = ("baseline", "cockpit_only", "nursta_only", "both")
INTERVENTIONS = ("cockpit_only", "nursta_only", "both")
PERSONAS = (
    "team_connector",
    "focus_protector",
    "patient_advocate",
    "vigilant_monitor",
    "adaptive_generalist",
)

SCENARIO_LABELS = {
    "normal_load": "Normal load",
    "high_load_high_acuity": "High load",
}
EXPLORER_SCENARIOS = {
    "normal_load": "normal_load",
    "high_load_high_acuity": "high_load",
}
CONDITION_LABELS = {
    "baseline": "Baseline",
    "cockpit_only": "COCPIT only",
    "nursta_only": "NURSTA only",
    "both": "Both",
}
CONTRAST_LABELS = {
    "cockpit_only": "COCPIT only − Baseline",
    "nursta_only": "NURSTA only − Baseline",
    "both": "Both − Baseline",
}
PERSONA_LABELS = {
    "team_connector": "Team Connector",
    "focus_protector": "Focus Protector",
    "patient_advocate": "Patient Advocate",
    "vigilant_monitor": "Vigilant Monitor",
    "adaptive_generalist": "Adaptive Generalist",
}
PERSONA_SLUGS = {
    "team_connector": "team-connector",
    "focus_protector": "focus-protector",
    "patient_advocate": "patient-advocate",
    "vigilant_monitor": "vigilant-monitor",
    "adaptive_generalist": "adaptive-generalist",
}
PERSONA_COLORS = {
    "team_connector": "#0B9F9A",
    "focus_protector": "#294A78",
    "patient_advocate": "#D76F5E",
    "vigilant_monitor": "#8C3F4A",
    "adaptive_generalist": "#718052",
}
COLORS = {
    "ink": "#24252A",
    "muted": "#737B87",
    "line": "#D9D7D2",
    "paper": "#FFFFFF",
    "baseline": "#AAA9A4",
    "observed": "#6FA8A2",
    "nursta": "#6B7B8E",
    "both": "#6FA8A2",
    "combined": "#74698C",
    "soft": "#F5F5F2",
}
OCEAN_LABELS = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}
SCORE_DIMENSIONS = {
    "overall_experience": "overall_person_space_fit",
    "coordination_support": "team_awareness",
    "focus_support": "task_continuity",
    "spatial_legibility": "spatial_legibility",
}
class ResultSource:
    """Read named files from an extracted result tree or .tar.gz archive."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._tar: tarfile.TarFile | None = None
        self._members: list[tarfile.TarInfo] | None = None
        self._cache: dict[str, bytes] = {}

    def __enter__(self) -> "ResultSource":
        if self.path.is_file():
            self._tar = tarfile.open(self.path, "r:gz")
            self._members = self._tar.getmembers()
        elif not self.path.is_dir():
            raise FileNotFoundError(self.path)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._tar is not None:
            self._tar.close()

    def _matching_paths(self, suffix: str) -> list[Any]:
        normalized = suffix.lstrip("/")
        if self.path.is_dir():
            matches = [
                path
                for path in self.path.rglob(Path(normalized).name)
                if path.as_posix().endswith(normalized)
            ]
        else:
            assert self._members is not None
            matches = [
                member
                for member in self._members
                if member.isfile() and member.name.endswith(normalized)
            ]
        return sorted(matches, key=lambda item: str(item))

    def read_bytes(self, suffix: str) -> bytes:
        if suffix in self._cache:
            return self._cache[suffix]
        matches = self._matching_paths(suffix)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one result file ending in {suffix!r} below {self.path}; "
                f"found {len(matches)}"
            )
        if self.path.is_dir():
            payload = matches[0].read_bytes()
        else:
            assert self._tar is not None
            handle = self._tar.extractfile(matches[0])
            if handle is None:
                raise ValueError(f"Could not read {matches[0].name}")
            payload = handle.read()
        self._cache[suffix] = payload
        return payload

    def read_csv(self, suffix: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(suffix)))

    def read_json(self, suffix: str) -> dict[str, Any]:
        return json.loads(self.read_bytes(suffix))

    def iter_jsonl_files(self, suffix: str) -> Iterator[dict[str, Any]]:
        matches = self._matching_paths(suffix)
        if not matches:
            raise ValueError(f"No result files ending in {suffix!r} below {self.path}")
        for match in matches:
            if self.path.is_dir():
                handle: Iterable[bytes] = match.open("rb")
            else:
                assert self._tar is not None
                extracted = self._tar.extractfile(match)
                if extracted is None:
                    raise ValueError(f"Could not read {match.name}")
                handle = extracted
            try:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            finally:
                if hasattr(handle, "close"):
                    handle.close()

    def iter_jsonl_groups(
        self, suffix: str
    ) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        """Yield each matching JSONL file separately with its source name."""

        if self.path.is_file():
            matched = 0
            # Stream the compressed archive once. Random seeks within a .tar.gz
            # repeatedly decompress earlier members and become prohibitively slow.
            with tarfile.open(self.path, "r|gz") as stream:
                for member in stream:
                    if not member.isfile() or not member.name.endswith(suffix):
                        continue
                    extracted = stream.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"Could not read {member.name}")
                    rows = [
                        json.loads(line) for line in extracted if line.strip()
                    ]
                    matched += 1
                    yield member.name, rows
            if not matched:
                raise ValueError(
                    f"No result files ending in {suffix!r} below {self.path}"
                )
            return

        matches = self._matching_paths(suffix)
        if not matches:
            raise ValueError(f"No result files ending in {suffix!r} below {self.path}")
        for match in matches:
            name = match.as_posix()
            handle: Iterable[bytes] = match.open("rb")
            try:
                rows = [json.loads(line) for line in handle if line.strip()]
            finally:
                if hasattr(handle, "close"):
                    handle.close()
            yield name, rows


def _configure_style() -> str:
    family = "DejaVu Sans"
    for candidate in ("Helvetica", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family, "Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": COLORS["paper"],
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
        }
    )
    return family


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def _despine(axis: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(left)
    axis.spines["bottom"].set_visible(bottom)
    if not left:
        axis.tick_params(axis="y", left=False)
    if not bottom:
        axis.tick_params(axis="x", bottom=False)


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return math.nan
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def _paired_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        raise ValueError("Paired summary requires at least two finite observations")
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    half = _student_t_critical_975(len(array) - 1) * se
    observed = abs(mean)
    permutations = (
        abs(float(np.mean(array * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    )
    exact_p = sum(value >= observed - 1e-12 for value in permutations) / (2 ** len(array))
    return {
        "n": float(len(array)),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "exact_sign_flip_p": exact_p,
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - half, center + half


def _format_effect(
    mean: float,
    low: float,
    high: float,
    *,
    scale: float = 1.0,
    decimals: int = 1,
) -> str:
    threshold = 0.5 * 10 ** (-decimals)

    def normalized(value: float) -> float:
        return 0.0 if abs(value * scale) < threshold else value * scale

    return (
        f"{normalized(mean):+.{decimals}f} "
        f"[{normalized(low):.{decimals}f}, {normalized(high):.{decimals}f}]"
    )


def _format_p(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def _markdown_table(
    frame: pd.DataFrame,
    *,
    bold_columns: Sequence[str] = (),
    bold_cells: set[tuple[int, str]] | None = None,
) -> str:
    columns = [str(column) for column in frame.columns]
    emphasized = set(bold_columns)
    cells = bold_cells or set()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        values = [
            f"**{value}**"
            if column in emphasized or (row_index, column) in cells
            else value
            for column, value in zip(columns, values)
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    ascii_replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in ascii_replacements.items():
        text = text.replace(old, new)
    unicode_replacements = {
        "Δ": r"\ensuremath{\Delta}",
        "→": r"\ensuremath{\rightarrow}",
        "−": r"\ensuremath{-}",
        "×": r"\ensuremath{\times}",
        "²": r"\textsuperscript{2}",
        "–": "--",
        "—": "---",
        "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}",
    }
    for old, new in unicode_replacements.items():
        text = text.replace(old, new)
    return text


def _latex_table(
    frame: pd.DataFrame,
    *,
    bold_columns: Sequence[str] = (),
    bold_cells: set[tuple[int, str]] | None = None,
) -> str:
    columns = [str(column) for column in frame.columns]
    emphasized = set(bold_columns)
    cells = bold_cells or set()
    rows = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(r"\textbf{" + _latex_escape(column) + "}" for column in columns)
        + r" \\",
        r"\midrule",
    ]
    previous_scenario: str | None = None
    scenario_index = columns.index("Scenario") if "Scenario" in columns else None
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        if scenario_index is not None:
            scenario = str(row[scenario_index])
            if previous_scenario is not None and scenario != previous_scenario:
                rows.append(r"\addlinespace[2pt]")
            previous_scenario = scenario
        values = []
        for column, value in zip(columns, row):
            escaped = _latex_escape(value)
            values.append(
                r"\textbf{" + escaped + "}"
                if column in emphasized or (row_index, column) in cells
                else escaped
            )
        rows.append(" & ".join(values) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows)


def _write_table(
    frame: pd.DataFrame,
    base_path: Path,
    *,
    note: str,
    bold_columns: Sequence[str] = (),
    bold_cells: set[tuple[int, str]] | None = None,
    latex: bool = True,
    markdown: bool = True,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base_path.with_suffix(".csv"), index=False)
    if markdown:
        text = _markdown_table(
            frame,
            bold_columns=bold_columns,
            bold_cells=bold_cells,
        )
        base_path.with_suffix(".md").write_text(text + f"\n*Note.* {note}\n")
    if latex:
        text = _latex_table(
            frame,
            bold_columns=bold_columns,
            bold_cells=bold_cells,
        )
        base_path.with_suffix(".tex").write_text(
            text
            + "\n\\par\\footnotesize\\textit{Note.} "
            + _latex_escape(note)
            + "\n"
        )


def _validate_sources(
    main_summary: Mapping[str, Any],
    appraisal_summary: Mapping[str, Any],
    ablation_summary: Mapping[str, Any],
) -> None:
    if main_summary.get("analysis_pass") is not True or main_summary.get("run_count") != 400:
        raise ValueError("The accepted 400-run Part 3 main analysis is not complete")
    required_dimensions = {
        "team_awareness",
        "task_continuity",
        "spatial_legibility",
        "overall_person_space_fit",
    }
    complete = {
        key
        for key, value in appraisal_summary.get("dimension_completeness", {}).items()
        if value.get("status") == "fully_estimable"
    }
    if appraisal_summary.get("analysis_pass") is not True or not required_dimensions <= complete:
        raise ValueError("Required Part 3 appraisal dimensions are not fully estimable")
    if ablation_summary.get("analysis_pass") is not True:
        raise ValueError("The matched Part 3 architecture ablation is not complete")


def _load_template(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    expected = set(PERSONAS)
    observed = {str(persona["id"]) for persona in data["personas"]}
    if observed != expected:
        raise ValueError(f"Persona template mismatch: {sorted(observed)}")
    return data


def _soft_color(hex_color: str, weight: float = 0.14) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(hex_color))
    return tuple((1.0 - weight) * np.ones(3) + weight * rgb)


def _load_persona_sprite(persona_id: str) -> Image.Image:
    sprite_path = (
        PROJECT_DIR
        / "web"
        / "persona-explorer"
        / "assets"
        / "personas"
        / f"{PERSONA_SLUGS[persona_id]}.png"
    )
    sprite = Image.open(sprite_path).convert("RGBA")
    if persona_id == "focus_protector":
        # The generated source sheet contains a detached fragment at the right edge.
        sprite = sprite.crop((0, 0, 313, sprite.height))
    return sprite.crop(sprite.getbbox())


def _plot_persona_atlas(template: Mapping[str, Any], figures_dir: Path) -> None:
    persona_lookup = {str(item["id"]): item for item in template["personas"]}
    fig = plt.figure(figsize=(11.8, 4.95))
    grid = fig.add_gridspec(
        1,
        5,
        left=0.035,
        right=0.985,
        top=0.975,
        bottom=0.035,
        wspace=0.06,
    )
    for index, persona_id in enumerate(PERSONAS):
        persona = persona_lookup[persona_id]
        accent = PERSONA_COLORS[persona_id]
        axis = fig.add_subplot(grid[0, index])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                facecolor=_soft_color(accent, 0.08),
                edgecolor=COLORS["line"],
                linewidth=0.8,
                zorder=0,
            )
        )
        axis.add_patch(
            Rectangle(
                (0.0, 0.985),
                1.0,
                0.015,
                facecolor=accent,
                edgecolor="none",
                zorder=1,
            )
        )
        crop = _load_persona_sprite(persona_id)
        sprite_bounds = (
            (0.0, 0.44, 1.0, 0.51)
            if persona_id == "team_connector"
            else (0.08, 0.44, 0.84, 0.51)
        )
        sprite_axis = axis.inset_axes(sprite_bounds)
        sprite_axis.imshow(crop, interpolation="none")
        sprite_axis.axis("off")
        axis.text(
            0.5,
            0.39,
            PERSONA_LABELS[persona_id],
            ha="center",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ocean = persona["ocean"]
        y_values = np.linspace(0.285, 0.075, 5)
        for y, (dimension, initial) in zip(y_values, OCEAN_LABELS.items()):
            value = float(ocean[dimension])
            axis.text(
                0.12,
                y,
                initial,
                color=accent,
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
            )
            axis.plot(
                [0.20, 0.82],
                [y, y],
                color="#D6D6D2",
                linewidth=3.4,
                solid_capstyle="butt",
                zorder=1,
            )
            axis.plot(
                [0.20, 0.20 + 0.62 * value / 100.0],
                [y, y],
                color=accent,
                linewidth=3.4,
                solid_capstyle="butt",
                zorder=2,
            )
            axis.text(
                0.94,
                y,
                f"{int(value)}",
                color=COLORS["muted"],
                ha="right",
                va="center",
                fontsize=6.3,
            )
    _save_figure(fig, figures_dir / "persona_explorer_preview")


def _diagram_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    detail: str = "",
    *,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#3F4145",
    heading_color: str = "#24252A",
    detail_color: str = "#24252A",
    linewidth: float = 0.95,
    radius: float = 0.009,
    heading_size: float = 8.6,
    detail_size: float = 7.0,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + 0.04 * width,
        y + (0.76 if detail else 0.50) * height,
        heading,
        ha="left",
        va="center",
        fontsize=heading_size,
        fontweight="heavy",
        color=heading_color,
        path_effects=[
            path_effects.withStroke(linewidth=0.28, foreground=heading_color)
        ],
        zorder=zorder + 1,
    )
    if detail:
        inner_x = x + 0.04 * width
        inner_y = y + 0.08 * height
        inner_width = 0.92 * width
        inner_height = 0.50 * height
        axis.add_patch(
            FancyBboxPatch(
                (inner_x, inner_y),
                inner_width,
                inner_height,
                boxstyle=f"round,pad=0,rounding_size={0.45 * radius}",
                facecolor="#FFFFFF",
                edgecolor=edgecolor,
                linewidth=0.72,
                zorder=zorder + 1,
            )
        )
        axis.text(
            inner_x + 0.04 * inner_width,
            inner_y + 0.50 * inner_height,
            detail,
            ha="left",
            va="center",
            fontsize=detail_size,
            color=detail_color,
            linespacing=1.22,
            zorder=zorder + 2,
        )
    return patch


def _diagram_inner_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    detail: str,
    *,
    edgecolor: str = "#3F4145",
    heading_size: float = 7.2,
    detail_size: float = 6.1,
    radius: float = 0.004,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor="#FFFFFF",
        edgecolor=edgecolor,
        linewidth=0.72,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + 0.04 * width,
        y + 0.68 * height,
        heading,
        ha="left",
        va="center",
        fontsize=heading_size,
        fontweight="heavy",
        color=COLORS["ink"],
        path_effects=[
            path_effects.withStroke(linewidth=0.24, foreground=COLORS["ink"])
        ],
        zorder=zorder + 1,
    )
    axis.text(
        x + 0.04 * width,
        y + 0.28 * height,
        detail,
        ha="left",
        va="center",
        fontsize=detail_size,
        color=COLORS["ink"],
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def _diagram_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#3F4145",
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    linewidth: float = 0.95,
    mutation_scale: float = 8.0,
    shrink_a: float = 3.5,
    shrink_b: float = 6.0,
    zorder: int = 4,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
            connectionstyle=connectionstyle,
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            zorder=zorder,
        )
    )


def _plot_study_overview(study_figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.2, 3.55))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    cards = (
        (
            0.025,
            "Validate",
            "Does the model reproduce\nobserved interaction ecology?",
            "#F2F2EF",
            COLORS["observed"],
        ),
        (
            0.3575,
            "Intervene",
            "What changes when spatial\ndesign features change?",
            "#E3F1EF",
            "#297C76",
        ),
        (
            0.690,
            "Interpret",
            "Who takes up the new\naffordances, and how?",
            "#ECE9F2",
            COLORS["combined"],
        ),
    )
    card_y = 0.15
    card_height = 0.69
    card_width = 0.285
    for index, (x, heading, question, fill, accent) in enumerate(cards):
        axis.add_patch(
            FancyBboxPatch(
                (x, card_y),
                card_width,
                card_height,
                boxstyle="round,pad=0,rounding_size=0.015",
                facecolor=fill,
                edgecolor="#4A4C50",
                linewidth=1.05,
            )
        )
        axis.text(
            x + 0.025,
            0.775,
            heading,
            fontsize=14.0,
            fontweight="heavy",
            color=COLORS["ink"],
            path_effects=[
                path_effects.withStroke(linewidth=0.34, foreground=COLORS["ink"])
            ],
            ha="left",
            va="center",
        )
        axis.text(
            x + 0.025,
            0.645,
            question,
            fontsize=8.2,
            color=COLORS["ink"],
            ha="left",
            va="center",
            linespacing=1.25,
        )
        if index == 0:
            rng = np.random.default_rng(17)
            theta = rng.uniform(0.0, 2.0 * np.pi, size=48)
            radius = np.sqrt(rng.uniform(0.0, 1.0, size=48))
            base_points = np.column_stack(
                (
                    x + 0.1425 + 0.086 * radius * np.cos(theta),
                    0.385 + 0.092 * radius * np.sin(theta),
                )
            )
            for color, noise in (
                ("#9FA3A8", 0.0060),
                (accent, 0.0045),
            ):
                points = base_points + rng.normal(scale=(noise, noise), size=(48, 2))
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=10,
                    color=color,
                    alpha=0.68,
                    edgecolors="none",
                )
            axis.text(
                x + 0.1425,
                0.205,
                "Observed vs. simulated",
                fontsize=7.4,
                fontweight="bold",
                color=COLORS["ink"],
                ha="center",
            )
        elif index == 1:
            panel_y = 0.315
            panel_w = 0.092
            panel_h = 0.155
            left_x = x + 0.035
            right_x = x + 0.158
            for panel_x, label in ((left_x, "Before"), (right_x, "After")):
                axis.add_patch(
                    Rectangle(
                        (panel_x, panel_y),
                        panel_w,
                        panel_h,
                        facecolor="#FFFFFF",
                        edgecolor="#4A4C50",
                        linewidth=0.85,
                    )
                )
                axis.text(
                    panel_x + panel_w / 2,
                    panel_y - 0.019,
                    label,
                    fontsize=5.8,
                    color=COLORS["ink"],
                    ha="center",
                    va="top",
                )
            for panel_x, partition_color, partition_alpha in (
                (left_x, "#4F5256", 1.0),
                (right_x, COLORS["observed"], 0.26),
            ):
                axis.plot(
                    [panel_x + panel_w / 2, panel_x + panel_w / 2],
                    [panel_y + 0.018, panel_y + panel_h - 0.018],
                    color=partition_color,
                    linewidth=2.1,
                    alpha=partition_alpha,
                )
                for eye_index, eye_x in enumerate(
                    (panel_x + 0.025, panel_x + panel_w - 0.025)
                ):
                    inward = 0.0
                    if panel_x == right_x:
                        inward = 0.0042 if eye_index == 0 else -0.0042
                    axis.add_patch(
                        Ellipse(
                            (eye_x, panel_y + 0.091),
                            width=0.030,
                            height=0.022,
                            facecolor="#FFFFFF",
                            edgecolor=COLORS["ink"],
                            linewidth=1.0,
                            zorder=3,
                        )
                    )
                    axis.add_patch(
                        Circle(
                            (eye_x + inward, panel_y + 0.091),
                            0.0061,
                            facecolor=COLORS["observed"],
                            edgecolor=COLORS["ink"],
                            linewidth=0.55,
                            zorder=4,
                        )
                    )
                    axis.add_patch(
                        Circle(
                            (eye_x + inward, panel_y + 0.091),
                            0.0024,
                            facecolor=COLORS["ink"],
                            edgecolor="none",
                            zorder=5,
                        )
                    )
                    if panel_x == right_x:
                        target_x = panel_x + panel_w / 2
                        axis.plot(
                            [eye_x + inward, target_x],
                            [panel_y + 0.091, panel_y + 0.091],
                            color=COLORS["ink"],
                            linewidth=0.55,
                            alpha=0.55,
                            zorder=2,
                        )
            _diagram_arrow(
                axis,
                (left_x + panel_w, panel_y + panel_h / 2),
                (right_x, panel_y + panel_h / 2),
                color=COLORS["ink"],
                linewidth=0.9,
                mutation_scale=7.0,
            )
            axis.text(
                x + 0.1425,
                0.205,
                "Space shapes behavior",
                fontsize=7.4,
                fontweight="bold",
                color=COLORS["ink"],
                ha="center",
            )
        else:
            sprite_width = 0.051
            sprite_height = 0.185
            gap = 0.003
            sprite_span = len(PERSONAS) * sprite_width + (len(PERSONAS) - 1) * gap
            start_x = x + (card_width - sprite_span) / 2.0
            for sprite_index, persona in enumerate(PERSONAS):
                sprite_axis = axis.inset_axes(
                    (
                        start_x + sprite_index * (sprite_width + gap),
                        0.31,
                        sprite_width,
                        sprite_height,
                    ),
                    transform=axis.transData,
                )
                sprite_axis.imshow(
                    _load_persona_sprite(persona),
                    interpolation="none",
                )
                sprite_axis.axis("off")
            axis.text(
                x + 0.1425,
                0.205,
                "Exposure, uptake, and fit",
                fontsize=7.4,
                fontweight="bold",
                color=COLORS["ink"],
                ha="center",
            )
    _diagram_arrow(axis, (0.310, 0.50), (0.3575, 0.50), linewidth=1.1)
    _diagram_arrow(axis, (0.6425, 0.50), (0.690, 0.50), linewidth=1.1)
    _save_figure(fig, study_figures_dir / "study_design_overview")


def _plot_interaction_pipeline(study_figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.2, 3.65))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.97, bottom=0.03)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    input_fill = "#F3F3F0"
    opportunity_fill = "#E3F1EF"
    policy_fill = "#E7EBF0"
    record_fill = "#ECE9F2"
    border = "#3F4145"
    flow_y = 0.49

    def stage_shell(
        x: float,
        y: float,
        width: float,
        height: float,
        heading: str,
        fill: str,
    ) -> float:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.009",
                facecolor=fill,
                edgecolor=border,
                linewidth=0.95,
                zorder=2,
            )
        )
        heading_y = y + height - 0.05
        axis.text(
            x + 0.04 * width,
            heading_y,
            heading,
            fontsize=9.0,
            fontweight="heavy",
            color=COLORS["ink"],
            path_effects=[
                path_effects.withStroke(linewidth=0.30, foreground=COLORS["ink"])
            ],
            ha="left",
            va="center",
            zorder=3,
        )
        return heading_y - 0.05

    prerequisite_x, prerequisite_y = 0.035, 0.2745
    prerequisite_w, prerequisite_h = 0.22, 0.431
    prerequisite_top = stage_shell(
        prerequisite_x,
        prerequisite_y,
        prerequisite_w,
        prerequisite_h,
        "Interaction prerequisites",
        input_fill,
    )
    prerequisite_rows = (
        ("Proximity", "Partners are close enough"),
        ("Mutual visibility", "Condition-specific sightline"),
        ("Operational feasibility", "Task, pressure, and repeat-contact gates"),
    )
    prerequisite_row_height = 0.085
    prerequisite_gap = 0.018
    for row_index, (heading, detail) in enumerate(prerequisite_rows):
        row_top = prerequisite_top - row_index * (
            prerequisite_row_height + prerequisite_gap
        )
        _diagram_inner_box(
            axis,
            prerequisite_x + 0.015,
            row_top - prerequisite_row_height,
            0.19,
            prerequisite_row_height,
            heading,
            detail,
            edgecolor=border,
            heading_size=7.1,
            detail_size=5.7,
        )
    _diagram_arrow(axis, (0.255, flow_y), (0.315, flow_y))

    def single_detail_stage(
        x: float,
        width: float,
        heading: str,
        detail: str,
        fill: str,
    ) -> None:
        stage_y, stage_height = 0.36, 0.26
        content_top = stage_shell(
            x,
            stage_y,
            width,
            stage_height,
            heading,
            fill,
        )
        inner_x = x + 0.04 * width
        inner_width = 0.92 * width
        inner_height = 0.13
        axis.add_patch(
            FancyBboxPatch(
                (inner_x, content_top - inner_height),
                inner_width,
                inner_height,
                boxstyle="round,pad=0,rounding_size=0.004",
                facecolor="#FFFFFF",
                edgecolor=border,
                linewidth=0.72,
                zorder=3,
            )
        )
        axis.text(
            inner_x + 0.04 * inner_width,
            content_top - inner_height / 2,
            detail,
            fontsize=6.5,
            color=COLORS["ink"],
            ha="left",
            va="center",
            linespacing=1.22,
            zorder=4,
        )

    single_detail_stage(
        0.315,
        0.19,
        "Actionable opportunity",
        "A feasible staff contact\nat a specific time and place",
        opportunity_fill,
    )
    _diagram_arrow(axis, (0.505, flow_y), (0.555, flow_y))
    single_detail_stage(
        0.555,
        0.17,
        "Bounded decision",
        "Engage, defer, or decline\nReason and topic",
        policy_fill,
    )
    _diagram_arrow(axis, (0.725, flow_y), (0.775, flow_y))

    outcome_x, outcome_y = 0.775, 0.28
    outcome_w, outcome_h = 0.20, 0.42
    outcome_top = stage_shell(
        outcome_x,
        outcome_y,
        outcome_w,
        outcome_h,
        "Recorded outcome",
        record_fill,
    )
    outcome_rows = (
        ("F2F interaction", "Time, coordinate, partners,\nreason, and topic"),
        ("Missed opportunity", "Time, coordinate, partners,\nand non-contact reason"),
    )
    outcome_row_height = 0.135
    outcome_gap = 0.020
    for row_index, (heading, detail) in enumerate(outcome_rows):
        row_top = outcome_top - row_index * (outcome_row_height + outcome_gap)
        _diagram_inner_box(
            axis,
            outcome_x + 0.017,
            row_top - outcome_row_height,
            0.166,
            outcome_row_height,
            heading,
            detail,
            edgecolor=border,
            heading_size=7.4,
            detail_size=5.8,
        )
    for x, label in (
        (0.145, "SPACE AND WORKFLOW"),
        (0.410, "OPPORTUNITY"),
        (0.640, "COGNITIVE OR RULE POLICY"),
        (0.875, "EVENT RECORD"),
    ):
        axis.text(
            x,
            0.785,
            label,
            fontsize=7.2,
            fontweight="heavy",
            color=COLORS["ink"],
            path_effects=[
                path_effects.withStroke(linewidth=0.30, foreground=COLORS["ink"])
            ],
            ha="center",
            va="center",
        )
    _save_figure(fig, study_figures_dir / "interaction_generation_pipeline")


def _plot_cognitive_architecture(figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.2, 5.2))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.02)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.55),
            0.96,
            0.36,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor="#FFFFFF",
            edgecolor="#777A7E",
            linewidth=0.85,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.36,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor="#FFFFFF",
            edgecolor="#777A7E",
            linewidth=0.85,
        )
    )
    axis.text(
        0.04,
        0.875,
        "DURING THE SHIFT",
        fontsize=7.2,
        fontweight="heavy",
        color=COLORS["ink"],
        path_effects=[
            path_effects.withStroke(linewidth=0.30, foreground=COLORS["ink"])
        ],
        ha="left",
    )
    _diagram_box(
        axis,
        0.04,
        0.645,
        0.17,
        0.155,
        "Feasible opportunity",
        "ABM supplies partner,\nplace, task, and urgency",
        facecolor="#F3F3F0",
        heading_size=8.4,
        detail_size=6.7,
    )
    _diagram_box(
        axis,
        0.27,
        0.645,
        0.19,
        0.155,
        "Cognitive context",
        "Orientation and recent\nrealized contacts",
        facecolor="#E3F1EF",
        heading_size=8.4,
        detail_size=6.7,
    )
    _diagram_box(
        axis,
        0.52,
        0.645,
        0.17,
        0.155,
        "Bounded LLM policy",
        "Action, reason, and topic\nwith cited evidence",
        facecolor="#E7EBF0",
        heading_size=8.5,
        detail_size=6.7,
    )
    _diagram_box(
        axis,
        0.76,
        0.645,
        0.20,
        0.155,
        "Realized outcome",
        "Interaction or missed opportunity\nis written to the trace",
        facecolor="#ECE9F2",
        heading_size=8.5,
        detail_size=6.7,
    )
    _diagram_arrow(axis, (0.21, 0.7225), (0.27, 0.7225))
    _diagram_arrow(axis, (0.46, 0.7225), (0.52, 0.7225))
    _diagram_arrow(axis, (0.69, 0.7225), (0.76, 0.7225))
    axis.plot(
        [0.86, 0.86, 0.365],
        [0.625, 0.590, 0.590],
        color="#4A4C50",
        linewidth=0.9,
        linestyle=(0, (2, 2)),
        solid_capstyle="butt",
        zorder=4,
    )
    _diagram_arrow(
        axis,
        (0.365, 0.590),
        (0.365, 0.645),
        color="#4A4C50",
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        mutation_scale=8.0,
        shrink_a=0.0,
        shrink_b=6.0,
    )
    axis.text(
        0.61,
        0.568,
        "Only realized contacts become causal memory",
        fontsize=6.6,
        color=COLORS["ink"],
        ha="center",
    )

    axis.text(
        0.04,
        0.375,
        "AFTER THE SHIFT",
        fontsize=7.2,
        fontweight="heavy",
        color=COLORS["ink"],
        path_effects=[
            path_effects.withStroke(linewidth=0.30, foreground=COLORS["ink"])
        ],
        ha="left",
    )
    axis.text(
        0.96,
        0.375,
        "Read-only: appraisal never alters the completed trajectory",
        fontsize=6.5,
        color=COLORS["ink"],
        ha="right",
    )
    _diagram_box(
        axis,
        0.04,
        0.14,
        0.20,
        0.155,
        "Episodic spatial experience",
        "Travel, visibility, tasks, contacts,\nmissed opportunities, and places",
        facecolor="#F3F3F0",
        heading_size=8.4,
        detail_size=6.7,
    )
    _diagram_box(
        axis,
        0.30,
        0.14,
        0.17,
        0.155,
        "Evidence retrieval",
        "Representative events and\nplace-linked summaries",
        facecolor="#E3F1EF",
        heading_size=8.4,
        detail_size=6.7,
    )
    _diagram_box(
        axis,
        0.53,
        0.14,
        0.20,
        0.155,
        "Appraisal and interview",
        "Seven structured scores\nand synthetic responses",
        facecolor="#E7EBF0",
        heading_size=8.4,
        detail_size=6.7,
    )
    _diagram_box(
        axis,
        0.79,
        0.14,
        0.17,
        0.155,
        "Claim layers",
        "Pattern and interpretation\nNeed and design hypothesis",
        facecolor="#ECE9F2",
        heading_size=8.4,
        detail_size=6.5,
    )
    _diagram_arrow(axis, (0.24, 0.2175), (0.30, 0.2175))
    _diagram_arrow(axis, (0.47, 0.2175), (0.53, 0.2175))
    _diagram_arrow(axis, (0.73, 0.2175), (0.79, 0.2175))
    _save_figure(fig, figures_dir / "figA_cognitive_architecture")


def _effect_row(
    frame: pd.DataFrame,
    *,
    scenario: str,
    persona: str,
    outcome: str,
) -> pd.Series:
    selected = frame[
        (frame["scenario"] == SCENARIO_LABELS[scenario])
        & (frame["persona"] == PERSONA_LABELS[persona])
        & (frame["contrast"] == "Both - Baseline")
        & (frame["outcome"] == outcome)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one effect row for {scenario}, {persona}, {outcome}; got {len(selected)}"
        )
    return selected.iloc[0]


def _plot_exposure_uptake(
    experience_effects: pd.DataFrame,
    persona_effects: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.23, wspace=0.18)
    x_positions = np.arange(len(PERSONAS), dtype=float)
    width = 0.32
    panel_specs = (
        (
            axes[0],
            experience_effects,
            "Share of time with a mutually visible colleague",
            "Time with a visible colleague",
        ),
        (
            axes[1],
            persona_effects,
            "Role-standardized model engagement-rate change",
            "LLM engagement rate",
        ),
    )
    for axis, source, outcome, panel_label in panel_specs:
        axis.axhline(0, color="#A6A6A2", linewidth=0.8, linestyle=(0, (2, 2)))
        for persona_index, persona in enumerate(PERSONAS):
            for scenario_index, scenario in enumerate(SCENARIOS):
                result = _effect_row(
                    source,
                    scenario=scenario,
                    persona=persona,
                    outcome=outcome,
                )
                mean = 100.0 * float(result["mean_delta"])
                low = 100.0 * float(result["ci95_low"])
                high = 100.0 * float(result["ci95_high"])
                x = x_positions[persona_index] + (-0.5 if scenario_index == 0 else 0.5) * width
                alpha = 0.96 if scenario_index == 0 else 0.44
                axis.bar(
                    x,
                    mean,
                    width=0.90 * width,
                    color=PERSONA_COLORS[persona],
                    alpha=alpha,
                    edgecolor=COLORS["ink"],
                    linewidth=0.7,
                    zorder=2,
                )
                axis.errorbar(
                    [x],
                    [mean],
                    yerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor=COLORS["ink"],
                    elinewidth=0.75,
                    capsize=2.0,
                    capthick=0.75,
                    zorder=3,
                )
        axis.text(
            0.025,
            0.96,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=COLORS["ink"],
        )
        axis.set_xticks(
            x_positions,
            [PERSONA_LABELS[item].replace(" ", "\n") for item in PERSONAS],
        )
        axis.tick_params(axis="x", length=0, pad=7)
        axis.set_ylabel("Change (percentage points)", labelpad=8)
        axis.grid(axis="y", color="#ECECE8", linewidth=0.65)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS["line"])
            spine.set_linewidth(0.85)
        axis.spines["top"].set_visible(False)
    axes[0].set_ylim(0.0, 9.3)
    axes[0].set_yticks([0, 2, 4, 6, 8])
    axes[1].set_ylim(-48.0, 32.0)
    axes[1].set_yticks([-40, -20, 0, 20])
    scenario_handles = [
        Patch(
            facecolor="#6F7479",
            edgecolor=COLORS["ink"],
            linewidth=0.7,
            alpha=0.96,
            label="Normal load",
        ),
        Patch(
            facecolor="#6F7479",
            edgecolor=COLORS["ink"],
            linewidth=0.7,
            alpha=0.44,
            label="High load",
        ),
    ]
    fig.legend(
        handles=scenario_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.055),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.55,
    )
    _save_figure(fig, figures_dir / "figB_common_affordance_different_uptake")


def _fit_rows(survey: pd.DataFrame) -> pd.DataFrame:
    selected = survey[
        (survey["dimension"] == "overall_person_space_fit")
        & (survey["rateability"] == "rateable")
    ].copy()
    selected["fit"] = pd.to_numeric(selected["score_1_to_7"], errors="raise")
    expected = len(SCENARIOS) * len(CONDITIONS) * len(PERSONAS) * 10
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} rateable fit rows; got {len(selected)}")
    return selected


def _ensemble_seed_summary(fit: pd.DataFrame) -> pd.DataFrame:
    persona_seed = (
        fit.groupby(["scenario", "condition", "seed", "persona_id"], as_index=False)["fit"]
        .mean()
    )
    records = []
    for keys, group in persona_seed.groupby(["scenario", "condition", "seed"]):
        values = group["fit"].to_numpy(dtype=float)
        if len(values) != len(PERSONAS):
            raise ValueError(f"Incomplete persona ensemble for {keys}")
        records.append(
            {
                "scenario": keys[0],
                "condition": keys[1],
                "seed": keys[2],
                "average_fit": float(np.mean(values)),
                "fit_floor": float(np.min(values)),
                "between_orientation_sd": float(np.std(values, ddof=1)),
                "between_orientation_range": float(np.max(values) - np.min(values)),
            }
        )
    return pd.DataFrame(records)


def _plot_fit_ensemble(
    fit: pd.DataFrame,
    ensemble: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.05), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.91, bottom=0.21, wspace=0.16)
    rng = np.random.default_rng(20260723)
    for axis, scenario in zip(axes, SCENARIOS):
        scenario_fit = fit[
            (fit["scenario"] == scenario) & fit["condition"].isin(("baseline", "both"))
        ]
        scenario_ensemble = ensemble[
            (ensemble["scenario"] == scenario)
            & ensemble["condition"].isin(("baseline", "both"))
        ]
        for persona_index, persona in enumerate(PERSONAS):
            persona_rows = scenario_fit[scenario_fit["persona_id"] == persona]
            means = []
            for x, condition in enumerate(("baseline", "both")):
                values = persona_rows[persona_rows["condition"] == condition]["fit"].to_numpy()
                jitter = rng.uniform(-0.024, 0.024, size=len(values))
                offset = (persona_index - 2) * 0.021
                axis.scatter(
                    np.full(len(values), x + offset) + jitter,
                    values,
                    s=13,
                    color=PERSONA_COLORS[persona],
                    alpha=0.18,
                    edgecolors="none",
                    zorder=1,
                )
                means.append(float(np.mean(values)))
            axis.plot(
                (0, 1),
                means,
                color=PERSONA_COLORS[persona],
                linewidth=1.4,
                alpha=0.9,
                zorder=2,
            )
            axis.scatter(
                (0, 1),
                means,
                s=54,
                color=PERSONA_COLORS[persona],
                edgecolor=COLORS["ink"],
                linewidth=0.75,
                zorder=3,
            )
        baseline = scenario_ensemble[
            scenario_ensemble["condition"] == "baseline"
        ].set_index("seed")
        both = scenario_ensemble[scenario_ensemble["condition"] == "both"].set_index(
            "seed"
        )
        average = _paired_summary(both["average_fit"] - baseline["average_fit"])
        floor = _paired_summary(both["fit_floor"] - baseline["fit_floor"])
        summary_positions = {"baseline": -0.115, "both": 1.115}
        for condition, x in summary_positions.items():
            rows = scenario_ensemble[scenario_ensemble["condition"] == condition]
            average_y = float(rows["average_fit"].mean())
            floor_y = float(rows["fit_floor"].mean())
            axis.scatter(
                [x],
                [average_y],
                s=52,
                marker="D",
                color=COLORS["ink"],
                edgecolor=COLORS["paper"],
                linewidth=0.6,
                zorder=4,
            )
            axis.scatter(
                [x],
                [floor_y],
                s=56,
                marker="v",
                color=COLORS["paper"],
                edgecolor=COLORS["ink"],
                linewidth=1.0,
                zorder=4,
            )
        axis.add_patch(
            FancyBboxPatch(
                (0.770, 0.0575),
                0.155,
                0.165,
                transform=axis.transAxes,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor="#FFFFFF",
                edgecolor=COLORS["line"],
                linewidth=0.8,
                zorder=5,
            )
        )
        axis.scatter(
            [0.805],
            [0.175],
            transform=axis.transAxes,
            s=34,
            marker="D",
            color=COLORS["ink"],
            edgecolor=COLORS["ink"],
            linewidth=0.6,
            zorder=6,
        )
        axis.scatter(
            [0.805],
            [0.105],
            transform=axis.transAxes,
            s=38,
            marker="v",
            color=COLORS["paper"],
            edgecolor=COLORS["ink"],
            linewidth=0.9,
            zorder=6,
        )
        axis.text(
            0.845,
            0.175,
            f"{average['mean']:+.2f}",
            transform=axis.transAxes,
            fontsize=7.3,
            fontweight="semibold",
            color=COLORS["ink"],
            ha="left",
            va="center",
            zorder=6,
        )
        axis.text(
            0.845,
            0.105,
            f"{floor['mean']:+.2f}",
            transform=axis.transAxes,
            fontsize=7.3,
            fontweight="semibold",
            color=COLORS["ink"],
            ha="left",
            va="center",
            zorder=6,
        )
        axis.set_title(
            SCENARIO_LABELS[scenario],
            loc="left",
            pad=12,
            fontsize=10.0,
            fontweight="bold",
        )
        axis.set_xlim(-0.27, 1.27)
        axis.set_ylim(1.0, 7.1)
        axis.set_xticks((0, 1), ("Baseline", "Both"))
        axis.set_yticks(range(1, 8))
        axis.grid(axis="y", color="#ECECE8", linewidth=0.65)
        axis.set_axisbelow(True)
        _despine(axis, left=True, bottom=True)
    axes[0].set_ylabel("Person–space fit (1–7)", labelpad=9)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=PERSONA_COLORS[item],
            markerfacecolor=PERSONA_COLORS[item],
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.6,
            linewidth=1.4,
            label=PERSONA_LABELS[item],
        )
        for item in PERSONAS
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=COLORS["ink"],
                markeredgecolor=COLORS["ink"],
                markersize=5.2,
                label="Average",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color="none",
                markerfacecolor=COLORS["paper"],
                markeredgecolor=COLORS["ink"],
                markeredgewidth=0.9,
                markersize=5.8,
                label="Fit floor",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.025),
        ncol=7,
        frameon=False,
        handletextpad=0.38,
        columnspacing=0.82,
    )
    _save_figure(fig, figures_dir / "figC_inclusive_person_space_fit")


def _find_effect(
    frame: pd.DataFrame,
    *,
    scenario: str,
    persona: str,
    outcome: str,
) -> pd.Series:
    selected = frame[
        (frame["scenario"] == SCENARIO_LABELS[scenario])
        & (frame["persona"] == PERSONA_LABELS[persona])
        & (frame["contrast"] == "Both - Baseline")
        & (frame["outcome"] == outcome)
    ]
    if len(selected) != 1:
        raise ValueError(f"Missing Part 3 effect: {scenario}, {persona}, {outcome}")
    return selected.iloc[0]


def _mean_appraisal_score(
    survey: pd.DataFrame,
    *,
    persona: str,
    condition: str,
    dimension: str,
) -> float:
    selected = survey[
        (survey["scenario"] == "high_load_high_acuity")
        & (survey["persona_id"] == persona)
        & (survey["condition"] == condition)
        & (survey["dimension"] == dimension)
        & (survey["rateability"] == "rateable")
    ]
    if len(selected) != 10:
        raise ValueError(
            f"Expected 10 appraisal scores for {persona}, {condition}, {dimension}; "
            f"got {len(selected)}"
        )
    return float(pd.to_numeric(selected["score_1_to_7"], errors="raise").mean())


def _score_transition(baseline: float, both: float) -> str:
    delta = both - baseline
    return f"{baseline:.1f} → {both:.1f} ({delta:+.1f})"


def _table_persona_appraisals(
    survey: pd.DataFrame,
    reviewed_interviews: pd.DataFrame,
) -> pd.DataFrame:
    dimensions = {
        "Overall fit": "overall_person_space_fit",
        "Coordination": "team_awareness",
        "Focus": "task_continuity",
        "Legibility": "spatial_legibility",
    }
    records = []
    for persona in PERSONAS:
        scores = {}
        for label, dimension in dimensions.items():
            baseline = _mean_appraisal_score(
                survey,
                persona=persona,
                condition="baseline",
                dimension=dimension,
            )
            both = _mean_appraisal_score(
                survey,
                persona=persona,
                condition="both",
                dimension=dimension,
            )
            scores[label] = _score_transition(baseline, both)
        candidates = reviewed_interviews[
            (reviewed_interviews["scenario"] == "high_load_high_acuity")
            & (reviewed_interviews["condition"] == "both")
            & (reviewed_interviews["persona_id"] == persona)
            & reviewed_interviews["include_in_explorer"].astype(bool)
        ].copy()
        if candidates.empty:
            response = "No grounded qualitative excerpt was retained."
        else:
            candidates["question_priority"] = (
                candidates["question_id"] != "counterfactual_change"
            ).astype(int)
            candidates = candidates.sort_values(
                ["question_priority", "explorer_display_order"]
            )
            response = _quote_excerpt(str(candidates.iloc[0]["answer"]))
        response_sentences = _sentences(response)
        records.append(
            {
                "Orientation": PERSONA_LABELS[persona],
                **scores,
                "Representative grounded response": (
                    response_sentences[0] if response_sentences else response
                ),
            }
        )
    return pd.DataFrame(records)


def _table_inclusive_fit(ensemble: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scenario in SCENARIOS:
        scenario_rows = ensemble[ensemble["scenario"] == scenario]
        baseline = scenario_rows[scenario_rows["condition"] == "baseline"].set_index("seed")
        for intervention in INTERVENTIONS:
            treatment = scenario_rows[
                scenario_rows["condition"] == intervention
            ].set_index("seed")
            summaries = {
                column: _paired_summary(treatment[column] - baseline[column])
                for column in (
                    "average_fit",
                    "fit_floor",
                    "between_orientation_sd",
                )
            }
            records.append(
                {
                    "Scenario": SCENARIO_LABELS[scenario],
                    "Contrast": CONTRAST_LABELS[intervention],
                    "Δ average fit [95% CI]": _format_effect(**{
                            "mean": summaries["average_fit"]["mean"],
                            "low": summaries["average_fit"]["ci95_low"],
                            "high": summaries["average_fit"]["ci95_high"],
                        }),
                    "Δ fit floor [95% CI]": _format_effect(**{
                            "mean": summaries["fit_floor"]["mean"],
                            "low": summaries["fit_floor"]["ci95_low"],
                            "high": summaries["fit_floor"]["ci95_high"],
                        }),
                    "Δ orientation SD [95% CI]": _format_effect(
                        **{
                            "mean": summaries["between_orientation_sd"]["mean"],
                            "low": summaries["between_orientation_sd"]["ci95_low"],
                            "high": summaries["between_orientation_sd"]["ci95_high"],
                        },
                        decimals=2,
                    ),
                }
            )
    return pd.DataFrame(records)


def _table_architecture_ablation_matrix(
    cells: pd.DataFrame,
    effects: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    grouped = cells.groupby("cell")["engage_indicator"]

    def estimate(cell: str) -> str:
        values = grouped.get_group(cell)
        successes = int(values.sum())
        total = int(len(values))
        low, high = _wilson_interval(successes, total)
        return (
            f"{100.0 * successes / total:.1f}% "
            f"[{100.0 * low:.1f}, {100.0 * high:.1f}]"
        )

    frame = pd.DataFrame(
        [
            {
                "Orientation input": "Generic",
                "No memory": estimate("neutral_no_memory"),
                "Grounded memory": estimate("neutral_orientation"),
            },
            {
                "Orientation input": "Persona-conditioned",
                "No memory": estimate("persona_no_memory"),
                "Grounded memory": estimate("full_persona_full_memory"),
            },
        ]
    )
    effect_labels = (
        ("orientation_main_effect", "persona conditioning"),
        ("memory_main_effect", "grounded memory"),
        ("orientation_memory_interaction", "orientation × memory"),
    )
    notes = []
    for effect, label in effect_labels:
        row = effects[
            (effects["scope"] == "overall") & (effects["effect"] == effect)
        ]
        if len(row) != 1:
            raise ValueError(f"Missing architecture effect: {effect}")
        result = row.iloc[0]
        notes.append(
            f"{label}: {100.0 * result['mean']:+.1f} pp "
            f"[{100.0 * result['ci95_low']:.1f}, "
            f"{100.0 * result['ci95_high']:.1f}]"
        )
    return frame, "; ".join(notes) + "."


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _sentences(value: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", _normalize_text(value))
        if sentence.strip()
    ]


def _quote_excerpt(answer: str) -> str:
    sentences = _sentences(answer)
    if not sentences:
        return ""
    selected = [sentences[0]]
    if len(sentences) > 1 and len(selected[0].split()) < 24:
        if len((selected[0] + " " + sentences[1]).split()) <= 52:
            selected.append(sentences[1])
    return " ".join(selected)


def _select_reviewed_quotes(
    reviewed_interviews: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], list[str]], pd.DataFrame]:
    selections: dict[tuple[str, str, str], list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    required = {
        "scenario",
        "condition",
        "persona_id",
        "pair_id",
        "question_id",
        "answer",
        "prompt_id",
        "review_grounding_pass",
        "review_claim_layer_pass",
        "include_in_explorer",
        "explorer_display_order",
        "reviewer_type",
        "human_review_completed",
    }
    missing = required - set(reviewed_interviews.columns)
    if missing:
        raise ValueError(f"Reviewed interview file is missing columns: {sorted(missing)}")
    if reviewed_interviews["human_review_completed"].astype(bool).any():
        raise ValueError("Codex audit must not be mislabeled as human review")
    for key, cell in reviewed_interviews.groupby(
        ["scenario", "condition", "persona_id"]
    ):
        selected = cell[cell["include_in_explorer"].astype(bool)].copy()
        selected = selected.sort_values("explorer_display_order")
        if not selected.empty and (
            not selected["review_grounding_pass"].astype(bool).all()
            or not selected["review_claim_layer_pass"].astype(bool).all()
        ):
            raise ValueError(f"Failed answer selected for explorer in {key}")
        explorer_key = (key[2], EXPLORER_SCENARIOS[key[0]], key[1])
        if selected.empty:
            placeholder = "No grounded qualitative excerpt was retained for this cell."
            selections[explorer_key] = [placeholder]
            audit_rows.append(
                {
                    "Scenario": SCENARIO_LABELS[key[0]],
                    "Condition": CONDITION_LABELS[key[1]],
                    "Orientation": PERSONA_LABELS[key[2]],
                    "Pair ID": "",
                    "Display order": 1,
                    "Question": "No retained excerpt",
                    "Exact excerpt": placeholder,
                    "Source prompt ID": "",
                    "Verbatim check": False,
                    "Review status": "no_grounded_quote_retained",
                    "Reviewer type": "independent_model_assisted_audit",
                    "Human review completed": False,
                }
            )
            continue
        excerpts = [_quote_excerpt(str(value)) for value in selected["answer"]]
        selections[explorer_key] = excerpts
        for excerpt, (_, row) in zip(excerpts, selected.iterrows()):
            source_answer = _normalize_text(row["answer"])
            if excerpt not in source_answer:
                raise ValueError(
                    f"Explorer excerpt is not verbatim for {row['pair_id']}, "
                    f"{row['question_id']}"
                )
            audit_rows.append(
                {
                    "Scenario": SCENARIO_LABELS[key[0]],
                    "Condition": CONDITION_LABELS[key[1]],
                    "Orientation": PERSONA_LABELS[key[2]],
                    "Pair ID": row["pair_id"],
                    "Display order": int(row["explorer_display_order"]),
                    "Question": str(row["question_id"]).replace("_", " ").title(),
                    "Exact excerpt": excerpt,
                    "Source prompt ID": row["prompt_id"],
                    "Verbatim check": True,
                    "Review status": "retained_after_independent_audit",
                    "Reviewer type": row["reviewer_type"],
                    "Human review completed": False,
                }
            )
    expected = len(SCENARIOS) * len(CONDITIONS) * len(PERSONAS)
    if len(selections) != expected:
        raise ValueError(f"Expected {expected} explorer quote bundles; got {len(selections)}")
    return selections, pd.DataFrame(audit_rows)


def _export_explorer(
    template: dict[str, Any],
    survey: pd.DataFrame,
    reviewed_interviews: pd.DataFrame,
    appraisal_summary: Mapping[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    quotes, quote_audit = _select_reviewed_quotes(reviewed_interviews)
    fit = _fit_rows(survey)
    score_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    paired_units: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in survey.itertuples(index=False):
        if row.rateability != "rateable" or pd.isna(row.score_1_to_7):
            continue
        scenario = EXPLORER_SCENARIOS[str(row.scenario)]
        base = (str(row.persona_id), scenario, str(row.condition))
        score_groups[(*base, str(row.dimension))].append(float(row.score_1_to_7))
        paired_units[base].add(str(row.pair_id))
    expected_keys = {
        (persona["id"], scenario["id"], condition["id"])
        for persona in template["personas"]
        for scenario in template["scenarios"]
        for condition in template["conditions"]
    }
    if set(quotes) != expected_keys:
        raise ValueError("Explorer quote coverage does not match its full design")
    results = []
    for persona_id, scenario, condition in sorted(expected_keys):
        base = (persona_id, scenario, condition)
        scores = {
            display: round(float(np.mean(score_groups[(*base, dimension)])), 1)
            for display, dimension in SCORE_DIMENSIONS.items()
        }
        results.append(
            {
                "persona": persona_id,
                "scenario": scenario,
                "condition": condition,
                "source": "verified_part3_appraisals",
                "sample_size": len(paired_units[base]),
                "scores": scores,
                "quotes": quotes[base],
            }
        )
    template["meta"] = {
        "status": "observed",
        "scientific_result": True,
        "not_human_data": True,
        "claim_boundary": appraisal_summary["claim_boundary"],
        "score_dimension_map": SCORE_DIMENSIONS,
        "quote_selection": (
            "One prespecified synthetic interview bundle per scenario-condition-"
            "orientation cell was independently audited by Codex. Only answers passing "
            "grounding and claim-layer checks were eligible; displayed text is a "
            "verbatim excerpt."
        ),
        "qualitative_review": {
            "reviewer_type": "independent_model_assisted_audit",
            "reviewer_system": "OpenAI Codex",
            "human_review_completed": False,
            "answer_count": int(len(reviewed_interviews)),
            "answers_passing_scientific_use_gate": int(
                (
                    reviewed_interviews["review_grounding_pass"].astype(bool)
                    & reviewed_interviews["review_claim_layer_pass"].astype(bool)
                ).sum()
            ),
            "selected_display_answer_count": int(
                reviewed_interviews["include_in_explorer"].astype(bool).sum()
            ),
            "cells_without_retained_excerpt": int(
                sum(
                    values == [
                        "No grounded qualitative excerpt was retained for this cell."
                    ]
                    for values in quotes.values()
                )
            ),
        },
        "note": (
            "Scores are direct means of verified synthetic appraisal dimensions. "
            "Quotes are exact excerpts retained by an independent Codex audit, not "
            "human-reviewed material or Zurich ED staff testimony."
        ),
    }
    template["results"] = results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n")
    if len(fit) != 400:
        raise ValueError("Unexpected fit record count after explorer export")
    return quote_audit


def _agent_experience_frame(main_source: ResultSource) -> pd.DataFrame:
    rows = []
    for item in main_source.iter_jsonl_files("part3_agent_experience.jsonl"):
        hours = float(item["window_duration_seconds"]) / 3600.0
        dwell = np.asarray(
            [float(value) for value in item.get("zone_dwell_seconds", {}).values() if value > 0]
        )
        probabilities = dwell / dwell.sum() if dwell.size else np.asarray([])
        zone_dispersion = (
            float(-(probabilities * np.log(probabilities)).sum() / np.log(len(probabilities)))
            if len(probabilities) > 1
            else 0.0
        )
        zone_interactions = np.asarray(
            [
                float(value)
                for value in item.get("interaction_counts_by_zone", {}).values()
                if value > 0
            ]
        )
        interaction_probabilities = (
            zone_interactions / zone_interactions.sum()
            if zone_interactions.size
            else np.asarray([])
        )
        interaction_dispersion = (
            float(
                -(
                    interaction_probabilities * np.log(interaction_probabilities)
                ).sum()
                / np.log(len(interaction_probabilities))
            )
            if len(interaction_probabilities) > 1
            else 0.0
        )
        pair_id = (
            f"{item['scenario']}|{item['seed']}|{item['assignment_round']}|"
            f"{item['persona_id']}|{item['role']}|{item['agent_id']}"
        )
        decisions = int(item["model_decision_count"])
        interaction_count = float(item["interaction_count"])
        visible_person_hours = float(item["visible_colleague_person_minutes"]) / 60.0
        staff_staff_count = float(item["staff_staff_interaction_count"])
        rows.append(
            {
                "pair_id": pair_id,
                "scenario": item["scenario"],
                "condition": item["condition"],
                "seed": int(item["seed"]),
                "persona_id": item["persona_id"],
                "role": item["role"],
                "movement_m_per_hour": float(item["movement_distance_m"]) / hours,
                "mean_visible_colleagues": float(
                    item["mean_mutually_visible_colleagues"]
                ),
                "visible_time_share": float(
                    item["share_of_time_with_any_mutually_visible_colleague"]
                ),
                "interaction_participations_per_hour": interaction_count / hours,
                "staff_staff_participations_per_hour": staff_staff_count / hours,
                "patient_facing_participations_per_hour": float(
                    item["patient_facing_interaction_count"]
                )
                / hours,
                "unique_staff_partners": float(
                    item["unique_staff_interaction_partner_count"]
                ),
                "model_engagement_rate": (
                    float(item["model_engage_decision_count"]) / decisions
                    if decisions
                    else math.nan
                ),
                "zone_use_dispersion": zone_dispersion,
                "interaction_zone_dispersion": interaction_dispersion,
                "movement_m_per_interaction": (
                    float(item["movement_distance_m"]) / interaction_count
                    if interaction_count
                    else math.nan
                ),
                "staff_contacts_per_visible_colleague_hour": (
                    staff_staff_count / visible_person_hours
                    if visible_person_hours
                    else math.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 3600:
        raise ValueError(f"Expected 3,600 agent experience rows; got {len(frame)}")
    return frame


def _agent_interaction_spatial_frame(main_source: ResultSource) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name, rows in main_source.iter_jsonl_groups("part3_experience_events.jsonl"):
        parts = Path(name).parts
        runs_index = parts.index("runs")
        scenario, condition, seed_part, round_part = parts[
            runs_index + 1 : runs_index + 5
        ]
        seed = int(seed_part.rsplit("_", 1)[-1])
        assignment_round = int(round_part.rsplit("_", 1)[-1])
        events_by_agent: dict[int, list[tuple[int, float, float, str]]] = defaultdict(
            list
        )
        agent_metadata: dict[int, tuple[str, str]] = {}
        for item in rows:
            if item.get("event_type") != "interaction":
                continue
            position = item.get("place", {}).get("position")
            if not isinstance(position, list) or len(position) != 2:
                continue
            agent_id = int(item["agent_id"])
            events_by_agent[agent_id].append(
                (
                    int(item["timestep"]),
                    float(position[0]),
                    float(position[1]),
                    str(item.get("place", {}).get("zone_id", "")),
                )
            )
            agent_metadata[agent_id] = (str(item["persona_id"]), str(item["role"]))
        for agent_id, events in events_by_agent.items():
            ordered = sorted(events)
            coordinates = np.asarray(
                [(x, y) for _, x, y, _ in ordered], dtype=float
            )
            steps = np.sqrt(np.sum(np.diff(coordinates, axis=0) ** 2, axis=1))
            centroid = np.mean(coordinates, axis=0)
            radius = float(
                np.sqrt(np.mean(np.sum((coordinates - centroid) ** 2, axis=1)))
            )
            occupied_cells = {
                (math.floor(x / 2.0), math.floor(y / 2.0))
                for x, y in coordinates
            }
            occupied_zones = {zone for *_, zone in ordered if zone}
            persona_id, role = agent_metadata[agent_id]
            pair_id = (
                f"{scenario}|{seed}|{assignment_round}|{persona_id}|{role}|{agent_id}"
            )
            records.append(
                {
                    "pair_id": pair_id,
                    "scenario": scenario,
                    "condition": condition,
                    "seed": seed,
                    "persona_id": persona_id,
                    "role": role,
                    "mean_interaction_site_step_m": (
                        float(np.mean(steps)) if len(steps) else math.nan
                    ),
                    "interaction_radius_m": radius,
                    "interaction_footprint_cells": float(len(occupied_cells)),
                    "interaction_zone_count": float(len(occupied_zones)),
                }
            )
    frame = pd.DataFrame(records)
    if len(frame) != 3600:
        raise ValueError(
            f"Expected 3,600 agent interaction-spatial rows; got {len(frame)}"
        )
    return frame


def _residualize(values: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    design = pd.get_dummies(controls.astype(str), drop_first=True, dtype=float)
    matrix = np.column_stack(
        [np.ones(len(design), dtype=float), design.to_numpy(dtype=float)]
    )
    outcome = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    return outcome - matrix @ coefficients


def _trace_fit_associations(
    survey: pd.DataFrame,
    experience: pd.DataFrame,
    interaction_spatial: pd.DataFrame,
) -> pd.DataFrame:
    fit = _fit_rows(survey)[
        ["pair_id", "scenario", "condition", "seed", "persona_id", "fit"]
    ]
    joined = fit.merge(
        experience,
        on=["pair_id", "scenario", "condition", "seed", "persona_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["movement_m_per_hour"].isna().any():
        raise ValueError("Appraisal-to-experience join is incomplete")
    joined = joined.merge(
        interaction_spatial,
        on=["pair_id", "scenario", "condition", "seed", "persona_id", "role"],
        how="left",
        validate="one_to_one",
    )
    if joined["mean_interaction_site_step_m"].isna().any():
        raise ValueError("Appraisal-to-interaction-spatial join is incomplete")
    metric_labels = {
        "movement_m_per_hour": "Movement distance",
        "movement_m_per_interaction": "Movement per interaction",
        "mean_visible_colleagues": "Mean visible colleagues",
        "visible_time_share": "Time with a visible colleague",
        "interaction_participations_per_hour": "Interaction participations",
        "staff_contacts_per_visible_colleague_hour": (
            "Staff contacts per visible-colleague hour"
        ),
        "unique_staff_partners": "Unique staff partners",
        "model_engagement_rate": "LLM engagement rate",
        "zone_use_dispersion": "Zone-use dispersion",
        "interaction_zone_dispersion": "Interaction-zone dispersion",
        "mean_interaction_site_step_m": "Distance between successive interaction sites",
        "interaction_radius_m": "Interaction radius",
        "interaction_footprint_cells": "Occupied interaction cells",
        "interaction_zone_count": "Interaction-zone count",
    }
    scenario_correlations: dict[str, dict[str, float]] = defaultdict(dict)
    adjusted_correlations: dict[str, dict[str, float]] = defaultdict(dict)
    for scenario in SCENARIOS:
        scenario_rows = joined[joined["scenario"] == scenario]
        baseline = scenario_rows[scenario_rows["condition"] == "baseline"].set_index(
            ["seed", "persona_id"]
        )
        deltas = []
        for intervention in INTERVENTIONS:
            treatment = scenario_rows[
                scenario_rows["condition"] == intervention
            ].set_index(["seed", "persona_id"])
            columns = ["fit", *metric_labels]
            delta = treatment[columns] - baseline[columns]
            delta["contrast"] = intervention
            delta["role"] = treatment["role"]
            deltas.append(delta.reset_index())
        combined = pd.concat(deltas, ignore_index=True)
        for metric in metric_labels:
            scenario_correlations[metric][scenario] = float(
                combined["fit"].corr(combined[metric])
            )
            complete = combined[["fit", metric, "role", "contrast"]].dropna()
            fit_residual = _residualize(
                complete["fit"], complete[["role", "contrast"]]
            )
            metric_residual = _residualize(
                complete[metric], complete[["role", "contrast"]]
            )
            adjusted_correlations[metric][scenario] = float(
                np.corrcoef(fit_residual, metric_residual)[0, 1]
            )
    return pd.DataFrame(
        [
            {
                "Experience channel": label,
                "Normal load r": f"{scenario_correlations[metric]['normal_load']:.2f}",
                "Normal load adjusted r": (
                    f"{adjusted_correlations[metric]['normal_load']:.2f}"
                ),
                "High load r": (
                    f"{scenario_correlations[metric]['high_load_high_acuity']:.2f}"
                ),
                "High load adjusted r": (
                    f"{adjusted_correlations[metric]['high_load_high_acuity']:.2f}"
                ),
            }
            for metric, label in metric_labels.items()
        ]
    )


def _reviewed_design_suggestions(
    reviewed_interviews: pd.DataFrame,
    reviewed_counterfactuals: pd.DataFrame,
) -> pd.DataFrame:
    dispositions = reviewed_interviews[
        reviewed_interviews["question_id"] == "counterfactual_change"
    ][
        [
            "prompt_id",
            "review_grounding_pass",
            "review_claim_layer_pass",
        ]
    ]
    coded = reviewed_counterfactuals.merge(
        dispositions, on="prompt_id", validate="one_to_one"
    )
    coded = coded[
        coded["review_grounding_pass"].astype(bool)
        & coded["review_claim_layer_pass"].astype(bool)
    ].copy()
    if len(coded) != 30:
        raise ValueError(f"Expected 30 retained counterfactual answers; got {len(coded)}")
    labels = {
        "semi_private": "Semi-private coordination space",
        "visual": "Visual permeability or awareness",
        "distributed": "Distributed support near active work",
        "availability": "Availability or focus signalling",
        "acoustic": "Acoustic control or quiet",
    }
    records = []
    for code, label in labels.items():
        matched = coded[f"manual_{code}"].astype(bool)
        personas = coded.loc[matched, "persona_id"].nunique()
        count = int(matched.sum())
        records.append(
            {
                "Synthetic design suggestion": label,
                "Retained answers, n (%)": f"{count}/30 ({100.0 * count / 30:.1f}%)",
                "Orientations represented": f"{personas}/5",
            }
        )
    return pd.DataFrame(records)


def _write_report(output_dir: Path, font_family: str) -> None:
    text = f"""# Part 3 cognitive-persona findings

## Figures

- **Persona explorer preview (unnumbered).** Static link preview for the supplementary interactive explorer. OCEAN values are presentation crosswalks, not measured psychometrics.
- **FigA — Cognitive architecture.** The during-shift decision loop and the strictly post-shift appraisal pathway. Generated prose never feeds back into movement or workflow.
- **FigB — Common affordance, different uptake.** The combined intervention exposes all five orientations to nearly the same visibility gain, while role-standardized LLM engagement changes diverge.
- **FigC — Inclusive person–space fit.** Baseline and Both fit distributions show persona means, paired-seed observations, the ensemble average, and the least-served-orientation floor.

## Tables

- **TableA — Persona appraisal profiles.** High-load Baseline-to-Both changes in four core appraisal dimensions plus one independently audited synthetic response.
- **TableB — Inclusive-fit effects.** Changes in average fit, fit floor, and between-orientation dispersion for all three interventions.
- **TableC — Audited synthetic design suggestions.** Independent Codex coding of the 30 grounded counterfactual answers retained from the prespecified 40-bundle audit.
- **Appendix TableA — Architecture ablation.** Compact matched 2 × 2 persona-conditioning-by-memory results on fixed observed opportunities.

## Interpretation boundary

The Part 3 results support a synthetic-model claim: a spatial affordance can be commonly available while its behavioral uptake and appraised value differ across designed cognitive orientations. Under high load, Both raises average fit and the least-served-orientation floor while narrowing between-orientation dispersion. This is not evidence of Zurich ED staff preferences, psychometric types, or a universally optimal design.

Exploratory trace-to-fit correlations are not promoted to a paper table. No
movement, visibility, interaction-volume, or interaction-spacing channel
consistently explains fit across both workload scenarios after role and
intervention contrast are controlled. The appraisal model also received
compressed trace evidence, so those post-hoc associations cannot be interpreted
as causal mediation.

The counterfactual table is hypothesis-generating. It records recurrent proposals
from synthetic interviews, not preferences reported by Zurich ED staff. Among the
30 independently audited and retained answers, semi-private coordination space was
the most recurrent suggestion, followed by visual permeability, distributed
support, availability signalling, and acoustic control. These categories were
assigned in a post-hoc model-assisted audit, not a human thematic analysis.

The qualitative display sample was reviewed independently by Codex against
grounding and claim-layer criteria. This process retained 184 of 240 answers and
selected 107 for display. No human interview review or inter-rater reliability
assessment was performed, and two persona-condition cells have no retained
qualitative excerpt.

## Build

Figures use {font_family}. The build reads completed outputs only and does not run the simulation or an LLM.
"""
    (output_dir / "part3_cognitive_personas_n10_report.md").write_text(text)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    study_figures_dir = Path(args.study_figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    study_figures_dir.mkdir(parents=True, exist_ok=True)
    font_family = _configure_style()

    with (
        ResultSource(Path(args.main_source)) as main_source,
        ResultSource(Path(args.appraisal_source)) as appraisal_source,
        ResultSource(Path(args.ablation_source)) as ablation_source,
    ):
        main_summary = main_source.read_json("analysis/analysis_summary.json")
        appraisal_summary = appraisal_source.read_json(
            "analysis/appraisal_analysis_summary.json"
        )
        ablation_summary = ablation_source.read_json(
            "analysis/architecture_ablation_summary.json"
        )
        _validate_sources(main_summary, appraisal_summary, ablation_summary)

        experience_effects = main_source.read_csv(
            "analysis/paired_persona_experience_effects.csv"
        )
        persona_effects = main_source.read_csv("analysis/paired_persona_effects.csv")
        survey = appraisal_source.read_csv("analysis/survey_responses.csv")
        reviewed_interviews = pd.read_csv(Path(args.reviewed_interviews).resolve())
        reviewed_counterfactuals = pd.read_csv(
            Path(args.reviewed_counterfactuals).resolve()
        )
        ablation_cells = ablation_source.read_csv(
            "analysis/architecture_ablation_cells.csv"
        )
        ablation_effects = ablation_source.read_csv(
            "analysis/architecture_ablation_effects.csv"
        )
        template = _load_template(DEFAULT_EXPLORER_DATA)

        fit = _fit_rows(survey)
        ensemble = _ensemble_seed_summary(fit)
        _, quote_audit = _select_reviewed_quotes(reviewed_interviews)
        _plot_study_overview(study_figures_dir)
        _plot_interaction_pipeline(study_figures_dir)
        _plot_persona_atlas(template, figures_dir)
        _plot_cognitive_architecture(figures_dir)
        _plot_exposure_uptake(experience_effects, persona_effects, figures_dir)
        _plot_fit_ensemble(fit, ensemble, figures_dir)

        table_a = _table_persona_appraisals(survey, reviewed_interviews)
        _write_table(
            table_a,
            tables_dir / "tableA_persona_appraisal_profiles",
            bold_columns=("Orientation", "Overall fit"),
            note=(
                "High load; each score is Baseline → Both (change) on a 1–7 scale, "
                "averaged across 10 paired seeds. The response was retained from the "
                "prespecified high-load/Both bundle after an independent Codex grounding "
                "and claim-layer audit. These are designed-orientation appraisals and "
                "synthetic responses, not Zurich ED staff testimony or human-coded "
                "interviews."
            ),
        )

        table_b = _table_inclusive_fit(ensemble)
        significant_fit_cells = {
            (row_index, column)
            for row_index, row in table_b.iterrows()
            for column in (
                "Δ average fit [95% CI]",
                "Δ fit floor [95% CI]",
                "Δ orientation SD [95% CI]",
            )
            if (
                (match := re.search(
                    r"\[([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\]",
                    str(row[column]),
                ))
                and (
                    float(match.group(1)) > 0.0
                    or float(match.group(2)) < 0.0
                )
            )
        }
        _write_table(
            table_b,
            tables_dir / "tableB_inclusive_person_space_fit",
            bold_cells=significant_fit_cells,
            note=(
                "The fit floor is the lowest of the five orientation scores within each "
                "seed. Orientation SD is the within-seed standard deviation across the "
                "five designed orientations; a negative change indicates less disparity. "
                "Confidence intervals use seed-paired means (n=10). These are "
                "synthetic-orientation ensemble metrics, not population accessibility "
                "estimates. Bold values have 95% confidence intervals that exclude zero."
            ),
        )

        table_c = _reviewed_design_suggestions(
            reviewed_interviews, reviewed_counterfactuals
        )
        priority_cells = {
            (row_index, column)
            for row_index in range(min(2, len(table_c)))
            for column in (
                "Synthetic design suggestion",
                "Retained answers, n (%)",
            )
        }
        _write_table(
            table_c,
            tables_dir / "tableC_recurrent_design_priorities",
            bold_cells=priority_cells,
            note=(
                "Independent Codex coding of 30 grounded counterfactual answers retained "
                "from the 40 prespecified bundles; ten were excluded for grounding or "
                "claim-layer failures. Categories overlap. The two most recurrent "
                "suggestions are bold. This is a post-hoc model-assisted audit, not human "
                "thematic analysis or Zurich ED staff preference evidence."
            ),
        )

        appendix_a, architecture_effect_note = _table_architecture_ablation_matrix(
            ablation_cells, ablation_effects
        )
        _write_table(
            appendix_a,
            tables_dir / "appendix_tableA_cognitive_architecture_ablation",
            note=(
                "Engagement rate [95% Wilson CI] on the same 120 fixed observed "
                "opportunities per cell. Matched factorial effects (95% CI): "
                f"{architecture_effect_note} This ablation does not represent complete "
                "recursive closed-loop trajectories."
            ),
        )

        exported_quote_audit = _export_explorer(
            template,
            survey,
            reviewed_interviews,
            appraisal_summary,
            Path(args.explorer_output).resolve(),
        )
        if not quote_audit.equals(exported_quote_audit):
            raise ValueError("Explorer quote selection changed within one build")
        quote_audit.to_csv(
            tables_dir / "audit_explorer_quote_selection.csv", index=False
        )
    _write_report(output_dir, font_family)
    expected = [
        figures_dir / f"{stem}.{suffix}"
        for stem in (
            "persona_explorer_preview",
            "figA_cognitive_architecture",
            "figB_common_affordance_different_uptake",
            "figC_inclusive_person_space_fit",
        )
        for suffix in ("png", "pdf")
    ]
    expected.extend(
        study_figures_dir / f"{stem}.{suffix}"
        for stem in (
            "study_design_overview",
            "interaction_generation_pipeline",
        )
        for suffix in ("png", "pdf")
    )
    expected.extend(
        tables_dir / f"{stem}.{suffix}"
        for stem in (
            "tableA_persona_appraisal_profiles",
            "tableB_inclusive_person_space_fit",
            "tableC_recurrent_design_priorities",
            "appendix_tableA_cognitive_architecture_ablation",
        )
        for suffix in ("csv", "md", "tex")
    )
    expected.extend(
        [
            tables_dir / "audit_explorer_quote_selection.csv",
            Path(args.explorer_output).resolve(),
            output_dir / "part3_cognitive_personas_n10_report.md",
        ]
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ValueError(f"Part 3 build is incomplete: {missing}")
    return {
        "build_pass": True,
        "font_family": font_family,
        "part3_figure_count": 4,
        "explorer_preview_count": 1,
        "cross_study_figure_count": 2,
        "main_table_count": 3,
        "appendix_table_count": 1,
        "explorer_result_count": 40,
        "output_dir": str(output_dir),
        "study_figures_dir": str(study_figures_dir),
        "explorer_output": str(Path(args.explorer_output).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-source", default=str(DEFAULT_MAIN_SOURCE))
    parser.add_argument("--appraisal-source", default=str(DEFAULT_APPRAISAL_SOURCE))
    parser.add_argument("--ablation-source", default=str(DEFAULT_ABLATION_SOURCE))
    parser.add_argument(
        "--reviewed-interviews", default=str(DEFAULT_REVIEWED_INTERVIEWS)
    )
    parser.add_argument(
        "--reviewed-counterfactuals", default=str(DEFAULT_REVIEWED_COUNTERFACTUALS)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--study-figures-dir", default=str(DEFAULT_STUDY_FIGURES_DIR)
    )
    parser.add_argument("--explorer-output", default=str(DEFAULT_EXPLORER_DATA))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
