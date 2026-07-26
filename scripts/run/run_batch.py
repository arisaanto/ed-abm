#!/usr/bin/env python3
"""Run a manifest of independent ABM single-run jobs."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parents[2]
RUN_SINGLE = PROJECT_DIR / "scripts" / "run" / "run_single.py"


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _chunk_rows(
    rows: list[dict[str, str]], chunk_index: int | None, chunk_size: int | None
) -> list[dict[str, str]]:
    if chunk_index is None or chunk_size is None:
        return rows
    start = max(0, int(chunk_index) * int(chunk_size))
    end = start + int(chunk_size)
    return rows[start:end]


def _rebased_output_dir(original: str, output_root: Path | None) -> str:
    if output_root is None:
        return original
    path = Path(original)
    try:
        parts = path.parts
        if "outputs" in parts and "batch" in parts:
            batch_index = parts.index("batch")
            relative = Path(*parts[batch_index + 1 :])
        else:
            relative = path.name
    except ValueError:
        relative = path.name
    return str(output_root / relative)


def _command_for_row(row: dict[str, str], output_root: Path | None) -> list[str]:
    output_dir = _rebased_output_dir(row["output_dir"], output_root)
    cmd = [
        sys.executable,
        str(RUN_SINGLE),
        "--scenario-mode",
        row["scenario_mode"],
        "--condition",
        row["condition"],
        "--seed",
        str(row["seed"]),
        "--duration",
        str(row["duration"]),
        "--warmup-seconds",
        str(row["warmup_seconds"]),
        "--scenario-start-hour",
        str(row["scenario_start_hour"]),
        "--output-dir",
        output_dir,
    ]
    if row.get("batch_name"):
        cmd.extend(["--batch-name", row["batch_name"]])
    if row.get("run_id"):
        cmd.extend(["--run-id", row["run_id"]])
    if row.get("validation_target"):
        cmd.extend(["--validation-target", row["validation_target"]])
    if row.get("sensitivity_parameter"):
        cmd.extend(["--sensitivity-parameter", row["sensitivity_parameter"]])
        cmd.extend(["--sensitivity-level", row["sensitivity_level"]])
        cmd.extend(["--sensitivity-value", str(row["sensitivity_value"])])
        cmd.extend(["--sensitivity-default", str(row["sensitivity_default_value"])])
    if _truthy(row.get("export_part3_episodes", "")):
        cmd.append("--export-part3-episodes")
        if row.get("part3_max_episodes"):
            cmd.extend(["--part3-max-episodes", str(row["part3_max_episodes"])])
    return cmd


def _run_command(cmd: list[str], dry_run: bool = False) -> tuple[int, str]:
    if dry_run:
        return 0, " ".join(cmd)
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout


def run_rows(
    rows: Iterable[dict[str, str]],
    max_workers: int,
    output_root: Path | None,
    dry_run: bool = False,
) -> int:
    rows = list(rows)
    if not rows:
        print("No manifest rows selected.")
        return 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _run_command, _command_for_row(row, output_root), dry_run
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            return_code, output = future.result()
            label = f"{row['scenario_mode']}/{row['condition']}/seed_{row['seed']}"
            if return_code == 0:
                print(f"[OK] {label}")
            else:
                failures += 1
                print(f"[FAIL] {label}")
            if output.strip():
                print(output.strip())
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=None, help="Zero-based chunk index, e.g. SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Rows per chunk/array task.")
    parser.add_argument("--output-root", default=None, help="Optional root replacing outputs/batch/... from the manifest.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_manifest(Path(args.manifest))
    selected = _chunk_rows(rows, args.chunk_index, args.chunk_size)
    output_root = Path(args.output_root) if args.output_root else None
    failures = run_rows(selected, args.max_workers, output_root, args.dry_run)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
