#!/usr/bin/env python3
"""Run independent Part 3 ABMs in parallel through one inference broker.

The engine is initialized once for the allocation. Independent simulations run
in spawned CPU processes and block at each cognitive decision until the parent
broker returns a schema-constrained result. Requests that are ready together
are micro-batched without advancing any simulation past an unanswered request.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from scripts.run_part3_closed_loop import (
    VLLMCategoricalCausalProvider,
    _persona_assignment,
    _validate_runtime_contract,
)
from scripts.run_single import run_single
from scripts.verify_part3_closed_loop_gate import verify as verify_run
from src.part3_closed_loop import Part3ClosedLoopController
from src.vllm_backend import VLLMOfflineBackend


INTEGER_FIELDS = (
    "run_index",
    "seed",
    "assignment_round",
    "duration_seconds",
    "warmup_seconds",
    "max_model_decisions",
    "max_model_decisions_per_agent",
    "decision_window_seconds",
    "max_model_decisions_per_window",
)
FLOAT_FIELDS = ("model_decision_sample_rate",)
CONDITIONS = {"baseline", "cockpit_only", "nursta_only", "both"}
ORCHESTRATION_MODE = "parallel_processes_shared_inference_broker_v1"
TEMPORAL_COVERAGE_POLICY = "aggregate_sampled_opportunity_fidelity_v2"


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Closed-loop manifest is empty")
    required = {"scenario", "condition", *INTEGER_FIELDS, *FLOAT_FIELDS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    normalized: list[dict[str, Any]] = []
    identities = set()
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for field in INTEGER_FIELDS:
            row[field] = int(row[field])
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
        if row["scenario"] not in config.SCENARIO_MODES:
            raise ValueError(f"Unknown scenario: {row['scenario']}")
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"Unknown condition: {row['condition']}")
        if row["assignment_round"] not in range(1, 6):
            raise ValueError("assignment_round must be 1..5")
        if row["duration_seconds"] <= row["warmup_seconds"]:
            raise ValueError("Each duration must exceed its warmup")
        if row["decision_window_seconds"] <= 0:
            raise ValueError("Pilot rows require a positive decision window")
        if not 0.0 < row["model_decision_sample_rate"] <= 1.0:
            raise ValueError("model_decision_sample_rate must be in (0, 1]")
        active_seconds = row["duration_seconds"] - row["warmup_seconds"]
        if active_seconds % row["decision_window_seconds"]:
            raise ValueError("Active duration must divide exactly into decision windows")
        safety_ceilings = (
            row["max_model_decisions"],
            row["max_model_decisions_per_agent"],
            row["max_model_decisions_per_window"],
        )
        if any(value <= 0 for value in safety_ceilings):
            raise ValueError("Pilot safety ceilings must all be positive")
        if any(value > row["max_model_decisions"] for value in safety_ceilings[1:]):
            raise ValueError("Agent/window safety ceilings cannot exceed the run ceiling")
        identity = (
            row["scenario"],
            row["condition"],
            row["seed"],
            row["assignment_round"],
        )
        if identity in identities:
            raise ValueError(f"Duplicate run identity: {identity}")
        identities.add(identity)
        normalized.append(row)
    indexes = [row["run_index"] for row in normalized]
    if indexes != list(range(1, len(normalized) + 1)):
        raise ValueError("run_index must be ordered and contiguous from 1")
    return normalized


def _run_dir(output_root: Path, row: dict[str, Any]) -> Path:
    return (
        output_root
        / "runs"
        / str(row["scenario"])
        / str(row["condition"])
        / f"seed_{row['seed']}"
        / f"assignment_round_{row['assignment_round']}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _temporal_diagnostics(
    row: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Audit sampled-call fidelity without forcing calls into sparse windows."""

    closed_loop = summary.get("part3_closed_loop", {})
    expected_window_count = (
        row["duration_seconds"] - row["warmup_seconds"]
    ) // row["decision_window_seconds"]

    def window_counts(name: str) -> dict[int, int]:
        return {
            int(index): int(count)
            for index, count in closed_loop.get(name, {}).items()
        }

    eligible = window_counts("eligible_decisions_by_window")
    sampled = window_counts("sampled_candidates_by_window")
    called = window_counts("provider_calls_by_window")
    valid_indexes = set(range(expected_window_count))
    invalid_indexes = sorted(
        (set(eligible) | set(sampled) | set(called)) - valid_indexes
    )
    sampled_call_mismatches = {
        str(index): {
            "sampled_candidates": sampled.get(index, 0),
            "provider_calls": called.get(index, 0),
        }
        for index in range(expected_window_count)
        if sampled.get(index, 0) != called.get(index, 0)
    }
    return {
        "temporal_coverage_policy": TEMPORAL_COVERAGE_POLICY,
        "expected_decision_window_count": expected_window_count,
        "eligible_decision_window_count": sum(
            eligible.get(index, 0) > 0 for index in range(expected_window_count)
        ),
        "sampled_decision_window_count": sum(
            sampled.get(index, 0) > 0 for index in range(expected_window_count)
        ),
        "covered_decision_window_count": sum(
            called.get(index, 0) > 0 for index in range(expected_window_count)
        ),
        "invalid_decision_window_indexes": invalid_indexes,
        "sampled_call_mismatches_by_window": sampled_call_mismatches,
        "sampled_call_fidelity_pass": (
            not invalid_indexes and not sampled_call_mismatches
        ),
    }


def _run_result_from_existing(
    output_root: Path, row: dict[str, Any]
) -> dict[str, Any] | None:
    """Re-verify and recover one complete run without rerunning its ABM."""

    run_dir = _run_dir(output_root, row)
    summary_path = run_dir / "summary.json"
    decisions_path = run_dir / "part3_cognitive_decisions.jsonl"
    if not summary_path.is_file() or not decisions_path.is_file():
        return None
    try:
        summary = _read_json(summary_path)
        verification = verify_run(run_dir, require_model_interaction=False)
        temporal = _temporal_diagnostics(row, summary)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    verification.update(temporal)
    verification.pop("minimum_covered_decision_window_count", None)
    verification["verification_pass"] = bool(
        verification.get("verification_pass")
        and temporal["sampled_call_fidelity_pass"]
    )
    if not temporal["sampled_call_fidelity_pass"]:
        verification.setdefault("errors", []).append(
            "Sampled candidates did not map one-to-one to model calls by window"
        )
    if verification["verification_pass"] is not True:
        return None
    _write_json(run_dir / "closed_loop_verification.json", verification)
    return {
        **row,
        "run_dir": str(run_dir.relative_to(output_root)),
        "elapsed_seconds": None,
        "workflow_status": summary.get("workflow_health", {}).get(
            "workflow_health_status"
        ),
        "provider_call_count": verification.get("provider_call_count"),
        "eligible_decision_count": summary.get("part3_closed_loop", {}).get(
            "eligible_decision_count"
        ),
        "sampled_candidate_count": summary.get("part3_closed_loop", {}).get(
            "sampled_candidate_count"
        ),
        "observed_sample_rate": summary.get("part3_closed_loop", {}).get(
            "observed_sample_rate"
        ),
        "covered_decision_window_count": temporal[
            "covered_decision_window_count"
        ],
        "model_applied_interaction_count": verification.get(
            "model_applied_interaction_count"
        ),
        "model_missed_opportunity_count": verification.get(
            "model_missed_opportunity_count"
        ),
        "verification_pass": True,
        "verification_errors": [],
        "recovered_from_prior_execution": True,
    }


class BrokerBackendProxy:
    """Present the backend API to one simulation over a private pipe."""

    def __init__(
        self,
        connection: Connection,
        worker_id: str,
        backend_metadata: dict[str, Any],
    ) -> None:
        self.connection = connection
        self.worker_id = worker_id
        self.model = str(backend_metadata["model"])
        self.model_revision = str(backend_metadata["model_revision"])
        self.enable_thinking = bool(backend_metadata["thinking_enabled"])
        self._request_index = 0

    def generate_prompt_packets(self, packets) -> list[dict[str, Any]]:
        packet_list = [dict(packet) for packet in packets]
        self._request_index += 1
        request_id = f"{self.worker_id}:{self._request_index}"
        self.connection.send(
            {
                "type": "inference_request",
                "worker_id": self.worker_id,
                "request_id": request_id,
                "packets": packet_list,
                "submitted_at": time.perf_counter(),
            }
        )
        response = self.connection.recv()
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"Inference broker response mismatch for {request_id}: {response}"
            )
        if response.get("type") != "inference_result":
            raise RuntimeError(
                f"Inference broker failed for {request_id}: "
                f"{response.get('error', response)}"
            )
        results = [dict(result) for result in response["results"]]
        for result in results:
            result["broker_batch_size"] = int(response["broker_batch_size"])
            result["broker_elapsed_seconds"] = float(
                response["broker_elapsed_seconds"]
            )
        return results


def _run_args(
    row: dict[str, Any], run_dir: Path, batch_name: str
) -> argparse.Namespace:
    return argparse.Namespace(
        scenario_mode=row["scenario"],
        condition=row["condition"],
        seed=row["seed"],
        duration=row["duration_seconds"],
        warmup_seconds=row["warmup_seconds"],
        scenario_start_hour=10,
        validation_target="care_area",
        batch_name=batch_name,
        run_id=(
            f"{row['scenario']}_{row['condition']}_seed_{row['seed']}_"
            f"assignment_round_{row['assignment_round']}"
        ),
        output_dir=str(run_dir),
        sensitivity_parameter=None,
        sensitivity_level=None,
        sensitivity_value=None,
        sensitivity_default=None,
        export_part3_episodes=False,
        part3_max_episodes=0,
    )


def _simulation_worker(
    row: dict[str, Any],
    output_root_value: str,
    connection: Connection,
    backend_metadata: dict[str, Any],
    batch_name: str,
) -> None:
    """Run one ABM in an isolated process and synchronously query the broker."""

    worker_id = f"run_{row['run_index']}"
    output_root = Path(output_root_value)
    run_dir = _run_dir(output_root, row)
    started = time.perf_counter()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        proxy = BrokerBackendProxy(connection, worker_id, backend_metadata)
        provider = VLLMCategoricalCausalProvider(proxy)

        def controller_factory(simulation):
            return Part3ClosedLoopController(
                provider,
                _persona_assignment(simulation, row["assignment_round"]),
                start_seconds=row["warmup_seconds"],
                max_decisions_per_run=row["max_model_decisions"],
                max_decisions_per_agent=row["max_model_decisions_per_agent"],
                decision_window_seconds=row["decision_window_seconds"],
                max_decisions_per_window=row[
                    "max_model_decisions_per_window"
                ],
                model_decision_sample_rate=row[
                    "model_decision_sample_rate"
                ],
                sampling_replication_id=row["assignment_round"],
            )

        summary = run_single(
            _run_args(row, run_dir, batch_name),
            part3_controller_factory=controller_factory,
        )
        verification = verify_run(run_dir, require_model_interaction=False)
        temporal = _temporal_diagnostics(row, summary)
        verification.update(temporal)
        if not temporal["sampled_call_fidelity_pass"]:
            verification["errors"].append(
                "Sampled candidates did not map one-to-one to model calls by window"
            )
            verification["verification_pass"] = False
        _write_json(run_dir / "closed_loop_verification.json", verification)
        run_result = {
            **row,
            "run_dir": str(run_dir.relative_to(output_root)),
            "elapsed_seconds": time.perf_counter() - started,
            "workflow_status": summary.get("workflow_health", {}).get(
                "workflow_health_status"
            ),
            "provider_call_count": verification.get("provider_call_count"),
            "eligible_decision_count": summary.get("part3_closed_loop", {}).get(
                "eligible_decision_count"
            ),
            "sampled_candidate_count": summary.get("part3_closed_loop", {}).get(
                "sampled_candidate_count"
            ),
            "observed_sample_rate": summary.get("part3_closed_loop", {}).get(
                "observed_sample_rate"
            ),
            "covered_decision_window_count": temporal[
                "covered_decision_window_count"
            ],
            "model_applied_interaction_count": verification.get(
                "model_applied_interaction_count"
            ),
            "model_missed_opportunity_count": verification.get(
                "model_missed_opportunity_count"
            ),
            "verification_pass": verification.get("verification_pass"),
            "verification_errors": list(verification.get("errors", [])),
        }
        connection.send(
            {
                "type": "worker_complete",
                "worker_id": worker_id,
                "run_result": run_result,
            }
        )
    except BaseException as error:
        try:
            connection.send(
                {
                    "type": "worker_failed",
                    "worker_id": worker_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def _parallel_progress(
    output_root: Path,
    expected_run_count: int,
    completed: list[dict[str, Any]],
    active: dict[Connection, dict[str, Any]],
    broker_stats: Counter[str],
) -> None:
    ordered = sorted(completed, key=lambda item: int(item["run_index"]))
    _write_json(
        output_root / "batch_progress.json",
        {
            "expected_run_count": expected_run_count,
            "completed_run_count": len(ordered),
            "active_run_indexes": sorted(
                int(state["row"]["run_index"]) for state in active.values()
            ),
            "engine_load_count": (
                broker_stats["prior_engine_load_count"] + 1
            ),
            "orchestration_mode": ORCHESTRATION_MODE,
            "broker_inference_batch_count": (
                broker_stats["prior_batch_count"]
                + broker_stats["batch_count"]
            ),
            "broker_packet_count": (
                broker_stats["prior_packet_count"]
                + broker_stats["packet_count"]
            ),
            "maximum_inference_batch_size": broker_stats["maximum_batch_size"],
            "worker_attempt_count": (
                broker_stats["prior_worker_attempt_count"]
                + broker_stats["workers_started"]
            ),
            "maximum_active_workers": broker_stats["maximum_active_workers"],
            "runs": ordered,
        },
    )


def run_parallel_workers(
    rows: list[dict[str, Any]],
    output_root: Path,
    backend,
    backend_metadata: dict[str, Any],
    *,
    max_workers: int,
    inference_batch_wait_seconds: float,
    batch_name: str = "part3_closed_loop",
    initial_completed: list[dict[str, Any]] | None = None,
    prior_progress: dict[str, Any] | None = None,
    total_expected_run_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run manifest rows concurrently while the parent owns the LLM engine."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    context = mp.get_context("spawn")
    pending = iter(rows)
    active: dict[Connection, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = list(initial_completed or [])
    expected_run_count = int(
        total_expected_run_count
        if total_expected_run_count is not None
        else len(rows) + len(completed)
    )
    prior_progress = dict(prior_progress or {})
    broker_stats: Counter[str] = Counter()
    broker_stats["prior_batch_count"] = int(
        prior_progress.get("broker_inference_batch_count", 0)
    )
    broker_stats["prior_packet_count"] = int(
        prior_progress.get("broker_packet_count", 0)
    )
    broker_stats["prior_worker_attempt_count"] = int(
        prior_progress.get("worker_attempt_count", 0)
    )
    broker_stats["prior_engine_load_count"] = int(
        prior_progress.get("engine_load_count", 0)
    )
    broker_stats["maximum_batch_size"] = int(
        prior_progress.get("maximum_inference_batch_size", 0)
    )
    broker_stats["maximum_active_workers"] = int(
        prior_progress.get("maximum_active_workers", 0)
    )
    batch_size_counts: Counter[int] = Counter()

    def start_next() -> bool:
        try:
            row = next(pending)
        except StopIteration:
            return False
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_simulation_worker,
            args=(
                row,
                str(output_root),
                child_connection,
                backend_metadata,
                batch_name,
            ),
            name=f"part3_run_{row['run_index']}",
        )
        process.start()
        child_connection.close()
        active[parent_connection] = {"process": process, "row": row}
        broker_stats["workers_started"] += 1
        broker_stats["maximum_active_workers"] = max(
            broker_stats["maximum_active_workers"], len(active)
        )
        return True

    def handle_non_inference(
        connection: Connection, message: dict[str, Any]
    ) -> None:
        state = active[connection]
        if message.get("type") == "worker_failed":
            raise RuntimeError(
                f"{message.get('worker_id')} failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        if message.get("type") != "worker_complete":
            raise RuntimeError(f"Unexpected broker message: {message}")
        result = dict(message["run_result"])
        if result.get("verification_pass") is not True:
            details = "; ".join(result.get("verification_errors", []))
            raise RuntimeError(
                f"Run verification failed for {result.get('run_dir')}: "
                f"{details or 'no verifier details were returned'}"
            )
        completed.append(result)
        process = state["process"]
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise RuntimeError(
                f"Worker {process.name} did not exit after reporting completion"
            )
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker {process.name} exited with code {process.exitcode}"
            )
        connection.close()
        del active[connection]
        start_next()

    try:
        for _ in range(min(max_workers, len(rows))):
            start_next()
        _parallel_progress(
            output_root, expected_run_count, completed, active, broker_stats
        )

        while active:
            ready = wait(list(active), timeout=1.0)
            if not ready:
                crashed = [
                    state["process"]
                    for state in active.values()
                    if state["process"].exitcode not in (None, 0)
                ]
                if crashed:
                    raise RuntimeError(
                        "Worker exited without diagnostics: "
                        + ", ".join(
                            f"{process.name}={process.exitcode}"
                            for process in crashed
                        )
                    )
                continue

            inference_messages: list[tuple[Connection, dict[str, Any]]] = []
            for connection in ready:
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"Worker pipe closed unexpectedly: {active[connection]['row']}"
                    ) from error
                if message.get("type") == "inference_request":
                    inference_messages.append((connection, message))
                else:
                    handle_non_inference(connection, message)

            if inference_messages and inference_batch_wait_seconds > 0:
                deadline = time.perf_counter() + inference_batch_wait_seconds
                occupied = {connection for connection, _ in inference_messages}
                while time.perf_counter() < deadline:
                    candidates = [
                        connection
                        for connection in active
                        if connection not in occupied
                    ]
                    if not candidates:
                        break
                    remaining = deadline - time.perf_counter()
                    more_ready = wait(candidates, timeout=max(remaining, 0.0))
                    if not more_ready:
                        break
                    for connection in more_ready:
                        message = connection.recv()
                        if message.get("type") == "inference_request":
                            inference_messages.append((connection, message))
                            occupied.add(connection)
                        else:
                            handle_non_inference(connection, message)

            if inference_messages:
                packets = [
                    packet
                    for _, message in inference_messages
                    for packet in message["packets"]
                ]
                started = time.perf_counter()
                results = backend.generate_prompt_packets(packets)
                elapsed = time.perf_counter() - started
                if len(results) != len(packets):
                    raise RuntimeError(
                        "Inference broker returned the wrong number of results"
                    )
                broker_stats["batch_count"] += 1
                broker_stats["packet_count"] += len(packets)
                broker_stats["maximum_batch_size"] = max(
                    broker_stats["maximum_batch_size"], len(packets)
                )
                batch_size_counts[len(packets)] += 1
                offset = 0
                for connection, message in inference_messages:
                    count = len(message["packets"])
                    connection.send(
                        {
                            "type": "inference_result",
                            "request_id": message["request_id"],
                            "results": results[offset : offset + count],
                            "broker_batch_size": len(packets),
                            "broker_elapsed_seconds": elapsed,
                        }
                    )
                    offset += count
            _parallel_progress(
                output_root,
                expected_run_count,
                completed,
                active,
                broker_stats,
            )
    finally:
        for connection, state in list(active.items()):
            process = state["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            connection.close()

    retained_packet_count = sum(
        int(result.get("provider_call_count") or 0) for result in completed
    )
    broker_summary = {
        "orchestration_mode": ORCHESTRATION_MODE,
        "configured_max_workers": max_workers,
        "workers_started": (
            broker_stats["prior_worker_attempt_count"]
            + broker_stats["workers_started"]
        ),
        "execution_workers_started": broker_stats["workers_started"],
        "recovered_complete_run_count": len(initial_completed or []),
        "unique_completed_run_count": len(completed),
        "worker_attempt_count": (
            broker_stats["prior_worker_attempt_count"]
            + broker_stats["workers_started"]
        ),
        "maximum_active_workers": broker_stats["maximum_active_workers"],
        "inference_batch_wait_seconds": inference_batch_wait_seconds,
        "inference_batch_count": (
            broker_stats["prior_batch_count"] + broker_stats["batch_count"]
        ),
        "inference_packet_count": retained_packet_count,
        "attempted_inference_packet_count": (
            broker_stats["prior_packet_count"] + broker_stats["packet_count"]
        ),
        "maximum_inference_batch_size": broker_stats["maximum_batch_size"],
        "inference_batch_size_counts": {
            str(size): count for size, count in sorted(batch_size_counts.items())
        },
    }
    return sorted(completed, key=lambda item: int(item["run_index"])), broker_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-name", default="part3_closed_loop")
    parser.add_argument("--model", default=config.VLLM_MODEL_NAME)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=("eager", "prefetch", "lazy", "default"),
        default="eager",
    )
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=50.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Re-verify complete run folders, remove only incomplete run folders, "
            "and execute the remaining manifest rows."
        ),
    )
    parser.add_argument(
        "--resume-inventory-only",
        action="store_true",
        help=(
            "With --resume, re-verify and inventory existing run folders, then "
            "exit before initializing the model or starting simulations."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_runtime_contract()
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.resume_inventory_only and not args.resume:
        raise SystemExit("--resume-inventory-only requires --resume")
    if any(output_root.iterdir()) and not args.resume:
        raise SystemExit(f"Output root must be empty: {output_root}")
    rows = _read_manifest(manifest_path)
    if args.max_output_tokens < config.VLLM_MAX_OUTPUT_TOKENS:
        raise SystemExit(
            "Pilot max-output-tokens is below the validated categorical contract"
        )

    manifest_copy = output_root / "manifest.csv"
    prior_progress: dict[str, Any] = {}
    prior_engine_metadata: dict[str, Any] = {}
    recovered: list[dict[str, Any]] = []
    incomplete_existing_run_indexes: list[int] = []
    if args.resume:
        if not manifest_copy.is_file():
            raise SystemExit("Resume output is missing its archived manifest")
        if manifest_copy.read_bytes() != manifest_path.read_bytes():
            raise SystemExit("Resume manifest differs from the archived manifest")
        progress_path = output_root / "batch_progress.json"
        engine_path = output_root / "engine_metadata.json"
        if progress_path.is_file():
            prior_progress = _read_json(progress_path)
            started_indexes = {
                int(result["run_index"])
                for result in prior_progress.get("runs", [])
            } | {
                int(index)
                for index in prior_progress.get("active_run_indexes", [])
            }
            prior_progress["worker_attempt_count"] = max(
                int(prior_progress.get("worker_attempt_count", 0)),
                len(started_indexes),
            )
            prior_progress["maximum_active_workers"] = max(
                int(prior_progress.get("maximum_active_workers", 0)),
                len(prior_progress.get("active_run_indexes", [])),
            )
        if engine_path.is_file():
            prior_engine_metadata = _read_json(engine_path)
        for row in rows:
            recovered_result = _run_result_from_existing(output_root, row)
            if recovered_result is not None:
                recovered.append(recovered_result)
                continue
            run_dir = _run_dir(output_root, row)
            if run_dir.exists():
                incomplete_existing_run_indexes.append(int(row["run_index"]))
                if not args.resume_inventory_only:
                    shutil.rmtree(run_dir)
        pending_rows = [
            row
            for row in rows
            if int(row["run_index"])
            not in {int(result["run_index"]) for result in recovered}
        ]
        resume_inventory = {
            "inventory_pass": (
                len(recovered) + len(pending_rows) == len(rows)
                and not (
                    {int(result["run_index"]) for result in recovered}
                    & {int(row["run_index"]) for row in pending_rows}
                )
            ),
            "inventory_only": args.resume_inventory_only,
            "expected_run_count": len(rows),
            "recovered_complete_run_count": len(recovered),
            "pending_run_count": len(pending_rows),
            "incomplete_existing_run_count": len(
                incomplete_existing_run_indexes
            ),
            "recovered_run_indexes": sorted(
                int(result["run_index"]) for result in recovered
            ),
            "pending_run_indexes": [
                int(row["run_index"]) for row in pending_rows
            ],
            "incomplete_existing_run_indexes": sorted(
                incomplete_existing_run_indexes
            ),
            "temporal_coverage_policy": TEMPORAL_COVERAGE_POLICY,
        }
        _write_json(output_root / "resume_inventory.json", resume_inventory)
        if not resume_inventory["inventory_pass"]:
            raise SystemExit("Resume inventory is internally inconsistent")
    else:
        manifest_copy.write_bytes(manifest_path.read_bytes())
        pending_rows = rows
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    load_strategy = (
        None if args.safetensors_load_strategy == "default"
        else args.safetensors_load_strategy
    )
    if args.resume and prior_engine_metadata:
        expected_backend = {
            "model": args.model,
            "model_revision": args.model_revision,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "thinking_enabled": False,
        }
        mismatches = {
            key: (prior_engine_metadata.get(key), expected)
            for key, expected in expected_backend.items()
            if prior_engine_metadata.get(key) != expected
        }
        if mismatches:
            raise SystemExit(
                f"Resume backend differs from prior execution: {mismatches}"
            )
    if args.resume_inventory_only:
        print(
            json.dumps(
                {
                    "inventory_pass": resume_inventory["inventory_pass"],
                    "expected_run_count": resume_inventory[
                        "expected_run_count"
                    ],
                    "recovered_complete_run_count": resume_inventory[
                        "recovered_complete_run_count"
                    ],
                    "pending_run_count": resume_inventory[
                        "pending_run_count"
                    ],
                    "incomplete_existing_run_count": resume_inventory[
                        "incomplete_existing_run_count"
                    ],
                    "temporal_coverage_policy": resume_inventory[
                        "temporal_coverage_policy"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    backend = VLLMOfflineBackend(
        model=args.model,
        model_revision=args.model_revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=False,
        enable_prefix_caching=not args.disable_prefix_caching,
        safetensors_load_strategy=load_strategy,
    )

    batch_started = time.perf_counter()
    load_seconds = backend.load()
    prior_engine_load_count = int(
        prior_engine_metadata.get("engine_load_count", 0)
    )
    engine_segments = list(prior_engine_metadata.get("execution_segments", []))
    if prior_engine_metadata and not engine_segments:
        engine_segments.append(
            {
                "engine_load_index": prior_engine_load_count or 1,
                "engine_load_seconds": prior_engine_metadata.get(
                    "engine_load_seconds"
                ),
                "started_at_utc": prior_engine_metadata.get("started_at_utc"),
                "source": "initial_execution",
            }
        )
    engine_segments.append(
        {
            "engine_load_index": prior_engine_load_count + 1,
            "engine_load_seconds": load_seconds,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "resume_execution" if args.resume else "initial_execution",
        }
    )
    engine_metadata = {
        "model": backend.model,
        "model_revision": backend.model_revision,
        "tensor_parallel_size": backend.tensor_parallel_size,
        "max_model_len": backend.max_model_len,
        "max_output_tokens": backend.max_output_tokens,
        "thinking_enabled": backend.enable_thinking,
        "prefix_caching_enabled": backend.enable_prefix_caching,
        "safetensors_load_strategy": backend.safetensors_load_strategy,
        "engine_load_seconds": sum(
            float(segment.get("engine_load_seconds") or 0.0)
            for segment in engine_segments
        ),
        "engine_load_count": len(engine_segments),
        "execution_segments": engine_segments,
        "resumed_execution": args.resume,
        "manifest_sha256": manifest_sha256,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "orchestration_mode": ORCHESTRATION_MODE,
        "configured_max_workers": min(
            args.max_workers, max(len(pending_rows), 1)
        ),
        "inference_batch_wait_ms": args.inference_batch_wait_ms,
        "batch_name": args.batch_name,
    }
    _write_json(output_root / "engine_metadata.json", engine_metadata)

    completed, broker_summary = run_parallel_workers(
        pending_rows,
        output_root,
        backend,
        engine_metadata,
        max_workers=min(args.max_workers, max(len(pending_rows), 1)),
        inference_batch_wait_seconds=max(
            0.0, args.inference_batch_wait_ms / 1000.0
        ),
        batch_name=args.batch_name,
        initial_completed=recovered,
        prior_progress=prior_progress,
        total_expected_run_count=len(rows),
    )

    _write_json(
        output_root / "batch_execution.json",
        {
            "expected_run_count": len(rows),
            "completed_run_count": len(completed),
            "engine_load_count": len(engine_segments),
            "engine_load_seconds": engine_metadata["engine_load_seconds"],
            "execution_segment_count": len(engine_segments),
            "resumed_execution": args.resume,
            "recovered_complete_run_count": len(recovered),
            "newly_completed_run_count": len(completed) - len(recovered),
            "total_elapsed_seconds": time.perf_counter() - batch_started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **broker_summary,
            "runs": completed,
        },
    )


if __name__ == "__main__":
    main()
