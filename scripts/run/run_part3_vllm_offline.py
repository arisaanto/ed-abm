#!/usr/bin/env python3
"""Run schema-constrained Part 3 prompt packets with vLLM offline inference.

This is an explicit GPU workflow. Importing this script does not initialize a
model, and it is never called by the default ABM pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from src.vllm_backend import (
    VLLMOfflineBackend,
    packet_json_schema,
    scientific_packet_prompt_leakage,
)


def _packet_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    paths = sorted(input_path.glob("*_prompt_packets.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No *_prompt_packets.jsonl files found in {input_path}")
    return paths


def _read_packets(
    paths: Iterable[Path],
    limit: int | None,
    *,
    reject_fixtures: bool = False,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            packet = json.loads(line)
            if not packet.get("prompt_id") or not packet.get("packet_type"):
                raise ValueError(f"Missing prompt_id/packet_type in {path}:{line_number}")
            if reject_fixtures and packet.get("fixture_only_not_scientific_data"):
                raise ValueError(
                    f"Scientific pilot rejected schema fixture in {path}:{line_number}"
                )
            leakage = scientific_packet_prompt_leakage(packet)
            if leakage:
                raise ValueError(
                    f"Scientific packet exposes evaluator-only prompt terms in "
                    f"{path}:{line_number}: {leakage}"
                )
            if (
                packet.get("fixture_only_not_scientific_data") is False
                and packet.get("packet_type") == "in_simulation_decision"
                and packet.get("comparator_hidden_from_model") is not True
            ):
                raise ValueError(
                    f"Scientific packet lacks comparator blinding marker in "
                    f"{path}:{line_number}"
                )
            packet_json_schema(str(packet["packet_type"]), packet)
            packets.append(packet)
            if limit is not None and len(packets) >= limit:
                return packets
    return packets


def _chunks(rows: list[dict[str, Any]], size: int):
    for index in range(0, len(rows), size):
        yield rows[index:index + size]


def _canonical_sha256(values: Iterable[Any]) -> str:
    """Hash selected scientific inputs independently of JSONL formatting."""

    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _schema_sha256(packet: dict[str, Any]) -> str:
    schema = packet_json_schema(str(packet.get("packet_type", "")), packet)
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _read_previous_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    results: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"Previous result is not an object at {path}:{line_number}")
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id:
            raise ValueError(f"Previous result lacks prompt_id at {path}:{line_number}")
        if prompt_id in results:
            raise ValueError(f"Duplicate previous result prompt_id: {prompt_id}")
        results[prompt_id] = row
    return results


def _validate_resume_metadata(
    response_path: Path,
    *,
    selected_packet_sha256: str,
    selected_prompt_id_sha256: str,
) -> dict[str, Any]:
    metadata_path = response_path.with_suffix(response_path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Resume metadata is required for packet-lineage verification: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "selected_packet_sha256": selected_packet_sha256,
        "selected_prompt_id_sha256": selected_prompt_id_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Previous responses were generated from a different packet inventory: "
            f"{mismatches}"
        )
    return metadata


def _revalidate_previous_results(
    packets: list[dict[str, Any]],
    previous_results: dict[str, dict[str, Any]],
    backend: VLLMOfflineBackend,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reuse only records that pass the current packet and semantic contracts."""

    packets_by_id = {str(packet["prompt_id"]): packet for packet in packets}
    unexpected = sorted(set(previous_results) - set(packets_by_id))
    if unexpected:
        raise ValueError(
            "Previous output contains prompt IDs outside the selected packet inventory: "
            f"{unexpected[:10]}"
        )

    reusable: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    rejection_reason_counts: dict[str, int] = {}
    for packet in packets:
        prompt_id = str(packet["prompt_id"])
        previous = previous_results.get(prompt_id)
        rejection = "missing_previous_result"
        normalized: dict[str, Any] | None = None
        if previous is not None:
            if previous.get("packet_type") != packet.get("packet_type"):
                rejection = "packet_type_mismatch"
            else:
                raw_response = previous.get("raw_response")
                try:
                    payload = (
                        json.loads(raw_response)
                        if isinstance(raw_response, str)
                        else None
                    )
                    if not isinstance(payload, dict):
                        raise TypeError("raw_response is not a JSON object")
                    normalized = backend.normalize_packet_response(packet, payload)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    rejection = f"{type(error).__name__}: {error}"

        if normalized is None:
            pending.append(packet)
            category = rejection.split(":", 1)[0]
            rejection_counts[category] = rejection_counts.get(category, 0) + 1
            rejection_reason_counts[rejection] = (
                rejection_reason_counts.get(rejection, 0) + 1
            )
            continue

        rebuilt = dict(previous)
        rebuilt.update(
            {
                "prompt_id": prompt_id,
                "packet_type": packet.get("packet_type"),
                "response_schema_sha256": _schema_sha256(packet),
                "response": normalized,
                "synthetic_design_probe_not_human_data": True,
                "resumed_from_previous_output": True,
                "local_revalidation_pass": True,
            }
        )
        rebuilt.pop("generation_error", None)
        reusable[prompt_id] = rebuilt

    repair_ids = [str(packet["prompt_id"]) for packet in pending]
    audit = {
        "previous_response_count": len(previous_results),
        "locally_reusable_response_count": len(reusable),
        "repair_packet_count": len(pending),
        "repair_packet_type_counts": {
            packet_type: sum(
                packet.get("packet_type") == packet_type for packet in pending
            )
            for packet_type in sorted(
                {str(packet.get("packet_type", "")) for packet in pending}
            )
        },
        "repair_prompt_id_sha256": _canonical_sha256(repair_ids),
        "local_revalidation_rejection_counts": rejection_counts,
        "local_revalidation_rejection_reason_counts": dict(
            sorted(rejection_reason_counts.items())
        ),
    }
    return reusable, pending, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prompt JSONL file or prompt_packets directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default=config.VLLM_MODEL_NAME)
    parser.add_argument(
        "--model-revision",
        help="Immutable Hugging Face commit hash used for weights and tokenizer",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=config.VLLM_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--max-model-len", type=int, default=config.VLLM_MAX_MODEL_LEN)
    parser.add_argument("--max-output-tokens", type=int, default=config.VLLM_MAX_OUTPUT_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=config.VLLM_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--semantic-retries",
        type=int,
        default=2,
        help="Bounded correction attempts for schema-valid but semantically invalid JSON",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-thinking", action="store_true", default=config.VLLM_ENABLE_THINKING)
    parser.add_argument(
        "--reject-fixtures",
        action="store_true",
        help="Fail if any packet is marked fixture_only_not_scientific_data",
    )
    parser.add_argument(
        "--resume-from",
        help=(
            "Previous response JSONL to revalidate locally. Only missing or invalid "
            "records are sent to the model; valid records are preserved."
        ),
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path for the dry-run/run inventory summary JSON",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate/count packets without importing vLLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    packets = _read_packets(
        _packet_paths(input_path),
        args.limit,
        reject_fixtures=args.reject_fixtures,
    )
    if not packets:
        raise RuntimeError("No prompt packets selected")

    estimates = [int(packet.get("estimated_tokens", 0) or 0) for packet in packets]
    prompt_ids = [str(packet["prompt_id"]) for packet in packets]
    selected_packet_sha256 = _canonical_sha256(packets)
    selected_prompt_id_sha256 = _canonical_sha256(prompt_ids)
    backend = VLLMOfflineBackend(
        model=args.model,
        model_revision=args.model_revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=args.enable_thinking,
        max_semantic_retries=args.semantic_retries,
    )
    reusable_by_id: dict[str, dict[str, Any]] = {}
    previous_by_id: dict[str, dict[str, Any]] = {}
    packets_for_inference = packets
    resume_audit: dict[str, Any] | None = None
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
        _validate_resume_metadata(
            resume_path,
            selected_packet_sha256=selected_packet_sha256,
            selected_prompt_id_sha256=selected_prompt_id_sha256,
        )
        previous_by_id = _read_previous_results(resume_path)
        reusable_by_id, packets_for_inference, resume_audit = (
            _revalidate_previous_results(
                packets,
                previous_by_id,
                backend,
            )
        )

    summary = {
        "packet_count": len(packets),
        "estimated_input_tokens": sum(estimates),
        "maximum_estimated_input_tokens": max(estimates, default=0),
        "model": args.model,
        "model_revision": args.model_revision,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "thinking_enabled": args.enable_thinking,
        "decoding": "greedy_schema_constrained",
        "sampling_temperature": 0.0,
        "maximum_semantic_retries": max(0, args.semantic_retries),
        "fixtures_rejected": args.reject_fixtures,
        "selected_packet_sha256": selected_packet_sha256,
        "selected_prompt_id_sha256": selected_prompt_id_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dry_run": args.dry_run,
        "resume_from": str(resume_path) if resume_path else None,
        "resume_audit": resume_audit,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.dry_run:
        return

    generated_by_id: dict[str, dict[str, Any]] = {}
    for packet_batch in _chunks(packets_for_inference, max(1, args.batch_size)):
        for result in backend.generate_prompt_packets(
            packet_batch, preserve_failures=True
        ):
            prompt_id = str(result.get("prompt_id", ""))
            previous = previous_by_id.get(prompt_id)
            if previous is not None:
                result["repair_from_previous_output"] = True
                result["prior_semantic_retry_count"] = int(
                    previous.get("semantic_retry_count", 0) or 0
                )
                result["prior_raw_response_sha256"] = hashlib.sha256(
                    str(previous.get("raw_response", "")).encode("utf-8")
                ).hexdigest()
            generated_by_id[prompt_id] = result

    merged_by_id = {**reusable_by_id, **generated_by_id}
    missing_after_merge = [
        str(packet["prompt_id"])
        for packet in packets
        if str(packet["prompt_id"]) not in merged_by_id
    ]
    if missing_after_merge:
        raise RuntimeError(
            f"Resume merge lacks {len(missing_after_merge)} selected packets"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        for packet in packets:
            prompt_id = str(packet["prompt_id"])
            handle.write(json.dumps(merged_by_id[prompt_id], sort_keys=True) + "\n")
            handle.flush()
    temporary_path.replace(output_path)
    import torch
    import vllm

    summary.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "vllm_version": vllm.__version__,
            "pytorch_version": torch.__version__,
            "pytorch_cuda_runtime": torch.version.cuda,
            "allocated_gpu_count": torch.cuda.device_count(),
            "inference_packet_count": len(packets_for_inference),
            "reused_response_count": len(reusable_by_id),
        }
    )
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    main()
