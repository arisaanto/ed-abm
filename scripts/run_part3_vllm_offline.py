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

PROJECT_DIR = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-thinking", action="store_true", default=config.VLLM_ENABLE_THINKING)
    parser.add_argument(
        "--reject-fixtures",
        action="store_true",
        help="Fail if any packet is marked fixture_only_not_scientific_data",
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
        "fixtures_rejected": args.reject_fixtures,
        "selected_packet_sha256": _canonical_sha256(packets),
        "selected_prompt_id_sha256": _canonical_sha256(prompt_ids),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    backend = VLLMOfflineBackend(
        model=args.model,
        model_revision=args.model_revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=args.enable_thinking,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        for packet_batch in _chunks(packets, max(1, args.batch_size)):
            for result in backend.generate_prompt_packets(packet_batch):
                handle.write(json.dumps(result, sort_keys=True) + "\n")
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
        }
    )
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    main()
