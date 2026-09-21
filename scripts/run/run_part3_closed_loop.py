#!/usr/bin/env python3
"""Run one bounded live Part 3 mechanics gate with offline vLLM inference.

This is not a scientific Part 3 batch. It proves that the already-grounded
categorical policy can act at an existing ABM opportunity without controlling
movement, workflow, geometry, patient state, or free-text generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from scripts.build.plan_part3_synthetic_study import (
    scientific_decision_packet,
    scientific_state_update_packet,
)
from scripts.run.run_single import run_single
from src.part3_closed_loop import (
    CausalDecision,
    CausalDecisionRequest,
    EvolvingStateUpdate,
    EvolvingStateUpdateRequest,
    Part3ClosedLoopController,
)
from src.interviews import required_output_schema
from src.personas import (
    balanced_cognitive_persona_assignments,
    cognitive_persona_by_id,
)
from src.vllm_backend import (
    PacketResponseDecodeError,
    VLLMOfflineBackend,
    packet_json_schema,
    scientific_packet_prompt_leakage,
)


def _validate_runtime_contract() -> None:
    """Fail before simulation/model loading if deployed Part 3 modules disagree."""

    schema = required_output_schema(
        "in_simulation_decision",
        decision_output_contract="categorical_causal_v1",
    )
    expected = {
        "decision_id",
        "selected_action",
        "selected_reason",
        "topic_family",
        "evidence_ids",
    }
    if set(schema) != expected:
        raise RuntimeError(
            "Deployed categorical decision schema is incompatible with the "
            f"closed-loop runner: {sorted(schema)}"
        )
    state_schema = packet_json_schema(
        "in_simulation_state_update",
        {
            "prompt_id": "state-contract-check",
            "evidence_ids": ["state-evidence-contract-check"],
        },
    )
    if set(state_schema.get("properties", {})) != {
        "checkpoint_id",
        "state",
        "evidence_by_dimension",
    }:
        raise RuntimeError("Deployed evolving-state schema is incompatible")


class VLLMCategoricalCausalProvider:
    """Adapt the validated Part 3 packet contract to the live controller API."""

    policy_name = "vllm_offline_categorical_causal_v1"

    def __init__(self, backend: VLLMOfflineBackend) -> None:
        self.backend = backend
        self.inference_log: list[dict] = []
        self.state_inference_log: list[dict] = []

    def decide_many(
        self, requests: Sequence[CausalDecisionRequest]
    ) -> list[CausalDecision]:
        personas = cognitive_persona_by_id()
        packets = []
        for request in requests:
            packet = scientific_decision_packet(
                personas[request.persona_id], dict(request.evidence)
            )
            leakage = scientific_packet_prompt_leakage(packet)
            if leakage:
                raise ValueError(
                    f"Evaluator-only information leaked into live packet: {leakage}"
                )
            packets.append(packet)

        started = time.perf_counter()
        try:
            results = self.backend.generate_prompt_packets(packets)
        except PacketResponseDecodeError as error:
            self.inference_log.append(
                {
                    "request": asdict(requests[0]) if len(requests) == 1 else None,
                    "packet": packets[0] if len(packets) == 1 else None,
                    "model": self.backend.model,
                    "model_revision": self.backend.model_revision,
                    "thinking_enabled": self.backend.enable_thinking,
                    "status": "malformed_json",
                    "prompt_id": error.prompt_id,
                    "raw_response": error.raw_response,
                    "error": str(error.original_error),
                    "free_text_used_as_causal_input": False,
                }
            )
            raise
        elapsed_seconds = time.perf_counter() - started
        decisions = []
        for request, packet, result in zip(requests, packets, results):
            response = result["response"]
            decisions.append(
                CausalDecision(
                    decision_id=request.decision_id,
                    evidence_id=request.evidence_id,
                    persona_id=request.persona_id,
                    selected_action=str(response["selected_action"]),
                    selected_reason=str(response["selected_reason"]),
                    topic_family=str(response["topic_family"]),
                    evidence_ids=(request.evidence_id,),
                    policy_name=self.policy_name,
                )
            )
            self.inference_log.append(
                {
                    "request": asdict(request),
                    "packet": packet,
                    "packet_sha256": hashlib.sha256(
                        json.dumps(packet, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "model": self.backend.model,
                    "model_revision": self.backend.model_revision,
                    "thinking_enabled": self.backend.enable_thinking,
                    "elapsed_seconds_for_batch": elapsed_seconds,
                    "result": result,
                    "free_text_used_as_causal_input": False,
                }
            )
        return decisions

    def update_states_many(
        self, requests: Sequence[EvolvingStateUpdateRequest]
    ) -> list[EvolvingStateUpdate]:
        personas = cognitive_persona_by_id()
        packets = [
            scientific_state_update_packet(
                personas[request.persona_id], request
            )
            for request in requests
        ]
        leakage_by_prompt = {
            str(packet["prompt_id"]): scientific_packet_prompt_leakage(packet)
            for packet in packets
        }
        leakage_by_prompt = {
            key: value for key, value in leakage_by_prompt.items() if value
        }
        if leakage_by_prompt:
            raise ValueError(
                f"Evaluator-only information leaked into state packets: {leakage_by_prompt}"
            )
        started = time.perf_counter()
        results = self.backend.generate_prompt_packets(packets)
        elapsed_seconds = time.perf_counter() - started
        updates = []
        for request, packet, result in zip(requests, packets, results):
            response = result["response"]
            updates.append(
                EvolvingStateUpdate(
                    checkpoint_id=request.checkpoint_id,
                    persona_id=request.persona_id,
                    state={
                        key: int(value)
                        for key, value in response["state"].items()
                    },
                    evidence_by_dimension={
                        name: tuple(values)
                        for name, values in response[
                            "evidence_by_dimension"
                        ].items()
                    },
                    policy_name="vllm_offline_evidence_linked_state_v1",
                )
            )
            self.state_inference_log.append(
                {
                    "request": asdict(request),
                    "packet": packet,
                    "packet_sha256": hashlib.sha256(
                        json.dumps(packet, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "model": self.backend.model,
                    "model_revision": self.backend.model_revision,
                    "thinking_enabled": self.backend.enable_thinking,
                    "elapsed_seconds_for_batch": elapsed_seconds,
                    "result": result,
                    "free_text_used_as_causal_input": False,
                }
            )
        return updates


def _persona_assignment(simulation, assignment_round: int) -> dict[int, str]:
    staff_ids = [int(agent.gid) for agent in simulation.staff_agents]
    rows = balanced_cognitive_persona_assignments(staff_ids)
    selected = {
        int(row["staff_id"]): str(row["persona_id"])
        for row in rows
        if int(row["assignment_round"]) == assignment_round
    }
    if set(selected) != set(staff_ids):
        raise ValueError(
            f"Assignment round {assignment_round} did not cover the staff roster"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=config.VLLM_MODEL_NAME)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=config.VLLM_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--scenario", default="normal_load", choices=sorted(config.SCENARIO_MODES))
    parser.add_argument(
        "--condition",
        default="both",
        choices=("baseline", "cockpit_only", "nursta_only", "both"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=int, default=9000)
    parser.add_argument("--warmup-seconds", type=int, default=7200)
    parser.add_argument("--assignment-round", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--max-model-decisions", type=int, default=5)
    parser.add_argument("--max-model-decisions-per-agent", type=int, default=2)
    parser.add_argument("--decision-window-seconds", type=int, default=0)
    parser.add_argument("--max-model-decisions-per-window", type=int, default=0)
    parser.add_argument("--model-decision-sample-rate", type=float, default=1.0)
    parser.add_argument("--evolving-state", action="store_true")
    parser.add_argument("--evolving-state-interval-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_runtime_contract()
    if args.duration <= args.warmup_seconds:
        raise SystemExit("Duration must exceed the warmup/controller start time")
    if args.max_output_tokens < config.VLLM_MAX_OUTPUT_TOKENS:
        raise SystemExit(
            "Closed-loop max-output-tokens cannot be lower than the validated "
            f"categorical setting ({config.VLLM_MAX_OUTPUT_TOKENS})"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = VLLMOfflineBackend(
        model=args.model,
        model_revision=args.model_revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=False,
    )
    provider = VLLMCategoricalCausalProvider(backend)

    def controller_factory(simulation):
        return Part3ClosedLoopController(
            provider,
            _persona_assignment(simulation, args.assignment_round),
            start_seconds=args.warmup_seconds,
            max_decisions_per_run=args.max_model_decisions,
            max_decisions_per_agent=args.max_model_decisions_per_agent,
            decision_window_seconds=args.decision_window_seconds,
            max_decisions_per_window=args.max_model_decisions_per_window,
            model_decision_sample_rate=args.model_decision_sample_rate,
            sampling_replication_id=args.assignment_round,
            evolving_state_enabled=args.evolving_state,
            evolving_state_interval_seconds=args.evolving_state_interval_seconds,
        )

    run_args = argparse.Namespace(
        scenario_mode=args.scenario,
        condition=args.condition,
        seed=args.seed,
        duration=args.duration,
        warmup_seconds=args.warmup_seconds,
        scenario_start_hour=10,
        validation_target="care_area",
        batch_name="part3_qwen36_closed_loop_gate",
        run_id=(
            f"{args.scenario}_{args.condition}_seed_{args.seed}_"
            f"assignment_round_{args.assignment_round}"
        ),
        output_dir=str(output_dir),
        sensitivity_parameter=None,
        sensitivity_level=None,
        sensitivity_value=None,
        sensitivity_default=None,
        export_part3_episodes=False,
        part3_max_episodes=0,
    )
    run_single(run_args, part3_controller_factory=controller_factory)


if __name__ == "__main__":
    main()
