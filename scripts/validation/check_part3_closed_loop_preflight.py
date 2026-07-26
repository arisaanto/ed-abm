#!/usr/bin/env python3
"""Run the complete Part 3 live integration path without loading an LLM.

This preflight uses the real packet builder, provider adapter, controller,
simulation hook, exports, and final verifier. Only token generation is replaced
with deterministic schema-valid JSON. It is a deployment/integration gate, not
a scientific simulation or persona result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Iterable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from scripts.build.build_part3_appraisal_packets import build_packets
from scripts.analysis.analyze_part3_appraisals import analyze as analyze_appraisals
from scripts.run.run_part3_closed_loop import (
    VLLMCategoricalCausalProvider,
    _persona_assignment,
    _validate_runtime_contract,
)
from scripts.run.run_part3_closed_loop_batch import (
    _read_manifest,
    run_parallel_workers,
)
from scripts.build.plan_part3_synthetic_study import scientific_decision_packet
from scripts.run.run_single import run_single
from scripts.validation.verify_part3_closed_loop_gate import (
    DECISION_KEYS,
    GROUNDED_MEMORY_FIELDS,
    VISIBLE_MEMORY_FIELDS,
    verify,
)
from scripts.validation.verify_part3_closed_loop_pilot import (
    _sampling_value as aggregate_sampling_value,
    verify as verify_aggregate_pilot,
)
from scripts.validation.verify_part3_vllm_responses import verify as verify_vllm_responses
from src.part3_closed_loop import (
    DeterministicMockCausalProvider,
    Part3ClosedLoopController,
)
from src.interaction import MemoryEvent, MemoryStream
from src.interviews import INTERVIEW_QUESTIONS
from src.personas import PERSONA_APPRAISAL_DIMENSIONS, cognitive_persona_by_id
from src.simulation import Simulation
from src.vllm_backend import (
    VLLMOfflineBackend,
    packet_json_schema,
    scientific_packet_prompt_leakage,
)


class DeterministicSchemaBackend:
    """Exercise live packet normalization without importing or loading vLLM."""

    model = "deterministic-schema-preflight"
    model_revision = "no-gpu-preflight-v1"
    enable_thinking = False

    def __init__(self, *, forced_action: str | None = None) -> None:
        if forced_action not in {None, "engage", "defer", "decline"}:
            raise ValueError(f"Unsupported deterministic action: {forced_action!r}")
        self._normalizer = VLLMOfflineBackend(
            model=self.model,
            model_revision=self.model_revision,
            tensor_parallel_size=1,
            max_model_len=8192,
            max_output_tokens=config.VLLM_MAX_OUTPUT_TOKENS,
            enable_thinking=False,
        )
        self.forced_action = forced_action
        self.call_count = 0
        self.raw_lengths: list[int] = []

    def generate_prompt_packets(
        self, packets: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        results = []
        actions = ("engage", "defer", "decline")
        for packet_value in packets:
            packet = dict(packet_value)
            schema = packet_json_schema("in_simulation_decision", packet)
            action = self.forced_action or actions[self.call_count % len(actions)]
            payload = {
                "decision_id": str(packet["prompt_id"]),
                "selected_action": action,
                "selected_reason": str(packet["allowed_reasons"][0]),
                "topic_family": str(packet["allowed_topic_families"][0]),
                "evidence_ids": [str(packet["evidence_ids"][0])],
            }
            raw = json.dumps(payload, sort_keys=True)
            self.raw_lengths.append(len(raw))
            normalized = self._normalizer.normalize_packet_response(packet, payload)
            results.append(
                {
                    "prompt_id": packet["prompt_id"],
                    "packet_type": packet["packet_type"],
                    "model": self.model,
                    "thinking_enabled": False,
                    "response_schema_sha256": hashlib.sha256(
                        json.dumps(schema, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "response": normalized,
                    "raw_response": raw,
                    "synthetic_design_probe_not_human_data": True,
                }
            )
            self.call_count += 1
        return results


def _appraisal_packet_audit(
    run_dir: Path,
    output_dir: Path,
    normalizer: VLLMOfflineBackend,
) -> dict[str, Any]:
    """Exercise post-run packet construction and response normalization without GPU."""

    args = argparse.Namespace(
        results_root=str(run_dir),
        output_dir=str(output_dir),
        replicates_per_stratum=1,
        max_events=18,
        allow_incomplete_strata=True,
    )
    result = build_packets(args)
    packet_path = (
        output_dir
        / "prompt_packets"
        / "end_of_shift_appraisal_prompt_packets.jsonl"
    )
    packets = [
        json.loads(line)
        for line in packet_path.read_text().splitlines()
        if line.strip()
    ]
    errors = []
    normalized_types = []
    response_rows = []
    for packet in packets:
        packet_type = str(packet["packet_type"])
        packet_json_schema(packet_type, packet)
        evidence_id = str(packet["evidence_ids"][0])
        if packet_type == "end_of_shift_survey_bundle":
            payload = {
                "responses": [
                    {
                        "dimension": dimension,
                        "rateability": "rateable",
                        "score_1_to_7": 4,
                        "confidence_1_to_5": 3,
                        "evidence_event_ids": [evidence_id],
                        "short_rationale": "The supplied shift contains a relevant moment.",
                        "uncertainty_note": "One shift cannot establish a stable preference.",
                    }
                    for dimension in PERSONA_APPRAISAL_DIMENSIONS
                ],
                "not_human_data": True,
            }
        elif packet_type == "end_of_shift_interview_bundle":
            payload = {
                "answers": [
                    {
                        "question_id": question["question_id"],
                        "answer": "One recurring moment shaped how I approached the work.",
                        "grounded_pattern": "A cited work moment occurred in the supplied shift.",
                        "persona_conditioned_interpretation": "That moment mattered to this work orientation.",
                        "latent_need": None,
                        "design_hypothesis": None,
                        "tradeoff": None,
                        "evidence_event_ids": [evidence_id],
                        "uncertainty_note": "This is one simulated shift.",
                    }
                    for question in INTERVIEW_QUESTIONS
                ],
                "role_perspective": str(packet["role"]),
                "claim_layer_contract": "grounded_pattern_interpretation_conjecture_v1",
                "not_human_data": True,
            }
        else:
            errors.append(f"Unexpected appraisal packet type: {packet_type}")
            continue
        try:
            normalized = normalizer.normalize_packet_response(packet, payload)
        except Exception as exc:  # pragma: no cover - surfaced in preflight output
            errors.append(f"{packet['prompt_id']}: {exc}")
            continue
        normalized_types.append(packet_type)
        if normalized.get("not_human_data") is not True:
            errors.append(f"{packet['prompt_id']}: normalized marker missing")
        schema = packet_json_schema(packet_type, packet)
        response_rows.append(
            {
                "prompt_id": packet["prompt_id"],
                "packet_type": packet_type,
                "model": "deterministic-appraisal-preflight",
                "thinking_enabled": False,
                "response_schema_sha256": hashlib.sha256(
                    json.dumps(schema, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "response": normalized,
                "raw_response": json.dumps(payload, sort_keys=True),
                "synthetic_design_probe_not_human_data": True,
            }
        )
    expected_count = 18
    if len(packets) != expected_count:
        errors.append(
            f"Expected {expected_count} appraisal packets from nine staff, found {len(packets)}"
        )
    response_path = output_dir / "deterministic_appraisal_responses.jsonl"
    response_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in response_rows)
    )
    verification = verify_vllm_responses(
        packet_path,
        response_path,
        limit=None,
        reject_fixtures=True,
    )
    verification_path = output_dir / "deterministic_appraisal_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    if verification.get("technical_verification_pass") is not True:
        errors.append("Deterministic appraisal response verification failed")
    analysis = analyze_appraisals(
        argparse.Namespace(
            packets=str(packet_path),
            responses=str(response_path),
            verification=str(verification_path),
            output_dir=str(output_dir / "deterministic_appraisal_analysis"),
        )
    )
    if analysis.get("analysis_pass") is not True:
        errors.append("Deterministic appraisal analysis export failed")
    return {
        "audit_pass": not errors and result.get("preflight_pass") is True,
        "packet_count": len(packets),
        "packet_type_counts": dict(Counter(normalized_types)),
        "maximum_estimated_tokens": result.get("maximum_estimated_tokens"),
        "claim_layer_contract": result.get("claim_layer_contract"),
        "response_verification_pass": verification.get(
            "technical_verification_pass"
        ),
        "human_review_required": verification.get(
            "appraisal_human_review_required"
        ),
        "analysis_export_pass": analysis.get("analysis_pass"),
        "gpu_used": False,
        "errors": errors,
    }


def _exogenous_arrival_stream_audit() -> dict[str, Any]:
    """Prove Part 3 common arrivals without changing the Part 1/2 RNG path."""

    legacy = Simulation(random_seed=17, scenario_mode="normal_load")
    baseline = Simulation(
        random_seed=17,
        scenario_mode="normal_load",
        condition_name="baseline",
        part3_isolate_exogenous_arrival_stream=True,
    )
    both = Simulation(
        random_seed=17,
        scenario_mode="normal_load",
        condition_name="both",
        part3_isolate_exogenous_arrival_stream=True,
    )
    for _ in range(1000):
        baseline.random.random()
    baseline_arrival_draws = [
        baseline.arrival_random.random() for _ in range(32)
    ]
    both_arrival_draws = [both.arrival_random.random() for _ in range(32)]
    errors = []
    if legacy.arrival_random is not legacy.random:
        errors.append("Part 1/2 no longer uses its validated shared RNG lifecycle")
    if baseline.arrival_random is baseline.random:
        errors.append("Part 3 did not isolate exogenous arrival randomness")
    if baseline_arrival_draws != both_arrival_draws:
        errors.append(
            "Matched Part 3 conditions do not share exogenous arrival draws"
        )
    if baseline.part3_exogenous_arrival_stream_version != (
        "scenario_seed_common_random_numbers_v1"
    ):
        errors.append("Unexpected Part 3 arrival-stream version")
    return {
        "audit_pass": not errors,
        "gpu_used": False,
        "part1_part2_legacy_rng_preserved": legacy.arrival_random is legacy.random,
        "part3_arrival_rng_isolated": baseline.arrival_random is not baseline.random,
        "matched_condition_draws_identical": (
            baseline_arrival_draws == both_arrival_draws
        ),
        "arrival_stream_version": (
            baseline.part3_exogenous_arrival_stream_version
        ),
        "errors": errors,
    }


def _grounded_memory_policy_audit() -> dict[str, Any]:
    """Exercise causal-memory inclusion, exclusion, and provenance rules."""

    stream = MemoryStream()

    def event(
        memory_id: str,
        *,
        timestamp: int,
        event_type: str = "communicative_interaction",
        outcome: str = "realized",
        partner_id: int = 2,
        patient_id: int | None = None,
        reason_type: str = "HANDOFF_NEED",
        interaction_type: str = "opportunistic_station",
        zone_id: str = "COCPIT",
    ) -> MemoryEvent:
        source_event_id = f"event:{memory_id}"
        return MemoryEvent(
            memory_id=memory_id,
            timestamp=timestamp,
            event_type=event_type,
            owner_agent_id=1,
            owner_agent_name="hidden owner",
            owner_role="Nurse",
            partner_id=partner_id,
            partner_name="hidden partner",
            partner_role="Nurse",
            patient_id=patient_id,
            esi_level=2 if patient_id is not None else None,
            zone_id=zone_id,
            condition_name="baseline",
            model_variant="perception_rule_v3",
            topic="patient_status_update",
            duration_seconds=30,
            interaction_type=interaction_type,
            summary="prose must not enter the causal packet",
            importance=0.7,
            emotional_tags=[],
            retrieval_keywords=[],
            source_event_id=source_event_id,
            support_ids=[source_event_id],
            owner_was_initiator=True,
            reason_type=reason_type,
            outcome=outcome,
        )

    stream.add(event("same_colleague", timestamp=9600, partner_id=2))
    stream.add(
        event(
            "same_patient_other_colleague",
            timestamp=9500,
            partner_id=3,
            patient_id=1004,
        )
    )
    stream.add(
        event(
            "unrelated_recent",
            timestamp=8950,
            partner_id=4,
            reason_type="POST_TASK_UPDATE",
            interaction_type="opportunistic_corridor",
            zone_id="CORR03",
        )
    )
    stream.add(event("stale_same_colleague", timestamp=1000, partner_id=2))
    stream.add(
        event(
            "missed_choice",
            timestamp=9700,
            event_type="missed_opportunity",
            outcome="missed",
        )
    )
    stream.add(
        event(
            "generated_reflection",
            timestamp=9800,
            event_type="reflection",
            outcome="reflection",
        )
    )

    retrieved = stream.retrieve_grounded_part3_context(
        current_timestamp=10000,
        partner_id=2,
        patient_context_id=1004,
        reason_type="HANDOFF_NEED",
        interaction_type="opportunistic_station",
        zone_id="COCPIT",
        limit=5,
        salience_modifiers={"coordination": 1.2, "handoff": 1.3},
    )
    returned_ids = [str(row["memory_id"]) for row in retrieved]
    expected_ids = {"same_colleague", "same_patient_other_colleague"}
    errors = []
    if set(returned_ids) != expected_ids:
        errors.append(
            f"Grounded retrieval returned {returned_ids}, expected {sorted(expected_ids)}"
        )
    if any("summary" in row or "selected_action" in row for row in retrieved):
        errors.append("Generated prose or prior categorical choice entered causal memory")
    if retrieved:
        exported_memory_keys = {
            "agent_id",
            "decision_id",
            "decision_evidence_id",
            *retrieved[0].keys(),
        }
        if exported_memory_keys != GROUNDED_MEMORY_FIELDS:
            errors.append(
                "Grounded-memory exporter/verifier schema mismatch: "
                f"missing={sorted(GROUNDED_MEMORY_FIELDS - exported_memory_keys)}, "
                f"unexpected={sorted(exported_memory_keys - GROUNDED_MEMORY_FIELDS)}"
            )
    if not any(row["same_partner"] for row in retrieved):
        errors.append("Same-colleague continuity was not represented")
    if not any(
        row["same_patient_context"] and not row["same_partner"]
        for row in retrieved
    ):
        errors.append("Bounded same-patient cross-colleague relay was not represented")

    observable_salience_features = {
        "coordination",
        "handoff",
        "patient_facing",
        "high_acuity",
        "interruption",
        "context_switch",
        "tradeoff",
        "task_continuity",
    }
    unsupported_modifiers = {
        persona_id: sorted(
            set(persona.memory_salience_modifiers) - observable_salience_features
        )
        for persona_id, persona in cognitive_persona_by_id().items()
        if set(persona.memory_salience_modifiers) - observable_salience_features
    }
    if unsupported_modifiers:
        errors.append(f"Persona memory modifiers are not observable: {unsupported_modifiers}")

    return {
        "audit_pass": not errors,
        "memory_policy": "grounded_realized_interactions_v1",
        "maximum_age_seconds": 7200,
        "returned_memory_ids": returned_ids,
        "excluded_memory_ids": sorted(
            {
                "unrelated_recent",
                "stale_same_colleague",
                "missed_choice",
                "generated_reflection",
            }
        ),
        "same_colleague_present": any(row["same_partner"] for row in retrieved),
        "same_patient_cross_colleague_present": any(
            row["same_patient_context"] and not row["same_partner"]
            for row in retrieved
        ),
        "unsupported_persona_memory_modifiers": unsupported_modifiers,
        "errors": errors,
    }


def _module_sources() -> dict[str, str]:
    import src.interviews as interviews
    import src.part3_closed_loop as closed_loop
    import src.personas as personas
    import src.simulation as simulation
    import src.vllm_backend as backend
    import scripts.build.plan_part3_synthetic_study as planner
    import scripts.run.run_part3_closed_loop as runner
    import scripts.run.run_single as single
    import scripts.validation.verify_part3_closed_loop_gate as verifier

    modules = {
        "interviews": interviews,
        "part3_closed_loop": closed_loop,
        "personas": personas,
        "simulation": simulation,
        "vllm_backend": backend,
        "planner": planner,
        "runner": runner,
        "run_single": single,
        "verifier": verifier,
    }
    sources = {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}
    outside = {
        name: source
        for name, source in sources.items()
        if PROJECT_DIR not in Path(source).parents
    }
    if outside:
        raise RuntimeError(f"Part 3 modules imported outside the deployed project: {outside}")
    return sources


def _deployed_source_hashes(
    guarded_paths: Iterable[str] = (),
) -> dict[str, str]:
    """Fingerprint every loaded project module plus the guarded GPU job."""

    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and PROJECT_DIR in path.parents:
            paths.add(path)
    paths.add(
        (
            PROJECT_DIR
            / "jobs/snellius_part3_qwen36_closed_loop_main_n10.sbatch"
        ).resolve()
    )
    for relative in (
        "scripts/run/run_part3_closed_loop_batch.py",
        "scripts/validation/check_part3_closed_loop.py",
    ):
        path = (PROJECT_DIR / relative).resolve()
        if path.is_file():
            paths.add(path)
    for value in guarded_paths:
        path = (PROJECT_DIR / value).resolve()
        if path.is_file() and PROJECT_DIR in path.parents:
            paths.add(path)
    return {
        str(path.relative_to(PROJECT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _static_packet_audit(
    episode_root: Path, normalizer: VLLMOfflineBackend
) -> dict[str, Any]:
    """Check every retained real episode, with and without causal memory."""

    episode_files = sorted(episode_root.rglob("part3_decision_episodes.jsonl"))
    episodes = [
        json.loads(line)
        for path in episode_files
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    errors = []
    packet_count = 0
    memory_packet_count = 0
    maximum_estimated_tokens = 0
    personas = cognitive_persona_by_id()
    grounded_memory_fixture = {
        "memory_id": "mem_1_both_1_1",
        "timestamp": 6800,
        "seconds_ago": 200,
        "event_type": "communicative_interaction",
        "outcome": "realized",
        "partner_id": 2,
        "partner_role": "Nurse",
        "owner_was_initiator": True,
        "patient_context_id": 1004,
        "zone": "COCPIT",
        "topic_family": "next_steps_and_planning",
        "interaction_type": "opportunistic_station",
        "reason_type": "POST_TASK_UPDATE",
        "importance": 0.7,
        "same_partner": True,
        "same_patient_context": True,
        "retrieval_rank": 1,
        "retrieval_score": 2.4,
        "retrieval_components": {
            "recency": 0.8,
            "relevance": 1.0,
            "importance": 0.7,
            "salience_multiplier": 1.0,
        },
        "salience_features": ["coordination", "task_continuity"],
        "retrieval_match_features": [
            "same_colleague",
            "same_patient_context",
        ],
        "source_event_id": "comm_1_both_1",
        "support_ids": ["comm_1_both_1"],
        "source": "realized_communicative_interaction",
        "memory_policy": "grounded_realized_interactions_v1",
    }
    for episode in episodes:
        for persona_id, persona in personas.items():
            for with_grounded_memory in (False, True):
                evidence = json.loads(json.dumps(episode))
                if with_grounded_memory:
                    evidence["memory_state_before"] = [grounded_memory_fixture]
                try:
                    packet = scientific_decision_packet(persona, evidence)
                    visible_memory = packet.get("llm_visible_evidence", {}).get(
                        "memory_state_before", []
                    )
                    if with_grounded_memory and (
                        len(visible_memory) != 1
                        or set(visible_memory[0]) != VISIBLE_MEMORY_FIELDS
                    ):
                        observed_keys = (
                            set(visible_memory[0]) if visible_memory else set()
                        )
                        raise ValueError(
                            "Visible-memory exporter/verifier schema mismatch: "
                            f"missing={sorted(VISIBLE_MEMORY_FIELDS - observed_keys)}, "
                            f"unexpected={sorted(observed_keys - VISIBLE_MEMORY_FIELDS)}"
                        )
                    leakage = scientific_packet_prompt_leakage(packet)
                    if leakage:
                        raise ValueError(f"prompt leakage: {leakage}")
                    schema = packet_json_schema("in_simulation_decision", packet)
                    payload = {
                        "decision_id": packet["prompt_id"],
                        "selected_action": packet["feasible_actions"][
                            packet_count % len(packet["feasible_actions"])
                        ],
                        "selected_reason": packet["allowed_reasons"][0],
                        "topic_family": packet["allowed_topic_families"][0],
                        "evidence_ids": [packet["evidence_ids"][0]],
                    }
                    if normalizer.normalize_packet_response(packet, payload) != payload:
                        raise ValueError("response normalization changed a valid payload")
                    if schema.get("additionalProperties") is not False:
                        raise ValueError("categorical schema permits extra fields")
                    maximum_estimated_tokens = max(
                        maximum_estimated_tokens, int(packet["estimated_tokens"])
                    )
                    packet_count += 1
                    memory_packet_count += int(with_grounded_memory)
                except Exception as error:
                    errors.append(
                        {
                            "evidence_id": episode.get("evidence_id"),
                            "persona_id": persona_id,
                            "with_grounded_memory": with_grounded_memory,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    if len(episodes) != 320:
        errors.append(
            {
                "error_type": "EpisodeCountError",
                "error": f"Expected 320 retained evidence episodes, got {len(episodes)}",
            }
        )
    if maximum_estimated_tokens > 8192:
        errors.append(
            {
                "error_type": "ContextLengthError",
                "error": (
                    f"Packet estimate {maximum_estimated_tokens} exceeds 8192 tokens"
                ),
            }
        )
    return {
        "audit_pass": not errors,
        "episode_file_count": len(episode_files),
        "episode_count": len(episodes),
        "persona_count": len(personas),
        "packet_variants_checked": packet_count,
        "memory_packet_count": memory_packet_count,
        "maximum_estimated_tokens": maximum_estimated_tokens,
        "errors": errors,
    }


def _temporal_budget_audit(episode_root: Path) -> dict[str, Any]:
    """Exercise all five quota windows without relying on event frequency."""

    episode_file = next(
        iter(sorted(episode_root.rglob("part3_decision_episodes.jsonl"))),
        None,
    )
    if episode_file is None:
        return {"audit_pass": False, "errors": ["No retained episode file"]}
    first_line = next(
        (line for line in episode_file.read_text().splitlines() if line.strip()),
        None,
    )
    if first_line is None:
        return {"audit_pass": False, "errors": ["Retained episode file is empty"]}
    base_episode = json.loads(first_line)
    agent_id = int(base_episode["metadata"]["staff_id"])
    backend = DeterministicSchemaBackend()
    controller = Part3ClosedLoopController(
        VLLMCategoricalCausalProvider(backend),
        {agent_id: "team_connector"},
        start_seconds=1000,
        max_decisions_per_run=5,
        max_decisions_per_agent=10,
        decision_window_seconds=100,
        max_decisions_per_window=1,
    )
    for window_index in range(5):
        for within_window_index in (1, 2):
            episode = deepcopy(base_episode)
            timestep = 1000 + window_index * 100 + within_window_index
            episode["timestep"] = timestep
            episode["metadata"]["timestep"] = timestep
            episode["evidence_id"] = (
                f"temporal-preflight:w{window_index}:e{within_window_index}"
            )
            controller.decide_episode(episode)
    summary = controller.summary()
    expected_windows = {str(index): 1 for index in range(5)}
    errors = []
    if summary["provider_calls_by_window"] != expected_windows:
        errors.append("Provider calls were not distributed one per test window")
    if summary["stats"].get("provider_calls") != 5:
        errors.append("Temporal audit did not make exactly five provider calls")
    if summary["stats"].get("window_decision_cap", 0) < 4:
        errors.append("Temporal audit did not exercise window-cap fallback")
    decision_key_mismatches = [
        sorted(set(row) ^ DECISION_KEYS)
        for row in controller.decision_log
        if set(row) != DECISION_KEYS
    ]
    if decision_key_mismatches:
        errors.append(
            "Decision exporter/verifier schema mismatch: "
            f"{decision_key_mismatches}"
        )
    return {
        "audit_pass": not errors,
        "provider_calls_by_window": summary["provider_calls_by_window"],
        "provider_call_count": summary["stats"].get("provider_calls", 0),
        "window_cap_fallback_count": summary["stats"].get(
            "window_decision_cap", 0
        ),
        "run_cap_fallback_count": summary["stats"].get("run_decision_cap", 0),
        "errors": errors,
    }


def _deterministic_sampling_audit(episode_root: Path) -> dict[str, Any]:
    """Prove that proportional selection is stable and independent of ABM RNG."""

    episode_file = next(
        iter(sorted(episode_root.rglob("part3_decision_episodes.jsonl"))),
        None,
    )
    if episode_file is None:
        return {"audit_pass": False, "errors": ["No retained episode file"]}
    first_line = next(
        (line for line in episode_file.read_text().splitlines() if line.strip()),
        None,
    )
    if first_line is None:
        return {"audit_pass": False, "errors": ["Retained episode file is empty"]}
    base_episode = json.loads(first_line)
    agent_id = int(base_episode["metadata"]["staff_id"])
    sample_rate = 0.10

    key_probe_controller = Part3ClosedLoopController(
        DeterministicMockCausalProvider(),
        {agent_id: "team_connector"},
        model_decision_sample_rate=sample_rate,
        sampling_replication_id=1,
    )
    baseline_probe = deepcopy(base_episode)
    both_probe = deepcopy(base_episode)
    baseline_probe["metadata"]["condition"] = "baseline"
    both_probe["metadata"]["condition"] = "both"
    baseline_key = key_probe_controller._sampling_key(baseline_probe)
    both_key = key_probe_controller._sampling_key(both_probe)
    next_round_controller = Part3ClosedLoopController(
        DeterministicMockCausalProvider(),
        {agent_id: "team_connector"},
        model_decision_sample_rate=sample_rate,
        sampling_replication_id=2,
    )
    next_round_key = next_round_controller._sampling_key(baseline_probe)

    def run_once() -> tuple[list[bool], list[float], list[str], dict[str, Any]]:
        controller = Part3ClosedLoopController(
            DeterministicMockCausalProvider(),
            {agent_id: "team_connector"},
            start_seconds=0,
            max_decisions_per_run=1000,
            max_decisions_per_agent=1000,
            model_decision_sample_rate=sample_rate,
            sampling_replication_id=1,
        )
        for index in range(1000):
            episode = deepcopy(base_episode)
            evidence_id = f"sampling-preflight:{index:04d}"
            episode["evidence_id"] = evidence_id
            episode["timestep"] = index
            episode["metadata"]["timestep"] = index
            controller.decide_episode(episode)
        return (
            [bool(row["sampled_for_model"]) for row in controller.decision_log],
            [float(row["sampling_value"]) for row in controller.decision_log],
            [str(row["sampling_key"]) for row in controller.decision_log],
            controller.summary(),
        )

    first_flags, first_values, first_keys, first_summary = run_once()
    second_flags, second_values, second_keys, second_summary = run_once()
    expected_flags = [
        Part3ClosedLoopController._sampling_value(sampling_key)
        < sample_rate
        for sampling_key in first_keys
    ]
    sampled_count = sum(first_flags)
    errors = []
    try:
        aggregate_values = [
            aggregate_sampling_value(sampling_key)
            for sampling_key in first_keys
        ]
    except Exception as error:
        aggregate_values = []
        errors.append(
            "Aggregate verifier sampling helper failed: "
            f"{type(error).__name__}: {error}"
        )
    else:
        if aggregate_values != first_values:
            errors.append(
                "Aggregate verifier sampling values differ from the controller"
            )
    if first_flags != expected_flags:
        errors.append("Controller selection does not match the SHA-256 threshold")
    if (
        first_flags != second_flags
        or first_values != second_values
        or first_keys != second_keys
    ):
        errors.append("Repeated sampling produced different selections")
    if first_summary != second_summary:
        errors.append("Repeated sampling produced different summaries")
    if baseline_key != both_key:
        errors.append("Condition labels changed a matched-opportunity sampling key")
    if baseline_key == next_round_key:
        errors.append("Assignment rounds did not create independent sampling keys")
    if first_summary["stats"].get("provider_calls", 0) != sampled_count:
        errors.append("Sampled episodes did not map one-to-one to provider calls")
    if first_summary["stats"].get("deterministic_sampling_skip", 0) != (
        1000 - sampled_count
    ):
        errors.append("Unselected episodes did not use the declared rule fallback")
    if not 70 <= sampled_count <= 130:
        errors.append("The deterministic 1,000-episode probe is implausibly far from 10%")
    return {
        "audit_pass": not errors,
        "sampling_method": (
            "deterministic_sha256_matched_opportunity_threshold_v1"
        ),
        "target_sample_rate": sample_rate,
        "eligible_episode_count": 1000,
        "sampled_episode_count": sampled_count,
        "observed_sample_rate": sampled_count / 1000,
        "repeat_identical": first_flags == second_flags,
        "aggregate_verifier_sampling_match": aggregate_values == first_values,
        "condition_matched": baseline_key == both_key,
        "assignment_rounds_independent": baseline_key != next_round_key,
        "provider_call_count": first_summary["stats"].get("provider_calls", 0),
        "fallback_count": first_summary["stats"].get(
            "deterministic_sampling_skip", 0
        ),
        "errors": errors,
    }


def _parallel_broker_audit(
    manifest_rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """Exercise spawned simulations and the shared inference broker without GPUs."""

    preferred = [
        row
        for row in manifest_rows
        if row["scenario"] == "normal_load"
        and int(row["seed"]) == 1
        and int(row["assignment_round"]) == 1
        and row["condition"] in {"baseline", "both"}
    ]
    preferred.sort(key=lambda row: (row["condition"] != "baseline", row["condition"]))
    rows = [deepcopy(row) for row in preferred[:2]]
    if len(rows) != 2:
        return {
            "audit_pass": False,
            "errors": [
                "Need matched normal-load Baseline/Both rows for the broker audit"
            ],
        }
    for run_index, row in enumerate(rows, 1):
        row.update(
            {
                "run_index": run_index,
                "duration_seconds": args.duration,
                "warmup_seconds": args.warmup_seconds,
                # The standalone integration probe above exercises call-cap
                # fallback. This miniature aggregate package instead mirrors
                # the pilot's nonbinding safety ceilings so the strict final
                # verifier can traverse its clean success path.
                "max_model_decisions": 300,
                "max_model_decisions_per_agent": 100,
                "decision_window_seconds": args.decision_window_seconds,
                "max_model_decisions_per_window": 100,
                # Sampling reproducibility and the preregistered 10% rate are
                # audited separately over 1,000 deterministic opportunities.
                # This two-run broker probe uses complete capture so a short,
                # low-opportunity run cannot fail merely because it sampled no
                # provider decision.
                "model_decision_sample_rate": 1.0,
                "minimum_covered_decision_windows": 0,
            }
        )
    # Broker completion must not depend on asynchronous worker arrival order.
    # A non-engage decision always yields a recorded missed-opportunity outcome;
    # the standalone integration audit separately exercises all action classes.
    backend = DeterministicSchemaBackend(forced_action="decline")
    metadata = {
        "model": backend.model,
        "model_revision": backend.model_revision,
        "thinking_enabled": backend.enable_thinking,
    }
    errors = []
    with tempfile.TemporaryDirectory(prefix="part3_parallel_preflight_") as temp:
        output_root = Path(temp) / "parallel"
        output_root.mkdir()
        completed, broker = run_parallel_workers(
            rows,
            output_root,
            backend,
            metadata,
            max_workers=2,
            inference_batch_wait_seconds=0.05,
        )
        observed_summaries = len(list(output_root.rglob("summary.json")))
        observed_inference_rows = sum(
            len(
                [
                    line
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
            )
            for path in output_root.rglob("part3_provider_inference.jsonl")
        )
        mini_manifest = Path(temp) / "manifest.csv"
        with mini_manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output_root / "manifest.csv").write_bytes(mini_manifest.read_bytes())
        mini_manifest_sha256 = hashlib.sha256(
            mini_manifest.read_bytes()
        ).hexdigest()
        (output_root / "engine_metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "engine_load_count": 1,
                    "prefix_caching_enabled": True,
                    "safetensors_load_strategy": "eager",
                    "orchestration_mode": broker["orchestration_mode"],
                    "manifest_sha256": mini_manifest_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        (output_root / "batch_execution.json").write_text(
            json.dumps(
                {
                    "engine_load_count": 1,
                    "expected_run_count": len(rows),
                    "completed_run_count": len(completed),
                    **broker,
                    "resumed_execution": False,
                    "recovered_complete_run_count": 0,
                    "newly_completed_run_count": len(completed),
                    "unique_completed_run_count": len(completed),
                    "runs": completed,
                },
                indent=2,
                sort_keys=True,
            )
        )
        aggregate_verification = verify_aggregate_pilot(
            output_root,
            mini_manifest,
            # This two-run audit proves process and broker integration. The
            # real pilot verifier separately requires grounded memory in every
            # scenario-condition cell.
            enforce_grounded_memory_coverage=False,
        )
    expected_calls = sum(
        int(row.get("provider_call_count", 0)) for row in completed
    )
    if len(completed) != 2 or observed_summaries != 2:
        errors.append("Parallel preflight did not complete both simulation workers")
    if broker.get("maximum_active_workers", 0) != 2:
        errors.append("Parallel preflight did not overlap two simulation workers")
    if broker.get("inference_packet_count") != expected_calls:
        errors.append("Broker packet count does not match worker provider calls")
    if observed_inference_rows != expected_calls:
        errors.append("Worker inference logs do not match broker packet count")
    if not all(row.get("verification_pass") for row in completed):
        errors.append("A parallel preflight run failed its closed-loop verifier")
    expected_aggregate_errors = {
        "One or more staff did not receive every persona exactly once",
        "Manifest does not match the declared pilot design (20 runs)",
        (
            "One or more scenario-condition study cells lack aggregate "
            "model-call coverage in a full-shift window"
        ),
    }
    unexpected_aggregate_errors = [
        error
        for error in aggregate_verification.get("errors", [])
        if error not in expected_aggregate_errors
    ]
    if aggregate_verification.get("observed_summary_count") != 2:
        errors.append("Aggregate verifier did not inspect both miniature runs")
    if aggregate_verification.get("provider_call_count") != expected_calls:
        errors.append("Aggregate verifier provider-call count is inconsistent")
    temporal_fidelity_failures = aggregate_verification.get(
        "temporal_sampling_fidelity_failures", []
    )
    if temporal_fidelity_failures:
        errors.append(
            "Aggregate verifier found sampled/provider-call mismatches in the "
            "miniature runs"
        )
    observed_fixture_cells = {
        (str(row["scenario"]), str(row["condition"])) for row in rows
    }
    observed_cell_temporal_failures = [
        failure
        for failure in aggregate_verification.get(
            "temporal_aggregate_coverage_failures", []
        )
        if (str(failure["scenario"]), str(failure["condition"]))
        in observed_fixture_cells
        and int(failure.get("eligible_decision_count", 0)) > 0
    ]
    if observed_cell_temporal_failures:
        errors.append(
            "Aggregate verifier found missing model-call coverage in an "
            "observed miniature scenario-condition cell"
        )
    if unexpected_aggregate_errors:
        errors.append(
            "Aggregate verifier failed its miniature output audit: "
            + "; ".join(unexpected_aggregate_errors)
        )
    return {
        "audit_pass": not errors,
        "completed_run_count": len(completed),
        "observed_summary_count": observed_summaries,
        "observed_inference_row_count": observed_inference_rows,
        "worker_provider_call_count": expected_calls,
        "aggregate_verifier_executed": True,
        "aggregate_verifier_observed_summary_count": (
            aggregate_verification.get("observed_summary_count")
        ),
        "aggregate_verifier_provider_call_count": (
            aggregate_verification.get("provider_call_count")
        ),
        "aggregate_verifier_expected_design_errors": sorted(
            expected_aggregate_errors
            & set(aggregate_verification.get("errors", []))
        ),
        "aggregate_verifier_temporal_sampling_fidelity_failures": (
            temporal_fidelity_failures
        ),
        "aggregate_verifier_observed_cell_temporal_failures": (
            observed_cell_temporal_failures
        ),
        "aggregate_verifier_unexpected_errors": unexpected_aggregate_errors,
        **broker,
        "errors": errors,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _validate_runtime_contract()
    exogenous_arrival_stream_audit = _exogenous_arrival_stream_audit()
    if config.VLLM_MAX_OUTPUT_TOKENS != 512:
        raise RuntimeError(
            "The preregistered categorical output ceiling must remain 512 tokens"
        )
    sources = _module_sources()
    grounded_memory_audit = _grounded_memory_policy_audit()
    backend = DeterministicSchemaBackend()
    provider = VLLMCategoricalCausalProvider(backend)
    packet_audit = (
        _static_packet_audit(Path(args.episode_log), backend._normalizer)
        if args.episode_log
        else None
    )
    temporal_budget_audit = (
        _temporal_budget_audit(Path(args.episode_log))
        if args.episode_log and args.decision_window_seconds
        else None
    )
    sampling_policy_audit = (
        _deterministic_sampling_audit(Path(args.episode_log))
        if args.episode_log
        else None
    )
    static_failures = [
        label
        for label, audit in (
            ("grounded memory policy", grounded_memory_audit),
            ("retained-evidence packet schema", packet_audit),
            ("temporal decision/export schema", temporal_budget_audit),
            ("deterministic sampling policy", sampling_policy_audit),
        )
        if audit is not None and audit.get("audit_pass") is not True
    ]
    if static_failures:
        return {
            "preflight_pass": False,
            "scientific_result": False,
            "gpu_used": False,
            "errors": [
                "Static pre-simulation audit failed: " + ", ".join(static_failures)
            ],
            "module_sources": sources,
            "grounded_memory_policy_audit": grounded_memory_audit,
            "static_packet_audit": packet_audit,
            "temporal_budget_audit": temporal_budget_audit,
            "sampling_policy_audit": sampling_policy_audit,
            "parallel_broker_audit": None,
            "pilot_manifest_audit": None,
            "study_manifest_audit": None,
        }

    def controller_factory(simulation):
        return Part3ClosedLoopController(
            provider,
            _persona_assignment(simulation, args.assignment_round),
            start_seconds=args.warmup_seconds,
            max_decisions_per_run=args.max_model_decisions,
            max_decisions_per_agent=args.max_model_decisions_per_agent,
            decision_window_seconds=args.decision_window_seconds,
            max_decisions_per_window=args.max_model_decisions_per_window,
            sampling_replication_id=args.assignment_round,
        )

    with tempfile.TemporaryDirectory(prefix="part3_closed_loop_preflight_") as temp:
        run_dir = Path(temp) / "run"
        run_args = argparse.Namespace(
            scenario_mode="normal_load",
            condition="both",
            seed=1,
            duration=args.duration,
            warmup_seconds=args.warmup_seconds,
            scenario_start_hour=10,
            validation_target="care_area",
            batch_name="part3_closed_loop_no_gpu_preflight",
            run_id="no_gpu_preflight",
            output_dir=str(run_dir),
            sensitivity_parameter=None,
            sensitivity_level=None,
            sensitivity_value=None,
            sensitivity_default=None,
            export_part3_episodes=False,
            part3_max_episodes=0,
        )
        run_single(run_args, part3_controller_factory=controller_factory)
        verification = verify(run_dir)
        decisions = [
            json.loads(line)
            for line in (run_dir / "part3_cognitive_decisions.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        inference = [
            json.loads(line)
            for line in (run_dir / "part3_provider_inference.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        provider_errors = [
            json.loads(line)
            for line in (run_dir / "part3_provider_errors.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        appraisal_packet_audit = _appraisal_packet_audit(
            run_dir,
            Path(temp) / "appraisal_packet_audit",
            backend._normalizer,
        )
        exported_files = sorted(path.name for path in run_dir.iterdir())

    errors = list(verification.get("errors", []))
    if not exogenous_arrival_stream_audit["audit_pass"]:
        errors.append("Part 3 exogenous arrival-stream audit failed")
    if not grounded_memory_audit["audit_pass"]:
        errors.append("Grounded causal-memory policy audit failed")
    if packet_audit is not None and not packet_audit["audit_pass"]:
        errors.append("Full retained-evidence packet audit failed")
    if sampling_policy_audit is not None and not sampling_policy_audit["audit_pass"]:
        errors.append("Deterministic proportional-sampling audit failed")
    if not appraisal_packet_audit["audit_pass"]:
        errors.append("Post-run appraisal packet audit failed")
    if args.decision_window_seconds:
        if backend.call_count < 2:
            errors.append(
                "Temporal integration run produced fewer than two real provider calls"
            )
        if temporal_budget_audit is None or not temporal_budget_audit["audit_pass"]:
            errors.append("Deterministic temporal budget audit failed")
    elif backend.call_count != args.max_model_decisions:
        errors.append(
            f"Expected {args.max_model_decisions} provider calls, got {backend.call_count}"
        )
    if len(inference) != backend.call_count:
        errors.append("Inference audit rows do not match backend calls")
    if not any(row.get("was_fallback") for row in decisions):
        errors.append("Call-cap fallback path was not exercised")
    action_counts = Counter(
        row["selected_action"] for row in decisions if not row["was_fallback"]
    )
    fallback_reason_counts = Counter(
        str(row.get("rejected_reason")) for row in decisions if row["was_fallback"]
    )
    if len(action_counts) < 2:
        errors.append("No-GPU provider did not exercise multiple action categories")
    if args.decision_window_seconds:
        observed_windows = verification.get("provider_calls_by_window", {})
        if not observed_windows:
            errors.append("Temporal call-budget windows were not exercised")
        if any(
            int(count) > args.max_model_decisions_per_window
            for count in observed_windows.values()
        ):
            errors.append("Temporal call-budget quota was exceeded")
        if not fallback_reason_counts.get("window_decision_cap"):
            errors.append("Temporal window-cap fallback path was not exercised")
    maximum_raw_characters = max(backend.raw_lengths, default=0)
    if maximum_raw_characters >= config.VLLM_MAX_OUTPUT_TOKENS * 4:
        errors.append(
            "A valid categorical response approaches the validated output ceiling"
        )

    manifest_audit = None
    parallel_broker_audit = None
    if args.manifest:
        manifest_rows = _read_manifest(Path(args.manifest))
        identities = {
            (
                row["scenario"],
                row["condition"],
                row["seed"],
                row["assignment_round"],
            )
            for row in manifest_rows
        }
        if args.study_design == "pilot":
            expected_conditions = {"baseline", "both"}
            expected_seeds = {1}
        else:
            if args.expected_seed_count is None or args.expected_seed_count < 2:
                raise ValueError(
                    "Main preflight requires --expected-seed-count >= 2"
                )
            expected_conditions = {
                "baseline",
                "cockpit_only",
                "nursta_only",
                "both",
            }
            expected_seeds = set(range(1, args.expected_seed_count + 1))
        expected_identities = {
            (scenario, condition, seed, assignment_round)
            for scenario in ("normal_load", "high_load_high_acuity")
            for condition in expected_conditions
            for seed in expected_seeds
            for assignment_round in range(1, 6)
        }
        manifest_errors = []
        if identities != expected_identities:
            manifest_errors.append(
                f"Manifest does not match the declared {args.study_design} "
                f"design ({len(expected_identities)} runs)"
            )
        sample_rates = {
            float(row["model_decision_sample_rate"])
            for row in manifest_rows
        }
        if sample_rates != {0.10}:
            manifest_errors.append(
                "Manifest must use the preregistered 10% sampling rate"
            )
        safety_contracts = {
            (
                int(row["max_model_decisions"]),
                int(row["max_model_decisions_per_agent"]),
                int(row["max_model_decisions_per_window"]),
            )
            for row in manifest_rows
        }
        if safety_contracts != {(300, 100, 100)}:
            manifest_errors.append(
                "Manifest safety ceilings do not match 300/100/100"
            )
        manifest_audit = {
            "audit_pass": not manifest_errors,
            "row_count": len(manifest_rows),
            "study_design": args.study_design,
            "expected_seed_count": len(expected_seeds),
            "expected_conditions": sorted(expected_conditions),
            "identities": [list(value) for value in sorted(identities)],
            "model_decision_sample_rates": sorted(sample_rates),
            "safety_contracts": [
                list(value) for value in sorted(safety_contracts)
            ],
            "errors": manifest_errors,
        }
        errors.extend(manifest_errors)
        if not manifest_errors:
            parallel_broker_audit = _parallel_broker_audit(
                manifest_rows, args
            )
            if not parallel_broker_audit["audit_pass"]:
                errors.append("Parallel shared-inference broker audit failed")

    return {
        "preflight_pass": not errors,
        "scientific_result": False,
        "gpu_used": False,
        "errors": errors,
        "module_sources": sources,
        "deployed_source_sha256": _deployed_source_hashes(
            [
                *args.guarded_path,
                *([args.manifest] if args.manifest else []),
            ]
        ),
        "validated_output_token_ceiling": config.VLLM_MAX_OUTPUT_TOKENS,
        "maximum_valid_response_characters": maximum_raw_characters,
        "provider_call_count": backend.call_count,
        "provider_errors": provider_errors,
        "exogenous_arrival_stream_audit": exogenous_arrival_stream_audit,
        "grounded_memory_policy_audit": grounded_memory_audit,
        "static_packet_audit": packet_audit,
        "temporal_budget_audit": temporal_budget_audit,
        "sampling_policy_audit": sampling_policy_audit,
        "appraisal_packet_audit": appraisal_packet_audit,
        "parallel_broker_audit": parallel_broker_audit,
        "pilot_manifest_audit": manifest_audit,
        "study_manifest_audit": manifest_audit,
        "study_design": args.study_design,
        "decision_count": len(decisions),
        "model_action_counts": dict(action_counts),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "fallback_count": sum(bool(row.get("was_fallback")) for row in decisions),
        "exported_files": exported_files,
        "closed_loop_verification": verification,
        "decision_window_seconds": args.decision_window_seconds,
        "max_model_decisions_per_window": args.max_model_decisions_per_window,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json")
    parser.add_argument("--duration", type=int, default=9000)
    parser.add_argument("--warmup-seconds", type=int, default=7200)
    parser.add_argument("--assignment-round", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--max-model-decisions", type=int, default=5)
    parser.add_argument("--max-model-decisions-per-agent", type=int, default=2)
    parser.add_argument("--decision-window-seconds", type=int, default=0)
    parser.add_argument("--max-model-decisions-per-window", type=int, default=0)
    parser.add_argument("--episode-log")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--study-design", choices=("pilot", "main"), default="pilot"
    )
    parser.add_argument("--expected-seed-count", type=int)
    parser.add_argument(
        "--guarded-path",
        action="append",
        default=[],
        help="Additional project-relative file included in deployment hashes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_preflight(args)
    except BaseException as error:
        result = {
            "preflight_pass": False,
            "scientific_result": False,
            "gpu_used": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "fatal_error_type": type(error).__name__,
            "fatal_error": str(error),
            "fatal_traceback": traceback.format_exc(),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        output = Path(args.out_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered)
    if not result["preflight_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
