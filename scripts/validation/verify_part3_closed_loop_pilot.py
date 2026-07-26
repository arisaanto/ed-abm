#!/usr/bin/env python3
"""Verify a matched Part 3 closed-loop pilot or main study.

This verifier separates technical integrity, design completeness, temporal
model-call coverage, and sampling coverage. Pilot mode remains a sizing gate;
main mode verifies the complete paired all-condition experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run.run_part3_closed_loop_batch import (
    ORCHESTRATION_MODE,
    TEMPORAL_COVERAGE_POLICY,
    _read_manifest,
    _run_dir,
)
from src.personas import cognitive_persona_by_id


ALLOWED_FALLBACK_REASONS = {
    "deterministic_sampling_skip",
}

SAFETY_CAP_REASONS = {
    "run_decision_cap",
    "agent_decision_cap",
    "window_decision_cap",
}


def _sampling_key(decision: dict[str, Any]) -> str:
    return "|".join(
        (
            str(decision.get("scenario")),
            str(decision.get("seed")),
            str(decision.get("sampling_replication_id")),
            str(decision.get("timestep")),
            str(decision.get("agent_id")),
            str(decision.get("partner_id")),
            str(decision.get("interaction_type")),
            str(decision.get("abm_reason_type")),
        )
    )


def _sampling_value(sampling_key: str) -> float:
    integer = int.from_bytes(
        hashlib.sha256(sampling_key.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return integer / float(1 << 64)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _declared_design(
    study_design: str, expected_seed_count: int | None
) -> tuple[set[str], set[int], str, bool]:
    if study_design == "pilot":
        if expected_seed_count not in (None, 1):
            raise ValueError("Pilot design requires exactly one seed")
        return {"baseline", "both"}, {1}, "one_seed_closed_loop_sizing_pilot", False
    if study_design == "main":
        if expected_seed_count is None or expected_seed_count < 2:
            raise ValueError("Main design requires --expected-seed-count >= 2")
        return (
            {"baseline", "cockpit_only", "nursta_only", "both"},
            set(range(1, expected_seed_count + 1)),
            f"paired_closed_loop_main_n{expected_seed_count}",
            True,
        )
    raise ValueError(f"Unknown study design: {study_design}")


def verify(
    output_root: Path,
    manifest_path: Path,
    *,
    study_design: str = "pilot",
    expected_seed_count: int | None = None,
    enforce_grounded_memory_coverage: bool = True,
) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    expected_conditions, expected_seeds, scope, scientific_result = (
        _declared_design(study_design, expected_seed_count)
    )
    errors: list[str] = []
    warnings: list[str] = []
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    archived_manifest_path = output_root / "manifest.csv"
    if not archived_manifest_path.is_file():
        errors.append("Missing archived pilot manifest")
    elif archived_manifest_path.read_bytes() != manifest_bytes:
        errors.append("Archived pilot manifest differs from the verified manifest")
    observed_identities = {
        (
            row["scenario"],
            row["condition"],
            row["seed"],
            row["assignment_round"],
        )
        for row in rows
    }
    declared_identities = {
        (scenario, condition, seed, assignment_round)
        for scenario in ("normal_load", "high_load_high_acuity")
        for condition in expected_conditions
        for seed in expected_seeds
        for assignment_round in range(1, 6)
    }
    declared_design_matches = observed_identities == declared_identities
    if not declared_design_matches:
        errors.append(
            f"Manifest does not match the declared {study_design} design "
            f"({len(declared_identities)} runs)"
        )
    sample_rates = {float(row["model_decision_sample_rate"]) for row in rows}
    if len(sample_rates) != 1:
        errors.append("Pilot manifest uses inconsistent model sampling rates")
    expected_run_dirs = {_run_dir(output_root, row) for row in rows}
    observed_summaries = set(output_root.rglob("summary.json"))
    observed_run_dirs = {path.parent for path in observed_summaries}
    if observed_run_dirs != expected_run_dirs:
        missing = sorted(str(path) for path in expected_run_dirs - observed_run_dirs)
        unexpected = sorted(str(path) for path in observed_run_dirs - expected_run_dirs)
        errors.append(f"Run-folder mismatch; missing={missing}, unexpected={unexpected}")

    batch_execution_path = output_root / "batch_execution.json"
    engine_metadata_path = output_root / "engine_metadata.json"
    if not batch_execution_path.is_file():
        errors.append("Missing batch_execution.json")
    if not engine_metadata_path.is_file():
        errors.append("Missing engine_metadata.json")
    batch_execution = _json(batch_execution_path) if batch_execution_path.is_file() else {}
    engine_metadata = _json(engine_metadata_path) if engine_metadata_path.is_file() else {}
    if engine_metadata.get("manifest_sha256") != manifest_sha256:
        errors.append("Engine metadata manifest hash does not match the verified manifest")
    if int(batch_execution.get("expected_run_count", -1)) != len(rows):
        errors.append("Batch expected-run count does not match the pilot manifest")
    if int(batch_execution.get("completed_run_count", -1)) != len(rows):
        errors.append("Batch did not record completion of every manifest run")
    batch_engine_loads = int(batch_execution.get("engine_load_count", -1))
    metadata_engine_loads = int(engine_metadata.get("engine_load_count", -1))
    if batch_engine_loads < 1 or metadata_engine_loads < 1:
        errors.append("The batch did not record a valid model-engine load")
    if batch_engine_loads != metadata_engine_loads:
        errors.append("Batch and engine metadata disagree on engine-load count")
    if metadata_engine_loads > 1:
        segments = engine_metadata.get("execution_segments", [])
        if (
            batch_execution.get("resumed_execution") is not True
            or engine_metadata.get("resumed_execution") is not True
            or len(segments) != metadata_engine_loads
        ):
            errors.append(
                "Resumed execution did not preserve complete engine-load provenance"
            )
    if engine_metadata.get("prefix_caching_enabled") is not True:
        errors.append("Automatic prefix caching was not enabled")
    if engine_metadata.get("safetensors_load_strategy") != "eager":
        warnings.append("Study did not use the planned eager Lustre loading strategy")
    for source_name, source in (
        ("batch execution", batch_execution),
        ("engine metadata", engine_metadata),
    ):
        if source.get("orchestration_mode") != ORCHESTRATION_MODE:
            errors.append(
                f"{source_name.title()} did not record parallel broker orchestration"
            )
    resumed_execution = batch_execution.get("resumed_execution") is True
    workers_started = int(batch_execution.get("workers_started", -1))
    execution_workers_started = int(
        batch_execution.get("execution_workers_started", -1)
    )
    recovered_complete_runs = int(
        batch_execution.get("recovered_complete_run_count", 0)
    )
    newly_completed_runs = int(
        batch_execution.get("newly_completed_run_count", -1)
    )
    unique_completed_runs = int(
        batch_execution.get("unique_completed_run_count", -1)
    )
    if resumed_execution:
        if (
            recovered_complete_runs <= 0
            or newly_completed_runs <= 0
            or recovered_complete_runs + newly_completed_runs != len(rows)
            or execution_workers_started != newly_completed_runs
            or unique_completed_runs != len(rows)
            or workers_started < len(rows)
        ):
            errors.append(
                "Resumed worker provenance does not reconcile to the complete design"
            )
    elif (
        workers_started != len(rows)
        or execution_workers_started != len(rows)
        or recovered_complete_runs != 0
        or newly_completed_runs != len(rows)
        or unique_completed_runs != len(rows)
    ):
        errors.append("The broker did not start exactly one worker per study run")
    if int(batch_execution.get("maximum_active_workers", 0)) < 2:
        errors.append("Independent simulations did not overlap in execution")

    persona_ids = set(cognitive_persona_by_id())
    assignments: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    topic_counts_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    engaged_topic_counts: Counter[str] = Counter()
    engaged_topic_counts_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    persona_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    persona_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    persona_realized_topic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    persona_cell_action_counts: dict[
        str, dict[str, Counter[str]]
    ] = defaultdict(lambda: defaultdict(Counter))
    persona_provider_calls: Counter[str] = Counter()
    persona_role_provider_calls: dict[str, Counter[str]] = defaultdict(Counter)
    fallback_reasons: Counter[str] = Counter()
    sampling_by_cell: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    run_results: list[dict[str, Any]] = []
    temporal_sampling_fidelity_failures = []
    low_per_run_temporal_coverage = []
    temporal_counts_by_cell_window: dict[
        tuple[str, str, int], Counter[str]
    ] = defaultdict(Counter)
    total_provider_calls = 0
    total_decisions = 0
    total_model_interactions = 0
    total_model_missed = 0
    total_eligible_decisions = 0
    total_sampled_candidates = 0
    total_memory_context_records = 0
    total_agent_experience_rows = 0
    total_experience_events = 0
    total_spatial_experience_maps = 0
    memory_context_records_by_cell: Counter[tuple[str, str]] = Counter()
    observed_memory_policies: set[str] = set()
    arrival_stream_signatures: dict[
        tuple[str, int], list[dict[str, Any]]
    ] = defaultdict(list)

    for row in rows:
        run_dir = _run_dir(output_root, row)
        verification_path = run_dir / "closed_loop_verification.json"
        decisions_path = run_dir / "part3_cognitive_decisions.jsonl"
        if not verification_path.is_file() or not decisions_path.is_file():
            errors.append(f"Missing run verification/decisions: {run_dir}")
            continue
        verification = _json(verification_path)
        summary = _json(run_dir / "summary.json")
        decisions = _jsonl(decisions_path)
        if verification.get("verification_pass") is not True:
            errors.append(
                f"Run verification failed for {run_dir}: {verification.get('errors')}"
            )
        run_experience_rows = int(
            verification.get("agent_experience_row_count", 0)
        )
        total_agent_experience_rows += run_experience_rows
        if (
            verification.get("agent_experience_complete") is not True
            or run_experience_rows != 9
        ):
            errors.append(
                f"Agent-experience evidence is incomplete in {run_dir}"
            )
        run_experience_events = int(
            verification.get("experience_event_count", 0)
        )
        run_spatial_maps = int(
            verification.get("spatial_experience_map_count", 0)
        )
        total_experience_events += run_experience_events
        total_spatial_experience_maps += run_spatial_maps
        if (
            verification.get("experience_event_export_pass") is not True
            or run_experience_events <= 0
        ):
            errors.append(f"Episodic experience evidence is incomplete in {run_dir}")
        if (
            verification.get("spatial_experience_map_export_pass") is not True
            or run_spatial_maps != 9
        ):
            errors.append(f"Spatial-experience maps are incomplete in {run_dir}")
        observed_memory_policies.add(str(verification.get("memory_policy")))
        run_memory_context_records = int(
            verification.get("memory_context_record_count", 0)
        )
        total_memory_context_records += run_memory_context_records
        memory_context_records_by_cell[(row["scenario"], row["condition"])] += (
            run_memory_context_records
        )
        if verification.get("decision_history_used_as_memory") is not False:
            errors.append(f"Decision history was used as memory in {run_dir}")
        if (
            verification.get("generated_reflection_used_as_causal_input")
            is not False
        ):
            errors.append(f"Generated reflection entered causal input in {run_dir}")
        metadata = summary.get("metadata", {})
        expected_metadata = {
            "scenario_mode": row["scenario"],
            "condition": row["condition"],
            "seed": row["seed"],
            "duration_seconds": row["duration_seconds"],
            "warmup_seconds": row["warmup_seconds"],
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            errors.append(f"Run metadata mismatch for {run_dir}: {mismatches}")
        if metadata.get("part3_exogenous_arrival_stream_isolated") is not True:
            errors.append(
                f"Part 3 exogenous arrival stream was not isolated in {run_dir}"
            )
        if metadata.get("part3_exogenous_arrival_stream_version") != (
            "scenario_seed_common_random_numbers_v1"
        ):
            errors.append(
                f"Unexpected Part 3 arrival-stream version in {run_dir}"
            )
        patient_flow = summary.get("patient_flow_metrics", {})
        arrival_stream_signatures[(row["scenario"], int(row["seed"]))].append(
            {
                "condition": row["condition"],
                "assignment_round": int(row["assignment_round"]),
                "arrival_attempts": int(
                    patient_flow.get("arrival_attempts", -1)
                ),
                "arrival_attempts_by_esi": tuple(
                    sorted(
                        (
                            str(key),
                            int(value),
                        )
                        for key, value in patient_flow.get(
                            "arrival_attempts_by_esi", {}
                        ).items()
                    )
                ),
            }
        )

        closed_loop = summary.get("part3_closed_loop", {})
        expected_controller = {
            "max_decisions_per_run": row["max_model_decisions"],
            "max_decisions_per_agent": row["max_model_decisions_per_agent"],
            "decision_window_seconds": row["decision_window_seconds"],
            "max_decisions_per_window": row["max_model_decisions_per_window"],
            "model_decision_sample_rate": row[
                "model_decision_sample_rate"
            ],
            "sampling_replication_id": row["assignment_round"],
        }
        controller_mismatch = {
            key: (closed_loop.get(key), expected)
            for key, expected in expected_controller.items()
            if closed_loop.get(key) != expected
        }
        if controller_mismatch:
            errors.append(
                f"Controller configuration mismatch for {run_dir}: {controller_mismatch}"
            )

        persona_by_agent = closed_loop.get("persona_by_agent", {})
        if set(persona_by_agent.values()) - persona_ids:
            errors.append(f"Unknown persona assignment in {run_dir}")
        for agent_id, persona_id in persona_by_agent.items():
            assignments[
                (row["scenario"], row["condition"], row["seed"], int(agent_id))
            ].append(str(persona_id))

        calls_by_window = {
            int(index): int(count)
            for index, count in closed_loop.get("provider_calls_by_window", {}).items()
        }
        eligible_by_window = {
            int(index): int(count)
            for index, count in closed_loop.get(
                "eligible_decisions_by_window", {}
            ).items()
        }
        sampled_by_window = {
            int(index): int(count)
            for index, count in closed_loop.get(
                "sampled_candidates_by_window", {}
            ).items()
        }
        expected_window_count = (
            row["duration_seconds"] - row["warmup_seconds"]
        ) // row["decision_window_seconds"]
        invalid_window = {
            index: count
            for index, count in {
                **eligible_by_window,
                **sampled_by_window,
                **calls_by_window,
            }.items()
            if index not in range(expected_window_count)
            or calls_by_window.get(index, 0)
            > row["max_model_decisions_per_window"]
        }
        if invalid_window:
            errors.append(f"Invalid temporal quota use in {run_dir}: {invalid_window}")
        sampled_call_mismatches = {
            index: {
                "sampled_candidates": sampled_by_window.get(index, 0),
                "provider_calls": calls_by_window.get(index, 0),
            }
            for index in range(expected_window_count)
            if sampled_by_window.get(index, 0)
            != calls_by_window.get(index, 0)
        }
        if sampled_call_mismatches:
            temporal_sampling_fidelity_failures.append(
                {
                    "run_dir": str(run_dir.relative_to(output_root)),
                    "window_mismatches": sampled_call_mismatches,
                }
            )
            errors.append(
                f"Sampled-call temporal fidelity failed in {run_dir}: "
                f"{sampled_call_mismatches}"
            )
        covered_windows = sum(
            calls_by_window.get(index, 0) > 0
            for index in range(expected_window_count)
        )
        if covered_windows < expected_window_count - 1:
            low_per_run_temporal_coverage.append(
                {
                    "run_dir": str(run_dir.relative_to(output_root)),
                    "covered_windows": covered_windows,
                    "expected_windows": expected_window_count,
                }
            )
        for index in range(expected_window_count):
            counts = temporal_counts_by_cell_window[
                (str(row["scenario"]), str(row["condition"]), index)
            ]
            counts["eligible"] += eligible_by_window.get(index, 0)
            counts["sampled"] += sampled_by_window.get(index, 0)
            counts["provider_calls"] += calls_by_window.get(index, 0)

        sampling_errors = []
        for decision in decisions:
            expected_key = _sampling_key(decision)
            expected_value = _sampling_value(expected_key)
            observed_value = decision.get("sampling_value")
            expected_sampled = (
                expected_value < row["model_decision_sample_rate"]
            )
            if decision.get("sampling_method") != (
                "deterministic_sha256_matched_opportunity_threshold_v1"
            ):
                sampling_errors.append("sampling_method")
            if decision.get("sampling_key") != expected_key:
                sampling_errors.append("sampling_key")
            if observed_value is None or not math.isclose(
                float(observed_value), expected_value, abs_tol=1e-15
            ):
                sampling_errors.append("sampling_value")
            if decision.get("sampled_for_model") is not expected_sampled:
                sampling_errors.append("sampled_for_model")
            if not math.isclose(
                float(decision.get("model_decision_sample_rate", -1.0)),
                float(row["model_decision_sample_rate"]),
                abs_tol=1e-15,
            ):
                sampling_errors.append("sample_rate")
            if (
                not expected_sampled
                and (
                    decision.get("was_fallback") is not True
                    or decision.get("rejected_reason")
                    != "deterministic_sampling_skip"
                )
            ):
                sampling_errors.append("unsampled_action")
        if sampling_errors:
            errors.append(
                f"Deterministic sampling audit failed in {run_dir}: "
                f"{dict(Counter(sampling_errors))}"
            )

        invalid_fallbacks = [
            row_value.get("rejected_reason")
            for row_value in decisions
            if row_value.get("was_fallback")
            and row_value.get("rejected_reason") not in ALLOWED_FALLBACK_REASONS
        ]
        if invalid_fallbacks:
            errors.append(f"Unexpected fallback in {run_dir}: {invalid_fallbacks}")
        safety_cap_hits = [
            row_value.get("rejected_reason")
            for row_value in decisions
            if row_value.get("rejected_reason") in SAFETY_CAP_REASONS
        ]
        if safety_cap_hits:
            errors.append(f"Safety ceiling reached in {run_dir}: {safety_cap_hits}")
        model_decisions = [row_value for row_value in decisions if not row_value.get("was_fallback")]
        sampled_candidates = [
            row_value
            for row_value in decisions
            if row_value.get("sampled_for_model") is True
        ]
        eligible_count = len(decisions)
        sampled_count = len(sampled_candidates)
        if len(model_decisions) != sampled_count:
            errors.append(
                f"Sampled candidates did not all receive model decisions in {run_dir}"
            )
        if int(closed_loop.get("eligible_decision_count", -1)) != eligible_count:
            errors.append(f"Eligible-decision count mismatch in {run_dir}")
        if int(closed_loop.get("sampled_candidate_count", -1)) != sampled_count:
            errors.append(f"Sampled-candidate count mismatch in {run_dir}")
        observed_rate = sampled_count / eligible_count if eligible_count else 0.0
        if not math.isclose(
            float(closed_loop.get("observed_sample_rate", -1.0)),
            observed_rate,
            abs_tol=1e-15,
        ):
            errors.append(f"Observed sample-rate mismatch in {run_dir}")
        for decision in model_decisions:
            action_counts[str(decision.get("selected_action"))] += 1
            reason_counts[str(decision.get("selected_reason"))] += 1
            topic_counts[str(decision.get("topic_family"))] += 1
            topic_counts_by_reason[str(decision.get("abm_reason_type"))][
                str(decision.get("topic_family"))
            ] += 1
            if decision.get("selected_action") == "engage":
                engaged_topic_counts[str(decision.get("topic_family"))] += 1
                engaged_topic_counts_by_reason[
                    str(decision.get("abm_reason_type"))
                ][str(decision.get("topic_family"))] += 1
            if decision.get("topic_family") not in decision.get(
                "allowed_topic_families", []
            ):
                errors.append(f"Unbounded topic in {run_dir}")
            persona_id = str(decision.get("persona_id"))
            role = str(decision.get("role"))
            persona_provider_calls[persona_id] += 1
            persona_role_provider_calls[persona_id][role] += 1
            persona_action_counts[str(decision.get("persona_id"))][
                str(decision.get("selected_action"))
            ] += 1
            persona_reason_counts[persona_id][
                str(decision.get("selected_reason"))
            ] += 1
            cell_label = f"{row['scenario']}|{row['condition']}"
            persona_cell_action_counts[persona_id][cell_label][
                str(decision.get("selected_action"))
            ] += 1
            if decision.get("selected_action") == "engage":
                persona_realized_topic_counts[persona_id][
                    str(decision.get("topic_family"))
                ] += 1
        fallback_reasons.update(
            str(decision.get("rejected_reason"))
            for decision in decisions
            if decision.get("was_fallback")
        )
        provider_calls = int(verification.get("provider_call_count", 0))
        if provider_calls != len(model_decisions):
            errors.append(f"Provider-call count mismatch in {run_dir}")
        total_provider_calls += provider_calls
        total_decisions += len(decisions)
        total_eligible_decisions += eligible_count
        total_sampled_candidates += sampled_count
        cell_sampling = sampling_by_cell[(row["scenario"], row["condition"])]
        cell_sampling["eligible"] += eligible_count
        cell_sampling["sampled"] += sampled_count
        total_model_interactions += int(
            verification.get("model_applied_interaction_count", 0)
        )
        total_model_missed += int(
            verification.get("model_missed_opportunity_count", 0)
        )
        run_results.append(
            {
                "scenario": row["scenario"],
                "condition": row["condition"],
                "seed": row["seed"],
                "assignment_round": row["assignment_round"],
                "provider_call_count": provider_calls,
                "eligible_decision_count": eligible_count,
                "sampled_candidate_count": sampled_count,
                "observed_sample_rate": observed_rate,
                "covered_decision_windows": covered_windows,
                "workflow_status": verification.get("workflow_status"),
                "verification_pass": verification.get("verification_pass"),
            }
        )

    assignment_failures = [
        {
            "scenario": key[0],
            "condition": key[1],
            "seed": key[2],
            "agent_id": key[3],
            "personas": values,
        }
        for key, values in sorted(assignments.items())
        if len(values) != 5 or set(values) != persona_ids
    ]
    if assignment_failures:
        errors.append("One or more staff did not receive every persona exactly once")
    exogenous_arrival_pairing_failures = []
    for (scenario, seed), signatures in sorted(
        arrival_stream_signatures.items()
    ):
        unique_signatures = {
            (
                row["arrival_attempts"],
                row["arrival_attempts_by_esi"],
            )
            for row in signatures
        }
        if len(unique_signatures) != 1:
            exogenous_arrival_pairing_failures.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "observed_signatures": signatures,
                }
            )
    exogenous_arrival_pairing_pass = not exogenous_arrival_pairing_failures
    if not exogenous_arrival_pairing_pass:
        errors.append(
            "Matched Part 3 runs did not share an identical exogenous "
            "arrival-attempt stream"
        )
    if low_per_run_temporal_coverage:
        warnings.append(
            "Some valid proportional samples contain model calls in fewer than "
            "four of five full-shift windows; inspect the reported observations"
        )
    temporal_aggregate_coverage = []
    temporal_aggregate_coverage_failures = []
    expected_window_count = (
        int(rows[0]["duration_seconds"])
        - int(rows[0]["warmup_seconds"])
    ) // int(rows[0]["decision_window_seconds"])
    for scenario in ("normal_load", "high_load_high_acuity"):
        for condition in sorted(expected_conditions):
            for window_index in range(expected_window_count):
                counts = temporal_counts_by_cell_window[
                    (scenario, condition, window_index)
                ]
                row_value = {
                    "scenario": scenario,
                    "condition": condition,
                    "window_index": window_index,
                    "eligible_decision_count": int(counts["eligible"]),
                    "sampled_candidate_count": int(counts["sampled"]),
                    "provider_call_count": int(counts["provider_calls"]),
                    "coverage_pass": (
                        counts["eligible"] > 0
                        and counts["sampled"] > 0
                        and counts["provider_calls"] == counts["sampled"]
                    ),
                }
                temporal_aggregate_coverage.append(row_value)
                if not row_value["coverage_pass"]:
                    temporal_aggregate_coverage_failures.append(row_value)
    if temporal_aggregate_coverage_failures:
        errors.append(
            "One or more scenario-condition study cells lack aggregate model-call "
            "coverage in a full-shift window"
        )
    target_sample_rate = float(rows[0]["model_decision_sample_rate"])
    sampling_coverage = []
    sampling_coverage_pass = True
    for (scenario, condition), counts in sorted(sampling_by_cell.items()):
        eligible = int(counts["eligible"])
        sampled = int(counts["sampled"])
        observed = sampled / eligible if eligible else 0.0
        tolerance = max(
            0.03,
            4.0
            * math.sqrt(
                target_sample_rate * (1.0 - target_sample_rate)
                / max(eligible, 1)
            ),
        )
        passed = eligible > 0 and abs(observed - target_sample_rate) <= tolerance
        sampling_coverage_pass = sampling_coverage_pass and passed
        sampling_coverage.append(
            {
                "scenario": scenario,
                "condition": condition,
                "eligible_decision_count": eligible,
                "sampled_candidate_count": sampled,
                "observed_sample_rate": observed,
                "target_sample_rate": target_sample_rate,
                "audit_tolerance": tolerance,
                "coverage_pass": passed,
            }
        )
    if not sampling_coverage_pass:
        errors.append(
            "One or more scenario-condition cells failed proportional-sampling coverage"
        )

    persona_call_values = {
        persona_id: int(persona_provider_calls.get(persona_id, 0))
        for persona_id in sorted(persona_ids)
    }
    maximum_persona_calls = max(persona_call_values.values(), default=0)
    minimum_persona_calls = min(persona_call_values.values(), default=0)
    persona_exposure_ratio = (
        minimum_persona_calls / maximum_persona_calls
        if maximum_persona_calls
        else 0.0
    )
    observed_roles = {
        role
        for counts in persona_role_provider_calls.values()
        for role in counts
        if role not in ("None", "")
    }
    persona_role_coverage_failures = {
        persona_id: sorted(
            observed_roles - set(persona_role_provider_calls.get(persona_id, {}))
        )
        for persona_id in sorted(persona_ids)
        if observed_roles
        - set(persona_role_provider_calls.get(persona_id, {}))
    }
    persona_exposure_balance_pass = (
        minimum_persona_calls > 0
        and persona_exposure_ratio >= 0.70
        and not persona_role_coverage_failures
    )
    if not persona_exposure_balance_pass:
        warnings.append(
            "Balanced assignment completed, but sampled persona exposure remains uneven"
        )

    realized_topic_count = sum(engaged_topic_counts.values())
    maximum_topic_share = (
        max(engaged_topic_counts.values(), default=0) / realized_topic_count
        if realized_topic_count
        else 0.0
    )
    if maximum_topic_share > 0.80:
        warnings.append(
            "One canonical topic exceeds 80% of realized model engagements; "
            "inspect evidence composition"
        )
    broker_packet_count = int(
        batch_execution.get("inference_packet_count", -1)
    )
    if broker_packet_count != total_provider_calls:
        errors.append(
            "Broker packet count does not equal verified provider-call count"
        )
    maximum_inference_batch_size = int(
        batch_execution.get("maximum_inference_batch_size", 0)
    )
    if maximum_inference_batch_size < 2:
        warnings.append(
            "Parallel workers completed without observable inference micro-batching"
        )
    if observed_memory_policies != {"grounded_realized_interactions_v1"}:
        errors.append(
            "Runs did not use one grounded realized-interaction memory policy"
        )
    grounded_memory_coverage_failures = sorted(
        f"{scenario}|{condition}"
        for scenario in ("normal_load", "high_load_high_acuity")
        for condition in expected_conditions
        if memory_context_records_by_cell[(scenario, condition)] <= 0
    )
    grounded_memory_coverage_pass = not grounded_memory_coverage_failures
    if enforce_grounded_memory_coverage and not grounded_memory_coverage_pass:
        errors.append(
            "Grounded memory was not exercised in every scenario-condition cell: "
            f"{grounded_memory_coverage_failures}"
        )

    technical_integrity_pass = not errors
    temporal_coverage_pass = (
        not temporal_sampling_fidelity_failures
        and not temporal_aggregate_coverage_failures
    )
    design_complete = (
        len(rows) == len(declared_identities)
        and len(observed_summaries) == len(declared_identities)
        and declared_design_matches
        and not assignment_failures
    )
    result = {
        "verification_pass": (
            technical_integrity_pass and design_complete and temporal_coverage_pass
        ),
        "technical_integrity_pass": technical_integrity_pass,
        "pilot_design_complete": design_complete if study_design == "pilot" else None,
        "main_design_complete": design_complete if study_design == "main" else None,
        "study_design_complete": design_complete,
        "temporal_coverage_pass": temporal_coverage_pass,
        "scientific_readiness_for_main": (
            technical_integrity_pass
            and design_complete
            and temporal_coverage_pass
            and sampling_coverage_pass
            and persona_exposure_balance_pass
            and grounded_memory_coverage_pass
        ),
        "scientific_result": scientific_result and technical_integrity_pass and design_complete,
        "study_design": study_design,
        "temporal_coverage_policy": TEMPORAL_COVERAGE_POLICY,
        "scope": scope,
        "expected_seed_count": len(expected_seeds),
        "expected_conditions": sorted(expected_conditions),
        "aggregate_verifier_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "errors": errors,
        "warnings": warnings,
        "expected_run_count": len(rows),
        "observed_summary_count": len(observed_summaries),
        "engine_load_count": batch_execution.get("engine_load_count"),
        "engine_load_seconds": batch_execution.get("engine_load_seconds"),
        "total_elapsed_seconds": batch_execution.get("total_elapsed_seconds"),
        "orchestration_mode": batch_execution.get("orchestration_mode"),
        "configured_max_workers": batch_execution.get("configured_max_workers"),
        "maximum_active_workers": batch_execution.get("maximum_active_workers"),
        "resumed_execution": resumed_execution,
        "recovered_complete_run_count": recovered_complete_runs,
        "newly_completed_run_count": newly_completed_runs,
        "unique_completed_run_count": unique_completed_runs,
        "workers_started": workers_started,
        "execution_workers_started": execution_workers_started,
        "worker_attempt_count": batch_execution.get("worker_attempt_count"),
        "inference_batch_count": batch_execution.get("inference_batch_count"),
        "inference_packet_count": broker_packet_count,
        "memory_policy": (
            next(iter(observed_memory_policies))
            if len(observed_memory_policies) == 1
            else None
        ),
        "memory_context_record_count": total_memory_context_records,
        "agent_experience_schema_version": 1,
        "agent_experience_row_count": total_agent_experience_rows,
        "expected_agent_experience_row_count": len(rows) * 9,
        "agent_experience_export_pass": (
            total_agent_experience_rows == len(rows) * 9
        ),
        "experience_event_schema_version": 1,
        "experience_event_count": total_experience_events,
        "experience_event_export_pass": total_experience_events > 0,
        "spatial_experience_map_schema_version": 1,
        "spatial_experience_map_count": total_spatial_experience_maps,
        "expected_spatial_experience_map_count": len(rows) * 9,
        "spatial_experience_map_export_pass": (
            total_spatial_experience_maps == len(rows) * 9
        ),
        "grounded_memory_coverage_pass": grounded_memory_coverage_pass,
        "grounded_memory_coverage_enforced": enforce_grounded_memory_coverage,
        "grounded_memory_coverage_failures": grounded_memory_coverage_failures,
        "exogenous_arrival_pairing_pass": exogenous_arrival_pairing_pass,
        "exogenous_arrival_stream_version": (
            "scenario_seed_common_random_numbers_v1"
        ),
        "exogenous_arrival_pairing_failures": (
            exogenous_arrival_pairing_failures
        ),
        "memory_context_record_count_by_scenario_condition": {
            f"{scenario}|{condition}": int(count)
            for (scenario, condition), count in sorted(
                memory_context_records_by_cell.items()
            )
        },
        "decision_history_used_as_memory": False,
        "generated_reflection_used_as_causal_input": False,
        "maximum_inference_batch_size": maximum_inference_batch_size,
        "inference_batch_size_counts": batch_execution.get(
            "inference_batch_size_counts", {}
        ),
        "provider_call_count": total_provider_calls,
        "decision_count": total_decisions,
        "eligible_decision_count": total_eligible_decisions,
        "sampled_candidate_count": total_sampled_candidates,
        "target_sample_rate": target_sample_rate,
        "observed_sample_rate": (
            total_sampled_candidates / total_eligible_decisions
            if total_eligible_decisions
            else 0.0
        ),
        "sampling_coverage_pass": sampling_coverage_pass,
        "sampling_coverage_by_scenario_condition": sampling_coverage,
        "model_applied_interaction_count": total_model_interactions,
        "model_missed_opportunity_count": total_model_missed,
        "model_action_counts": dict(sorted(action_counts.items())),
        "model_reason_counts": dict(sorted(reason_counts.items())),
        "model_decision_topic_counts": dict(sorted(topic_counts.items())),
        "model_decision_topic_counts_by_abm_reason": {
            reason: dict(sorted(counts.items()))
            for reason, counts in sorted(topic_counts_by_reason.items())
        },
        "realized_interaction_topic_counts": dict(
            sorted(engaged_topic_counts.items())
        ),
        "realized_interaction_topic_counts_by_abm_reason": {
            reason: dict(sorted(counts.items()))
            for reason, counts in sorted(
                engaged_topic_counts_by_reason.items()
            )
        },
        "maximum_realized_interaction_topic_share": maximum_topic_share,
        "persona_provider_call_counts": persona_call_values,
        "persona_role_provider_call_counts": {
            persona: dict(sorted(counts.items()))
            for persona, counts in sorted(persona_role_provider_calls.items())
        },
        "persona_exposure_ratio": persona_exposure_ratio,
        "persona_role_coverage_failures": persona_role_coverage_failures,
        "persona_exposure_balance_pass": persona_exposure_balance_pass,
        "persona_action_counts": {
            persona: dict(sorted(counts.items()))
            for persona, counts in sorted(persona_action_counts.items())
        },
        "persona_reason_counts": {
            persona: dict(sorted(counts.items()))
            for persona, counts in sorted(persona_reason_counts.items())
        },
        "persona_realized_interaction_topic_counts": {
            persona: dict(sorted(counts.items()))
            for persona, counts in sorted(
                persona_realized_topic_counts.items()
            )
        },
        "persona_action_counts_by_scenario_condition": {
            persona: {
                cell: dict(sorted(counts.items()))
                for cell, counts in sorted(cells.items())
            }
            for persona, cells in sorted(persona_cell_action_counts.items())
        },
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        "assignment_failure_count": len(assignment_failures),
        "assignment_failures": assignment_failures,
        "temporal_sampling_fidelity_failures": (
            temporal_sampling_fidelity_failures
        ),
        "temporal_aggregate_coverage": temporal_aggregate_coverage,
        "temporal_aggregate_coverage_failures": (
            temporal_aggregate_coverage_failures
        ),
        "low_per_run_temporal_coverage_observations": (
            low_per_run_temporal_coverage
        ),
        # Backward-compatible alias; failures now mean aggregate/fidelity
        # failures rather than random zero-sample windows within one run.
        "temporal_coverage_failures": (
            temporal_sampling_fidelity_failures
            + temporal_aggregate_coverage_failures
        ),
        "runs": run_results,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--study-design", choices=("pilot", "main"), default="pilot"
    )
    parser.add_argument("--expected-seed-count", type=int)
    args = parser.parse_args()
    result = verify(
        Path(args.output_root),
        Path(args.manifest),
        study_design=args.study_design,
        expected_seed_count=args.expected_seed_count,
    )
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
