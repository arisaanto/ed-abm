#!/usr/bin/env python3
"""Strictly verify Part 3 offline-vLLM responses against their source packets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.vllm_backend import packet_json_schema, scientific_packet_prompt_leakage
from src.interviews import INTERVIEW_QUESTIONS
from src.personas import PERSONA_APPRAISAL_DIMENSIONS, default_cognitive_personas
from scripts.plan_part3_synthetic_study import (
    SCIENTIFIC_EPISODE_COUNT,
    SCIENTIFIC_PACKET_COUNT,
)


LEGACY_DECISION_KEYS = {
    "decision_id",
    "selected_action",
    "selected_reason",
    "topic_family",
    "topic_text",
    "rationale_short",
    "evidence_ids",
    "confidence",
}

CATEGORICAL_CAUSAL_DECISION_KEYS = {
    "decision_id",
    "selected_action",
    "selected_reason",
    "topic_family",
    "evidence_ids",
}

PUBLIC_ANSWER_FORBIDDEN_PATTERNS = {
    "analysis_language": re.compile(
        r"\b(?:log|logs|logged|metric|metrics|dataset|data point|evidence id|"
        r"agent-based model|abm|prompt|persona)\b",
        re.IGNORECASE,
    ),
    "synthetic_process_language": re.compile(
        r"\b(?:simulated agent|language model|llm|model output)\b",
        re.IGNORECASE,
    ),
    "opaque_id_leakage": re.compile(r"\bshift_event_\d+\b", re.IGNORECASE),
}


def _decision_keys(packet: dict[str, Any]) -> set[str]:
    if packet.get("decision_output_contract") == "categorical_causal_v1":
        return CATEGORICAL_CAUSAL_DECISION_KEYS
    return LEGACY_DECISION_KEYS


def _packet_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    paths = sorted(path.glob("*_prompt_packets.jsonl")) if path.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No prompt packet JSONL files found under {path}")
    return paths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def _read_packets(path: Path, limit: int | None) -> list[dict[str, Any]]:
    packets = []
    for packet_path in _packet_paths(path):
        for packet in _read_jsonl(packet_path):
            packets.append(packet)
            if limit is not None and len(packets) >= limit:
                return packets
    return packets


def _schema_digest(packet: dict[str, Any]) -> str:
    schema = packet_json_schema(str(packet.get("packet_type", "")), packet)
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _decision_errors(
    packet: dict[str, Any], result: dict[str, Any], raw_payload: dict[str, Any]
) -> list[str]:
    prompt_id = str(packet.get("prompt_id", ""))
    errors = []
    expected_keys = _decision_keys(packet)
    if set(raw_payload) != expected_keys:
        errors.append(
            f"{prompt_id}: raw decision keys differ: "
            f"missing={sorted(expected_keys - set(raw_payload))}, "
            f"extra={sorted(set(raw_payload) - expected_keys)}"
        )
    if str(raw_payload.get("decision_id", "")) != prompt_id:
        errors.append(f"{prompt_id}: decision_id does not match prompt_id")
    for response_key, packet_key in (
        ("selected_action", "feasible_actions"),
        ("selected_reason", "allowed_reasons"),
        ("topic_family", "allowed_topic_families"),
    ):
        if raw_payload.get(response_key) not in packet.get(packet_key, []):
            errors.append(
                f"{prompt_id}: {response_key}={raw_payload.get(response_key)!r} "
                f"not in {packet_key}"
            )
    categorical_causal = (
        packet.get("decision_output_contract") == "categorical_causal_v1"
    )
    if not categorical_causal:
        for key in ("topic_text", "rationale_short"):
            if not isinstance(raw_payload.get(key), str) or not raw_payload[key].strip():
                errors.append(f"{prompt_id}: {key} is empty or not a string")
    evidence = raw_payload.get("evidence_ids")
    allowed_evidence = {str(value) for value in packet.get("evidence_ids", [])}
    if not isinstance(evidence, list) or not evidence:
        errors.append(f"{prompt_id}: evidence_ids must be a non-empty list")
    elif any(str(value) not in allowed_evidence for value in evidence):
        errors.append(f"{prompt_id}: response cites evidence outside the packet")
    elif len(evidence) != len({str(value) for value in evidence}):
        errors.append(f"{prompt_id}: response cites duplicate evidence IDs")
    if not categorical_causal:
        confidence = raw_payload.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{prompt_id}: confidence is not numeric within 0..1")
    if result.get("response") != raw_payload:
        errors.append(f"{prompt_id}: normalized decision differs from schema-valid raw response")
    return errors


def _appraisal_errors(
    packet: dict[str, Any], result: dict[str, Any], raw_payload: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    prompt_id = str(packet.get("prompt_id", ""))
    packet_type = str(packet.get("packet_type", ""))
    errors = []
    review_flags = []
    allowed_evidence = {str(value) for value in packet.get("evidence_ids", [])}
    if raw_payload.get("not_human_data") is not True:
        errors.append(f"{prompt_id}: appraisal lacks not_human_data=true")

    if packet_type == "end_of_shift_survey_bundle":
        rows = raw_payload.get("responses")
        if not isinstance(rows, list):
            return [f"{prompt_id}: survey responses must be a list"], review_flags
        dimensions = [str(row.get("dimension", "")) for row in rows]
        if len(dimensions) != len(set(dimensions)):
            errors.append(f"{prompt_id}: duplicate survey dimensions")
        if set(dimensions) != set(PERSONA_APPRAISAL_DIMENSIONS):
            errors.append(f"{prompt_id}: survey dimension set is incomplete")
        for row in rows:
            dimension = str(row.get("dimension", ""))
            rateability = str(row.get("rateability", ""))
            score = row.get("score_1_to_7")
            if rateability == "rateable" and not (
                isinstance(score, int) and 1 <= score <= 7
            ):
                errors.append(f"{prompt_id}: invalid rateable score for {dimension}")
            if rateability == "insufficient_evidence" and score is not None:
                errors.append(
                    f"{prompt_id}: insufficient-evidence score is not null for {dimension}"
                )
            confidence = row.get("confidence_1_to_5")
            if not isinstance(confidence, int) or not 1 <= confidence <= 5:
                errors.append(f"{prompt_id}: invalid confidence for {dimension}")
            if not str(row.get("short_rationale", "")).strip():
                errors.append(f"{prompt_id}: {dimension} lacks a short rationale")
            evidence = row.get("evidence_event_ids")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{prompt_id}: {dimension} cites no evidence")
            elif set(map(str, evidence)) - allowed_evidence:
                errors.append(f"{prompt_id}: {dimension} cites unknown evidence")

    elif packet_type == "end_of_shift_interview_bundle":
        rows = raw_payload.get("answers")
        if not isinstance(rows, list):
            return [f"{prompt_id}: interview answers must be a list"], review_flags
        expected = {row["question_id"] for row in INTERVIEW_QUESTIONS}
        question_ids = [str(row.get("question_id", "")) for row in rows]
        if len(question_ids) != len(set(question_ids)):
            errors.append(f"{prompt_id}: duplicate interview questions")
        if set(question_ids) != expected:
            errors.append(f"{prompt_id}: interview question set is incomplete")
        if raw_payload.get("claim_layer_contract") != (
            "grounded_pattern_interpretation_conjecture_v1"
        ):
            errors.append(f"{prompt_id}: interview claim-layer contract differs")
        for row in rows:
            question_id = str(row.get("question_id", ""))
            answer = str(row.get("answer", "")).strip()
            grounded = str(row.get("grounded_pattern", "")).strip()
            interpretation = str(
                row.get("persona_conditioned_interpretation", "")
            ).strip()
            evidence = row.get("evidence_event_ids")
            if not answer or not grounded or not interpretation:
                errors.append(
                    f"{prompt_id}: {question_id} lacks answer, pattern, or interpretation"
                )
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{prompt_id}: {question_id} cites no evidence")
            elif set(map(str, evidence)) - allowed_evidence:
                errors.append(f"{prompt_id}: {question_id} cites unknown evidence")
            design_hypothesis = row.get("design_hypothesis")
            tradeoff = row.get("tradeoff")
            if design_hypothesis and not tradeoff:
                errors.append(
                    f"{prompt_id}: {question_id} proposes a design without a tradeoff"
                )
            hits = [
                category
                for category, pattern in PUBLIC_ANSWER_FORBIDDEN_PATTERNS.items()
                if pattern.search(answer)
            ]
            if hits:
                review_flags.append(
                    {
                        "prompt_id": prompt_id,
                        "question_id": question_id,
                        "categories": hits,
                        "reason": "public answer exposes analysis/process language",
                    }
                )
    normalized = result.get("response")
    if not isinstance(normalized, dict) or normalized.get("not_human_data") is not True:
        errors.append(f"{prompt_id}: normalized appraisal lacks not-human marker")
    return errors, review_flags


def verify(
    packet_path: Path,
    response_path: Path,
    *,
    limit: int | None,
    reject_fixtures: bool,
) -> dict[str, Any]:
    packets = _read_packets(packet_path, limit)
    responses = _read_jsonl(response_path)
    errors = []
    packet_ids = [str(packet.get("prompt_id", "")) for packet in packets]
    response_ids = [str(row.get("prompt_id", "")) for row in responses]
    if not packets:
        errors.append("No input packets selected")
    if any(not prompt_id for prompt_id in packet_ids):
        errors.append("One or more input packets lacks prompt_id")
    if len(packet_ids) != len(set(packet_ids)):
        errors.append("Input prompt IDs are not unique")
    if len(response_ids) != len(set(response_ids)):
        errors.append("Response prompt IDs are not unique")
    leakage_packet_count = 0
    for packet in packets:
        prompt_id = str(packet.get("prompt_id", ""))
        leakage = scientific_packet_prompt_leakage(packet)
        if leakage:
            leakage_packet_count += 1
            errors.append(
                f"{prompt_id}: evaluator-only prompt terms exposed: {leakage}"
            )
        if (
            packet.get("fixture_only_not_scientific_data") is False
            and packet.get("packet_type") == "in_simulation_decision"
            and packet.get("comparator_hidden_from_model") is not True
        ):
            leakage_packet_count += 1
            errors.append(f"{prompt_id}: comparator blinding marker is absent")
    missing = sorted(set(packet_ids) - set(response_ids))
    unexpected = sorted(set(response_ids) - set(packet_ids))
    if missing:
        errors.append(f"Missing response IDs: {missing}")
    if unexpected:
        errors.append(f"Unexpected response IDs: {unexpected}")

    packets_by_id = {str(packet.get("prompt_id", "")): packet for packet in packets}
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    persona_counts: Counter[str] = Counter()
    persona_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    persona_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    persona_topic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    persona_confidences: dict[str, list[float]] = defaultdict(list)
    decision_contract_counts: Counter[str] = Counter()
    persona_reference_matches: Counter[str] = Counter()
    persona_reference_totals: Counter[str] = Counter()
    evidence_group_decisions: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    manual_grounding_review_flags: list[dict[str, Any]] = []
    appraisal_manual_review_flags: list[dict[str, Any]] = []
    appraisal_packet_counts: Counter[str] = Counter()
    reference_matches = 0
    comparable_reference_actions = 0

    for result in responses:
        prompt_id = str(result.get("prompt_id", ""))
        packet = packets_by_id.get(prompt_id)
        if packet is None:
            continue
        if reject_fixtures and packet.get("fixture_only_not_scientific_data") is not False:
            errors.append(f"{prompt_id}: fixture packet rejected")
        if result.get("packet_type") != packet.get("packet_type"):
            errors.append(f"{prompt_id}: packet_type differs from source packet")
        if result.get("synthetic_design_probe_not_human_data") is not True:
            errors.append(f"{prompt_id}: synthetic-not-human marker is absent")
        expected_digest = _schema_digest(packet)
        if result.get("response_schema_sha256") != expected_digest:
            errors.append(f"{prompt_id}: response schema digest does not match packet")
        raw_response = result.get("raw_response")
        try:
            raw_payload = json.loads(raw_response) if isinstance(raw_response, str) else None
        except json.JSONDecodeError:
            raw_payload = None
        if not isinstance(raw_payload, dict):
            errors.append(f"{prompt_id}: raw_response is not a JSON object")
            continue
        if packet.get("packet_type") == "in_simulation_decision":
            decision_contract_counts[
                str(packet.get("decision_output_contract") or "legacy_narrative")
            ] += 1
            errors.extend(_decision_errors(packet, result, raw_payload))
            action = str(raw_payload.get("selected_action", ""))
            reason = str(raw_payload.get("selected_reason", ""))
            topic = str(raw_payload.get("topic_family", ""))
            action_counts[action] += 1
            reason_counts[reason] += 1
            topic_counts[topic] += 1
            persona_id = str(
                packet.get("user_model_profile", {}).get("persona_id", "unknown")
            )
            persona_counts[persona_id] += 1
            persona_action_counts[persona_id][action] += 1
            persona_reason_counts[persona_id][reason] += 1
            persona_topic_counts[persona_id][topic] += 1
            confidence = raw_payload.get("confidence")
            if isinstance(confidence, (int, float)):
                persona_confidences[persona_id].append(float(confidence))
            evidence_id = str(packet.get("evidence_ids", [prompt_id])[0])
            evidence_group_decisions[evidence_id][persona_id] = {
                "action": action,
                "reason": reason,
                "topic": topic,
            }
            reference_action = str(
                packet.get("trace_evidence", {})
                .get("rule_reference", {})
                .get("selected_action", "")
            )
            if reference_action:
                comparable_reference_actions += 1
                reference_matches += int(action == reference_action)
                persona_reference_totals[persona_id] += 1
                persona_reference_matches[persona_id] += int(
                    action == reference_action
                )

            categorical_causal = (
                packet.get("decision_output_contract") == "categorical_causal_v1"
            )
            contact = packet.get("llm_visible_evidence", {}).get(
                "contact_opportunity", {}
            )
            response_text = " ".join(
                (
                    str(raw_payload.get("topic_text", "")),
                    str(raw_payload.get("rationale_short", "")),
                )
            ).lower()
            flag_categories = []
            flag_reasons = []

            def add_flag(category: str, reason_text: str) -> None:
                if category not in flag_categories:
                    flag_categories.append(category)
                    flag_reasons.append(reason_text)

            if not categorical_causal and contact.get("patient_context_present") is False:
                patient_specific_topics = {
                    "patient_status_update",
                    "treatment_and_orders",
                    "diagnostic_findings",
                    "patient_facing_care",
                }
                if topic in patient_specific_topics:
                    add_flag(
                        "patient_topic_without_context",
                        "patient-specific topic selected without a supplied patient context"
                    )
                if any(
                    phrase in response_text
                    for phrase in (
                        "this patient",
                        "the patient",
                        "active patient",
                        "active cases",
                        "current patient status",
                        "patient's",
                        "waiting patient",
                    )
                ):
                    add_flag(
                        "patient_claim_without_context",
                        "free text may imply an unsupplied patient or clinical state",
                    )

            patient_context = contact.get("patient_context") or {}
            waiting_supported = bool(
                patient_context.get("waiting_for_placement")
            )
            discharge_supported = str(
                patient_context.get("current_care_stage", "")
            ).lower() == "discharge or admission"
            waiting_claim = any(
                phrase in response_text
                for phrase in (
                    "waiting patient",
                    "patient waiting for a doctor",
                    "patient waiting for the doctor",
                    "patient waiting for a physician",
                    "waiting for doctor assignment",
                    "waiting for physician assignment",
                )
            )
            discharge_claim = any(
                phrase in response_text
                for phrase in (
                    "discharge readiness",
                    "returning home patient",
                    "patient discharge",
                )
            )
            if (waiting_claim and not waiting_supported) or (
                discharge_claim and not discharge_supported
            ):
                add_flag(
                    "unsupported_patient_disposition",
                    "free text may invent a waiting or disposition state not supplied",
                )

            if contact.get("active_task_present") is False:
                task_text = response_text
                for pattern in (
                    r"\bno active task(?: is)? present\b",
                    r"\bno active task\b",
                    r"\bwithout an? active task\b",
                    r"\black of an? active task\b",
                    r"\bwithout interrupting active tasks\b",
                ):
                    task_text = re.sub(pattern, "", task_text)
                if any(
                    phrase in task_text
                    for phrase in (
                        "current task",
                        "task continuity",
                        "protect the task",
                        "interrupting the task",
                        "active task was",
                    )
                ):
                    add_flag(
                        "active_task_claim_without_context",
                        "free text may imply an active task when none was supplied",
                    )

            if contact.get("mutual_visibility") is False and any(
                phrase in response_text
                for phrase in (
                    "partner is visible",
                    "visible partner",
                    "visible colleague",
                    "mutually visible",
                    "idle and visible",
                )
            ):
                add_flag(
                    "visibility_claim_without_evidence",
                    "free text describes visibility when mutual_visibility is false",
                )

            unsupported_operational_phrases = (
                "incoming patients",
                "transfer queue",
                "patient is stable",
                "stable patient",
                "doctor assignment",
                "partner availability",
                "partner unavailability",
            )
            waiting_mode_supplied = any(
                contact.get(key) == "staff_waiting_for_doctor_coordination"
                for key in ("initiator_staff_mode", "partner_staff_mode")
            )
            if "waiting doctor" in response_text and not waiting_mode_supplied:
                unsupported_operational_phrases += ("waiting doctor",)
            if contact.get("active_task_present") is False:
                unsupported_operational_phrases += (
                    "until task completion",
                    "after task completion",
                )
            unsupported_hits = [
                phrase
                for phrase in unsupported_operational_phrases
                if phrase in response_text
            ]
            if unsupported_hits:
                add_flag(
                    "unsupported_operational_detail",
                    "free text may infer unsupplied operational state: "
                    + ", ".join(unsupported_hits),
                )

            band_patterns = {
                "urgency_band": r"\b(low|medium|moderate|high)\s+urgency\b",
                "diversion_cost_band": (
                    r"\b(low|medium|moderate|high)[ -]diversion(?:\s+cost)?\b"
                ),
                "ed_pressure_band": (
                    r"\b(low|medium|moderate|high)\s+(?:ed\s+)?pressure\b"
                ),
            }
            for band_key, pattern in band_patterns.items():
                expected_band = str(contact.get(band_key, ""))
                for match in re.finditer(pattern, response_text):
                    prefix = response_text[max(0, match.start() - 80):match.start()]
                    if re.search(
                        r"(?:\blacks?|\bnot|\bwithout|\bno)\b[^.!?]{0,70}$",
                        prefix,
                    ):
                        continue
                    claimed_band = match.group(1)
                    if claimed_band == "moderate":
                        claimed_band = "medium"
                    if expected_band and claimed_band != expected_band:
                        add_flag(
                            "numeric_band_contradiction",
                            f"free text calls {band_key} {claimed_band} but supplied "
                            f"band is {expected_band}",
                        )

            rationale = str(raw_payload.get("rationale_short", "")).strip()
            if rationale and not re.search(r"[.!?]$", rationale):
                add_flag(
                    "incomplete_rationale",
                    "rationale_short does not end as a complete sentence",
                )

            persona_echo_terms = (
                "team connector",
                "focus protector",
                "patient advocate",
                "vigilant monitor",
                "adaptive generalist",
                "orientation",
                "workplace prior",
            )
            numeric_prior_echo = re.search(
                r"\b(?:coordination orientation|focus protection|patient advocacy|"
                r"vigilance|adaptability|interaction initiative|privacy control "
                r"preference)\s*\(?\s*\d",
                response_text,
            )
            if any(term in response_text for term in persona_echo_terms) or numeric_prior_echo:
                add_flag(
                    "persona_prompt_echo",
                    "free text repeats a persona label or numeric workplace prior",
                )

            initiator_role = str(contact.get("initiator_role", "")).lower()
            for claimed_role in ("nurse", "doctor", "coordination nurse"):
                if (
                    f"{claimed_role} initiator" in response_text
                    and claimed_role.replace(" ", "")
                    != initiator_role.replace(" ", "")
                ):
                    add_flag(
                        "initiator_role_inversion",
                        "free text assigns the wrong role to the initiator",
                    )
                    break

            if flag_reasons:
                manual_grounding_review_flags.append(
                    {
                        "prompt_id": prompt_id,
                        "persona_id": persona_id,
                        "categories": flag_categories,
                        "reasons": flag_reasons,
                    }
                )
        elif packet.get("packet_type") in {
            "end_of_shift_survey_bundle",
            "end_of_shift_interview_bundle",
        }:
            packet_type = str(packet["packet_type"])
            appraisal_packet_counts[packet_type] += 1
            appraisal_errors, appraisal_flags = _appraisal_errors(
                packet, result, raw_payload
            )
            errors.extend(appraisal_errors)
            appraisal_manual_review_flags.extend(appraisal_flags)

    def nested_counts(
        counters: dict[str, Counter[str]],
    ) -> dict[str, dict[str, int]]:
        return {
            persona: dict(sorted(counts.items()))
            for persona, counts in sorted(counters.items())
        }

    evidence_group_count = len(evidence_group_decisions)
    action_divergent_groups = sum(
        len({row["action"] for row in decisions.values()}) > 1
        for decisions in evidence_group_decisions.values()
    )
    reason_divergent_groups = sum(
        len({row["reason"] for row in decisions.values()}) > 1
        for decisions in evidence_group_decisions.values()
    )
    topic_divergent_groups = sum(
        len({row["topic"] for row in decisions.values()}) > 1
        for decisions in evidence_group_decisions.values()
    )

    persona_ids = sorted(persona_counts)
    expected_persona_ids = {
        persona.persona_id for persona in default_cognitive_personas()
    }
    scientific_design_errors = []
    scientific_design_checked = bool(
        reject_fixtures
        and packets
        and all(
            packet.get("fixture_only_not_scientific_data") is False
            and packet.get("packet_type") == "in_simulation_decision"
            for packet in packets
        )
    )
    if scientific_design_checked:
        if len(packets) != SCIENTIFIC_PACKET_COUNT:
            scientific_design_errors.append(
                f"expected {SCIENTIFIC_PACKET_COUNT} matched scientific packets, "
                f"found {len(packets)}"
            )
        if evidence_group_count != SCIENTIFIC_EPISODE_COUNT:
            scientific_design_errors.append(
                f"expected {SCIENTIFIC_EPISODE_COUNT} matched evidence groups, "
                f"found {evidence_group_count}"
            )
        if set(persona_ids) != expected_persona_ids:
            scientific_design_errors.append(
                "persona set differs from the five preregistered cognitive orientations"
            )
        for evidence_id, decisions in sorted(evidence_group_decisions.items()):
            if set(decisions) != expected_persona_ids:
                scientific_design_errors.append(
                    f"{evidence_id}: incomplete five-persona crossing"
                )
        for persona_id in sorted(expected_persona_ids):
            if persona_counts.get(persona_id, 0) != SCIENTIFIC_EPISODE_COUNT:
                scientific_design_errors.append(
                    f"{persona_id}: expected {SCIENTIFIC_EPISODE_COUNT} responses, found "
                    f"{persona_counts.get(persona_id, 0)}"
                )
        errors.extend(scientific_design_errors)
    pairwise_agreement = {}
    indistinguishable_pairs = []
    for left_index, left_persona in enumerate(persona_ids):
        for right_persona in persona_ids[left_index + 1 :]:
            shared = [
                decisions
                for decisions in evidence_group_decisions.values()
                if left_persona in decisions and right_persona in decisions
            ]
            if not shared:
                continue
            action_matches = sum(
                decisions[left_persona]["action"]
                == decisions[right_persona]["action"]
                for decisions in shared
            )
            reason_matches = sum(
                decisions[left_persona]["reason"]
                == decisions[right_persona]["reason"]
                for decisions in shared
            )
            topic_matches = sum(
                decisions[left_persona]["topic"]
                == decisions[right_persona]["topic"]
                for decisions in shared
            )
            triple_matches = sum(
                decisions[left_persona] == decisions[right_persona]
                for decisions in shared
            )
            pair_key = f"{left_persona}|{right_persona}"
            pairwise_agreement[pair_key] = {
                "matched_evidence_count": len(shared),
                "action_agreement": action_matches / len(shared),
                "reason_agreement": reason_matches / len(shared),
                "topic_agreement": topic_matches / len(shared),
                "categorical_triple_agreement": triple_matches / len(shared),
            }
            if len(shared) >= 8 and triple_matches == len(shared):
                indistinguishable_pairs.append(pair_key)

    persona_action_signal_present = bool(
        action_divergent_groups > 0 and len(action_counts) > 1
    )
    appraisal_human_review_required = bool(appraisal_packet_counts)
    manual_review_required = bool(
        manual_grounding_review_flags
        or appraisal_manual_review_flags
        or indistinguishable_pairs
        or (scientific_design_checked and not persona_action_signal_present)
        or appraisal_human_review_required
    )
    automated_scientific_readiness = bool(
        not errors
        and not manual_grounding_review_flags
        and not appraisal_manual_review_flags
        and not indistinguishable_pairs
        and (not scientific_design_checked or persona_action_signal_present)
        and not appraisal_human_review_required
    )
    return {
        "verification_pass": not errors,
        "technical_verification_pass": not errors,
        "scientific_manual_review_required": manual_review_required,
        "automated_scientific_readiness_pass": automated_scientific_readiness,
        "scientific_design_checked": scientific_design_checked,
        "scientific_design_errors": scientific_design_errors,
        "persona_action_signal_present": persona_action_signal_present,
        "closed_loop_progression_note": (
            "Do not advance to closed-loop use until all manual grounding flags "
            "are reviewed and resolved."
            if manual_review_required
            else "No automated manual-review flags remain; retain human spot-checking."
        ),
        "packet_count": len(packets),
        "response_count": len(responses),
        "missing_response_count": len(missing),
        "unexpected_response_count": len(unexpected),
        "schema_or_grounding_error_count": len(errors),
        "evaluator_leakage_packet_count": leakage_packet_count,
        "evaluator_blinding_pass": leakage_packet_count == 0,
        "errors": errors,
        "decision_action_counts": dict(sorted(action_counts.items())),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "decision_topic_counts": dict(sorted(topic_counts.items())),
        "decision_output_contract_counts": dict(
            sorted(decision_contract_counts.items())
        ),
        "free_text_used_as_causal_input": False,
        "persona_response_counts": dict(sorted(persona_counts.items())),
        "persona_action_counts": nested_counts(persona_action_counts),
        "persona_reason_counts": nested_counts(persona_reason_counts),
        "persona_topic_counts": nested_counts(persona_topic_counts),
        "persona_mean_confidence": {
            persona: sum(values) / len(values)
            for persona, values in sorted(persona_confidences.items())
            if values
        },
        "within_evidence_divergence": {
            "evidence_group_count": evidence_group_count,
            "groups_with_action_divergence": action_divergent_groups,
            "groups_with_reason_divergence": reason_divergent_groups,
            "groups_with_topic_divergence": topic_divergent_groups,
            "action_divergence_rate": (
                action_divergent_groups / evidence_group_count
                if evidence_group_count
                else None
            ),
            "reason_divergence_rate": (
                reason_divergent_groups / evidence_group_count
                if evidence_group_count
                else None
            ),
            "topic_divergence_rate": (
                topic_divergent_groups / evidence_group_count
                if evidence_group_count
                else None
            ),
        },
        "persona_pairwise_categorical_agreement": pairwise_agreement,
        "categorically_indistinguishable_persona_pairs": indistinguishable_pairs,
        "manual_grounding_review_flag_count": len(
            manual_grounding_review_flags
        ),
        "manual_grounding_review_category_counts": dict(
            sorted(
                Counter(
                    category
                    for flag in manual_grounding_review_flags
                    for category in flag.get("categories", [])
                ).items()
            )
        ),
        "manual_grounding_review_flags": manual_grounding_review_flags,
        "appraisal_packet_counts": dict(sorted(appraisal_packet_counts.items())),
        "appraisal_human_review_required": appraisal_human_review_required,
        "appraisal_human_review_note": (
            "Automated checks cannot establish naturalness or the credibility of "
            "persona-conditioned interpretation and design conjecture. Review the "
            "prespecified balanced sample before scientific reporting."
            if appraisal_human_review_required
            else None
        ),
        "appraisal_manual_review_flag_count": len(appraisal_manual_review_flags),
        "appraisal_manual_review_flags": appraisal_manual_review_flags,
        "rule_reference_action_agreement": (
            reference_matches / comparable_reference_actions
            if comparable_reference_actions
            else None
        ),
        "rule_reference_action_agreement_note": (
            "Descriptive only; agreement is not a validity target for the cognitive policy."
        ),
        "rule_reference_action_agreement_by_persona": {
            persona: (
                persona_reference_matches[persona]
                / persona_reference_totals[persona]
            )
            for persona in sorted(persona_reference_totals)
            if persona_reference_totals[persona]
        },
        "fixtures_rejected": reject_fixtures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prompt JSONL file or packet directory")
    parser.add_argument("--responses", required=True, help="Generated response JSONL")
    parser.add_argument("--out-json", help="Optional verification-summary JSON")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reject-fixtures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify(
        Path(args.input).expanduser().resolve(),
        Path(args.responses).expanduser().resolve(),
        limit=args.limit,
        reject_fixtures=args.reject_fixtures,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered)
    if not summary["verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
