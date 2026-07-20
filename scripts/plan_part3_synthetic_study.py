#!/usr/bin/env python3
"""Write the frozen Part 3 protocol and offline prompt packets.

Without an episode log, this writes explicit fixtures for testing the vLLM code
path. With ``--episode-log``, it converts logging-only ABM decision episodes
into paired scientific packets without running a simulation or an LLM.
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
from typing import Any, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.cognitive_policy import estimate_context_tokens
from src.interviews import (
    CANONICAL_TOPIC_FAMILIES,
    INTERVIEW_QUESTIONS,
    TOPIC_FAMILIES_BY_ABM_REASON,
    categorical_decision_bounds,
    required_output_schema,
)
from src.personas import (
    PERSONA_APPRAISAL_DIMENSIONS,
    balanced_cognitive_persona_assignments,
    default_cognitive_personas,
)


SYSTEM_MESSAGE = (
    "You are a constrained synthetic cognitive policy for an Emergency Department agent-based model. "
    "Use only supplied evidence. Never invent patient facts, movement, outcomes, emotions, or testimony. "
    "Choose only allowed actions, reasons, and canonical topic families. Return only schema-valid JSON."
)

DECISION_OUTPUT_CONTRACT = "categorical_causal_v1"

ACTION_SEMANTICS = {
    "engage": (
        "Initiate this optional interaction now because its expected operational "
        "value exceeds its attention and diversion cost."
    ),
    "defer": (
        "Do not interact now; the topic may remain relevant but this is not the "
        "right moment."
    ),
    "decline": (
        "Do not interact because the supplied evidence does not establish enough "
        "relevance or value."
    ),
}

STAFF_MODE_LABELS = {
    "returning_home": "returning_to_staff_station",
    "waiting_for_doctor": "staff_waiting_for_doctor_coordination",
    "post_task_station_check": "staff_post_task_station_check",
    "moving_to_post_task_station_check": "staff_moving_to_post_task_station_check",
    "moving_to_secondary_station": "staff_moving_to_secondary_station",
}

SCIENTIFIC_EPISODE_COUNT = 12
SCIENTIFIC_PACKET_COUNT = SCIENTIFIC_EPISODE_COUNT * 5

REQUIRED_EVIDENCE_FEATURES = {
    "role:CoordinationNurse",
    "role:Nurse",
    "role:Doctor",
    "reason:BED_FLOW_COORDINATION",
    "reason:HANDOFF_NEED",
    "reason:POST_TASK_UPDATE",
    "reason:SAME_STATION_BRIEF_CHECKIN",
    "type:station",
    "type:corridor",
    "visibility:True",
    "visibility:False",
    "patient:True",
    "patient:False",
    "acuity:high",
    "acuity:lower",
    "urgency:low",
    "urgency:high",
    "diversion:low",
    "diversion:high",
    "pressure:low",
    "pressure:high",
    *{
        f"cell:{scenario}|{condition}"
        for scenario in ("normal_load", "high_load_high_acuity")
        for condition in ("baseline", "cockpit_only", "nursta_only", "both")
    },
}

ALLOWED_INTERACTION_REASONS = [
    "same_patient_update",
    "urgent_escalation",
    "coordination_need",
    "task_unblocking",
    "patient_explanation",
    "relational_check_in",
    "protect_task_continuity",
    "insufficient_relevance",
]

ABLATION_VARIANTS = [
    {
        "variant": "rule_based_reference",
        "llm": False,
        "persona": False,
        "longitudinal_memory": False,
        "reflection": False,
    },
    {
        "variant": "generic_llm",
        "llm": True,
        "persona": False,
        "longitudinal_memory": False,
        "reflection": False,
    },
    {
        "variant": "persona_llm_no_memory",
        "llm": True,
        "persona": True,
        "longitudinal_memory": False,
        "reflection": False,
    },
    {
        "variant": "persona_memory_reflection",
        "llm": True,
        "persona": True,
        "longitudinal_memory": True,
        "reflection": True,
    },
]

EVALUATION_FAMILIES = [
    "integrity and schema validity",
    "persona fidelity and separability",
    "behavior-belief alignment",
    "evidence grounding and hallucination rate",
    "temporal coherence",
    "spatial sensitivity and persona-by-environment differentiation",
    "run-to-run stability",
    "system-level interaction consequences",
    "compute cost per valid grounded response",
]

HARD_CONSTRAINTS = [
    "The ABM owns geometry, movement, workflow, proximity, and feasible partners.",
    "Choose only from feasible_actions, allowed_reasons, and allowed_topic_families.",
    "Cite only evidence_ids present in this packet.",
    "Do not invent patient outcomes, diagnoses, events, feelings, or human testimony.",
    (
        "When patient_context_present is false, do not imply a particular patient, "
        "case, transfer, diagnosis, or clinical status."
    ),
    (
        "When active_task_present is false, do not claim that an active task was "
        "interrupted, protected, accelerated, or completed."
    ),
    (
        "active_task_present=false identifies an ABM-approved interruptible boundary; "
        "it is not evidence that the opportunity lacks operational relevance."
    ),
    (
        "patient_context_present means only that a patient is linked to the "
        "opportunity; do not infer waiting, discharge, disposition, diagnosis, "
        "treatment, or outcome unless explicitly supplied."
    ),
    (
        "Initiator and partner modes describe staff workflow states only; they "
        "never describe a patient's disposition or movement."
    ),
    (
        "A partner waiting for doctor coordination is a staff member waiting to "
        "coordinate with a doctor; it does not mean that a doctor is waiting."
    ),
    (
        "Do not infer patient stability, staff availability, incoming patients, "
        "queue status, transfer status, or assignment state unless supplied explicitly."
    ),
    (
        "mutual_visibility is the only supplied visibility fact. Never describe "
        "the partner as visible when mutual_visibility is false."
    ),
    (
        "Use the supplied study-specific urgency, diversion, and pressure bands "
        "when applying qualitative labels; do not relabel them."
    ),
    (
        "Apply the cognitive orientation silently. Do not mention its title, "
        "persona name, workplace-prior labels, or numeric prior values in the response."
    ),
    "Interpret numeric evidence only according to score_semantics.",
    "Keep free-text topics subordinate to the selected canonical topic family.",
]

SCORE_SEMANTICS = {
    "urgency": {
        "range": [0.0, 1.0],
        "meaning": (
            "Higher values indicate stronger operational urgency. The supplied "
            "urgency_band is a study-specific descriptive bin."
        ),
    },
    "salience": {
        "range": "relative; no universal upper bound",
        "meaning": (
            "Higher values rank this visible opportunity as more salient within "
            "the ABM; this is not a probability."
        ),
    },
    "diversion_cost": {
        "range": [0.0, 1.0],
        "meaning": (
            "Higher values indicate greater normalized spatial diversion. The "
            "supplied diversion_cost_band is a study-specific descriptive bin."
        ),
    },
    "ed_pressure_index": {
        "range": [0.0, 2.0],
        "meaning": (
            "Higher values indicate greater aggregate ED operational pressure. The "
            "supplied ed_pressure_band is a study-specific descriptive bin."
        ),
    },
    "esi_level": {
        "range": [1, 5],
        "meaning": (
            "Lower values indicate higher acuity; null means no particular patient "
            "context is supplied."
        ),
    },
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _fixture_evidence(index: int) -> dict[str, Any]:
    evidence_id = f"fixture:event:{index:02d}"
    return {
        "evidence_id": evidence_id,
        "fixture_only_not_scientific_data": True,
        "metadata": {
            "run_id": f"schema_fixture_{index:02d}",
            "scenario_mode": "normal_load" if index % 2 else "high_load_high_acuity",
            "condition": ["baseline", "cockpit_only", "nursta_only", "both"][(index - 1) % 4],
            "seed": 0,
            "staff_id": [1, 10, 11, 12, 13, 14, 20, 21, 22][(index - 1) % 9],
            "role": ["CoordinationNurse", "Nurse", "Doctor"][(index - 1) % 3],
        },
        "event_summary": (
            "Synthetic schema fixture: a feasible staff contact opportunity was observed. "
            "No patient outcome or unobserved clinical fact is supplied."
        ),
    }


def _user_message(packet_type: str, persona: dict[str, Any], evidence: dict[str, Any]) -> str:
    metadata = evidence["metadata"]
    return "\n".join(
        [
            f"Packet type: {packet_type}",
            *_orientation_prompt_lines(persona),
            f"ABM role: {metadata['role']}",
            f"Scenario: {metadata['scenario_mode']}",
            f"Condition: {metadata['condition']}",
            f"Evidence: {evidence['evidence_id']} - {evidence['event_summary']}",
            "This is a schema-validation fixture, not a scientific observation.",
        ]
    )


def _orientation_prompt_lines(persona: dict[str, Any]) -> list[str]:
    """Expose operational policy guidance without persona labels or scores."""

    return [
        "Decision policy priorities: " + " ".join(persona["decision_principles"]),
        "Topic priorities: " + ", ".join(persona["topic_family_preferences"]),
    ]


def _fixture_packet(packet_type: str, index: int) -> dict[str, Any]:
    personas = default_cognitive_personas()
    persona = personas[(index - 1) % len(personas)].prompt_profile()
    evidence = _fixture_evidence(index)
    packet: dict[str, Any] = {
        "prompt_id": f"schema_fixture_{packet_type}_{index:02d}",
        "packet_type": packet_type,
        "system_message": SYSTEM_MESSAGE,
        "user_message": _user_message(packet_type, persona, evidence),
        "required_output_schema": required_output_schema(packet_type),
        "role": evidence["metadata"]["role"],
        "user_model_profile": persona,
        "trace_evidence": evidence,
        "evidence_ids": [evidence["evidence_id"]],
        "hard_constraints": HARD_CONSTRAINTS,
        "fixture_only_not_scientific_data": True,
        "synthetic_design_probe_not_human_data": True,
    }
    if packet_type == "in_simulation_decision":
        packet.update(
            {
                "feasible_actions": ["engage", "defer", "decline"],
                "allowed_reasons": ALLOWED_INTERACTION_REASONS,
                "allowed_topic_families": CANONICAL_TOPIC_FAMILIES,
            }
        )
    elif packet_type == "end_of_shift_survey":
        packet["survey_dimension"] = PERSONA_APPRAISAL_DIMENSIONS[(index - 1) % len(PERSONA_APPRAISAL_DIMENSIONS)]
        packet["user_message"] += (
            f"\nRate only this appraisal dimension: {packet['survey_dimension']}. "
            "Use rateability=insufficient_evidence and a null score when the supplied "
            "trace cannot support that appraisal; do not infer a rating from persona "
            "description alone."
        )
    elif packet_type == "end_of_shift_interview":
        question = INTERVIEW_QUESTIONS[(index - 1) % len(INTERVIEW_QUESTIONS)]
        packet["question_id"] = question["question_id"]
        packet["question"] = question["question"]
        packet[
            "user_message"
        ] += f"\nQuestion ({question['question_id']}): {question['question']}"
    packet["estimated_tokens"] = estimate_context_tokens(packet)
    return packet


def _load_episode_log(path: Path) -> list[dict[str, Any]]:
    paths = (
        sorted(path.rglob("part3_decision_episodes.jsonl")) if path.is_dir() else [path]
    )
    episodes: list[dict[str, Any]] = []
    for source_path in paths:
        with source_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                episode = json.loads(line)
                if episode.get("fixture_only_not_scientific_data") is not False:
                    raise ValueError(
                        f"Episode is not marked as real logging evidence: {source_path}:{line_number}"
                    )
                if not episode.get("evidence_id") or not episode.get(
                    "decision_context"
                ):
                    raise ValueError(
                        f"Incomplete episode evidence: {source_path}:{line_number}"
                    )
                episodes.append(episode)
    if not episodes:
        raise ValueError(f"No Part 3 decision episodes found under {path}")
    return episodes


def _opaque_evidence_id(source_evidence_id: str) -> str:
    digest = hashlib.sha256(source_evidence_id.encode("utf-8")).hexdigest()[:16]
    return f"evidence-{digest}"


def _opaque_entity_id(kind: str, metadata: Mapping[str, Any], value: Any) -> str | None:
    if value in (None, -1, "-1", ""):
        return None
    source = "|".join(
        (
            str(kind),
            str(metadata.get("scenario_mode")),
            str(metadata.get("seed")),
            str(value),
        )
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{kind}-{digest}"


def _memory_reason_label(value: Any) -> str | None:
    labels = {
        "POST_TASK_UPDATE": "post-task update",
        "SAME_PATIENT_UPDATE": "same-patient continuity",
        "HANDOFF_NEED": "handoff coordination",
        "BED_FLOW_COORDINATION": "bed-flow coordination",
        "HIGH_ACUITY_ESCALATION": "high-acuity escalation",
        "SAME_STATION_BRIEF_CHECKIN": "brief station check-in",
        "CORRIDOR_PASSING_UPDATE": "corridor update",
    }
    if value in (None, ""):
        return None
    return labels.get(str(value), str(value).lower().replace("_", " "))


def _llm_visible_decision_evidence(
    evidence: dict[str, Any], model_evidence_id: str
) -> dict[str, Any]:
    """Expose observable context while withholding evaluator/comparator fields."""

    metadata = evidence["metadata"]
    decision = evidence["decision_context"]
    spatial = evidence["spatial_context"]
    workflow = evidence["workflow_context"]
    perception = evidence.get("perception_context", {})
    patient_id = workflow.get("patient_id")
    patient_context_present = patient_id not in (None, -1, "-1")
    active_task_present = bool(
        workflow.get("initiator_is_task_busy", False)
    )

    def staff_mode_label(value: Any) -> Any:
        if value in (None, ""):
            return None
        return STAFF_MODE_LABELS.get(str(value), str(value))

    current_partner_key = _opaque_entity_id(
        "colleague", metadata, metadata.get("partner_id")
    )
    current_patient_key = _opaque_entity_id("patient", metadata, patient_id)
    visible_memory = []
    for memory in evidence.get("memory_state_before", []):
        source_memory_id = str(memory.get("memory_id", ""))
        visible_memory.append(
            {
                "memory_id": (
                    _opaque_evidence_id(source_memory_id)
                    if source_memory_id
                    else None
                ),
                "seconds_ago": memory.get("seconds_ago"),
                "outcome": memory.get("outcome"),
                "colleague_key": _opaque_entity_id(
                    "colleague", metadata, memory.get("partner_id")
                ),
                "colleague_role": memory.get("partner_role"),
                "interaction_direction": (
                    "initiated_by_self"
                    if memory.get("owner_was_initiator")
                    else "received_from_colleague"
                ),
                "same_current_colleague": memory.get("same_partner"),
                "patient_context_key": _opaque_entity_id(
                    "patient", metadata, memory.get("patient_context_id")
                ),
                "same_patient_context": memory.get("same_patient_context"),
                "topic_family": memory.get("topic_family"),
                "interaction_type": memory.get("interaction_type"),
                "reason_context": _memory_reason_label(
                    memory.get("reason_type")
                ),
                "zone": memory.get("zone"),
                "salience_features": list(
                    memory.get("salience_features") or []
                ),
                "retrieval_match_features": list(
                    memory.get("retrieval_match_features") or []
                ),
            }
        )

    return {
        "evidence_id": model_evidence_id,
        "action_semantics": ACTION_SEMANTICS,
        "score_semantics": SCORE_SEMANTICS,
        "patient_context_scope": (
            "A patient is linked. Only fields in patient_context may be used; "
            "nothing else about diagnosis, treatment, disposition, outcome, or "
            "subjective experience is known."
            if patient_context_present
            else "No particular patient or patient state is supplied."
        ),
        "staff_mode_semantics": (
            "All initiator_staff_mode and partner_staff_mode values describe staff "
            "workflow. returning_to_staff_station never means patient discharge. "
            "staff_waiting_for_doctor_coordination means that the staff member is "
            "waiting to coordinate with a doctor, not that a doctor is waiting."
        ),
        "memory_semantics": (
            "Memory records are retrieved, realized prior interactions. Use them "
            "only to judge repetition, continuity, prior coordination with the same "
            "colleague, or bounded information relay when a prior colleague discussed "
            "the same patient context. Do not infer unlisted facts or assume that a "
            "topic family contains any detail not shown here."
        ),
        "contact_opportunity": {
            "initiator_role": metadata.get("role"),
            "partner_role": metadata.get("partner_role"),
            "partner_key": current_partner_key,
            "interaction_type": decision.get("interaction_type"),
            "feasible_actions": list(decision.get("feasible_actions", [])),
            "allowed_reasons": list(decision.get("allowed_reasons", [])),
            "allowed_topic_families": list(
                decision.get("allowed_topic_families", [])
            ),
            "urgency": decision.get("urgency"),
            "urgency_band": _evidence_bucket(
                float(decision.get("urgency", 0.0)), 0.5, 0.8
            ),
            "salience": decision.get("salience"),
            "diversion_cost": decision.get("diversion_cost"),
            "diversion_cost_band": _evidence_bucket(
                float(decision.get("diversion_cost", 0.0)), 0.33, 0.67
            ),
            "zone": spatial.get("zone"),
            "distance_m": spatial.get("distance_m"),
            "mutual_visibility": spatial.get("mutual_visibility"),
            "visible_staff_count": perception.get("visible_staff_count"),
            "eligible_staff_opportunity_count": perception.get(
                "eligible_staff_opportunity_count"
            ),
            "attention_rank": perception.get("attention_rank"),
            "attention_capacity": perception.get("attention_capacity"),
            "initiator_staff_mode": staff_mode_label(
                workflow.get("initiator_mode")
            ),
            "initiator_task": workflow.get("initiator_task"),
            "active_task_present": active_task_present,
            "partner_staff_mode": staff_mode_label(workflow.get("target_mode")),
            "partner_task": workflow.get("target_task"),
            "opportunity_boundary": workflow.get("opportunity_boundary"),
            "patient_context_present": patient_context_present,
            "patient_context_key": current_patient_key,
            "active_patient_count": workflow.get("active_patient_count"),
            "ed_pressure_index": workflow.get("ed_pressure_index"),
            "ed_pressure_band": _evidence_bucket(
                float(workflow.get("ed_pressure_index", 0.0)), 0.5, 0.85
            ),
            "patient_context": (
                {
                    "esi_level": workflow.get("esi_level"),
                    "current_care_stage": workflow.get("patient_current_task"),
                    "time_in_stage_seconds": workflow.get(
                        "patient_task_elapsed_seconds"
                    ),
                    "bed_assigned": workflow.get("patient_bed_assigned"),
                    "zone": workflow.get("patient_zone"),
                    "high_acuity_escalation": workflow.get(
                        "patient_high_acuity_escalation"
                    ),
                    "waiting_for_placement": workflow.get(
                        "patient_waiting_for_placement"
                    ),
                }
                if patient_context_present
                else None
            ),
        },
        "memory_state_before": visible_memory,
    }


def _scientific_user_message(
    persona: dict[str, Any], visible_evidence: dict[str, Any]
) -> str:
    contact = visible_evidence["contact_opportunity"]
    return "\n".join(
        [
            "Packet type: in_simulation_decision",
            *_orientation_prompt_lines(persona),
            f"ABM role: {contact['initiator_role']}",
            f"Evidence id: {visible_evidence['evidence_id']}",
            "Observable decision evidence: "
            f"{json.dumps(visible_evidence, sort_keys=True)}",
            "Decision constraints:\n- " + "\n- ".join(HARD_CONSTRAINTS),
            (
                "A feasible opportunity is possible, not automatically worthwhile. "
                "Compare its operational value with attention and diversion cost."
            ),
            "Choose only from the supplied feasible actions, reasons, and topic families.",
            (
                "Return only the categorical decision fields required by the schema. "
                "Do not add narrative explanation or infer operational details."
            ),
        ]
    )


def _scientific_decision_packet(persona, evidence: dict[str, Any]) -> dict[str, Any]:
    profile = persona.prompt_profile()
    evidence = deepcopy(evidence)
    bounds = categorical_decision_bounds(evidence)
    evidence["decision_context"].update(bounds)
    decision_context = dict(evidence["decision_context"])
    source_evidence_id = str(evidence["evidence_id"])
    model_evidence_id = _opaque_evidence_id(source_evidence_id)
    visible_evidence = _llm_visible_decision_evidence(
        evidence, model_evidence_id
    )
    packet = {
        "prompt_id": f"decision-{model_evidence_id.removeprefix('evidence-')}-"
        f"{persona.persona_id}",
        "packet_type": "in_simulation_decision",
        "decision_output_contract": DECISION_OUTPUT_CONTRACT,
        "system_message": SYSTEM_MESSAGE,
        "user_message": _scientific_user_message(profile, visible_evidence),
        "required_output_schema": required_output_schema(
            "in_simulation_decision",
            decision_output_contract=DECISION_OUTPUT_CONTRACT,
        ),
        "role": evidence["metadata"]["role"],
        "user_model_profile": profile,
        "llm_visible_evidence": visible_evidence,
        "trace_evidence": evidence,
        "evidence_ids": [model_evidence_id],
        "feasible_actions": list(decision_context["feasible_actions"]),
        "allowed_reasons": list(decision_context["allowed_reasons"]),
        "allowed_topic_families": list(decision_context["allowed_topic_families"]),
        "hard_constraints": HARD_CONSTRAINTS,
        "fixture_only_not_scientific_data": False,
        "synthetic_design_probe_not_human_data": True,
        "ocean_display_excluded_from_prompt": True,
        "comparator_hidden_from_model": True,
    }
    packet["estimated_tokens"] = estimate_context_tokens(packet)
    return packet


def scientific_decision_packet(persona, evidence: dict[str, Any]) -> dict[str, Any]:
    """Build the preregistered categorical packet used by offline and live gates."""

    return _scientific_decision_packet(persona, evidence)


def _evidence_bucket(value: float, low_cut: float, high_cut: float) -> str:
    if value < low_cut:
        return "low"
    if value < high_cut:
        return "medium"
    return "high"


def _episode_features(episode: dict[str, Any]) -> set[str]:
    metadata = episode["metadata"]
    decision = episode["decision_context"]
    workflow = episode["workflow_context"]
    spatial = episode["spatial_context"]
    rule = episode["rule_reference"]
    esi = workflow.get("esi_level")
    interaction_type = str(decision.get("interaction_type", "unknown"))
    return {
        f"cell:{metadata.get('scenario_mode')}|{metadata.get('condition')}",
        f"scenario:{metadata.get('scenario_mode')}",
        f"condition:{metadata.get('condition')}",
        f"role:{metadata.get('role')}",
        f"reason:{decision.get('abm_reason_type')}",
        f"action:{rule.get('selected_action')}",
        f"type:{'corridor' if 'corridor' in interaction_type else 'station'}",
        f"visibility:{bool(spatial.get('mutual_visibility'))}",
        f"patient:{esi is not None}",
        f"acuity:{'none' if esi is None else 'high' if int(esi) <= 2 else 'lower'}",
        "urgency:"
        + _evidence_bucket(float(decision.get("urgency", 0.0)), 0.5, 0.8),
        "diversion:"
        + _evidence_bucket(float(decision.get("diversion_cost", 0.0)), 0.33, 0.67),
        "pressure:"
        + _evidence_bucket(
            float(workflow.get("ed_pressure_index", 0.0)), 0.5, 0.85
        ),
    }


def _evidence_coverage_summary(
    episodes: list[dict[str, Any]], *, selected: bool
) -> dict[str, Any]:
    feature_counts: Counter[str] = Counter()
    for episode in episodes:
        feature_counts.update(_episode_features(episode))
    missing = sorted(REQUIRED_EVIDENCE_FEATURES - set(feature_counts))
    return {
        "scope": "selected_matched_episodes" if selected else "retained_logging_pool",
        "episode_count": len(episodes),
        "feature_counts": dict(sorted(feature_counts.items())),
        "required_features": sorted(REQUIRED_EVIDENCE_FEATURES),
        "missing_required_features": missing,
        "coverage_pass": not missing,
        "active_task_episode_count": sum(
            bool(episode["workflow_context"].get("initiator_is_task_busy"))
            for episode in episodes
        ),
        "active_task_scope_note": (
            "External opportunistic contacts are logged only at ABM-approved, "
            "interruptible boundaries. Protected active clinical work is therefore "
            "not a persona-controlled action in this experiment."
        ),
    }


def _balanced_episode_sample(
    episodes: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Select matched episodes that cover the preregistered evidence contrasts."""

    feature_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def selection_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
        features = _episode_features(row)
        newly_covered = len(
            (features & REQUIRED_EVIDENCE_FEATURES)
            - {feature for feature, count in feature_counts.items() if count}
        )
        diversity = sum(
            1.0 / (1.0 + feature_counts[feature])
            for feature in features
            if not feature.startswith(("cell:", "action:"))
        )
        cell = next(feature for feature in features if feature.startswith("cell:"))
        cell_balance = 1.0 / (1.0 + feature_counts[cell])
        digest = hashlib.sha256(
            str(row["evidence_id"]).encode("utf-8")
        ).hexdigest()
        return (-float(newly_covered), -diversity, -cell_balance, digest)

    cell_order = [
        (scenario, condition)
        for scenario in ("high_load_high_acuity", "normal_load")
        for condition in ("baseline", "both", "cockpit_only", "nursta_only")
    ]
    for scenario, condition in cell_order:
        candidates = [
            episode
            for episode in episodes
            if episode["metadata"].get("scenario_mode") == scenario
            and episode["metadata"].get("condition") == condition
        ]
        if not candidates:
            raise ValueError(f"No evidence for required cell {scenario}/{condition}")
        chosen = min(candidates, key=selection_key)
        selected.append(chosen)
        selected_ids.add(str(chosen["evidence_id"]))
        feature_counts.update(_episode_features(chosen))

    while len(selected) < limit:
        candidates = [
            episode
            for episode in episodes
            if str(episode["evidence_id"]) not in selected_ids
        ]
        if not candidates:
            break
        chosen = min(candidates, key=selection_key)
        selected.append(chosen)
        selected_ids.add(str(chosen["evidence_id"]))
        feature_counts.update(_episode_features(chosen))

    coverage = _evidence_coverage_summary(selected, selected=True)
    if not coverage["coverage_pass"]:
        raise ValueError(
            "Selected evidence does not cover preregistered contrasts: "
            + ", ".join(coverage["missing_required_features"])
        )
    return selected


def build_scientific_packets(
    out_dir: Path,
    episode_log: Path,
    packet_limit: int | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    episodes = _load_episode_log(episode_log)
    legacy = [
        str(episode.get("evidence_id"))
        for episode in episodes
        if int(episode.get("episode_schema_version", 0)) < 2
    ]
    if legacy:
        raise ValueError(
            "Scientific packets require logging schema version 2. Rerun the "
            f"CPU-only evidence logger; legacy episode count={len(legacy)}."
        )
    pool_coverage = _evidence_coverage_summary(episodes, selected=False)
    if not pool_coverage["coverage_pass"]:
        raise ValueError(
            "Logging pool does not cover preregistered evidence contrasts: "
            + ", ".join(pool_coverage["missing_required_features"])
        )
    personas = default_cognitive_personas()
    if packet_limit is None:
        raise ValueError(
            f"Set --scientific-packet-limit explicitly; the go/no-go design uses "
            f"{SCIENTIFIC_PACKET_COUNT} packets."
        )
    if packet_limit != SCIENTIFIC_PACKET_COUNT:
        raise ValueError(
            f"the bounded go/no-go design requires exactly "
            f"{SCIENTIFIC_PACKET_COUNT} packets: {SCIENTIFIC_EPISODE_COUNT} real "
            f"episodes crossed with all {len(personas)} personas"
        )
    episodes = _balanced_episode_sample(episodes, packet_limit // len(personas))
    selected_coverage = _evidence_coverage_summary(episodes, selected=True)
    packets = [
        _scientific_decision_packet(persona, episode)
        for episode in episodes
        for persona in personas
    ]
    if not packets:
        raise ValueError("Scientific packet limit produced an empty packet set")
    prompt_path = (
        out_dir
        / "prompt_packets"
        / "scientific_decision_prompt_packets.jsonl"
    )
    inventory_path = out_dir / "scientific_prompt_inventory.csv"
    _write_jsonl(prompt_path, packets)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "packet_type",
                "persona_id",
                "evidence_id",
                "model_evidence_id",
                "scenario",
                "condition",
                "role",
                "partner_role",
                "abm_reason",
                "rule_reference_action",
                "zone",
                "initiator_mode",
                "patient_context_present",
                "active_task_present",
                "patient_care_stage",
                "urgency",
                "diversion_cost",
                "ed_pressure_index",
                "mutual_visibility",
                "construct_features",
                "allowed_topic_families",
                "estimated_tokens",
            ],
        )
        writer.writeheader()
        for packet in packets:
            writer.writerow(
                {
                    "prompt_id": packet["prompt_id"],
                    "packet_type": packet["packet_type"],
                    "persona_id": packet["user_model_profile"]["persona_id"],
                    "evidence_id": packet["trace_evidence"]["evidence_id"],
                    "model_evidence_id": packet["evidence_ids"][0],
                    "scenario": packet["trace_evidence"]["metadata"]["scenario_mode"],
                    "condition": packet["trace_evidence"]["metadata"]["condition"],
                    "role": packet["trace_evidence"]["metadata"]["role"],
                    "partner_role": packet["trace_evidence"]["metadata"][
                        "partner_role"
                    ],
                    "abm_reason": packet["trace_evidence"]["decision_context"][
                        "abm_reason_type"
                    ],
                    "rule_reference_action": packet["trace_evidence"][
                        "rule_reference"
                    ]["selected_action"],
                    "zone": packet["trace_evidence"]["spatial_context"]["zone"],
                    "initiator_mode": packet["llm_visible_evidence"][
                        "contact_opportunity"
                    ]["initiator_staff_mode"],
                    "patient_context_present": packet["llm_visible_evidence"][
                        "contact_opportunity"
                    ]["patient_context_present"],
                    "active_task_present": packet["llm_visible_evidence"][
                        "contact_opportunity"
                    ]["active_task_present"],
                    "patient_care_stage": packet["trace_evidence"][
                        "workflow_context"
                    ].get("patient_current_task"),
                    "urgency": packet["trace_evidence"]["decision_context"].get(
                        "urgency"
                    ),
                    "diversion_cost": packet["trace_evidence"][
                        "decision_context"
                    ].get("diversion_cost"),
                    "ed_pressure_index": packet["trace_evidence"][
                        "workflow_context"
                    ].get("ed_pressure_index"),
                    "mutual_visibility": packet["trace_evidence"][
                        "spatial_context"
                    ].get("mutual_visibility"),
                    "construct_features": "|".join(
                        sorted(_episode_features(packet["trace_evidence"]))
                    ),
                    "allowed_topic_families": "|".join(
                        packet["allowed_topic_families"]
                    ),
                    "estimated_tokens": packet["estimated_tokens"],
                }
            )
    return prompt_path, inventory_path, {
        "pool": pool_coverage,
        "selected": selected_coverage,
        "matched_episode_count": len(episodes),
        "persona_count": len(personas),
        "packet_count": len(packets),
    }


def build_validation_packets(out_dir: Path, packet_count: int = 8) -> tuple[Path, Path]:
    if packet_count < 4:
        raise ValueError("packet_count must be at least 4 to exercise all schemas")
    packet_types = [
        "in_simulation_decision",
        "end_of_shift_survey",
        "end_of_shift_interview",
        "critical_incident",
    ]
    packets = [_fixture_packet(packet_types[(index - 1) % 4], index) for index in range(1, packet_count + 1)]
    prompt_path = out_dir / "prompt_packets" / "schema_validation_prompt_packets.jsonl"
    inventory_path = out_dir / "prompt_inventory.csv"
    _write_jsonl(prompt_path, packets)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_id", "packet_type", "persona_id", "estimated_tokens", "fixture_only"],
        )
        writer.writeheader()
        for packet in packets:
            writer.writerow(
                {
                    "prompt_id": packet["prompt_id"],
                    "packet_type": packet["packet_type"],
                    "persona_id": packet["user_model_profile"]["persona_id"],
                    "estimated_tokens": packet["estimated_tokens"],
                    "fixture_only": True,
                }
            )
    return prompt_path, inventory_path


def write_protocol(
    out_dir: Path,
    scientific_sampling_summary: dict[str, Any] | None = None,
) -> Path:
    personas = default_cognitive_personas()
    assignments = balanced_cognitive_persona_assignments([1, 10, 11, 12, 13, 14, 20, 21, 22])
    protocol = {
        "status": "frozen_before_outcome_inspection",
        "scientific_unit": "role-independent cognitive orientation crossed with ABM role",
        "persona_ids": [persona.persona_id for persona in personas],
        "persona_count": len(personas),
        "balanced_assignment_rounds": 5,
        "balanced_assignment_rows": len(assignments),
        "conditions": ["baseline", "cockpit_only", "nursta_only", "both"],
        "scenarios": ["normal_load", "high_load_high_acuity"],
        "survey_dimensions": PERSONA_APPRAISAL_DIMENSIONS,
        "interview_questions": INTERVIEW_QUESTIONS,
        "canonical_topic_families": CANONICAL_TOPIC_FAMILIES,
        "topic_families_by_abm_reason": TOPIC_FAMILIES_BY_ABM_REASON,
        "topic_constraint_policy": (
            "union of logged empirical topic families and preregistered "
            "reason-grounded canonical families, with patient-specific families "
            "removed when no particular patient context is supplied"
        ),
        "ablation_variants": ABLATION_VARIANTS,
        "evaluation_families": EVALUATION_FAMILIES,
        "reasoning_mode_default": "disabled",
        "decision_output_contract": DECISION_OUTPUT_CONTRACT,
        "decision_output_boundary": (
            "Only action, bounded reason, bounded canonical topic family, and "
            "cited evidence enter the causal decision record. Free-text explanation "
            "and self-reported confidence are excluded from in-simulation decisions; "
            "qualitative reflection is a separate evidence-grounded endpoint."
        ),
        "prompt_blinding": (
            "Qwen receives observable decision evidence and opaque ids only; "
            "the rule action, random draw, ABM reason label, scenario label, "
            "condition label, and source ids remain evaluator-only."
        ),
        "scientific_packet_sampling": (
            "Twelve real matched episodes, including every scenario-condition cell, "
            "are each crossed with all five personas. Deterministic coverage selection "
            "requires contrasts in role, reason, patient linkage and acuity, urgency, "
            "diversion cost, pressure, visibility, and interaction setting. The hidden "
            "randomized rule-reference action is not a sampling criterion."
        ),
        "persona_action_boundary": (
            "The ABM exposes only externally interruptible optional staff-contact "
            "opportunities. Protected active clinical tasks are never persona-controlled."
        ),
        "scientific_sampling_summary": scientific_sampling_summary,
        "qualitative_method": "computational qualitative content analysis",
        "validity_boundary": [
            "synthetic cognitive design probes, not staff testimony",
            "no inference of human personality prevalence or clinical outcomes",
            "OCEAN is a derived ordinal presentation crosswalk and never conditions prompts",
            "scientific packets require agent-level event traces; run summaries are not substituted",
        ],
    }
    path = out_dir / "part3_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/part3/pilot_plan")
    parser.add_argument("--packet-count", type=int, default=8)
    parser.add_argument(
        "--episode-log",
        type=Path,
        default=None,
        help="File or directory containing logging-only part3_decision_episodes.jsonl",
    )
    parser.add_argument("--scientific-packet-limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true", help="Validate schemas without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for packet_type in (
        "in_simulation_decision",
        "end_of_shift_survey",
        "end_of_shift_interview",
        "critical_incident",
    ):
        required_output_schema(packet_type)
    packets = [_fixture_packet("in_simulation_decision", 1), _fixture_packet("end_of_shift_survey", 2)]
    if not all(packet["fixture_only_not_scientific_data"] for packet in packets):
        raise AssertionError("schema fixtures must be marked non-scientific")
    if args.check_only:
        print(json.dumps({"status": "PASS", "persona_count": 5, "schemas_checked": 4}, indent=2))
        return
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    scientific_sampling_summary = None
    if args.episode_log is None:
        prompt_path, inventory_path = build_validation_packets(out_dir, args.packet_count)
    else:
        episode_log = args.episode_log if args.episode_log.is_absolute() else PROJECT_DIR / args.episode_log
        prompt_path, inventory_path, scientific_sampling_summary = build_scientific_packets(
            out_dir,
            episode_log,
            args.scientific_packet_limit,
        )
    protocol_path = write_protocol(out_dir, scientific_sampling_summary)
    for path in (prompt_path, inventory_path, protocol_path):
        print(path)


if __name__ == "__main__":
    main()
