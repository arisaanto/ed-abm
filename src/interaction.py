"""Interaction backends, empirical personas, and lightweight memory streams.

The simulation keeps movement and task order rule-based. This module only
decides whether an eligible encounter becomes an interaction, what gets talked
about, and the remembered summary of that encounter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Mapping, Optional, Sequence

import config


ROLE_TO_PROFILE_KEY = {
    "CoordinationNurse": "coordination_nurse",
    "Nurse": "nurse",
    "Doctor": "doctor",
    "Patient": "support",
}

PART3_GROUNDED_MEMORY_POLICY = "grounded_realized_interactions_v1"
PART3_GROUNDED_MEMORY_MAX_AGE_SECONDS = 7200


@dataclass
class MemoryEvent:
    """One traceable remembered event or reflection."""

    memory_id: str
    timestamp: int
    event_type: str
    owner_agent_id: int
    owner_agent_name: str
    owner_role: str
    partner_id: Optional[int]
    partner_name: Optional[str]
    partner_role: str
    patient_id: Optional[int]
    esi_level: Optional[int]
    zone_id: str
    condition_name: str
    model_variant: str
    topic: str
    duration_seconds: int
    interaction_type: str
    summary: str
    importance: float
    emotional_tags: List[str]
    retrieval_keywords: List[str]
    source_event_id: Optional[str]
    support_ids: List[str]
    owner_was_initiator: bool
    reason_type: Optional[str] = None
    outcome: str = "realized"


@dataclass
class InteractionDecision:
    """Structured interaction result consumed by the simulation logger."""

    interact: bool
    partner_id: int
    interaction_type: str
    topic: str
    duration_seconds: int
    memory_summary: str
    reason: str = ""
    importance: float = 0.5
    retrieved_memory_ids: List[str] = None
    retrieved_memory_summaries: List[str] = None
    raw_response: Optional[str] = None
    parsed_response: Optional[dict] = None
    used_fallback: bool = False
    parse_error: Optional[str] = None
    memory_effect_applied: bool = False
    memory_effect_type: str = ""
    memory_ids_used: List[str] = None
    accepted_because_of_memory: bool = False
    rejected_because_of_memory: bool = False
    decision_flipped_by_memory: bool = False
    probability_before: Optional[float] = None
    probability_after: Optional[float] = None
    duration_before: Optional[int] = None
    duration_after: Optional[int] = None
    topic_before: Optional[str] = None
    topic_after: Optional[str] = None
    backend_name: str = "rule_stub"
    prompt_context_hash: Optional[str] = None
    context_summary: Optional[str] = None
    persona_id: Optional[str] = None


class MemoryStream:
    """Smallville-style ordered memory store with lightweight retrieval."""

    def __init__(self) -> None:
        self.events: List[MemoryEvent] = []
        self.retrieval_traces: List[dict] = []

    def add(self, event: MemoryEvent) -> None:
        self.events.append(event)

    def reflect(self, current_timestamp: int, owner_role: str) -> Optional[MemoryEvent]:
        if not self.events:
            return None
        recent_events = [
            event for event in self.events[-config.LLM_REFLECTION_WINDOW:]
            if event.event_type != "reflection"
        ]
        if not recent_events:
            return None

        top_topics: Dict[str, int] = {}
        top_zones: Dict[str, int] = {}
        for event in recent_events:
            top_topics[event.topic] = top_topics.get(event.topic, 0) + 1
            top_zones[event.zone_id] = top_zones.get(event.zone_id, 0) + 1

        main_topic = max(top_topics, key=top_topics.get)
        main_zone = max(top_zones, key=top_zones.get)
        summary = (
            f"As a {owner_role}, recent work has centered on {main_topic} "
            f"around {main_zone}."
        )
        return MemoryEvent(
            memory_id=f"reflection_{owner_role}_{current_timestamp}_{len(self.events) + 1}",
            timestamp=current_timestamp,
            event_type="reflection",
            owner_agent_id=-1,
            owner_agent_name=owner_role,
            owner_role=owner_role,
            partner_id=None,
            partner_name=None,
            partner_role="self_reflection",
            patient_id=None,
            esi_level=None,
            zone_id=main_zone,
            condition_name="unknown",
            model_variant="unknown",
            topic=main_topic,
            duration_seconds=0,
            interaction_type="reflection",
            summary=summary,
            importance=0.6,
            emotional_tags=["synthesis"],
            retrieval_keywords=[main_topic, main_zone, owner_role],
            source_event_id=None,
            support_ids=[event.memory_id for event in recent_events],
            owner_was_initiator=False,
            reason_type=None,
            outcome="reflection",
        )

    def retrieve(
        self,
        current_timestamp: int,
        query_terms: Sequence[str],
        limit: int,
    ) -> List[MemoryEvent]:
        lowered_terms = {term.lower() for term in query_terms if term}

        scored_events = []

        def score(event: MemoryEvent) -> float:
            recency_seconds = max(current_timestamp - event.timestamp, 0)
            recency_score = 1.0 / (1.0 + (recency_seconds / 300.0))
            relevance_score = 0.0
            searchable_text = " ".join(
                [
                    event.partner_role,
                    event.zone_id,
                    event.topic,
                    event.summary,
                    event.interaction_type,
                    " ".join(event.retrieval_keywords),
                ]
            ).lower()
            for term in lowered_terms:
                if term in searchable_text:
                    relevance_score += 1.0

            return (
                (recency_score * config.MEMORY_RECENCY_WEIGHT)
                + (relevance_score * config.MEMORY_RELEVANCE_WEIGHT)
                + (event.importance * config.MEMORY_IMPORTANCE_WEIGHT)
            )

        for event in self.events:
            scored_events.append((score(event), event))
        ranked_events = [event for _, event in sorted(scored_events, key=lambda item: item[0], reverse=True)]
        self.retrieval_traces.append(
            {
                "timestamp": current_timestamp,
                "query_terms": list(query_terms),
                "returned_memory_ids": [event.memory_id for event in ranked_events[:limit]],
                "scores": [
                    {"memory_id": event.memory_id, "score": round(score_value, 4)}
                    for score_value, event in sorted(scored_events, key=lambda item: item[0], reverse=True)[:limit]
                ],
            }
        )
        return ranked_events[:limit]

    def retrieve_grounded_part3_context(
        self,
        *,
        current_timestamp: int,
        partner_id: Optional[int],
        patient_context_id: Optional[int],
        reason_type: str,
        interaction_type: str,
        zone_id: str,
        limit: int,
        salience_modifiers: Mapping[str, float],
    ) -> List[Dict[str, object]]:
        """Retrieve realized interactions for the bounded Part 3 policy.

        This deliberately excludes reflections, missed opportunities, and prior
        model choices. The LLM sees structured facts about experienced contacts,
        while generated prose remains outside the causal decision path.
        """

        candidates = [
            event
            for event in self.events
            if event.event_type == "communicative_interaction"
            and event.outcome == "realized"
            and event.source_event_id
            and 0
            <= int(current_timestamp) - int(event.timestamp)
            <= PART3_GROUNDED_MEMORY_MAX_AGE_SECONDS
        ]
        scored: List[
            tuple[float, MemoryEvent, Dict[str, float], List[str], List[str]]
        ] = []
        for event in candidates:
            age_seconds = max(int(current_timestamp) - int(event.timestamp), 0)
            recency = 1.0 / (1.0 + (age_seconds / 900.0))
            same_partner = partner_id is not None and event.partner_id == partner_id
            same_patient = (
                patient_context_id is not None
                and event.patient_id == patient_context_id
            )
            match_features = []
            if same_partner:
                match_features.append("same_colleague")
            if same_patient:
                match_features.append("same_patient_context")
            if event.reason_type == reason_type:
                match_features.append("same_operational_reason")
            if event.interaction_type == interaction_type:
                match_features.append("same_interaction_context")
            if event.zone_id == zone_id and age_seconds <= 900:
                match_features.append("same_recent_zone")
            if not match_features:
                continue
            relevance_points = (
                (3.0 if same_partner else 0.0)
                + (3.0 if same_patient else 0.0)
                + (1.5 if event.reason_type == reason_type else 0.0)
                + (1.0 if event.interaction_type == interaction_type else 0.0)
                + (0.5 if event.zone_id == zone_id else 0.0)
            )
            relevance = relevance_points / 9.0
            salience_features = self._grounded_salience_features(event)
            salience_multiplier = max(
                [
                    float(salience_modifiers.get(feature, 1.0))
                    for feature in salience_features
                ]
                or [1.0]
            )
            components = {
                "recency": recency,
                "relevance": relevance,
                "importance": float(event.importance),
                "salience_multiplier": salience_multiplier,
            }
            score = (
                (recency * config.MEMORY_RECENCY_WEIGHT)
                + (relevance * config.MEMORY_RELEVANCE_WEIGHT)
                + (float(event.importance) * config.MEMORY_IMPORTANCE_WEIGHT)
            ) * salience_multiplier
            scored.append(
                (
                    score,
                    event,
                    components,
                    salience_features,
                    match_features,
                )
            )

        ranked = sorted(
            scored,
            key=lambda item: (-item[0], -item[1].timestamp, item[1].memory_id),
        )[: max(int(limit), 0)]
        records: List[Dict[str, object]] = []
        for rank, (
            score,
            event,
            components,
            features,
            match_features,
        ) in enumerate(ranked, start=1):
            records.append(
                {
                    "memory_id": event.memory_id,
                    "timestamp": int(event.timestamp),
                    "seconds_ago": max(
                        int(current_timestamp) - int(event.timestamp), 0
                    ),
                    "event_type": event.event_type,
                    "outcome": event.outcome,
                    "partner_id": event.partner_id,
                    "partner_role": event.partner_role,
                    "owner_was_initiator": bool(event.owner_was_initiator),
                    "patient_context_id": event.patient_id,
                    "zone": event.zone_id,
                    "topic_family": event.topic,
                    "interaction_type": event.interaction_type,
                    "reason_type": event.reason_type,
                    "importance": float(event.importance),
                    "same_partner": bool(
                        partner_id is not None and event.partner_id == partner_id
                    ),
                    "same_patient_context": bool(
                        patient_context_id is not None
                        and event.patient_id == patient_context_id
                    ),
                    "retrieval_rank": rank,
                    "retrieval_score": round(score, 8),
                    "retrieval_components": {
                        key: round(value, 8) for key, value in components.items()
                    },
                    "salience_features": features,
                    "retrieval_match_features": sorted(match_features),
                    "source_event_id": event.source_event_id,
                    "support_ids": list(event.support_ids),
                    "source": "realized_communicative_interaction",
                    "memory_policy": PART3_GROUNDED_MEMORY_POLICY,
                }
            )
        self.retrieval_traces.append(
            {
                "timestamp": int(current_timestamp),
                "retrieval_policy": PART3_GROUNDED_MEMORY_POLICY,
                "partner_id": partner_id,
                "patient_context_id": patient_context_id,
                "reason_type": reason_type,
                "interaction_type": interaction_type,
                "zone_id": zone_id,
                "returned_memory_ids": [row["memory_id"] for row in records],
                "scores": [
                    {
                        "memory_id": row["memory_id"],
                        "score": row["retrieval_score"],
                    }
                    for row in records
                ],
            }
        )
        return records

    @staticmethod
    def _grounded_salience_features(event: MemoryEvent) -> List[str]:
        features: List[str] = []
        if event.partner_role != "Patient":
            features.append("coordination")
        if "handoff" in event.interaction_type.lower() or "handoff" in str(
            event.reason_type or ""
        ).lower():
            features.append("handoff")
        if event.partner_role == "Patient" or event.patient_id is not None:
            features.append("patient_facing")
        if event.esi_level in {1, 2}:
            features.append("high_acuity")
        if "opportunistic" in event.interaction_type.lower():
            features.extend(["interruption", "context_switch", "tradeoff"])
        if str(event.reason_type or "") in {
            "POST_TASK_UPDATE",
            "SAME_PATIENT_UPDATE",
        }:
            features.append("task_continuity")
        return sorted(set(features))


class EmpiricalBehaviorLibrary:
    """Loads persona text and empirically observed topic/duration priors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
        else:
            self.data = {
                "behavioral_profiles": {},
                "quantitative": {},
            }
        self.quantitative_topics = self.data.get("quantitative", {}).get(
            "topic_distributions_by_role_and_zone", {}
        )

    def role_profile(self, role: str) -> Mapping[str, object]:
        profile_key = ROLE_TO_PROFILE_KEY.get(role, role.lower())
        return self.data.get("behavioral_profiles", {}).get(profile_key, {})

    def persona_paragraph(self, role: str, source: str = "empirical") -> str:
        if source == "generic":
            return (
                f"You are a {role} working in a busy Swiss emergency department. "
                "You coordinate safely, speak briefly, and focus on clinically relevant updates."
            )
        profile = self.role_profile(role)
        return str(profile.get("persona_paragraph", f"You are a {role} in the ED workflow."))

    def topics_for(self, role: str, zone_id: Optional[str]) -> List[str]:
        profile = self.role_profile(role)
        topics_by_zone = profile.get("typical_topics_by_zone", {})
        if zone_id and zone_id in topics_by_zone:
            return list(topics_by_zone[zone_id])

        collected_topics = []
        for topic_list in topics_by_zone.values():
            collected_topics.extend(topic_list)
        return collected_topics or ["patient_status_update"]

    def topic_weights_for(self, role: str, zone_id: Optional[str]) -> Dict[str, float]:
        role_key = ROLE_TO_PROFILE_KEY.get(role, role.lower())
        zone_topics = self.quantitative_topics.get(role_key, {})
        weighted_topics = zone_topics.get(zone_id or "", {})
        if isinstance(weighted_topics, dict) and weighted_topics:
            return {
                topic_name: float(payload.get("proportion_within_role_zone", 0.0))
                if isinstance(payload, dict)
                else float(payload)
                for topic_name, payload in weighted_topics.items()
            }
        topics = self.topics_for(role, zone_id)
        if not topics:
            return {"patient_status_update": 1.0}
        uniform_weight = 1.0 / len(topics)
        return {topic_name: uniform_weight for topic_name in topics}

    def duration_prior(self, role: str, partner_role: str) -> tuple[int, int]:
        profile = self.role_profile(role)
        partner_key = partner_role.lower()
        partner_stats = profile.get("typical_durations_by_partner", {}).get(partner_key)
        if isinstance(partner_stats, dict):
            mean_seconds = max(int(float(partner_stats.get("mean_seconds", 30))), 10)
            std_seconds = max(int(float(partner_stats.get("std_seconds", 10))), 5)
            return (max(mean_seconds - std_seconds, 10), mean_seconds + std_seconds)
        return (15, 45)


def build_interaction_context(
    agent,
    partner,
    simulation,
    interaction_type: str,
    task_name: Optional[str],
    position,
    memories: Sequence[MemoryEvent],
    library: EmpiricalBehaviorLibrary,
) -> Dict[str, object]:
    zone_id = simulation.which_zone(*position)
    persona_source = getattr(simulation, "persona_source", config.PERSONA_SOURCE)
    patient = partner if getattr(partner, "role", None) == "Patient" else None
    if patient is None and task_name is not None:
        patient = simulation.patient_by_id.get(getattr(agent, "target_patient_id", None))
    persona = getattr(agent, "persona", None)
    partner_persona = getattr(partner, "persona", None)
    if persona_source == "generic":
        generic_topics = sorted(set(library.topics_for(agent.role, None)))
        if not generic_topics:
            generic_topics = ["patient_status_update"]
        uniform_weight = 1.0 / len(generic_topics)
        topic_weights = {topic_name: uniform_weight for topic_name in generic_topics}
    else:
        topic_weights = library.topic_weights_for(agent.role, zone_id)
    topic_weights = _contextual_topic_weights(
        topic_weights,
        role=agent.role,
        partner_role=partner.role,
        interaction_type=interaction_type,
        task_name=task_name,
        zone_id=zone_id,
        esi_level=getattr(patient, "esi_level", None),
        persona=persona,
    )
    return {
        "role": agent.role,
        "partner_role": partner.role,
        "partner_id": partner.gid,
        "agent_id": agent.gid,
        "agent_name": getattr(agent, "name", f"{agent.role} {agent.gid}"),
        "partner_name": getattr(partner, "name", f"{partner.role} {partner.gid}"),
        "zone_id": zone_id,
        "interaction_type": interaction_type,
        "task_name": task_name,
        "position": position,
        "current_mode": getattr(agent, "mode", "idle"),
        "persona_paragraph": (
            getattr(persona, "base_persona_paragraph", None)
            or library.persona_paragraph(agent.role, persona_source)
        ),
        "persona": persona.as_dict() if hasattr(persona, "as_dict") else None,
        "partner_persona": partner_persona.as_dict() if hasattr(partner_persona, "as_dict") else None,
        "persona_source": persona_source,
        "topic_candidates": list(topic_weights),
        "topic_weights": topic_weights,
        "duration_prior": library.duration_prior(agent.role, partner.role),
        "distance_to_partner": getattr(agent, "distance_to_agent", lambda _: 0.0)(partner),
        "condition_name": simulation.condition_spec.name,
        "model_variant": getattr(simulation, "model_variant", config.MODEL_VARIANT),
        "patient_id": getattr(patient, "gid", None),
        "esi_level": getattr(patient, "esi_level", None),
        "urgency": getattr(patient, "urgency", None),
        "complaint_type": getattr(patient, "complaint_type", None),
        "workload": len(simulation.active_patients),
        "recent_memories": [asdict(memory) for memory in memories],
    }


class InteractionBackend(ABC):
    """Backend interface for deterministic or LLM-driven interaction decisions."""

    @abstractmethod
    def decide_interaction(self, context: Mapping[str, object], rng: random.Random) -> InteractionDecision:
        raise NotImplementedError


class RuleStubBackend(InteractionBackend):
    """Deterministic offline backend for debugging and calibration iteration."""

    def decide_interaction(self, context: Mapping[str, object], rng: random.Random) -> InteractionDecision:
        return _rule_decision(context, rng, apply_memory_effects=False, backend_name="rule_stub")


def _choose_weighted_topic(topic_weights: Mapping[str, float], rng: random.Random) -> str:
    topic_candidates = list(topic_weights)
    if topic_candidates:
        return rng.choices(
            population=topic_candidates,
            weights=[max(topic_weights.get(candidate, 0.0), 0.001) for candidate in topic_candidates],
            k=1,
        )[0]
    return "patient_status_update"


def _contextual_topic_weights(
    base_weights: Mapping[str, float],
    *,
    role: str,
    partner_role: str,
    interaction_type: str,
    task_name: Optional[str],
    zone_id: Optional[str],
    esi_level: Optional[int],
    persona,
) -> Dict[str, float]:
    """Blend empirical priors with ED task, role-pair, acuity, and persona context."""

    weights = {str(topic): max(float(weight), 0.01) for topic, weight in base_weights.items()}

    def add(topic: str, amount: float) -> None:
        weights[topic] = weights.get(topic, 0.0) + amount

    placement_context = (
        interaction_type in {"placement_handoff", "placement_nurse_alert"}
        or task_name == config.PLACEMENT_TASK_NAME
    )
    transition_context = interaction_type in {
        "task_transition_update",
        "doctor_doctor_coreview",
        "senior_doctor_oversight_review",
        "senior_doctor_room_assist",
    }

    if not weights or set(weights) == {"patient_status_update"}:
        add("patient_status_update", 0.6)
        add("next_steps_and_planning", 0.35)
        add("bed_and_capacity_management", 0.15)
        add("treatment_and_orders", 0.25)
        add("diagnostic_findings", 0.2)
        add("patient_facing_care", 0.2)

    if partner_role == "Patient":
        add("patient_facing_care", 1.2)
        add("next_steps_and_planning", 0.5)
        add("diagnostic_findings" if role == "Doctor" else "treatment_and_orders", 0.55)
    else:
        add("patient_status_update", 0.7 if transition_context else 0.55)
        if placement_context:
            add("bed_and_capacity_management", 0.45)
        else:
            add("bed_and_capacity_management", 0.08)
            add("next_steps_and_planning", 0.25)

    if "handoff" in interaction_type:
        add("patient_status_update", 0.8)
        add("patient_flow_and_transfer", 0.55)
        add("next_steps_and_planning", 0.45)
    if placement_context:
        add("bed_and_capacity_management", 1.0)
        add("patient_flow_and_transfer", 0.55)
    if interaction_type == "task_transition_update":
        add("patient_status_update", 1.25)
        add("next_steps_and_planning", 0.45)
    if interaction_type in {"doctor_doctor_coreview", "senior_doctor_oversight_review", "senior_doctor_room_assist"}:
        add("patient_status_update", 1.1)
        add("diagnostic_findings", 0.9)
        add("next_steps_and_planning", 0.7)
    if task_name == config.INITIAL_NURSING_ASSESSMENT_TASK_NAME:
        add("patient_facing_care", 0.7)
        add("patient_status_update", 0.65)
    if task_name == config.MEDICAL_EVALUATION_TASK_NAME:
        add("diagnostic_findings", 0.75)
        add("next_steps_and_planning", 0.55)
        add("patient_status_update", 0.45)
    if task_name == config.DIAGNOSTICS_TASK_NAME:
        add("diagnostic_findings", 1.0)
        add("patient_status_update", 0.45)
    if task_name == config.TREATMENT_TASK_NAME:
        add("treatment_and_orders", 1.0)
    if task_name == config.DISCHARGE_TASK_NAME:
        add("next_steps_and_planning", 0.8)
        add("patient_flow_and_transfer", 0.45)
        add("patient_status_update", 0.35)

    if zone_id in config.STATION_ZONE_IDS:
        if placement_context:
            add("bed_and_capacity_management", 0.6)
        else:
            add("patient_status_update", 0.55)
            add("bed_and_capacity_management", 0.12)
        add("administrative_and_logistics", 0.25)
    if zone_id and str(zone_id).startswith("CORR"):
        add("patient_flow_and_transfer", 0.55)
        add("patient_status_update", 0.35)
    if esi_level in {1, 2}:
        add("patient_status_update", 0.8)
        add("treatment_and_orders", 0.65)
        add("diagnostic_findings", 0.55)

    if hasattr(persona, "topic_preferences"):
        for index, topic in enumerate(getattr(persona, "topic_preferences", [])):
            add(str(topic), max(0.45 - (0.08 * index), 0.18))

    total = sum(weights.values())
    if total <= 0:
        return {"patient_status_update": 1.0}
    return {topic: weight / total for topic, weight in weights.items()}


def _base_duration_bounds(context: Mapping[str, object]) -> tuple[int, int]:
    interaction_type = str(context.get("interaction_type", "workflow_task"))
    duration_low, duration_high = context.get("duration_prior", (15, 45))
    esi_level = context.get("esi_level")
    if interaction_type in {"placement_handoff", "nurse_doctor_handoff"}:
        duration_low = max(duration_low, 15)
        duration_high = max(duration_high, 45)
    elif interaction_type.startswith("opportunistic_"):
        duration_low = 8
        duration_high = 28
    elif interaction_type in {"nursing_task", "doctor_task"}:
        duration_low = max(duration_low, 20)
        duration_high = max(duration_high, 90)
    if esi_level in {1, 2}:
        duration_low = max(8, int(duration_low * 0.85))
        duration_high = max(duration_low + 5, int(duration_high * 1.15))
    return int(duration_low), int(duration_high)


def _memory_effect(
    context: Mapping[str, object],
    probability: float,
    duration_seconds: int,
    topic: str,
) -> tuple[float, int, str, dict]:
    persona = context.get("persona") or {}
    memories = list(context.get("recent_memories", []))
    interaction_type = str(context.get("interaction_type", ""))
    partner_role = str(context.get("partner_role", ""))
    esi_level = context.get("esi_level")
    effect = {
        "applied": False,
        "type": "",
        "memory_ids": [],
        "probability_before": probability,
        "duration_before": duration_seconds,
        "topic_before": topic,
    }
    if not memories:
        effect["probability_after"] = probability
        effect["duration_after"] = duration_seconds
        effect["topic_after"] = topic
        effect["accepted_because_of_memory"] = False
        effect["rejected_because_of_memory"] = False
        return probability, duration_seconds, topic, effect

    relevant = memories[: config.MEMORY_RETRIEVAL_LIMIT]
    effect["memory_ids"] = [str(memory.get("memory_id")) for memory in relevant if memory.get("memory_id")]
    modifier = 0.0
    duration_modifier = 0
    effect_types = []
    topic_after = topic

    if any(str(memory.get("partner_role")) == partner_role for memory in relevant):
        modifier += 0.16
        duration_modifier += 6
        effect_types.append("recent_partner_followup")
    if any("missed" in str(memory.get("event_type", "")) or "missed" in " ".join(memory.get("emotional_tags", [])) for memory in relevant):
        modifier += 0.22
        effect_types.append("recent_missed_opportunity")
    if esi_level in {1, 2} or any(float(memory.get("importance", 0.0)) >= 0.8 for memory in relevant):
        modifier += 0.14
        duration_modifier += 10
        effect_types.append("high_importance_patient_memory")
        if str(context.get("partner_role")) != "Patient":
            topic_after = "patient_status_update"

    interruption_tolerance = float(persona.get("interruption_tolerance", 0.5)) if isinstance(persona, dict) else 0.5
    current_mode = str(context.get("current_mode", "idle"))
    if (
        interaction_type.startswith("opportunistic_")
        and interruption_tolerance < 0.45
        and esi_level not in {1, 2}
        and current_mode not in {"idle", "returning_home"}
    ):
        modifier -= 0.18
        duration_modifier -= 5
        effect_types.append("interruption_averse_persona")
    if interaction_type.startswith("opportunistic_") and interruption_tolerance > 0.7:
        modifier += 0.14
        effect_types.append("high_collaboration_persona")
        if str(context.get("zone_id")) in set(config.STATION_ZONE_IDS):
            topic_after = "bed_and_capacity_management"

    probability_after = min(max(probability + modifier, 0.05), 0.99)
    duration_after = max(5, duration_seconds + duration_modifier)
    effect["applied"] = (
        abs(probability_after - probability) > 1e-9
        or duration_after != duration_seconds
        or topic_after != topic
    )
    effect["type"] = "+".join(effect_types)
    effect["probability_after"] = probability_after
    effect["duration_after"] = duration_after
    effect["topic_after"] = topic_after
    effect["accepted_because_of_memory"] = probability < 0.5 <= probability_after
    effect["rejected_because_of_memory"] = probability >= 0.5 > probability_after
    return probability_after, duration_after, topic_after, effect


def _rule_decision(
    context: Mapping[str, object],
    rng: random.Random,
    *,
    apply_memory_effects: bool,
    backend_name: str,
) -> InteractionDecision:
        topic_weights = dict(context.get("topic_weights", {}))
        topic = _choose_weighted_topic(topic_weights, rng)

        interaction_type = str(context.get("interaction_type", "workflow_task"))
        duration_low, duration_high = _base_duration_bounds(context)
        esi_level = context.get("esi_level")
        duration_seconds = rng.randint(int(duration_low), int(duration_high))
        zone_id = context.get("zone_id", "unknown_zone")
        partner_role = context.get("partner_role", "colleague")
        retrieved = list(context.get("recent_memories", []))
        memory_summary = (
            f"{context.get('agent_name', context.get('role', 'Staff'))} discussed {topic} with {partner_role} "
            f"in {zone_id} during {interaction_type}."
        )
        if esi_level in {1, 2}:
            memory_summary += f" Patient acuity was ESI {esi_level}, making the exchange more salient."
        interact_probability = 1.0
        if interaction_type.startswith("opportunistic_"):
            interact_probability = 0.72
            if context.get("model_variant") == "traditional_rule":
                interact_probability = 0.75
            elif context.get("model_variant") in {"memory_rule", "generative_interaction"}:
                interact_probability = min(0.86, 0.45 + (0.05 * len(retrieved)))
            if esi_level in {1, 2}:
                interact_probability = min(0.99, interact_probability + 0.08)
        probability_before = interact_probability
        duration_before = duration_seconds
        memory_effect = {
            "applied": False,
            "type": "",
            "memory_ids": [str(memory.get("memory_id")) for memory in retrieved if memory.get("memory_id")],
            "probability_before": probability_before,
            "probability_after": interact_probability,
            "duration_before": duration_before,
            "duration_after": duration_seconds,
        }
        if apply_memory_effects:
            interact_probability, duration_seconds, topic, memory_effect = _memory_effect(
                context,
                interact_probability,
                duration_seconds,
                topic,
            )
        decision_roll = rng.random()
        would_interact_before_memory = (
            True if not interaction_type.startswith("opportunistic_") else decision_roll < probability_before
        )
        interact_after_memory = (
            True if not interaction_type.startswith("opportunistic_") else decision_roll < interact_probability
        )
        decision_flipped_by_memory = would_interact_before_memory != interact_after_memory
        if decision_flipped_by_memory:
            memory_effect["accepted_because_of_memory"] = (not would_interact_before_memory) and interact_after_memory
            memory_effect["rejected_because_of_memory"] = would_interact_before_memory and (not interact_after_memory)
        return InteractionDecision(
            interact=interact_after_memory,
            partner_id=int(context["partner_id"]),
            interaction_type=interaction_type,
            topic=topic,
            duration_seconds=duration_seconds,
            memory_summary=memory_summary,
            reason="rule_fallback_eligible_encounter",
            importance=0.85 if esi_level in {1, 2} else min(1.0, duration_seconds / 90.0),
            retrieved_memory_ids=[str(memory.get("memory_id")) for memory in retrieved],
            retrieved_memory_summaries=[str(memory.get("summary")) for memory in retrieved],
            parsed_response={
                "interact_probability": interact_probability,
                "bounded_decision": True,
                "memory_effect": memory_effect,
            },
            memory_effect_applied=bool(memory_effect["applied"]),
            memory_effect_type=str(memory_effect["type"]),
            memory_ids_used=list(memory_effect["memory_ids"]),
            accepted_because_of_memory=bool(memory_effect.get("accepted_because_of_memory", False)),
            rejected_because_of_memory=bool(memory_effect.get("rejected_because_of_memory", False)),
            decision_flipped_by_memory=decision_flipped_by_memory,
            probability_before=float(memory_effect["probability_before"]),
            probability_after=float(memory_effect["probability_after"]),
            duration_before=int(memory_effect["duration_before"]),
            duration_after=int(memory_effect["duration_after"]),
            topic_before=str(memory_effect.get("topic_before", topic)),
            topic_after=str(memory_effect.get("topic_after", topic)),
            backend_name=backend_name,
        )


class MemoryRuleBackend(InteractionBackend):
    """No-LLM backend where retrieved memories change bounded decisions."""

    def decide_interaction(self, context: Mapping[str, object], rng: random.Random) -> InteractionDecision:
        return _rule_decision(
            context,
            rng,
            apply_memory_effects=True,
            backend_name="memory_rule",
        )


class StructuredGenerativeFallbackBackend(InteractionBackend):
    """Seeded JSON-like fallback that uses persona, context, and memory distinctly."""

    def decide_interaction(self, context: Mapping[str, object], rng: random.Random) -> InteractionDecision:
        decision = _rule_decision(
            context,
            rng,
            apply_memory_effects=True,
            backend_name="structured_generative_fallback",
        )
        persona = context.get("persona") or {}
        persona_id = persona.get("persona_id") if isinstance(persona, dict) else None
        memories = list(context.get("recent_memories", []))
        context_summary = {
            "agent": context.get("agent_name"),
            "role": context.get("role"),
            "partner_role": context.get("partner_role"),
            "zone": context.get("zone_id"),
            "interaction_type": context.get("interaction_type"),
            "esi": context.get("esi_level"),
            "memory_ids": [memory.get("memory_id") for memory in memories[:3]],
        }
        prompt_context_hash = hashlib.sha1(
            json.dumps(context_summary, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        style = "focused"
        if isinstance(persona, dict):
            style = str(persona.get("communication_style", "focused")).split(",")[0]
        topic_preferences = persona.get("topic_preferences", []) if isinstance(persona, dict) else []
        if topic_preferences and rng.random() < 0.55:
            allowed = set(context.get("topic_candidates", []))
            preferred = [topic for topic in topic_preferences if topic in allowed]
            if preferred:
                decision.topic_before = decision.topic_before or decision.topic
                decision.topic = rng.choice(preferred)
                decision.topic_after = decision.topic
        base_probability = float(decision.probability_after or decision.probability_before or 1.0)
        adjusted_probability = base_probability
        adjustment_reasons = []
        if context.get("esi_level") in {1, 2}:
            adjusted_probability += 0.12
            adjustment_reasons.append("urgent_patient")
        if context.get("partner_role") == "Patient":
            adjusted_probability += 0.10
            adjustment_reasons.append("patient_facing_need")
        if str(context.get("zone_id")) in set(config.STATION_ZONE_IDS):
            adjusted_probability += 0.08
            adjustment_reasons.append("station_coordination")
        if any("missed" in str(memory.get("event_type", "")) for memory in memories):
            adjusted_probability += 0.14
            adjustment_reasons.append("recent_missed_opportunity")
        if not memories and str(context.get("interaction_type", "")).startswith("opportunistic_"):
            adjusted_probability -= 0.04
            adjustment_reasons.append("low_memory_relevance")
        interruption_tolerance = (
            float(persona.get("interruption_tolerance", 0.5))
            if isinstance(persona, dict)
            else 0.5
        )
        if (
            str(context.get("interaction_type", "")).startswith("opportunistic_")
            and interruption_tolerance < 0.45
            and context.get("esi_level") not in {1, 2}
        ):
            adjusted_probability -= 0.10
            adjustment_reasons.append("interruption_averse_nonurgent")
        adjusted_probability = min(max(adjusted_probability, 0.08), 0.99)
        previous_interact = decision.interact
        if str(context.get("interaction_type", "")).startswith("opportunistic_"):
            decision.interact = rng.random() < adjusted_probability
        decision.accepted_because_of_memory = bool(
            decision.accepted_because_of_memory or (not previous_interact and decision.interact and adjusted_probability > base_probability)
        )
        decision.rejected_because_of_memory = bool(
            decision.rejected_because_of_memory or (previous_interact and not decision.interact and adjusted_probability < base_probability)
        )
        decision.probability_before = decision.probability_before if decision.probability_before is not None else base_probability
        decision.probability_after = adjusted_probability
        if context.get("partner_role") == "Patient":
            patient_style = persona.get("patient_facing_style", "clear explanation") if isinstance(persona, dict) else "clear explanation"
            decision.interaction_type = f"{decision.interaction_type}_styled"
            decision.memory_summary = (
                f"{context.get('agent_name')} used a {style} style with the patient about "
                f"{decision.topic}; the exchange reflected {patient_style}."
            )
        else:
            decision.memory_summary = (
                f"{context.get('agent_name')} used a {style} style during {decision.topic} "
                f"with {context.get('partner_name')} in {context.get('zone_id')}."
            )
        decision.reason = (
            "structured fallback combined persona style, eligible workflow context, "
            "and retrieved memories without changing movement"
        )
        decision.importance = min(1.0, decision.importance + (0.08 if memories else 0.03))
        decision.used_fallback = True
        decision.backend_name = "structured_generative_fallback"
        decision.persona_id = persona_id
        decision.prompt_context_hash = prompt_context_hash
        decision.context_summary = json.dumps(context_summary, sort_keys=True)
        decision.parsed_response = {
            "backend": "structured_generative_fallback",
            "context_summary": context_summary,
            "persona_id": persona_id,
            "topic": decision.topic,
            "duration_seconds": decision.duration_seconds,
            "importance": decision.importance,
            "interact_probability": adjusted_probability,
            "probability_adjustment_reasons": adjustment_reasons,
            "bounded_decision": True,
        }
        return decision


class InteractionEngine:
    """Facade used by Simulation.log_interaction to enrich and remember events."""

    def __init__(
        self,
        backend_name: str,
        rng: random.Random,
        model_variant: str = config.MODEL_VARIANT,
        memory_enabled: bool = config.MEMORY_ENABLED,
    ) -> None:
        self.rng = rng
        self.model_variant = model_variant
        self.memory_enabled = memory_enabled
        self.behavior_library = EmpiricalBehaviorLibrary(config.EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH)
        self.memory_streams: Dict[int, MemoryStream] = {}
        self.decision_log: List[Dict[str, object]] = []
        self.stats = {
            "decision_calls": 0,
            "llm_calls": 0,
            "llm_cache_hits": 0,
            "llm_fallbacks": 0,
            "parse_successes": 0,
            "rule_stub_calls": 0,
            "backend_calls": 0,
            "schema_parse_failures": 0,
            "memory_retrieval_calls": 0,
            "memories_retrieved": 0,
            "memory_effects_applied": 0,
            "accepted_because_of_memory": 0,
            "rejected_because_of_memory": 0,
            "decision_flipped_by_memory": 0,
            "duration_changed_by_memory": 0,
            "topic_changed_by_memory": 0,
            "structured_fallback_calls": 0,
        }
        self.backend = self._build_backend(backend_name)

    def _build_backend(self, backend_name: str) -> InteractionBackend:
        if backend_name == "rule_stub":
            if self.model_variant == "memory_rule":
                return MemoryRuleBackend()
            if self.model_variant == "generative_interaction":
                return StructuredGenerativeFallbackBackend()
            return RuleStubBackend()
        if backend_name == "structured_generative_fallback":
            return StructuredGenerativeFallbackBackend()
        if backend_name == "memory_rule":
            return MemoryRuleBackend()
        if backend_name in {"vllm_offline", "local_llm"}:
            raise ValueError(
                "Live in-simulation vLLM inference is disabled. Export evidence packets and run "
                "scripts/run_part3_vllm_offline.py so Part 3 uses the frozen cognitive-persona protocol."
            )
        raise ValueError(f"Unsupported interaction backend: {backend_name}")

    def stream_for(self, agent_id: int) -> MemoryStream:
        if agent_id not in self.memory_streams:
            self.memory_streams[agent_id] = MemoryStream()
        return self.memory_streams[agent_id]

    def grounded_part3_memory_context(
        self,
        *,
        agent_id: int,
        current_timestamp: int,
        partner_id: Optional[int],
        patient_context_id: Optional[int],
        reason_type: str,
        interaction_type: str,
        zone_id: str,
        salience_modifiers: Mapping[str, float],
        limit: int = 5,
    ) -> List[Dict[str, object]]:
        """Return bounded, realized experience records for Part 3 only."""

        return self.stream_for(agent_id).retrieve_grounded_part3_context(
            current_timestamp=current_timestamp,
            partner_id=partner_id,
            patient_context_id=patient_context_id,
            reason_type=reason_type,
            interaction_type=interaction_type,
            zone_id=zone_id,
            limit=limit,
            salience_modifiers=salience_modifiers,
        )

    def decide(
        self,
        agent_1,
        agent_2,
        simulation,
        position,
        interaction_type: str,
        task_name: Optional[str],
        ) -> InteractionDecision:
        memory_stream = self.stream_for(agent_1.gid)
        if self.memory_enabled:
            self.stats["memory_retrieval_calls"] += 1
        memories = (
            memory_stream.retrieve(
                current_timestamp=simulation.timestep,
                query_terms=[interaction_type, task_name or "", agent_2.role],
                limit=config.MEMORY_RETRIEVAL_LIMIT,
            )
            if self.memory_enabled
            else []
        )
        self.stats["memories_retrieved"] += len(memories)
        context = build_interaction_context(
            agent=agent_1,
            partner=agent_2,
            simulation=simulation,
            interaction_type=interaction_type,
            task_name=task_name,
            position=position,
            memories=memories,
            library=self.behavior_library,
        )
        self.stats["decision_calls"] += 1
        self.stats["backend_calls"] += 1
        if isinstance(self.backend, RuleStubBackend):
            self.stats["rule_stub_calls"] += 1
        if isinstance(self.backend, StructuredGenerativeFallbackBackend):
            self.stats["structured_fallback_calls"] += 1
        decision = self.backend.decide_interaction(context, self.rng)
        if decision.memory_effect_applied:
            self.stats["memory_effects_applied"] += 1
        if decision.accepted_because_of_memory:
            self.stats["accepted_because_of_memory"] += 1
        if decision.rejected_because_of_memory:
            self.stats["rejected_because_of_memory"] += 1
        if decision.decision_flipped_by_memory:
            self.stats["decision_flipped_by_memory"] += 1
        if decision.duration_before is not None and decision.duration_after is not None and decision.duration_before != decision.duration_after:
            self.stats["duration_changed_by_memory"] += 1
        if decision.topic_before is not None and decision.topic_after is not None and decision.topic_before != decision.topic_after:
            self.stats["topic_changed_by_memory"] += 1
        if decision.used_fallback:
            self.stats["llm_fallbacks"] += 1
        if decision.parse_error:
            self.stats["schema_parse_failures"] += 1

        self.decision_log.append(
            {
                "timestamp": simulation.timestep,
                "agent_id": agent_1.gid,
                "agent_role": agent_1.role,
                "partner_id": agent_2.gid,
                "partner_role": agent_2.role,
                "zone_id": context["zone_id"],
                "interaction_type": interaction_type,
                "task_name": task_name,
                "used_fallback": decision.used_fallback,
                "persona_source": context["persona_source"],
                "memory_count": len(memories),
                "retrieved_memory_ids": [memory.memory_id for memory in memories],
                "model_variant": self.model_variant,
                "raw_response": decision.raw_response,
                "parsed_response": decision.parsed_response,
                "parse_error": decision.parse_error,
                "backend_name": decision.backend_name,
                "memory_effect_applied": decision.memory_effect_applied,
                "memory_effect_type": decision.memory_effect_type,
                "probability_before": decision.probability_before,
                "probability_after": decision.probability_after,
                "duration_before": decision.duration_before,
                "duration_after": decision.duration_after,
                "accepted_because_of_memory": decision.accepted_because_of_memory,
                "rejected_because_of_memory": decision.rejected_because_of_memory,
                "decision_flipped_by_memory": decision.decision_flipped_by_memory,
                "topic_before": decision.topic_before,
                "topic_after": decision.topic_after,
                "prompt_context_hash": decision.prompt_context_hash,
                "persona_id": decision.persona_id,
            }
        )
        if decision.retrieved_memory_ids is None:
            decision.retrieved_memory_ids = [memory.memory_id for memory in memories]
        if decision.retrieved_memory_summaries is None:
            decision.retrieved_memory_summaries = [memory.summary for memory in memories]
        return decision

    def remember(
        self,
        decision: InteractionDecision,
        agent_1,
        agent_2,
        simulation,
        zone_id: str,
        source_event_id: Optional[str] = None,
        reason_type: Optional[str] = None,
        patient_context_id: Optional[int] = None,
        force_part3_grounded: bool = False,
    ) -> None:
        if not self.memory_enabled and not force_part3_grounded:
            return
        duration_seconds = int(max(decision.duration_seconds, 1))
        importance = max(0.0, min(1.0, float(decision.importance)))
        patient = agent_1 if getattr(agent_1, "role", None) == "Patient" else agent_2 if getattr(agent_2, "role", None) == "Patient" else None
        if patient is None and patient_context_id is not None:
            patient = simulation.patient_by_id.get(int(patient_context_id))
        tags = []
        if getattr(patient, "esi_level", None) in {1, 2}:
            tags.append("high_acuity")
        if "handoff" in decision.interaction_type:
            tags.append("handoff")
        if getattr(patient, "role", None) == "Patient":
            tags.append("patient_facing")
        for owner, partner, owner_was_initiator in (
            (agent_1, agent_2, True),
            (agent_2, agent_1, False),
        ):
            if owner.role == "Patient":
                continue
            stream = self.stream_for(owner.gid)
            memory_id = f"mem_{simulation.random_seed}_{simulation.condition_spec.name}_{owner.gid}_{len(stream.events) + 1}"
            stream.add(
                MemoryEvent(
                    memory_id=memory_id,
                    timestamp=simulation.timestep,
                    event_type="communicative_interaction",
                    owner_agent_id=owner.gid,
                    owner_agent_name=getattr(owner, "name", f"{owner.role} {owner.gid}"),
                    owner_role=owner.role,
                    partner_id=getattr(partner, "gid", None),
                    partner_name=getattr(partner, "name", f"{partner.role} {partner.gid}"),
                    partner_role=partner.role,
                    patient_id=getattr(patient, "gid", None),
                    esi_level=getattr(patient, "esi_level", None),
                    zone_id=zone_id,
                    condition_name=simulation.condition_spec.name,
                    model_variant=self.model_variant,
                    topic=decision.topic,
                    duration_seconds=duration_seconds,
                    interaction_type=decision.interaction_type,
                    summary=decision.memory_summary,
                    importance=importance,
                    emotional_tags=tags,
                    retrieval_keywords=[
                        owner.role,
                        partner.role,
                        decision.topic,
                        decision.interaction_type,
                        zone_id,
                        str(getattr(patient, "complaint_type", "")),
                    ],
                    source_event_id=source_event_id,
                    support_ids=[source_event_id] if source_event_id else [],
                    owner_was_initiator=owner_was_initiator,
                    reason_type=reason_type,
                    outcome="realized",
                )
            )
            owner.memory_stream = stream
            if (
                self.memory_enabled
                and len(stream.events) % max(config.LLM_REFLECTION_INTERVAL, 1) == 0
            ):
                reflection = stream.reflect(simulation.timestep, owner.role)
                if reflection is not None:
                    reflection.owner_agent_id = owner.gid
                    reflection.owner_agent_name = getattr(owner, "name", f"{owner.role} {owner.gid}")
                    reflection.condition_name = simulation.condition_spec.name
                    reflection.model_variant = self.model_variant
                    stream.add(reflection)
                    owner.reflection_summaries.append(asdict(reflection))

    def remember_missed_opportunity(self, event: Mapping[str, object], owner) -> None:
        if not self.memory_enabled or getattr(owner, "role", None) == "Patient":
            return
        stream = self.stream_for(owner.gid)
        partner_role = str(event.get("potential_partner_role") or "unknown_partner")
        reason = str(event.get("reason") or "missed_opportunity")
        zone_id = str(event.get("zone_id") or "unknown_zone")
        memory_id = f"missmem_{event.get('simulation_seed')}_{event.get('condition_name')}_{owner.gid}_{len(stream.events) + 1}"
        stream.add(
            MemoryEvent(
                memory_id=memory_id,
                timestamp=int(event.get("timestep", 0)),
                event_type="missed_opportunity",
                owner_agent_id=owner.gid,
                owner_agent_name=getattr(owner, "name", f"{owner.role} {owner.gid}"),
                owner_role=owner.role,
                partner_id=event.get("potential_partner_id"),
                partner_name=event.get("potential_partner_name"),
                partner_role=partner_role,
                patient_id=event.get("patient_id"),
                esi_level=event.get("esi_level"),
                zone_id=zone_id,
                condition_name=str(event.get("condition_name") or "unknown_condition"),
                model_variant=self.model_variant,
                topic="missed_coordination",
                duration_seconds=int(event.get("episode_duration_seconds", 0)),
                interaction_type="missed_opportunity",
                summary=(
                    f"Missed opportunity in {zone_id}: {owner.role} could not complete "
                    f"a {reason} exchange with {partner_role}."
                ),
                importance=0.62,
                emotional_tags=["missed_opportunity", reason],
                retrieval_keywords=[owner.role, partner_role, reason, zone_id, "missed"],
                source_event_id=str(event.get("event_id")),
                support_ids=[str(event.get("event_id"))],
                owner_was_initiator=True,
                reason_type=reason,
                outcome="missed",
            )
        )
        owner.memory_stream = stream
