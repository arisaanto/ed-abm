"""Interaction backends, empirical personas, and lightweight memory streams.

The simulation keeps movement and task order rule-based. This module only
decides whether an eligible encounter becomes an interaction, what gets talked
about, and the remembered summary of that encounter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List, Mapping, Optional

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
    raw_response: Optional[str] = None
    parsed_response: Optional[dict] = None
    backend_name: str = "rule_stub"
    persona_id: Optional[str] = None


class MemoryStream:
    """Ordered store for grounded Part 3 interaction memories."""

    def __init__(self) -> None:
        self.events: List[MemoryEvent] = []
        self.retrieval_traces: List[dict] = []

    def add(self, event: MemoryEvent) -> None:
        self.events.append(event)

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

    def persona_paragraph(self, role: str) -> str:
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
    library: EmpiricalBehaviorLibrary,
) -> Dict[str, object]:
    zone_id = simulation.which_zone(*position)
    persona_source = getattr(simulation, "persona_source", config.PERSONA_SOURCE)
    patient = partner if getattr(partner, "role", None) == "Patient" else None
    if patient is None and task_name is not None:
        patient = simulation.patient_by_id.get(getattr(agent, "target_patient_id", None))
    persona = getattr(agent, "persona", None)
    partner_persona = getattr(partner, "persona", None)
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
            or library.persona_paragraph(agent.role)
        ),
        "persona": persona.as_dict() if hasattr(persona, "as_dict") else None,
        "partner_persona": partner_persona.as_dict() if hasattr(partner_persona, "as_dict") else None,
        "persona_source": persona_source,
        "topic_candidates": list(topic_weights),
        "topic_weights": topic_weights,
        "duration_prior": library.duration_prior(agent.role, partner.role),
        "distance_to_partner": getattr(agent, "distance_to_agent", lambda _: 0.0)(partner),
        "condition_name": simulation.condition_spec.name,
        "model_variant": getattr(
            simulation,
            "model_variant",
            config.BASELINE_MODEL_ID,
        ),
        "patient_id": getattr(patient, "gid", None),
        "esi_level": getattr(patient, "esi_level", None),
        "urgency": getattr(patient, "urgency", None),
        "complaint_type": getattr(patient, "complaint_type", None),
        "workload": len(simulation.active_patients),
    }


class InteractionBackend(ABC):
    """Backend interface for deterministic or LLM-driven interaction decisions."""

    @abstractmethod
    def decide_interaction(self, context: Mapping[str, object], rng: random.Random) -> InteractionDecision:
        raise NotImplementedError


class RuleStubBackend(InteractionBackend):
    """Deterministic offline backend for debugging and calibration iteration."""

    def decide_interaction(self, context: Mapping[str, object], rng: random.Random) -> InteractionDecision:
        return _rule_decision(context, rng)


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
    if interaction_type == "doctor_doctor_coreview":
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


def _rule_decision(
    context: Mapping[str, object],
    rng: random.Random,
) -> InteractionDecision:
    topic_weights = dict(context.get("topic_weights", {}))
    topic = _choose_weighted_topic(topic_weights, rng)
    interaction_type = str(context.get("interaction_type", "workflow_task"))
    duration_low, duration_high = _base_duration_bounds(context)
    esi_level = context.get("esi_level")
    duration_seconds = rng.randint(int(duration_low), int(duration_high))
    zone_id = context.get("zone_id", "unknown_zone")
    partner_role = context.get("partner_role", "colleague")
    memory_summary = (
        f"{context.get('agent_name', context.get('role', 'Staff'))} discussed "
        f"{topic} with {partner_role} in {zone_id} during {interaction_type}."
    )
    if esi_level in {1, 2}:
        memory_summary += (
            f" Patient acuity was ESI {esi_level}, making the exchange more salient."
        )

    interact_probability = 1.0
    if interaction_type.startswith("opportunistic_"):
        interact_probability = 0.72
        if esi_level in {1, 2}:
            interact_probability = min(0.99, interact_probability + 0.08)
    interact = (
        True
        if not interaction_type.startswith("opportunistic_")
        else rng.random() < interact_probability
    )
    return InteractionDecision(
        interact=interact,
        partner_id=int(context["partner_id"]),
        interaction_type=interaction_type,
        topic=topic,
        duration_seconds=duration_seconds,
        memory_summary=memory_summary,
        reason="rule_fallback_eligible_encounter",
        importance=(
            0.85
            if esi_level in {1, 2}
            else min(1.0, duration_seconds / 90.0)
        ),
        parsed_response={
            "interact_probability": interact_probability,
            "bounded_decision": True,
        },
        backend_name="rule_stub",
    )


class InteractionEngine:
    """Facade used by Simulation.log_interaction to enrich and remember events."""

    def __init__(
        self,
        rng: random.Random,
        model_variant: str,
    ) -> None:
        self.rng = rng
        self.model_variant = model_variant
        self.behavior_library = EmpiricalBehaviorLibrary(config.EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH)
        self.memory_streams: Dict[int, MemoryStream] = {}
        self.decision_log: List[Dict[str, object]] = []
        self.stats = {
            "decision_calls": 0,
            "rule_stub_calls": 0,
            "backend_calls": 0,
        }
        self.backend = RuleStubBackend()

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
        context = build_interaction_context(
            agent=agent_1,
            partner=agent_2,
            simulation=simulation,
            interaction_type=interaction_type,
            task_name=task_name,
            position=position,
            library=self.behavior_library,
        )
        self.stats["decision_calls"] += 1
        self.stats["backend_calls"] += 1
        if isinstance(self.backend, RuleStubBackend):
            self.stats["rule_stub_calls"] += 1
        decision = self.backend.decide_interaction(context, self.rng)

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
                "used_fallback": False,
                "persona_source": context["persona_source"],
                "memory_count": 0,
                "retrieved_memory_ids": [],
                "model_variant": self.model_variant,
                "raw_response": decision.raw_response,
                "parsed_response": decision.parsed_response,
                "parse_error": None,
                "backend_name": decision.backend_name,
                "memory_effect_applied": False,
                "memory_effect_type": "",
                "probability_before": decision.parsed_response.get(
                    "interact_probability"
                )
                if decision.parsed_response
                else None,
                "probability_after": decision.parsed_response.get(
                    "interact_probability"
                )
                if decision.parsed_response
                else None,
                "duration_before": decision.duration_seconds,
                "duration_after": decision.duration_seconds,
                "accepted_because_of_memory": False,
                "rejected_because_of_memory": False,
                "decision_flipped_by_memory": False,
                "topic_before": decision.topic,
                "topic_after": decision.topic,
                "prompt_context_hash": None,
                "persona_id": decision.persona_id,
            }
        )
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
        if not force_part3_grounded:
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
