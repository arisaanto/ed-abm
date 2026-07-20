"""Trace-grounded synthetic interview smoke tests for named staff agents."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable, Mapping, Optional

import config
from src.personas import PERSONA_APPRAISAL_DIMENSIONS


INTERVIEW_QUESTIONS = [
    {
        "question_id": "overall_experience",
        "question": (
            "How would you describe your overall experience of working in this layout? "
            "Describe recurring patterns rather than listing your itinerary."
        ),
    },
    {
        "question_id": "critical_moments",
        "question": "Which evidence-backed moments most shaped that experience, and why did they matter?",
    },
    {
        "question_id": "coordination_and_focus",
        "question": "How did the layout affect the balance between coordinating with others and protecting task focus?",
    },
    {
        "question_id": "supportive_feature",
        "question": "Which spatial feature most supported your work, based on the supplied evidence?",
    },
    {
        "question_id": "difficult_feature",
        "question": "Which spatial feature created the greatest difficulty, based on the supplied evidence?",
    },
    {
        "question_id": "counterfactual_change",
        "question": "What one spatial change would best support someone with your priorities, and what tradeoff might it create?",
    },
]
INTERVIEW_PROTOCOL = [item["question"] for item in INTERVIEW_QUESTIONS]

APPRAISAL_CLAIM_LAYERS = {
    "grounded_pattern": (
        "A concise statement supported directly by cited simulated events or "
        "place summaries. It must not contain invented experience."
    ),
    "persona_conditioned_interpretation": (
        "A synthetic interpretation of that pattern through the preregistered "
        "workplace priorities. It is not a human psychological measurement."
    ),
    "latent_need": (
        "An optional, explicitly conjectural user need inferred from the grounded "
        "pattern and orientation. It is a design hypothesis, not observed testimony."
    ),
    "design_hypothesis": (
        "An optional spatial or operational proposal. It must include a plausible "
        "tradeoff and must never be reported as an observed outcome."
    ),
}

APPRAISAL_LANGUAGE_RULES = [
    "Write in natural first-person workplace language, not as a data analyst.",
    "Answer directly; do not repeat or paraphrase the question before answering it.",
    (
        "Do not announce the study context with phrases such as 'under high load', "
        "'during normal load', 'in the baseline', or 'in this condition'. Let the "
        "experience itself carry the context."
    ),
    (
        "Begin with a concrete observation, tension, or consequence a colleague "
        "might naturally remember, not with a scenario or layout label."
    ),
    "Describe one or two representative moments; do not recite an itinerary.",
    "Do not mention logs, metrics, evidence IDs, the ABM, prompts, or persona titles in the answer.",
    "Do not invent diagnoses, outcomes, emotions, motives, speech, or events.",
    "Do not claim that a spatial feature caused speed, efficiency, safety, care quality, or staff wellbeing; those outcomes were not modeled.",
    "A listed feature is context, not proof of effect; connect any interpretation to cited recurring patterns or moments.",
    "General workplace knowledge may shape an interpretation or design hypothesis, never an event claim.",
    "Use insufficient evidence when the supplied shift cannot support a credible answer.",
]

CANONICAL_TOPIC_FAMILIES = [
    "patient_status_update",
    "bed_and_capacity_management",
    "administrative_and_logistics",
    "treatment_and_orders",
    "patient_flow_and_transfer",
    "diagnostic_findings",
    "next_steps_and_planning",
    "patient_facing_care",
    "informal_or_relational",
]

TOPIC_FAMILIES_BY_ABM_REASON = {
    "POST_TASK_UPDATE": [
        "patient_status_update",
        "treatment_and_orders",
        "diagnostic_findings",
        "next_steps_and_planning",
    ],
    "SAME_PATIENT_UPDATE": [
        "patient_status_update",
        "treatment_and_orders",
        "diagnostic_findings",
        "next_steps_and_planning",
    ],
    "HANDOFF_NEED": [
        "patient_status_update",
        "treatment_and_orders",
        "diagnostic_findings",
        "patient_flow_and_transfer",
        "next_steps_and_planning",
    ],
    "BED_FLOW_COORDINATION": [
        "bed_and_capacity_management",
        "patient_flow_and_transfer",
        "administrative_and_logistics",
        "next_steps_and_planning",
    ],
    "HIGH_ACUITY_ESCALATION": [
        "patient_status_update",
        "treatment_and_orders",
        "diagnostic_findings",
        "next_steps_and_planning",
    ],
    "SAME_STATION_BRIEF_CHECKIN": [
        "informal_or_relational",
        "patient_status_update",
        "administrative_and_logistics",
    ],
    "CORRIDOR_PASSING_UPDATE": [
        "patient_status_update",
        "administrative_and_logistics",
        "next_steps_and_planning",
    ],
}


def categorical_decision_bounds(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared reason/topic bounds for one scientific decision."""

    decision_context = evidence["decision_context"]
    workflow = evidence["workflow_context"]
    active_task_present = bool(workflow.get("initiator_is_task_busy", False))
    patient_id = workflow.get("patient_id")
    patient_context_present = patient_id not in (None, -1, "-1")

    original_reasons = list(decision_context["allowed_reasons"])
    allowed_reasons = [
        reason
        for reason in original_reasons
        if active_task_present or reason != "protect_task_continuity"
    ]
    if not allowed_reasons:
        raise ValueError(
            f"No context-valid reasons for {evidence.get('evidence_id')}"
        )

    logged_topics = [
        str(topic)
        for topic in decision_context.get("allowed_topic_families", [])
        if str(topic) in CANONICAL_TOPIC_FAMILIES
    ]
    abm_reason = str(decision_context.get("abm_reason_type", ""))
    reason_topics = list(TOPIC_FAMILIES_BY_ABM_REASON.get(abm_reason, []))
    allowed_topics = [
        topic
        for topic in CANONICAL_TOPIC_FAMILIES
        if topic in set(logged_topics) | set(reason_topics)
    ]
    patient_specific_topics = {
        "patient_status_update",
        "treatment_and_orders",
        "diagnostic_findings",
        "patient_facing_care",
    }
    allowed_topics = [
        topic
        for topic in allowed_topics
        if patient_context_present or topic not in patient_specific_topics
    ]
    if not allowed_topics:
        raise ValueError(
            f"No context-valid topic families for {evidence.get('evidence_id')} "
            f"(ABM reason {abm_reason!r})"
        )

    return {
        "allowed_reasons": allowed_reasons,
        "allowed_topic_families": allowed_topics,
        "reason_constraint_basis": {
            "original_allowed_reasons": original_reasons,
            "final_allowed_reasons": allowed_reasons,
            "active_task_present": active_task_present,
            "policy": "remove task-continuity reason when no active task is supplied",
        },
        "topic_constraint_basis": {
            "logged_topic_families": logged_topics,
            "reason_grounded_topic_families": reason_topics,
            "final_allowed_topic_families": allowed_topics,
            "patient_context_present": patient_context_present,
            "policy": (
                "union of logged and preregistered reason-grounded families, then "
                "remove patient-specific families when no patient context is supplied"
            ),
        },
    }

@dataclass
class SurveyResponse:
    """Part 3 trace-grounded synthetic survey response schema."""

    dimension: str
    rateability: str
    score_1_to_7: int | None
    confidence_1_to_5: int
    evidence_event_ids: list[str]
    short_rationale: str
    uncertainty_note: str
    user_model_id: str
    role: str
    condition: str
    scenario_mode: str
    not_human_data: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class InterviewResponse:
    """Part 3 trace-grounded synthetic interview response schema."""

    question_id: str
    answer: str
    evidence_event_ids: list[str]
    role_perspective: str
    uncertainty_note: str
    user_model_id: str
    role: str
    condition: str
    scenario_mode: str
    not_human_data: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SurveyDimensionResponse:
    """One dimension inside a fixed end-of-shift synthetic appraisal."""

    dimension: str
    rateability: str
    score_1_to_7: int | None
    confidence_1_to_5: int
    evidence_event_ids: list[str]
    short_rationale: str
    uncertainty_note: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SurveyBundleResponse:
    """All preregistered survey dimensions for one simulated agent-shift."""

    responses: list[SurveyDimensionResponse]
    user_model_id: str
    role: str
    condition: str
    scenario_mode: str
    source_is_simulated_experience: bool = True
    not_human_data: bool = True

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["responses"] = [response.as_dict() for response in self.responses]
        return payload


@dataclass
class InterviewAnswer:
    """One bounded-generative answer with explicit epistemic layers."""

    question_id: str
    answer: str
    grounded_pattern: str
    persona_conditioned_interpretation: str
    latent_need: str | None
    design_hypothesis: str | None
    tradeoff: str | None
    evidence_event_ids: list[str]
    uncertainty_note: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class InterviewBundleResponse:
    """Natural-language appraisal for one simulated shift, not human testimony."""

    answers: list[InterviewAnswer]
    role_perspective: str
    user_model_id: str
    role: str
    condition: str
    scenario_mode: str
    claim_layer_contract: str = "grounded_pattern_interpretation_conjecture_v1"
    source_is_simulated_experience: bool = True
    not_human_data: bool = True

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["answers"] = [answer.as_dict() for answer in self.answers]
        return payload


@dataclass
class CriticalIncidentReport:
    incident_type: str
    why_it_mattered_operationally: str
    spatial_factor_if_any: str
    coordination_factor_if_any: str
    supporting_event_ids: list[str]
    uncertainty_note: str

    def as_dict(self) -> dict:
        return asdict(self)


def required_output_schema(
    packet_type: str,
    *,
    decision_output_contract: str | None = None,
) -> dict[str, Any]:
    """Return the compact structured-output schema expected for a Part 3 packet."""

    schemas = {
        "end_of_shift_survey": {
            "dimension": f"one of {PERSONA_APPRAISAL_DIMENSIONS}",
            "rateability": "rateable or insufficient_evidence",
            "score_1_to_7": (
                "integer 1..7 when rateable; null when evidence is insufficient"
            ),
            "confidence_1_to_5": "integer 1..5",
            "evidence_event_ids": "list of cited trace ids",
            "short_rationale": "brief trace-grounded rationale",
            "uncertainty_note": "what the trace cannot support",
            "not_human_data": True,
        },
        "end_of_shift_interview": {
            "question_id": "string",
            "answer": "trace-grounded synthetic answer",
            "evidence_event_ids": "list of cited trace ids",
            "role_perspective": "role/user model viewpoint",
            "uncertainty_note": "what is uncertain or unsupported",
            "not_human_data": True,
        },
        "end_of_shift_survey_bundle": {
            "responses": (
                "one response for every preregistered appraisal dimension; each "
                "contains rateability, score, confidence, evidence, rationale, and uncertainty"
            ),
            "not_human_data": True,
        },
        "end_of_shift_interview_bundle": {
            "answers": (
                "one answer per preregistered question with separate grounded pattern, "
                "persona-conditioned interpretation, optional latent need and design hypothesis, "
                "tradeoff, cited evidence, and uncertainty"
            ),
            "role_perspective": "brief role context without a persona title",
            "claim_layer_contract": "grounded_pattern_interpretation_conjecture_v1",
            "not_human_data": True,
        },
        "critical_incident": {
            "incident_type": "string",
            "why_it_mattered_operationally": "string",
            "spatial_factor_if_any": "string",
            "coordination_factor_if_any": "string",
            "supporting_event_ids": "list of cited trace ids",
            "uncertainty_note": "string",
        },
        "in_simulation_decision": {
            "decision_id": "string",
            "selected_action": "must be one of feasible_actions",
            "selected_reason": "must be one of allowed_reasons",
            "topic_family": "must be one of allowed_topic_families",
            "topic_text": "brief evidence-grounded subtopic within topic_family",
            "rationale_short": "brief trace-grounded rationale",
            "evidence_ids": "list of cited evidence ids",
            "confidence": "0..1",
        },
    }
    if (
        packet_type == "in_simulation_decision"
        and decision_output_contract == "categorical_causal_v1"
    ):
        return {
            "decision_id": "string",
            "selected_action": "must be one of feasible_actions",
            "selected_reason": "must be one of allowed_reasons",
            "topic_family": "must be one of allowed_topic_families",
            "evidence_ids": "list of cited evidence ids",
        }
    return schemas.get(packet_type, {"answer": "structured JSON", "not_human_data": True})


def _top_memories(stream, limit: int = 5) -> list:
    memories = [
        event for event in stream.events
        if event.event_type in {"communicative_interaction", "reflection", "missed_opportunity"}
    ]
    memories.sort(key=lambda event: (event.importance, event.timestamp), reverse=True)
    return memories[:limit]


def _answer_from_trace(agent, simulation, question: str, memories: list) -> tuple[str, str, list[str], bool]:
    persona = getattr(agent, "persona", None)
    role = agent.role
    tolerance = getattr(persona, "interruption_tolerance", 0.5)
    concern = getattr(persona, "priorities", ["safe care"])[0] if getattr(persona, "priorities", None) else "safe care"
    top_memory = memories[0] if memories else None
    top_memory_text = top_memory.summary if top_memory is not None else "no single strong incident"
    missed_for_agent = [
        event for event in simulation.missed_opportunity_log
        if event.get("agent_id") == agent.gid
    ]
    top_missed = missed_for_agent[0] if missed_for_agent else None
    top_topics = Counter(event.topic for event in memories if event.topic)
    top_zones = Counter(event.zone_id for event in memories if event.zone_id)
    topic = top_topics.most_common(1)[0][0] if top_topics else "coordination"
    zone = top_zones.most_common(1)[0][0] if top_zones else "the care area"
    high_acuity = [event for event in memories if event.esi_level in {1, 2}]
    missed_count = len(missed_for_agent)
    interruption_count = len([
        event for event in simulation.interruption_log
        if event.get("agent_1_id") == agent.gid or event.get("agent_2_id") == agent.gid
    ])
    unsupported = []

    if "visibility" in question.lower():
        if top_memory is None:
            answer = "I do not have one clear moment that supports a visibility claim."
        else:
            answer = (
                f"The clearest moment for me was in {zone}: "
                f"{top_memory_text} It affected timing of contact, not where I was allowed to move."
            )
    elif "movement" in question.lower() or "effort" in question.lower():
        distance = simulation.movement_distance_by_agent.get(agent.gid, 0.0)
        answer = (
            f"I covered about {distance:.0f} meters. What mattered was whether that "
            f"movement supported {concern}, not the distance by itself."
        )
    elif "coordination" in question.lower():
        if top_memory is None:
            answer = "I do not have one clear coordination moment to point to."
        else:
            answer = (
                f"The moment I would point to involved {topic} around {zone}: {top_memory_text}"
            )
    elif "interrupt" in question.lower():
        answer = (
            f"I had {interruption_count} interruptions and {missed_count} moments when "
            f"contact did not happen. Given my relatively "
            f"{'low' if tolerance < 0.45 else 'high' if tolerance > 0.7 else 'moderate'} interruption tolerance, "
            "I would treat that as a workload signal rather than a simple benefit."
        )
    elif "urgent" in question.lower() or "routine" in question.lower():
        if high_acuity:
            answer = (
                f"The moment that pulled most of my attention was: {high_acuity[0].summary}. "
                "Visibility helped most when it supported rapid clarification without making anyone invent a new route."
            )
        else:
            answer = (
                "I do not have a strong ESI 1 or 2 incident to draw from. "
                "I cannot make a supported claim about urgent-patient benefit from this run."
            )
            unsupported.append("urgent_patient_claim_not_supported")
    elif "privacy" in question.lower() or "exposure" in question.lower():
        answer = (
            f"The relevant moment involved {topic} around {zone}. "
            "I would be cautious about privacy or exposure claims unless a longer run shows repeated bedside or station incidents."
        )
    elif "prefer" in question.lower():
        answer = (
            "I cannot compare conditions from a single-condition smoke run. "
            f"For this run, I would only say that the layout should protect {concern} while making availability legible."
        )
        unsupported.append("single_condition_no_preference_comparison")
    elif "specific" in question.lower() or "incident" in question.lower():
        if memories:
            answer = f"The clearest incident I can cite was: {memories[0].summary}"
        else:
            answer = "This smoke run did not leave enough memories for a specific incident."
    else:
        answer = (
            f"From my {role} perspective, the clearest pattern involved {topic} around "
            f"{zone}: {top_memory_text}."
        )
    if not memories:
        unsupported.append("no_supporting_memory")
    if "specific" in question.lower() and top_memory is None:
        unsupported.append("specific_incident_unavailable")
    if top_missed is not None and "missed" not in answer.lower() and "interrupt" in question.lower():
        unsupported.append("missed_opportunity_not_described")
    prompt_leakage = "Answer as" in answer
    if prompt_leakage:
        unsupported.append("prompt_leakage_detected")
    return answer, "trace_supported" if memories else "low_support", unsupported, prompt_leakage


def interview_agent(agent, simulation, questions: Optional[Iterable[str]] = None) -> dict:
    stream = simulation.interaction_engine.stream_for(agent.gid)
    memories = _top_memories(stream)
    memory_ids = [event.memory_id for event in memories]
    event_ids = sorted({
        support_id for event in memories for support_id in event.support_ids if support_id
    })
    answers = []
    for question in (questions or INTERVIEW_PROTOCOL):
        answer, support_level, unsupported, prompt_leakage = _answer_from_trace(agent, simulation, question, memories)
        answers.append(
            {
                "question": question,
                "answer": answer,
                "answer_mode": "rule_trace_narrative",
                "cited_memory_ids": memory_ids[:5],
                "cited_event_ids": event_ids[:5],
                "confidence": support_level,
                "support_level": support_level,
                "unsupported_claim_flags": unsupported,
                "prompt_leakage_detected": prompt_leakage,
                "condition": simulation.condition_spec.name,
                "agent_name": getattr(agent, "name", f"{agent.role} {agent.gid}"),
                "agent_role": agent.role,
                "model_backend": simulation.interaction_backend_name,
                "raw_response": None,
                "fallback": True,
                "support_check": "answers are generated from local trace summaries, not human testimony",
            }
        )
    persona_payload = agent.persona.as_dict() if hasattr(agent.persona, "as_dict") else None
    if isinstance(persona_payload, dict):
        persona_payload.pop("interview_voice_guidelines", None)
    return {
        "agent_id": agent.gid,
        "agent_name": getattr(agent, "name", f"{agent.role} {agent.gid}"),
        "agent_role": agent.role,
        "condition": simulation.condition_spec.name,
        "seed": simulation.random_seed,
        "model_variant": simulation.model_variant,
        "included_in_behavioral_ablation": False,
        "simulation_metrics_should_match_generative_interaction": True,
        "interview_outputs_generated": True,
        "backend": simulation.interaction_backend_name,
        "persona": persona_payload,
        "answers": answers,
    }


def run_post_condition_interviews(simulation) -> dict:
    config.INTERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {}
    for agent in simulation.staff_agents:
        interview = interview_agent(agent, simulation)
        safe_name = interview["agent_name"].replace(" ", "_")
        payload[str(agent.gid)] = interview
        if config.SAVE_INTERVIEW_OUTPUTS:
            path = config.INTERVIEW_DIR / f"{safe_name}_{simulation.condition_spec.name}_{simulation.random_seed}.json"
            path.write_text(json.dumps(interview, indent=2))
    return payload
