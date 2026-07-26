"""Trace-grounded synthetic interview smoke tests for named staff agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

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
        "A spatial or operational proposal used only for counterfactual_change. "
        "That question requires both a non-empty design_hypothesis and tradeoff; "
        "both fields must be null for every other question. A proposal must never "
        "be reported as an observed outcome."
    ),
}

APPRAISAL_LANGUAGE_RULES = [
    "Write in natural first-person workplace language, not as a data analyst.",
    "Answer directly; do not repeat or paraphrase the question before answering it.",
    (
        "Keep each public answer to two or three complete sentences and each "
        "analytic field to one complete sentence. Never stop mid-sentence."
    ),
    (
        "Write every prose field in plain English. Cite event IDs only in the "
        "evidence_event_ids array, never inside prose."
    ),
    (
        "Apply the designed orientation silently. Never mention an orientation, "
        "persona, archetype, or profile in any answer or rationale."
    ),
    (
        "Do not announce the study context with phrases such as 'under high load', "
        "'during normal load', 'in the baseline', or 'in this condition'. Let the "
        "experience itself carry the context."
    ),
    (
        "Begin with a concrete observation, tension, or consequence a colleague "
        "might naturally remember, not with a scenario or layout label."
    ),
    (
        "Avoid canned openings such as 'The shift felt like', 'Two moments stood "
        "out', and 'The layout forces a trade-off'; begin with the substance."
    ),
    "Describe one or two representative moments; do not recite an itinerary.",
    "Do not mention logs, metrics, evidence IDs, the ABM, prompts, or persona titles in the answer.",
    "Do not invent diagnoses, outcomes, emotions, motives, speech, or events.",
    "Do not claim that a spatial feature caused speed, efficiency, safety, care quality, or staff wellbeing; those outcomes were not modeled.",
    "A listed feature is context, not proof of effect; connect any interpretation to cited recurring patterns or moments.",
    "General workplace knowledge may shape an interpretation or design hypothesis, never an event claim.",
    (
        "Use design_hypothesis and tradeoff only for counterfactual_change, where "
        "both must be non-empty strings. Return both as null for every other question."
    ),
    "Use insufficient evidence when the supplied shift cannot support a credible answer.",
]

APPRAISAL_PROSE_LIMITS = {
    "short_rationale": 420,
    "answer": 520,
    "grounded_pattern": 360,
    "persona_conditioned_interpretation": 360,
    "latent_need": 300,
    "design_hypothesis": 300,
    "tradeoff": 300,
    "uncertainty_note": 240,
}


_EVENT_ID_IN_PROSE = re.compile(r"\b(?:shift_)?event_\d+\b", re.IGNORECASE)
_PUBLIC_PROCESS_LANGUAGE = re.compile(
    r"\b(?:log|logs|logged|metric|metrics|dataset|data point|evidence id|"
    r"agent-based model|abm|prompt|persona|archetype|profile|simulated agent|"
    r"language model|llm|model output)\b|"
    r"\b(?:my|the|this|designed|cognitive|assigned)\s+"
    r"(?:workplace\s+)?orientation\b",
    re.IGNORECASE,
)
_PUBLIC_CONTEXT_OPENING = re.compile(
    r"^(?:under|during|in)\s+(?:the\s+)?(?:high(?:[- ]load)?|normal(?:[- ]load)?|"
    r"baseline|this (?:scenario|condition)|(?:cockpit|nursta|both) condition)\b",
    re.IGNORECASE,
)
_CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_COMPLETE_SENTENCE_END = re.compile(r"[.!?](?:[\"']|\))?$")


def appraisal_prose_issues(
    text: Any,
    *,
    public: bool = False,
    maximum_length: int | None = None,
) -> list[str]:
    """Return deterministic quality defects in generated appraisal prose."""

    value = str(text or "").strip()
    issues = []
    if not value:
        return ["empty prose"]
    if _CJK_CHARACTER.search(value):
        issues.append("non-English CJK fragment appears inside prose")
    if public:
        complete = bool(_COMPLETE_SENTENCE_END.search(value))
        if not complete:
            issues.append("prose does not end with a complete sentence")
        if maximum_length is not None and len(value) >= maximum_length and not complete:
            issues.append("prose reaches its schema character limit")
        if _EVENT_ID_IN_PROSE.search(value):
            issues.append("event ID appears inside public prose")
        if _PUBLIC_PROCESS_LANGUAGE.search(value):
            issues.append(
                "analysis or persona-construction language appears in public prose"
            )
        if _PUBLIC_CONTEXT_OPENING.search(value):
            issues.append("public prose restates its scenario or condition")
    return issues

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
