"""Scientific contract for the Part 3 empirical-questionnaire audit.

The audit asks Qwen to complete the eight end-of-shift questions used in the
Zurich shadowing study from simulated shift evidence.  It is a bounded scale-
and-face-validity benchmark, not a validation of the synthetic personas as
human categories.
"""

from __future__ import annotations

from typing import Any, Mapping


QUESTIONNAIRE_ITEMS = (
    {
        "question_id": "q1",
        "short_name": "max_workload",
        "question": "What was the maximum workload for you today, compared with other days?",
        "scale": "1-7",
    },
    {
        "question_id": "q2",
        "short_name": "critical_workload_share",
        "question": "What proportion of the entire shift was your workload critically high?",
        "scale": "0, 25, 50, 75, or 100 percent",
    },
    {
        "question_id": "q3",
        "short_name": "concentration",
        "question": "Given the work situation today, how concentrated were you able to work?",
        "scale": "1-7",
    },
    {
        "question_id": "q4",
        "short_name": "communication_satisfaction",
        "question": "How satisfactory was communication throughout the whole emergency team today?",
        "scale": "1-7",
    },
    {
        "question_id": "q5",
        "short_name": "collaboration",
        "question": "To what extent did the whole emergency team manage to work hand in hand today?",
        "scale": "1-7",
    },
    {
        "question_id": "q6",
        "short_name": "operational_helpfulness_to_patients",
        "question": "How much did the emergency department manage to be helpful to patients today?",
        "scale": "1-7",
    },
    {
        "question_id": "q7",
        "short_name": "information_access",
        "question": "To what extent did you have access to the information needed to do a good job today?",
        "scale": "1-7",
    },
    {
        "question_id": "q8",
        "short_name": "noise",
        "question": "How high did you find the noise level today?",
        "scale": "1-7",
    },
)

QUESTION_IDS = tuple(row["question_id"] for row in QUESTIONNAIRE_ITEMS)
RATEABLE_QUESTION_IDS = QUESTION_IDS[:7]
Q2_PERCENTAGES = (0, 25, 50, 75, 100)


def questionnaire_json_schema(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return the constrained-output schema for one eight-question bundle."""

    evidence_ids = [str(value) for value in packet.get("evidence_ids", [])]
    evidence_item: dict[str, Any] = {"type": "string"}
    if evidence_ids:
        evidence_item["enum"] = evidence_ids
    fixed_q2 = (packet.get("fixed_answers") or {}).get("q2")
    share_values = [*Q2_PERCENTAGES, None]
    if fixed_q2 is not None:
        share_values = [int(fixed_q2), None]
    response = {
        "type": "object",
        "properties": {
            "question_id": {"type": "string", "enum": list(QUESTION_IDS)},
            "rateability": {
                "type": "string",
                "enum": ["rateable", "insufficient_evidence"],
            },
            "score_1_to_7": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 7,
            },
            "share_percent": {
                "type": ["integer", "null"],
                "enum": share_values,
            },
            "confidence_1_to_5": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
            "evidence_event_ids": {
                "type": "array",
                "items": evidence_item,
                "minItems": 1,
                "maxItems": max(1, len(evidence_ids)),
                "uniqueItems": True,
            },
            "short_rationale": {"type": "string", "minLength": 1, "maxLength": 320},
            "uncertainty_note": {"type": "string", "maxLength": 240},
        },
        "required": [
            "question_id",
            "rateability",
            "score_1_to_7",
            "share_percent",
            "confidence_1_to_5",
            "evidence_event_ids",
            "short_rationale",
            "uncertainty_note",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "responses": {
                "type": "array",
                "items": response,
                "minItems": len(QUESTION_IDS),
                "maxItems": len(QUESTION_IDS),
            },
            "not_human_data": {"type": "boolean", "const": True},
        },
        "required": ["responses", "not_human_data"],
        "additionalProperties": False,
    }


def normalize_questionnaire_response(
    packet: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate cross-field semantics without forcing scientific outcomes."""

    if payload.get("not_human_data") is not True:
        raise ValueError("Questionnaire response requires not_human_data=true")
    raw_rows = payload.get("responses")
    if not isinstance(raw_rows, list):
        raise ValueError("Questionnaire responses must be a list")
    allowed_evidence = {str(value) for value in packet.get("evidence_ids", [])}
    required_rateable = {
        str(value) for value in packet.get("required_rateable_question_ids", [])
    }
    required_insufficient = {
        str(value) for value in packet.get("required_insufficient_question_ids", [])
    }
    fixed_answers = dict(packet.get("fixed_answers") or {})
    by_question: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("Each questionnaire response must be an object")
        question_id = str(raw.get("question_id", ""))
        if question_id not in QUESTION_IDS:
            raise ValueError(f"Unknown questionnaire item: {question_id!r}")
        if question_id in by_question:
            raise ValueError(f"Duplicate questionnaire item: {question_id}")
        rateability = str(raw.get("rateability", ""))
        score = raw.get("score_1_to_7")
        share = raw.get("share_percent")
        if rateability == "insufficient_evidence":
            if score is not None or share is not None:
                raise ValueError(
                    f"{question_id} insufficient_evidence requires both scores to be null"
                )
        elif rateability == "rateable":
            if question_id == "q2":
                if score is not None or share not in Q2_PERCENTAGES:
                    raise ValueError("q2 requires one allowed percentage and a null 1-7 score")
            elif share is not None or not isinstance(score, int) or not 1 <= score <= 7:
                raise ValueError(
                    f"{question_id} requires a 1-7 integer and a null percentage"
                )
        else:
            raise ValueError(f"Unknown rateability for {question_id}: {rateability!r}")
        if question_id in required_rateable and rateability != "rateable":
            raise ValueError(f"{question_id} must be rateable under this evidence contract")
        if question_id in required_insufficient and rateability != "insufficient_evidence":
            raise ValueError(
                f"{question_id} must be insufficient_evidence under this evidence contract"
            )
        if question_id in fixed_answers and share != int(fixed_answers[question_id]):
            raise ValueError(
                f"{question_id} must report the supplied questionnaire category "
                f"{int(fixed_answers[question_id])}"
            )

        evidence = raw.get("evidence_event_ids")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{question_id} must cite at least one evidence ID")
        evidence = [str(value) for value in evidence]
        if len(evidence) != len(set(evidence)):
            raise ValueError(f"{question_id} cites duplicate evidence IDs")
        unknown = set(evidence) - allowed_evidence
        if unknown:
            raise ValueError(f"{question_id} cites unknown evidence IDs: {sorted(unknown)}")
        rationale = str(raw.get("short_rationale", "")).strip()
        if not rationale:
            raise ValueError(f"{question_id} requires a concise rationale")
        confidence = raw.get("confidence_1_to_5")
        if not isinstance(confidence, int) or not 1 <= confidence <= 5:
            raise ValueError(f"{question_id} confidence must be an integer from 1 to 5")
        by_question[question_id] = {
            "question_id": question_id,
            "rateability": rateability,
            "score_1_to_7": score,
            "share_percent": share,
            "confidence_1_to_5": confidence,
            "evidence_event_ids": evidence,
            "short_rationale": rationale,
            "uncertainty_note": str(raw.get("uncertainty_note", "")).strip(),
        }
    missing = set(QUESTION_IDS) - set(by_question)
    if missing:
        raise ValueError(f"Questionnaire bundle is missing items: {sorted(missing)}")
    return {
        "responses": [by_question[question_id] for question_id in QUESTION_IDS],
        "not_human_data": True,
    }


__all__ = [
    "QUESTIONNAIRE_ITEMS",
    "QUESTION_IDS",
    "RATEABLE_QUESTION_IDS",
    "Q2_PERCENTAGES",
    "normalize_questionnaire_response",
    "questionnaire_json_schema",
]
