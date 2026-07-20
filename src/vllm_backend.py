"""Explicit, GPU-only vLLM offline backend for Part 3 prompt packets.

This module is inert until instantiated by an explicit Part 3 command or
backend selection. It does not participate in default Part 1/2 execution.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Iterable, Mapping, Optional

import config


class PacketResponseDecodeError(ValueError):
    """Expose a malformed model response without treating it as valid output."""

    def __init__(
        self,
        *,
        prompt_id: str,
        raw_response: str,
        original_error: json.JSONDecodeError,
    ) -> None:
        super().__init__(
            f"Malformed JSON for {prompt_id}: {original_error}"
        )
        self.prompt_id = prompt_id
        self.raw_response = raw_response
        self.original_error = original_error


EVALUATOR_ONLY_PROMPT_TERMS = (
    "rule_reference",
    "rule_reference_action",
    "selected_action\":",
    "random_draw",
    "engagement_probability",
    "abm_reason_type",
    "logged_topic_families",
    "\nScenario:",
    "\nCondition:",
)


def scientific_packet_prompt_leakage(packet: Mapping[str, Any]) -> list[str]:
    """Return evaluator-only terms exposed by a scientific decision prompt."""

    if (
        packet.get("packet_type") != "in_simulation_decision"
        or packet.get("fixture_only_not_scientific_data") is not False
    ):
        return []
    prompt_text = str(packet.get("user_message", ""))
    visible_evidence = json.dumps(
        packet.get("llm_visible_evidence", {}), sort_keys=True
    )
    model_identifiers = json.dumps(
        {
            "prompt_id": packet.get("prompt_id"),
            "evidence_ids": packet.get("evidence_ids", []),
        },
        sort_keys=True,
    )
    searchable = f"{prompt_text}\n{visible_evidence}\n{model_identifiers}"
    leakage = [term for term in EVALUATOR_ONLY_PROMPT_TERMS if term in searchable]
    trace = packet.get("trace_evidence", {})
    source_evidence_id = str(trace.get("evidence_id", ""))
    if source_evidence_id and source_evidence_id in searchable:
        leakage.append("source_evidence_id")
    abm_reason = str(trace.get("decision_context", {}).get("abm_reason_type", ""))
    if abm_reason and abm_reason in searchable:
        leakage.append("abm_reason_value")
    return leakage


def _object_schema(properties: Mapping[str, Any], required: Iterable[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _vllm_grammar_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the subset of the scientific schema supported by vLLM grammars.

    vLLM/xgrammar does not implement JSON Schema's ``uniqueItems`` keyword.
    Evidence-ID uniqueness remains part of ``packet_json_schema`` and is
    enforced after generation by ``_validated_evidence_ids``; only the schema
    copy passed to constrained decoding drops that unsupported keyword.
    """

    def compatible(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: compatible(child)
                for key, child in value.items()
                if key != "uniqueItems"
            }
        if isinstance(value, list):
            return [compatible(child) for child in value]
        return value

    return compatible(schema)


def packet_json_schema(packet_type: str, packet: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return an executable JSON schema matching the consolidated Part 3 schemas."""

    packet = packet or {}
    allowed_evidence = [str(value) for value in packet.get("evidence_ids", [])]
    evidence_items: dict[str, Any] = {"type": "string"}
    if allowed_evidence:
        evidence_items["enum"] = allowed_evidence
    evidence_array: dict[str, Any] = {
        "type": "array",
        "items": evidence_items,
        "uniqueItems": True,
    }
    if allowed_evidence:
        evidence_array["minItems"] = 1
        evidence_array["maxItems"] = len(allowed_evidence)

    if packet_type == "end_of_shift_survey":
        from src.personas import PERSONA_APPRAISAL_DIMENSIONS

        requested_dimension = str((packet or {}).get("survey_dimension", ""))
        dimensions = (
            [requested_dimension]
            if requested_dimension in PERSONA_APPRAISAL_DIMENSIONS
            else list(PERSONA_APPRAISAL_DIMENSIONS)
        )
        properties = {
            "dimension": {"type": "string", "enum": dimensions},
            "rateability": {
                "type": "string",
                "enum": ["rateable", "insufficient_evidence"],
            },
            "score_1_to_7": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 7,
            },
            "confidence_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
            "evidence_event_ids": evidence_array,
            "short_rationale": {"type": "string"},
            "uncertainty_note": {"type": "string"},
            "not_human_data": {"type": "boolean", "const": True},
        }
        return _object_schema(properties, properties)

    if packet_type == "end_of_shift_survey_bundle":
        from src.personas import PERSONA_APPRAISAL_DIMENSIONS

        response_properties = {
            "dimension": {
                "type": "string",
                "enum": list(PERSONA_APPRAISAL_DIMENSIONS),
            },
            "rateability": {
                "type": "string",
                "enum": ["rateable", "insufficient_evidence"],
            },
            "score_1_to_7": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 7,
            },
            "confidence_1_to_5": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
            "evidence_event_ids": evidence_array,
            "short_rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 260,
            },
            "uncertainty_note": {"type": "string", "maxLength": 180},
        }
        response_schema = _object_schema(
            response_properties, response_properties
        )
        properties = {
            "responses": {
                "type": "array",
                "items": response_schema,
                "minItems": len(PERSONA_APPRAISAL_DIMENSIONS),
                "maxItems": len(PERSONA_APPRAISAL_DIMENSIONS),
            },
            "not_human_data": {"type": "boolean", "const": True},
        }
        return _object_schema(properties, properties)

    if packet_type == "end_of_shift_interview":
        properties = {
            "question_id": {"type": "string"},
            "answer": {"type": "string"},
            "evidence_event_ids": evidence_array,
            "role_perspective": {"type": "string"},
            "uncertainty_note": {"type": "string"},
            "not_human_data": {"type": "boolean", "const": True},
        }
        return _object_schema(properties, properties)

    if packet_type == "end_of_shift_interview_bundle":
        from src.interviews import INTERVIEW_QUESTIONS

        question_ids = [row["question_id"] for row in INTERVIEW_QUESTIONS]
        nullable_text = {"type": ["string", "null"], "maxLength": 220}
        answer_properties = {
            "question_id": {"type": "string", "enum": question_ids},
            "answer": {"type": "string", "minLength": 1, "maxLength": 360},
            "grounded_pattern": {
                "type": "string",
                "minLength": 1,
                "maxLength": 260,
            },
            "persona_conditioned_interpretation": {
                "type": "string",
                "minLength": 1,
                "maxLength": 260,
            },
            "latent_need": nullable_text,
            "design_hypothesis": nullable_text,
            "tradeoff": nullable_text,
            "evidence_event_ids": evidence_array,
            "uncertainty_note": {"type": "string", "maxLength": 180},
        }
        answer_schema = _object_schema(answer_properties, answer_properties)
        properties = {
            "answers": {
                "type": "array",
                "items": answer_schema,
                "minItems": len(question_ids),
                "maxItems": len(question_ids),
            },
            "role_perspective": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            },
            "claim_layer_contract": {
                "type": "string",
                "const": "grounded_pattern_interpretation_conjecture_v1",
            },
            "not_human_data": {"type": "boolean", "const": True},
        }
        return _object_schema(properties, properties)

    if packet_type == "critical_incident":
        properties = {
            "incident_type": {"type": "string"},
            "why_it_mattered_operationally": {"type": "string"},
            "spatial_factor_if_any": {"type": "string"},
            "coordination_factor_if_any": {"type": "string"},
            "supporting_event_ids": evidence_array,
            "uncertainty_note": {"type": "string"},
        }
        return _object_schema(properties, properties)

    if packet_type == "in_simulation_decision":
        categorical_causal = (
            packet.get("decision_output_contract") == "categorical_causal_v1"
        )
        feasible = [str(value) for value in packet.get("feasible_actions", [])]
        reasons = [str(value) for value in packet.get("allowed_reasons", [])]
        topic_families = [str(value) for value in packet.get("allowed_topic_families", [])]
        action_schema: dict[str, Any] = {"type": "string"}
        reason_schema: dict[str, Any] = {"type": "string"}
        topic_schema: dict[str, Any] = {"type": "string"}
        if feasible:
            action_schema["enum"] = feasible
        if reasons:
            reason_schema["enum"] = reasons
        if topic_families:
            topic_schema["enum"] = topic_families
        decision_id_schema: dict[str, Any] = {"type": "string", "minLength": 1}
        if packet.get("prompt_id"):
            decision_id_schema["const"] = str(packet["prompt_id"])
        if categorical_causal:
            properties = {
                "decision_id": decision_id_schema,
                "selected_action": action_schema,
                "selected_reason": reason_schema,
                "topic_family": topic_schema,
                "evidence_ids": evidence_array,
            }
        else:
            # Keep the legacy insertion order because it is part of the recorded
            # JSON-schema digest in completed grounding packages.
            properties = {
                "decision_id": decision_id_schema,
                "selected_action": action_schema,
                "selected_reason": reason_schema,
                "topic_family": topic_schema,
                "topic_text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 320,
                },
                "rationale_short": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 480,
                },
                "evidence_ids": evidence_array,
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            }
        return _object_schema(properties, properties)

    properties = {
        "answer": {"type": "string"},
        "evidence_event_ids": evidence_array,
        "uncertainty_note": {"type": "string"},
        "not_human_data": {"type": "boolean", "const": True},
    }
    return _object_schema(properties, properties)


class VLLMOfflineBackend:
    """Lazy vLLM runner for batched, schema-constrained Part 3 inference."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        model_revision: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        language_model_only: bool = True,
        enable_prefix_caching: bool = True,
        safetensors_load_strategy: str | None = "eager",
    ) -> None:
        self.model = model or config.VLLM_MODEL_NAME
        self.model_revision = model_revision or None
        self.tensor_parallel_size = int(
            config.VLLM_TENSOR_PARALLEL_SIZE if tensor_parallel_size is None else tensor_parallel_size
        )
        self.max_model_len = int(config.VLLM_MAX_MODEL_LEN if max_model_len is None else max_model_len)
        self.gpu_memory_utilization = float(
            config.VLLM_GPU_MEMORY_UTILIZATION
            if gpu_memory_utilization is None
            else gpu_memory_utilization
        )
        self.max_output_tokens = int(
            config.VLLM_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
        )
        self.enable_thinking = bool(
            config.VLLM_ENABLE_THINKING if enable_thinking is None else enable_thinking
        )
        self.language_model_only = bool(language_model_only)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.safetensors_load_strategy = safetensors_load_strategy
        self._engine = None
        self.engine_load_seconds: float | None = None

    def _ensure_engine(self):
        if self._engine is None:
            try:
                from vllm import LLM
            except ImportError as error:  # pragma: no cover - GPU environment only.
                raise RuntimeError(
                    "vLLM is not installed. Use the separate Part 3 LLM environment and "
                    "requirements-llm.txt."
                ) from error
            engine_options = {
                "model": self.model,
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": False,
                "dtype": "auto",
                "enable_prefix_caching": self.enable_prefix_caching,
            }
            if self.safetensors_load_strategy:
                engine_options["safetensors_load_strategy"] = (
                    self.safetensors_load_strategy
                )
            if self.model_revision:
                # Pin both weights and tokenizer to the same immutable Hub
                # commit for reproducible scientific runs.
                engine_options["revision"] = self.model_revision
                engine_options["tokenizer_revision"] = self.model_revision
            if self.language_model_only:
                # Offline equivalent of the text-only serving mode: skip
                # multimodal profiling/cache allocation for text packets.
                engine_options["limit_mm_per_prompt"] = {"image": 0, "video": 0}
            started = time.perf_counter()
            self._engine = LLM(**engine_options)
            self.engine_load_seconds = time.perf_counter() - started
        return self._engine

    def load(self) -> float:
        """Load the engine once and return measured initialization seconds."""

        self._ensure_engine()
        return float(self.engine_load_seconds or 0.0)

    @staticmethod
    def _packet_messages(packet: Mapping[str, Any]) -> list[dict[str, str]]:
        schema = packet_json_schema(str(packet.get("packet_type", "")), packet)
        user_message = (
            f"{packet.get('user_message', '')}\n\n"
            f"Required JSON schema:\n{json.dumps(schema, sort_keys=True)}\n"
            "Return one JSON object only. Cite only supplied evidence IDs."
        )
        return [
            {"role": "system", "content": str(packet.get("system_message", "Return JSON only."))},
            {"role": "user", "content": user_message},
        ]

    def _sampling_params(self, schema: Mapping[str, Any]):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=self.max_output_tokens,
            structured_outputs=StructuredOutputsParams(
                json=_vllm_grammar_schema(schema)
            ),
        )

    def _chat_batch(
        self,
        messages: list[list[dict[str, str]]],
        schemas: list[Mapping[str, Any]],
    ) -> list[str]:
        if len(messages) != len(schemas):
            raise ValueError("Each chat conversation must have its own response schema")
        engine = self._ensure_engine()
        outputs = engine.chat(
            messages,
            sampling_params=[self._sampling_params(schema) for schema in schemas],
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            use_tqdm=False,
        )
        return [output.outputs[0].text.strip() for output in outputs]

    @staticmethod
    def _validated_evidence_ids(
        payload: Mapping[str, Any],
        packet: Mapping[str, Any],
        key: str,
        *,
        require_one: bool = False,
    ) -> list[str]:
        allowed = {str(value) for value in packet.get("evidence_ids", [])}
        raw_values = payload.get(key, [])
        if not isinstance(raw_values, list):
            raise ValueError(f"{key} must be a list")
        values = [str(value) for value in raw_values]
        invalid = [value for value in values if value not in allowed]
        if invalid:
            raise ValueError(f"Response cited evidence not supplied by the packet: {invalid}")
        if len(values) != len(set(values)):
            raise ValueError(f"Response cited duplicate evidence IDs in {key}")
        if require_one and not values:
            raise ValueError(f"Response must cite at least one supplied evidence ID in {key}")
        return values

    def normalize_packet_response(
        self,
        packet: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Map generated JSON to the existing Part 3 response dataclasses."""

        packet_type = str(packet.get("packet_type", ""))
        profile = packet.get("user_model_profile", {})
        metadata = packet.get("trace_evidence", {}).get("metadata", {})
        role = str(packet.get("role") or profile.get("role_name") or "")
        user_model_id = str(profile.get("persona_id") or profile.get("user_model_id") or "")
        condition = str(metadata.get("condition") or "")
        scenario_mode = str(metadata.get("scenario_mode") or "")

        if packet_type == "end_of_shift_survey":
            from src.interviews import SurveyResponse

            rateability = str(payload["rateability"])
            raw_score = payload["score_1_to_7"]
            if rateability == "rateable":
                if raw_score is None:
                    raise ValueError("Rateable survey response requires a 1-7 score")
                score = max(1, min(7, int(raw_score)))
            elif rateability == "insufficient_evidence":
                if raw_score is not None:
                    raise ValueError(
                        "Insufficient-evidence survey response must use a null score"
                    )
                score = None
            else:
                raise ValueError(f"Unknown survey rateability: {rateability!r}")

            response = SurveyResponse(
                dimension=str(payload["dimension"]),
                rateability=rateability,
                score_1_to_7=score,
                confidence_1_to_5=max(1, min(5, int(payload["confidence_1_to_5"]))),
                evidence_event_ids=self._validated_evidence_ids(
                    payload, packet, "evidence_event_ids"
                ),
                short_rationale=str(payload["short_rationale"]),
                uncertainty_note=str(payload["uncertainty_note"]),
                user_model_id=user_model_id,
                role=role,
                condition=condition,
                scenario_mode=scenario_mode,
                not_human_data=True,
            )
            return response.as_dict()

        if packet_type == "end_of_shift_survey_bundle":
            from src.interviews import SurveyBundleResponse, SurveyDimensionResponse
            from src.personas import PERSONA_APPRAISAL_DIMENSIONS

            raw_responses = payload.get("responses")
            if not isinstance(raw_responses, list):
                raise ValueError("Survey bundle responses must be a list")
            by_dimension: dict[str, SurveyDimensionResponse] = {}
            for raw in raw_responses:
                dimension = str(raw.get("dimension", ""))
                if dimension not in PERSONA_APPRAISAL_DIMENSIONS:
                    raise ValueError(f"Unknown survey dimension: {dimension!r}")
                if dimension in by_dimension:
                    raise ValueError(f"Duplicate survey dimension: {dimension!r}")
                rateability = str(raw.get("rateability", ""))
                raw_score = raw.get("score_1_to_7")
                if rateability == "rateable":
                    if raw_score is None:
                        raise ValueError(
                            f"Rateable {dimension} response requires a score"
                        )
                    score = max(1, min(7, int(raw_score)))
                elif rateability == "insufficient_evidence":
                    if raw_score is not None:
                        raise ValueError(
                            f"Insufficient-evidence {dimension} response must use null"
                        )
                    score = None
                else:
                    raise ValueError(
                        f"Unknown survey rateability for {dimension}: {rateability!r}"
                    )
                by_dimension[dimension] = SurveyDimensionResponse(
                    dimension=dimension,
                    rateability=rateability,
                    score_1_to_7=score,
                    confidence_1_to_5=max(
                        1, min(5, int(raw.get("confidence_1_to_5", 1)))
                    ),
                    evidence_event_ids=self._validated_evidence_ids(
                        raw, packet, "evidence_event_ids", require_one=True
                    ),
                    short_rationale=str(raw.get("short_rationale", "")),
                    uncertainty_note=str(raw.get("uncertainty_note", "")),
                )
            missing = set(PERSONA_APPRAISAL_DIMENSIONS) - set(by_dimension)
            if missing:
                raise ValueError(
                    f"Survey bundle is missing dimensions: {sorted(missing)}"
                )
            response = SurveyBundleResponse(
                responses=[by_dimension[key] for key in PERSONA_APPRAISAL_DIMENSIONS],
                user_model_id=user_model_id,
                role=role,
                condition=condition,
                scenario_mode=scenario_mode,
            )
            return response.as_dict()

        if packet_type == "end_of_shift_interview":
            from src.interviews import InterviewResponse

            response = InterviewResponse(
                question_id=str(payload["question_id"]),
                answer=str(payload["answer"]),
                evidence_event_ids=self._validated_evidence_ids(
                    payload, packet, "evidence_event_ids"
                ),
                role_perspective=str(payload["role_perspective"]),
                uncertainty_note=str(payload["uncertainty_note"]),
                user_model_id=user_model_id,
                role=role,
                condition=condition,
                scenario_mode=scenario_mode,
                not_human_data=True,
            )
            return response.as_dict()

        if packet_type == "end_of_shift_interview_bundle":
            from src.interviews import (
                INTERVIEW_QUESTIONS,
                InterviewAnswer,
                InterviewBundleResponse,
            )

            expected_ids = [row["question_id"] for row in INTERVIEW_QUESTIONS]
            raw_answers = payload.get("answers")
            if not isinstance(raw_answers, list):
                raise ValueError("Interview bundle answers must be a list")
            by_question: dict[str, InterviewAnswer] = {}
            for raw in raw_answers:
                question_id = str(raw.get("question_id", ""))
                if question_id not in expected_ids:
                    raise ValueError(f"Unknown interview question: {question_id!r}")
                if question_id in by_question:
                    raise ValueError(f"Duplicate interview question: {question_id!r}")
                by_question[question_id] = InterviewAnswer(
                    question_id=question_id,
                    answer=str(raw.get("answer", "")).strip(),
                    grounded_pattern=str(raw.get("grounded_pattern", "")).strip(),
                    persona_conditioned_interpretation=str(
                        raw.get("persona_conditioned_interpretation", "")
                    ).strip(),
                    latent_need=(
                        str(raw["latent_need"]).strip()
                        if raw.get("latent_need") is not None
                        else None
                    ),
                    design_hypothesis=(
                        str(raw["design_hypothesis"]).strip()
                        if raw.get("design_hypothesis") is not None
                        else None
                    ),
                    tradeoff=(
                        str(raw["tradeoff"]).strip()
                        if raw.get("tradeoff") is not None
                        else None
                    ),
                    evidence_event_ids=self._validated_evidence_ids(
                        raw, packet, "evidence_event_ids", require_one=True
                    ),
                    uncertainty_note=str(raw.get("uncertainty_note", "")).strip(),
                )
            missing = set(expected_ids) - set(by_question)
            if missing:
                raise ValueError(
                    f"Interview bundle is missing questions: {sorted(missing)}"
                )
            for question_id, answer in by_question.items():
                if not answer.answer or not answer.grounded_pattern:
                    raise ValueError(
                        f"Interview answer {question_id!r} lacks natural or grounded text"
                    )
                if answer.design_hypothesis and not answer.tradeoff:
                    raise ValueError(
                        f"Design hypothesis for {question_id!r} requires a tradeoff"
                    )
            response = InterviewBundleResponse(
                answers=[by_question[key] for key in expected_ids],
                role_perspective=str(payload.get("role_perspective", "")).strip(),
                user_model_id=user_model_id,
                role=role,
                condition=condition,
                scenario_mode=scenario_mode,
            )
            return response.as_dict()

        if packet_type == "critical_incident":
            from src.interviews import CriticalIncidentReport

            response = CriticalIncidentReport(
                incident_type=str(payload["incident_type"]),
                why_it_mattered_operationally=str(payload["why_it_mattered_operationally"]),
                spatial_factor_if_any=str(payload["spatial_factor_if_any"]),
                coordination_factor_if_any=str(payload["coordination_factor_if_any"]),
                supporting_event_ids=self._validated_evidence_ids(
                    payload, packet, "supporting_event_ids"
                ),
                uncertainty_note=str(payload["uncertainty_note"]),
            )
            return response.as_dict()

        if packet_type == "in_simulation_decision":
            categorical_causal = (
                packet.get("decision_output_contract") == "categorical_causal_v1"
            )
            feasible = [str(value) for value in packet.get("feasible_actions", [])]
            allowed_reasons = [str(value) for value in packet.get("allowed_reasons", [])]
            allowed_topics = [str(value) for value in packet.get("allowed_topic_families", [])]
            decision_id = str(payload.get("decision_id", ""))
            expected_decision_id = str(packet.get("prompt_id", ""))
            if decision_id != expected_decision_id:
                raise ValueError(
                    f"Decision ID {decision_id!r} does not match prompt ID "
                    f"{expected_decision_id!r}"
                )
            action = str(payload.get("selected_action", ""))
            if action not in feasible:
                raise ValueError(f"Disallowed action {action!r}; expected one of {feasible}")
            reason = str(payload.get("selected_reason", ""))
            if reason not in allowed_reasons:
                raise ValueError(
                    f"Disallowed reason {reason!r}; expected one of {allowed_reasons}"
                )
            topic_family = str(payload.get("topic_family", ""))
            if topic_family not in allowed_topics:
                raise ValueError(
                    f"Disallowed topic {topic_family!r}; expected one of {allowed_topics}"
                )
            normalized = {
                "decision_id": decision_id,
                "selected_action": action,
                "selected_reason": reason,
                "topic_family": topic_family,
                "evidence_ids": self._validated_evidence_ids(
                    payload, packet, "evidence_ids", require_one=True
                ),
            }
            if categorical_causal:
                return normalized
            topic_text = str(payload.get("topic_text", ""))
            rationale = str(payload.get("rationale_short", ""))
            if not topic_text.strip() or not rationale.strip():
                raise ValueError("Decision topic_text and rationale_short must be non-empty")
            confidence = float(payload.get("confidence", -1.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Decision confidence outside 0..1: {confidence}")
            normalized.update(
                {
                    "topic_text": topic_text,
                    "rationale_short": rationale,
                    "confidence": confidence,
                }
            )
            return normalized

        return {
            "answer": str(payload.get("answer", "")),
            "evidence_event_ids": self._validated_evidence_ids(
                payload, packet, "evidence_event_ids"
            ),
            "uncertainty_note": str(payload.get("uncertainty_note", "")),
            "not_human_data": True,
        }

    def generate_prompt_packets(self, packets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        packet_list = [dict(packet) for packet in packets]
        schemas = [
            packet_json_schema(str(packet.get("packet_type", "")), packet)
            for packet in packet_list
        ]
        messages = [self._packet_messages(packet) for packet in packet_list]
        raw_outputs = self._chat_batch(messages, schemas)
        results = []
        for packet, schema, raw_text in zip(packet_list, schemas, raw_outputs):
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as error:
                raise PacketResponseDecodeError(
                    prompt_id=str(packet.get("prompt_id", "")),
                    raw_response=raw_text,
                    original_error=error,
                ) from error
            normalized = self.normalize_packet_response(packet, payload)
            results.append(
                {
                    "prompt_id": packet.get("prompt_id"),
                    "packet_type": packet.get("packet_type"),
                    "model": self.model,
                    "thinking_enabled": self.enable_thinking,
                    "response_schema_sha256": hashlib.sha256(
                        json.dumps(schema, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "response": normalized,
                    "raw_response": raw_text,
                    "synthetic_design_probe_not_human_data": True,
                }
            )
        return results

__all__ = [
    "PacketResponseDecodeError",
    "VLLMOfflineBackend",
    "packet_json_schema",
    "scientific_packet_prompt_leakage",
]
