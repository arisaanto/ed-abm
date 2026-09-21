"""Bounded Part 3 controller for optional staff-contact decisions.

The controller receives ABM-created, route-supported opportunities and may only
select a categorical action, reason, and canonical topic family. It cannot
create opportunities or modify geometry, movement, workflow, or patient state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Protocol, Sequence

from src.interaction import PART3_GROUNDED_MEMORY_POLICY
from src.interviews import categorical_decision_bounds
from src.personas import cognitive_persona_by_id


_GROUNDED_MEMORY_REQUIRED_FIELDS = {
    "memory_id",
    "timestamp",
    "seconds_ago",
    "event_type",
    "outcome",
    "partner_id",
    "partner_role",
    "owner_was_initiator",
    "patient_context_id",
    "zone",
    "topic_family",
    "interaction_type",
    "reason_type",
    "importance",
    "same_partner",
    "same_patient_context",
    "retrieval_rank",
    "retrieval_score",
    "retrieval_components",
    "salience_features",
    "retrieval_match_features",
    "source_event_id",
    "support_ids",
    "source",
    "memory_policy",
}
_PROHIBITED_DECISION_HISTORY_FIELDS = {
    "selected_action",
    "selected_reason",
    "prior_action",
    "prior_reason",
}


@dataclass(frozen=True)
class CausalDecisionRequest:
    decision_id: str
    evidence_id: str
    persona_id: str
    agent_id: int
    timestep: int
    feasible_actions: tuple[str, ...]
    allowed_reasons: tuple[str, ...]
    allowed_topic_families: tuple[str, ...]
    rule_reference_action: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class CausalDecision:
    decision_id: str
    evidence_id: str
    persona_id: str
    selected_action: str
    selected_reason: str
    topic_family: str
    evidence_ids: tuple[str, ...]
    policy_name: str
    was_fallback: bool = False
    rejected_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EVOLVING_STATE_DIMENSIONS = (
    "coordination_need",
    "interruption_strain",
    "task_continuity",
    "team_support",
)
EVOLVING_STATE_MIN = -2
EVOLVING_STATE_MAX = 2
EVOLVING_STATE_INITIAL = {name: 0 for name in EVOLVING_STATE_DIMENSIONS}


@dataclass(frozen=True)
class EvolvingStateUpdateRequest:
    checkpoint_id: str
    persona_id: str
    agent_id: int
    role: str
    timestep: int
    window_start: int
    prior_state: Mapping[str, int]
    evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class EvolvingStateUpdate:
    checkpoint_id: str
    persona_id: str
    state: Mapping[str, int]
    evidence_by_dimension: Mapping[str, tuple[str, ...]]
    policy_name: str
    was_fallback: bool = False
    rejected_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = dict(self.state)
        payload["evidence_by_dimension"] = {
            name: list(values)
            for name, values in self.evidence_by_dimension.items()
        }
        return payload


class CausalDecisionProvider(Protocol):
    policy_name: str

    def decide_many(
        self, requests: Sequence[CausalDecisionRequest]
    ) -> list[CausalDecision]: ...

    def update_states_many(
        self, requests: Sequence[EvolvingStateUpdateRequest]
    ) -> list[EvolvingStateUpdate]: ...


class DeterministicMockCausalProvider:
    """Evidence-sensitive mock used only to test closed-loop plumbing."""

    policy_name = "deterministic_mock_categorical"

    @staticmethod
    def _action(request: CausalDecisionRequest) -> str:
        contact = request.evidence.get("decision_context", {})
        workflow = request.evidence.get("workflow_context", {})
        urgency = float(contact.get("urgency", 0.0) or 0.0)
        diversion = float(contact.get("diversion_cost", 0.0) or 0.0)
        pressure = float(workflow.get("ed_pressure_index", 0.0) or 0.0)
        patient_linked = workflow.get("patient_id") not in (None, -1, "-1")
        high_acuity = bool(workflow.get("patient_high_acuity_escalation"))

        if request.persona_id == "team_connector":
            return "defer" if urgency < 0.5 and diversion >= 0.67 else "engage"
        if request.persona_id == "focus_protector":
            if high_acuity or urgency >= 0.8:
                return "engage"
            return "decline" if urgency < 0.5 else "defer"
        if request.persona_id == "patient_advocate":
            if patient_linked:
                return "engage"
            return "defer" if urgency >= 0.5 else "decline"
        if request.persona_id == "vigilant_monitor":
            if high_acuity or urgency >= 0.8 or pressure >= 0.85:
                return "engage"
            return "decline" if urgency < 0.5 else "defer"
        value = urgency + (0.20 if patient_linked else 0.0) + (0.15 if pressure >= 0.85 else 0.0)
        return "engage" if value > diversion else "defer"

    @staticmethod
    def _bounded_choice(preferred: Sequence[str], allowed: tuple[str, ...]) -> str:
        for value in preferred:
            if value in allowed:
                return value
        return allowed[0]

    def decide_many(
        self, requests: Sequence[CausalDecisionRequest]
    ) -> list[CausalDecision]:
        decisions = []
        for request in requests:
            action = self._action(request)
            if action == "engage":
                reason_preference = (
                    "urgent_escalation",
                    "same_patient_update",
                    "coordination_need",
                    "relational_check_in",
                    "insufficient_relevance",
                )
            else:
                reason_preference = (
                    "protect_task_continuity",
                    "insufficient_relevance",
                    "coordination_need",
                    "relational_check_in",
                )
            persona = cognitive_persona_by_id()[request.persona_id]
            topic_preference = tuple(persona.topic_family_preferences)
            decisions.append(
                CausalDecision(
                    decision_id=request.decision_id,
                    evidence_id=request.evidence_id,
                    persona_id=request.persona_id,
                    selected_action=action,
                    selected_reason=self._bounded_choice(
                        reason_preference, request.allowed_reasons
                    ),
                    topic_family=self._bounded_choice(
                        topic_preference, request.allowed_topic_families
                    ),
                    evidence_ids=(request.evidence_id,),
                    policy_name=self.policy_name,
                )
            )
        return decisions

    def update_states_many(
        self, requests: Sequence[EvolvingStateUpdateRequest]
    ) -> list[EvolvingStateUpdate]:
        updates = []
        for request in requests:
            state = dict(request.prior_state)
            decisions = [
                row for row in request.evidence if row.get("event_type") == "persona_decision"
            ]
            contacts = [
                row for row in request.evidence if row.get("event_type") == "realized_interaction"
            ]
            def move(name: str, direction: int) -> None:
                state[name] = max(
                    EVOLVING_STATE_MIN,
                    min(EVOLVING_STATE_MAX, int(state[name]) + direction),
                )
            if sum(row.get("selected_action") == "decline" for row in decisions) >= 2:
                move("interruption_strain", 1)
            if sum(row.get("selected_action") == "defer" for row in decisions) >= 2:
                move("task_continuity", -1)
            if contacts:
                move("team_support", 1)
                move("coordination_need", -1)
            updates.append(
                EvolvingStateUpdate(
                    checkpoint_id=request.checkpoint_id,
                    persona_id=request.persona_id,
                    state=state,
                    evidence_by_dimension={
                        name: (
                            tuple(str(row["evidence_id"]) for row in request.evidence)
                            if state[name] != request.prior_state[name]
                            else ()
                        )
                        for name in EVOLVING_STATE_DIMENSIONS
                    },
                    policy_name=self.policy_name,
                )
            )
        return updates


class Part3ClosedLoopController:
    """Validate decisions and expose only grounded experiential memory."""

    def __init__(
        self,
        provider: CausalDecisionProvider,
        persona_by_agent: Mapping[int, str],
        *,
        start_seconds: int = 0,
        max_decisions_per_run: int = 40,
        max_decisions_per_agent: int = 6,
        decision_window_seconds: int = 0,
        max_decisions_per_window: int = 0,
        model_decision_sample_rate: float = 1.0,
        sampling_replication_id: int = 0,
        evolving_state_enabled: bool = False,
        evolving_state_interval_seconds: int = 7200,
    ) -> None:
        self.provider = provider
        self.persona_by_agent = {
            int(agent_id): str(persona_id)
            for agent_id, persona_id in persona_by_agent.items()
        }
        known_personas = cognitive_persona_by_id()
        unknown_personas = sorted(
            set(self.persona_by_agent.values()) - set(known_personas)
        )
        if unknown_personas:
            raise ValueError(f"Unknown cognitive persona ids: {unknown_personas}")
        self.start_seconds = max(int(start_seconds), 0)
        self.max_decisions_per_run = max(int(max_decisions_per_run), 0)
        self.max_decisions_per_agent = max(int(max_decisions_per_agent), 0)
        self.decision_window_seconds = max(int(decision_window_seconds), 0)
        self.max_decisions_per_window = max(int(max_decisions_per_window), 0)
        self.model_decision_sample_rate = float(model_decision_sample_rate)
        self.sampling_replication_id = int(sampling_replication_id)
        self.evolving_state_enabled = bool(evolving_state_enabled)
        self.evolving_state_interval_seconds = max(
            int(evolving_state_interval_seconds), 1
        )
        if not 0.0 < self.model_decision_sample_rate <= 1.0:
            raise ValueError("model_decision_sample_rate must be in (0, 1]")
        if bool(self.decision_window_seconds) != bool(self.max_decisions_per_window):
            raise ValueError(
                "decision_window_seconds and max_decisions_per_window must "
                "both be zero or both be positive"
            )
        self.decision_log: list[dict[str, Any]] = []
        self.provider_error_log: list[dict[str, Any]] = []
        self.evolving_state_log: list[dict[str, Any]] = []
        self.evolving_state_by_agent: dict[int, dict[str, int]] = {
            agent_id: dict(EVOLVING_STATE_INITIAL)
            for agent_id in self.persona_by_agent
        }
        self._last_state_checkpoint = self.start_seconds
        self._next_state_checkpoint = (
            self.start_seconds + self.evolving_state_interval_seconds
        )
        self.memory_exposures_by_agent: dict[
            int, list[dict[str, Any]]
        ] = defaultdict(list)
        self.stats: Counter[str] = Counter()
        self._calls_by_agent: Counter[int] = Counter()
        self._calls_by_window: Counter[int] = Counter()
        self._eligible_by_agent: Counter[int] = Counter()
        self._eligible_by_window: Counter[int] = Counter()
        self._sampled_by_agent: Counter[int] = Counter()
        self._sampled_by_window: Counter[int] = Counter()

    def active_at(self, timestep: int) -> bool:
        """Return whether the optional cognitive policy is active at this time."""

        return int(timestep) >= self.start_seconds

    def memory_salience_modifiers_for(self, agent_id: int) -> Mapping[str, float]:
        """Return the assigned persona's preregistered retrieval modifiers."""

        persona_id = self.persona_by_agent[int(agent_id)]
        return cognitive_persona_by_id()[persona_id].memory_salience_modifiers

    @staticmethod
    def _state_evidence_id(source: str) -> str:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return f"state-evidence-{digest}"

    def _state_evidence_for_agent(
        self, simulation: Any, agent_id: int, window_start: int, timestep: int
    ) -> tuple[Mapping[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        stream = simulation.interaction_engine.stream_for(agent_id)
        for event in stream.events:
            if not (
                window_start <= int(event.timestamp) < timestep
                and event.event_type == "communicative_interaction"
                and event.outcome == "realized"
                and event.source_event_id
            ):
                continue
            source = str(event.source_event_id)
            rows.append(
                {
                    "evidence_id": self._state_evidence_id(source),
                    "source_evidence_id": source,
                    "event_type": "realized_interaction",
                    "timestamp": int(event.timestamp),
                    "colleague_role": event.partner_role,
                    "interaction_type": event.interaction_type,
                    "reason_context": event.reason_type,
                    "topic_family": event.topic,
                    "zone": event.zone_id,
                    "owner_was_initiator": bool(event.owner_was_initiator),
                    "patient_context_present": event.patient_id is not None,
                }
            )
        for decision in self.decision_log:
            if (
                int(decision.get("agent_id", -1)) != agent_id
                or not (window_start <= int(decision.get("timestep", -1)) < timestep)
                or decision.get("was_fallback") is True
                or decision.get("sampled_for_model") is not True
            ):
                continue
            source = str(decision["decision_id"])
            rows.append(
                {
                    "evidence_id": self._state_evidence_id(source),
                    "source_evidence_id": source,
                    "event_type": "persona_decision",
                    "timestamp": int(decision["timestep"]),
                    "selected_action": decision["selected_action"],
                    "selected_reason": decision["selected_reason"],
                    "colleague_role": decision.get("partner_role"),
                    "interaction_type": decision.get("interaction_type"),
                    "patient_context_present": bool(
                        decision.get("patient_context_present")
                    ),
                }
            )
        # Preserve the whole bounded checkpoint window while keeping prompts compact.
        # Recent events are preferred if an unusually active agent exceeds the cap.
        rows.sort(key=lambda row: (int(row["timestamp"]), row["evidence_id"]))
        return tuple(rows[-24:])

    @staticmethod
    def _state_validation_error(
        request: EvolvingStateUpdateRequest, update: EvolvingStateUpdate
    ) -> str | None:
        if update.checkpoint_id != request.checkpoint_id:
            return "checkpoint_id_mismatch"
        if update.persona_id != request.persona_id:
            return "persona_id_mismatch"
        if set(update.state) != set(EVOLVING_STATE_DIMENSIONS):
            return "invalid_state_dimensions"
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not EVOLVING_STATE_MIN <= value <= EVOLVING_STATE_MAX
            for value in update.state.values()
        ):
            return "state_value_out_of_bounds"
        if any(
            abs(int(update.state[name]) - int(request.prior_state[name])) > 1
            for name in EVOLVING_STATE_DIMENSIONS
        ):
            return "state_transition_exceeds_one_step"
        if set(update.evidence_by_dimension) != set(EVOLVING_STATE_DIMENSIONS):
            return "invalid_state_evidence_dimensions"
        allowed = {str(row["evidence_id"]) for row in request.evidence}
        for name in EVOLVING_STATE_DIMENSIONS:
            cited_values = tuple(update.evidence_by_dimension[name])
            cited = set(cited_values)
            changed = int(update.state[name]) != int(request.prior_state[name])
            if (
                len(cited) != len(cited_values)
                or not cited <= allowed
                or (changed and not cited)
                or (not changed and cited)
            ):
                return f"invalid_state_evidence_citation:{name}"
        return None

    def maybe_update_evolving_states(self, simulation: Any) -> None:
        """Update bounded transient state at fixed, condition-blind checkpoints."""

        if not self.evolving_state_enabled:
            return
        timestep = int(simulation.timestep)
        if timestep < self._next_state_checkpoint:
            return
        if timestep != self._next_state_checkpoint:
            raise RuntimeError(
                "Simulation skipped a configured Part 3 state checkpoint"
            )
        requests: list[EvolvingStateUpdateRequest] = []
        run_key = self._state_evidence_id(
            "|".join(
                (
                    str(simulation.scenario_mode),
                    str(simulation.condition_spec.name),
                    str(simulation.random_seed),
                    str(self.sampling_replication_id),
                )
            )
        ).removeprefix("state-evidence-")
        for agent in simulation.staff_agents:
            agent_id = int(agent.gid)
            evidence = self._state_evidence_for_agent(
                simulation, agent_id, self._last_state_checkpoint, timestep
            )
            if not evidence:
                self.evolving_state_log.append(
                    {
                        "checkpoint_id": (
                            f"state:{run_key}:{timestep}:agent:{agent_id}:"
                            f"{self.persona_by_agent[agent_id]}"
                        ),
                        "persona_id": self.persona_by_agent[agent_id],
                        "agent_id": agent_id,
                        "role": str(agent.role),
                        "timestep": timestep,
                        "window_start": self._last_state_checkpoint,
                        "prior_state": dict(self.evolving_state_by_agent[agent_id]),
                        "state": dict(self.evolving_state_by_agent[agent_id]),
                        "evidence_by_dimension": {
                            name: [] for name in EVOLVING_STATE_DIMENSIONS
                        },
                        "source_evidence_ids_by_dimension": {
                            name: [] for name in EVOLVING_STATE_DIMENSIONS
                        },
                        "policy_name": "deterministic_no_new_evidence",
                        "was_fallback": False,
                        "rejected_reason": None,
                        "model_called": False,
                    }
                )
                self.stats["state_no_evidence_carry_forward"] += 1
                continue
            requests.append(
                EvolvingStateUpdateRequest(
                    checkpoint_id=(
                        f"state:{run_key}:{timestep}:agent:{agent_id}:"
                        f"{self.persona_by_agent[agent_id]}"
                    ),
                    persona_id=self.persona_by_agent[agent_id],
                    agent_id=agent_id,
                    role=str(agent.role),
                    timestep=timestep,
                    window_start=self._last_state_checkpoint,
                    prior_state=dict(self.evolving_state_by_agent[agent_id]),
                    evidence=evidence,
                )
            )
        if requests:
            try:
                updates = self.provider.update_states_many(requests)
            except Exception as error:
                self.provider_error_log.append(
                    {
                        "checkpoint_id": f"state_batch:{timestep}",
                        "timestep": timestep,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                self.stats["state_provider_exceptions"] += 1
                raise
            if len(updates) != len(requests):
                raise RuntimeError("State provider returned the wrong cardinality")
            for request, update in zip(requests, updates):
                error = self._state_validation_error(request, update)
                if error is not None:
                    raise RuntimeError(
                        f"Invalid evolving-state update {request.checkpoint_id}: {error}"
                    )
                prior_state = dict(request.prior_state)
                new_state = {name: int(update.state[name]) for name in EVOLVING_STATE_DIMENSIONS}
                self.evolving_state_by_agent[request.agent_id] = new_state
                source_ids_by_dimension = {}
                for name in EVOLVING_STATE_DIMENSIONS:
                    cited = set(update.evidence_by_dimension[name])
                    source_ids_by_dimension[name] = [
                        str(row["source_evidence_id"])
                        for row in request.evidence
                        if str(row["evidence_id"]) in cited
                    ]
                self.evolving_state_log.append(
                    {
                        **update.as_dict(),
                        "agent_id": request.agent_id,
                        "role": request.role,
                        "timestep": request.timestep,
                        "window_start": request.window_start,
                        "prior_state": prior_state,
                        "source_evidence_ids_by_dimension": source_ids_by_dimension,
                        "model_called": True,
                    }
                )
                self.stats["state_provider_calls"] += 1
        self._last_state_checkpoint = timestep
        self._next_state_checkpoint += self.evolving_state_interval_seconds
        self.stats["state_checkpoint_count"] += 1

    @staticmethod
    def _validate_grounded_memory(memory_rows: Sequence[Mapping[str, Any]]) -> None:
        for row in memory_rows:
            missing = _GROUNDED_MEMORY_REQUIRED_FIELDS - set(row)
            prohibited = _PROHIBITED_DECISION_HISTORY_FIELDS & set(row)
            if missing or prohibited:
                raise ValueError(
                    "Invalid Part 3 grounded memory record: "
                    f"missing={sorted(missing)}, prohibited={sorted(prohibited)}"
                )
            if (
                row.get("event_type") != "communicative_interaction"
                or row.get("outcome") != "realized"
                or row.get("source") != "realized_communicative_interaction"
                or row.get("memory_policy") != PART3_GROUNDED_MEMORY_POLICY
                or not row.get("source_event_id")
                or int(row.get("seconds_ago", -1)) < 0
            ):
                raise ValueError("Part 3 causal memory is not a realized interaction")

    def _fallback(
        self, request: CausalDecisionRequest, reason: str
    ) -> CausalDecision:
        action = request.rule_reference_action
        if action not in request.feasible_actions:
            action = request.feasible_actions[0]
        return CausalDecision(
            decision_id=request.decision_id,
            evidence_id=request.evidence_id,
            persona_id=request.persona_id,
            selected_action=action,
            selected_reason=request.allowed_reasons[0],
            topic_family=request.allowed_topic_families[0],
            evidence_ids=(request.evidence_id,),
            policy_name=getattr(self.provider, "policy_name", "unknown"),
            was_fallback=True,
            rejected_reason=reason,
        )

    @staticmethod
    def _validation_error(
        request: CausalDecisionRequest, decision: CausalDecision
    ) -> str | None:
        if decision.decision_id != request.decision_id:
            return "decision_id_mismatch"
        if decision.evidence_id != request.evidence_id:
            return "evidence_id_mismatch"
        if decision.persona_id != request.persona_id:
            return "persona_id_mismatch"
        if decision.selected_action not in request.feasible_actions:
            return "infeasible_action"
        if decision.selected_reason not in request.allowed_reasons:
            return "unbounded_reason"
        if decision.topic_family not in request.allowed_topic_families:
            return "unbounded_topic_family"
        if decision.evidence_ids != (request.evidence_id,):
            return "invalid_evidence_citation"
        return None

    def _sampling_key(self, episode: Mapping[str, Any]) -> str:
        """Build a condition-invariant key for matched opportunity sampling."""

        metadata = episode["metadata"]
        decision_context = episode["decision_context"]
        return "|".join(
            (
                str(metadata.get("scenario_mode")),
                str(metadata.get("seed")),
                str(self.sampling_replication_id),
                str(episode.get("timestep")),
                str(metadata.get("staff_id")),
                str(metadata.get("partner_id")),
                str(decision_context.get("interaction_type")),
                str(decision_context.get("abm_reason_type")),
            )
        )

    @staticmethod
    def _sampling_value(sampling_key: str) -> float:
        """Map a matched-opportunity key to [0, 1) without ABM RNG."""

        integer = int.from_bytes(
            hashlib.sha256(sampling_key.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        return integer / float(1 << 64)

    def decide_episode(self, episode: Mapping[str, Any]) -> CausalDecision:
        metadata = episode["metadata"]
        decision_context = episode["decision_context"]
        rule_reference = episode["rule_reference"]
        agent_id = int(metadata["staff_id"])
        persona_id = self.persona_by_agent[agent_id]
        evidence_id = str(episode["evidence_id"])
        evidence = deepcopy(dict(episode))
        memory_state = list(evidence.get("memory_state_before") or [])
        self._validate_grounded_memory(memory_state)
        evidence["memory_state_before"] = memory_state
        evidence["evolving_state_before"] = dict(
            self.evolving_state_by_agent[agent_id]
        )
        bounds = categorical_decision_bounds(evidence)
        request = CausalDecisionRequest(
            decision_id=f"closed-loop:{evidence_id}:{persona_id}",
            evidence_id=evidence_id,
            persona_id=persona_id,
            agent_id=agent_id,
            timestep=int(episode["timestep"]),
            feasible_actions=tuple(decision_context["feasible_actions"]),
            allowed_reasons=tuple(bounds["allowed_reasons"]),
            allowed_topic_families=tuple(bounds["allowed_topic_families"]),
            rule_reference_action=str(rule_reference["selected_action"]),
            evidence=evidence,
        )
        cap_reason = None
        window_index = (
            max(0, (request.timestep - self.start_seconds) // self.decision_window_seconds)
            if self.decision_window_seconds
            else 0
        )
        sampling_key = self._sampling_key(episode)
        sampling_value = self._sampling_value(sampling_key)
        sampled_for_model = sampling_value < self.model_decision_sample_rate
        self.stats["eligible_decisions"] += 1
        self._eligible_by_agent[agent_id] += 1
        self._eligible_by_window[window_index] += 1
        if sampled_for_model:
            self.stats["sampled_candidates"] += 1
            self._sampled_by_agent[agent_id] += 1
            self._sampled_by_window[window_index] += 1

        if not sampled_for_model:
            cap_reason = "deterministic_sampling_skip"
        elif self.stats["provider_calls"] >= self.max_decisions_per_run:
            cap_reason = "run_decision_cap"
        elif self._calls_by_agent[agent_id] >= self.max_decisions_per_agent:
            cap_reason = "agent_decision_cap"
        elif (
            self.max_decisions_per_window
            and self._calls_by_window[window_index] >= self.max_decisions_per_window
        ):
            cap_reason = "window_decision_cap"

        if cap_reason:
            decision = self._fallback(request, cap_reason)
            self.stats[cap_reason] += 1
        else:
            self._calls_by_agent[agent_id] += 1
            self._calls_by_window[window_index] += 1
            try:
                supplied = self.provider.decide_many([request])
                if len(supplied) != 1:
                    decision = self._fallback(request, "provider_cardinality_error")
                else:
                    error = self._validation_error(request, supplied[0])
                    decision = (
                        self._fallback(request, error)
                        if error is not None
                        else supplied[0]
                    )
            except Exception as error:  # Fail closed; verifier treats this as fatal.
                decision = self._fallback(request, "provider_exception")
                self.provider_error_log.append(
                    {
                        "decision_id": request.decision_id,
                        "evidence_id": request.evidence_id,
                        "agent_id": agent_id,
                        "timestep": request.timestep,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                self.stats["provider_exceptions"] += 1
            self.stats["provider_calls"] += 1

        record = {
            **decision.as_dict(),
            "agent_id": agent_id,
            "timestep": request.timestep,
            "condition": metadata.get("condition"),
            "scenario": metadata.get("scenario_mode"),
            "seed": metadata.get("seed"),
            "role": metadata.get("role"),
            "partner_id": metadata.get("partner_id"),
            "partner_role": metadata.get("partner_role"),
            "zone": episode.get("spatial_context", {}).get("zone"),
            "interaction_type": decision_context.get("interaction_type"),
            "abm_reason_type": decision_context.get("abm_reason_type"),
            "allowed_reasons": list(request.allowed_reasons),
            "allowed_topic_families": list(request.allowed_topic_families),
            "patient_context_present": (
                episode.get("workflow_context", {}).get("patient_id")
                not in (None, -1, "-1")
            ),
            "sampling_method": (
                "deterministic_sha256_matched_opportunity_threshold_v1"
            ),
            "sampling_replication_id": self.sampling_replication_id,
            "sampling_key": sampling_key,
            "model_decision_sample_rate": self.model_decision_sample_rate,
            "sampling_value": sampling_value,
            "sampled_for_model": sampled_for_model,
            "decision_output_contract": "categorical_causal_v1",
            "free_text_used_as_causal_input": False,
            "evolving_state_before": dict(
                self.evolving_state_by_agent[agent_id]
            ),
        }
        self.decision_log.append(record)
        if not decision.was_fallback:
            for memory_row in memory_state:
                self.memory_exposures_by_agent[agent_id].append(
                    {
                        "decision_id": request.decision_id,
                        "decision_evidence_id": evidence_id,
                        **dict(memory_row),
                    }
                )
        self.stats[f"action_{decision.selected_action}"] += 1
        self.stats["fallbacks"] += int(decision.was_fallback)
        return decision

    def summary(self) -> dict[str, Any]:
        eligible_count = int(self.stats.get("eligible_decisions", 0))
        sampled_count = int(self.stats.get("sampled_candidates", 0))
        memory_rows = [
            row
            for rows in self.memory_exposures_by_agent.values()
            for row in rows
        ]
        return {
            "decision_count": len(self.decision_log),
            "memory_count": sum(
                len(rows) for rows in self.memory_exposures_by_agent.values()
            ),
            "unique_grounded_memory_event_count": len(
                {str(row["memory_id"]) for row in memory_rows}
            ),
            "memory_policy": PART3_GROUNDED_MEMORY_POLICY,
            "decision_history_used_as_memory": False,
            "decision_history_used_in_bounded_state": self.evolving_state_enabled,
            "generated_reflection_used_as_causal_input": False,
            "bounded_evolving_state_used_as_causal_input": self.evolving_state_enabled,
            "evolving_state_enabled": self.evolving_state_enabled,
            "evolving_state_contract": (
                "evidence_linked_bounded_state_v1"
                if self.evolving_state_enabled
                else None
            ),
            "evolving_state_interval_seconds": self.evolving_state_interval_seconds,
            "evolving_state_checkpoint_count": int(
                self.stats.get("state_checkpoint_count", 0)
            ),
            "evolving_state_update_count": len(self.evolving_state_log),
            "evolving_state_by_agent": {
                str(agent_id): dict(state)
                for agent_id, state in sorted(self.evolving_state_by_agent.items())
            },
            "stats": dict(sorted(self.stats.items())),
            "free_text_used_as_causal_input": False,
            "start_seconds": self.start_seconds,
            "max_decisions_per_run": self.max_decisions_per_run,
            "max_decisions_per_agent": self.max_decisions_per_agent,
            "decision_window_seconds": self.decision_window_seconds,
            "max_decisions_per_window": self.max_decisions_per_window,
            "sampling_method": (
                "deterministic_sha256_matched_opportunity_threshold_v1"
            ),
            "model_decision_sample_rate": self.model_decision_sample_rate,
            "sampling_replication_id": self.sampling_replication_id,
            "eligible_decision_count": eligible_count,
            "sampled_candidate_count": sampled_count,
            "observed_sample_rate": (
                sampled_count / eligible_count if eligible_count else 0.0
            ),
            "eligible_decisions_by_window": {
                str(index): count
                for index, count in sorted(self._eligible_by_window.items())
            },
            "sampled_candidates_by_window": {
                str(index): count
                for index, count in sorted(self._sampled_by_window.items())
            },
            "provider_calls_by_window": {
                str(index): count
                for index, count in sorted(self._calls_by_window.items())
            },
            "eligible_decisions_by_agent": {
                str(agent_id): count
                for agent_id, count in sorted(self._eligible_by_agent.items())
            },
            "sampled_candidates_by_agent": {
                str(agent_id): count
                for agent_id, count in sorted(self._sampled_by_agent.items())
            },
            "persona_by_agent": {
                str(agent_id): persona_id
                for agent_id, persona_id in sorted(self.persona_by_agent.items())
            },
            "provider_error_count": len(self.provider_error_log),
        }
