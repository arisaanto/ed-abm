#!/usr/bin/env python3
"""Build bounded-generative Part 3 appraisals from completed run evidence.

This script never runs the ABM or an LLM. For the scientific design it selects
one deterministic, four-condition-paired staff shift for every
scenario/persona/seed combination, balances staff roles across the ten seeds,
and writes one fixed survey bundle plus one interview bundle for each shift.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.cognitive_policy import estimate_context_tokens
from src.interviews import (
    APPRAISAL_CLAIM_LAYERS,
    APPRAISAL_LANGUAGE_RULES,
    INTERVIEW_QUESTIONS,
    required_output_schema,
)
from src.personas import PERSONA_APPRAISAL_DIMENSIONS, default_cognitive_personas
from src.vllm_backend import packet_json_schema


APPRAISAL_SYSTEM_MESSAGE = (
    "You are producing a bounded synthetic appraisal of one simulated Emergency "
    "Department staff shift. Write naturally in the first person, but never imply "
    "that this is real staff testimony. Event claims must come only from supplied "
    "evidence. Keep observed patterns, persona-conditioned interpretation, and design "
    "conjecture epistemically separate. In public-facing answers, respond directly "
    "without repeating the question or announcing the scenario or condition label. "
    "Return only schema-valid JSON."
)

SURVEY_ANCHORS = {
    "team_awareness": "1 = colleagues were difficult to locate or reach; 7 = colleagues were readily available when needed",
    "interruption_burden": "1 = little experienced interruption burden; 7 = heavy experienced interruption burden",
    "task_continuity": "1 = work felt repeatedly fragmented; 7 = work remained continuous",
    "patient_accessibility": "1 = patient-facing destinations felt difficult to reach or support; 7 = they felt readily accessible",
    "privacy_and_control": "1 = little control over exposure and contact; 7 = strong control over exposure and contact",
    "spatial_legibility": "1 = it was difficult to understand where work and colleagues were available; 7 = those patterns were clear",
    "overall_person_space_fit": "1 = poor fit with this orientation's work priorities; 7 = strong fit",
}

EVENT_TYPE_ORDER = (
    "interaction",
    "cognitive_opportunity_decision",
    "missed_opportunity",
    "workflow_event",
    "travel_episode",
    "visibility_episode",
    "task_context_transition",
    "zone_transition",
    "mode_transition",
)

PHASE_ORDER = (
    "early in the shift",
    "mid-shift",
    "later in the shift",
)

MAIN_STUDY_SEEDS = tuple(range(1, 11))

INTERNAL_LANGUAGE = (
    "log",
    "metric",
    "dataset",
    "evidence id",
    "agent-based model",
    "abm",
    "persona",
    "condition",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    )


def _digest(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_agent_shifts(results_root: Path) -> list[dict[str, Any]]:
    map_paths = sorted(results_root.rglob("part3_spatial_experience.jsonl"))
    if not map_paths:
        raise FileNotFoundError(
            f"No part3_spatial_experience.jsonl files found under {results_root}"
        )
    shifts = []
    for map_path in map_paths:
        run_dir = map_path.parent
        event_path = run_dir / "part3_experience_events.jsonl"
        summary_path = run_dir / "part3_agent_experience.jsonl"
        accumulated_experience_path = run_dir / "part3_evolving_state.jsonl"
        if (
            not event_path.exists()
            or not summary_path.exists()
            or not accumulated_experience_path.exists()
        ):
            raise FileNotFoundError(
                f"Incomplete Part 3 experience evidence in {run_dir}"
            )
        summaries = {
            int(row["agent_id"]): row for row in _read_jsonl(summary_path)
        }
        for experience_map in _read_jsonl(map_path):
            agent_id = int(experience_map["agent_id"])
            summary = summaries.get(agent_id)
            if summary is None:
                raise ValueError(
                    f"Map for agent {agent_id} has no aggregate summary in {run_dir}"
                )
            if experience_map.get("generated_interpretation_present") is not False:
                raise ValueError(f"Experience map already contains interpretation: {run_dir}")
            if experience_map.get("not_human_data") is not True:
                raise ValueError(f"Experience map lacks not-human marker: {run_dir}")
            shifts.append(
                {
                    "run_dir": str(run_dir),
                    "map": experience_map,
                    "summary": summary,
                    "event_path": str(event_path),
                    "accumulated_experience_path": str(
                        accumulated_experience_path
                    ),
                }
            )
    return shifts


def _load_selected_agent_events(shift: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Stream one selected staff member's events without retaining every shift."""

    agent_id = int(shift["map"]["agent_id"])
    events = []
    event_path = Path(str(shift["event_path"]))
    with event_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {event_path}:{line_number}"
                ) from exc
            if int(row["agent_id"]) == agent_id:
                events.append(row)
    return events


def _load_selected_accumulated_experience(
    shift: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load the five evidence-linked experience checkpoints for one staff member."""

    agent_id = int(shift["map"]["agent_id"])
    rows = [
        row
        for row in _read_jsonl(Path(str(shift["accumulated_experience_path"])))
        if int(row["agent_id"]) == agent_id
    ]
    return sorted(rows, key=lambda row: int(row["timestep"]))


def _select_balanced_shifts(
    shifts: list[dict[str, Any]],
    *,
    replicates_per_stratum: int,
    allow_incomplete_strata: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if replicates_per_stratum < 1:
        raise ValueError("replicates_per_stratum must be at least 1")
    personas = [persona.persona_id for persona in default_cognitive_personas()]
    roles = ("CoordinationNurse", "Nurse", "Doctor")
    scenarios = ("normal_load", "high_load_high_acuity")
    conditions = ("baseline", "cockpit_only", "nursta_only", "both")
    if allow_incomplete_strata:
        # Structural preflights contain only a few runs and cannot establish
        # four-condition pairing. Exercise packet construction without
        # pretending that this is an inferential sample.
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for shift in shifts:
            row = shift["map"]
            grouped[
                (
                    str(row["scenario"]),
                    str(row["condition"]),
                    str(row["persona_id"]),
                    str(row["role"]),
                )
            ].append(shift)
        selected = []
        for key in sorted(grouped):
            candidates = sorted(
                grouped[key],
                key=lambda row: _digest(
                    row["map"]["run_id"], row["map"]["agent_id"], "appraisal"
                ),
            )
            selected.extend(candidates[:replicates_per_stratum])
        return selected, {
            "available_agent_shift_count": len(shifts),
            "selected_agent_shift_count": len(selected),
            "observed_stratum_count": len(grouped),
            "replicates_per_stratum": replicates_per_stratum,
            "paired_condition_sampling": False,
            "incomplete_strata_allowed_for_smoke_only": True,
            "selection_policy": (
                "sha256-stable independent smoke sample; not an inferential design"
            ),
            "missing_strata": [],
            "short_strata": [],
        }

    expected_seed_strata = {
        (scenario, persona, seed)
        for scenario in scenarios
        for persona in personas
        for seed in MAIN_STUDY_SEEDS
    }
    pair_units: dict[
        tuple[str, int, int, str, str, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for shift in shifts:
        row = shift["map"]
        pair_key = (
            str(row["scenario"]),
            int(row["seed"]),
            int(row.get("assignment_round", 0)),
            str(row["persona_id"]),
            str(row["role"]),
            int(row["agent_id"]),
        )
        condition = str(row["condition"])
        if condition in pair_units[pair_key]:
            raise ValueError(f"Duplicate appraisal pair member: {pair_key}/{condition}")
        pair_units[pair_key][condition] = shift

    required_conditions = set(conditions)
    incomplete_pairs = [
        {"pair_key": list(key), "conditions": sorted(members)}
        for key, members in sorted(pair_units.items())
        if set(members) != required_conditions
    ]
    complete_pairs = {
        key: members
        for key, members in pair_units.items()
        if set(members) == required_conditions
    }
    paired_by_seed: dict[
        tuple[str, str, int],
        list[
            tuple[
                tuple[str, int, int, str, str, int],
                dict[str, dict[str, Any]],
            ]
        ],
    ] = defaultdict(list)
    for key, members in complete_pairs.items():
        scenario, seed, _round, persona, _role, _agent_id = key
        paired_by_seed[(scenario, persona, seed)].append((key, members))

    missing = sorted(expected_seed_strata - set(paired_by_seed))
    selected_pairs = []
    role_counts: dict[str, Counter[str]] = {}
    missing_role_candidates = []
    for scenario in scenarios:
        for persona in personas:
            # Every scenario/persona uses every seed. Roles are allocated 4/3/3;
            # the role receiving the fourth seed rotates deterministically.
            extra_role_index = int(_digest(scenario, persona, "extra_role")[:8], 16) % len(roles)
            role_slots = list(roles) * 3 + [roles[extra_role_index]]
            role_slots = [
                role
                for _rank, role in sorted(
                    (
                        _digest(scenario, persona, index, role, "role_schedule"),
                        role,
                    )
                    for index, role in enumerate(role_slots)
                )
            ]
            counts: Counter[str] = Counter()
            for seed, target_role in zip(MAIN_STUDY_SEEDS, role_slots):
                seed_key = (scenario, persona, seed)
                candidates = [
                    item
                    for item in paired_by_seed.get(seed_key, [])
                    if item[0][4] == target_role
                ]
                candidates.sort(
                    key=lambda item: _digest(*item[0], "all_seed_paired_appraisal")
                )
                if not candidates:
                    missing_role_candidates.append(
                        {
                            "seed_stratum": list(seed_key),
                            "required_role": target_role,
                        }
                    )
                    continue
                selected_pairs.append(candidates[0])
                counts[target_role] += 1
            role_counts[f"{scenario}|{persona}"] = counts
    if missing or missing_role_candidates or incomplete_pairs:
        raise ValueError(
            "Appraisal evidence does not satisfy the all-seed paired design: "
            f"missing_seed_strata={len(missing)}, "
            f"missing_role_candidates={len(missing_role_candidates)}, "
            f"incomplete_pairs={len(incomplete_pairs)}"
        )

    selected = [
        members[condition]
        for _key, members in selected_pairs
        for condition in conditions
    ]
    return selected, {
        "available_agent_shift_count": len(shifts),
        "selected_agent_shift_count": len(selected),
        "expected_seed_stratum_count": len(expected_seed_strata),
        "observed_seed_stratum_count": len(expected_seed_strata - set(missing)),
        "missing_strata": [list(row) for row in missing],
        "missing_role_candidates": missing_role_candidates,
        "incomplete_pair_count": len(incomplete_pairs),
        "selected_paired_unit_count": len(selected_pairs),
        "selected_seed_count_per_scenario_persona": len(MAIN_STUDY_SEEDS),
        "conditions_per_paired_unit": len(conditions),
        "paired_condition_sampling": True,
        "pair_key": "scenario-seed-assignment_round-persona-role-agent_id",
        "role_counts_by_scenario_persona": {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(role_counts.items())
        },
        "selection_policy": (
            "all ten seeds retained per scenario/persona; one sha256-stable "
            "four-condition pair per seed with deterministic 4/3/3 role balance"
        ),
    }


def _phase(timestep: int, start: int, duration: int) -> str:
    fraction = (int(timestep) - int(start)) / max(int(duration), 1)
    if fraction < 1 / 3:
        return "early in the shift"
    if fraction < 2 / 3:
        return "mid-shift"
    return "later in the shift"


def _band(value: float, *, low: float, high: float) -> str:
    if value <= low:
        return "limited"
    if value >= high:
        return "substantial"
    return "moderate"


def _event_is_informative(row: Mapping[str, Any]) -> bool:
    """Exclude within-area movement snippets that add no experiential context."""

    if row.get("event_type") != "travel_episode":
        return True
    start = str(row.get("start_place_label", ""))
    end = str(row.get("end_place_label", ""))
    if start and end and start != end:
        return True
    return (
        float(row.get("distance_meters", 0.0)) >= 2.0
        or int(row.get("duration_seconds", 0)) >= 5
    )


def _humanize_role(value: Any) -> str:
    role = str(value or "colleague")
    role = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", role).replace("_", " ")
    return role.lower()


def _humanize_event_summary(row: Mapping[str, Any], summary: str) -> str:
    """Remove implementation labels while preserving the underlying event."""

    value = summary.replace("coordinationnurse", "coordination nurse")
    value = value.replace(
        "because part3 cognitive defer",
        "because the contact was deferred after weighing timing and relevance",
    )
    value = value.replace(
        "because part3 cognitive decline",
        "because the contact was declined after weighing timing and relevance",
    )
    value = value.replace(
        "because task transition update not in close physical contact",
        "because the colleague was not close enough during a task transition",
    )
    if row.get("event_type") == "cognitive_opportunity_decision":
        partner = _humanize_role(row.get("partner_role"))
        place = str(row.get("place", {}).get("label", "the clinical area"))
        action = str(row.get("selected_action", "responded")).replace("_", " ")
        action_phrase = {
            "engage": "went ahead",
            "defer": "was deferred",
            "decline": "was declined",
            "reject": "was declined",
        }.get(action, f"was handled as {action}")
        value = f"An optional contact with a {partner} in {place} {action_phrase}."
    return value


def _select_events(
    experience_map: Mapping[str, Any],
    events: list[dict[str, Any]],
    *,
    limit: int,
    window_start: int,
    window_duration: int,
) -> list[dict[str, Any]]:
    """Select temporally balanced, type-diverse evidence cards.

    The spatial map's salient-event list is preferred, but salience alone tends
    to favor later-shift interactions after experience has accumulated. Fill
    equal early/middle/late quotas from the complete grounded event stream so a
    synthetic appraisal cannot mistake the end of a shift for the whole shift.
    """

    if limit < 1:
        return []
    by_id = {str(row["event_id"]): row for row in events}
    salient_ids = [
        str(value)
        for value in experience_map.get("salient_event_ids", [])
        if str(value) in by_id
    ]
    salient_rank = {event_id: rank for rank, event_id in enumerate(salient_ids)}
    event_type_rank = {
        event_type: rank for rank, event_type in enumerate(EVENT_TYPE_ORDER)
    }

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        event_id = str(row["event_id"])
        return (
            0 if event_id in salient_rank else 1,
            salient_rank.get(event_id, len(salient_rank)),
            event_type_rank.get(str(row.get("event_type")), len(EVENT_TYPE_ORDER)),
            int(row.get("timestep", 0)),
            event_id,
        )

    phase_rows: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in PHASE_ORDER
    }
    for row in events:
        if not _event_is_informative(row):
            continue
        phase_rows[
            _phase(int(row.get("timestep", 0)), window_start, window_duration)
        ].append(row)
    for rows in phase_rows.values():
        rows.sort(key=sort_key)

    base_quota, remainder = divmod(limit, len(PHASE_ORDER))
    phase_quotas = {
        phase: base_quota + (1 if index < remainder else 0)
        for index, phase in enumerate(PHASE_ORDER)
    }
    selected_ids: list[str] = []
    for phase in PHASE_ORDER:
        candidates = phase_rows[phase]
        phase_selected: list[str] = []
        for event_type in EVENT_TYPE_ORDER:
            candidate = next(
                (
                    str(row["event_id"])
                    for row in candidates
                    if row.get("event_type") == event_type
                    and str(row["event_id"]) not in phase_selected
                ),
                None,
            )
            if candidate is not None:
                phase_selected.append(candidate)
            if len(phase_selected) >= phase_quotas[phase]:
                break
        for row in candidates:
            event_id = str(row["event_id"])
            if event_id not in phase_selected:
                phase_selected.append(event_id)
            if len(phase_selected) >= phase_quotas[phase]:
                break
        selected_ids.extend(phase_selected[: phase_quotas[phase]])

    if len(selected_ids) < limit:
        for row in sorted(events, key=sort_key):
            if not _event_is_informative(row):
                continue
            event_id = str(row["event_id"])
            if event_id not in selected_ids:
                selected_ids.append(event_id)
            if len(selected_ids) >= limit:
                break

    selected = [by_id[event_id] for event_id in selected_ids[:limit]]
    return sorted(
        selected,
        key=lambda row: (int(row.get("timestep", 0)), str(row["event_id"])),
    )


def _visible_experience(
    shift: Mapping[str, Any], *, max_events: int
) -> tuple[dict[str, Any], dict[str, str]]:
    experience_map = shift["map"]
    summary = shift["summary"]
    window_start = int(summary.get("window_start_seconds", 0))
    window_duration = int(summary.get("window_duration_seconds", 1))
    selected = _select_events(
        experience_map,
        shift["events"],
        limit=max_events,
        window_start=window_start,
        window_duration=window_duration,
    )
    id_map = {
        str(row["event_id"]): f"shift_event_{index:02d}"
        for index, row in enumerate(selected, start=1)
    }
    cards = []
    for row in selected:
        public_id = id_map[str(row["event_id"])]
        event_summary = str(row.get("event_summary", "A work event occurred."))
        if (
            row.get("event_type") == "travel_episode"
            and row.get("start_place_label")
            and row.get("start_place_label") == row.get("end_place_label")
        ):
            distance = float(row.get("distance_meters", 0.0))
            duration = int(row.get("duration_seconds", 0))
            duration_unit = "second" if duration == 1 else "seconds"
            event_summary = (
                f"Moved {distance:.1f} m within {row['start_place_label']} over "
                f"{duration} {duration_unit}."
            )
        event_summary = _humanize_event_summary(row, event_summary)
        kind = str(row.get("event_type", "event")).replace("_", " ")
        if kind == "cognitive opportunity decision":
            kind = "optional contact decision"
        card = {
            "evidence_id": public_id,
            "when": _phase(row.get("timestep", 0), window_start, window_duration),
            "where": str(row.get("place", {}).get("label", "clinical area")),
            "kind": kind,
            "what_happened": event_summary,
        }
        if row.get("partner_role"):
            card["colleague_role"] = _humanize_role(row["partner_role"])
        if row.get("current_task"):
            card["task_context"] = str(row["current_task"])
        cards.append(card)

    place_nodes_by_label: dict[str, dict[str, float]] = {}
    for node in experience_map.get("place_nodes", []):
        label = str(node.get("label", "clinical area"))
        aggregate = place_nodes_by_label.setdefault(
            label,
            {"dwell_seconds": 0.0, "interaction_count": 0.0, "visibility_seconds": 0.0},
        )
        aggregate["dwell_seconds"] += float(node.get("dwell_seconds", 0))
        aggregate["interaction_count"] += int(node.get("interaction_count", 0))
        aggregate["visibility_seconds"] += float(node.get("visibility_seconds", 0))
    place_nodes = [
        {"label": label, **values}
        for label, values in place_nodes_by_label.items()
    ]
    max_dwell = max((float(node.get("dwell_seconds", 0)) for node in place_nodes), default=1.0)
    place_patterns = []
    for node in place_nodes:
        dwell = float(node.get("dwell_seconds", 0))
        interactions = int(node.get("interaction_count", 0))
        visibility = float(node.get("visibility_seconds", 0))
        if not dwell and not interactions and not visibility:
            continue
        place_patterns.append(
            {
                "place": str(node.get("label", "clinical area")),
                "presence": _band(dwell / max(max_dwell, 1.0), low=0.12, high=0.55),
                "contact_activity": _band(interactions, low=1, high=5),
                "colleague_visibility": _band(visibility, low=30, high=600),
            }
        )

    transition_counts: dict[tuple[str, str], float] = defaultdict(float)
    for edge in experience_map.get("transition_edges", []):
        from_place = str(edge.get("from_place", "clinical area"))
        to_place = str(edge.get("to_place", "clinical area"))
        if from_place == to_place:
            continue
        transition_counts[(from_place, to_place)] += float(edge.get("trip_count", 0))
    transition_patterns = []
    for (from_place, to_place), trip_count in sorted(
        transition_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        transition_patterns.append(
            {
                "from": from_place,
                "to": to_place,
                "frequency": _band(trip_count, low=1, high=4),
            }
        )

    visible = {
        "staff_role": str(experience_map["role"]),
        "spatial_features": list(experience_map.get("spatial_features", [])),
        "place_experience": place_patterns,
        "recurring_transitions": transition_patterns[:12],
        "representative_shift_events": cards,
        "scope_note": (
            "This is a sparse reconstruction of one simulated shift. Absence from "
            "these cards is not proof that an event never occurred."
        ),
    }
    experience_checkpoints = []
    for row in shift.get("accumulated_experience", []):
        values = row["state"]
        experience_checkpoints.append(
            {
                "hours_into_evaluated_shift": round(
                    (int(row["timestep"]) - window_start) / 3600, 1
                ),
                "unmet_coordination_need": int(values["coordination_need"]),
                "interruption_strain": int(values["interruption_strain"]),
                "task_continuity": int(values["task_continuity"]),
                "team_support": int(values["team_support"]),
            }
        )
    visible["accumulated_experience"] = {
        "scale": (
            "Each measure ranges from -2 to +2. Zero means that no positive or "
            "negative effect had accumulated at that checkpoint."
        ),
        "meanings": {
            "unmet_coordination_need": (
                "higher values mean stronger unmet coordination needs"
            ),
            "interruption_strain": (
                "higher values mean greater strain from optional interruptions"
            ),
            "task_continuity": (
                "higher values mean work maintained stronger continuity"
            ),
            "team_support": (
                "higher values mean stronger support from recent colleague contact"
            ),
        },
        "checkpoints": experience_checkpoints,
    }
    return visible, id_map


def _orientation_text(profile: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(profile["summary"]),
            *[str(value) for value in profile["decision_principles"]],
            "Reflection style: " + str(profile["reflection_voice"]),
        ]
    )


def _base_packet(
    shift: Mapping[str, Any],
    *,
    packet_type: str,
    visible: Mapping[str, Any],
    id_map: Mapping[str, str],
) -> dict[str, Any]:
    experience_map = shift["map"]
    personas = {row.persona_id: row for row in default_cognitive_personas()}
    persona = personas[str(experience_map["persona_id"])]
    profile = persona.prompt_profile()
    public_ids = list(id_map.values())
    stable_id = _digest(
        experience_map["run_id"], experience_map["agent_id"], packet_type
    )[:20]
    return {
        "prompt_id": f"appraisal_{packet_type}_{stable_id}",
        "packet_type": packet_type,
        "system_message": APPRAISAL_SYSTEM_MESSAGE,
        "role": str(experience_map["role"]),
        "user_model_profile": profile,
        "trace_evidence": {
            "metadata": {
                "run_id": experience_map["run_id"],
                "scenario_mode": experience_map["scenario"],
                "condition": experience_map["condition"],
                "seed": experience_map["seed"],
                "agent_id": experience_map["agent_id"],
            },
            "source_event_id_map": dict(id_map),
            "source_experience_map": experience_map,
            "source_agent_summary": shift["summary"],
        },
        "llm_visible_evidence": dict(visible),
        "evidence_ids": public_ids,
        "hard_constraints": [
            *APPRAISAL_LANGUAGE_RULES,
            *APPRAISAL_CLAIM_LAYERS.values(),
        ],
        "fixture_only_not_scientific_data": False,
        "synthetic_design_probe_not_human_data": True,
        "source_is_simulated_experience": True,
    }


def _estimate_model_visible_tokens(packet: Mapping[str, Any]) -> int:
    """Estimate the actual chat payload, excluding evaluator-only provenance."""

    schema = packet_json_schema(str(packet["packet_type"]), packet)
    return estimate_context_tokens(
        {
            "system_message": packet["system_message"],
            "user_message": packet["user_message"],
            "response_schema": schema,
            "format_instruction": (
                "Return one JSON object only. Cite only supplied evidence IDs."
            ),
        }
    )


def _survey_packet(
    shift: Mapping[str, Any], visible: Mapping[str, Any], id_map: Mapping[str, str]
) -> dict[str, Any]:
    packet = _base_packet(
        shift,
        packet_type="end_of_shift_survey_bundle",
        visible=visible,
        id_map=id_map,
    )
    profile = packet["user_model_profile"]
    packet["expected_dimensions"] = list(PERSONA_APPRAISAL_DIMENSIONS)
    packet["survey_anchors"] = SURVEY_ANCHORS
    packet["required_output_schema"] = required_output_schema(packet["packet_type"])
    packet["user_message"] = "\n".join(
        [
            "Complete one end-of-shift appraisal for every listed dimension.",
            "Apply this designed workplace orientation silently: " + _orientation_text(profile),
            "Use these score anchors: " + json.dumps(SURVEY_ANCHORS, sort_keys=True),
            "Use insufficient_evidence with a null score rather than filling gaps from the orientation.",
            (
                "Keep each rationale to one or two complete English sentences. "
                "Express the perspective implicitly and put event IDs only in the "
                "evidence_event_ids array."
            ),
            "Shift evidence: " + json.dumps(visible, sort_keys=True),
        ]
    )
    packet["estimated_tokens"] = _estimate_model_visible_tokens(packet)
    return packet


def _interview_packet(
    shift: Mapping[str, Any], visible: Mapping[str, Any], id_map: Mapping[str, str]
) -> dict[str, Any]:
    packet = _base_packet(
        shift,
        packet_type="end_of_shift_interview_bundle",
        visible=visible,
        id_map=id_map,
    )
    profile = packet["user_model_profile"]
    packet["expected_question_ids"] = [
        row["question_id"] for row in INTERVIEW_QUESTIONS
    ]
    packet["questions"] = list(INTERVIEW_QUESTIONS)
    packet["claim_layer_contract"] = APPRAISAL_CLAIM_LAYERS
    packet["required_output_schema"] = required_output_schema(packet["packet_type"])
    packet["user_message"] = "\n".join(
        [
            "Answer every interview question as the staff member at the end of this one simulated shift.",
            "Apply this designed workplace orientation silently: " + _orientation_text(profile),
            "The public answer should sound like a concise colleague reflecting on work, not a report translating numbers.",
            "Do not open by naming or restating the scenario, workload, condition, intervention, or question.",
            "Start with a concrete remembered observation, tension, or consequence; context labels remain metadata, not dialogue.",
            "Use one or two representative moments where useful. Never list the full route or mention evidence IDs in prose.",
            "A latent need or design suggestion may extend beyond literal events only when marked in its dedicated conjecture field.",
            (
                "Use design_hypothesis and tradeoff only for counterfactual_change, "
                "where both are required. Return both as null for every other question."
            ),
            "Questions: " + json.dumps(INTERVIEW_QUESTIONS, sort_keys=True),
            "Shift evidence: " + json.dumps(visible, sort_keys=True),
        ]
    )
    packet["estimated_tokens"] = _estimate_model_visible_tokens(packet)
    return packet


def build_packets(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    shifts = _load_agent_shifts(results_root)
    selected, selection = _select_balanced_shifts(
        shifts,
        replicates_per_stratum=args.replicates_per_stratum,
        allow_incomplete_strata=args.allow_incomplete_strata,
    )
    packets = []
    inventory = []
    evidence_audits = []
    for shift in selected:
        selected_shift = dict(shift)
        selected_shift["events"] = _load_selected_agent_events(shift)
        selected_shift["accumulated_experience"] = (
            _load_selected_accumulated_experience(shift)
        )
        visible, id_map = _visible_experience(
            selected_shift, max_events=args.max_events
        )
        phases = Counter(
            row["when"] for row in visible["representative_shift_events"]
        )
        place_labels = [row["place"] for row in visible["place_experience"]]
        self_transition_count = sum(
            row["from"] == row["to"] for row in visible["recurring_transitions"]
        )
        redundant_travel_wording_count = 0
        uninformative_travel_count = 0
        internal_label_count = 0
        for row in visible["representative_shift_events"]:
            if row["kind"] == "travel episode":
                match = re.match(
                    r"^Travelled .*? from (.+?) to (.+?) over ",
                    row["what_happened"],
                )
                redundant_travel_wording_count += int(
                    match is not None and match.group(1) == match.group(2)
                )
                within_match = re.match(r"^Moved ([0-9.]+) m within .* over (\d+) ", row["what_happened"])
                if within_match:
                    uninformative_travel_count += int(
                        float(within_match.group(1)) < 2.0
                        and int(within_match.group(2)) < 5
                    )
            searchable = json.dumps(row, sort_keys=True).lower()
            internal_label_count += int(
                "part3" in searchable or "coordinationnurse" in searchable
            )
        evidence_audits.append(
            {
                "event_count": len(visible["representative_shift_events"]),
                "covered_phases": sorted(phases),
                "phase_counts": dict(phases),
                "duplicate_place_label_count": len(place_labels) - len(set(place_labels)),
                "self_transition_count": self_transition_count,
                "redundant_travel_wording_count": redundant_travel_wording_count,
                "uninformative_travel_count": uninformative_travel_count,
                "internal_label_count": internal_label_count,
                "accumulated_experience_checkpoint_count": len(
                    selected_shift["accumulated_experience"]
                ),
            }
        )
        for builder in (_survey_packet, _interview_packet):
            packet = builder(shift, visible, id_map)
            packets.append(packet)
            inventory.append(
                {
                    "prompt_id": packet["prompt_id"],
                    "packet_type": packet["packet_type"],
                    "run_id": shift["map"]["run_id"],
                    "scenario": shift["map"]["scenario"],
                    "condition": shift["map"]["condition"],
                    "seed": shift["map"]["seed"],
                    "assignment_round": shift["map"].get("assignment_round"),
                    "agent_id": shift["map"]["agent_id"],
                    "role": shift["map"]["role"],
                    "persona_id": shift["map"]["persona_id"],
                    "evidence_event_count": len(id_map),
                    "estimated_tokens": packet["estimated_tokens"],
                }
            )
    prompt_path = output_dir / "prompt_packets" / "end_of_shift_appraisal_prompt_packets.jsonl"
    inventory_path = output_dir / "appraisal_prompt_inventory.csv"
    preflight_path = output_dir / "appraisal_packet_preflight.json"
    _write_jsonl(prompt_path, packets)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]) if inventory else [])
        if inventory:
            writer.writeheader()
            writer.writerows(inventory)
    packet_types = defaultdict(int)
    for packet in packets:
        packet_types[packet["packet_type"]] += 1
    missing_phase_shift_count = sum(
        set(row["covered_phases"]) != set(PHASE_ORDER) for row in evidence_audits
    )
    duplicate_place_label_shift_count = sum(
        row["duplicate_place_label_count"] > 0 for row in evidence_audits
    )
    self_transition_shift_count = sum(
        row["self_transition_count"] > 0 for row in evidence_audits
    )
    redundant_travel_wording_shift_count = sum(
        row["redundant_travel_wording_count"] > 0 for row in evidence_audits
    )
    uninformative_travel_shift_count = sum(
        row["uninformative_travel_count"] > 0 for row in evidence_audits
    )
    internal_label_shift_count = sum(
        row["internal_label_count"] > 0 for row in evidence_audits
    )
    incomplete_accumulated_experience_shift_count = sum(
        row["accumulated_experience_checkpoint_count"] != 5
        for row in evidence_audits
    )
    evidence_audit_pass = not (
        missing_phase_shift_count
        or duplicate_place_label_shift_count
        or self_transition_shift_count
        or redundant_travel_wording_shift_count
        or uninformative_travel_shift_count
        or internal_label_shift_count
        or incomplete_accumulated_experience_shift_count
    )
    preflight = {
        "preflight_pass": (
            bool(packets)
            and not selection.get("missing_strata")
            and evidence_audit_pass
        ),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "abm_rerun": False,
        "llm_run": False,
        "source_is_simulated_experience": True,
        "not_human_data": True,
        "claim_layer_contract": "grounded_pattern_interpretation_conjecture_v1",
        "selection": selection,
        "packet_count": len(packets),
        "packet_type_counts": dict(sorted(packet_types.items())),
        "experience_evidence_audit": {
            "audit_pass": evidence_audit_pass,
            "selected_shift_count": len(evidence_audits),
            "missing_phase_shift_count": missing_phase_shift_count,
            "duplicate_place_label_shift_count": duplicate_place_label_shift_count,
            "self_transition_shift_count": self_transition_shift_count,
            "redundant_travel_wording_shift_count": (
                redundant_travel_wording_shift_count
            ),
            "uninformative_travel_shift_count": uninformative_travel_shift_count,
            "internal_label_shift_count": internal_label_shift_count,
            "incomplete_accumulated_experience_shift_count": (
                incomplete_accumulated_experience_shift_count
            ),
            "minimum_representative_event_count": min(
                (row["event_count"] for row in evidence_audits), default=0
            ),
            "maximum_representative_event_count": max(
                (row["event_count"] for row in evidence_audits), default=0
            ),
            "required_phases": list(PHASE_ORDER),
        },
        "maximum_estimated_tokens": max(
            (row["estimated_tokens"] for row in packets), default=0
        ),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "internal_language_forbidden_in_public_answers": list(INTERNAL_LANGUAGE),
    }
    if args.allow_incomplete_strata:
        preflight["preflight_pass"] = bool(packets)
        preflight["incomplete_strata_allowed_for_smoke_only"] = True
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
    return preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--replicates-per-stratum",
        type=int,
        default=3,
        help="Smoke-only sample count; the complete scientific design always uses all seeds.",
    )
    parser.add_argument("--max-events", type=int, default=18)
    parser.add_argument(
        "--allow-incomplete-strata",
        action="store_true",
        help="Allow partial coverage for a structural smoke test only.",
    )
    return parser.parse_args()


def main() -> None:
    result = build_packets(parse_args())
    print(json.dumps(result, indent=2))
    if not result["preflight_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
