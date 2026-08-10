"""Fast contracts for frozen study boundaries and prepared Part 3 workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import unittest

import config
from scripts.build.build_part3_ablation_packets import (
    CONDITIONS as ABLATION_CONDITIONS,
    PERSONAS as ABLATION_PERSONAS,
    ROLES as ABLATION_ROLES,
    SCENARIOS as ABLATION_SCENARIOS,
    VARIANTS,
    _select_balanced as select_balanced_ablation_sources,
)
from scripts.analysis.analyze_part3_architecture_ablation import _aggregate, factorial_effects
from scripts.analysis.analyze_part3_affordance_translation import construct_coverage
from scripts.build.build_part3_appraisal_packets import (
    _event_is_informative,
    _humanize_event_summary,
    _select_balanced_shifts,
)
from scripts.build.finalize_part3_appraisal_responses import trim_incomplete_prose
from scripts.run.run_part3_vllm_offline import _revalidate_previous_results
from scripts.validation.verify_part3_vllm_responses import _is_architecture_ablation
from src.conditions import get_condition_spec
from src.personas import default_cognitive_personas
from src.interviews import INTERVIEW_QUESTIONS, appraisal_prose_issues
from src.vllm_backend import VLLMOfflineBackend


ROOT = Path(__file__).resolve().parents[1]


class FrozenStudyContracts(unittest.TestCase):
    def test_part3_is_disabled_by_default(self) -> None:
        self.assertIs(config.PART3_EPISODE_LOGGING_ENABLED, False)

    def test_perception_gate_uses_the_frozen_validated_setting(self) -> None:
        source = (ROOT / "src" / "simulation.py").read_text()
        self.assertNotIn("variant_settings", source)
        self.assertIn("config.PERCEPTION_BASED_BASELINE_ENABLED", source)

    def test_condition_factor_isolation(self) -> None:
        baseline = get_condition_spec("baseline")
        cockpit = get_condition_spec("cockpit_only")
        nursta = get_condition_spec("nursta_only")
        both = get_condition_spec("both")

        self.assertEqual(baseline.zone_polygon_overrides, cockpit.zone_polygon_overrides)
        self.assertEqual(baseline.station_attractor_overrides, cockpit.station_attractor_overrides)
        self.assertEqual(baseline.routing_waypoint_overrides, cockpit.routing_waypoint_overrides)
        self.assertEqual(nursta.zone_polygon_overrides, both.zone_polygon_overrides)
        self.assertEqual(nursta.station_attractor_overrides, both.station_attractor_overrides)
        self.assertEqual(nursta.routing_waypoint_overrides, both.routing_waypoint_overrides)
        self.assertEqual(nursta.added_wall_segments, both.added_wall_segments)
        self.assertEqual(nursta.removed_wall_segments, both.removed_wall_segments)
        self.assertEqual(cockpit.visibility_transparent_segments, both.visibility_transparent_segments)
        self.assertEqual(baseline.visibility_transparent_segments, nursta.visibility_transparent_segments)


class AppraisalDesignContracts(unittest.TestCase):
    def test_deterministic_finalizer_only_trims_incomplete_tail(self) -> None:
        trimmed, changed = trim_incomplete_prose(
            "I could coordinate clearly. The nursing work area felt",
            220,
            require_character_cap=False,
        )
        self.assertTrue(changed)
        self.assertEqual(trimmed, "I could coordinate clearly.")

        complete, changed = trim_incomplete_prose(
            "I could coordinate clearly.",
            220,
            require_character_cap=False,
        )
        self.assertFalse(changed)
        self.assertEqual(complete, "I could coordinate clearly.")

    @staticmethod
    def _interview_packet() -> dict:
        return {
            "prompt_id": "appraisal_test",
            "packet_type": "end_of_shift_interview_bundle",
            "system_message": "Return JSON only.",
            "user_message": "Answer the interview.",
            "user_model_profile": {"persona_id": "team_connector"},
            "trace_evidence": {
                "metadata": {
                    "condition": "baseline",
                    "scenario_mode": "normal_load",
                }
            },
            "role": "Nurse",
            "evidence_ids": ["event_1"],
        }

    @staticmethod
    def _interview_payload(*, paired_tradeoff: bool) -> dict:
        answers = []
        for question in INTERVIEW_QUESTIONS:
            is_counterfactual = question["question_id"] == "counterfactual_change"
            answers.append(
                {
                    "question_id": question["question_id"],
                    "answer": "I could coordinate without losing track of my work.",
                    "grounded_pattern": "A recurring coordination pattern was present.",
                    "persona_conditioned_interpretation": (
                        "Shared awareness supported this designed orientation."
                    ),
                    "latent_need": None,
                    "design_hypothesis": (
                        "Provide a clearer shared coordination edge."
                        if is_counterfactual
                        else None
                    ),
                    "tradeoff": (
                        "Greater openness may reduce privacy."
                        if is_counterfactual and paired_tradeoff
                        else None
                    ),
                    "evidence_event_ids": ["event_1"],
                    "uncertainty_note": "This is a synthetic interpretation.",
                }
            )
        return {
            "answers": answers,
            "role_perspective": "Nursing work",
            "claim_layer_contract": (
                "grounded_pattern_interpretation_conjecture_v1"
            ),
            "not_human_data": True,
        }

    def test_semantic_retry_corrects_atomic_design_pair(self) -> None:
        packet = self._interview_packet()
        invalid = json.dumps(self._interview_payload(paired_tradeoff=False))
        valid = json.dumps(self._interview_payload(paired_tradeoff=True))
        batches = [[invalid], [valid]]
        observed_messages = []
        backend = VLLMOfflineBackend(max_semantic_retries=2)

        def fake_chat(messages, schemas):
            observed_messages.append(messages)
            return batches.pop(0)

        backend._chat_batch = fake_chat
        result = backend.generate_prompt_packets([packet])[0]
        self.assertEqual(result["semantic_retry_count"], 1)
        self.assertNotIn("generation_error", result)
        self.assertEqual(len(result["semantic_retry_history"]), 1)
        self.assertIn("counterfactual_change", observed_messages[1][0][-1]["content"])

    def test_semantic_retry_corrects_incomplete_public_prose(self) -> None:
        packet = self._interview_packet()
        invalid_payload = self._interview_payload(paired_tradeoff=True)
        invalid_payload["answers"][0]["answer"] = "I lost sight of the team"
        invalid = json.dumps(invalid_payload)
        valid = json.dumps(self._interview_payload(paired_tradeoff=True))
        batches = [[invalid], [valid]]
        backend = VLLMOfflineBackend(max_semantic_retries=1)
        backend._chat_batch = lambda messages, schemas: batches.pop(0)

        result = backend.generate_prompt_packets([packet])[0]

        self.assertEqual(result["semantic_retry_count"], 1)
        self.assertNotIn("generation_error", result)

    def test_public_and_internal_prose_have_distinct_contracts(self) -> None:
        self.assertIn(
            "prose does not end with a complete sentence",
            appraisal_prose_issues("I lost sight of the team", public=True),
        )
        self.assertIn(
            "event ID appears inside public prose",
            appraisal_prose_issues("I returned after shift_event_12.", public=True),
        )
        self.assertIn(
            "analysis or persona-construction language appears in public prose",
            appraisal_prose_issues(
                "My designed orientation favored quiet work.", public=True
            ),
        )
        self.assertIn(
            "public prose restates its scenario or condition",
            appraisal_prose_issues(
                "Under high load, I stayed near the nursing work area.", public=True
            ),
        )
        self.assertEqual(
            appraisal_prose_issues(
                "shift_event_12: repeated contact at nursing work area",
                public=False,
            ),
            [],
        )

    def test_resume_revalidates_and_selects_only_invalid_records(self) -> None:
        valid_packet = self._interview_packet()
        invalid_packet = dict(valid_packet, prompt_id="appraisal_test_invalid")
        valid_payload = self._interview_payload(paired_tradeoff=True)
        invalid_payload = self._interview_payload(paired_tradeoff=True)
        invalid_payload["answers"][0]["answer"] = "I lost sight of the team"
        previous = {
            valid_packet["prompt_id"]: {
                "prompt_id": valid_packet["prompt_id"],
                "packet_type": valid_packet["packet_type"],
                "raw_response": json.dumps(valid_payload),
                "generation_error": {"error": "obsolete strict failure"},
            },
            invalid_packet["prompt_id"]: {
                "prompt_id": invalid_packet["prompt_id"],
                "packet_type": invalid_packet["packet_type"],
                "raw_response": json.dumps(invalid_payload),
            },
        }

        reusable, pending, audit = _revalidate_previous_results(
            [valid_packet, invalid_packet],
            previous,
            VLLMOfflineBackend(),
        )

        self.assertEqual(set(reusable), {valid_packet["prompt_id"]})
        self.assertNotIn("generation_error", reusable[valid_packet["prompt_id"]])
        self.assertEqual(
            [packet["prompt_id"] for packet in pending],
            [invalid_packet["prompt_id"]],
        )
        self.assertEqual(audit["locally_reusable_response_count"], 1)
        self.assertEqual(audit["repair_packet_count"], 1)

    def test_uninformative_travel_is_excluded_from_appraisal_cards(self) -> None:
        self.assertFalse(
            _event_is_informative(
                {
                    "event_type": "travel_episode",
                    "start_place_label": "corridor",
                    "end_place_label": "corridor",
                    "distance_meters": 1.0,
                    "duration_seconds": 1,
                }
            )
        )
        self.assertTrue(
            _event_is_informative(
                {
                    "event_type": "travel_episode",
                    "start_place_label": "corridor",
                    "end_place_label": "patient room",
                    "distance_meters": 1.0,
                    "duration_seconds": 1,
                }
            )
        )

    def test_internal_decision_labels_are_humanized(self) -> None:
        declined = _humanize_event_summary(
            {"event_type": "missed_opportunity"},
            "A contact did not proceed because part3 cognitive decline.",
        )
        deferred = _humanize_event_summary(
            {
                "event_type": "cognitive_opportunity_decision",
                "partner_role": "CoordinationNurse",
                "place": {"label": "nursing work area"},
                "selected_action": "defer",
            },
            "unused",
        )
        self.assertEqual(
            declined,
            "A contact did not proceed because the contact was declined after "
            "weighing timing and relevance.",
        )
        self.assertEqual(
            deferred,
            "An optional contact with a coordination nurse in nursing work area "
            "was deferred.",
        )

    def test_exhausted_semantic_retry_is_preserved_for_verifier(self) -> None:
        packet = self._interview_packet()
        invalid = json.dumps(self._interview_payload(paired_tradeoff=False))
        batches = [[invalid], [invalid]]
        backend = VLLMOfflineBackend(max_semantic_retries=1)

        def fake_chat(messages, schemas):
            return batches.pop(0)

        backend._chat_batch = fake_chat
        result = backend.generate_prompt_packets(
            [packet], preserve_failures=True
        )[0]
        self.assertEqual(result["semantic_retry_count"], 1)
        self.assertIsNone(result["response"])
        self.assertEqual(result["generation_error"]["attempt_count"], 2)

    def test_all_seed_paired_selection_and_role_balance(self) -> None:
        scenarios = ("normal_load", "high_load_high_acuity")
        conditions = ("baseline", "cockpit_only", "nursta_only", "both")
        roles = ("CoordinationNurse", "Nurse", "Doctor")
        personas = [row.persona_id for row in default_cognitive_personas()]
        shifts = []
        for scenario in scenarios:
            for persona in personas:
                for seed in range(1, 11):
                    for role_index, role in enumerate(roles, start=1):
                        for condition in conditions:
                            shifts.append(
                                {
                                    "map": {
                                        "run_id": f"{scenario}-{condition}-{seed}-{role}",
                                        "scenario": scenario,
                                        "condition": condition,
                                        "seed": seed,
                                        "assignment_round": role_index,
                                        "persona_id": persona,
                                        "role": role,
                                        "agent_id": role_index,
                                    }
                                }
                            )

        selected, audit = _select_balanced_shifts(
            shifts,
            replicates_per_stratum=3,
            allow_incomplete_strata=False,
        )
        self.assertEqual(len(selected), 400)
        self.assertEqual(audit["selected_paired_unit_count"], 100)
        self.assertEqual(audit["observed_seed_stratum_count"], 100)
        self.assertEqual(audit["missing_strata"], [])
        self.assertEqual(audit["incomplete_pair_count"], 0)

        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for shift in selected:
            row = shift["map"]
            grouped[(row["scenario"], row["persona_id"])].append(row)
        for rows in grouped.values():
            self.assertEqual({row["seed"] for row in rows}, set(range(1, 11)))
            role_counts = Counter(row["role"] for row in rows)
            self.assertEqual(sorted(role_counts.values()), [12, 12, 16])

    def test_ablation_is_complete_two_by_two(self) -> None:
        self.assertEqual(
            set(VARIANTS),
            {
                "full_persona_full_memory",
                "neutral_orientation",
                "persona_no_memory",
                "neutral_no_memory",
            },
        )

    def test_ablation_source_selection_is_exactly_seed_balanced(self) -> None:
        records = []
        for scenario in ABLATION_SCENARIOS:
            for condition in ABLATION_CONDITIONS:
                for persona in ABLATION_PERSONAS:
                    for role in ABLATION_ROLES:
                        for seed in range(1, 11):
                            prompt_id = f"{scenario}-{condition}-{persona}-{role}-{seed}"
                            records.append(
                                {
                                    "packet": {
                                        "prompt_id": prompt_id,
                                        "user_model_profile": {"persona_id": persona},
                                        "trace_evidence": {
                                            "metadata": {
                                                "scenario_mode": scenario,
                                                "condition": condition,
                                                "role": role,
                                                "seed": seed,
                                            }
                                        },
                                    },
                                    "packet_sha256": prompt_id,
                                }
                            )
        selected, _ = select_balanced_ablation_sources(records, 1)
        self.assertEqual(len(selected), 120)
        seed_counts = Counter(
            row["packet"]["trace_evidence"]["metadata"]["seed"]
            for row in selected
        )
        self.assertEqual(seed_counts, Counter({seed: 12 for seed in range(1, 11)}))

    def test_ablation_packets_bypass_the_legacy_60_packet_design_gate(self) -> None:
        packets = [
            {
                "ablation_not_main_result": True,
                "ablation_variant": variant,
                "ablation_pair_id": "pair-1",
            }
            for variant in VARIANTS
        ]
        self.assertTrue(_is_architecture_ablation(packets))
        self.assertFalse(
            _is_architecture_ablation(
                [{"ablation_variant": "neutral_orientation"}]
            )
        )

    def test_ablation_factorial_effects_reconstruct_expected_contrasts(self) -> None:
        cells = {
            "full_persona_full_memory": {"selected_action": "engage"},
            "neutral_orientation": {"selected_action": "decline"},
            "persona_no_memory": {"selected_action": "decline"},
            "neutral_no_memory": {"selected_action": "decline"},
        }
        effects = factorial_effects(cells)
        self.assertEqual(effects["orientation_effect_with_memory"], 1.0)
        self.assertEqual(effects["memory_effect_with_persona"], 1.0)
        self.assertEqual(effects["orientation_memory_interaction"], 1.0)

    def test_ablation_confidence_interval_clusters_by_seed(self) -> None:
        result = _aggregate(
            [-1.0, -1.0, 0.0, 0.0],
            clusters=["1", "1", "2", "2"],
        )
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["ci_method"], "seed_cluster_robust_t")
        self.assertEqual(result["mean"], -0.5)
        self.assertGreater(result["se"], 0.0)

    def test_privacy_is_excluded_when_confidentiality_is_unobserved(self) -> None:
        rows = [
            {"dimension": "privacy_and_control", "rateability": "rateable"},
            {"dimension": "privacy_and_control", "rateability": "insufficient_evidence"},
        ]
        coverage = {
            row["dimension"]: row for row in construct_coverage(rows)
        }
        privacy = coverage["privacy_and_control"]
        self.assertEqual(privacy["rateable_share"], 0.5)
        self.assertEqual(privacy["analysis_status"], "exclude_from_primary_inference")
        self.assertIn("confidentiality", privacy["claim_boundary"])

    def test_independent_interview_audit_is_complete_and_not_human_review(self) -> None:
        manifest = json.loads(
            (ROOT / "manifests" / "part3_independent_interview_review.json").read_text()
        )
        self.assertEqual(
            manifest["reviewer_type"],
            "independent_model_assisted_audit",
        )
        self.assertIs(
            manifest["human_review_completed"],
            False,
        )
        self.assertEqual(len(manifest["bundles"]), 40)

        review_dir = (
            ROOT
            / "outputs"
            / "findings"
            / "part3_cognitive_personas_n10"
            / "review"
        )
        with (review_dir / "reviewed_interview_sample.csv").open(newline="") as handle:
            answers = list(csv.DictReader(handle))
        self.assertEqual(len(answers), 240)
        self.assertEqual(len({row["prompt_id"] for row in answers}), 40)
        for row in answers:
            if row["include_in_explorer"].lower() == "true":
                self.assertEqual(row["review_grounding_pass"].lower(), "true")
                self.assertEqual(row["review_claim_layer_pass"].lower(), "true")
                self.assertEqual(
                    row["reviewer_type"],
                    "independent_model_assisted_audit",
                )

        with (review_dir / "dictionary_validation.csv").open(newline="") as handle:
            dictionary_rows = list(csv.DictReader(handle))
        self.assertEqual(len(dictionary_rows), 5)


class PersonaExplorerContracts(unittest.TestCase):
    def test_observed_data_and_assets_are_complete(self) -> None:
        explorer = ROOT / "web" / "persona-explorer"
        data = json.loads((explorer / "data" / "persona-results.json").read_text())
        self.assertEqual(data["meta"]["status"], "observed")
        self.assertTrue(data["meta"]["scientific_result"])
        self.assertTrue(data["meta"]["not_human_data"])
        self.assertIn("Synthetic persona-conditioned", data["meta"]["claim_boundary"])
        qualitative_review = data["meta"]["qualitative_review"]
        self.assertEqual(
            qualitative_review["reviewer_type"],
            "independent_model_assisted_audit",
        )
        self.assertIs(qualitative_review["human_review_completed"], False)
        self.assertEqual(qualitative_review["answer_count"], 240)
        self.assertEqual(
            qualitative_review["answers_passing_scientific_use_gate"],
            184,
        )
        self.assertEqual(qualitative_review["cells_without_retained_excerpt"], 2)
        expected_count = len(data["personas"]) * len(data["scenarios"]) * len(data["conditions"])
        self.assertEqual(len(data["results"]), expected_count)
        keys = {
            (row["persona"], row["scenario"], row["condition"])
            for row in data["results"]
        }
        self.assertEqual(len(keys), expected_count)
        forbidden_openings = (
            "under high load",
            "under heavier pressure",
            "during normal load",
            "in this baseline",
            "in this condition",
        )
        for result in data["results"]:
            self.assertEqual(result["source"], "verified_part3_appraisals")
            self.assertEqual(result["sample_size"], 10)
            self.assertEqual(
                set(result["scores"]),
                {
                    "overall_experience",
                    "coordination_support",
                    "focus_support",
                    "spatial_legibility",
                },
            )
            self.assertGreaterEqual(len(result.get("quotes", [])), 1)
            self.assertLessEqual(len(result.get("quotes", [])), 3)
            for quote in result["quotes"]:
                self.assertTrue(quote.strip())
                self.assertFalse(
                    quote.lower().startswith(forbidden_openings),
                    quote,
                )
        self.assertTrue((explorer / "app.js").is_file())
        for persona in data["personas"]:
            sprite = explorer / persona["sprite"]
            portrait = sprite.with_name(f"{sprite.stem}-portrait.png")
            self.assertTrue(sprite.is_file(), sprite)
            self.assertTrue(portrait.is_file(), portrait)


class PublishedFindingsContracts(unittest.TestCase):
    def test_questionnaire_convergence_table_is_aggregate_only(self) -> None:
        path = (
            ROOT
            / "outputs"
            / "findings"
            / "part3_cognitive_personas_n10"
            / "tables"
            / "tableD_questionnaire_convergence.csv"
        )
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 7)
        self.assertEqual(
            set(rows[0]),
            {
                "construct",
                "empirical_mean",
                "empirical_sd",
                "synthetic_mean",
                "synthetic_sd",
                "absolute_difference",
                "scale",
            },
        )
        self.assertNotIn("participant_id", rows[0])
        self.assertEqual(rows[0]["construct"], "Maximum workload")
        self.assertEqual(rows[-1]["construct"], "Information sufficiency")


if __name__ == "__main__":
    unittest.main()
