"""Fast contracts for frozen study boundaries and prepared Part 3 workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import unittest

import config
from scripts.build_part3_ablation_packets import VARIANTS
from scripts.build_part3_appraisal_packets import _select_balanced_shifts
from src.conditions import get_condition_spec
from src.personas import default_cognitive_personas


ROOT = Path(__file__).resolve().parents[1]


class FrozenStudyContracts(unittest.TestCase):
    def test_part3_is_disabled_by_default(self) -> None:
        self.assertIs(config.PART3_EPISODE_LOGGING_ENABLED, False)

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
            {"neutral_orientation", "persona_no_memory", "neutral_no_memory"},
        )


class PersonaExplorerContracts(unittest.TestCase):
    def test_preview_data_and_assets_are_complete(self) -> None:
        explorer = ROOT / "web" / "persona-explorer"
        data = json.loads((explorer / "data" / "persona-results.json").read_text())
        self.assertEqual(data["meta"]["status"], "preview")
        self.assertFalse(data["meta"]["scientific_result"])
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
            self.assertEqual(len(result.get("quotes", [])), 3)
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


if __name__ == "__main__":
    unittest.main()
