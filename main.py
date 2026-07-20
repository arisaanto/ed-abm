"""Entry point for the Phase 1 ED ABM."""

import argparse

import config
from src.simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergency Department ABM")
    parser.add_argument("--verify", action="store_true", help="Run laptop verification harness")
    parser.add_argument("--ablation-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ablation-ensemble", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ablation-plausibility", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mechanism-audit", action="store_true", help="Run diagnostic-mode-on mechanism audit")
    parser.add_argument("--validation-candidate", action="store_true", help="Run diagnostic-mode-off validation candidate")
    parser.add_argument("--high-acuity-smoke", action="store_true", help="Run focused ESI 1/2 high-acuity smoke check")
    parser.add_argument("--empirical-window", action="store_true", help="Print empirical shadowing observation-window timing")
    parser.add_argument("--condition-smoke", action="store_true", help="Run short condition comparison")
    parser.add_argument("--part2-scenarios", action="store_true", help="Run paired-seed Part 2 spatial/design scenario comparison")
    parser.add_argument("--interview-smoke", action="store_true", help="Run one generative-interview condition smoke")
    parser.add_argument("--duration", type=int, default=None, help="Duration in simulated seconds")
    parser.add_argument("--n-runs", type=int, default=None, help="Number of paired seeds for ablation/condition runs")
    parser.add_argument("--warmup-seconds", type=int, default=0, help="Discard this many initial simulated seconds from validation metrics")
    parser.add_argument(
        "--scenario-mode",
        default=config.DEFAULT_SCENARIO_MODE,
        choices=sorted(config.SCENARIO_MODES),
        help="Arrival/load scenario mode for validation-candidate sanity runs",
    )
    parser.add_argument(
        "--scenario-start-hour",
        type=int,
        default=config.DEFAULT_SCENARIO_START_HOUR,
        help="Simulated 24h clock hour used as the start of scenario-mode arrival profiles",
    )
    parser.add_argument("--matched-empirical-windows", action="store_true", help="Sample simulated windows matching empirical observation-window durations")
    parser.add_argument("--senior-doctor-oversight", action="store_true", help="Enable optional SeniorDoctor oversight candidate for validation")
    parser.add_argument("--condition", default=None, help="Condition name")
    parser.add_argument("--conditions", default=None, help="Comma-separated Part 2 condition list")
    parser.add_argument("--seed", type=int, default=None, help="Single seed for cluster-style Part 2 runs")
    parser.add_argument("--output-run-id", default=None, help="Optional run id for cluster-style Part 2 runs")
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--validation-target",
        default="care_area",
        choices=["care_area", "full_empirical"],
        help="Empirical F2F target used by --validation-candidate",
    )
    args = parser.parse_args()

    if args.verify:
        from src.verification import run_full_verification

        payload = run_full_verification()
        print(f"\nVerification passed {payload['passed']}/{payload['total']}")
        return

    if args.empirical_window:
        from src.empirical import audit_ecological_scope, audit_scope_alignment, describe_empirical_observation_window

        payload = describe_empirical_observation_window()
        scope_payload = audit_scope_alignment()
        ecological_payload = audit_ecological_scope()
        print("\n=== EMPIRICAL OBSERVATION WINDOW ===")
        for key, value in payload.items():
            print(f"{key}: {value}")
        print("\n=== SCOPE ALIGNMENT AUDIT ===")
        print(f"recommended_strategy: {scope_payload['validation_strategy']['recommended_strategy']}")
        print(f"scope_mismatch_flag: {scope_payload['validation_strategy']['scope_mismatch_flag']}")
        print(
            "inside_active_simulated_scope_share: "
            f"{scope_payload['scope_alignment']['inside_active_simulated_scope_share']:.3f}"
        )
        print(
            "inactive_adjacent_outside_or_unclear_share: "
            f"{scope_payload['scope_alignment']['inactive_adjacent_outside_or_unclear_share']:.3f}"
        )
        print("Validation should use the interaction log only, matched to empirical observation timing.")
        print("\n=== ECOLOGICAL SCOPE AUDIT ===")
        print(f"recommendation: {ecological_payload['recommendation']['decision']}")
        print(f"rationale: {ecological_payload['recommendation']['rationale']}")
        print(f"Saved: {config.EMPIRICAL_WINDOW_SUMMARY_PATH}")
        print(f"Scope audit: {config.SCOPE_ALIGNMENT_AUDIT_PATH}")
        print(f"Ecological audit: {config.ECOLOGICAL_SCOPE_AUDIT_PATH}")
        return

    if args.mechanism_audit:
        from src.ablation import run_mechanism_audit

        duration = args.duration or 3600
        n_runs = args.n_runs or 10
        results = run_mechanism_audit(duration_seconds=duration, n_runs=n_runs)
        print("\n=== MECHANISM AUDIT COMPLETE ===")
        print("Diagnostic mode: ON. This audits the single rule-based perception baseline, not empirical validation.")
        _print_variant_brief(results)
        print(f"Summary: {config.MECHANISM_AUDIT_SUMMARY_PATH}")
        print(f"Perception audit: {config.PERCEPTION_MECHANISM_AUDIT_MD_PATH}")
        return

    if args.validation_candidate:
        from src.ablation import run_validation_candidate

        duration = args.duration or 3600
        n_runs = args.n_runs or 10
        variants = list(config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS)
        results = run_validation_candidate(
            duration_seconds=duration,
            n_runs=n_runs,
            validation_target=args.validation_target,
            variants=variants,
            warmup_seconds=args.warmup_seconds,
            matched_empirical_windows=args.matched_empirical_windows,
            enable_senior_doctor_oversight=args.senior_doctor_oversight,
            scenario_mode=args.scenario_mode,
            scenario_start_hour=args.scenario_start_hour,
        )
        print("\n=== VALIDATION CANDIDATE COMPLETE ===")
        print("Diagnostic mode: OFF. Interaction logs only; workflow events excluded.")
        print(f"Validation target: {args.validation_target}")
        if args.warmup_seconds:
            print(f"Warm-up excluded from validation metrics: {args.warmup_seconds}s")
        if args.matched_empirical_windows:
            print("Matched empirical-window sampling: ON")
        print("Scope mode: single active simulation setup")
        print(f"SeniorDoctor oversight: {'ON' if args.senior_doctor_oversight else 'OFF'}")
        print("Run purpose: baseline_validation")
        print(f"Scenario mode: {args.scenario_mode}")
        print(f"Scenario start hour: {args.scenario_start_hour % 24}:00")
        print(f"Baseline model: {config.BASELINE_MODEL_ID}")
        _print_variant_brief(results, include_validation=True)
        print(f"Summary: {config.VALIDATION_SUMMARY_PATH}")
        scenario_summary_path = config.LATEST_OUTPUT_DIR / f"validation_summary_{args.scenario_mode}.json"
        print(f"Scenario summary: {scenario_summary_path}")
        return

    if args.ablation_smoke or args.ablation_ensemble or args.ablation_plausibility:
        from src.ablation import run_ablation_smoke

        duration = args.duration or (14_400 if args.ablation_plausibility else config.SMOKE_DURATION_SECONDS)
        n_runs = args.n_runs or (10 if args.ablation_ensemble else config.SMOKE_N_RUNS)
        results = run_ablation_smoke(duration_seconds=duration, n_runs=n_runs)
        if args.ablation_ensemble:
            print("\n=== ABLATION ENSEMBLE COMPLETE ===")
        elif args.ablation_plausibility:
            print("\n=== ABLATION PLAUSIBILITY COMPLETE ===")
        else:
            print("\n=== ABLATION SMOKE COMPLETE ===")
        if duration <= 600:
            print("Mode: fast debugging; use this for decision-path checks only.")
        elif duration <= 1800:
            print("Mode: intermediate diagnostic; check whether mechanisms remain differentiated.")
        elif duration <= 3600:
            print("Mode: one-hour plausibility check; still not scientific validation.")
        elif duration <= 14400:
            print("Mode: four-hour plausibility check; run only after verify and 1-hour diagnostics pass.")
        if n_runs > 1:
            print(f"Paired seeds: {n_runs}; inspect mean/std and paired comparisons before making claims.")
        for variant, payload in results.items():
            diagnostics = payload.get("diagnostics", {})
            print(
                f"{variant}: distance={payload['distance_to_empirical']:.3f}, "
                f"interactions={payload['interaction_count']}, "
                f"missed={diagnostics.get('missed_opportunity_episodes', 0)}, "
                f"backend={diagnostics.get('backend_calls', 0)}, "
                f"retrieval={diagnostics.get('memory_retrieval_calls', 0)}, "
                f"memory_effects={diagnostics.get('memory_effects_applied', 0)}, "
                f"fallbacks={diagnostics.get('backend_fallbacks', 0)}, "
                f"fov_hard={diagnostics.get('rejected_fov_hard', 0)}, "
                f"fov_soft={diagnostics.get('rejected_fov_soft_probability', 0)}"
            )
        if config.ABLATION_DIAGNOSTIC_MODE:
            print("Diagnostic mode was used; treat these as decision-path checks, not calibrated validation.")
        return

    if args.high_acuity_smoke:
        from src.ablation import write_model_status

        duration = args.duration or 900
        simulation = Simulation(
            condition_name=args.condition or "baseline",
            interaction_backend="structured_generative_fallback",
            persona_source=config.PERSONA_SOURCE,
            model_variant="generative_interaction",
        )
        patient = simulation.force_high_acuity_patient()
        simulation.run(duration)
        simulation.save_trace_outputs()
        high_acuity_interactions = [
            event for event in simulation.interaction_log
            if event.get("esi_level") in {1, 2}
        ]
        high_importance_memories = [
            event
            for agent in simulation.staff_agents
            for event in simulation.interaction_engine.stream_for(agent.gid).events
            if event.esi_level in {1, 2} and event.importance >= 0.8
        ]
        payload = {
            "forced_patient_id": patient.gid,
            "forced_esi_level": patient.esi_level,
            "high_acuity_interactions": len(high_acuity_interactions),
            "high_importance_memories": len(high_importance_memories),
            "traceable_event_ids": [str(event.get("event_id")) for event in high_acuity_interactions[:5]],
        }
        write_model_status(
            source="high-acuity smoke",
            duration_seconds=duration,
            high_acuity_payload=payload,
        )
        print("\n=== HIGH-ACUITY SMOKE COMPLETE ===")
        print(
            f"Forced patient {patient.gid} ESI {patient.esi_level}; "
            f"urgent interactions={payload['high_acuity_interactions']}, "
            f"high-importance memories={payload['high_importance_memories']}"
        )
        print(f"Status: {config.MODEL_STATUS_PATH}")
        return

    if args.condition_smoke:
        from src.experiments import run_part2_scenarios

        duration = args.duration or config.SMOKE_DURATION_SECONDS
        conditions = (
            [name.strip() for name in args.conditions.split(",") if name.strip()]
            if args.conditions
            else list(config.EXPERIMENT_CONDITIONS)
        )
        results = run_part2_scenarios(
            duration_seconds=duration,
            warmup_seconds=args.warmup_seconds,
            n_runs=args.n_runs or 1,
            conditions=conditions,
            save_outputs=True,
        )
        print("\n=== CONDITION SMOKE COMPLETE ===")
        print(f"Simulated duration: {duration}s")
        print(f"Warm-up excluded from reported interaction metrics: {args.warmup_seconds}s")
        print(f"Evaluated window: {max(duration - args.warmup_seconds, 0)}s")
        for condition_name, payload in results["conditions"].items():
            print(
                f"{condition_name}: interactions={payload['interaction_count']}, "
                f"f2f/hr={payload['f2f_per_hour']:.2f}, "
                f"workflow={payload['workflow_health'].get('workflow_health_status', 'UNKNOWN')}"
            )
        return

    if args.part2_scenarios:
        from src.experiments import run_part2_scenarios

        duration = args.duration or 43_200
        n_runs = args.n_runs or 3
        conditions = (
            [name.strip() for name in args.conditions.split(",") if name.strip()]
            if args.conditions
            else list(config.EXPERIMENT_CONDITIONS)
        )
        results = run_part2_scenarios(
            duration_seconds=duration,
            warmup_seconds=args.warmup_seconds,
            n_runs=n_runs,
            conditions=conditions,
            seed=args.seed,
            output_run_id=args.output_run_id,
            save_outputs=True,
        )
        print("\n=== PART 2 SCENARIO COMPARISON COMPLETE ===")
        print("Interpretation: preliminary predicted effects relative to baseline, not empirical proof.")
        print(f"Conditions: {', '.join(results['condition_order'])}")
        print(f"Paired seeds: {', '.join(str(seed) for seed in results['seeds'])}")
        for condition_name, payload in results["conditions"].items():
            print(
                f"{condition_name}: f2f/hr={payload['f2f_per_hour']:.2f}, "
                f"HCW-HCW={payload['hcw_hcw_share']:.3f}, "
                f"corridor={payload['corridor_share']:.3f}, "
                f"missed={payload['missed_opportunity_episodes']}, "
                f"workflow={payload['workflow_health'].get('workflow_health_status', 'UNKNOWN')}"
            )
        print(f"Summary: {config.PART2_SCENARIO_SUMMARY_PATH}")
        print(f"Latest notes: {config.PART2_PRELIMINARY_NOTES_LATEST_PATH}")
        return

    if args.interview_smoke:
        from src.interviews import run_post_condition_interviews

        simulation = Simulation(
            condition_name=args.condition or config.CONDITION_NAME,
            interaction_backend=config.INTERVIEW_BACKEND,
            persona_source=config.PERSONA_SOURCE,
            model_variant=args.variant or "generative_interview",
        )
        simulation.run(args.duration or config.SMOKE_DURATION_SECONDS)
        run_post_condition_interviews(simulation)
        print("\n=== INTERVIEW SMOKE COMPLETE ===")
        print(f"Outputs: {config.INTERVIEW_DIR}")
        return

    simulation = Simulation(
        condition_name=args.condition,
        model_variant=args.variant,
    )

    if config.ANIMATION_MODE:
        simulation.animate()
    else:
        duration = (
            config.TEST_DURATION_SECONDS
            if config.TEST_MODE
            else config.SIMULATION_DURATION_SECONDS
        )
        simulation.run(duration)
        simulation.visualize()
        from src.reporting import write_supervisor_outputs

        write_supervisor_outputs(simulation)
        _print_summary(simulation)


def _print_summary(simulation: Simulation) -> None:
    print("\n=== SIMULATION COMPLETE ===")
    print(
        f"Duration: {simulation.timestep}s "
        f"({simulation.timestep // 3600}h {(simulation.timestep % 3600) // 60}m)"
    )
    print(
        "Patients admitted: "
        f"{len(simulation.completed_patients) + len(simulation.active_patients)}"
    )
    print(f"Patients discharged: {len(simulation.completed_patients)}")
    print(f"Interactions logged: {len(simulation.interaction_log)}")
    print(f"Workflow events logged: {len(simulation.workflow_event_log)}")
    print(f"Missed opportunities logged: {len(simulation.missed_opportunity_log)}")
    print(f"Condition: {simulation.condition_spec.name}")
    print(f"Model variant: {simulation.model_variant}")
    print(f"Interaction backend: {simulation.interaction_backend_name}")
    print(f"Persona source: {simulation.persona_source}")
    print(f"Interaction decisions evaluated: {simulation.llm_metrics()['decision_calls']}")
    print()

    zone_counts = simulation.interaction_zone_counts()
    type_counts = simulation.interaction_type_counts()
    pair_counts = simulation.interaction_role_pair_counts()

    print("Interactions by zone:")
    for zone_id, count in zone_counts.most_common():
        print(f"  {zone_id}: {count}")

    print("\nInteractions by type:")
    for interaction_type, count in type_counts.most_common():
        print(f"  {interaction_type}: {count}")

    opportunistic_count = type_counts.get("opportunistic_corridor", 0)
    total_interactions = len(simulation.interaction_log)
    opportunistic_share = (
        opportunistic_count / total_interactions if total_interactions else 0.0
    )
    print(
        "\nOpportunistic corridor share: "
        f"{opportunistic_share:.1%} "
        f"(target {config.OPPORTUNISTIC_CORRIDOR_TARGET_SHARE_MIN:.0%}-"
        f"{config.OPPORTUNISTIC_CORRIDOR_TARGET_SHARE_MAX:.0%})"
    )

    print("\nInteractions by role pair:")
    for pair, count in pair_counts.most_common():
        print(f"  {pair[0]} - {pair[1]}: {count}")


def _print_variant_brief(results: dict, *, include_validation: bool = False) -> None:
    for _, payload in results.items():
        diagnostics = payload.get("diagnostics", {})
        line = (
            f"{config.BASELINE_MODEL_ID}: distance={payload['distance_to_empirical']:.3f}, "
            f"interactions={payload['interaction_count']}, "
            f"missed={diagnostics.get('missed_opportunity_episodes', 0)}, "
            f"perceived_staff={diagnostics.get('perceived_staff_interaction_count', 0)}, "
            f"nonproximate_logged={diagnostics.get('nonproximate_logged_interaction_count', 0)}, "
            f"approach_intents={diagnostics.get('approach_intent_count', 0)}, "
            f"approach_failures={diagnostics.get('approach_failure_count', 0)}"
        )
        if include_validation:
            line += f", validation={payload.get('validation_fit', {}).get('overall_status', 'UNKNOWN')}"
        print(line)


if __name__ == "__main__":
    main()
