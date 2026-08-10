# Study Status And Experiment Plan

This is the authoritative study-status handoff for paper drafting. Code and canonical outputs remain the source of truth.

## Part 1 - Validated Baseline

**Completed and frozen.** Part 1 validates a non-LLM, embodied, workflow-constrained ED interaction engine under `normal_load / baseline`.

- Runs: n=100
- Simulated F2F/hour: approximately 22.14
- Empirical F2F/hour: approximately 22.78
- Residual gap: approximately -0.64/hour (-2.8%)
- Workflow: 100/100 PASS
- Hard embodied/routing gates: clean
- Composite distance: approximately 0.878
- KDE/spatial component: approximately 0.206
- Zone component: approximately 0.055
- Role-pair component: approximately 0.046
- Topic component: approximately 0.396
- Duration component: approximately 0.145
- Simulated zone shares: station/desk 0.446, corridor 0.261, bedside/patient room 0.293

The remaining station-heavy redistribution and approximate topic semantics are limitations, not reasons to recalibrate during Part 2.

Authoritative report: `outputs/findings/part1_validation_n100/part1_validation_n100_report.md`.

Canonical one-run entrypoint, retained for software reproducibility rather
than routine rerunning of the accepted n100 batch:

```bash
python3 main.py \
  --scenario-mode normal_load \
  --condition baseline \
  --seed 1 \
  --duration 43200 \
  --warmup-seconds 7200 \
  --validation-target care_area \
  --output-dir outputs/manual/baseline_seed_1
```

## Part 2 - Spatial Interventions

**Completed at n=100.** The paired design includes 2 scenarios x 4 conditions x 100 seeds = 800 runs. Canonical raw results contain 800 `summary.json` and 800 `interaction_events.csv` files. Workflow passed 800/800; integrity and hard gates passed.

The causal chain is:

`objective spatial affordance -> experienced visibility -> reason-bearing opportunity -> approach -> embodied F2F ecology -> operational pressure`

Headline completed-n100 paired F2F/hour effects:

| Scenario | COCPIT | NURSTA | Both |
|---|---:|---:|---:|
| `normal_load` | +9.25 (95% CI 8.81 to 9.69) | +0.07 (-0.35 to 0.49) | +8.96 (8.54 to 9.38) |
| `high_load_high_acuity` | +6.95 (6.52 to 7.38) | +0.23 (-0.13 to 0.59) | +7.67 (7.23 to 8.11) |

Current interpretation: COCPIT transparency is dominant; NURSTA-only is weak/near-neutral at default parameters; `both` is largely COCPIT-driven. COCPIT conditions increase HCW-HCW/corridor coordination and reduce patient-facing/bedside share, while ED pressure effects are modest. These are simulated conditional effects, not empirical post-occupancy evidence.

Authoritative protocol: `docs/PART2_SPATIAL_INTERVENTIONS_PROTOCOL.md`.

## Part 2 - Completed Robustness And Sensitivity

**Completed-n100 robustness** reuses the existing 800 runs and evaluates paired CIs, median/IQR, sign consistency, leave-one-out means, outlier influence, scenario consistency, and synergy at default parameters.

The original n20-per-cell screen contains 1,280 integrity-clean runs. Visibility response, repeat-contact delay, and station-return tendency are informative. The old crowding-suppression factor is excluded because it was inactive on the operative reason-gate path and produced identical low/high outputs. Its 320-run pressure-activation-threshold replacement completed with 320/320 workflow PASS, all hard gates zero, and complete runtime diagnostics. The authoritative final set combines 960 retained original runs with these 320 replacement runs.

Authoritative results: `outputs/findings/part2_parameter_sensitivity_n20/`.

## Part 3 - Bounded Synthetic Design-Evaluation Layer

**The 400-run closed-loop causal study, end-of-shift appraisal inference, and matched architecture ablation are complete.** Part 3 treats the ED intervention as the interface and crosses five role-independent cognitive orientations with the validated staff roles. The ABM remains responsible for physics, workflow, patient state, proximity, and feasible actions. The opt-in Qwen policy chooses among feasible discretionary interaction actions and retrieves source-linked memories; it never enters the default Part 1/2 pipeline.

The paired main design contains 2 scenarios x 4 conditions x 10 seeds x 5 persona-assignment rounds = 400 runs. All 400 summaries are present. Technical integrity, persona exposure balance, common exogenous arrival streams, temporal coverage, and sampling coverage passed. The final condition-paired selection retained 100 paired units across all seeds, producing 800 verified survey/interview packets (400 survey and 400 interview packets). The matched prompt-level 2 x 2 architecture ablation used 120 fixed observed opportunities per cell and completed without rerunning the ABM.

A post hoc aggregate questionnaire audit compared role-standardized normal-load
synthetic ratings with authorized summary statistics from observed staff
shifts. Across six independently comparable 1–7 items, mean absolute error was
0.25 points and five means met a prespecified 0.5-point margin. Q2 was reported
separately because its 25% synthetic mean was partly fixed by construction; Q8
served as an unsupported-acoustics negative control. The benchmark informed
audit development, so this is descriptive convergence rather than held-out
validation.

Personas are preregistered synthetic cognitive orientations with transparent workplace priors. Role is retained as a blocking variable, and every staff id cycles through every persona. Outputs must cite evidence IDs and carry `not_human_data: true`; the optional OCEAN character display is a derived ordinal crosswalk, not psychometric measurement and not a policy input.

Authoritative plan: `docs/PART3_PLAN.md`.

## Implemented / Planned / Deprecated

- **Implemented:** Part 1 n100, Part 2 n100, repaired visibility-layer spatial audit, completed-n100 robustness, final two-phase four-factor n20 sensitivity evidence, the Part 3 400-run closed-loop causal study, 800 verified appraisal packets, and the matched 2 x 2 cognitive-architecture ablation.
- **Completed sensitivity screen:** 960 retained valid original runs plus 320 pressure-threshold replacement runs; final total 1,280.
- **Completed Part 3 appraisal layer:** 100 condition-paired units, 800 verified packets, prespecified seed-paired fit analysis, reviewed qualitative bundles, and recurrent design-priority coding.
- **Completed questionnaire bridge:** 390 verified audit packets, correct evidence-withheld and acoustic refusals, and authorized aggregate normal-load convergence results.
- **Deprecated:** descriptions of Part 2 as an n20/n50 pilot; the n10 +/-20% sensitivity design; seven generic Part 2 plots as the final paper figure set.

## Do Not Claim

Do not claim clinical outcomes, universal ED design recommendations, human experience from synthetic outputs, invariance beyond the informative tested assumptions, or formal DepthmapX analysis.
