# Canonical Project Map

## Source

- `main.py`: convenience wrapper for one canonical run.
- `config.py`: active frozen defaults used by the retained model and workflows.
- `src/agents.py`, `src/simulation.py`: embodied staff/patient state and the
  simulation orchestrator.
- `src/environment.py`, `src/conditions.py`, `src/perception.py`: routing,
  intervention geometry, and visibility.
- `src/interaction.py`, `src/metrics.py`, `src/empirical.py`: interaction
  behavior, shared outcome metrics, and empirical parsing/comparison.
- `src/personas.py`, `src/part3_closed_loop.py`, `src/interviews.py`,
  `src/cognitive_policy.py`, `src/vllm_backend.py`: opt-in Part 3 personas,
  grounded memory, appraisal contracts, and inference.
- `scripts/run/`: simulation and inference execution.
- `scripts/validation/`: preflight and completed-result checks.
- `scripts/analysis/`: scientific post-processing.
- `scripts/build/`: manifests, packets, figures, tables, and web exports.
- `scripts/run/run_single.py`, `scripts/run/run_batch.py`: reproducible single-run and manifest execution.
- `scripts/build/build_part2_figures_tables.py`: authoritative Part 2 figure/table generation from completed outputs.
- `scripts/analysis/analyze_part2_parameter_sensitivity.py`: final two-phase combination of three retained Phase 1 factors plus the Phase 2 pressure-threshold replacement.
- `scripts/build/build_part3_user_models.py`, `scripts/build/plan_part3_synthetic_study.py`: Part 3 persona protocol, balanced assignments, fixtures, and evidence-packet construction.
- `scripts/run/run_part3_closed_loop.py`, `scripts/run/run_part3_closed_loop_batch.py`: explicit live causal path with one shared vLLM engine per GPU job.
- `scripts/run/run_part3_vllm_offline.py`: batched post-run appraisal inference.
- `scripts/analysis/analyze_part3_closed_loop_main.py`: accepted 400-run analysis, including paired persona effects and hidden-rule versus Qwen same-opportunity substitution diagnostics.
- `scripts/build/build_part3_appraisal_packets.py`, `scripts/analysis/analyze_part3_appraisals.py`: condition-paired end-of-shift evidence packets and the prespecified seed-level quantitative/manual qualitative analysis.
- `scripts/build/build_part3_figures_tables.py`: authoritative Part 3 figure/table and persona-explorer export from completed main-study, appraisal, and architecture-ablation outputs.
- `scripts/build/export_part3_persona_explorer_data.py`: strict post-review adapter from verified appraisal tables to the supplementary explorer JSON; it does not synthesize or auto-select quotations.
- `scripts/build/build_part3_ablation_packets.py`: CPU-only matched 2 x 2 orientation-by-memory packet design; it does not execute inference.
- `web/persona-explorer/`: supplementary interactive persona browser with verified synthetic appraisal scores, exact representative excerpts, and a query-addressable print-card fallback.
- `jobs/`: final Part 3 preflight, main-study, appraisal, and architecture-
  ablation Slurm definitions. Superseded smoke, gate, and pilot launchers are
  not retained.

## Input Data

- `data/wall_positions.json`: regular local JSON file, not a symlink.
- `data/zone_boundaries.json`: condition-independent zone geometry used by the environment.
- Restricted local source outside the repository: `../shadowing_with-participant-info.csv`.
- Restricted local exploration notebook outside the repository: `../EVIDENT_ShadowingDataset_ARIS.ipynb`.

## Part 1

- Local canonical source: `outputs/source_results/part1_validation_n100/`
- Retained report: `outputs/findings/part1_validation_n100/`

## Part 2

- Local canonical source: `outputs/source_results/part2_spatial_interventions_n100/`
- Retained report/tables: `outputs/findings/part2_spatial_interventions_n100/`

The retained package contains the current intervention atlas, spatial redistribution, coordination-ecology figure, paired movement/visibility/interaction tables, and supporting source tables.

## Sensitivity

- Local raw completed Phase 2 pressure-threshold input: `outputs/source_results/part2_parameter_sensitivity_n20_outputs.zip` (excluded from version control)
- Single authoritative findings folder: `outputs/findings/part2_parameter_sensitivity_n20/`
- Current analyzer: `scripts/analysis/analyze_part2_parameter_sensitivity.py`

The old `SCENARIO_PRESSURE_STATION_SOCIAL_SUPPRESSION` factor is non-authoritative and excluded. The completed Phase 2 replacement is `SCENARIO_PRESSURE_ACTION_THRESHOLD`. Final interpretation uses 960 retained Phase 1 runs plus 320 Phase 2 runs.

## Part 3

- Final findings: `outputs/findings/part3_cognitive_personas_n10/`
- Interactive supplement: `web/persona-explorer/`
- Local completed sources: `outputs/source_results/part3_closed_loop_main_n10_24719095.tar.gz`, `outputs/source_results/part3_appraisals_final_24789077.tar.gz`, and `outputs/source_results/part3_architecture_ablation_v2_24817925.tar.gz`

The findings package contains the five-orientation atlas, the common-affordance
versus heterogeneous-uptake result, the inclusive person-space-fit result,
paired persona effects, the matched architecture ablation, recurrent synthetic
design hypotheses, and explicitly descriptive role-adjusted trace-fit
diagnostics.

## Documentation

- `docs/EXPERIMENT_PLAN.md`: study status.
- `docs/ARCHITECTURE.md`: active model boundaries.
- `docs/PART2_SPATIAL_INTERVENTIONS_PROTOCOL.md`: completed Part 2 protocol.
- `docs/PAPER_FIGURE_TABLE_STRATEGY.md`: retained/future quantitative outputs.
- `docs/PART3_PLAN.md`: frozen cognitive-persona protocol and phased Part 3 execution gates.
- `docs/PART3_APPRAISAL_ANALYSIS_PROTOCOL.md`: locked paired appraisal estimands, uncertainty, convergence, polarization, worst-case-fit, and manual-review contract.
- `docs/PART3_LLM_BACKEND_SETUP.md`: separate vLLM environment, offline jobs, and provisional SBU budget.
- `docs/LITERATURE_REFERENCE_MAP.md`: bounded reference map.

The machine-readable appraisal contract is
`manifests/part3_appraisal_analysis_spec.json`.

`manifests/part3_main_execution_source.sha256` is the immutable code snapshot
for the completed 400-run Part 3 main execution. It is provenance, not a claim
that the current post-run analysis/appraisal code is byte-identical. The live
repository checkpoint is `manifests/part3_current_source.sha256`.

## Output Formats

- Figures: PDF for Overleaf/arXiv and PNG for repository previews.
- Tables: TeX for the manuscript, CSV for machine-readable reuse, and Markdown
  for direct review.
- SVG copies are intentionally omitted because arXiv's standard pdfLaTeX
  workflow does not use them and the PDFs already preserve vector content.

## Deprecated Or Removed

- Historical execution files, local smoke outputs, generated Part 3 traces,
  review archives, and non-authoritative dashboard figures have been removed.
