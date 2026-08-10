# Spatially Explicit ED Interaction Model

A workflow-constrained, spatially embodied agent-based model of face-to-face
interaction in an emergency department. The study validates the model against
empirical shadowing data, tests visibility and workstation interventions, and
examines how five designed cognitive orientations take up the resulting spatial
affordances.

![Study design: validate, intervene, and interpret](outputs/findings/study_overview/figures/study_design_overview.png)

## Study at a Glance

| Part | Question | Design |
|---|---|---|
| 1. Validate | Does the model reproduce the observed interaction ecology? | Empirical comparison and negative controls across `100` runs |
| 2. Intervene | What changes when spatial design features change? | Four spatial conditions, two operating scenarios, `800` paired runs, and a four-factor sensitivity screen |
| 3. Interpret | Who takes up the new affordances, and how? | Five cognitive orientations balanced across role, condition, scenario, and assignment round in `400` closed-loop runs, followed by synthetic appraisals and a matched architecture ablation |

Part 1 produced `22.14` simulated face-to-face interactions per hour against an
empirical target of `22.78`, with `100/100` workflow passes and clean hard
gates. Part 2 completed `800/800` workflow-clean runs. Part 3 completed
`400/400` runs with balanced orientation exposure, isolated exogenous arrival
streams, temporal and sampling coverage, and clean technical-integrity gates.
In a post hoc bridge to observed staff experience, six comparable normal-load
synthetic questionnaire means were `0.25` scale points from the observed means
on average; five met the prespecified half-point margin.

## Model Boundary

The rule-based ABM owns geometry, routing, workflow, patient flow, acuity,
feasible partners, proximity, mutual visibility, and event logging. Parts 1
and 2 never use an LLM. Part 3 is opt-in: Qwen may choose among feasible
discretionary interaction actions supplied by the ABM, but it cannot invent
movement, patients, partners, or events.

The five Part 3 personas are preregistered synthetic workplace orientations,
not discovered or validated personality types. Their OCEAN profiles are
illustrative display labels and are never causal model inputs. Synthetic
appraisals are design probes, not testimony from Zurich ED staff.

## Repository Map

```text
src/          model, geometry, interaction, metrics, and Part 3 policy code
scripts/
  run/        simulation and inference entry points
  validation/ preflight and result-verification commands
  analysis/   scientific post-processing
  build/      manifests, packets, figures, tables, and web-data builders
jobs/         final Snellius batch definitions retained for reproducibility
manifests/    paired experiment manifests and source-integrity lock
data/         non-participant geometry inputs
docs/         architecture, completed-study protocols, and analysis plans
tests/        compact study-contract and publication-artifact regression suite
outputs/
  findings/   publication-ready aggregate figures, tables, and reports
web/
  persona-explorer/  interactive Part 3 results interface
```

Participant-linked data, imported run trees, model caches, cluster logs,
archives, and generated traces are intentionally excluded from version
control.

## Installation

Parts 1 and 2 use the CPU environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The Part 3 vLLM backend requires a separate GPU environment:

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
python3 -m pip install -r requirements-llm.txt
```

See the [Part 3 backend setup](docs/PART3_LLM_BACKEND_SETUP.md) for the
reproducible Snellius workflow. The CPU and vLLM environments should not be
mixed.

## Verification

Repository-level checks do not require restricted research data:

```bash
python3 -m py_compile \
  config.py main.py src/*.py scripts/*/*.py tests/*.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
find jobs -maxdepth 1 -name '*.sbatch' -print -exec bash -n {} \;
shasum -a 256 -c manifests/part3_current_source.sha256
```

Result-level verification commands live in `scripts/validation/` and require
the corresponding restricted source data or completed run tree. The completed
Part 1, Part 2, and Part 3 batches should not be rerun for ordinary
development.

## Results

- [Study overview and interaction pipeline](outputs/findings/study_overview/)
- [Part 1 validation report](outputs/findings/part1_validation_n100/part1_validation_n100_report.md)
- [Part 2 intervention findings](outputs/findings/part2_spatial_interventions_n100/)
- [Part 2 sensitivity findings](outputs/findings/part2_parameter_sensitivity_n20/)
- [Part 3 cognitive-orientation findings](outputs/findings/part3_cognitive_personas_n10/)
- [Part 3 questionnaire-convergence table](outputs/findings/part3_cognitive_personas_n10/tables/tableD_questionnaire_convergence.md)
- [Interactive persona explorer](https://arisaanto.github.io/ed-abm/)
- [Persona explorer source](web/persona-explorer/)

Figures are retained as publication-ready PDF and review-friendly PNG files.
Tables are retained as LaTeX for the manuscript, CSV for analysis, and
Markdown for repository review.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Project and experiment status](docs/EXPERIMENT_PLAN.md)
- [Part 2 intervention protocol](docs/PART2_SPATIAL_INTERVENTIONS_PROTOCOL.md)
- [Part 3 protocol](docs/PART3_PLAN.md)
- [Part 3 appraisal analysis plan](docs/PART3_APPRAISAL_ANALYSIS_PROTOCOL.md)

## Data Availability

This repository contains software, non-participant geometry inputs, aggregate
reported findings, and the interactive results interface. The
participant-linked shadowing source and full run-level outputs are not public
repository assets. Any future release of deidentified source data remains
subject to the source study's approval and data-management conditions.

## Interpretation Limits

The results do not establish clinical outcomes, a universally optimal ED
layout, real staff experience, validated personality types, or transfer to
other sites without recalibration. Part 3 estimates responses within this
model and its prespecified synthetic orientations.

## Citation

The manuscript is in preparation. Until a formal citation is available, cite
this repository by its title, author, URL, and archived release or commit.

## License

The software is released under the [MIT License](LICENSE). Data and third-party
assets remain subject to their original access and licensing terms.
