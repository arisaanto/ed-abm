# Script Guide

The model library lives in `src/`. Its `analysis.py` module contains two
importable spatial-statistics helpers used by the simulator and empirical
comparison code. This directory contains executable research
workflows grouped by what they do:

- `run/`: launch one or more simulations, the bounded Part 3 closed loop, or
  offline vLLM inference.
- `validation/`: check inputs, execution contracts, and completed outputs.
- `analysis/`: compute scientific summaries from accepted result packages.
- `build/`: create manifests, prompt packets, figures, tables, and
  persona-explorer data.

Run commands from the repository root, for example:

```bash
python3 scripts/validation/verify_part2_spatial_interventions.py --help
python3 scripts/build/build_part3_figures_tables.py --help
```

The categories are intentionally separate from `src/`: importing the model
does not execute a study workflow, and workflow scripts can reuse the same
tested model and metric functions.
