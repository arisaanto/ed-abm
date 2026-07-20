# Batch Workflow

## Canonical Completed Batches

| Batch | Design | Local canonical path | Status |
|---|---|---|---|
| Part 1 | 1 scenario x 1 condition x 100 seeds | `outputs/latest/ABM_results/part1_validation_n100/` | Completed/frozen |
| Part 2 | 2 scenarios x 4 conditions x 100 seeds | `outputs/latest/ABM_results/part2_spatial_interventions_n100/` | Completed/integrity PASS |
| Part 2 sensitivity, original | 4 parameters x 2 levels x 2 scenarios x 4 conditions x 20 seeds | `outputs/findings/part2_parameter_sensitivity_n20/` (compact) | Completed/integrity PASS; crowding factor excluded |
| Sensitivity Phase 2 replacement | 1 parameter x 2 levels x 2 scenarios x 4 conditions x 20 seeds | `outputs/findings/part2_parameter_sensitivity_n20/` | Completed/integrity and effectiveness PASS |
| Final two-phase sensitivity set | 4 active parameters x 2 levels x 2 scenarios x 4 conditions x 20 seeds | `outputs/findings/part2_parameter_sensitivity_n20/` | Authoritative, 1,280 runs |

The historical Part 2 n20 pilot is not the current scientific result.

## Current Reporting Commands

Completed raw results are read in place; no simulation is required:

```bash
python3 scripts/verify_part2_spatial_interventions.py \
  --batch-dir outputs/latest/ABM_results/part2_spatial_interventions_n100 \
  --expected full

python3 scripts/build_part2_figures_tables.py
```

The sensitivity analyzer reads the two retained raw archives and writes the
single authoritative folder under `outputs/findings/part2_parameter_sensitivity_n20/`.
Historical SLURM jobs and manifests are intentionally not retained.
