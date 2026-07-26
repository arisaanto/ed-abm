# Part 1 validation n100

## Protocol

This package reports the frozen Part 1 baseline: `normal_load`, `baseline`, seeds 1-100, 43,200 simulated seconds per seed, 7,200 seconds warm-up, and a 10-hour evaluation window per seed. This reporting pass did not run simulations or change model behavior.

Canonical local input/result path: `outputs/source_results/part1_validation_n100`

## Main validation result

The model is validated here as an ED interaction fingerprint, not as a single-number fit. The main result is:

- Observed F2F/hour: 22.78
- Simulated F2F/hour: 22.14 (SD 1.46, SE 0.15, 95% CI 21.85-22.43)
- Residual gap: -0.64/hour (-2.8%)
- Coefficient of variation across runs: 0.066
- Workflow pass: 100/100
- Pooled simulated contact points: 22,142
- Missing x/y coordinates: 0

Official seed-level KDE/spatial validation component: 0.206. This is the spatial component used in the original validation/composite framework.

## Figure list

- `figures/figA_interaction_fingerprint.pdf`
- `figures/figB_rate_convergence.pdf`
- `figures/figC_spatial_similarity.pdf`

## Scientific table list

- `tables/appendix_role_pair_distribution.csv`
- `tables/appendix_topic_distribution.csv`
- `tables/appendix_zone_distribution.csv`
- `tables/table1_spatial_negative_control.csv`

## Spatial negative-control interpretation

The pooled point-cloud null comparison is auxiliary. It asks whether the exported ABM contact cloud is closer to empirical locations than random spatial baselines under the same pooled method. In Table 1, the ABM reference distance is lower than the global random null mean by 0.099 and lower than the zone-preserving null mean by 0.035. The conservative Monte Carlo p-value is <0.002 for both nulls.

This supports the interpretation that the ABM captures hotspot structure beyond random placement and beyond broad zone proportions alone.

## KDE metric distinction

The report uses this hierarchy to avoid presenting two competing KDE values:

- Official seed-level KDE/spatial validation component: 0.206.
- Auxiliary pooled point-cloud reference distance: reported only in Table 1 as part of the null comparison.

These values use different aggregation/scaling and should not be compared directly. The seed-level component is the official mean of compact per-run validation distances. The pooled reference rebuilds one density from all 22,142 exported coordinates and is interpreted only relative to the null distances computed with that same pooled method. The figures therefore foreground the official spatial component and the null-comparison result, not the pooled reference as a headline score.

## Spatial figure diagnostic

Figure C now uses a left-hand point-overlay panel plus two right-hand heatmap panels: empirical above, simulated below. The point panel uses tiny deterministic display-only jitter to reveal stacked markers; statistics and exported coordinates remain unjittered. The previous dominance/overlap panel was removed because blending separately normalized densities from 359 empirical points and 22,142 simulated points produced a pale, foggy overlay that obscured the visual comparison. The heatmap panels use the same floorplan, spatial extent, bandwidth, and quantile-scaling rule. All empirical and simulated coordinates were included in the plotted extent, KDE density, and null comparison; the issue was visualization, not coordinate exclusion.

## Statistics used

For Part 1 baseline validation, the package reports repeated-run mean, SD, SE, 95% CI, coefficient of variation, residual gap, percent gap, null intervals, and conservative Monte Carlo p-values. Standardized effect sizes such as Hedges' g are not used here because they are more appropriate for Part 2 paired baseline-vs-intervention comparisons.

## Bed / high-acuity room coverage

Bed 8 has 19 empirical and 586 simulated interactions within the configured bed-proximity radius; coordinates are present and included in the plotted extent/KDE.

## Validity boundary

Validated for: normal-load baseline interaction ecology, F2F interaction rate, workflow-constrained embodied contact, proximity-valid logged interactions, broad role-pair/zone/spatial hotspot structure, and use as the frozen baseline for Part 2 spatial intervention experiments.

Not validated for: perfect free-text conversational semantics, individual-level psychology, clinical patient outcomes, generalization to other ED layouts without recalibration, or Part 2 intervention effects before those experiments are run.

## Limitations

Topic categories are approximate semantic labels, not full conversational validation. Station/desk interactions remain somewhat overrepresented relative to empirical data. The empirical variation check is descriptive because the empirical observation windows are sparse and uneven across dates.

## Conclusion

Part 1 is reporting-ready and frozen for Part 2: the interaction rate stabilizes close to the observed ED rate, the model reproduces recognizable place/people/topic structure, the spatial point cloud beats random placement nulls, and hard embodied-contact checks remain clean.
