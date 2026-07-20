# Paper Figure And Table Strategy

## Principle

Use figures only for results that need visual comparison. Use tables for exact multi-metric effects. Conceptual and quantitative graphics must remain visibly distinct.

## Part 1

Retain the three-figure validation package in `outputs/findings/part1_validation_n100/`:

1. interaction fingerprint;
2. rate convergence;
3. spatial similarity.

The spatial null comparison remains a table. The official seed-level KDE/spatial component and auxiliary pooled null distance are different metrics and should not be presented as competing scores.

## Part 2 Main Figures

1. **Four-condition intervention floorplan.** Show baseline, COCPIT transparency, moved NURSTA, and both. Distinguish physical barriers, visibility occluders, and station standing/attractor semantics.
2. **Comparable interaction spatial density.** Use the validation-counted `interaction_events.csv` points with identical spatial extent, binning/bandwidth, and scale across conditions.
3. **Final F2F sensitivity response.** Use the final low/high paired-delta figure as a main or near-main robustness visual. It communicates effect magnitude and uncertainty more directly than the verdict matrix.

## Part 2 Tables

1. Experiment design and integrity.
2. Corrected targeted spatial-affordance and movement metrics.
3. Main paired intervention effects by scenario.
4. Completed-n100 robustness supplement.
5. Compact sensitivity table listing the four active factors, the assumption each probes, and the qualitative outcome. Keep the final 528-row detailed table in the supplement.

Preferred compact sensitivity table:

| Active factor | Probe | Qualitative outcome |
|---|---|---|
| Visibility response | Visible-opportunity response strength | Core COCPIT and ecology claims stable |
| Pressure activation threshold | Earlier/later deterministic pressure filtering | Core claims stable; pressure outcome mixed at the low setting |
| Repeat-contact delay | Shorter/longer repeat-contact cooldown | Core claims stable |
| Station-return tendency | Nurse post-task station return/check behavior | Core claims stable; pressure outcome mixed |

## Current Draft Disposition

- Seven existing Part 2 figures are diagnostic drafts, not the final paper set.
- Operational tradeoff, movement/copresence, mechanism funnel, and social redistribution are better as tables.
- The mixed-scale forest plot is supplement-only or should be redesigned.
- The old space-syntax draft must not be reused because it preceded the visibility-layer repair.
- The final two-phase verdict matrix is supplementary diagnostics, not the preferred sensitivity figure.

## Full-Project Conceptual Figures

Possible presentation/thesis figures include the Part 1-Part 2-Part 3 roadmap, ED ecosystem, reason-gated interaction mechanism, routine versus high-load scenario, bounded Part 3 cognitive layer, and a `space -> experience -> coordination -> operation` summary.

AI assistance is acceptable for conceptual composition or stylized explanatory scenes. AI must not generate validation charts, KDE maps, effect plots, statistical tables, floorplan measurements, or any exact quantitative result.

## Deprecated

- Generic dashboard collections as the main Part 2 story.
- Presenting the non-informative crowding-suppression contrast as demonstrated invariance.
- Treating composite distance from the Part 1 empirical ecology as an intervention success target.
