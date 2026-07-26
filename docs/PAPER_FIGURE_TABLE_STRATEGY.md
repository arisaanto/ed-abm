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
3. **Spatial coordination ecology.** Show the Baseline-to-Both reduction in
   interaction separation and role-location segregation.

Retain the final low/high paired-delta sensitivity figure as a supplementary
robustness visual. It communicates effect magnitude and uncertainty more
directly than the verdict matrix.

## Part 2 Tables

1. movement effects;
2. whole-space VGA-style spatial metrics;
3. main paired intervention effects by scenario;
4. coordination-friction profile;
5. role-specific movement and staff-contact decomposition for Both versus
   Baseline;
6. compact sensitivity values and the corrected detailed sensitivity
   supplement.

The paper-facing tables intentionally omit station occupancy from the movement
table, the redundant through-vision proxy from the VGA table, and duplicate
separation/segregation change columns from the coordination profile. Exact
paired changes remain in the corresponding figures or source CSVs.

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

## Part 3

Use the static persona atlas as an **unnumbered preview** linked to the
interactive explorer. It is not a substantive result figure. OCEAN is a
presentation crosswalk, not a measured psychometric result.

Retain three numbered figures:

1. **Cognitive architecture.** Separate the during-shift causal decision loop
   from the read-only post-shift appraisal and interview pathway.
2. **Common affordance, heterogeneous uptake.** Contrast the nearly common
   visibility gain with persona-specific LLM engagement changes.
3. **Inclusive person-space fit.** Show the five persona distributions and the
   ensemble average and fit floor for Baseline versus Both. The primary result
   is the high-load increase in average fit and fit floor together with reduced
   between-orientation disparity.

Retain three main tables:

1. high-load persona appraisal profiles, combining four core quantitative
   dimensions with one representative exact design response;
2. ensemble average, fit-floor, and between-orientation-disparity effects;
3. recurrent synthetic design priorities as a persona-by-priority percentage
   matrix, with explicit overlap and synthetic-data boundaries.

Keep the matched 2 x 2 persona-conditioning by grounded-memory ablation as one
compact appendix table.

Use bold table values sparingly: only prespecified defaults, the principal
reference row, recurrent top-ranked qualitative priorities, or estimates whose
95% confidence intervals exclude zero. Do not bold an entire effect column when
some intervals include zero.

Do not promote trace-fit associations to a paper table. The movement,
experienced-visibility, interaction-volume, and interaction-spacing
correlations are weak, scenario-dependent, and partly endogenous because the
appraisal model received compressed trace evidence. Retain them only as an
internal analytic audit.

## Full-Project Conceptual Figures

Retain two cross-study explanatory figures:

1. **Study overview.** Part 1 validates, Part 2 intervenes, and Part 3
   interprets heterogeneous uptake and person-space fit.
2. **Interaction-generation pipeline.** Proximity, mutual visibility, and
   operational feasibility create an actionable opportunity; a bounded policy
   records either an F2F interaction or a missed opportunity.

The Part 3 cognitive-architecture figure already contains the evidence
retrieval-to-appraisal pathway. Do not add a separate memory-to-response figure
unless a venue specifically requests more methodological detail.

AI assistance is acceptable for conceptual composition or stylized explanatory scenes. AI must not generate validation charts, KDE maps, effect plots, statistical tables, floorplan measurements, or any exact quantitative result.

## Deprecated

- Generic dashboard collections as the main Part 2 story.
- Presenting the non-informative crowding-suppression contrast as demonstrated invariance.
- Treating composite distance from the Part 1 empirical ecology as an intervention success target.
