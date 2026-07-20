# Part 2 Spatial Interventions Protocol And Completed Status

## Completed Design

- Batch: `part2_spatial_interventions_n100`
- Scenarios: `normal_load`, `high_load_high_acuity`
- Conditions: `baseline`, `cockpit_only`, `nursta_only`, `both`
- Seeds: 1-100, paired within scenario
- Runs: 800
- Duration: 43,200 seconds; warmup: 7,200 seconds; start hour: 10
- Canonical results: `outputs/latest/ABM_results/part2_spatial_interventions_n100/`
- Integrity: 800 summaries, 800 interaction event files, 800/800 workflow PASS, paired completeness true, hard gates clean

Seed `k` baseline is compared with seed `k` in every intervention condition within the same scenario.

## Intervention Semantics

- `baseline`: frozen Part 1 spatial condition.
- `cockpit_only`: COCPIT partitions become transparent in the perception/visibility layer. Physical collision and routing geometry are unchanged.
- `nursta_only`: NURSTA desk/station location, walkable standing semantics, station-return anchors, and related route exposure change together.
- `both`: combines the two interventions.

Physical `effective_walls` and perceptual `visibility_walls` are distinct. COCPIT transparency should change objective and experienced visibility without changing route distance. NURSTA relocation may change NURSTA visibility, station-to-bed routes, and station/corridor exposure.

## Analytical Funnel

1. Objective spatial affordance: isovists, targeted intervisibility, local visibility graph, route directness.
2. Experienced visibility: staff-specific exposure during simulated work.
3. Opportunity conversion: raw percepts, eligible/selected opportunities, intents, successes, misses.
4. Movement/copresence: distance, dwell, route/station/corridor exposure.
5. Interaction ecology: F2F/hour, role-pair, patient-facing/HCW, zone, topic, duration.
6. Operational pressure: ED pressure, waiting, occupancy, arrivals/completions, high-acuity coordination.
7. Integrity: workflow, routes, stalls, proximity, and coordinate hard gates.

## Repaired Spatial-Affordance Audit

An earlier reporting calculation used physical walls for visibility and therefore failed to represent transparent COCPIT partitions. The repaired reporting logic uses `PerceptionService`/condition-specific `visibility_walls` for isovists and intervisibility while retaining physical walls for routes.

Key repaired values:

| Metric | Baseline | COCPIT | NURSTA | Both |
|---|---:|---:|---:|---:|
| COCPIT isovist area (m2) | 37.78 | 224.10 | 37.78 | 224.10 |
| Beds visible from COCPIT | 0 | 7 | 0 | 7 |
| NURSTA isovist area (m2) | 26.13 | 26.13 | 61.19 | 61.19 |
| Beds visible from NURSTA | 1 | 1 | 2 | 2 |
| COCPIT mean route to beds (m) | 12.83 | 12.83 | 12.83 | 12.83 |
| NURSTA mean route to beds (m) | 17.29 | 17.29 | 14.83 | 14.83 |

COCPIT-to-NUROPE intervisibility changes from absent to present under COCPIT transparency. The proportion of interactions in high-visibility grid cells rises from about 11.2% to 35.6% in normal load and from about 12.6% to 33.8% in high load under `cockpit_only`.

These are reproducible space-syntax-style proxies, not formal DepthmapX axial/segment outputs. The interaction-density association is not by itself causal evidence.

## Completed-n100 Interpretation

- COCPIT transparency strongly increases objective visibility, raw/eligible visibility opportunities, approaches, and F2F/hour.
- NURSTA-only is weak or near-neutral at default behavior settings and remains comparatively small/uncertain across the completed station-return sensitivity screen.
- `both` is mostly COCPIT-driven.
- COCPIT conditions shift relative interaction share toward HCW-HCW/corridor coordination and away from patient-facing/bedside interaction.
- Operational pressure differences are comparatively modest.
- Composite distance from the Part 1 empirical baseline is not a Part 2 success criterion because interventions are expected to alter the baseline ecology.

## Completed Parameter Sensitivity

The original n20-per-cell add-on completed 1,280 integrity-clean runs. Visibility response, repeat-contact delay, and station-return tendency are informative. The crowding-suppression factor is excluded because it was inactive on the operative path. Its 320-run `SCENARIO_PRESSURE_ACTION_THRESHOLD` replacement completed with integrity and sensitivity-effectiveness PASS. The authoritative set contains 960 retained Phase 1 runs plus 320 Phase 2 replacement runs.

Across all eight factor-level verdict rows, the COCPIT F2F, visibility-driven contact, HCW-HCW/corridor redistribution, patient-facing/bedside reduction, and smaller/local NURSTA conclusions remain stable. Operational-pressure conclusions are secondary: four cells are stable and four mixed.

## Statistics

Primary estimates are paired seed deltas (`condition - baseline`) with mean, SD, SE, 95% CI, median/IQR, sign consistency, and paired tests where useful. Paired standardized effects are secondary. P-values do not replace effect magnitude or uncertainty.

## Do Not Claim

The results are conditional counterfactuals in this calibrated ED model. They are not empirical proof of post-intervention benefit, clinical outcome evidence, or universal design recommendations.
