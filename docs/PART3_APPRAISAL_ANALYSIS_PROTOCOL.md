# Part 3 Appraisal Analysis Protocol

## Status and scope

This protocol is locked before Qwen generates the 800 end-of-shift appraisal
responses. The machine-readable source of truth is
`manifests/part3_appraisal_analysis_spec.json`; the analysis script validates
and uses that file directly.

The appraisals are synthetic, persona-conditioned interpretations of completed
simulated shifts. They are not Zurich ED staff testimony, population preference
estimates, psychometric measurements, or evidence of clinical outcomes.

## Design and estimands

The paired observation fixes scenario, seed, assignment round, persona, staff
role, and staff id, then compares all four spatial conditions for that same
unit. The seed remains the inferential unit: staff-shift deltas are averaged
within seed before uncertainty is calculated, avoiding false precision from
shifts that share an exogenous simulation stream.
The packet sample retains all ten seeds for every scenario and persona. One
four-condition-paired staff shift is selected per seed; staff roles are
allocated as evenly as possible (4/3/3) within each scenario-persona block.
This yields 100 paired units, 400 agent-shifts, 400 survey bundles, and 400
interview bundles.
The primary contrasts are COCPIT only, NURSTA only, and Both minus Baseline,
separately within scenario, persona, and appraisal dimension.

For every contrast, `scripts/analysis/analyze_part3_appraisals.py` reports the
equal-seed-weighted paired mean difference, SD and SE across seed-level means,
Student-t 95% confidence interval, median seed-level difference, and
positive/negative seed shares. Role-specific mean directions are retained as
a descriptive convergence check. Effect sizes and uncertainty are primary;
the study will not create a family of binary significance claims after seeing
the responses.

## Emergent-experience summaries

- **Convergence:** seed-mean sign consistency and the number of staff roles
  whose descriptive mean direction agrees with the overall direction.
- **Polarization:** change from Baseline in the SD and range of the five persona
  means for each dimension. This means dispersion among designed orientations,
  not psychological polarization in a real workforce.
- **Worst-case fit:** the smallest persona-specific paired effect on overall
  person-space fit, its interval, and the count of personas with positive and
  negative mean effects.

No unregistered composite score is primary. The seven anchored dimensions
remain visible so a favorable coordination score cannot conceal interruption,
privacy, accessibility, or continuity tradeoffs.

Raw deltas retain the survey direction. Higher scores are favorable for every
dimension except interruption burden, where lower is favorable. The analysis
exports that preferred direction explicitly and does not silently reverse or
combine scores.

## Qualitative analysis

One complete interview per scenario-condition-persona cell is selected
deterministically for mandatory manual review. Review checks evidence grounding,
naturalness, score/prose coherence, separation of observation from conjecture,
and credible tradeoffs for design suggestions. Responses failing these checks
are not used as illustrative quotations or thematic evidence.

The analyzer exports a coding template but performs no automated theme
discovery. Themes are coded only after the grounding review. Reported themes
must be linked back to the paired simulated evidence and described as synthetic
design probes, never as statements made by real clinicians.

## Required outputs

- `survey_responses.csv`
- `interview_responses.csv`
- `paired_survey_effects.csv`
- `between_persona_polarization.csv`
- `worst_case_person_space_fit.csv`
- `interview_manual_review_sample.csv`
- `interview_manual_coding_template.csv`
- `appraisal_analysis_summary.json`

Final figures and paper tables are deliberately deferred until the response
verification and manual-review gates pass.
