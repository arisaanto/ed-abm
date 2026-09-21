# Part 3 Appraisal Analysis Protocol

## Status and scope

This protocol was locked before Qwen generated the 800 end-of-shift appraisal
responses. The machine-readable analysis specification is
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

## Prespecified qualitative analysis

One complete interview per scenario-condition-persona cell was selected
deterministically for a prespecified review sample. The intended review checked
evidence grounding, naturalness, score/prose coherence, separation of
observation from conjecture, and credible tradeoffs for design suggestions.
As documented below, the completed study used an independent model-assisted
audit rather than human review. Responses failing those checks were not used as
illustrative quotations or thematic evidence.

The analyzer exports a coding template but performs no automated theme
discovery. Themes are coded only after the grounding review. Reported themes
must be linked back to the paired simulated evidence and described as synthetic
design probes, never as statements made by real clinicians.

## Post-hoc review disposition

The prespecified human-review step was not completed. At the investigator's
request, OpenAI Codex instead performed an independent model-assisted audit of
the same 40 prespecified bundles and all 240 public answers. This is a protocol
deviation and must not be described as human review, manual human coding, or
inter-rater validation.

The Codex audit assessed evidence grounding, claim-layer separation,
naturalness, directness, context labeling, and counterfactual tradeoffs. Its
final record retained 140 of 240 answers for scientific use, excluded 100,
and selected 109 excerpts for display. Thirty-seven retained design answers
underlie the exploratory category counts. Thirty-two of the 40 bundles had at
least three usable answers, and one display cell had no acceptable excerpt.

The five-category regular-expression dictionary was also compared with the
model-assisted audit. Its macro F1 was 0.704 and its minimum category F1 was
0.56, which was inadequate for final frequency claims over all 400 interviews.
Consequently, the paper table reports only the 37 retained design answers
from the prespecified audit sample. Categories may overlap. These counts are
exploratory synthetic design suggestions, not human thematic findings.

The frozen review record is
`manifests/part3_evolving_independent_interview_review.json`; reproducible exports are in
`outputs/findings/part3_cognitive_personas_n10/review/`.

## Required outputs

- `survey_responses.csv`
- `interview_responses.csv`
- `paired_survey_effects.csv`
- `between_persona_polarization.csv`
- `worst_case_person_space_fit.csv`
- `interview_manual_review_sample.csv`
- `interview_manual_coding_template.csv`
- `appraisal_analysis_summary.json`
- `reviewed_interview_sample.csv`
- `reviewed_counterfactual_codes.csv`
- `dictionary_validation.csv`
- `review_summary.json`

The original human-review gate remains unmet. Final qualitative figures and
tables may use the independent Codex audit only when its provenance and the
absence of human review are disclosed explicitly.
