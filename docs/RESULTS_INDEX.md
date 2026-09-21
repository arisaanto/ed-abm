# Public results index

The files below are selected, aggregate outputs of the completed study. PNG
figures are easy to preview on GitHub; matching PDFs are for print. CSV files
provide the numbers behind reported tables. The manuscript supplies the full
methods, captions, and interpretation. These files alone do not reproduce the
analysis from restricted shadowing records or the complete run trees.

## Study design

Start with the [study overview](../outputs/findings/study_overview/figures/study_design_overview.png),
[interaction pipeline](../outputs/findings/study_overview/figures/interaction_generation_pipeline.png),
and [modelled floor plan](../outputs/findings/study_overview/figures/study-context-floorplan.png).

## Part 1: baseline resemblance

The [interaction-composition figure](../outputs/findings/part1_validation_n100/figures/figA_interaction_fingerprint.png)
and [spatial-resemblance figure](../outputs/findings/part1_validation_n100/figures/figC_spatial_similarity.png)
show the main comparison with the shadowing extract. The
[negative-control table](../outputs/findings/part1_validation_n100/tables/table1_spatial_negative_control.csv),
[multiscale summary](../outputs/findings/part1_multiscale_kde/multiscale_summary.csv),
and [replication-convergence figure](../outputs/findings/part1_validation_n100/figures/part1-replication-convergence.png)
give supporting diagnostics. The comparison used observations that also
informed model development, so it is not an independent validation test.

## Part 2: spatial interventions

The [layout figure](../outputs/findings/part2_spatial_interventions_n100/figures/figA_intervention_layout.png)
shows the four conditions. The
[interaction-redistribution figure](../outputs/findings/part2_spatial_interventions_n100/figures/figB_interaction_redistribution.png)
and [paired-effects table](../outputs/findings/part2_spatial_interventions_n100/tables/tableC_paired_intervention_effects.csv)
show the principal contact results. Supporting tables cover
[movement](../outputs/findings/part2_spatial_interventions_n100/tables/tableA_movement_effects.csv)
and [role-specific effects](../outputs/findings/part2_spatial_interventions_n100/tables/tableE_role_specific_spatial_effects.csv).
The [local-sensitivity figure](../outputs/findings/part2_parameter_sensitivity_n20/figures/figA_sensitivity_response.png)
and [visibility-audit summary](../outputs/findings/part2_visibility_copresence_n20/visibility_copresence_summary.csv)
are checks on the within-model mechanism.

## Part 3: synthetic personas

The [persona-results figure](../outputs/findings/part3_cognitive_personas_n10/figures/figB_common_affordance_different_uptake.png)
and [person–space-fit figure](../outputs/findings/part3_cognitive_personas_n10/figures/figC_inclusive_person_space_fit.png)
show the main results. The tables report
[appraisals](../outputs/findings/part3_cognitive_personas_n10/tables/tableA_persona_appraisal_profiles.csv),
[decisions under common situations](../outputs/findings/part3_cognitive_personas_n10/tables/appendix_tableC_common_situation_construct_checks.csv),
[the accumulated-experience check](../outputs/findings/part3_cognitive_personas_n10/tables/appendix_tableA_experience_ablation.csv),
and [questionnaire comparison](../outputs/findings/part3_cognitive_personas_n10/tables/tableD_questionnaire_convergence.csv).
The [persona explorer](https://arisaanto.github.io/ed-abm/) is an interactive
view of selected results. All personas and appraisals are synthetic.

The [Part 3 review folder](../outputs/findings/part3_cognitive_personas_n10/review/)
is an **audit trail of synthetic text**, not a collection of clinician
interviews. In particular, `reviewed_interview_sample.csv` contains both
retained and **excluded** model-generated answers. Use its
`review_grounding_pass`, `review_claim_layer_pass`, and related review columns
before citing or displaying any answer. The audit was model-assisted; the
`human_review_completed` field is false. The [review summary](../outputs/findings/part3_cognitive_personas_n10/review/review_summary.json)
records the final disposition. For the curated public examples, start with the
[representative-response table](../outputs/findings/part3_cognitive_personas_n10/tables/appendix_tableB_representative_responses.csv).

Participant-linked shadowing records, approximate participant-date rate
diagnostics, and complete simulation traces are not published here. The
[README](../README.md#data-availability) explains the access boundary.
