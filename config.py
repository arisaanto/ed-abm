"""Central configuration for the Phase 1 ED ABM.

All tunable parameters, file paths, task definitions, staff composition,
spatial constants, and visualization settings live here so later phases can
swap implementations without chasing hidden constants across the codebase.
"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MPL_CONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", OUTPUTS_DIR / ".mplconfig"))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", OUTPUTS_DIR / ".xdg_cache"))
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_HOME)
PROJECT_ROOT = BASE_DIR.parent
RESEARCH_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARIS_DIR = BASE_DIR.parent.parent / "ARIS"

WALL_POSITIONS_PATH = DATA_DIR / "wall_positions.json"
ZONE_BOUNDARIES_PATH = DATA_DIR / "zone_boundaries.json"
OUTPUT_FIGURE_PATH = OUTPUTS_DIR / "simulation_output.png"
SIMULATION_OVERVIEW_PATH = OUTPUTS_DIR / "simulation_overview.png"
EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH = RESEARCH_OUTPUTS_DIR / "empirical_topic_distributions.json"
EMPIRICAL_SHADOWING_CSV_PATH = (
    PROJECT_ROOT / "shadowing_with-participant-info.csv"
    if (PROJECT_ROOT / "shadowing_with-participant-info.csv").exists()
    else ARIS_DIR / "shadowing_with-participant-info.csv"
)
CALIBRATION_SUMMARY_PATH = OUTPUTS_DIR / "calibration_summary.json"
CALIBRATION_DISTANCE_COMPONENTS_PATH = OUTPUTS_DIR / "calibration_distance_components.json"
POSTERIOR_PARAMETER_DISTRIBUTIONS_PATH = OUTPUTS_DIR / "posterior_parameter_distributions.json"
ABC_HISTORY_DB_PATH = OUTPUTS_DIR / "abc_history.db"
EXPERIMENT_SUMMARY_PATH = OUTPUTS_DIR / "experiment_summary.json"
EXPERIMENT_METRICS_CSV_PATH = OUTPUTS_DIR / "experiment_metrics.csv"
EXPERIMENT_HEATMAP_GRID_PATH = OUTPUTS_DIR / "experiment_heatmaps.png"
CONDITION_HEATMAPS_PATH = OUTPUTS_DIR / "condition_heatmaps.png"
CONDITION_DELTA_TABLE_PATH = OUTPUTS_DIR / "condition_delta_table.txt"
INTERACTION_TRANSCRIPT_SAMPLE_PATH = OUTPUTS_DIR / "interaction_transcript_sample.json"
PRIOR_PREDICTIVE_DISTANCES_PATH = OUTPUTS_DIR / "prior_predictive_distances.json"
LLM_VALIDATION_PATH = OUTPUTS_DIR / "llm_validation.json"
VISIBILITY_ANALYSIS_PATH = OUTPUTS_DIR / "visibility_metrics.json"
AGENT_SHIFT_NARRATIVES_PATH = OUTPUTS_DIR / "agent_shift_narratives.json"
THESIS_FIGURES_DIR = OUTPUTS_DIR / "figures"
LATEST_OUTPUT_DIR = OUTPUTS_DIR / "latest"
MEMORY_TRACE_DIR = OUTPUTS_DIR / "memory_traces"
RETRIEVAL_TRACE_DIR = OUTPUTS_DIR / "retrieval_traces"
REFLECTION_DIR = OUTPUTS_DIR / "reflections"
INTERVIEW_DIR = OUTPUTS_DIR / "interviews"
PART3_OUTPUT_DIR = OUTPUTS_DIR / "part3"
PART3_COGNITIVE_PERSONAS_PATH = PART3_OUTPUT_DIR / "cognitive_personas.json"
PART3_COGNITIVE_PERSONA_CARDS_PATH = PART3_OUTPUT_DIR / "cognitive_persona_cards.md"
PART3_PERSONA_ASSIGNMENTS_PATH = PART3_OUTPUT_DIR / "balanced_persona_assignments.csv"
PART3_PROTOCOL_PATH = PART3_OUTPUT_DIR / "part3_protocol.json"
VERIFICATION_REPORT_PATH = OUTPUTS_DIR / "verification_report.md"
ABLATION_SUMMARY_PATH = OUTPUTS_DIR / "ablation_summary.json"
MECHANISM_AUDIT_SUMMARY_PATH = LATEST_OUTPUT_DIR / "mechanism_audit_summary.json"
MECHANISM_AUDIT_DIAGNOSTICS_PATH = LATEST_OUTPUT_DIR / "mechanism_audit_diagnostics.json"
VALIDATION_SUMMARY_PATH = LATEST_OUTPUT_DIR / "validation_summary.json"
SCENARIO_SANITY_COMPARISON_PATH = LATEST_OUTPUT_DIR / "scenario_sanity_comparison.json"
EMPIRICAL_WINDOW_SUMMARY_PATH = LATEST_OUTPUT_DIR / "empirical_window_summary.json"
SCOPE_ALIGNMENT_AUDIT_PATH = LATEST_OUTPUT_DIR / "scope_alignment_audit.json"
ECOLOGICAL_SCOPE_AUDIT_PATH = LATEST_OUTPUT_DIR / "ecological_scope_audit.json"
SPATIAL_CLUSTER_AFFORDANCE_AUDIT_JSON_PATH = LATEST_OUTPUT_DIR / "spatial_cluster_affordance_audit.json"
SPATIAL_CLUSTER_AFFORDANCE_AUDIT_MD_PATH = LATEST_OUTPUT_DIR / "spatial_cluster_affordance_audit.md"
CARE_AREA_ROSTER_SUMMARY_JSON_PATH = LATEST_OUTPUT_DIR / "care_area_roster_summary.json"
CARE_AREA_ROSTER_SUMMARY_MD_PATH = LATEST_OUTPUT_DIR / "care_area_roster_summary.md"
CARE_AREA_FLOORPLAN_OVERLAY_PATH = LATEST_OUTPUT_DIR / "care_area_floorplan_overlay.png"
CARE_AREA_FLOORPLAN_OVERLAY_METADATA_PATH = LATEST_OUTPUT_DIR / "care_area_floorplan_overlay_metadata.json"
CARE_AREA_FLOORPLAN_OVERLAY_SUMMARY_PATH = LATEST_OUTPUT_DIR / "care_area_floorplan_overlay_summary.md"
VALIDATION_TARGET_COMPARISON_PATH = LATEST_OUTPUT_DIR / "validation_target_comparison.json"
PART1_BASELINE_FAILURE_DIAGNOSTIC_PATH = LATEST_OUTPUT_DIR / "part1_baseline_failure_diagnostic.md"
ENSEMBLE_SUMMARY_PATH = LATEST_OUTPUT_DIR / "ensemble_summary.json"
MECHANISM_AUDIT_DASHBOARD_PATH = THESIS_FIGURES_DIR / "mechanism_audit_dashboard.png"
VALIDATION_FIT_DASHBOARD_PATH = THESIS_FIGURES_DIR / "validation_fit_dashboard.png"
PAIRED_SEED_UNCERTAINTY_PATH = THESIS_FIGURES_DIR / "paired_seed_uncertainty.png"
MODEL_SUFFICIENCY_SUMMARY_PATH = LATEST_OUTPUT_DIR / "model_sufficiency_summary.json"
MODEL_SUFFICIENCY_DASHBOARD_PATH = THESIS_FIGURES_DIR / "model_sufficiency_dashboard.png"
ABLATION_DASHBOARD_PATH = THESIS_FIGURES_DIR / "ablation_dashboard.png"
ABLATION_DIAGNOSTICS_PATH = OUTPUTS_DIR / "ablation_diagnostics.json"
ABLATION_DECISION_COUNTS_PATH = THESIS_FIGURES_DIR / "ablation_decision_counts.png"
CONDITION_COMPARISON_DASHBOARD_PATH = THESIS_FIGURES_DIR / "condition_comparison_dashboard.png"
AGENT_EXPERIENCE_CARDS_PATH = OUTPUTS_DIR / "agent_experience_cards.md"
TIMELINE_PATH = OUTPUTS_DIR / "timeline.md"
MODEL_STATUS_PATH = OUTPUTS_DIR / "model_status.md"
OUTPUTS_README_PATH = OUTPUTS_DIR / "README.md"
FINDINGS_DIR = OUTPUTS_DIR / "findings"
PART1_PERCEPTION_MECHANISM_AUDIT_FINDING_PATH = FINDINGS_DIR / "part1_perception_mechanism_audit.md"
PART1_INTERACTION_MECHANISM_EXAMPLES_PATH = FINDINGS_DIR / "part1_interaction_mechanism_examples.md"
PERCEPTION_MECHANISM_AUDIT_JSON_PATH = LATEST_OUTPUT_DIR / "perception_mechanism_audit.json"
PERCEPTION_MECHANISM_AUDIT_MD_PATH = LATEST_OUTPUT_DIR / "perception_mechanism_audit.md"
PERCEPTION_AUDIT_JSON_PATH = LATEST_OUTPUT_DIR / "perception_audit.json"
PERCEPTION_AUDIT_MD_PATH = LATEST_OUTPUT_DIR / "perception_audit.md"
PERCEPTION_EVENT_EXAMPLES_PATH = LATEST_OUTPUT_DIR / "perception_event_examples.md"
PART2_CONDITION_MECHANISM_AUDIT_JSON_PATH = LATEST_OUTPUT_DIR / "part2_condition_mechanism_audit.json"
PART2_CONDITION_MECHANISM_AUDIT_MD_PATH = LATEST_OUTPUT_DIR / "part2_condition_mechanism_audit.md"
PART1_BASELINE_SUMMARY_PATH = FINDINGS_DIR / "part1_baseline_summary.md"
PART1_BASELINE_FIT_PATH = FINDINGS_DIR / "part1_baseline_fit.png"
PART1_RATE_FIT_PATH = FINDINGS_DIR / "part1_rate_fit.png"
PART1_SHARE_FIT_PATH = FINDINGS_DIR / "part1_share_fit.png"
PART2_SCENARIO_SUMMARY_PATH = LATEST_OUTPUT_DIR / "part2_scenario_summary.json"
PART2_PRELIMINARY_NOTES_LATEST_PATH = LATEST_OUTPUT_DIR / "part2_preliminary_notes.md"
PART2_PRELIMINARY_EFFECTS_LATEST_PATH = LATEST_OUTPUT_DIR / "part2_preliminary_effects.png"
PART2_PRELIMINARY_EFFECTS_HTML_LATEST_PATH = LATEST_OUTPUT_DIR / "part2_preliminary_effects.html"
PART2_SCENARIO_MATRIX_PATH = FINDINGS_DIR / "part2_scenario_matrix.md"
PART2_PRELIMINARY_RESULTS_PATH = FINDINGS_DIR / "part2_preliminary_results.md"
PART2_PRELIMINARY_EFFECTS_PATH = FINDINGS_DIR / "part2_preliminary_effects.png"
PART2_MECHANISM_EXPLANATION_PATH = FINDINGS_DIR / "part2_mechanism_explanation.md"
PART2_CLUSTER_PLAN_PATH = FINDINGS_DIR / "part2_cluster_plan.md"

RANDOM_SEED = 49
RUN_BASE_SEED = 42

TIMESTEP_SECONDS = 1
SIMULATION_DURATION_SECONDS = 43_200
HOURLY_REPORT_INTERVAL_SECONDS = 3_600

ARRIVAL_RATE_PER_SECOND = 0.002
MOVEMENT_SPEED_METERS_PER_SECOND = 1.0
PERCEPTION_RADIUS_METERS = 10.0
ARRIVAL_TOLERANCE_METERS = 0.25
WALL_CLEARANCE_METERS = 0.2

INTERACTION_DISTANCE_METERS = 1.5
INTERACTION_COOLDOWN_SECONDS = 60
SHARED_STATION_AWARENESS_DISTANCE_METERS = 2.2
TASK_DRIVEN_FOV_RELAXATION_PROBABILITY = 0.68
CORRIDOR_CLOSE_PASS_DISTANCE_METERS = 1.0
CORRIDOR_SOFT_FOV_RELAXATION_PROBABILITY = 0.42

DOCTOR_RECHECK_INTERVAL_SECONDS = 60
DOCTOR_RECHECK_MAX_ATTEMPTS = 5
PATIENT_STATE_SWEEP_INTERVAL_SECONDS = 300
MEDICAL_EVALUATION_STALE_WAIT_SECONDS = 1_800
SIMULATION_HEALTH_CHECK_INTERVAL_SECONDS = 3_600
STAFF_TRANSIT_STALL_TIMEOUT_SECONDS = 300

PATIENT_ID_START = 1_000
HOME_POSITION_WANDER_RADIUS = 0.3
STAFF_IDLE_POSITION_NOISE_METERS = 0.3
IDLE_PATROL_TRIGGER_SECONDS = 45
COORDINATION_PATROL_ROUTE_KEYS = [
    "entry_door",
    "corr01_mid",
    "corr04_door",
    "corr05_door",
    "nurope_door",
    "nursta_west",
    "nursta_east",
    "right_corridor_hub",
]
DOCTOR_PATROL_ROUTE_KEYS = [
    "right_room_door",
    "left_room_door",
    "corr01_mid",
    "corr04_door",
    "corr05_door",
    "corr07_lower",
]
NURSE_CIRCULATION_ROUTE_KEYS_BY_HOME = {
    "NURSTA": ["nursta_west", "nursta_east", "corr04_door", "corr05_door", "lower_corridor_hub"],
    "NUROPE": ["nurope_door", "corr04_door", "corr01_west", "left_upper_bed_access"],
    "COCPIT": ["corr01_mid", "lower_corridor_hub", "right_corridor_hub"],
}
NURSE_PATROL_TO_BED_PROBABILITY = 0.35
NURSE_SECONDARY_STATION_PROBABILITY = 0.05
NURSE_SECONDARY_STATION_DURATION = (60, 180)
SECONDARY_STATION_ARRIVAL_INTERACTION_PROBABILITY = 0.10
INTERACTION_POSITION_NOISE_METERS = 0.35
OPPORTUNISTIC_CORRIDOR_INTERACTION_PROBABILITY = 1.0
OPPORTUNISTIC_CORRIDOR_INTERACTION_COOLDOWN_SECONDS = 30
CORRIDOR_ENCOUNTER_DISTANCE_METERS = 2.3
OPPORTUNISTIC_CORRIDOR_TARGET_SHARE_MIN = 0.20
OPPORTUNISTIC_CORRIDOR_TARGET_SHARE_MAX = 0.35
OPPORTUNISTIC_STATION_INTERACTION_PROBABILITY = 0.25
OPPORTUNISTIC_STATION_INTERACTION_COOLDOWN_SECONDS = 150
INITIAL_STATION_INTERACTION_GRACE_SECONDS = 90
STATION_INTERACTION_MIN_DWELL_SECONDS = 30
STATION_EPISODE_LEAVE_GRACE_SECONDS = 20
STATION_EPISODE_RADIUS_METERS = 2.4
NURSE_DOCTOR_HANDOFF_INTERACTION_PROBABILITY = 0.42
HIGH_ACUITY_NURSE_DOCTOR_HANDOFF_INTERACTION_PROBABILITY = 0.85
STATION_COORDINATION_DISTANCE_METERS = 3.2
COORDINATION_HUB_INTERACTION_PROBABILITY = 0.015
COORDINATION_HUB_INTERACTION_COOLDOWN_SECONDS = 220
SECONDARY_STATION_DURATION_MEAN = 120.0
COORDINATION_PATROL_PROBABILITY = 0.0002
NURSE_CIRCULATION_PROBABILITY = 0.00016
DOCTOR_CIRCULATION_PROBABILITY = 0.00014
STAFF_CIRCULATION_MIN_IDLE_SECONDS = 60
DOCTOR_COREVIEW_PROBABILITY = 0.00012
DOCTOR_COREVIEW_COOLDOWN_SECONDS = 900
COCPIT_IDLE_COLLOCATION_INTERACTION_MULTIPLIER = 0.55
DOCTOR_RESPONSE_BIAS = 1.0
BEDSIDE_LINGER_MULTIPLIER = 1.0

STAFF_COUNTS = {
    "CoordinationNurse": 1,
    "Nurse": 5,
    "Doctor": 3,
}

CHECKLIST = [
    "Placement",
    "Initial Nursing Assessment",
    "Medical Evaluation",
    "Diagnostics",
    "Treatment",
    "Reassessment",
    "Disposition Decision",
    "Discharge or Admission",
]

TASK_INDEX_BY_NAME = {name: index for index, name in enumerate(CHECKLIST)}

PLACEMENT_TASK_NAME = "Placement"
INITIAL_NURSING_ASSESSMENT_TASK_NAME = "Initial Nursing Assessment"
MEDICAL_EVALUATION_TASK_NAME = "Medical Evaluation"
DIAGNOSTICS_TASK_NAME = "Diagnostics"
TREATMENT_TASK_NAME = "Treatment"
REASSESSMENT_TASK_NAME = "Reassessment"
DISPOSITION_DECISION_TASK_NAME = "Disposition Decision"
DISCHARGE_TASK_NAME = "Discharge or Admission"

PLACEMENT_TASK_INDEX = TASK_INDEX_BY_NAME[PLACEMENT_TASK_NAME]
MEDICAL_EVALUATION_TASK_INDEX = TASK_INDEX_BY_NAME[MEDICAL_EVALUATION_TASK_NAME]
DISCHARGE_TASK_INDEX = TASK_INDEX_BY_NAME[DISCHARGE_TASK_NAME]

ROLE_PERMISSIONS = {
    "CoordinationNurse": [PLACEMENT_TASK_INDEX],
    "Nurse": [
        TASK_INDEX_BY_NAME[INITIAL_NURSING_ASSESSMENT_TASK_NAME],
        TASK_INDEX_BY_NAME[DIAGNOSTICS_TASK_NAME],
        TASK_INDEX_BY_NAME[TREATMENT_TASK_NAME],
        TASK_INDEX_BY_NAME[DISCHARGE_TASK_NAME],
    ],
    "Doctor": [
        TASK_INDEX_BY_NAME[MEDICAL_EVALUATION_TASK_NAME],
        TASK_INDEX_BY_NAME[DIAGNOSTICS_TASK_NAME],
        TASK_INDEX_BY_NAME[TREATMENT_TASK_NAME],
        TASK_INDEX_BY_NAME[REASSESSMENT_TASK_NAME],
        TASK_INDEX_BY_NAME[DISPOSITION_DECISION_TASK_NAME],
    ],
}

# Optional baseline-extension candidate. Disabled by default; when enabled, one
# senior doctor is added as a stationary COCPIT reviewer who does not take
# ordinary patient-care workload.
ENABLE_SENIOR_DOCTOR_OVERSIGHT = False
SENIOR_DOCTOR_HOME_ZONE_ID = "COCPIT"
SENIOR_DOCTOR_GID = 29
SENIOR_DOCTOR_REVIEW_PROBABILITY = {
    MEDICAL_EVALUATION_TASK_NAME: 0.08,
    DIAGNOSTICS_TASK_NAME: 0.10,
    REASSESSMENT_TASK_NAME: 0.07,
    DISPOSITION_DECISION_TASK_NAME: 0.10,
}
SENIOR_DOCTOR_REVIEW_COOLDOWN_SECONDS = 300
SENIOR_DOCTOR_REVIEW_EXPIRY_SECONDS = 900
SENIOR_DOCTOR_REVIEW_DISTANCE_METERS = 4.5
SENIOR_DOCTOR_REVIEW_HOLD_SECONDS = (30, 90)
SENIOR_DOCTOR_MAX_REVIEWS_PER_PATIENT = 1
SENIOR_DOCTOR_MAX_COMPLEX_REVIEWS_PER_PATIENT = 2
SENIOR_DOCTOR_OFFSPA_REVIEW_PROBABILITY = 0.06
SENIOR_DOCTOR_ROOM_ASSIST_PROBABILITY = {
    MEDICAL_EVALUATION_TASK_NAME: 0.05,
    DIAGNOSTICS_TASK_NAME: 0.05,
    TREATMENT_TASK_NAME: 0.03,
    REASSESSMENT_TASK_NAME: 0.04,
    DISPOSITION_DECISION_TASK_NAME: 0.04,
}
SENIOR_DOCTOR_ROOM_ASSIST_COOLDOWN_SECONDS = 1200
SENIOR_DOCTOR_ROOM_ASSIST_THRESHOLD_LOCATION_PROBABILITY = 0.90

# Realistic clinical durations. These allow for a stable queue of ~5-10
# patients when paired with the 0.0015 arrival rate.
TASK_DURATIONS_SECONDS = {
    PLACEMENT_TASK_NAME: (60, 180),
    INITIAL_NURSING_ASSESSMENT_TASK_NAME: (300, 600),
    MEDICAL_EVALUATION_TASK_NAME: (600, 900),
    DIAGNOSTICS_TASK_NAME: (300, 600),
    TREATMENT_TASK_NAME: (600, 900),
    REASSESSMENT_TASK_NAME: (300, 600),
    DISPOSITION_DECISION_TASK_NAME: (300, 600),
    DISCHARGE_TASK_NAME: (300, 900),
}

COORDINATION_NURSE_HOME_ZONE_ID = "COCPIT"
DOCTOR_HOME_ZONE_ID = "OFFSPA"
OFFSPA_DOCTOR_HOME_POINT = (0.23, 4.35)
NURSE_HOME_ZONE_IDS = ["NURSTA", "NURSTA", "NUROPE", "NUROPE", "COCPIT"]
STATION_ZONE_IDS = ["COCPIT", "NUROPE", "NURSTA"]
STATION_ADJACENT_ZONE_IDS = ["CORR03", "CORR04", "CORR05", "CORR07"]
DEFAULT_STATION_ATTRACTOR_OVERRIDES = {
    "NUROPE": (-6.9, -4.15),
}
COCPIT_WORKSTATION_REGION_1 = {
    "x_min": -2.0,
    "x_max": 2.2,
    "y": 0.0,
    "heading_degrees": 90.0,
}
COCPIT_WORKSTATION_REGION_2 = {
    "x_min": -1.75,
    "x_max": 2.0,
    "y": -1.65,
    "heading_degrees": -90.0,
}
COCPIT_WORKSTATION_BAND_Y_JITTER_METERS = 0.08
DEFAULT_SCOPE_MODE = "simulation"
SCOPE_MODES = [DEFAULT_SCOPE_MODE]

BED_PROXIMITY_METERS = 2.3
PATIENT_ENTRY_ZONE_ID = "CORR07"
PATIENT_ENTRY_POINT = (5.4, 9.0)
INITIAL_PATIENT_COUNT = 1

# The model represents a bounded observed care-area subsystem, not the whole
# hospital waiting room. Once occupied beds plus waiting patients reaches this
# cap, new upstream arrivals are counted as deferred outside the modeled area.
MAX_ACTIVE_SYSTEM_LOAD = 13

# Clinical tasks do not automatically imply a validation-relevant F2F
# communication episode. These probabilities govern whether a task start
# generates a patient-facing interaction log entry; task execution itself is
# always still logged as workflow.
PATIENT_FACING_TASK_INTERACTION_PROBABILITY = {
    "CoordinationNurse": {
        "Placement": 0.35,
    },
    "Nurse": {
        "Initial Nursing Assessment": 0.42,
        "Diagnostics": 0.34,
        "Treatment": 0.40,
        "Discharge or Admission": 0.46,
    },
    "Doctor": {
        "Medical Evaluation": 0.14,
        "Diagnostics": 0.32,
        "Treatment": 0.34,
        "Reassessment": 0.40,
        "Disposition Decision": 0.16,
    },
}
HIGH_ACUITY_PATIENT_FACING_BONUS = 0.18
DOCTOR_PATIENT_THRESHOLD_LOG_PROBABILITY = 0.55

# Validation-counted patient-facing F2F episodes are stricter than clinical
# task attendance. Tasks still execute normally; this set only gates whether a
# task start can be logged as a shadowing-comparable patient-facing interaction.
PATIENT_FACING_VALIDATION_TASKS = {
    "CoordinationNurse": {"Placement"},
    "Nurse": {"Initial Nursing Assessment", "Discharge or Admission"},
    "Doctor": {"Medical Evaluation", "Disposition Decision"},
}
PATIENT_FACING_VALIDATION_MIN_INTERVAL_SECONDS = 900

# Local satellite-station check-ins let the coordination nurse use
# NURSTA/NUROPE as coordination stations without adding support/admin roles.
SATELLITE_STATION_CHECKIN_PROBABILITY = 0.05
SATELLITE_STATION_CHECKIN_DISTANCE_METERS = 4.2
PLACEMENT_NURSE_ALERT_PROBABILITY = 0.20
PLACEMENT_ESCORT_CORRIDOR_INTERACTION_PROBABILITY = 0.25
PATIENT_FACING_IN_TASK_INTERACTION_PROBABILITY = {
    "CoordinationNurse": {
        PLACEMENT_TASK_NAME: 0.04,
    },
    "Nurse": {
        INITIAL_NURSING_ASSESSMENT_TASK_NAME: 0.04,
        DISCHARGE_TASK_NAME: 0.03,
    },
    "Doctor": {
        MEDICAL_EVALUATION_TASK_NAME: 0.03,
        REASSESSMENT_TASK_NAME: 0.03,
        DISPOSITION_DECISION_TASK_NAME: 0.03,
    },
}
PATIENT_FACING_IN_TASK_MIN_SECONDS = 60
BEDSIDE_COTASK_INTERACTION_PROBABILITY = 0.22
BEDSIDE_COTASK_INTERACTION_COOLDOWN_SECONDS = 300

# Workflow task completion can generate a short, validation-counted staff-staff
# update only when the staff member later becomes co-present with a relevant
# colleague. This represents real ED transition talk without changing clinical
# task execution or injecting unsupported encounters at beds.
TASK_TRANSITION_UPDATE_PROBABILITY = {
    "Nurse": {
        "Initial Nursing Assessment": 0.42,
        "Diagnostics": 0.26,
        "Treatment": 0.22,
        "Discharge or Admission": 0.32,
    },
    "Doctor": {
        "Medical Evaluation": 0.55,
        "Diagnostics": 0.48,
        "Treatment": 0.30,
        "Reassessment": 0.42,
        "Disposition Decision": 0.50,
    },
}
TASK_TRANSITION_UPDATE_COOLDOWN_SECONDS = 180
TASK_TRANSITION_UPDATE_EXPIRY_SECONDS = 900
TASK_TRANSITION_UPDATE_DISTANCE_METERS = 4.2
TASK_TRANSITION_UPDATE_INTERACTION_PROBABILITY = 0.95
TASK_TRANSITION_STATION_EDGE_RELOCATION_PROBABILITY = 0.0

# Some clinical steps require short bounded coordination before the next task
# can start. This models real status/result/plan clarification without adding
# random wandering or changing the workflow checklist itself.
INTERTASK_COORDINATION_GAP_PROBABILITY = {
    PLACEMENT_TASK_NAME: 0.18,
    INITIAL_NURSING_ASSESSMENT_TASK_NAME: 0.36,
    MEDICAL_EVALUATION_TASK_NAME: 0.34,
    DIAGNOSTICS_TASK_NAME: 0.36,
    DISPOSITION_DECISION_TASK_NAME: 0.26,
}
INTERTASK_COORDINATION_GAP_SECONDS = {
    PLACEMENT_TASK_NAME: (30, 90),
    INITIAL_NURSING_ASSESSMENT_TASK_NAME: (45, 150),
    MEDICAL_EVALUATION_TASK_NAME: (45, 180),
    DIAGNOSTICS_TASK_NAME: (60, 210),
    DISPOSITION_DECISION_TASK_NAME: (45, 150),
}

# Role-pair and zone weights shape eligible HCW-HCW communication without
# hard-coding empirical counts. They make the coordination ecology plausible
# while leaving stochastic routing and co-presence in control.
HCW_HCW_ROLE_PAIR_INTERACTION_WEIGHTS = {
    "CoordinationNurse|Nurse": 1.60,
    "Nurse|Nurse": 1.25,
    "Doctor|Nurse": 0.70,
    "CoordinationNurse|Doctor": 1.25,
    "Doctor|Doctor": 1.90,
}
HCW_HCW_ZONE_INTERACTION_WEIGHTS = {
    "COCPIT": 0.75,
    "NUROPE": 1.18,
    "NURSTA": 1.30,
    "CORR03": 1.08,
    "CORR01": 1.18,
    "CORR02": 1.08,
    "CORR04": 1.18,
    "CORR05": 1.15,
    "CORR07": 1.12,
    "corridor": 1.12,
    "station": 1.10,
}

BASELINE_MODEL_ID = "rule_based_perception_model"
MODEL_VARIANT = BASELINE_MODEL_ID
MODEL_VARIANTS = [
    BASELINE_MODEL_ID,
]
POSTHOC_ANALYSIS_MODES = ["generative_interview"]
DEFAULT_BEHAVIORAL_ABLATION_VARIANTS = [BASELINE_MODEL_ID]
RUN_ABLATION_SMOKE = False
RUN_VERIFICATION = False
RUN_INTERVIEW_SMOKE = False
LAPTOP_SMOKE_MODE = True
SMOKE_DURATION_SECONDS = 600
SMOKE_N_RUNS = 1
SAVE_MEMORY_TRACES = True
SAVE_RETRIEVAL_TRACES = True
SAVE_INTERVIEW_OUTPUTS = True
ENABLE_MISSED_OPPORTUNITY_LOG = True
ENABLE_INTERRUPTION_METRICS = True
PERSONA_SOURCE = "named_default"
MISSED_OPPORTUNITY_COOLDOWN_SECONDS = 90
MISSED_OPPORTUNITY_MIN_DURATION_SECONDS = 8
MISSED_OPPORTUNITY_LOG_MODE = "episode"
ABLATION_DIAGNOSTIC_MODE = True
DIAGNOSTIC_ARRIVAL_MULTIPLIER = 3.0
DIAGNOSTIC_INTERACTION_PROBABILITY_MULTIPLIER = 1.35
DIAGNOSTIC_INITIAL_PATIENT_COUNT = 3
GEN_FALLBACK_MIN_INTERACTIONS_10M = 4
MIN_MISSED_OPPORTUNITIES_1H_DIAGNOSTIC = 1

# Provisional validation-readiness gates. These are transparent engineering
# thresholds, not proof of truth. They should be replaced or tightened after
# empirical bootstrap tolerance work and supervisor review.
VALIDATION_THRESHOLDS = {
    "interaction_count_relative_error_warn": 0.30,
    "interaction_count_relative_error_fail": 0.50,
    "f2f_rate_relative_error_warn": 0.30,
    "f2f_rate_relative_error_fail": 0.50,
    "kde_distance_warn": 0.65,
    "kde_distance_fail": 0.80,
    "zone_jsd_warn": 0.35,
    "zone_jsd_fail": 0.50,
    "role_pair_jsd_warn": 0.40,
    "role_pair_jsd_fail": 0.60,
    "topic_jsd_warn": 0.35,
    "topic_jsd_fail": 0.55,
    "topic_by_zone_jsd_warn": 0.55,
    "topic_by_zone_jsd_fail": 0.75,
    "duration_error_warn": 0.25,
    "duration_error_fail": 0.50,
    "hcw_patient_share_abs_error_warn": 0.20,
    "hcw_patient_share_abs_error_fail": 0.35,
    "patient_facing_share_abs_error_warn": 0.20,
    "patient_facing_share_abs_error_fail": 0.35,
    "hcw_hcw_share_abs_error_warn": 0.15,
    "hcw_hcw_share_abs_error_fail": 0.25,
    "station_share_abs_error_warn": 0.15,
    "station_share_abs_error_fail": 0.25,
    "corridor_share_abs_error_warn": 0.15,
    "corridor_share_abs_error_fail": 0.25,
    "outside_or_uncoded_share_warn": 0.35,
    "outside_or_uncoded_share_fail": 0.50,
    # No empirical ESI/acuity target exists in the current shadowing summary.
    # The validator should report high-acuity fit as UNKNOWN until one is added.
    "high_acuity_share_abs_error_warn": None,
    "high_acuity_share_abs_error_fail": None,
}

ARRIVAL_MODE = "simple"
DEFAULT_SCENARIO_MODE = "validated_baseline"
DEFAULT_SCENARIO_START_HOUR = 0
SCENARIO_MODES = {
    "validated_baseline": {
        "description": "Current calibrated care-area baseline used before scenario load separation.",
        "arrival_rate_multiplier": 1.0,
        "hourly_arrival_multipliers": [1.0] * 24,
        "esi_arrival_weights": None,
        "high_acuity_event_probability_multiplier": 1.0,
        "pressure_action_threshold": 0.65,
        "pressure_station_social_suppression": 0.30,
    },
    "normal_load": {
        "description": (
            "Time-varying normal ED care-area demand: low early hours, morning ramp, "
            "midday/afternoon busy period, and evening decline."
        ),
        "arrival_rate_multiplier": 0.76,
        "hourly_arrival_multipliers": [
            0.55, 0.45, 0.38, 0.32, 0.28, 0.30,
            0.34, 0.58, 1.18, 1.62, 1.55, 1.45,
            1.35, 1.38, 1.42, 1.38, 1.28, 1.15,
            1.05, 0.98, 0.88, 0.75, 0.64, 0.55,
        ],
        "esi_arrival_weights": None,
        "high_acuity_event_probability_multiplier": 0.8,
        "pressure_action_threshold": 0.62,
        "pressure_station_social_suppression": 0.35,
    },
    "high_load_high_acuity": {
        "description": (
            "High operational pressure scenario with stronger input pressure and "
            "a higher ESI 1/2/3 share."
        ),
        "arrival_rate_multiplier": 1.15,
        "hourly_arrival_multipliers": [
            0.70, 0.58, 0.48, 0.40, 0.36, 0.40,
            0.50, 0.78, 1.35, 1.85, 1.75, 1.65,
            1.58, 1.62, 1.70, 1.68, 1.55, 1.40,
            1.30, 1.20, 1.05, 0.92, 0.82, 0.72,
        ],
        "esi_arrival_weights": {
            1: 0.008,
            2: 0.14,
            3: 0.50,
            4: 0.23,
            5: 0.122,
        },
        "high_acuity_event_probability_multiplier": 0.8,
        "pressure_action_threshold": 0.50,
        "pressure_station_social_suppression": 0.55,
    },
}
ENABLE_ESI = True
ENABLE_HIGH_ACUITY_EVENTS = True
HIGH_ACUITY_EVENT_PROBABILITY_PER_SECOND = 0.00003
HIGH_ACUITY_SOFT_PREEMPTION_ENABLED = True
HIGH_ACUITY_PREEMPTION_COOLDOWN_SECONDS = 300
HIGH_ACUITY_PREEMPTION_MAX_DISTANCE_METERS = 18.0
ESI_ARRIVAL_WEIGHTS = {
    1: 0.01,
    2: 0.12,
    3: 0.42,
    4: 0.30,
    5: 0.15,
}
ESI_TASK_DURATION_MULTIPLIERS = {
    1: 1.40,
    2: 1.25,
    3: 1.00,
    4: 0.85,
    5: 0.75,
}
ESI_ENTRY_LEVELS = {1, 2, 3, 4, 5}
ESI_PRIORITY_WEIGHT = {
    1: 5.0,
    2: 3.0,
    3: 1.6,
    4: 0.8,
    5: 0.4,
}

POST_TASK_STATION_CHECK_ENABLED = True
POST_TASK_STATION_CHECK_PROBABILITY_DOCTOR = 0.55
POST_TASK_STATION_CHECK_PROBABILITY_NURSE = 0.42
POST_TASK_STATION_CHECK_DURATION_SECONDS_MIN = 20
POST_TASK_STATION_CHECK_DURATION_SECONDS_MAX = 90
POST_TASK_STATION_CHECK_STATIONS = ("COCPIT", "NURSTA", "NUROPE")
PATIENT_ARCHETYPES_BY_ESI = {
    1: [
        {
            "complaint": "polytrauma_resuscitation",
            "urgency": "resuscitation",
            "expected_resources": ["doctor", "nurse", "monitoring", "imaging", "iv_access"],
            "likely_pathway": "resuscitation_to_admission",
            "required_staff_roles": ["Doctor", "Nurse", "CoordinationNurse"],
            "high_acuity_escalation": True,
        }
    ],
    2: [
        {
            "complaint": "stroke_like_symptoms",
            "urgency": "emergent",
            "expected_resources": ["doctor", "nurse", "ct", "iv_access"],
            "likely_pathway": "rapid_diagnostics",
            "required_staff_roles": ["Doctor", "Nurse"],
            "high_acuity_escalation": True,
        },
        {
            "complaint": "sepsis_like_deterioration",
            "urgency": "emergent",
            "expected_resources": ["doctor", "nurse", "labs", "iv_antibiotics"],
            "likely_pathway": "rapid_treatment",
            "required_staff_roles": ["Doctor", "Nurse"],
            "high_acuity_escalation": True,
        },
        {
            "complaint": "allergic_reaction",
            "urgency": "emergent",
            "expected_resources": ["doctor", "nurse", "medication", "monitoring"],
            "likely_pathway": "treatment_and_observation",
            "required_staff_roles": ["Doctor", "Nurse"],
            "high_acuity_escalation": True,
        },
    ],
    3: [
        {
            "complaint": "appendicitis_like_abdominal_pain",
            "urgency": "urgent",
            "expected_resources": ["doctor", "nurse", "labs", "imaging"],
            "likely_pathway": "diagnostics_then_disposition",
            "required_staff_roles": ["Doctor", "Nurse"],
            "high_acuity_escalation": False,
        },
        {
            "complaint": "fracture_like_injury",
            "urgency": "urgent",
            "expected_resources": ["doctor", "nurse", "xray", "analgesia"],
            "likely_pathway": "diagnostics_and_treatment",
            "required_staff_roles": ["Doctor", "Nurse"],
            "high_acuity_escalation": False,
        },
        {
            "complaint": "dvt_like_leg_pain",
            "urgency": "urgent",
            "expected_resources": ["doctor", "nurse", "ultrasound", "labs"],
            "likely_pathway": "diagnostics_then_disposition",
            "required_staff_roles": ["Doctor", "Nurse"],
            "high_acuity_escalation": False,
        },
    ],
    4: [
        {
            "complaint": "minor_wound_or_pain",
            "urgency": "standard",
            "expected_resources": ["nurse", "doctor"],
            "likely_pathway": "brief_assessment_discharge",
            "required_staff_roles": ["Nurse", "Doctor"],
            "high_acuity_escalation": False,
        }
    ],
    5: [
        {
            "complaint": "low_acuity_check",
            "urgency": "nonurgent",
            "expected_resources": ["nurse"],
            "likely_pathway": "brief_care",
            "required_staff_roles": ["Nurse"],
            "high_acuity_escalation": False,
        }
    ],
}

BED_POSITIONS = [
    (8.05, -1.8),
    (8.05, 1.2),
    (8.05, 4.2),
    (-8.05, -1.8),
    (-8.05, 4.2),
    (5.3, -10.0),
    (5.3, -7.2),
    (-12.5, -2.0),
    (-8.0, -12.0),
    (-8.0, -7.0),
]

# Staff-facing access points for patient beds. Patient bed centers remain the
# clinical patient coordinates; staff route to these nearby access points when
# a bed center is valid but a direct center-line route would cross room walls.
BED_ACCESS_POSITIONS = [
    (7.2, -1.8),
    (7.2, 1.2),
    (7.2, 4.2),
    (-7.2, -1.8),
    (-7.2, 4.2),
    (5.3, -10.0),
    (5.3, -7.2),
    (-12.5, -2.0),
    (-8.0, -12.0),
    (-8.0, -7.0),
]

# Bed 8 is reserved for rare high-acuity ESI 1/2 cases. ESI 3-5 patients are
# never assigned there.
HIGH_ACUITY_RESERVED_BED_INDICES = {7}

# Five nurses cover the ten-bed care area. Nurse homes are configured as:
# two NURSTA, two NUROPE, and one COCPIT.
NURSE_BED_COVERAGE = {
    0: [5, 6],
    1: [0, 1, 2],
    2: [3, 4, 7],
    3: [8, 9],
    4: [0, 1, 2, 5, 6],
}

# This small, explicit waypoint graph is a Phase 1 routing scaffold. It keeps
# movement spatially grounded against walls without introducing a full navmesh.
ROUTING_WAYPOINTS = {
    "right_corridor_hub": (5.5, -1.6),
    "lower_corridor_hub": (0.18, -4.7),
    "cockpit_east_access": (3.3, -1.3),
    "cockpit_west_access": (-3.5, -1.2),
    "corr01_east": (3.2, 2.35),
    "corr01_mid": (0.176, 2.35),
    "corr01_west": (-3.2, 2.35),
    "offspa_access": (1.5, 2.35),
    "corr02_east": (3.2, 6.1),
    "corr02_mid": (0.176, 6.1),
    "corr02_west": (-3.2, 6.1),
    "corr04_door": (-2.8, -5.7),
    "corr05_door": (3.2, -5.7),
    "corr07_lower": (5.2, -2.0),
    "entry_door": (5.4, 9),
    "nurope_door": (-6.2, -4.64),
    "nursta_west": (-2.8, -10.5),
    "nursta_east": (2.8, -10.5),
    "right_room_door": (6.5, 1.2),
    "left_room_door": (-6.1, 1.2),
    "right_lower_room_door": (6.5, -1.8),
    "right_upper_room_door": (6.5, 4.2),
    "left_lower_room_door": (-6.1, -1.8),
    "left_upper_room_door": (-6.0, 2.7),
    "left_upper_bed_access": (-5.4, 2.2),
    "high_acuity_bed_access": (-10.4, -3.3),
    "lower_left_bed_access": (-6.4, -9.2),
    "lower_left_deep_bed_access": (-6.2, -11.5),
}

ROUTING_EDGES = [
    ("cockpit_east_access", "right_corridor_hub"),
    ("cockpit_west_access", "nurope_door"),
    ("right_corridor_hub", "lower_corridor_hub"),
    ("right_corridor_hub", "corr07_lower"),
    ("right_corridor_hub", "right_room_door"),
    ("right_corridor_hub", "right_upper_room_door"),
    ("right_room_door", "corr01_east"),
    ("right_upper_room_door", "corr01_east"),
    ("right_room_door", "corr02_east"),
    ("right_upper_room_door", "corr02_east"),
    ("corr01_east", "corr01_mid"),
    ("corr01_mid", "corr01_west"),
    ("corr01_mid", "offspa_access"),
    ("corr01_east", "offspa_access"),
    ("corr01_west", "left_upper_bed_access"),
    ("corr02_east", "corr02_mid"),
    ("corr02_mid", "corr02_west"),
    ("corr02_west", "left_upper_bed_access"),
    ("corr07_lower", "entry_door"),
    ("right_room_door", "corr07_lower"),
    ("corr07_lower", "right_lower_room_door"),
    ("lower_corridor_hub", "corr04_door"),
    ("lower_corridor_hub", "corr05_door"),
    ("lower_corridor_hub", "nurope_door"),
    ("lower_corridor_hub", "left_lower_room_door"),
    ("left_room_door", "left_upper_room_door"),
    ("nurope_door", "left_upper_bed_access"),
    ("nurope_door", "high_acuity_bed_access"),
    ("lower_left_bed_access", "lower_left_deep_bed_access"),
    ("corr04_door", "nursta_west"),
    ("corr05_door", "nursta_east"),
]

TEST_MODE = False
TEST_DURATION_SECONDS = 3_600

ANIMATION_MODE = True
ANIMATION_STEPS_PER_FRAME = 20
ANIMATION_INTERVAL_MS = 50
ANIMATION_DURATION_SECONDS = 43_200
ANIMATION_OUTPUT_PATH = OUTPUTS_DIR / "simulation_animation.mp4"
SAVE_ANIMATION = False

FIGURE_SIZE = (14, 10)
PLOT_PADDING_METERS = 1.0
HEATMAP_BINS = 60
HEATMAP_ALPHA = 0.35
HEATMAP_CMAP = "Reds"
MATPLOTLIB_BACKEND = "Agg"
TRAIL_ALPHA = 0.15
TRAIL_LINEWIDTH = 0.8
ZONE_LINEWIDTH = 1.0
WALL_LINEWIDTH = 1.5
ZONE_LABEL_FONTSIZE = 8
AGENT_MARKER_SIZE = 50
COORDINATION_NURSE_MARKER_SIZE = 70
STATUS_POSITION_DECIMALS = 2

WALL_COLOR = "black"
ZONE_COLOR = "lightgray"
ZONE_LABEL_COLOR = "dimgray"

ROLE_COLORS = {
    "Patient": "blue",
    "CoordinationNurse": "orange",
    "Nurse": "green",
    "Doctor": "red",
}

ROLE_MARKERS = {
    "Patient": "o",
    "CoordinationNurse": "^",
    "Nurse": "^",
    "Doctor": "^",
}

# Phase 2+ condition management extends the baseline geometry without
# modifying the core environment loader. Baseline remains the default.
CONDITION_NAME = "baseline"
CONDITIONS = {
    "baseline": {
        "notes": "Observed ED layout without intervention.",
        "visibility_transparent_segments": [],
        "removed_wall_segments": [],
        "added_wall_segments": [],
        "zone_polygon_overrides": {},
        "station_attractor_overrides": {},
        "routing_waypoint_overrides": {},
    },
    "cockpit_only": {
        "notes": "Cockpit wall segments become visually transparent but remain collision barriers.",
        "visibility_transparent_segments": [
            {"start": (0.176409652494, 1.42605248428), "end": (0.176409652494, 0.0975794953605)},
            {"start": (-3.597481837, -1.41096340304), "end": (-3.597481837, -2.46646061156)},
            {"start": (-3.597481837, -0.386066629534), "end": (-2.6897465448, -0.386066629534)},
            {"start": (-3.597481837, -0.386066629534), "end": (-3.597481837, 1.42605248428)},
            {"start": (-3.597481837, 1.42605248428), "end": (3.95030114198, 1.42605248428)},
            {"start": (3.95030114198, 1.42605248428), "end": (3.95283453306, -0.378749840708)},
            {"start": (3.95283453306, -1.41096340304), "end": (3.95283453306, -2.46646061156)},
            {"start": (-2.6897465448, -0.386066629534), "end": (-2.6897465448, 0.187944707007)},
            {"start": (-2.6897465448, 0.187944707007), "end": (-2.25255835192, 0.622010127081)},
            {"start": (-2.25255835192, 0.622010127081), "end": (0.176409652494, 0.622010127081)},
            {"start": (0.176409652494, 0.622010127081), "end": (2.60322936881, 0.622010127081)},
            {"start": (2.60322936881, 0.622010127081), "end": (3.03812607609, 0.187944707007)},
            {"start": (3.03812607609, 0.187944707007), "end": (3.04264332228, -0.38591458963)},
            {"start": (3.04264332228, -0.38591458963), "end": (3.95283453306, -0.378749840708)},
            {"start": (-3.597481837, -1.41096340304), "end": (-2.6897465448, -1.41096340304)},
            {"start": (-2.6897465448, -1.41096340304), "end": (-2.6897465448, -1.7796573699)},
            {"start": (3.95283453306, -1.41096340304), "end": (3.04264332228, -1.41096340304)},
            {"start": (3.04264332228, -1.41096340304), "end": (3.04264332228, -1.7796573699)},
            {"start": (3.95283453306, -2.46646061156), "end": (2.13596697993, -3.25420584376)},
            {"start": (2.13596697993, -3.25420584376), "end": (0.17767634803, -3.52343857955)},
            {"start": (0.17767634803, -3.52343857955), "end": (-1.78314767183, -3.25420584463)},
            {"start": (-1.78314767183, -3.25420584463), "end": (-3.597481837, -2.46646061156)},
            {"start": (3.04264332228, -1.7796573699), "end": (1.81733040136, -2.24881171747)},
            {"start": (1.81733040136, -2.24881171747), "end": (0.176448388743, -2.4691874107)},
            {"start": (0.176448388743, -2.4691874107), "end": (-1.47949411167, -2.24968504673)},
            {"start": (-1.47949411167, -2.24968504673), "end": (-2.6897465448, -1.7796573699)},
        ],
        "removed_wall_segments": [],
        "added_wall_segments": [],
        "zone_polygon_overrides": {},
        "station_attractor_overrides": {},
        "routing_waypoint_overrides": {},
    },
    "nursta_only": {
        "notes": "Satellite nurse station relocates upward and the obsolete vertical divider wall is removed.",
        "visibility_transparent_segments": [],
        "removed_wall_segments": [
            {"start": (0.176409652494, -5.92891719145), "end": (0.176409652494, -9.01517939175)},
        ],
        "added_wall_segments": [
            {"start": (-2.1855208564972446, -5.92891719145), "end": (2.422498356748878, -5.92891719145)},
            {"start": (2.422498356748878, -5.92891719145), "end": (2.4224983567488785, -7.29475)},
            {"start": (2.4224983567488785, -7.29475), "end": (-2.1855208564972446, -7.29475)},
            {"start": (-2.1855208564972446, -7.29475), "end": (-2.1855208564972446, -5.92891719145)},
        ],
        "zone_polygon_overrides": {
            "NURSTA": [
                (2.20, -7.45),
                (2.20, -8.85),
                (-2.00, -8.85),
                (-2.00, -7.45),
                (2.20, -7.45),
            ],
        },
        "station_attractor_overrides": {
            "NURSTA": (0.10, -8.10),
        },
        "routing_waypoint_overrides": {
            "nursta_west": (-2.05, -8.10),
            "nursta_east": (2.05, -8.10),
        },
    },
    "both": {
        "notes": "Transparent cockpit plus relocated NURSTA.",
        "visibility_transparent_segments": [
            {"start": (0.176409652494, 1.42605248428), "end": (0.176409652494, 0.0975794953605)},
            {"start": (-3.597481837, -1.41096340304), "end": (-3.597481837, -2.46646061156)},
            {"start": (-3.597481837, -0.386066629534), "end": (-2.6897465448, -0.386066629534)},
            {"start": (-3.597481837, -0.386066629534), "end": (-3.597481837, 1.42605248428)},
            {"start": (-3.597481837, 1.42605248428), "end": (3.95030114198, 1.42605248428)},
            {"start": (3.95030114198, 1.42605248428), "end": (3.95283453306, -0.378749840708)},
            {"start": (3.95283453306, -1.41096340304), "end": (3.95283453306, -2.46646061156)},
            {"start": (-2.6897465448, -0.386066629534), "end": (-2.6897465448, 0.187944707007)},
            {"start": (-2.6897465448, 0.187944707007), "end": (-2.25255835192, 0.622010127081)},
            {"start": (-2.25255835192, 0.622010127081), "end": (0.176409652494, 0.622010127081)},
            {"start": (0.176409652494, 0.622010127081), "end": (2.60322936881, 0.622010127081)},
            {"start": (2.60322936881, 0.622010127081), "end": (3.03812607609, 0.187944707007)},
            {"start": (3.03812607609, 0.187944707007), "end": (3.04264332228, -0.38591458963)},
            {"start": (3.04264332228, -0.38591458963), "end": (3.95283453306, -0.378749840708)},
            {"start": (-3.597481837, -1.41096340304), "end": (-2.6897465448, -1.41096340304)},
            {"start": (-2.6897465448, -1.41096340304), "end": (-2.6897465448, -1.7796573699)},
            {"start": (3.95283453306, -1.41096340304), "end": (3.04264332228, -1.41096340304)},
            {"start": (3.04264332228, -1.41096340304), "end": (3.04264332228, -1.7796573699)},
            {"start": (3.95283453306, -2.46646061156), "end": (2.13596697993, -3.25420584376)},
            {"start": (2.13596697993, -3.25420584376), "end": (0.17767634803, -3.52343857955)},
            {"start": (0.17767634803, -3.52343857955), "end": (-1.78314767183, -3.25420584463)},
            {"start": (-1.78314767183, -3.25420584463), "end": (-3.597481837, -2.46646061156)},
            {"start": (3.04264332228, -1.7796573699), "end": (1.81733040136, -2.24881171747)},
            {"start": (1.81733040136, -2.24881171747), "end": (0.176448388743, -2.4691874107)},
            {"start": (0.176448388743, -2.4691874107), "end": (-1.47949411167, -2.24968504673)},
            {"start": (-1.47949411167, -2.24968504673), "end": (-2.6897465448, -1.7796573699)},
        ],
        "removed_wall_segments": [
            {"start": (0.176409652494, -5.92891719145), "end": (0.176409652494, -9.01517939175)},
        ],
        "added_wall_segments": [
            {"start": (-2.1855208564972446, -5.92891719145), "end": (2.422498356748878, -5.92891719145)},
            {"start": (2.422498356748878, -5.92891719145), "end": (2.4224983567488785, -7.29475)},
            {"start": (2.4224983567488785, -7.29475), "end": (-2.1855208564972446, -7.29475)},
            {"start": (-2.1855208564972446, -7.29475), "end": (-2.1855208564972446, -5.92891719145)},
        ],
        "zone_polygon_overrides": {
            "NURSTA": [
                (2.20, -7.45),
                (2.20, -8.85),
                (-2.00, -8.85),
                (-2.00, -7.45),
                (2.20, -7.45),
            ],
        },
        "station_attractor_overrides": {
            "NURSTA": (0.10, -8.10),
        },
        "routing_waypoint_overrides": {
            "nursta_west": (-2.05, -8.10),
            "nursta_east": (2.05, -8.10),
        },
    },
}

PART2_SCENARIO_REGISTRY = {
    "baseline": {
        "human_name": "Baseline care-area layout",
        "description": "Validated Part 1 care-area baseline with no design intervention.",
        "mechanism_changed": "none",
        "parameters_changed": [],
        "expected_direction_of_effect": "Reference condition for paired deltas.",
        "geometry_changes": False,
        "visibility_changes": False,
        "station_attractor_changes": False,
        "wayfinding_or_movement_changes": False,
        "interaction_opportunity_rule_changes": False,
    },
    "cockpit_only": {
        "human_name": "COCPIT transparency only",
        "description": "COCPIT wall segments are visually transparent while remaining physical barriers.",
        "mechanism_changed": "visibility / line-of-sight / spatial awareness around COCPIT",
        "parameters_changed": ["visibility_transparent_segments"],
        "expected_direction_of_effect": (
            "May shift awareness, missed opportunities, and HCW-HCW interactions around "
            "COCPIT and adjacent corridors; direction is model-predicted, not hard-coded."
        ),
        "geometry_changes": False,
        "visibility_changes": True,
        "station_attractor_changes": False,
        "wayfinding_or_movement_changes": False,
        "interaction_opportunity_rule_changes": True,
    },
    "nursta_only": {
        "human_name": "NURSTA station-access intervention only",
        "description": "NURSTA station polygon, access, and station attractor are shifted within the care area.",
        "mechanism_changed": "station attractor / station return / local route accessibility",
        "parameters_changed": [
            "zone_polygon_overrides.NURSTA",
            "station_attractor_overrides.NURSTA",
            "routing_waypoint_overrides.nursta_west",
            "routing_waypoint_overrides.nursta_east",
            "removed_wall_segments",
        ],
        "expected_direction_of_effect": (
            "May shift nurse dwell, route exposure, Nurse|Nurse and CoordinationNurse|Nurse "
            "opportunities, and station-versus-corridor balance."
        ),
        "geometry_changes": True,
        "visibility_changes": False,
        "station_attractor_changes": True,
        "wayfinding_or_movement_changes": True,
        "interaction_opportunity_rule_changes": True,
    },
    "both": {
        "human_name": "COCPIT transparency + NURSTA intervention",
        "description": "Combined 2x2 condition with COCPIT transparency and NURSTA station-access changes.",
        "mechanism_changed": "combined visibility and station-attractor / movement effects",
        "parameters_changed": [
            "visibility_transparent_segments",
            "zone_polygon_overrides.NURSTA",
            "station_attractor_overrides.NURSTA",
            "routing_waypoint_overrides.nursta_west",
            "routing_waypoint_overrides.nursta_east",
            "removed_wall_segments",
        ],
        "expected_direction_of_effect": "Tests whether the two changes are additive, offsetting, or interacting.",
        "geometry_changes": True,
        "visibility_changes": True,
        "station_attractor_changes": True,
        "wayfinding_or_movement_changes": True,
        "interaction_opportunity_rule_changes": True,
    },
}

# The perception layer replaces radius-only visibility with line-of-sight,
# field of view, salience, and attention. Movement remains rule-based.
PERCEPTION_BASED_BASELINE_ENABLED = True
LEGACY_PROXIMITY_MODE_ENABLED = False
ISOVIST_BACKEND = "raycast"
VISIBILITY_MAX_DISTANCE = 18.0
FOV_DEGREES = 210.0
ATTENTION_CAPACITY = 2
SALIENCE_DISTANCE_DECAY = 6.0
SALIENCE_MIN_THRESHOLD = 0.1
ISOVIST_RAY_EPSILON_RADIANS = 1e-4
ISOVIST_SAMPLE_ANGLE_DEGREES = 6.0
VISIBILITY_GRAPH_RESOLUTION_METERS = 1.5
PERCEPTION_INTERACTION_RADIUS_METERS = 10.0
PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS = INTERACTION_DISTANCE_METERS
PERCEPTION_OPPORTUNITY_CHECK_INTERVAL_SECONDS = 10
PERCEPTION_STAFF_ATTENTION_CAPACITY = 3
PERCEPTION_STAFF_SALIENCE_THRESHOLD = 0.12
PERCEIVED_STAFF_INTERACTION_PROBABILITY = 0.14
PERCEPTION_VISIBILITY_INTERACTION_COOLDOWN_SECONDS = 180
PERCEPTION_STAFF_INTERACTION_INTENT_TIMEOUT_SECONDS = 60
PERCEPTION_STAFF_INTERACTION_TARGET_HOLD_SECONDS = 15
PERCEPTION_REASON_RECENT_COMPLETION_COOLDOWN_SECONDS = 900
PERCEPTION_REASON_LOW_STAKES_COMPLETION_COOLDOWN_SECONDS = 1200
PERCEPTION_MAX_EXAMPLES = 12
PERCEPTION_ROLE_PAIR_SALIENCE_WEIGHTS = {
    "CoordinationNurse|Nurse": 1.45,
    "CoordinationNurse|Doctor": 1.25,
    "Doctor|Nurse": 1.05,
    "Doctor|Doctor": 1.35,
    "Nurse|Nurse": 1.05,
}
PERCEPTION_ZONE_SALIENCE_BONUSES = {
    "COCPIT": 0.10,
    "NURSTA": 0.14,
    "NUROPE": 0.12,
    "CORR01": 0.12,
    "CORR02": 0.10,
    "CORR03": 0.12,
    "CORR04": 0.12,
    "CORR05": 0.12,
    "CORR07": 0.10,
}
PERCEPTION_TARGET_BUSYNESS_PENALTY = 0.45

# Interaction backend configuration keeps movement deterministic while making
# encounter content pluggable. The rule stub is the default offline backend.
INTERACTION_BACKEND = "rule_stub"
INTERVIEW_BACKEND = "rule_stub"
LLM_REFLECTION_INTERVAL = 8
LLM_REFLECTION_WINDOW = 5
MEMORY_ENABLED = True
MEMORY_RETRIEVAL_LIMIT = 5
MEMORY_RECENCY_WEIGHT = 1.0
MEMORY_IMPORTANCE_WEIGHT = 1.0
MEMORY_RELEVANCE_WEIGHT = 1.0

# Part 3 vLLM inference runs only through the offline packet runner. Reasoning
# is disabled for the first structured-output pilot to bound token use and GPU
# cost; the batch runner exposes an explicit override.
VLLM_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
VLLM_VALIDATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_MAX_MODEL_LEN = 8_192
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLM_MAX_OUTPUT_TOKENS = 512
VLLM_ENABLE_THINKING = False
PART3_EPISODE_LOGGING_ENABLED = False
PART3_MAX_DECISION_EPISODES_PER_RUN = 40

# Calibration helpers remain importable for validation metrics and future
# larger runs, but calibration is not a top-level smoke workflow.
CALIBRATION_TEST_MODE = True
ABC_POPULATION_SIZE = 50
ABC_MAX_POPULATIONS = 4
ABC_ENSEMBLE_RUNS = 3
ABC_DISTANCE_THRESHOLD = 2.0
ABC_DISTANCE_WEIGHTS = {
    "zone_histogram": 1.0,
    "role_pair_matrix": 1.0,
    "topic_by_zone": 1.0,
    "duration_quantiles": 0.5,
    "kde_grid": 1.5,
}
ABC_PRIORS = {
    "secondary_station_probability": ("beta", 2.0, 8.0),
    "secondary_station_duration_mean": ("uniform", 60.0, 240.0),
    "corridor_exchange_probability": ("beta", 2.0, 10.0),
    "station_exchange_probability": ("beta", 2.0, 8.0),
    "salience_distance_decay": ("uniform", 2.0, 10.0),
    "attention_capacity": ("randint", 1, 3),
    "doctor_response_bias": ("uniform", 0.5, 2.0),
    "bedside_linger_multiplier": ("uniform", 0.5, 2.0),
    "memory_recency_weight": ("uniform", 0.25, 2.0),
    "memory_importance_weight": ("uniform", 0.25, 2.0),
}
OBSERVATION_WINDOW_SPEC = {
    "start_seconds": 0,
    "end_seconds": SIMULATION_DURATION_SECONDS,
}
CALIBRATION_INTERACTION_BACKEND = "rule_stub"
CALIBRATION_ABC_POPULATION_SIZE = 20
CALIBRATION_ABC_POPULATION_SIZE_FULL = 100
CALIBRATION_ABC_MAX_POPULATIONS = 3
CALIBRATION_ABC_MAX_POPULATIONS_FULL = 5
CALIBRATION_ABC_ENSEMBLE_RUNS = 3
CALIBRATION_ABC_ENSEMBLE_RUNS_FULL = 5
CALIBRATION_TEST_DISTANCE_THRESHOLD = 1.5
CALIBRATION_POSTERIOR_SAMPLES = 3
CALIBRATION_TEST_DURATION_SECONDS = 120
CALIBRATION_FULL_DURATION_SECONDS = 10_800

EXPERIMENT_CONDITIONS = ["baseline", "cockpit_only", "nursta_only", "both"]
EXPERIMENT_ENSEMBLE_RUNS = 5
EXPERIMENT_DURATION_SECONDS = 43_200
EXPERIMENT_FULL_DURATION_SECONDS = 43_200
EXPERIMENT_TEST_DURATION_SECONDS = 3_600
EXPERIMENT_INTERACTION_BACKEND = "rule_stub"
EXPERIMENT_USE_POSTERIOR_MEAN = True
