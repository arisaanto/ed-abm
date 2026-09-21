"""Central configuration for the empirically informed ED agent-based model.

Active simulation parameters, data paths, task definitions, staff composition,
spatial constants, and Part 3 inference defaults live here.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
PROJECT_ROOT = BASE_DIR.parent

WALL_POSITIONS_PATH = DATA_DIR / "wall_positions.json"
ZONE_BOUNDARIES_PATH = DATA_DIR / "zone_boundaries.json"
EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH = OUTPUTS_DIR / "empirical_topic_distributions.json"
EMPIRICAL_SHADOWING_CSV_PATH = (
    PROJECT_ROOT / "source-materials" / "shadowing_with-participant-info.csv"
)
LATEST_OUTPUT_DIR = OUTPUTS_DIR / "latest"
EMPIRICAL_WINDOW_SUMMARY_PATH = LATEST_OUTPUT_DIR / "empirical_window_summary.json"

RANDOM_SEED = 49

TIMESTEP_SECONDS = 1
SIMULATION_DURATION_SECONDS = 43_200
HOURLY_REPORT_INTERVAL_SECONDS = 3_600

ARRIVAL_RATE_PER_SECOND = 0.002
MOVEMENT_SPEED_METERS_PER_SECOND = 1.0
ARRIVAL_TOLERANCE_METERS = 0.25
WALL_CLEARANCE_METERS = 0.2

INTERACTION_DISTANCE_METERS = 1.5
SHARED_STATION_AWARENESS_DISTANCE_METERS = 2.2
TASK_DRIVEN_FOV_RELAXATION_PROBABILITY = 0.68
CORRIDOR_CLOSE_PASS_DISTANCE_METERS = 1.0
CORRIDOR_SOFT_FOV_RELAXATION_PROBABILITY = 0.42

DOCTOR_RECHECK_INTERVAL_SECONDS = 60
DOCTOR_RECHECK_MAX_ATTEMPTS = 5
PATIENT_STATE_SWEEP_INTERVAL_SECONDS = 300
MEDICAL_EVALUATION_STALE_WAIT_SECONDS = 1_800
STAFF_TRANSIT_STALL_TIMEOUT_SECONDS = 300

PATIENT_ID_START = 1_000
HOME_POSITION_WANDER_RADIUS = 0.3
STAFF_IDLE_POSITION_NOISE_METERS = 0.3
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
OPPORTUNISTIC_CORRIDOR_INTERACTION_PROBABILITY = 1.0
OPPORTUNISTIC_CORRIDOR_INTERACTION_COOLDOWN_SECONDS = 30
OPPORTUNISTIC_STATION_INTERACTION_PROBABILITY = 0.25
STATION_EPISODE_LEAVE_GRACE_SECONDS = 20
STATION_EPISODE_RADIUS_METERS = 2.4
NURSE_DOCTOR_HANDOFF_INTERACTION_PROBABILITY = 0.42
HIGH_ACUITY_NURSE_DOCTOR_HANDOFF_INTERACTION_PROBABILITY = 0.85
SECONDARY_STATION_DURATION_MEAN = 120.0
COORDINATION_PATROL_PROBABILITY = 0.0002
NURSE_CIRCULATION_PROBABILITY = 0.00016
DOCTOR_CIRCULATION_PROBABILITY = 0.00014
STAFF_CIRCULATION_MIN_IDLE_SECONDS = 60
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

# Clinical task durations developed iteratively with the active arrival process.
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

BED_PROXIMITY_METERS = 2.3
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

# Validation-counted patient-facing F2F episodes are stricter than clinical
# task attendance. Tasks still execute normally; this set only gates whether a
# task start can be logged as a shadowing-comparable patient-facing interaction.
PATIENT_FACING_VALIDATION_TASKS = {
    "CoordinationNurse": {"Placement"},
    "Nurse": {"Initial Nursing Assessment", "Discharge or Admission"},
    "Doctor": {"Medical Evaluation", "Disposition Decision"},
}
PATIENT_FACING_VALIDATION_MIN_INTERVAL_SECONDS = 900

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
ENABLE_MISSED_OPPORTUNITY_LOG = True
ENABLE_INTERRUPTION_METRICS = True
PERSONA_SOURCE = "named_default"
MISSED_OPPORTUNITY_COOLDOWN_SECONDS = 90
MISSED_OPPORTUNITY_MIN_DURATION_SECONDS = 8
MISSED_OPPORTUNITY_LOG_MODE = "episode"

DEFAULT_SCENARIO_MODE = "validated_baseline"
DEFAULT_SCENARIO_START_HOUR = 0
SCENARIO_MODES = {
    "validated_baseline": {
        "description": "Current manually developed care-area baseline used before scenario load separation.",
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
ENABLE_HIGH_ACUITY_EVENTS = True
HIGH_ACUITY_EVENT_PROBABILITY_PER_SECOND = 0.00003
HIGH_ACUITY_SOFT_PREEMPTION_ENABLED = True
HIGH_ACUITY_PREEMPTION_COOLDOWN_SECONDS = 300
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

# This explicit waypoint graph keeps movement deterministic and spatially
# grounded against walls.
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

PLOT_PADDING_METERS = 1.0
STATUS_POSITION_DECIMALS = 2

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

# The perception layer replaces radius-only visibility with line-of-sight,
# field of view, salience, and attention. Movement remains rule-based.
PERCEPTION_BASED_BASELINE_ENABLED = True
LEGACY_PROXIMITY_MODE_ENABLED = False
VISIBILITY_MAX_DISTANCE = 18.0
FOV_DEGREES = 210.0
ATTENTION_CAPACITY = 2
SALIENCE_DISTANCE_DECAY = 6.0
SALIENCE_MIN_THRESHOLD = 0.1
ISOVIST_RAY_EPSILON_RADIANS = 1e-4
ISOVIST_SAMPLE_ANGLE_DEGREES = 6.0
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

# Part 3 vLLM inference runs through the offline packet runner.
VLLM_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_MAX_MODEL_LEN = 8_192
VLLM_GPU_MEMORY_UTILIZATION = 0.90

# Grounded Part 3 episodic-memory retrieval weights. These are deliberately
# equal so that recency, contextual relevance, and recorded importance each
# contribute without an unvalidated privileged term.
MEMORY_RECENCY_WEIGHT = 1.0
MEMORY_IMPORTANCE_WEIGHT = 1.0
MEMORY_RELEVANCE_WEIGHT = 1.0
VLLM_MAX_OUTPUT_TOKENS = 512
VLLM_ENABLE_THINKING = False
PART3_EPISODE_LOGGING_ENABLED = False
PART3_MAX_DECISION_EPISODES_PER_RUN = 40

# Retained because the publication metrics use the same documented component
# weights as the original validation analysis.
ABC_DISTANCE_WEIGHTS = {
    "zone_histogram": 1.0,
    "role_pair_matrix": 1.0,
    "topic_by_zone": 1.0,
    "duration_quantiles": 0.5,
    "kde_grid": 1.5,
}
