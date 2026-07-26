"""Empirical shadowing-data utilities for calibration and validation.

This module turns the observed shadowing CSV into the same summary-statistic
space used by simulation outputs. It reuses the planning-phase JSON where that
work already encoded translated topics and role profiles, and combines it with
the raw interaction coordinates for spatial calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ModuleNotFoundError:  # pragma: no cover - stdlib fallback for base environments
    import difflib

    class _FallbackFuzz:
        @staticmethod
        def partial_ratio(left: str, right: str) -> float:
            return difflib.SequenceMatcher(None, left, right).ratio() * 100.0

    class _FallbackProcess:
        @staticmethod
        def extractOne(query, choices, scorer):
            scored = [(choice, scorer(query, choice), None) for choice in choices]
            if not scored:
                return None
            return max(scored, key=lambda item: item[1])

    fuzz = _FallbackFuzz()
    process = _FallbackProcess()

import config
from src.analysis import compute_kde_grid
from src.environment import Environment


ROLE_MAP = {
    "coordination": "CoordinationNurse",
    "nurse": "Nurse",
    "assistant_doctor": "Doctor",
    "support": "Support",
}


PARTNER_ROLE_PATTERNS = [
    ("Patient", (r"patient", r"patientin", r"patienten", r"angehörige")),
    ("Nurse", (r"pflege", r"nurse", r"nurses")),
    ("Doctor", (r"arzt", r"ärzte", r"doctor", r"oberarzt", r"assistenzarzt", r"assistenzärzte")),
    ("Support", (r"rettungsdienst", r"transport", r"sanität", r"staff")),
]


TOPIC_PATTERNS = [
    ("patient_status_update", (r"info über patient", r"patient information", r"update about situation", r"updating")),
    ("bed_and_capacity_management", (r"patientenbelegung", r"bed", r"capacity", r"room change")),
    ("administrative_and_logistics", (r"administratives", r"organisator", r"logistic", r"general information")),
    ("treatment_and_orders", (r"medication", r"medikation", r"pvk", r"infusion", r"order")),
    ("patient_flow_and_transfer", (r"transport", r"room change", r"transfer", r"patient transportation")),
    ("diagnostic_findings", (r"labor", r"untersuchungsbefund", r"blood sample", r"befund", r"ekg")),
    ("next_steps_and_planning", (r"next steps", r"prozedere", r"planning")),
    ("patient_facing_care", (r"helping patient", r"explaining", r"teaching", r"consent")),
    ("informal_or_relational", (r"informal", r"persönliches gespräch", r"personal conversation", r"food")),
]


@dataclass
class EmpiricalSummary:
    """Empirical statistics projected into the simulator summary space."""

    zone_histogram: Dict[str, float]
    role_pair_matrix: Dict[str, float]
    topic_distribution: Dict[str, float]
    topic_by_zone: Dict[str, Dict[str, int]]
    zone_supergroup_distribution: Dict[str, float]
    duration_quantiles: Dict[str, Dict[str, float]]
    kde_grid: list
    raw_f2f_count: int

    def as_dict(self) -> dict:
        return {
            "zone_histogram": self.zone_histogram,
            "role_pair_matrix": self.role_pair_matrix,
            "topic_distribution": self.topic_distribution,
            "interaction_types": {},
            "topic_by_zone": self.topic_by_zone,
            "zone_supergroup_distribution": self.zone_supergroup_distribution,
            "duration_quantiles": self.duration_quantiles,
            "kde_grid": self.kde_grid,
            "raw_f2f_count": self.raw_f2f_count,
        }


VALIDATION_TARGETS = {
    "care_area",
    "full_empirical",
}


def _normalize_counter(counter: Mapping[str, int]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {key: 0.0 for key in counter}
    return {key: value / total for key, value in counter.items()}


def load_empirical_json(path: Path = config.EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH) -> dict:
    if not path.exists():
        return {
            "behavioral_profiles": {},
            "quantitative": {
                "interactions_by_zone": {},
                "topic_distributions_by_role_and_zone": {},
            },
        }
    return json.loads(path.read_text())


def load_f2f_dataframe(path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    mask = dataframe["interfaceType"].fillna("").astype(str).str.contains("Person", case=False)
    f2f = dataframe.loc[mask].copy()
    f2f["start"] = pd.to_datetime(f2f["start"], errors="coerce")
    f2f["end"] = pd.to_datetime(f2f["end"], errors="coerce")
    f2f["duration_seconds"] = (f2f["end"] - f2f["start"]).dt.total_seconds().clip(lower=0)
    f2f["sim_role"] = f2f["role_survey"].fillna("").astype(str).map(lambda value: ROLE_MAP.get(value, value))
    return f2f


def _infer_partner_role(interface_detail: str) -> str:
    lowered_detail = interface_detail.lower()
    for role_name, patterns in PARTNER_ROLE_PATTERNS:
        if any(re.search(pattern, lowered_detail) for pattern in patterns):
            return role_name
    return "Unknown"


def _infer_topic(transmitted_information: str) -> str:
    lowered_information = transmitted_information.lower()
    for topic_name, patterns in TOPIC_PATTERNS:
        if any(re.search(pattern, lowered_information) for pattern in patterns):
            return topic_name
    return "low_information_or_unspecified"


def _build_location_aliases(empirical_json: Mapping[str, object]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for zone_id, payload in empirical_json.get("quantitative", {}).get("interactions_by_zone", {}).items():
        if not isinstance(payload, dict):
            continue
        aliases[zone_id.lower()] = zone_id
        for example in payload.get("example_location_details", []):
            aliases[str(example).lower()] = zone_id
    return aliases


def _infer_zone(
    row: pd.Series,
    environment: Environment,
    location_aliases: Mapping[str, str],
) -> str:
    x_coord = float(row.get("x_shadowing", 0.0))
    y_coord = float(row.get("y_shadowing", 0.0))
    zone_id = environment.which_zone(x_coord, y_coord)
    if zone_id is not None:
        return zone_id

    location_detail = str(row.get("locationDetail", "")).strip().lower()
    if location_detail in location_aliases:
        return location_aliases[location_detail]

    if location_detail:
        match = process.extractOne(
            location_detail,
            list(location_aliases.keys()),
            scorer=fuzz.partial_ratio,
        )
        if match is not None and match[1] >= 80:
            return location_aliases[match[0]]

    return "Outside named zones"


def classify_zone_supergroup(zone_id: str, x_coord: float, y_coord: float) -> str:
    """Map detailed ED zones into validation-relevant interaction ecology groups."""

    if zone_id in config.STATION_ZONE_IDS or zone_id in {"NEUOFF", "OFFSPA"}:
        return "station_or_desk"
    if str(zone_id).startswith("CORR"):
        return "corridor"
    if any(
        ((x_coord - bed_x) ** 2 + (y_coord - bed_y) ** 2) ** 0.5 <= config.BED_PROXIMITY_METERS
        for bed_x, bed_y in config.BED_POSITIONS
    ):
        return "bedside_or_patient_room"
    if zone_id == "Outside named zones":
        return "other_or_uncoded"
    return "other_or_uncoded"


def _point_in_polygon(point: tuple[float, float], polygon: Iterable[tuple[float, float]]) -> bool:
    x_coord, y_coord = point
    vertices = list(polygon)
    inside = False
    previous_index = len(vertices) - 1
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[previous_index]
        intersects = ((y1 > y_coord) != (y2 > y_coord)) and (
            x_coord < ((x2 - x1) * (y_coord - y1) / ((y2 - y1) + 1e-12)) + x1
        )
        if intersects:
            inside = not inside
        previous_index = index
    return inside


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0.0 and dy == 0.0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, (((px - sx) * dx) + ((py - sy) * dy)) / ((dx * dx) + (dy * dy))))
    projection = (sx + (t * dx), sy + (t * dy))
    return math.dist(point, projection)


def _point_polygon_distance(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
    if not polygon:
        return math.inf
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _nearest_zone(point: tuple[float, float], environment: Environment) -> tuple[Optional[str], float]:
    nearest = None
    nearest_distance = math.inf
    for zone_id, polygon in environment.zones.items():
        distance = _point_polygon_distance(point, polygon)
        if distance < nearest_distance:
            nearest = zone_id
            nearest_distance = distance
    return nearest, nearest_distance


def _outside_named_zone_candidate(
    x_coord: float,
    y_coord: float,
    nearest_zone: Optional[str],
    nearest_distance_m: float,
) -> str:
    """Classify uncoded empirical coordinates by nearest named affordance."""

    point = (x_coord, y_coord)
    near_bed = any(
        math.dist(point, bed_position) <= config.BED_PROXIMITY_METERS
        for bed_position in config.BED_POSITIONS
    )
    if near_bed:
        return "bedside_or_patient_room_candidate"
    if nearest_zone in {"NUROPE", "NURSTA", "COCPIT"} and nearest_distance_m <= 2.0:
        return "station_edge_candidate"
    if nearest_zone in {"NEUOFF"} and nearest_distance_m <= 2.5:
        return "staff_office_or_workroom_candidate"
    if nearest_zone in {"OFFSPA"} and nearest_distance_m <= 2.5:
        return "staff_office_or_workroom_candidate"
    if str(nearest_zone).startswith("CORR") and nearest_distance_m <= 2.0:
        return "corridor_threshold_candidate"
    if nearest_zone in {"NUROPE", "NURSTA", "COCPIT"}:
        return "station_edge_candidate"
    if str(nearest_zone).startswith("CORR"):
        return "corridor_threshold_candidate"
    if nearest_zone in {"NEUOFF", "OFFSPA"}:
        return "staff_office_or_workroom_candidate"
    return "unclear_or_unmapped"


def _active_scope_spec(environment: Environment) -> dict:
    waypoint_zones = {
        zone_id
        for point in config.ROUTING_WAYPOINTS.values()
        for zone_id in [environment.which_zone(*point)]
        if zone_id is not None
    }
    bed_zone_ids = {
        zone_id
        for point in config.BED_POSITIONS
        for zone_id in [environment.which_zone(*point)]
        if zone_id is not None
    }
    active_named_zones = set(config.STATION_ZONE_IDS) | waypoint_zones | bed_zone_ids | {
        zone_id for zone_id in [environment.which_zone(*config.PATIENT_ENTRY_POINT)] if zone_id is not None
    }
    adjacent_named_zones = {"CORR01", "CORR02", "OFFSPA"}
    inactive_named_zones = set(environment.zones) - active_named_zones - adjacent_named_zones
    return {
        "active_named_zones": sorted(active_named_zones),
        "adjacent_named_zones": sorted(adjacent_named_zones & set(environment.zones)),
        "inactive_named_zones": sorted(inactive_named_zones),
        "active_bed_positions": [
            {"bed_index": index, "position": list(position), "zone_id": environment.which_zone(*position) or "Outside named zones"}
            for index, position in enumerate(config.BED_POSITIONS)
        ],
        "staff_home_stations": {
            "CoordinationNurse": config.COORDINATION_NURSE_HOME_ZONE_ID,
            "Doctor": config.DOCTOR_HOME_ZONE_ID,
            "Nurse": list(config.NURSE_HOME_ZONE_IDS),
        },
        "routing_waypoint_zones": {
            name: {
                "position": list(point),
                "zone_id": environment.which_zone(*point) or "Outside named zones",
            }
            for name, point in sorted(config.ROUTING_WAYPOINTS.items())
        },
    }


def _scope_classification(
    zone_id: str,
    x_coord: float,
    y_coord: float,
    environment: Environment,
    scope_spec: Mapping[str, object],
) -> tuple[str, dict]:
    point = (x_coord, y_coord)
    nearest, nearest_distance = _nearest_zone(point, environment)
    near_bed = any(
        math.dist(point, bed_position) <= config.BED_PROXIMITY_METERS
        for bed_position in config.BED_POSITIONS
    )
    active_zones = set(scope_spec["active_named_zones"])
    adjacent_zones = set(scope_spec["adjacent_named_zones"])
    inactive_zones = set(scope_spec["inactive_named_zones"])
    metadata = {
        "nearest_zone": nearest,
        "nearest_zone_distance_m": nearest_distance,
        "near_configured_bed": near_bed,
    }
    if near_bed:
        return "in_simulated_scope_active", metadata
    if zone_id in active_zones:
        return "in_simulated_scope_active", metadata
    if zone_id in adjacent_zones:
        return "adjacent_to_simulated_scope", metadata
    if zone_id in inactive_zones:
        return "in_geometry_but_behaviorally_inactive", metadata
    if zone_id == "Outside named zones":
        if nearest in active_zones and nearest_distance <= 1.5:
            return "adjacent_to_simulated_scope", metadata
        if nearest in inactive_zones and nearest_distance <= 1.5:
            return "outside_simulated_scope", metadata
        return "unmapped_or_unclear", metadata
    return "unmapped_or_unclear", metadata


def _prepare_empirical_f2f_records(
    csv_path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH,
    json_path: Path = config.EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH,
    environment: Optional[Environment] = None,
) -> pd.DataFrame:
    empirical_json = load_empirical_json(json_path)
    dataframe = load_f2f_dataframe(csv_path)
    environment = environment or Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    location_aliases = _build_location_aliases(empirical_json)
    dataframe["partner_role"] = dataframe["interfaceDetail"].fillna("").astype(str).map(_infer_partner_role)
    dataframe["topic"] = dataframe["transmittedInformation"].fillna("").astype(str).map(_infer_topic)
    dataframe["zone_id"] = dataframe.apply(_infer_zone, axis=1, args=(environment, location_aliases))
    dataframe["zone_supergroup"] = dataframe.apply(
        lambda row: classify_zone_supergroup(
            str(row["zone_id"]),
            float(row.get("x_shadowing", 0.0)),
            float(row.get("y_shadowing", 0.0)),
        ),
        axis=1,
    )
    dataframe["role_pair"] = dataframe.apply(
        lambda row: "|".join(sorted((str(row["sim_role"]), str(row["partner_role"])))),
        axis=1,
    )
    nearest_zones = []
    nearest_distances = []
    outside_candidates = []
    for _, row in dataframe.iterrows():
        x_coord = float(row.get("x_shadowing", 0.0))
        y_coord = float(row.get("y_shadowing", 0.0))
        nearest_zone, nearest_distance = _nearest_zone((x_coord, y_coord), environment)
        nearest_zones.append(nearest_zone or "unknown")
        nearest_distances.append(float(nearest_distance))
        if str(row["zone_id"]) == "Outside named zones":
            outside_candidates.append(
                _outside_named_zone_candidate(x_coord, y_coord, nearest_zone, nearest_distance)
            )
        else:
            outside_candidates.append("not_outside_named_zones")
    dataframe["nearest_zone"] = nearest_zones
    dataframe["nearest_zone_distance_m"] = nearest_distances
    dataframe["outside_named_zone_candidate"] = outside_candidates
    return dataframe


def _annotate_validation_targets(dataframe: pd.DataFrame, environment: Environment) -> pd.DataFrame:
    scope_spec = _active_scope_spec(environment)
    annotated = dataframe.copy()
    classifications = []
    target_memberships = []
    for _, row in annotated.iterrows():
        classification, _ = _scope_classification(
            str(row["zone_id"]),
            float(row.get("x_shadowing", 0.0)),
            float(row.get("y_shadowing", 0.0)),
            environment,
            scope_spec,
        )
        classifications.append(classification)
        memberships = ["care_area", "full_empirical"]
        target_memberships.append(memberships)
    annotated["scope_classification"] = classifications
    annotated["validation_target_memberships"] = target_memberships
    annotated["in_care_area"] = True
    annotated["in_full_empirical"] = True
    return annotated


def _summary_from_prepared_dataframe(dataframe: pd.DataFrame, environment: Environment) -> EmpiricalSummary:
    zone_counts = dataframe["zone_id"].value_counts().to_dict()
    zone_supergroup_counts = dataframe["zone_supergroup"].value_counts().to_dict()

    role_pair_counts: Dict[str, int] = {}
    duration_quantiles: Dict[str, Dict[str, float]] = {}
    for pair_key, pair_frame in dataframe.groupby("role_pair"):
        role_pair_counts[pair_key] = int(len(pair_frame))
        durations = pair_frame["duration_seconds"].dropna().astype(float)
        if len(durations) > 0:
            duration_quantiles[pair_key] = {
                "q25": float(durations.quantile(0.25)),
                "q50": float(durations.quantile(0.50)),
                "q75": float(durations.quantile(0.75)),
            }

    topic_distribution = _normalize_counter(dataframe["topic"].value_counts().to_dict())
    topic_by_zone: Dict[str, Dict[str, int]] = {}
    for (zone_id, topic_name), count_value in dataframe.groupby(["zone_id", "topic"]).size().items():
        topic_by_zone.setdefault(str(zone_id), {})
        topic_by_zone[str(zone_id)][str(topic_name)] = int(count_value)

    points_xy = list(
        zip(
            dataframe["x_shadowing"].astype(float).tolist(),
            dataframe["y_shadowing"].astype(float).tolist(),
        )
    )
    kde_grid = compute_kde_grid(
        points_xy,
        environment.plot_bounds,
        resolution=1.0,
        bandwidth=1.5,
    ).tolist()

    return EmpiricalSummary(
        zone_histogram=_normalize_counter(zone_counts),
        role_pair_matrix=_normalize_counter(role_pair_counts),
        topic_distribution=topic_distribution,
        topic_by_zone=topic_by_zone,
        zone_supergroup_distribution=_normalize_counter(zone_supergroup_counts),
        duration_quantiles=duration_quantiles,
        kde_grid=kde_grid,
        raw_f2f_count=len(dataframe),
    )


def compute_empirical_summary(
    csv_path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH,
    json_path: Path = config.EMPIRICAL_TOPIC_DISTRIBUTIONS_PATH,
    environment: Optional[Environment] = None,
    validation_target: str = "care_area",
) -> EmpiricalSummary:
    """Build calibration-ready empirical summary statistics."""

    if validation_target not in VALIDATION_TARGETS:
        raise ValueError(f"validation_target must be one of {sorted(VALIDATION_TARGETS)}")
    environment = environment or Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    dataframe = _prepare_empirical_f2f_records(
        csv_path=csv_path,
        json_path=json_path,
        environment=environment,
    )
    dataframe = _annotate_validation_targets(dataframe, environment)
    if validation_target == "care_area":
        dataframe = dataframe.loc[dataframe["in_care_area"]].copy()
    return _summary_from_prepared_dataframe(dataframe, environment)


def describe_empirical_observation_window(
    csv_path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH,
    *,
    write_output: bool = True,
    validation_target: str = "care_area",
) -> dict:
    """Describe the empirical F2F observation window used for validation timing.

    Shadowing occurred in separate daily observation windows. Validation timing
    therefore sums the within-day windows instead of treating the calendar gap
    from the first to last record as continuous ED observation time. It does not
    treat the scenario note about patient length of stay as ED operating
    duration.
    """

    if validation_target not in VALIDATION_TARGETS:
        raise ValueError(f"validation_target must be one of {sorted(VALIDATION_TARGETS)}")
    dataframe = load_f2f_dataframe(csv_path)
    all_records = dataframe.dropna(subset=["start", "end"]).copy()
    environment = Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    prepared = _annotate_validation_targets(
        _prepare_empirical_f2f_records(csv_path=csv_path, environment=environment),
        environment,
    )
    if validation_target == "care_area":
        target_dataframe = prepared.loc[prepared["in_care_area"]].copy()
    else:
        target_dataframe = prepared.copy()
    records = target_dataframe.dropna(subset=["start", "end"]).copy()
    if records.empty:
        payload = {
            "csv_path": str(csv_path),
            "validation_target": validation_target,
            "earliest_observation_timestamp": None,
            "latest_observation_timestamp": None,
            "total_observed_clock_span_seconds": 0,
            "total_f2f_interaction_count": int(len(target_dataframe)),
            "validation_record_count": 0,
            "recommended_validation_duration_seconds": 0,
            "sampling_recommendation": "no usable timestamped F2F records found",
        }
        if write_output and validation_target == "care_area":
            config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            config.EMPIRICAL_WINDOW_SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
        return payload

    empirical_summary = compute_empirical_summary(csv_path=csv_path, environment=environment, validation_target=validation_target).as_dict()
    earliest = all_records["start"].min()
    latest = all_records["end"].max()
    continuous_span_seconds = int(max((latest - earliest).total_seconds(), 0))
    continuous_span_hours = continuous_span_seconds / 3600.0 if continuous_span_seconds > 0 else 0.0
    observation_dates = sorted({str(value.date()) for value in all_records["start"].dropna()})
    all_records["observation_date"] = all_records["start"].dt.date
    per_day_windows = []
    summed_span_seconds = 0
    target_with_dates = records.copy()
    target_with_dates["observation_date"] = target_with_dates["start"].dt.date
    target_counts_by_day = target_with_dates.groupby("observation_date").size().to_dict()
    for observation_date, day_frame in all_records.groupby("observation_date"):
        day_earliest = day_frame["start"].min()
        day_latest = day_frame["end"].max()
        day_span_seconds = int(max((day_latest - day_earliest).total_seconds(), 0))
        day_span_hours = day_span_seconds / 3600.0 if day_span_seconds > 0 else 0.0
        target_count = int(target_counts_by_day.get(observation_date, 0))
        summed_span_seconds += day_span_seconds
        per_day_windows.append(
            {
                "date": str(observation_date),
                "earliest_timestamp": day_earliest.isoformat(),
                "latest_timestamp": day_latest.isoformat(),
                "observed_span_seconds": day_span_seconds,
                "observed_span_hours": round(day_span_hours, 3),
                "f2f_record_count": target_count,
                "f2f_records_per_hour": float(target_count / day_span_hours) if day_span_hours > 0 else 0.0,
            }
        )
    span_seconds = summed_span_seconds
    span_hours = span_seconds / 3600.0 if span_seconds > 0 else 0.0
    records["hour_of_day"] = records["start"].dt.hour
    if len(observation_dates) > 1:
        recommendation = "matched daily/shift-window sampling"
    else:
        recommendation = "one continuous matched duration"
    role_pair_distribution = empirical_summary.get("role_pair_matrix", {})
    hcw_patient_share = sum(
        float(value)
        for key, value in role_pair_distribution.items()
        if "Patient" in str(key).split("|")
    )
    hcw_hcw_share = max(1.0 - hcw_patient_share, 0.0)
    zone_supergroups = empirical_summary.get("zone_supergroup_distribution", {})
    payload = {
        "csv_path": str(csv_path),
        "validation_target": validation_target,
        "earliest_observation_timestamp": earliest.isoformat(),
        "latest_observation_timestamp": latest.isoformat(),
        "continuous_calendar_span_seconds": continuous_span_seconds,
        "continuous_calendar_span_hours": round(continuous_span_hours, 2),
        "total_observed_clock_span_seconds": span_seconds,
        "total_observed_clock_span_hours": round(span_hours, 2),
        "total_f2f_interaction_count": int(len(target_dataframe)),
        "validation_record_count": int(len(records)),
        "f2f_records_per_hour": float(len(records) / span_hours) if span_hours > 0 else 0.0,
        "recommended_validation_duration_seconds": span_seconds,
        "sampling_recommendation": recommendation,
        "observation_dates": observation_dates,
        "per_day_observation_windows": per_day_windows,
        "hour_of_day_distribution": _normalize_counter(records["hour_of_day"].value_counts().to_dict()),
        "role_pair_distribution": role_pair_distribution,
        "hcw_hcw_share": hcw_hcw_share,
        "hcw_patient_share": hcw_patient_share,
        "patient_facing_share": hcw_patient_share,
        "zone_distribution": empirical_summary.get("zone_histogram", {}),
        "zone_supergroup_distribution": empirical_summary.get("zone_supergroup_distribution", {}),
        "station_zone_share": float(zone_supergroups.get("station_or_desk", 0.0)),
        "corridor_zone_share": float(zone_supergroups.get("corridor", 0.0)),
        "bedside_or_patient_room_share": float(zone_supergroups.get("bedside_or_patient_room", 0.0)),
        "other_or_uncoded_share": float(zone_supergroups.get("other_or_uncoded", 0.0)),
        "topic_distribution": empirical_summary.get("topic_distribution", {}),
        "duration_quantiles_by_role_pair": empirical_summary.get("duration_quantiles", {}),
        "notes": [
            "The ED is assumed to operate continuously; this window describes empirical shadowing coverage, not ED opening hours.",
            "The validation duration sums within-day observation spans. The continuous calendar gap between first and last record is reported only as metadata.",
            "Any 4.4-hour scenario-building value is treated as patient length-of-stay/episode duration, not operating duration.",
        ],
    }
    if write_output and validation_target == "care_area":
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.EMPIRICAL_WINDOW_SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    return payload
