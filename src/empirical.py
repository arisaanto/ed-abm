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


def _care_area_object_for_point(x_coord: float, y_coord: float) -> Optional[str]:
    del x_coord, y_coord
    return None


def _apply_care_area_object_zones(dataframe: pd.DataFrame) -> pd.DataFrame:
    annotated = dataframe.copy()
    for index, row in annotated.iterrows():
        x_coord = float(row.get("x_shadowing", 0.0))
        y_coord = float(row.get("y_shadowing", 0.0))
        object_zone = _care_area_object_for_point(x_coord, y_coord)
        if object_zone is None:
            continue
        annotated.at[index, "zone_id"] = object_zone
        annotated.at[index, "zone_supergroup"] = classify_zone_supergroup(object_zone, x_coord, y_coord)
    return annotated


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


def audit_scope_alignment(
    csv_path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH,
    *,
    write_output: bool = True,
) -> dict:
    """Compare empirical F2F locations with the active care-area simulation scope."""

    environment = Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    scope_spec = _active_scope_spec(environment)
    dataframe = _prepare_empirical_f2f_records(csv_path=csv_path, environment=environment)
    classifications = []
    nearest_for_outside = []
    for _, row in dataframe.iterrows():
        x_coord = float(row.get("x_shadowing", 0.0))
        y_coord = float(row.get("y_shadowing", 0.0))
        classification, metadata = _scope_classification(
            str(row["zone_id"]),
            x_coord,
            y_coord,
            environment,
            scope_spec,
        )
        classifications.append(classification)
        if str(row["zone_id"]) == "Outside named zones":
            nearest_for_outside.append(metadata["nearest_zone"] or "unknown")
    dataframe["scope_classification"] = classifications

    def normalized_counts(series) -> dict:
        counts = series.value_counts().to_dict()
        total = max(sum(counts.values()), 1)
        return {
            str(key): {
                "count": int(value),
                "share": float(value / total),
            }
            for key, value in counts.items()
        }

    role_pair = dataframe["role_pair"]
    hcw_hcw_mask = ~role_pair.astype(str).str.contains("Patient", regex=False)
    hcw_patient_mask = role_pair.astype(str).str.contains("Patient", regex=False)
    total_records = max(len(dataframe), 1)
    classification_counts = dataframe["scope_classification"].value_counts().to_dict()
    out_of_active_share = sum(
        classification_counts.get(label, 0)
        for label in (
            "in_geometry_but_behaviorally_inactive",
            "adjacent_to_simulated_scope",
            "outside_simulated_scope",
            "unmapped_or_unclear",
        )
    ) / total_records
    hcw_hcw_records = dataframe.loc[hcw_hcw_mask]
    hcw_hcw_total = max(len(hcw_hcw_records), 1)
    hcw_hcw_classification_counts = hcw_hcw_records["scope_classification"].value_counts().to_dict()
    hcw_hcw_out_of_active_share = sum(
        hcw_hcw_classification_counts.get(label, 0)
        for label in (
            "in_geometry_but_behaviorally_inactive",
            "adjacent_to_simulated_scope",
            "outside_simulated_scope",
            "unmapped_or_unclear",
        )
    ) / hcw_hcw_total
    mismatch = out_of_active_share > 0.25 or hcw_hcw_out_of_active_share > 0.25
    if mismatch:
        strategy = "hybrid"
        recommendation = (
            "Use care-area validation for the current ABM and keep full-ED validation separate "
            "unless the model scope is expanded."
        )
    else:
        strategy = "experimental_scope_validation"
        recommendation = "The current empirical target is mostly aligned with active simulated scope; continue with experimental-scope validation checks."

    zone_status = {}
    for zone_id in sorted(environment.zones):
        if zone_id in scope_spec["active_named_zones"]:
            status = "active_destination_or_route_zone"
        elif zone_id in scope_spec["adjacent_named_zones"]:
            status = "geometrically_present_adjacent_but_not_destination"
        else:
            status = "geometrically_present_but_behaviorally_inactive"
        zone_status[zone_id] = status

    payload = {
        "purpose": "Audit whether empirical F2F shadowing scope matches the active care-area ED ABM scope.",
        "simulated_scope": {
            "scope_label": "care_area",
            "active_patient_care_areas": "Ten configured bed positions plus patient-entry route; bed positions are active by proximity.",
            "active_beds": scope_spec["active_bed_positions"],
            "staff_home_stations": scope_spec["staff_home_stations"],
            "zones_existing_in_geometry": sorted(environment.zones),
            "active_named_zones": scope_spec["active_named_zones"],
            "adjacent_named_zones": scope_spec["adjacent_named_zones"],
            "inactive_named_zones": scope_spec["inactive_named_zones"],
            "routing_waypoint_zones": scope_spec["routing_waypoint_zones"],
            "zone_behavior_status": zone_status,
            "neuoff_present": "NEUOFF" in environment.zones,
            "nurope_present": "NUROPE" in environment.zones,
            "nursta_present": "NURSTA" in environment.zones,
            "neuoff_behavior": zone_status.get("NEUOFF"),
            "nurope_behavior": zone_status.get("NUROPE"),
            "nursta_behavior": zone_status.get("NURSTA"),
        },
        "empirical_scope": {
            "total_f2f_records": int(len(dataframe)),
            "coordinate_available_records": int(dataframe[["x_shadowing", "y_shadowing"]].notna().all(axis=1).sum()),
            "records_by_zone": normalized_counts(dataframe["zone_id"]),
            "records_by_zone_supergroup": normalized_counts(dataframe["zone_supergroup"]),
            "records_by_role_pair": normalized_counts(dataframe["role_pair"]),
            "records_by_topic": normalized_counts(dataframe["topic"]),
            "duration_seconds_by_role_pair": {
                str(pair): {
                    "q25": float(group["duration_seconds"].quantile(0.25)),
                    "q50": float(group["duration_seconds"].quantile(0.50)),
                    "q75": float(group["duration_seconds"].quantile(0.75)),
                }
                for pair, group in dataframe.groupby("role_pair")
                if len(group["duration_seconds"].dropna()) > 0
            },
            "hour_of_day_distribution": _normalize_counter(dataframe["start"].dt.hour.value_counts().to_dict()),
            "outside_named_zones_nearest_zone_counts": normalized_counts(pd.Series(nearest_for_outside, dtype=str)),
        },
        "scope_alignment": {
            "overall": normalized_counts(dataframe["scope_classification"]),
            "hcw_hcw_only": normalized_counts(hcw_hcw_records["scope_classification"]),
            "hcw_patient_only": normalized_counts(dataframe.loc[hcw_patient_mask, "scope_classification"]),
            "inside_active_simulated_scope_share": float(classification_counts.get("in_simulated_scope_active", 0) / total_records),
            "inactive_adjacent_outside_or_unclear_share": float(out_of_active_share),
            "hcw_hcw_inside_active_simulated_scope_share": float(hcw_hcw_classification_counts.get("in_simulated_scope_active", 0) / hcw_hcw_total),
            "hcw_hcw_inactive_adjacent_outside_or_unclear_share": float(hcw_hcw_out_of_active_share),
        },
        "validation_strategy": {
            "recommended_strategy": strategy,
            "scope_mismatch_flag": mismatch,
            "threshold_note": "Flagged when more than roughly 25% of all F2F or HCW-HCW records fall outside active simulated scope.",
            "recommendation": recommendation,
            "full_empirical_target": "real ED shadowing scope",
            "experimental_scope_target": "subset classified as in_simulated_scope_active",
            "fairness_note": "Comparing the current care-area ABM to full empirical shadowing is not fair if substantial empirical interactions occur in inactive, adjacent, outside, or unclear scope.",
        },
    }
    if write_output:
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.SCOPE_ALIGNMENT_AUDIT_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _classify_missing_affordance(zone_id: str, nearest_zone: Optional[str] = None) -> dict:
    """Return a conservative affordance interpretation for scope-audit zones."""

    lookup_zone = zone_id if zone_id != "Outside named zones" else (nearest_zone or zone_id)
    if lookup_zone == "NEUOFF":
        return {
            "likely_affordance": "staff office / workroom",
            "confidence": "medium",
            "evidence": "Geometry name suggests a staff office/workroom; empirical outside-zone records also cluster nearest NEUOFF.",
        }
    if lookup_zone == "OFFSPA":
        return {
            "likely_affordance": "staff support / informal area",
            "confidence": "low_to_medium",
            "evidence": "Geometry name suggests office/support space, but current empirical mass is small and function is not explicit in code.",
        }
    if lookup_zone in {"NUROPE", "NURSTA"}:
        return {
            "likely_affordance": "nurse station / satellite station",
            "confidence": "high",
            "evidence": "Configured as nurse home/station zones and present as named empirical F2F locations.",
        }
    if lookup_zone == "COCPIT":
        return {
            "likely_affordance": "coordination station / central staff cockpit",
            "confidence": "high",
            "evidence": "Configured as coordination nurse and doctor home station.",
        }
    if str(lookup_zone).startswith("CORR"):
        return {
            "likely_affordance": "corridor / threshold",
            "confidence": "high",
            "evidence": "Named corridor geometry and empirical F2F records occur there.",
        }
    if zone_id == "Outside named zones":
        return {
            "likely_affordance": "uncoded bedside / room / threshold space",
            "confidence": "medium",
            "evidence": "Coordinates are outside named polygons; nearest-zone distribution indicates many are close to corridors, stations, or NEUOFF rather than truly out of map.",
        }
    return {
        "likely_affordance": "unclear",
        "confidence": "low",
        "evidence": "No strong functional label in current geometry/config.",
    }


def _load_scenario_building_evidence() -> dict:
    """Extract compact, non-authoritative notes from ScenarioBuildingAris if available."""

    pdf_path = config.PROJECT_ROOT / "ScenarioBuildingAris.pdf"
    if not pdf_path.exists():
        return {
            "available": False,
            "path": str(pdf_path),
            "notes": ["ScenarioBuildingAris.pdf was not found; audit relies on empirical data and geometry."],
        }
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pragma: no cover - depends on local PDF backend
        return {
            "available": True,
            "path": str(pdf_path),
            "text_extracted": False,
            "notes": [f"PDF exists but text extraction failed: {exc}"],
        }

    lowered = text.lower()
    notes = []
    if "125" in lowered and "patients" in lowered:
        notes.append("Scenario notes mention roughly 125 ED patients/day.")
    if "4.4" in lowered and "ed" in lowered:
        notes.append("Scenario notes mention ~4.4 hours in the ED; this is treated as patient length-of-stay/time-in-department, not ED operating duration.")
    if "flow coordinator" in lowered or "flowc" in lowered:
        notes.append("Scenario notes include a flow coordinator role, corresponding conceptually to the ABM CoordinationNurse.")
    if "treatment bay" in lowered or "koje" in lowered:
        notes.append("Scenario notes include treatment bays/Koje, supporting patient-room/bedside affordances around the modeled care area.")
    if "fast track" in lowered:
        notes.append("Scenario notes mention Fast Track, which is broader ED ecology and not represented by the current care-area ABM.")
    if "shock room" in lowered or "schockraum" in lowered:
        notes.append("Scenario notes mention shock/resuscitation rooms, represented here only by the reserved high-acuity bed rather than a separate active resuscitation room.")
    if "safe" in lowered and "room" in lowered:
        notes.append("Scenario notes mention a safe/quiet room for psychiatric crisis, outside the current care-area scope.")
    return {
        "available": True,
        "path": str(pdf_path),
        "text_extracted": True,
        "notes": notes or ["PDF text extraction succeeded, but no specific scope note was confidently extracted."],
    }


def audit_ecological_scope(
    csv_path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH,
    *,
    write_output: bool = True,
) -> dict:
    """Audit whether the current care-area scope is enough for broader ED ecology claims."""

    environment = Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    scope_alignment = audit_scope_alignment(csv_path=csv_path, write_output=False)
    scope_spec = _active_scope_spec(environment)
    dataframe = _prepare_empirical_f2f_records(csv_path=csv_path, environment=environment)

    classifications = []
    nearest_zones = []
    nearest_distances = []
    for _, row in dataframe.iterrows():
        classification, metadata = _scope_classification(
            str(row["zone_id"]),
            float(row.get("x_shadowing", 0.0)),
            float(row.get("y_shadowing", 0.0)),
            environment,
            scope_spec,
        )
        classifications.append(classification)
        nearest_zones.append(metadata.get("nearest_zone") or "unknown")
        nearest_distances.append(float(metadata.get("nearest_zone_distance_m", math.nan)))
    dataframe["scope_classification"] = classifications
    dataframe["nearest_zone"] = nearest_zones
    dataframe["nearest_zone_distance_m"] = nearest_distances

    excluded_labels = {
        "in_geometry_but_behaviorally_inactive",
        "adjacent_to_simulated_scope",
        "outside_simulated_scope",
        "unmapped_or_unclear",
    }
    excluded = dataframe.loc[dataframe["scope_classification"].isin(excluded_labels)].copy()

    def count_share(frame: pd.DataFrame, column: str) -> dict:
        counts = frame[column].value_counts(dropna=False).to_dict()
        total = max(len(frame), 1)
        return {
            str(key): {
                "count": int(value),
                "share": float(value / total),
            }
            for key, value in counts.items()
        }

    missing_zone_rows = []
    for zone_id, group in excluded.groupby("zone_id"):
        nearest_counts = group["nearest_zone"].value_counts().to_dict()
        top_nearest = max(nearest_counts, key=nearest_counts.get) if nearest_counts else None
        affordance = _classify_missing_affordance(str(zone_id), top_nearest)
        missing_zone_rows.append(
            {
                "zone_id": str(zone_id),
                "count": int(len(group)),
                "share_of_all_f2f": float(len(group) / max(len(dataframe), 1)),
                "scope_classification_counts": count_share(group, "scope_classification"),
                "top_nearest_zones": {
                    str(key): int(value)
                    for key, value in sorted(nearest_counts.items(), key=lambda item: item[1], reverse=True)[:8]
                },
                **affordance,
            }
        )
    missing_zone_rows.sort(key=lambda item: item["count"], reverse=True)

    outside = dataframe.loc[dataframe["zone_id"] == "Outside named zones"].copy()
    outside_nearest_rows = []
    for nearest_zone, group in outside.groupby("nearest_zone"):
        affordance = _classify_missing_affordance("Outside named zones", str(nearest_zone))
        outside_nearest_rows.append(
            {
                "nearest_zone": str(nearest_zone),
                "count": int(len(group)),
                "share_of_outside_named_zones": float(len(group) / max(len(outside), 1)),
                "share_of_all_f2f": float(len(group) / max(len(dataframe), 1)),
                "median_distance_to_nearest_zone_m": float(group["nearest_zone_distance_m"].median()),
                **affordance,
            }
        )
    outside_nearest_rows.sort(key=lambda item: item["count"], reverse=True)

    zone_status = scope_alignment["simulated_scope"]["zone_behavior_status"]
    zones_of_interest = {}
    for zone_id in ["NEUOFF", "OFFSPA", "CORR01", "CORR02", "CORRFT", "NUROPE", "NURSTA", "COCPIT"]:
        zone_frame = dataframe.loc[dataframe["zone_id"] == zone_id]
        zones_of_interest[zone_id] = {
            "exists_in_geometry": zone_id in environment.zones,
            "behavior_status": zone_status.get(zone_id, "not_in_geometry"),
            "empirical_count": int(len(zone_frame)),
            "empirical_share": float(len(zone_frame) / max(len(dataframe), 1)),
            **_classify_missing_affordance(zone_id),
        }

    latest_validation = {}
    if config.VALIDATION_SUMMARY_PATH.exists():
        try:
            validation_payload = json.loads(config.VALIDATION_SUMMARY_PATH.read_text())
            for variant, row in validation_payload.get("variants", {}).items():
                zone_distribution = row.get("zone_distribution", {})
                latest_validation[variant] = {
                    "nursta_share": float(zone_distribution.get("NURSTA", 0.0)),
                    "nurope_share": float(zone_distribution.get("NUROPE", 0.0)),
                    "cocpit_share": float(zone_distribution.get("COCPIT", 0.0)),
                    "outside_named_zones_share": float(zone_distribution.get("Outside named zones", 0.0)),
                }
        except Exception:
            latest_validation = {"parse_error": "Could not parse latest validation summary."}

    overall_out_of_active = scope_alignment["scope_alignment"]["inactive_adjacent_outside_or_unclear_share"]
    hcw_hcw_out_of_active = scope_alignment["scope_alignment"]["hcw_hcw_inactive_adjacent_outside_or_unclear_share"]
    hcw_patient_inside = scope_alignment["scope_alignment"]["hcw_patient_only"].get(
        "in_simulated_scope_active", {}
    ).get("share", 0.0)
    outside_share = scope_alignment["empirical_scope"]["records_by_zone"].get("Outside named zones", {}).get("share", 0.0)
    if overall_out_of_active > 0.25 or outside_share > 0.25 or hcw_patient_inside < 0.60:
        recommendation = "add_surrounding_care_area_ecology"
        rationale = (
            "The current care-area scope is defensible for controlled EVIDENT actor-experiment claims, "
            "but the full empirical shadowing target includes substantial adjacent/uncoded patient-care and threshold space."
        )
    elif hcw_hcw_out_of_active > 0.20:
        recommendation = "add_minimal_inactive_zone_affordances"
        rationale = "HCW-HCW empirical interactions spill into inactive or adjacent staff areas enough to affect ecology claims."
    else:
        recommendation = "stay_experimental_scope_only"
        rationale = "Most empirical F2F mass is aligned with the active care-area scope."

    payload = {
        "purpose": "Minimal ecological expansion audit before changing the care-area ED ABM scope.",
        "target_assessment": {
            "controlled_evident_care_area_setup": {
                "support": "supportable",
                "interpretation": "The current model matches the controlled care-area intent if empirical targets are filtered to active simulated scope.",
            },
            "care_area_ecology_around_redesign": {
                "support": "partially_supportable_with_minimal_expansion",
                "interpretation": "Empirical F2F records show meaningful mass in adjacent, station, corridor, and uncoded threshold/room spaces; broader care-area claims likely need minimal surrounding-care-area affordances or a filtered target.",
            },
            "full_zurich_ed_operational_ecology": {
                "support": "not_supportable_currently",
                "interpretation": "Scenario evidence includes broader ED functions such as Fast Track, shock room, safe/quiet room, support and specialist roles; the current ABM deliberately excludes those.",
            },
        },
        "scope_alignment_summary": {
            "inside_active_simulated_scope_share": scope_alignment["scope_alignment"]["inside_active_simulated_scope_share"],
            "inactive_adjacent_outside_or_unclear_share": overall_out_of_active,
            "hcw_hcw_inside_active_simulated_scope_share": scope_alignment["scope_alignment"]["hcw_hcw_inside_active_simulated_scope_share"],
            "hcw_hcw_inactive_adjacent_outside_or_unclear_share": hcw_hcw_out_of_active,
            "hcw_patient_scope_breakdown": scope_alignment["scope_alignment"]["hcw_patient_only"],
        },
        "top_excluded_adjacent_inactive_or_unclear_zones": missing_zone_rows,
        "outside_named_zones_nearest_clusters": outside_nearest_rows,
        "zones_of_interest": zones_of_interest,
        "nurope_nursta_use_note": {
            "empirical_nursta_share": scope_alignment["empirical_scope"]["records_by_zone"].get("NURSTA", {}).get("share", 0.0),
            "empirical_nurope_share": scope_alignment["empirical_scope"]["records_by_zone"].get("NUROPE", {}).get("share", 0.0),
            "latest_validation_zone_shares_if_available": latest_validation,
            "interpretation": "NUROPE and NURSTA are active in the code. Any underuse should be assessed against diagnostic-mode-off validation summaries rather than assumed from geometry alone.",
        },
        "scenario_building_evidence": _load_scenario_building_evidence(),
        "recommendation": {
            "decision": recommendation,
            "rationale": rationale,
            "minimal_next_code_changes_if_implementation_proceeds": [
                "Preserve the active care-area scope as the default simulation setup.",
                "Use the care-area validation target after supervisor agreement on the broader care-area scope.",
                "Add low-frequency staff-only NEUOFF/OFFSPA documentation or consultation dwell only if those areas are declared in-scope.",
                "Add explicit patient-room/bedside/threshold polygons or best-effort care-room clusters before treating Outside named zones as a validation failure.",
                "Wire validation target selection to full vs care-area empirical records before claiming ecological fit.",
            ],
        },
        "do_not_implement_yet": [
            "Do not add admin, triage, support, senior doctor, or specialist as full agents based only on this audit.",
            "Do not expand to Fast Track, shock room, safe room, imaging, ICU transfer, or full hospital pathways.",
            "Do not tune interaction probabilities merely to match full-ED empirical zone shares.",
            "Do not let generative logic select destinations or repair missing spatial affordances.",
            "Do not claim Part 2 readiness until the validation target is scope-aligned and diagnostic-mode-off validation supports the selected baseline.",
        ],
    }
    if write_output:
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.ECOLOGICAL_SCOPE_AUDIT_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _frame_distribution(frame: pd.DataFrame, column: str, *, limit: int = 8) -> dict:
    counts = frame[column].fillna("unknown").astype(str).value_counts().head(limit)
    total = max(len(frame), 1)
    return {
        str(label): {"count": int(count), "share": float(count / total)}
        for label, count in counts.items()
    }


def _duration_summary(frame: pd.DataFrame) -> dict:
    durations = frame["duration_seconds"].dropna().astype(float)
    if durations.empty:
        return {"count": 0}
    return {
        "count": int(len(durations)),
        "mean_seconds": float(durations.mean()),
        "median_seconds": float(durations.median()),
        "q25_seconds": float(durations.quantile(0.25)),
        "q75_seconds": float(durations.quantile(0.75)),
        "max_seconds": float(durations.max()),
    }


def _patient_facing_breakdown(frame: pd.DataFrame) -> dict:
    patient_mask = frame["role_pair"].astype(str).str.contains("Patient", regex=False)
    total = max(len(frame), 1)
    return {
        "patient_facing_count": int(patient_mask.sum()),
        "patient_facing_share": float(patient_mask.sum() / total),
        "hcw_hcw_count": int((~patient_mask).sum()),
        "hcw_hcw_share": float((~patient_mask).sum() / total),
    }


def _cluster_coordinate_summary(frame: pd.DataFrame) -> dict:
    return {
        "centroid": {
            "x": float(frame["x_shadowing"].astype(float).mean()),
            "y": float(frame["y_shadowing"].astype(float).mean()),
        },
        "bounds": {
            "min_x": float(frame["x_shadowing"].astype(float).min()),
            "min_y": float(frame["y_shadowing"].astype(float).min()),
            "max_x": float(frame["x_shadowing"].astype(float).max()),
            "max_y": float(frame["y_shadowing"].astype(float).max()),
        },
    }


def _cluster_affordance_inference(label: str, frame: pd.DataFrame) -> dict:
    zone_counts = frame["zone_id"].fillna("").astype(str).value_counts().to_dict()
    nearest_counts = frame["nearest_zone"].fillna("").astype(str).value_counts().to_dict()
    patient_share = _patient_facing_breakdown(frame)["patient_facing_share"]

    if label == "left_of_CORR04_lower_left":
        return {
            "likely_affordance": "patient room / bedside cluster or procedure-room threshold near NEUOFF/NUROPE",
            "confidence": "medium",
            "evidence": "Records are outside named polygons, nearest mostly NEUOFF/NUROPE/CORR04, and patient-facing interactions dominate.",
            "model_change_implication": "If broader care-area ecology is targeted, add room/threshold polygons before adding new staff.",
            "requires": ["new room/polygon affordances", "possibly added patient beds/care bays"],
        }
    if label == "below_central_corridor_lower_middle":
        return {
            "likely_affordance": "NURSTA station-edge plus adjacent lower-middle threshold/room ecology",
            "confidence": "medium_to_high",
            "evidence": "NURSTA named records and outside-near-NURSTA/CORR05/CORR04 records concentrate here; Nurse|Nurse and CoordinationNurse|Nurse are common.",
            "model_change_implication": "Prioritize station dwell/co-presence and station-edge geometry; do not add agents solely from this cluster.",
            "requires": ["added station-dwell behavior", "new room/polygon affordances"],
        }
    if label == "right_of_CORR05_lower_right_room_stack":
        return {
            "likely_affordance": "patient room / bedside stack or care-bay threshold near CORR05",
            "confidence": "medium",
            "evidence": "Outside-named-zone records nearest CORR05 include Doctor|Patient, Nurse|Patient, and Doctor|Doctor interactions.",
            "model_change_implication": "Likely missing patient-care room polygons or care bays if expanded care-area validation is desired.",
            "requires": ["new room/polygon affordances", "possibly added patient beds/care bays"],
        }
    if label == "above_and_slightly_left_of_NUROPE":
        return {
            "likely_affordance": "NUROPE/NEUOFF threshold or adjacent patient-care/workroom edge",
            "confidence": "medium",
            "evidence": "Outside records nearest NUROPE/NEUOFF dominate; many are CoordinationNurse|Patient, suggesting more than a staff-only office.",
            "model_change_implication": "Separate staff-office affordance from adjacent patient-care/threshold affordance before expanding behavior.",
            "requires": ["new room/polygon affordances", "added staff co-presence"],
        }
    if label == "NURSTA":
        return {
            "likely_affordance": "nurse station / station-edge coordination hub",
            "confidence": "high",
            "evidence": "Named NURSTA records are dominated by Nurse|Nurse and CoordinationNurse|Nurse interactions.",
            "model_change_implication": "The current zero simulated NURSTA share is more likely station dwell/co-presence failure than missing geometry.",
            "requires": ["added station-dwell behavior", "added staff co-presence"],
        }
    if label == "NUROPE":
        return {
            "likely_affordance": "open nurse station / documentation and coordination hub",
            "confidence": "high",
            "evidence": "Named NUROPE records are dominated by CoordinationNurse|Nurse, often with person/computer interfaces.",
            "model_change_implication": "The current zero simulated NUROPE share is more likely station dwell/co-presence failure than missing geometry.",
            "requires": ["added station-dwell behavior", "added staff co-presence"],
        }
    if label == "COCPIT":
        return {
            "likely_affordance": "central cockpit / coordination station",
            "confidence": "high",
            "evidence": "Named COCPIT records include Doctor|Doctor, Doctor|Nurse, and coordination interactions; this is already active in the model.",
            "model_change_implication": "No scope expansion is required for COCPIT, but overconcentration there can indicate missing satellite station ecology.",
            "requires": ["no model change / empirical artifact only"],
        }
    if str(label).startswith("auto_cluster"):
        inferred_zone = max(nearest_counts, key=nearest_counts.get) if nearest_counts else max(zone_counts, key=zone_counts.get)
        if patient_share >= 0.50 and str(inferred_zone).startswith("CORR"):
            affordance = "corridor-adjacent patient room / bedside threshold"
            requires = ["new room/polygon affordances", "possibly added patient beds/care bays"]
        elif inferred_zone in {"NURSTA", "NUROPE", "COCPIT"}:
            affordance = "station-edge or staff coordination cluster"
            requires = ["added station-dwell behavior", "added staff co-presence"]
        elif inferred_zone == "NEUOFF":
            affordance = "staff workroom / office edge or adjacent patient-care threshold"
            requires = ["new room/polygon affordances"]
        else:
            affordance = _classify_missing_affordance(str(inferred_zone)).get("likely_affordance", "unclear")
            requires = ["new room/polygon affordances"]
        return {
            "likely_affordance": affordance,
            "confidence": "low_to_medium",
            "evidence": "Automatic coordinate cluster; interpretation is based on nearest-zone, role-pair, and patient-facing composition.",
            "model_change_implication": "Use as supporting evidence only; manually reviewed clusters should drive modeling decisions.",
            "requires": requires,
        }
    return {
        "likely_affordance": "unclear",
        "confidence": "low",
        "evidence": "No specific manual interpretation was configured for this cluster.",
        "model_change_implication": "Do not implement behavior from this cluster without review.",
        "requires": ["no model change / empirical artifact only"],
    }


def _summarize_cluster(
    label: str,
    frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    experimental_total: int,
    expanded_total: int,
    *,
    cluster_kind: str,
) -> dict:
    if frame.empty:
        return {
            "label": label,
            "cluster_kind": cluster_kind,
            "record_count": 0,
            "share_of_all_empirical_records": 0.0,
            "note": "No empirical F2F records matched this cluster definition.",
        }
    care_area_count = int(frame["in_care_area"].sum())
    text_examples = [
        str(value)[:260]
        for value in frame["text"].dropna().astype(str).head(5).tolist()
    ]
    return {
        "label": label,
        "cluster_kind": cluster_kind,
        **_cluster_coordinate_summary(frame),
        "record_count": int(len(frame)),
        "share_of_all_empirical_records": float(len(frame) / max(len(full_frame), 1)),
        "care_area_record_count": care_area_count,
        "share_of_care_area_target": float(care_area_count / max(experimental_total, 1)),
        "full_empirical_record_count": int(len(frame)),
        "share_of_full_empirical_target": float(len(frame) / max(expanded_total, 1)),
        "zone_distribution": _frame_distribution(frame, "zone_id"),
        "nearest_named_zone_distribution": _frame_distribution(frame, "nearest_zone"),
        "outside_named_zone_candidate_distribution": _frame_distribution(frame, "outside_named_zone_candidate"),
        "scope_classification_distribution": _frame_distribution(frame, "scope_classification"),
        "role_pair_distribution": _frame_distribution(frame, "role_pair"),
        "topic_distribution": _frame_distribution(frame, "topic"),
        "description_or_transmitted_information_distribution": _frame_distribution(frame, "transmittedInformation"),
        "patient_facing_vs_hcw_hcw": _patient_facing_breakdown(frame),
        "duration_summary": _duration_summary(frame),
        "interface_type_distribution": _frame_distribution(frame, "interfaceType"),
        "interface_detail_distribution": _frame_distribution(frame, "interfaceDetail"),
        "text_examples": text_examples,
        "affordance_inference": _cluster_affordance_inference(label, frame),
    }


def _dbscan_labels(frame: pd.DataFrame, *, eps: float = 1.75, min_points: int = 4) -> dict:
    points = [
        (int(index), float(row["x_shadowing"]), float(row["y_shadowing"]))
        for index, row in frame.iterrows()
    ]
    neighbors = {
        index: [
            other_index
            for other_index, other_x, other_y in points
            if math.dist((x_coord, y_coord), (other_x, other_y)) <= eps
        ]
        for index, x_coord, y_coord in points
    }
    labels: dict[int, int] = {}
    cluster_id = 0
    for index, _, _ in points:
        if index in labels:
            continue
        if len(neighbors[index]) < min_points:
            labels[index] = -1
            continue
        labels[index] = cluster_id
        seeds = list(neighbors[index])
        cursor = 0
        while cursor < len(seeds):
            neighbor_index = seeds[cursor]
            if labels.get(neighbor_index) == -1:
                labels[neighbor_index] = cluster_id
            if neighbor_index not in labels:
                labels[neighbor_index] = cluster_id
                if len(neighbors[neighbor_index]) >= min_points:
                    for candidate in neighbors[neighbor_index]:
                        if candidate not in seeds:
                            seeds.append(candidate)
            cursor += 1
        cluster_id += 1
    return labels


def _write_spatial_cluster_audit_markdown(payload: Mapping[str, object]) -> None:
    clusters = payload.get("clusters", [])
    lines = [
        "# Spatial Cluster Affordance Audit",
        "",
        "Status: provisional spatial-forensic audit, not a finalized finding.",
        "",
        "## Summary",
        payload.get("plain_language_summary", ""),
        "",
        "## Top Dense / Underrepresented Clusters",
        "| Cluster | Records | Share all | Likely affordance | Confidence | Requires |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for cluster in clusters:
        inference = cluster.get("affordance_inference", {})
        lines.append(
            "| {label} | {count} | {share:.3f} | {affordance} | {confidence} | {requires} |".format(
                label=cluster.get("label", ""),
                count=int(cluster.get("record_count", 0)),
                share=float(cluster.get("share_of_all_empirical_records", 0.0)),
                affordance=str(inference.get("likely_affordance", "")),
                confidence=str(inference.get("confidence", "")),
                requires=", ".join(inference.get("requires", [])),
            )
        )
    lines.extend([
        "",
        "## NURSTA / NUROPE Interpretation",
        payload.get("nursta_nurope_diagnosis", {}).get("interpretation", ""),
        "",
        "## Evidence For / Against More Nurses",
        f"- For: {payload.get('staffing_and_bed_evidence', {}).get('evidence_for_more_nurses', '')}",
        f"- Against: {payload.get('staffing_and_bed_evidence', {}).get('evidence_against_more_nurses', '')}",
        "",
        "## Evidence For / Against More Beds or Rooms",
        f"- For: {payload.get('staffing_and_bed_evidence', {}).get('evidence_for_more_beds_or_rooms', '')}",
        f"- Against: {payload.get('staffing_and_bed_evidence', {}).get('evidence_against_more_beds_or_rooms', '')}",
        "",
        "## ScenarioBuildingAris",
        payload.get("scenario_building_interpretation", {}).get("interpretation", ""),
        "",
        "## Recommended Next Modeling Decision",
        payload.get("recommendation", {}).get("decision", ""),
        "",
        payload.get("recommendation", {}).get("rationale", ""),
        "",
    ])
    config.SPATIAL_CLUSTER_AFFORDANCE_AUDIT_MD_PATH.write_text("\n".join(lines))


def audit_spatial_cluster_affordances(
    csv_path: Path = config.EMPIRICAL_SHADOWING_CSV_PATH,
    *,
    write_output: bool = True,
) -> dict:
    """Forensically audit high-density empirical F2F clusters before model expansion."""

    environment = Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    dataframe = _annotate_validation_targets(
        _prepare_empirical_f2f_records(csv_path=csv_path, environment=environment),
        environment,
    )
    experimental_total = int(dataframe["in_care_area"].sum())
    expanded_total = int(len(dataframe))

    x_coord = dataframe["x_shadowing"].astype(float)
    y_coord = dataframe["y_shadowing"].astype(float)
    manual_masks = {
        "left_of_CORR04_lower_left": (x_coord >= -10.5) & (x_coord < -3.44) & (y_coord < -5.9),
        "below_central_corridor_lower_middle": (x_coord >= -3.44) & (x_coord <= 2.42) & (y_coord < -5.9),
        "right_of_CORR05_lower_right_room_stack": (x_coord > 2.42) & (x_coord <= 9.5) & (y_coord < -5.9),
        "above_and_slightly_left_of_NUROPE": (x_coord >= -14.0) & (x_coord <= -6.0) & (y_coord > -3.36) & (y_coord <= 2.8),
        "NURSTA": dataframe["zone_id"].astype(str).eq("NURSTA"),
        "NUROPE": dataframe["zone_id"].astype(str).eq("NUROPE"),
        "COCPIT": dataframe["zone_id"].astype(str).eq("COCPIT"),
    }

    clusters = [
        _summarize_cluster(
            label,
            dataframe.loc[mask].copy(),
            dataframe,
            experimental_total,
            expanded_total,
            cluster_kind="manual_region_or_zone",
        )
        for label, mask in manual_masks.items()
    ]

    candidate_frame = dataframe.loc[
        dataframe["scope_classification"].isin(
            ["adjacent_to_simulated_scope", "outside_simulated_scope", "unmapped_or_unclear"]
        )
        | dataframe["zone_id"].isin(["NURSTA", "NUROPE", "COCPIT"])
    ].copy()
    labels = _dbscan_labels(candidate_frame)
    candidate_frame["auto_cluster_id"] = candidate_frame.index.map(labels)
    auto_clusters = []
    for cluster_id, group in candidate_frame.loc[candidate_frame["auto_cluster_id"] >= 0].groupby("auto_cluster_id"):
        if len(group) < 4:
            continue
        auto_clusters.append(
            _summarize_cluster(
                f"auto_cluster_{int(cluster_id)}",
                group.copy(),
                dataframe,
                experimental_total,
                expanded_total,
                cluster_kind="automatic_coordinate_cluster",
            )
        )
    auto_clusters.sort(key=lambda item: int(item.get("record_count", 0)), reverse=True)

    clusters.extend(auto_clusters[:8])
    clusters.sort(key=lambda item: int(item.get("record_count", 0)), reverse=True)

    latest_validation = {}
    if config.VALIDATION_SUMMARY_PATH.exists():
        try:
            validation_payload = json.loads(config.VALIDATION_SUMMARY_PATH.read_text())
            row = validation_payload.get("variants", {}).get("traditional_rule", {})
            latest_validation = {
                "validation_target": validation_payload.get("validation_target"),
                "nursta_simulated_share": float(row.get("zone_distribution", {}).get("NURSTA", 0.0)),
                "nurope_simulated_share": float(row.get("zone_distribution", {}).get("NUROPE", 0.0)),
                "cocpit_simulated_share": float(row.get("zone_distribution", {}).get("COCPIT", 0.0)),
                "simulated_interactions_per_hour": float(row.get("validation_fit", {}).get("simulated_f2f_per_hour", 0.0)),
            }
        except Exception:
            latest_validation = {"parse_error": "Could not parse latest validation summary."}

    scenario_evidence = _load_scenario_building_evidence()
    payload = {
        "purpose": "Spatial-forensic audit of high-density empirical F2F clusters before expanding the ED ABM.",
        "method": {
            "manual_regions": "Bounding boxes around the user-identified circled regions plus named station zones.",
            "automatic_clusters": "Small deterministic DBSCAN-like clustering on empirical coordinates for adjacent/inactive/unclear records and major stations; eps=1.75m, min_points=4.",
            "non_claim": "This audit infers affordances from empirical F2F clusters; it does not change simulator behavior.",
        },
        "record_counts": {
            "full_empirical": int(len(dataframe)),
            "care_area": experimental_total,
        },
        "plain_language_summary": (
            "The strongest missing ecology is not a single LLM/rule mechanism. Empirical F2F records show station-edge "
            "and adjacent room/threshold clusters around NURSTA, NUROPE/NEUOFF, and CORR05/CORR07. The current model now "
            "represents a ten-bed care-area setup; broader full-ED claims would still need explicit surrounding "
            "room/threshold affordances and better station co-presence evidence."
        ),
        "clusters": clusters,
        "nursta_nurope_diagnosis": {
            "most_likely_cause": "missing station dwell/co-presence and station-edge encounter behavior",
            "less_likely_primary_causes": [
                "missing geometry, because NURSTA and NUROPE exist as named polygons",
                "wrong home-station allocation alone, because nurses already have NURSTA/NUROPE home zones",
                "validation filtering artifact, because NURSTA and NUROPE are present in the care-area target",
                "missing staff agents alone, because station records are dominated by co-presence patterns that the current agents could in principle produce if routed/dwelled there",
            ],
            "latest_validation_context": latest_validation,
            "interpretation": (
                "NURSTA and NUROPE are geometrically present and configured as staff stations, but latest baseline validation "
                "still produced zero share. That points first to station dwell/co-presence and encounter opportunity, not full-ED expansion."
            ),
        },
        "staffing_and_bed_evidence": {
            "evidence_for_more_nurses": (
                "NURSTA and the lower-middle cluster contain many Nurse|Nurse interactions, which a six-agent model may under-sample "
                "when nurses are busy at beds."
            ),
            "evidence_against_more_nurses": (
                "The current model already assigns nurses to NURSTA and NUROPE, yet produces zero station share; adding nurses before "
                "repairing dwell/co-presence would confound staff-count and affordance effects."
            ),
            "evidence_for_more_beds_or_rooms": (
                "Outside-named-zone clusters nearest CORR05, CORR07, NUROPE, and NEUOFF contain substantial patient-facing interactions, "
                "consistent with missing bedside/room/threshold polygons or care bays."
            ),
            "evidence_against_more_beds_or_rooms": (
                "Additional beds should now be evaluated against the care-area validation target rather than treated as an abstract scope toggle."
            ),
        },
        "scenario_building_interpretation": {
            "scenario_building_evidence": scenario_evidence,
            "interpretation": (
                "ScenarioBuildingAris is useful functional context for ESI, flow coordinator equivalence, treatment bays, and broader ED functions. "
                "It does not provide enough precise geometry or operational staffing detail to build a full Zurich ED model by itself."
            ),
        },
        "recommendation": {
            "decision": "Use the active ten-bed care-area setup for Part 1 baseline validation; do not expand to full ED behavior yet.",
            "smallest_expansion_if_care_area_is_selected": [
                "Add explicit lower-middle/NURSTA-edge, lower-right/CORR05-edge, and NUROPE/NEUOFF-threshold affordance polygons.",
                "Add station dwell/co-presence behavior for NURSTA and NUROPE before adding staff.",
                "Only after those are validated, consider whether extra beds/care bays are required for patient-facing clusters.",
            ],
            "do_not_implement_yet": [
                "Do not add full ED scope.",
                "Do not add new agents.",
                "Do not add beds until target scope is chosen.",
                "Do not tune probabilities to compensate for missing affordances.",
            ],
            "rationale": (
                "The circled empirical clusters are best interpreted as missing affordances and co-presence ecology, not evidence that "
                "memory/generative behavior should be expanded."
            ),
        },
    }
    if write_output:
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.SPATIAL_CLUSTER_AFFORDANCE_AUDIT_JSON_PATH.write_text(json.dumps(payload, indent=2))
        _write_spatial_cluster_audit_markdown(payload)
    return payload


def load_scope_filtered_f2f_dataframe(scope: str = "care_area") -> pd.DataFrame:
    """Return empirical F2F records for a validation target."""

    environment = Environment(
        wall_path=config.WALL_POSITIONS_PATH,
        zone_path=config.ZONE_BOUNDARIES_PATH,
    )
    target = {"full": "full_empirical", "care_area": "care_area"}.get(scope, scope)
    if target not in VALIDATION_TARGETS:
        raise ValueError(f"scope must be one of {sorted(VALIDATION_TARGETS)}")
    dataframe = _annotate_validation_targets(
        _prepare_empirical_f2f_records(environment=environment),
        environment,
    )
    if target == "care_area":
        return dataframe.loc[dataframe["in_care_area"]].copy()
    return dataframe.copy()
