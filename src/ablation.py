"""Laptop-scale ablation smoke runs across model variants."""

from __future__ import annotations

import json
from contextlib import contextmanager
from collections import Counter, defaultdict
import random
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Optional

import config

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import cosine_similarity
from src.calibration import compute_distance_debug, compute_summary_statistics
from src.empirical import (
    VALIDATION_TARGETS,
    classify_zone_supergroup,
    compute_empirical_summary,
    describe_empirical_observation_window,
    load_scope_filtered_f2f_dataframe,
)
from src.environment import Environment
from src.simulation import Simulation


def _scenario_clock_metadata(
    scenario_mode: str,
    scenario_start_hour: int,
    duration_seconds: int,
    warmup_seconds: int = 0,
) -> dict:
    scenario_definition = dict(config.SCENARIO_MODES.get(scenario_mode, {}))
    profile = list(scenario_definition.get("hourly_arrival_multipliers") or [1.0] * 24)
    duration_hours = int(np.ceil(max(duration_seconds, 0) / 3600))
    warmup_hours = int(np.ceil(max(warmup_seconds, 0) / 3600))
    start_hour = int(scenario_start_hour) % 24
    end_hour = (start_hour + duration_hours) % 24
    warmup_end_hour = (start_hour + warmup_hours) % 24
    evaluation_hours = [(start_hour + hour) % 24 for hour in range(warmup_hours, duration_hours)]
    return {
        "scenario_start_hour": start_hour,
        "simulated_clock_window": f"{start_hour:02d}:00-{end_hour:02d}:00",
        "warmup_clock_window": f"{start_hour:02d}:00-{warmup_end_hour:02d}:00",
        "evaluation_clock_window": f"{warmup_end_hour:02d}:00-{end_hour:02d}:00",
        "evaluation_hour_indices": evaluation_hours,
        "evaluation_hourly_arrival_multipliers": [
            float(profile[hour % len(profile)]) if profile else 1.0
            for hour in evaluation_hours
        ],
    }


@contextmanager
def _temporary_config_updates(updates: Mapping[str, object]):
    previous = {key: getattr(config, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(config, key, value)


def _duration_label(duration_seconds: int) -> str:
    if duration_seconds <= 600:
        return "latest_debug"
    if duration_seconds == 1800:
        return "30m"
    if duration_seconds == 3600:
        return "1h"
    if duration_seconds == 14400:
        return "4h"
    if duration_seconds % 3600 == 0:
        return f"{duration_seconds // 3600}h"
    return f"{duration_seconds // 60}m"


def _jsd(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    labels = sorted(set(left) | set(right))
    if not labels:
        return 0.0
    l = np.asarray([float(left.get(label, 0.0)) for label in labels], dtype=float)
    r = np.asarray([float(right.get(label, 0.0)) for label in labels], dtype=float)
    l = l / max(l.sum(), 1e-12)
    r = r / max(r.sum(), 1e-12)
    m = 0.5 * (l + r)
    mask_l = l > 0
    mask_r = r > 0
    return float(
        0.5 * np.sum(l[mask_l] * np.log2(l[mask_l] / np.maximum(m[mask_l], 1e-12)))
        + 0.5 * np.sum(r[mask_r] * np.log2(r[mask_r] / np.maximum(m[mask_r], 1e-12)))
    )


def _normalized(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {str(key): float(value) / total for key, value in counter.items()}


def _duration_stats(interactions: list[dict]) -> dict[str, float]:
    durations = [float(event.get("duration_seconds", 0.0)) for event in interactions]
    if not durations:
        return {"mean_duration": 0.0, "median_duration": 0.0}
    return {
        "mean_duration": float(mean(durations)),
        "median_duration": float(median(durations)),
    }


def _aggregate_workflow_health(records: list[Mapping[str, object]]) -> dict:
    if not records:
        return {
            "workflow_health_status": "UNKNOWN",
            "workflow_health_reasons": ["no workflow-health records were collected"],
        }
    sum_keys = [
        "arrival_attempts",
        "patient_arrivals",
        "admitted_arrivals",
        "deferred_arrivals",
        "bed_assignments",
        "successful_escorts_to_bed",
        "placement_tasks_started",
        "placement_tasks_completed",
        "workflow_tasks_started",
        "workflow_tasks_completed",
        "doctor_tasks_started",
        "doctor_tasks_completed",
        "doctor_task_backlog_patients",
        "discharges",
        "stuck_agent_events",
        "oscillation_warnings",
        "reserved_bed_restriction_violations",
        "active_waiting_patients",
    ]
    totals = {
        key: int(sum(int(record.get(key, 0)) for record in records))
        for key in sum_keys
    }
    occupancy_values = [float(record.get("average_bed_occupancy", 0.0)) for record in records]
    ordinary_occupancy_values = [float(record.get("average_ordinary_bed_occupancy", 0.0)) for record in records]
    p95_ordinary_occupancy_values = [float(record.get("p95_ordinary_bed_occupancy", 0.0)) for record in records]
    special_occupancy_values = [float(record.get("average_special_high_acuity_bed_occupancy", 0.0)) for record in records]
    p95_special_occupancy_values = [float(record.get("p95_special_high_acuity_bed_occupancy", 0.0)) for record in records]
    all_ordinary_full_values = [float(record.get("percent_time_all_ordinary_beds_full", 0.0)) for record in records]
    ordinary_open_values = [float(record.get("percent_time_at_least_one_ordinary_bed_open", 0.0)) for record in records]
    waiting_gt_zero_values = [float(record.get("percent_time_waiting_queue_gt_zero", 0.0)) for record in records]
    waiting_zero_values = [float(record.get("percent_time_waiting_queue_zero", 0.0)) for record in records]
    mean_waiting_values = [float(record.get("mean_waiting_queue_length", 0.0)) for record in records]
    p95_waiting_values = [float(record.get("p95_waiting_queue_length", 0.0)) for record in records]
    pressure_mean_values = [float(record.get("ed_pressure_index_mean", 0.0)) for record in records]
    pressure_p95_values = [float(record.get("ed_pressure_index_p95", 0.0)) for record in records]
    mean_concurrent_esi1_values = [float(record.get("mean_concurrent_esi1", 0.0)) for record in records]
    mean_concurrent_esi1_esi2_values = [float(record.get("mean_concurrent_esi1_esi2", 0.0)) for record in records]
    ordinary_bed_open_seconds_by_bed: Counter = Counter()
    beds_used = sorted({
        int(bed)
        for record in records
        for bed in record.get("beds_used", [])
    })
    bed_assignment_counts: Counter = Counter()
    bed_assignment_esi_by_bed: dict[str, Counter] = defaultdict(Counter)
    deferred_arrivals_by_esi: Counter = Counter()
    esi_counter_keys = [
        "arrival_attempts_by_esi",
        "admitted_arrivals_by_esi",
        "deferred_or_rejected_arrivals_by_esi",
        "rejected_external_by_esi",
        "accepted_waiting_by_esi",
        "bedded_by_esi",
        "completed_by_esi",
        "active_by_esi",
        "backlog_by_esi",
        "arrivals_by_esi",
        "triaged_by_esi",
        "bed_assignment_by_esi",
        "waiting_for_bed_by_esi",
        "waiting_in_bed_by_esi",
        "low_acuity_deprioritized_by_esi",
        "first_nurse_seen_by_esi",
        "first_doctor_seen_by_esi",
        "completed_patients_by_esi",
        "completed_events_by_esi",
        "active_patients_by_esi",
        "end_of_run_backlog_by_esi",
        "active_without_first_nurse_by_esi",
        "active_without_first_doctor_by_esi",
    ]
    esi_counters: dict[str, Counter] = {key: Counter() for key in esi_counter_keys}
    esi_time_mean_keys = [
        "time_to_bed_by_esi",
        "time_to_first_nurse_by_esi",
        "time_to_first_doctor_by_esi",
    ]
    esi_time_max_keys = [
        "max_time_to_bed_by_esi",
        "max_time_to_first_nurse_by_esi",
        "max_time_to_first_doctor_by_esi",
        "oldest_waiting_patient_age_by_esi",
    ]
    for record in records:
        bed_assignment_counts.update(record.get("bed_assignment_counts", {}))
        ordinary_bed_open_seconds_by_bed.update(record.get("ordinary_bed_open_seconds_by_bed", {}))
        deferred_arrivals_by_esi.update(record.get("deferred_arrivals_by_esi", {}))
        for key in esi_counter_keys:
            esi_counters[key].update(record.get(key, {}))
        for bed_label, esi_counts in record.get("bed_assignment_esi_by_bed", {}).items():
            bed_assignment_esi_by_bed[str(bed_label)].update(esi_counts)
    status_order = {"PASS": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": 3}
    worst_status = max(
        (str(record.get("workflow_health_status", "UNKNOWN")) for record in records),
        key=lambda status: status_order.get(status, 3),
    )
    reasons = sorted({
        str(reason)
        for record in records
        for reason in record.get("workflow_health_reasons", [])
    })
    utilization_keys = {
        key
        for record in records
        for key in dict(record.get("doctor_utilization", {})).keys()
    }
    doctor_utilization = {
        key: float(mean(float(dict(record.get("doctor_utilization", {})).get(key, 0.0)) for record in records))
        for key in utilization_keys
    }
    return {
        **totals,
        "beds_used": beds_used,
        "bed_assignment_counts": dict(bed_assignment_counts),
        "bed_assignment_esi_by_bed": {
            bed_label: dict(counter)
            for bed_label, counter in bed_assignment_esi_by_bed.items()
        },
        "deferred_arrivals_by_esi": dict(deferred_arrivals_by_esi),
        **{key: dict(counter) for key, counter in esi_counters.items()},
        "arrival_attempts_total": int(sum(esi_counters["arrival_attempts_by_esi"].values())),
        "admitted_arrivals_total": int(sum(esi_counters["admitted_arrivals_by_esi"].values())),
        "deferred_or_rejected_arrivals_total": int(
            sum(esi_counters["deferred_or_rejected_arrivals_by_esi"].values())
        ),
        "rejected_external_total": int(sum(esi_counters["rejected_external_by_esi"].values())),
        "ESI1_rejected_external_count": int(esi_counters["rejected_external_by_esi"].get("ESI 1", 0)),
        "ESI2_rejected_external_count": int(esi_counters["rejected_external_by_esi"].get("ESI 2", 0)),
        "urgent_waiting_esi1_esi2_count": int(
            esi_counters["accepted_waiting_by_esi"].get("ESI 1", 0)
            + esi_counters["accepted_waiting_by_esi"].get("ESI 2", 0)
        ),
        **{
            key: {
                esi_label: float(mean(
                    float(record.get(key, {}).get(esi_label, 0.0))
                    for record in records
                    if esi_label in record.get(key, {})
                ))
                for esi_label in sorted({
                    label
                    for record in records
                    for label in record.get(key, {}).keys()
                })
            }
            for key in esi_time_mean_keys
        },
        **{
            key: {
                esi_label: float(max(
                    float(record.get(key, {}).get(esi_label, 0.0))
                    for record in records
                    if esi_label in record.get(key, {})
                ))
                for esi_label in sorted({
                    label
                    for record in records
                    for label in record.get(key, {}).keys()
                })
            }
            for key in esi_time_max_keys
        },
        "max_active_system_load": int(max(int(record.get("max_active_system_load", 0)) for record in records)),
        "reserved_bed_numbers": sorted({
            int(bed)
            for record in records
            for bed in record.get("reserved_bed_numbers", [])
        }),
        "average_bed_occupancy": float(mean(occupancy_values)) if occupancy_values else 0.0,
        "ordinary_bed_capacity": int(max(int(record.get("ordinary_bed_capacity", 0)) for record in records)),
        "average_ordinary_bed_occupancy": float(mean(ordinary_occupancy_values)) if ordinary_occupancy_values else 0.0,
        "p95_ordinary_bed_occupancy": float(mean(p95_ordinary_occupancy_values)) if p95_ordinary_occupancy_values else 0.0,
        "special_high_acuity_bed_capacity": int(max(int(record.get("special_high_acuity_bed_capacity", 0)) for record in records)),
        "average_special_high_acuity_bed_occupancy": float(mean(special_occupancy_values)) if special_occupancy_values else 0.0,
        "p95_special_high_acuity_bed_occupancy": float(mean(p95_special_occupancy_values)) if p95_special_occupancy_values else 0.0,
        "percent_time_all_ordinary_beds_full": float(mean(all_ordinary_full_values)) if all_ordinary_full_values else 0.0,
        "percent_time_at_least_one_ordinary_bed_open": float(mean(ordinary_open_values)) if ordinary_open_values else 0.0,
        "ordinary_bed_open_seconds_by_bed": dict(ordinary_bed_open_seconds_by_bed),
        "ordinary_bed_open_minutes_by_bed": {
            str(bed): float(seconds) / 60.0
            for bed, seconds in ordinary_bed_open_seconds_by_bed.items()
        },
        "percent_time_waiting_queue_gt_zero": float(mean(waiting_gt_zero_values)) if waiting_gt_zero_values else 0.0,
        "percent_time_waiting_queue_zero": float(mean(waiting_zero_values)) if waiting_zero_values else 0.0,
        "mean_waiting_queue_length": float(mean(mean_waiting_values)) if mean_waiting_values else 0.0,
        "p95_waiting_queue_length": float(mean(p95_waiting_values)) if p95_waiting_values else 0.0,
        "ed_pressure_index_mean": float(mean(pressure_mean_values)) if pressure_mean_values else 0.0,
        "ed_pressure_index_p95": float(mean(pressure_p95_values)) if pressure_p95_values else 0.0,
        "waiting_pressure_minutes": float(sum(float(record.get("waiting_pressure_minutes", 0.0)) for record in records)),
        "ordinary_beds_full_minutes": float(sum(float(record.get("ordinary_beds_full_minutes", 0.0)) for record in records)),
        "open_ordinary_bed_minutes": float(sum(float(record.get("open_ordinary_bed_minutes", 0.0)) for record in records)),
        "zero_waiting_minutes": float(sum(float(record.get("zero_waiting_minutes", 0.0)) for record in records)),
        "high_acuity_active_minutes": float(sum(float(record.get("high_acuity_active_minutes", 0.0)) for record in records)),
        "active_esi1_patient_minutes": float(sum(float(record.get("active_esi1_patient_minutes", 0.0)) for record in records)),
        "active_esi2_patient_minutes": float(sum(float(record.get("active_esi2_patient_minutes", 0.0)) for record in records)),
        "max_concurrent_esi1": int(max(int(record.get("max_concurrent_esi1", 0)) for record in records)),
        "max_concurrent_esi1_esi2": int(max(int(record.get("max_concurrent_esi1_esi2", 0)) for record in records)),
        "mean_concurrent_esi1": float(mean(mean_concurrent_esi1_values)) if mean_concurrent_esi1_values else 0.0,
        "mean_concurrent_esi1_esi2": float(mean(mean_concurrent_esi1_esi2_values)) if mean_concurrent_esi1_esi2_values else 0.0,
        "average_doctor_task_wait_seconds": float(
            mean(float(record.get("average_doctor_task_wait_seconds", 0.0)) for record in records)
        ),
        "doctor_utilization": doctor_utilization,
        "average_active_waiting_seconds": float(
            mean(float(record.get("average_active_waiting_seconds", 0.0)) for record in records)
        ),
        "max_bed_occupancy": int(max(int(record.get("max_bed_occupancy", 0)) for record in records)),
        "max_staff_stalled_seconds": int(max(int(record.get("max_staff_stalled_seconds", 0)) for record in records)),
        "workflow_health_status": worst_status,
        "workflow_health_reasons": reasons,
        "per_run": list(records),
    }


def _zone_group_for_event(event: Mapping[str, object]) -> str:
    return classify_zone_supergroup(
        str(event.get("zone_id", "Outside named zones")),
        float(event.get("x", 0.0)),
        float(event.get("y", 0.0)),
    )


def _outcome_metrics(interactions: list[dict], missed_count: int) -> dict:
    interaction_count = len(interactions)
    role_pair_counter = Counter(
        "|".join(sorted((str(event["role_1"]), str(event["role_2"]))))
        for event in interactions
    )
    topic_counter = Counter(str(event.get("topic", "unknown")) for event in interactions)
    zone_counter = Counter(str(event.get("zone_id", "Outside named zones")) for event in interactions)
    hcw_patient_count = sum(1 for event in interactions if "Patient" in {event["role_1"], event["role_2"]})
    high_acuity_count = sum(1 for event in interactions if event.get("esi_level") in {1, 2})
    zone_group_counts = Counter(_zone_group_for_event(event) for event in interactions)
    duration_stats = _duration_stats(interactions)
    denominator = max(interaction_count, 1)
    return {
        "interaction_count": interaction_count,
        "missed_opportunity_episodes": missed_count,
        "hcw_hcw_share": (interaction_count - hcw_patient_count) / denominator,
        "hcw_patient_share": hcw_patient_count / denominator,
        "patient_facing_share": sum(1 for event in interactions if event.get("patient_id") is not None) / denominator,
        "high_acuity_interaction_share": high_acuity_count / denominator,
        "mean_duration": duration_stats["mean_duration"],
        "median_duration": duration_stats["median_duration"],
        "station_or_desk_interaction_share": zone_group_counts.get("station_or_desk", 0) / denominator,
        "corridor_interaction_share": zone_group_counts.get("corridor", 0) / denominator,
        "bedside_or_patient_room_interaction_share": zone_group_counts.get("bedside_or_patient_room", 0) / denominator,
        "other_or_uncoded_interaction_share": zone_group_counts.get("other_or_uncoded", 0) / denominator,
        "outside_named_zones_interaction_share": sum(
            1 for event in interactions if str(event.get("zone_id", "Outside named zones")) == "Outside named zones"
        ) / denominator,
        "role_pair_distribution": _normalized(role_pair_counter),
        "topic_distribution": _normalized(topic_counter),
        "zone_distribution": _normalized(zone_counter),
        "zone_supergroup_distribution": _normalized(zone_group_counts),
    }


def _senior_doctor_summary(interactions: list[dict]) -> dict:
    senior_interactions = [
        event for event in interactions
        if str(event.get("interaction_type")) in {
            "senior_doctor_oversight_review",
            "senior_doctor_room_assist",
        }
    ]
    role_pairs = Counter(
        "|".join(sorted((str(event.get("role_1")), str(event.get("role_2")))))
        for event in senior_interactions
    )
    by_trigger = Counter(str(event.get("task_name", "unknown")) for event in senior_interactions)
    by_location = Counter(str(event.get("zone_id", "Outside named zones")) for event in senior_interactions)
    by_topic = Counter(str(event.get("topic", "unknown")) for event in senior_interactions)
    by_type = Counter(str(event.get("interaction_type", "unknown")) for event in senior_interactions)
    by_acuity = Counter(
        f"ESI {event.get('esi_level')}" if event.get("esi_level") is not None else "unknown"
        for event in senior_interactions
    )
    return {
        "count": len(senior_interactions),
        "by_type": dict(by_type),
        "by_trigger_task": dict(by_trigger),
        "by_location": dict(by_location),
        "by_role_pair": dict(role_pairs),
        "by_topic": dict(by_topic),
        "by_acuity": dict(by_acuity),
    }


def _interactions_in_window(interactions: Iterable[dict], start_second: int, end_second: int) -> list[dict]:
    return [
        event for event in interactions
        if start_second <= int(event.get("timestamp", event.get("timestep", 0))) < end_second
    ]


def _time_sliced_validation_summary(
    interactions: list[dict],
    *,
    duration_seconds: int,
    n_runs: int,
    slice_seconds: int = 3600,
) -> list[dict]:
    slices = []
    run_count = max(int(n_runs), 1)
    for start_second in range(0, max(duration_seconds, 0), slice_seconds):
        end_second = min(start_second + slice_seconds, duration_seconds)
        window = _interactions_in_window(interactions, start_second, end_second)
        window_hours_per_run = max((end_second - start_second) / 3600.0, 1e-12)
        outcomes = _outcome_metrics(window, missed_count=0)
        slices.append({
            "start_second": start_second,
            "end_second": end_second,
            "hour_index": int(start_second // 3600),
            "interaction_count": len(window),
            "f2f_per_hour": float(len(window) / (window_hours_per_run * run_count)),
            "hcw_hcw_share": float(outcomes["hcw_hcw_share"]),
            "patient_facing_share": float(outcomes["patient_facing_share"]),
            "station_or_desk_share": float(outcomes["station_or_desk_interaction_share"]),
            "corridor_share": float(outcomes["corridor_interaction_share"]),
            "bedside_or_patient_room_share": float(outcomes["bedside_or_patient_room_interaction_share"]),
        })
    return slices


def _matched_window_validation_summary(
    interactions_by_run: list[list[dict]],
    empirical_window: Mapping[str, object],
    *,
    duration_seconds: int,
    warmup_seconds: int,
) -> dict:
    empirical_windows = list(empirical_window.get("per_day_observation_windows", []))
    if not empirical_windows:
        return {"enabled": False, "reason": "No empirical observation windows were available."}

    empirical_rates = [float(item.get("f2f_records_per_hour", 0.0)) for item in empirical_windows]
    empirical_total_records = sum(int(item.get("f2f_record_count", 0)) for item in empirical_windows)
    empirical_total_hours = sum(float(item.get("observed_span_seconds", 0)) for item in empirical_windows) / 3600.0
    empirical_pooled_rate = empirical_total_records / max(empirical_total_hours, 1e-12)

    sampled_windows = []
    sampled_total_interactions = 0
    sampled_total_hours = 0.0
    for run_index, interactions in enumerate(interactions_by_run):
        rng = random.Random(config.RUN_BASE_SEED + run_index + 10_000)
        for window_index, empirical_item in enumerate(empirical_windows):
            window_seconds = int(empirical_item.get("observed_span_seconds", 0))
            if window_seconds <= 0:
                continue
            latest_start = max(duration_seconds - window_seconds, warmup_seconds)
            if latest_start <= warmup_seconds:
                start_second = warmup_seconds
            else:
                start_second = rng.randint(warmup_seconds, latest_start)
            end_second = min(start_second + window_seconds, duration_seconds)
            window = _interactions_in_window(interactions, start_second, end_second)
            hours = max((end_second - start_second) / 3600.0, 1e-12)
            rate = len(window) / hours
            sampled_total_interactions += len(window)
            sampled_total_hours += hours
            sampled_windows.append({
                "run_index": run_index,
                "empirical_window_index": window_index,
                "empirical_date": empirical_item.get("date"),
                "empirical_duration_seconds": window_seconds,
                "simulated_start_second": start_second,
                "simulated_end_second": end_second,
                "simulated_interactions": len(window),
                "simulated_f2f_per_hour": float(rate),
                "empirical_f2f_per_hour": float(empirical_item.get("f2f_records_per_hour", 0.0)),
            })

    simulated_rates = [float(item["simulated_f2f_per_hour"]) for item in sampled_windows]
    simulated_pooled_rate = sampled_total_interactions / max(sampled_total_hours, 1e-12)
    empirical_range = [min(empirical_rates), max(empirical_rates)] if empirical_rates else [0.0, 0.0]
    simulated_range = [min(simulated_rates), max(simulated_rates)] if simulated_rates else [0.0, 0.0]
    empirical_width = empirical_range[1] - empirical_range[0]
    simulated_width = simulated_range[1] - simulated_range[0]
    return {
        "enabled": True,
        "warmup_seconds": int(warmup_seconds),
        "empirical_window_count": len(empirical_windows),
        "sampled_window_count": len(sampled_windows),
        "pooled_empirical_f2f_per_hour": float(empirical_pooled_rate),
        "pooled_simulated_f2f_per_hour": float(simulated_pooled_rate),
        "empirical_per_window_f2f_per_hour_range": empirical_range,
        "simulated_sampled_window_f2f_per_hour_range": simulated_range,
        "empirical_per_window_f2f_per_hour_values": empirical_rates,
        "simulated_sampled_window_f2f_per_hour_values": simulated_rates,
        "simulated_distribution_much_narrower_than_empirical": bool(
            empirical_width > 0 and simulated_width < (0.5 * empirical_width)
        ),
        "interpretation": (
            "Matched-window sampling evaluates duration sensitivity and sampling variability. "
            "It should not be interpreted as requiring an average-day baseline to reproduce each "
            "empirical day's extreme F2F/hour value."
        ),
        "sampled_windows": sampled_windows,
    }


def _standard_validation_protocol_recommendation(
    *,
    full_fit: Mapping[str, object],
    warmup_fit: Optional[Mapping[str, object]],
    time_slices: list[Mapping[str, object]],
    matched_summary: Optional[Mapping[str, object]],
    warmup_seconds: int,
) -> dict:
    rates = [float(item.get("f2f_per_hour", 0.0)) for item in time_slices]
    first_hour_rate = rates[0] if rates else 0.0
    later_rates = rates[1:] if len(rates) > 1 else []
    later_mean = float(mean(later_rates)) if later_rates else first_hour_rate
    startup_inflated = bool(later_rates and first_hour_rate > later_mean * 1.15)
    full_error = abs(
        float(full_fit.get("simulated_f2f_per_hour", 0.0))
        - float(full_fit.get("empirical_f2f_per_hour", 0.0))
    )
    if warmup_fit is not None:
        warmup_error = abs(
            float(warmup_fit.get("simulated_f2f_per_hour", 0.0))
            - float(warmup_fit.get("empirical_f2f_per_hour", 0.0))
        )
    else:
        warmup_error = full_error
    if warmup_fit is not None and startup_inflated:
        recommended = "12-hour run with 2-hour warm-up exclusion"
        rationale = (
            "Hourly slices indicate startup transient behavior; post-warm-up metrics "
            "are the cleaner baseline for steady-state care-area validation."
        )
    elif matched_summary and matched_summary.get("enabled"):
        recommended = "matched-window pooled validation"
        rationale = (
            "Matched empirical windows are available and should be used as a sensitivity "
            "analysis for observation-window duration effects."
        )
    elif full_error <= warmup_error:
        recommended = "12-hour full-run validation"
        rationale = "Full-run rate is at least as close to empirical as the post-warm-up rate in this run."
    else:
        recommended = "4-hour fixed validation"
        rationale = "No warm-up evidence was available; keep the previous 4-hour baseline until longer-window checks are rerun."
    return {
        "recommended_protocol": recommended,
        "rationale": rationale,
        "first_hour_f2f_per_hour": float(first_hour_rate),
        "later_hour_mean_f2f_per_hour": float(later_mean),
        "startup_first_hour_inflated": startup_inflated,
        "full_run_rate_abs_error": float(full_error),
        "warmup_rate_abs_error": float(warmup_error),
        "matched_window_note": (
            matched_summary.get("interpretation")
            if matched_summary and matched_summary.get("enabled")
            else "Matched-window sampling was not enabled."
        ),
    }


def _delta_metrics(current: Mapping[str, object], reference: Mapping[str, object]) -> dict[str, float]:
    keys = [
        "interaction_count",
        "missed_opportunity_episodes",
        "hcw_hcw_share",
        "hcw_patient_share",
        "patient_facing_share",
        "high_acuity_interaction_share",
        "mean_duration",
        "median_duration",
        "station_or_desk_interaction_share",
        "corridor_interaction_share",
        "bedside_or_patient_room_interaction_share",
        "other_or_uncoded_interaction_share",
        "outside_named_zones_interaction_share",
    ]
    return {
        f"{key}_delta": float(current.get(key, 0.0)) - float(reference.get(key, 0.0))
        for key in keys
    }


def _bed_care_flow_summary(simulation: Simulation) -> dict:
    """Summarize task progression by bed for validation sanity checks."""

    patient_bed: dict[int, int] = {}
    for event in simulation.workflow_event_log:
        if event.get("event_type") != "bed_assigned":
            continue
        patient_id = event.get("patient_id")
        new_state = str(event.get("new_state", ""))
        if patient_id is None or not new_state.startswith("bed_"):
            continue
        try:
            bed_number = int(new_state.split("_")[1])
        except (IndexError, ValueError):
            continue
        patient_bed[int(patient_id)] = bed_number

    task_event_counts: dict[str, Counter] = defaultdict(Counter)
    doctor_task_counts: dict[str, Counter] = defaultdict(Counter)
    nurse_ids_by_bed: dict[str, set[int]] = defaultdict(set)
    doctor_ids_by_bed: dict[str, set[int]] = defaultdict(set)

    for event in simulation.workflow_event_log:
        patient_id = event.get("patient_id")
        if patient_id is None:
            continue
        bed_number = patient_bed.get(int(patient_id))
        if bed_number is None:
            continue
        bed_label = f"bed_{bed_number}"
        key = f"{event.get('task_name')}|{event.get('event_type')}"
        task_event_counts[bed_label][key] += 1
        if event.get("agent_role") == "Nurse" and event.get("agent_id") is not None:
            nurse_ids_by_bed[bed_label].add(int(event["agent_id"]))
        if event.get("agent_role") == "Doctor" and event.get("agent_id") is not None:
            doctor_ids_by_bed[bed_label].add(int(event["agent_id"]))
            doctor_task_counts[bed_label][str(event.get("task_name"))] += 1

    discharges_by_bed: Counter = Counter()
    for patient in simulation.completed_patients:
        if patient.bed_index is not None:
            discharges_by_bed[f"bed_{patient.bed_index + 1}"] += 1

    assigned_nurse_by_bed = {}
    for bed_index in range(len(config.BED_POSITIONS)):
        nurse_index = simulation._nurse_index_for_bed(bed_index)
        if nurse_index is None or nurse_index >= len(simulation.nurses):
            assigned_nurse_by_bed[f"bed_{bed_index + 1}"] = None
            continue
        nurse = simulation.nurses[nurse_index]
        assigned_nurse_by_bed[f"bed_{bed_index + 1}"] = {
            "nurse_index": nurse_index,
            "nurse_id": nurse.gid,
            "home_zone_id": nurse.home_zone_id,
            "covered_bed_indices": [index + 1 for index in nurse.covered_bed_indices],
        }

    return {
        f"bed_{bed_index + 1}": {
            "assigned_nurse": assigned_nurse_by_bed.get(f"bed_{bed_index + 1}"),
            "task_event_counts": dict(task_event_counts.get(f"bed_{bed_index + 1}", Counter())),
            "doctor_task_counts": dict(doctor_task_counts.get(f"bed_{bed_index + 1}", Counter())),
            "nurse_ids": sorted(nurse_ids_by_bed.get(f"bed_{bed_index + 1}", set())),
            "doctor_ids": sorted(doctor_ids_by_bed.get(f"bed_{bed_index + 1}", set())),
            "discharges": int(discharges_by_bed.get(f"bed_{bed_index + 1}", 0)),
        }
        for bed_index in range(len(config.BED_POSITIONS))
    }


def _interaction_source_label(event: Mapping[str, object]) -> str:
    interaction_type = str(event.get("interaction_type", "unknown"))
    roles = {str(event.get("role_1")), str(event.get("role_2"))}
    reason = str(event.get("reason_for_interaction", ""))
    if reason == "placement_walk_patient_guidance":
        return "patient_escort_corridor"
    if "Patient" in roles:
        return "patient_facing"
    if interaction_type == "task_transition_update":
        return "task_transition_update"
    if interaction_type in {
        "bedside_cotask_update",
        "nurse_doctor_bedside_coordination",
        "nurse_nurse_bedside_assist",
        "doctor_doctor_bedside_review",
    }:
        return "bedside_cotask"
    if event.get("staff_interaction_intent_id"):
        return "opportunistic_perception_pending_intent"
    if interaction_type == "opportunistic_corridor":
        return "corridor_co_presence"
    if interaction_type == "opportunistic_station":
        if reason == "coordination_hub_check" or "CoordinationNurse" in roles:
            return "coordination_nurse_interaction"
        return "station_co_presence"
    if interaction_type == "doctor_doctor_coreview":
        return "doctor_doctor_coreview"
    if interaction_type == "nurse_doctor_handoff":
        return "nurse_doctor_handoff"
    if interaction_type in {"placement_nurse_alert", "staff_status_update"}:
        return "coordination_nurse_interaction" if "CoordinationNurse" in roles else "station_co_presence"
    return interaction_type


def _interaction_source_audit(
    interactions: Iterable[Mapping[str, object]],
    missed_opportunities: Iterable[Mapping[str, object]],
) -> dict[str, dict]:
    grouped: dict[str, dict] = defaultdict(
        lambda: {
            "logged_count": 0,
            "blocked_not_close_count": 0,
            "max_logged_distance": 0.0,
            "zone_distribution": Counter(),
            "role_pair_distribution": Counter(),
        }
    )
    for event in interactions:
        label = _interaction_source_label(event)
        row = grouped[label]
        row["logged_count"] += 1
        row["max_logged_distance"] = max(
            float(row["max_logged_distance"]),
            float(event.get("final_logged_distance_m", event.get("initiation_distance_m", 0.0)) or 0.0),
        )
        row["zone_distribution"].update([str(event.get("zone_id", "unknown"))])
        row["role_pair_distribution"].update([
            "|".join(sorted((str(event.get("role_1", "unknown")), str(event.get("role_2", "unknown")))))
        ])

    for event in missed_opportunities:
        reason = str(event.get("reason", ""))
        perception_state = event.get("perception_state", {})
        if not isinstance(perception_state, Mapping):
            perception_state = {}
        if "not_close" not in reason and not bool(perception_state.get("blocked_validation_counted_log", False)):
            continue
        interaction_type = str(perception_state.get("interaction_type", event.get("interaction_type", "unknown")))
        label = _interaction_source_label({"interaction_type": interaction_type})
        grouped[label]["blocked_not_close_count"] += 1

    return {
        label: {
            "logged_count": int(row["logged_count"]),
            "blocked_not_close_count": int(row["blocked_not_close_count"]),
            "max_logged_distance": float(row["max_logged_distance"]),
            "zone_distribution": dict(row["zone_distribution"]),
            "role_pair_distribution": dict(row["role_pair_distribution"]),
        }
        for label, row in sorted(grouped.items())
    }


def _static_visibility_diagnostics() -> dict[str, object]:
    diagnostics: dict[str, object] = {"fov_degrees_used": float(config.FOV_DEGREES)}
    baseline_walls = tuple(Simulation(random_seed=config.RANDOM_SEED, condition_name="baseline").condition_manager.effective_walls)

    def blockers_for(simulation: Simulation, start: tuple[float, float], end: tuple[float, float]) -> list[dict[str, object]]:
        blockers = []
        for index, wall in enumerate(simulation.environment.walls):
            if not simulation.perception._segments_intersect(start, end, wall[0], wall[1]):
                continue
            blockers.append({
                "wall_index": int(index),
                "start": [float(wall[0][0]), float(wall[0][1])],
                "end": [float(wall[1][0]), float(wall[1][1])],
                "transparent": bool(index in simulation.condition_manager.transparent_wall_indices),
                "removed": bool(index in simulation.condition_manager.removed_wall_indices),
            })
        return blockers

    cockpit_region_1_point = (
        (float(config.COCPIT_WORKSTATION_REGION_1["x_min"]) + float(config.COCPIT_WORKSTATION_REGION_1["x_max"])) / 2.0,
        float(config.COCPIT_WORKSTATION_REGION_1["y"]),
    )
    cockpit_region_2_point = (
        (float(config.COCPIT_WORKSTATION_REGION_2["x_min"]) + float(config.COCPIT_WORKSTATION_REGION_2["x_max"])) / 2.0,
        float(config.COCPIT_WORKSTATION_REGION_2["y"]),
    )

    for condition_name in ("baseline", "cockpit_only", "both"):
        simulation = Simulation(
            random_seed=config.RANDOM_SEED,
            condition_name=condition_name,
            interaction_backend=config.INTERACTION_BACKEND,
            persona_source=config.PERSONA_SOURCE,
            model_variant=config.MODEL_VARIANT,
            enable_senior_doctor_oversight=False,
        )
        nurope_point = simulation.condition_manager.station_attractor("NUROPE")
        corr03_point = simulation.condition_manager.zone_centroid("CORR03")
        for region_label, cockpit_point in {
            "region_1": cockpit_region_1_point,
            "region_2": cockpit_region_2_point,
        }.items():
            diagnostics[f"cocpit_{region_label}_to_nurope_visibility_{condition_name}"] = bool(
                simulation.perception._has_visibility(cockpit_point, nurope_point)
            )
            diagnostics[f"nurope_to_cocpit_{region_label}_visibility_{condition_name}"] = bool(
                simulation.perception._has_visibility(nurope_point, cockpit_point)
            )
            diagnostics[f"cocpit_{region_label}_to_nurope_blockers_{condition_name}"] = blockers_for(
                simulation,
                cockpit_point,
                nurope_point,
            )
            diagnostics[f"nurope_to_cocpit_{region_label}_blockers_{condition_name}"] = blockers_for(
                simulation,
                nurope_point,
                cockpit_point,
            )
            if region_label == "region_2":
                diagnostics[f"cocpit_to_nurope_visibility_{condition_name}"] = diagnostics[
                    f"cocpit_{region_label}_to_nurope_visibility_{condition_name}"
                ]
                diagnostics[f"nurope_to_cocpit_visibility_{condition_name}"] = diagnostics[
                    f"nurope_to_cocpit_{region_label}_visibility_{condition_name}"
                ]
        if condition_name in {"cockpit_only", "both"}:
            expected_transparent = len(config.CONDITIONS[condition_name].get("visibility_transparent_segments", []))
            diagnostics[f"transparent_cocpit_wall_segment_count_{condition_name}"] = int(
                len(simulation.condition_manager.transparent_wall_indices)
            )
            diagnostics[f"transparent_cocpit_wall_segments_verified_{condition_name}"] = bool(
                len(simulation.condition_manager.transparent_wall_indices) == expected_transparent
                and tuple(simulation.condition_manager.effective_walls) == baseline_walls
            )
            if condition_name == "cockpit_only":
                diagnostics["transparent_cocpit_wall_segment_count"] = diagnostics[
                    "transparent_cocpit_wall_segment_count_cockpit_only"
                ]
                diagnostics["transparent_cocpit_wall_segments_verified"] = diagnostics[
                    "transparent_cocpit_wall_segments_verified_cockpit_only"
                ]
        heading_checks = {}
        agent = simulation.coordination_nurse
        original_position = agent.position
        original_heading = agent.heading
        agent.position = cockpit_region_2_point
        for label, heading in {
            "east": 0.0,
            "north": np.pi / 2.0,
            "west": np.pi,
            "south": -np.pi / 2.0,
        }.items():
            agent.heading = float(heading)
            heading_checks[label] = bool(
                simulation.perception._has_visibility(cockpit_region_2_point, corr03_point)
                and simulation.perception._within_fov(agent, corr03_point)
            )
        agent.position = original_position
        agent.heading = original_heading
        diagnostics[f"cocpit_to_corr03_visibility_by_heading_{condition_name}"] = heading_checks
    return diagnostics


def _aggregate_bed_care_flow(records: list[Mapping[str, object]]) -> dict:
    aggregate = {}
    for bed_number in range(1, len(config.BED_POSITIONS) + 1):
        bed_label = f"bed_{bed_number}"
        task_counts: Counter = Counter()
        doctor_task_counts: Counter = Counter()
        nurse_ids: set[int] = set()
        doctor_ids: set[int] = set()
        discharges = 0
        assigned_nurse = None
        for record in records:
            bed_record = record.get(bed_label, {})
            task_counts.update(bed_record.get("task_event_counts", {}))
            doctor_task_counts.update(bed_record.get("doctor_task_counts", {}))
            nurse_ids.update(int(value) for value in bed_record.get("nurse_ids", []))
            doctor_ids.update(int(value) for value in bed_record.get("doctor_ids", []))
            discharges += int(bed_record.get("discharges", 0))
            assigned_nurse = assigned_nurse or bed_record.get("assigned_nurse")
        aggregate[bed_label] = {
            "assigned_nurse": assigned_nurse,
            "task_event_counts": dict(task_counts),
            "doctor_task_counts": dict(doctor_task_counts),
            "nurse_ids": sorted(nurse_ids),
            "doctor_ids": sorted(doctor_ids),
            "discharges": discharges,
        }
    return aggregate


def _validation_metric_audit(empirical: Mapping[str, object]) -> dict:
    weights = dict(config.ABC_DISTANCE_WEIGHTS)
    return {
        "distance_components": {
            "kde_component": {
                "weight": weights["kde_grid"],
                "empirical_target": "KDE over empirical F2F x/y shadowing points",
                "simulated_target": "KDE over simulated interaction x/y points",
            },
            "zone_component": {
                "weight": weights["zone_histogram"],
                "empirical_target": "zone histogram inferred from empirical F2F locations",
                "simulated_target": "zone histogram of simulated interactions",
            },
            "role_pair_component": {
                "weight": weights["role_pair_matrix"],
                "empirical_target": "empirical F2F role-pair distribution",
                "simulated_target": "simulated interaction role-pair distribution",
            },
            "topic_component": {
                "weight": weights["topic_by_zone"],
                "empirical_target": "empirical inferred topic-by-zone distribution from F2F records",
                "simulated_target": "simulated topic-by-zone distribution",
            },
            "duration_component": {
                "weight": weights["duration_quantiles"],
                "empirical_target": "empirical duration quantiles by role pair",
                "simulated_target": "simulated duration quantiles by role pair",
            },
        },
        "formula": "weighted_total = 1.5*kde + 1.0*zone + 1.0*role_pair + 1.0*topic + 0.5*duration",
        "workflow_events_excluded": True,
        "uses_interaction_log_only": True,
        "empirical_observation_window_affects_distance": False,
        "diagnostic_mode_changes_rates": {
            "enabled": bool(config.ABLATION_DIAGNOSTIC_MODE),
            "arrival_multiplier": float(config.DIAGNOSTIC_ARRIVAL_MULTIPLIER),
            "interaction_probability_multiplier": float(config.DIAGNOSTIC_INTERACTION_PROBABILITY_MULTIPLIER),
        },
        "composite_includes_interaction_count_error": False,
        "composite_includes_high_acuity_behavior": False,
        "composite_includes_missed_opportunities": False,
        "composite_includes_hcw_vs_patient_proportions": False,
        "empirical_topic_labels": sorted(empirical.get("topic_distribution", {}).keys()),
        "empirical_topic_target_nonempty": bool(empirical.get("topic_distribution")),
        "empirical_topic_by_zone_nonempty": bool(empirical.get("topic_by_zone")),
    }


def _validation_target_metadata(validation_target: str, empirical: Mapping[str, object], empirical_window: Mapping[str, object]) -> dict:
    fair_labels = {
        "care_area": {
            "fair_for_current_model_scope": True,
            "supports_claims": "care-area baseline claims for the active model",
            "interpretation": "This target uses the full mapped empirical F2F care-area shadowing data for the active care-area ABM.",
        },
        "full_empirical": {
            "fair_for_current_model_scope": True,
            "supports_claims": "same full mapped empirical F2F source retained as a compatibility label",
            "interpretation": "This compatibility target is equivalent to care_area in the current active model.",
        },
    }
    frame = load_scope_filtered_f2f_dataframe(validation_target)
    outside_frame = frame.loc[frame["zone_id"] == "Outside named zones"]
    return {
        "validation_target": validation_target,
        "empirical_record_count": int(empirical.get("raw_f2f_count", len(frame))),
        "empirical_f2f_rate": float(empirical_window.get("f2f_records_per_hour", 0.0)),
        "zone_distribution": empirical.get("zone_histogram", {}),
        "role_pair_distribution": empirical.get("role_pair_matrix", {}),
        "zone_supergroup_distribution": empirical.get("zone_supergroup_distribution", {}),
        "outside_named_zone_candidate_distribution": _normalized(Counter(outside_frame["outside_named_zone_candidate"])),
        **fair_labels[validation_target],
    }


def _display_variant_label(variant: str) -> str:
    return "baseline_rule_model" if variant == "traditional_rule" else variant


def _gate_metric(value: Optional[float], warn: Optional[float], fail: Optional[float], *, lower_is_better: bool = True) -> dict:
    if value is None or warn is None or fail is None or not np.isfinite(float(value)):
        return {"status": "UNKNOWN", "value": value, "warn": warn, "fail": fail}
    numeric = float(value)
    if lower_is_better:
        status = "PASS" if numeric <= warn else "WARN" if numeric <= fail else "FAIL"
    else:
        status = "PASS" if numeric >= warn else "WARN" if numeric >= fail else "FAIL"
    return {"status": status, "value": numeric, "warn": warn, "fail": fail}


def _empirical_share_targets(empirical: Mapping[str, object]) -> dict[str, float]:
    role_pairs = empirical.get("role_pair_matrix", {})
    hcw_patient_share = sum(
        float(value)
        for key, value in role_pairs.items()
        if "Patient" in str(key).split("|")
    )
    zone_supergroups = empirical.get("zone_supergroup_distribution", {})
    return {
        "hcw_patient_share": hcw_patient_share,
        "hcw_hcw_share": max(1.0 - hcw_patient_share, 0.0),
        "patient_facing_share": hcw_patient_share,
        "station_or_desk_share": float(zone_supergroups.get("station_or_desk", 0.0)),
        "corridor_share": float(zone_supergroups.get("corridor", 0.0)),
        "bedside_or_patient_room_share": float(zone_supergroups.get("bedside_or_patient_room", 0.0)),
        "other_or_uncoded_share": float(zone_supergroups.get("other_or_uncoded", 0.0)),
    }


def _validation_fit_for_row(
    row: Mapping[str, object],
    empirical: Mapping[str, object],
    empirical_window: Mapping[str, object],
) -> dict:
    thresholds = config.VALIDATION_THRESHOLDS
    evaluated_seconds = float(row.get("evaluation_duration_seconds", row.get("duration_seconds", 0)))
    duration_hours = max(evaluated_seconds / 3600.0, 1e-12)
    empirical_rate = float(empirical_window.get("f2f_records_per_hour", 0.0))
    expected_count = empirical_rate * duration_hours * max(int(row.get("n_runs", 1)), 1)
    interaction_count = float(row.get("interaction_count", 0.0))
    interaction_count_relative_error = (
        abs(interaction_count - expected_count) / max(expected_count, 1e-12)
        if expected_count > 0
        else None
    )
    simulated_rate = interaction_count / max(duration_hours * max(int(row.get("n_runs", 1)), 1), 1e-12)
    f2f_rate_relative_error = (
        abs(simulated_rate - empirical_rate) / max(empirical_rate, 1e-12)
        if empirical_rate > 0
        else None
    )
    components = row.get("distance_components", {})
    outcomes = row.get("outcome_metrics", {})
    empirical_shares = _empirical_share_targets(empirical)
    hcw_patient_share_error = abs(float(outcomes.get("hcw_patient_share", 0.0)) - empirical_shares["hcw_patient_share"])
    hcw_hcw_share_error = abs(float(outcomes.get("hcw_hcw_share", 0.0)) - empirical_shares["hcw_hcw_share"])
    patient_facing_share_error = abs(float(outcomes.get("patient_facing_share", 0.0)) - empirical_shares["patient_facing_share"])
    station_share_error = abs(float(outcomes.get("station_or_desk_interaction_share", 0.0)) - empirical_shares["station_or_desk_share"])
    corridor_share_error = abs(float(outcomes.get("corridor_interaction_share", 0.0)) - empirical_shares["corridor_share"])
    other_or_uncoded_share = float(outcomes.get("other_or_uncoded_interaction_share", 0.0))
    topic_jsd = float(row.get("topic_divergence", 0.0))
    metrics = {
        "interaction_count_relative_error": _gate_metric(
            interaction_count_relative_error,
            thresholds["interaction_count_relative_error_warn"],
            thresholds["interaction_count_relative_error_fail"],
        ),
        "f2f_rate_relative_error": _gate_metric(
            f2f_rate_relative_error,
            thresholds["f2f_rate_relative_error_warn"],
            thresholds["f2f_rate_relative_error_fail"],
        ),
        "kde_distance": _gate_metric(
            float(components.get("kde_component", 0.0)),
            thresholds["kde_distance_warn"],
            thresholds["kde_distance_fail"],
        ),
        "zone_jsd": _gate_metric(
            float(components.get("zone_component", 0.0)),
            thresholds["zone_jsd_warn"],
            thresholds["zone_jsd_fail"],
        ),
        "role_pair_jsd": _gate_metric(
            float(components.get("role_pair_component", 0.0)),
            thresholds["role_pair_jsd_warn"],
            thresholds["role_pair_jsd_fail"],
        ),
        "topic_jsd": _gate_metric(
            topic_jsd,
            thresholds["topic_jsd_warn"],
            thresholds["topic_jsd_fail"],
        ),
        "topic_by_zone_jsd": _gate_metric(
            float(components.get("topic_component", 0.0)),
            thresholds["topic_by_zone_jsd_warn"],
            thresholds["topic_by_zone_jsd_fail"],
        ),
        "duration_error": _gate_metric(
            float(components.get("duration_component", 0.0)),
            thresholds["duration_error_warn"],
            thresholds["duration_error_fail"],
        ),
        "hcw_patient_share_abs_error": _gate_metric(
            hcw_patient_share_error,
            thresholds["hcw_patient_share_abs_error_warn"],
            thresholds["hcw_patient_share_abs_error_fail"],
        ),
        "hcw_hcw_share_abs_error": _gate_metric(
            hcw_hcw_share_error,
            thresholds["hcw_hcw_share_abs_error_warn"],
            thresholds["hcw_hcw_share_abs_error_fail"],
        ),
        "patient_facing_share_abs_error": _gate_metric(
            patient_facing_share_error,
            thresholds["patient_facing_share_abs_error_warn"],
            thresholds["patient_facing_share_abs_error_fail"],
        ),
        "station_share_abs_error": _gate_metric(
            station_share_error,
            thresholds["station_share_abs_error_warn"],
            thresholds["station_share_abs_error_fail"],
        ),
        "corridor_share_abs_error": _gate_metric(
            corridor_share_error,
            thresholds["corridor_share_abs_error_warn"],
            thresholds["corridor_share_abs_error_fail"],
        ),
        "outside_or_uncoded_share": _gate_metric(
            other_or_uncoded_share,
            thresholds["outside_or_uncoded_share_warn"],
            thresholds["outside_or_uncoded_share_fail"],
        ),
        "high_acuity_share_abs_error": _gate_metric(
            None,
            thresholds["high_acuity_share_abs_error_warn"],
            thresholds["high_acuity_share_abs_error_fail"],
        ),
    }
    interpretations = {
        "interaction_count_relative_error": "Compares simulated interaction count with the empirical F2F rate scaled to the run duration.",
        "f2f_rate_relative_error": "Compares simulated interactions/hour with empirical observed interactions/hour.",
        "kde_distance": "Spatial KDE mismatch; lower is better, and PASS is only provisional tolerance, not proof of good spatial fit.",
        "zone_jsd": "Detailed zone distribution divergence.",
        "role_pair_jsd": "Role-pair distribution divergence.",
        "topic_jsd": "Global topic distribution divergence.",
        "topic_by_zone_jsd": "Topic-by-zone distribution divergence.",
        "duration_error": "Role-pair duration quantile error.",
        "hcw_patient_share_abs_error": "Absolute error in HCW-patient share; large values indicate patient-facing inflation.",
        "hcw_hcw_share_abs_error": "Absolute error in HCW-HCW share; large values indicate an incorrect coordination ecology.",
        "patient_facing_share_abs_error": "Absolute error in patient-facing share.",
        "station_share_abs_error": "Absolute error in station/desk interaction share.",
        "corridor_share_abs_error": "Absolute error in corridor interaction share.",
        "outside_or_uncoded_share": "Warns when simulated interactions remain too concentrated in uncoded or room-like locations.",
        "high_acuity_share_abs_error": "UNKNOWN until empirical acuity-coded F2F targets exist.",
    }
    for metric_name, payload in metrics.items():
        payload["interpretation"] = interpretations.get(metric_name, "Provisional validation-readiness gate.")
    status_counts = Counter(metric["status"] for metric in metrics.values())
    unknown_metrics = [
        metric_name
        for metric_name, payload in metrics.items()
        if payload.get("status") == "UNKNOWN"
    ]
    if status_counts["FAIL"] > 0:
        overall = "FAIL"
    elif status_counts["WARN"] > 0:
        overall = "WARN"
    else:
        overall = "PASS"
    if status_counts["FAIL"] > 0:
        status_reason = "One or more calibrated validation gates failed."
    elif status_counts["WARN"] > 0:
        status_reason = "One or more calibrated validation gates are borderline."
    elif unknown_metrics:
        status_reason = (
            "All calibrated validation gates passed; unknown metrics are excluded "
            "from the overall gate because no empirical target is available."
        )
    else:
        status_reason = "All calibrated validation gates passed."
    return {
        "overall_status": overall,
        "known_metric_status": overall,
        "evidence_completeness_status": "WARN" if unknown_metrics else "PASS",
        "unknown_metrics": unknown_metrics,
        "status_reason": status_reason,
        "expected_interactions_for_duration": expected_count,
        "simulated_interactions": interaction_count,
        "empirical_f2f_per_hour": empirical_rate,
        "simulated_f2f_per_hour": simulated_rate,
        "empirical_hcw_patient_share": empirical_shares["hcw_patient_share"],
        "simulated_hcw_patient_share": float(outcomes.get("hcw_patient_share", 0.0)),
        "empirical_hcw_hcw_share": empirical_shares["hcw_hcw_share"],
        "simulated_hcw_hcw_share": float(outcomes.get("hcw_hcw_share", 0.0)),
        "empirical_patient_facing_share": empirical_shares["patient_facing_share"],
        "simulated_patient_facing_share": float(outcomes.get("patient_facing_share", 0.0)),
        "empirical_station_or_desk_share": empirical_shares["station_or_desk_share"],
        "simulated_station_or_desk_share": float(outcomes.get("station_or_desk_interaction_share", 0.0)),
        "empirical_corridor_share": empirical_shares["corridor_share"],
        "simulated_corridor_share": float(outcomes.get("corridor_interaction_share", 0.0)),
        "simulated_other_or_uncoded_share": other_or_uncoded_share,
        "metrics": metrics,
    }


def _variant_interpretation(variant: str, row: Mapping[str, object], traditional_row: Mapping[str, object]) -> dict:
    diagnostics = row.get("diagnostics", {})
    mechanism_active = False
    if variant == "traditional_rule":
        mechanism_active = True
    elif variant == "perception_rule":
        mechanism_active = any(int(diagnostics.get(key, 0)) > 0 for key in ("rejected_by_visibility", "rejected_fov_hard", "accepted_shared_station_awareness", "accepted_patient_facing"))
    elif variant == "memory_rule":
        mechanism_active = int(diagnostics.get("memory_effects_applied", 0)) > 0
    elif variant == "generative_interaction":
        mechanism_active = int(diagnostics.get("structured_fallback_calls", 0)) > 0
    outcome_signals = int(row.get("outcome_difference_signal_count", 0))
    fit_delta = float(row.get("distance_to_empirical", 0.0)) - float(traditional_row.get("distance_to_empirical", 0.0))
    if abs(fit_delta) < max(float(traditional_row.get("distance_std", 0.0)), 0.05):
        fit_improved = "unclear"
    else:
        fit_improved = fit_delta < 0
    if variant == "traditional_rule":
        interpretation = "Task/movement baseline; useful as a conservative comparator."
    elif variant == "perception_rule":
        interpretation = "Perception gates alter eligible encounters; large global count shifts are not required for this to be useful."
    elif variant == "memory_rule":
        interpretation = "Memory affects salience/topic/duration and occasional decisions; global outcomes may remain workflow-dominated."
    else:
        interpretation = "Structured generative fallback changes bounded interaction interpretation; this is not open-ended LLM emergence."
    return {
        "mechanism_active": mechanism_active,
        "outcome_difference_detected": "partial" if outcome_signals > 0 else False,
        "empirical_fit_improved_vs_traditional": fit_improved,
        "interpretation": interpretation,
    }


def _select_baseline_recommendation(summary: Mapping[str, Mapping[str, object]]) -> dict:
    candidates = []
    for variant, payload in summary.items():
        if not isinstance(payload, Mapping):
            continue
        validation_fit = payload.get("validation_fit", {})
        status = str(validation_fit.get("overall_status", "UNKNOWN"))
        evidence_status = str(validation_fit.get("evidence_completeness_status", "UNKNOWN"))
        distance = float(payload.get("distance_to_empirical", 999.0))
        stable_bonus = 0.0
        if payload.get("ensemble_summary"):
            stable_bonus = -0.05
        penalty = {"PASS": 0.0, "WARN": 1.0, "FAIL": 2.0, "UNKNOWN": 3.0}.get(status, 3.0)
        candidates.append((penalty + distance + stable_bonus, variant, status, evidence_status, distance))
    if not candidates:
        return {
            "recommended_model": None,
            "status": "none_ready",
            "rationale": "No validation candidates were available.",
        }
    candidates.sort()
    _, variant, status, evidence_status, distance = candidates[0]
    if status == "FAIL":
        return {
            "recommended_model": None,
            "status": "none_ready",
            "rationale": "The baseline does not pass provisional validation gates; do not proceed to Part 2 yet.",
            "closest_model": config.BASELINE_MODEL_ID,
            "closest_model_distance": distance,
        }
    rationale = (
        "Use the single rule-based perception baseline for Part 1 and Part 2 if workflow health, "
        "Bed 8 checks, empirical fit, and perception-mechanism diagnostics pass."
    )
    return {
        "recommended_model": config.BASELINE_MODEL_ID,
        "status": status,
        "evidence_completeness_status": evidence_status,
        "rationale": rationale,
        "distance_to_empirical": distance,
    }


def _update_validation_target_comparison(payload: Mapping[str, object]) -> None:
    """Merge the latest validation-candidate run into the compact target comparison."""

    if int(payload.get("duration_seconds", 0)) < 3600 or int(payload.get("n_runs", 0)) < 10:
        return
    target = str(payload.get("validation_target", "full_empirical"))
    existing = {}
    if config.VALIDATION_TARGET_COMPARISON_PATH.exists():
        try:
            existing = json.loads(config.VALIDATION_TARGET_COMPARISON_PATH.read_text())
        except json.JSONDecodeError:
            existing = {}
    raw_targets = existing.get("targets", {}) if isinstance(existing, Mapping) else {}
    current_duration = payload.get("duration_seconds")
    current_n_runs = payload.get("n_runs")
    targets = {
        key: value
        for key, value in raw_targets.items()
        if value.get("duration_seconds") == current_duration
        and value.get("n_runs") == current_n_runs
    }
    variants = payload.get("variants", {})
    compact_variants = {}
    for variant, row in variants.items():
        fit = row.get("validation_fit", {}) if isinstance(row, Mapping) else {}
        compact_variants[variant] = {
            "display_label": _display_variant_label(variant),
            "overall_status": fit.get("overall_status", "UNKNOWN"),
            "distance_to_empirical": row.get("distance_to_empirical"),
            "interaction_count": row.get("interaction_count"),
            "simulated_f2f_per_hour": fit.get("simulated_f2f_per_hour"),
            "empirical_f2f_per_hour": fit.get("empirical_f2f_per_hour"),
            "simulated_hcw_hcw_share": fit.get("simulated_hcw_hcw_share"),
            "empirical_hcw_hcw_share": fit.get("empirical_hcw_hcw_share"),
            "simulated_patient_facing_share": fit.get("simulated_patient_facing_share"),
            "empirical_patient_facing_share": fit.get("empirical_patient_facing_share"),
        }
    targets[target] = {
        "run_purpose": payload.get("run_purpose"),
        "selected_variants": payload.get("selected_variants", []),
        "validation_target_metadata": payload.get("validation_target_metadata", {}),
        "duration_seconds": current_duration,
        "n_runs": current_n_runs,
        "diagnostic_mode": payload.get("diagnostic_mode"),
        "empirical_record_count": payload.get("validation_target_metadata", {}).get("empirical_record_count"),
        "empirical_f2f_rate": payload.get("validation_target_metadata", {}).get("empirical_f2f_rate"),
        "baseline_recommendation": payload.get("baseline_recommendation", {}),
        "variants": compact_variants,
    }
    comparison = {
        "purpose": "Compare diagnostic-mode-off validation candidates under scope-aware empirical targets.",
        "targets": targets,
        "notes": [
            "care_area is the active validation target for the current care-area ABM.",
            "full_empirical is retained only as a compatibility label and uses the same mapped F2F source.",
        ],
    }
    config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.VALIDATION_TARGET_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2))


def _write_part1_variant_audit(payload: Mapping[str, object]) -> None:
    """Save the latest four-variant comparison as a Part 1 mechanism finding."""

    variants = payload.get("variants", {})
    if set(variants) != set(config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS):
        return
    lines = [
        "# Part 1 Variant Audit Finding",
        "",
        "Purpose: preserve the four-variant comparison as a mechanism audit, not as routine baseline validation.",
        "",
        f"Validation target: `{payload.get('validation_target')}`",
        f"Run purpose: `{payload.get('run_purpose')}`",
        f"Duration: {payload.get('duration_seconds')} seconds",
        f"Paired seeds / runs: {payload.get('n_runs')}",
        "",
        "## Interpretation",
        "The four variants test whether perception, memory, and structured generative interpretation substantially change aggregate spatial-validation outcomes under the current workflow-constrained ED model. Similar aggregate metrics are interpretable as evidence that task routing and spatial opportunity structure dominate global behavior, while higher-cognition layers mainly leave trace-level mechanism changes.",
        "",
        "## Variant Results",
        "| Variant | Display label | Validation | Distance | Interactions | F2F/hour | HCW-HCW | Patient-facing | Station | Corridor | Bedside | Missed | Memory effects | Decision flips | Topic changes | Duration changes | Fallbacks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS:
        row = variants.get(variant, {})
        fit = row.get("validation_fit", {})
        outcomes = row.get("outcome_metrics", {})
        diagnostics = row.get("diagnostics", {})
        lines.append(
            "| "
            f"{variant} | {_display_variant_label(variant)} | "
            f"{fit.get('overall_status', 'UNKNOWN')} | "
            f"{float(row.get('distance_to_empirical', 0.0)):.3f} | "
            f"{int(row.get('interaction_count', 0))} | "
            f"{float(fit.get('simulated_f2f_per_hour', 0.0)):.2f} | "
            f"{float(outcomes.get('hcw_hcw_share', 0.0)):.3f} | "
            f"{float(outcomes.get('patient_facing_share', 0.0)):.3f} | "
            f"{float(outcomes.get('station_or_desk_interaction_share', 0.0)):.3f} | "
            f"{float(outcomes.get('corridor_interaction_share', 0.0)):.3f} | "
            f"{float(outcomes.get('bedside_or_patient_room_interaction_share', 0.0)):.3f} | "
            f"{int(row.get('missed_opportunity_episodes', 0))} | "
            f"{int(diagnostics.get('memory_effects_applied', row.get('memory_effects_applied', 0)))} | "
            f"{int(diagnostics.get('decision_flipped_by_memory', row.get('decision_flipped_by_memory', 0)))} | "
            f"{int(diagnostics.get('topic_changed_by_memory', row.get('topic_changed_by_memory', 0)))} | "
            f"{int(diagnostics.get('duration_changed_by_memory', row.get('duration_changed_by_memory', 0)))} | "
            f"{int(diagnostics.get('backend_fallbacks', row.get('backend_fallbacks', 0)))} |"
        )
    lines.extend(["", "## Paired-Delta Interpretation"])
    for variant in config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS:
        row = variants.get(variant, {})
        ensemble = row.get("ensemble_summary", {})
        lines.append(f"- `{variant}`: {ensemble.get('plain_language', 'No paired-delta summary available.')}")
    distances = {
        variant: float(variants.get(variant, {}).get("distance_to_empirical", 0.0))
        for variant in config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS
    }
    spread = max(distances.values()) - min(distances.values()) if distances else 0.0
    conclusion = (
        "Aggregate validation fit changed only modestly across variants; this supports treating the four-variant result as a mechanism audit rather than evidence of generative superiority."
        if spread < 0.15
        else "Aggregate validation fit differs across variants enough to inspect component-level gates before selecting a baseline."
    )
    lines.extend([
        "",
        "## Conclusion",
        conclusion,
        "",
        "Routine Part 1 validation should use the baseline rule model unless an ablation audit is explicitly requested.",
        "",
    ])
    config.FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    config.PART1_VARIANT_AUDIT_PATH.write_text("\n".join(lines))


def _format_share_table(rows: list[tuple[str, float, float]]) -> list[str]:
    lines = [
        "| Item | Empirical share | Simulated share | Sim - Emp |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, empirical_value, simulated_value in rows:
        lines.append(
            f"| {label} | {empirical_value:.3f} | {simulated_value:.3f} | {simulated_value - empirical_value:+.3f} |"
        )
    return lines


def _write_baseline_failure_diagnostic(
    summary: Mapping[str, Mapping[str, object]],
    empirical: Mapping[str, object],
    empirical_window: Mapping[str, object],
    validation_target_metadata: Mapping[str, object],
) -> None:
    """Write a focused provisional diagnostic for the baseline experimental-scope failure."""

    if set(summary.keys()) != {"traditional_rule"}:
        return
    if validation_target_metadata.get("validation_target") != "care_area":
        return
    row = summary["traditional_rule"]
    outcomes = row.get("outcome_metrics", {})
    fit = row.get("validation_fit", {})
    empirical_roles = empirical.get("role_pair_matrix", {})
    simulated_roles = outcomes.get("role_pair_distribution", {})
    role_rows = [
        (label, float(empirical_roles.get(label, 0.0)), float(simulated_roles.get(label, 0.0)))
        for label in sorted(set(empirical_roles) | set(simulated_roles))
    ]
    role_rows.sort(key=lambda item: item[2] - item[1])

    empirical_zones = empirical.get("zone_histogram", {})
    simulated_zones = outcomes.get("zone_distribution", {})
    zone_rows = [
        (label, float(empirical_zones.get(label, 0.0)), float(simulated_zones.get(label, 0.0)))
        for label in sorted(set(empirical_zones) | set(simulated_zones))
    ]
    zone_rows.sort(key=lambda item: item[2] - item[1])

    interaction_type_counts = row.get("interaction_type_counts", {})
    patient_facing_by_type = row.get("patient_facing_interaction_type_counts", {})
    patient_facing_by_task = row.get("patient_facing_task_counts", {})
    task_counts = row.get("interaction_task_counts", {})
    workflow_task_counts = row.get("workflow_task_counts", {})
    patient_facing_attendance = row.get("patient_facing_task_attendance_counts", {})
    patient_facing_validation = row.get("patient_facing_validation_interaction_counts", {})
    total_interactions = max(int(row.get("interaction_count", 0)), 1)
    patient_facing_count = sum(int(value) for value in patient_facing_by_type.values())

    role_over = sorted(role_rows, key=lambda item: item[2] - item[1], reverse=True)[:4]
    role_under = sorted(role_rows, key=lambda item: item[2] - item[1])[:4]
    zone_over = sorted(zone_rows, key=lambda item: item[2] - item[1], reverse=True)[:4]
    zone_under = sorted(zone_rows, key=lambda item: item[2] - item[1])[:4]
    nursta_sim = float(simulated_zones.get("NURSTA", 0.0))
    nurope_sim = float(simulated_zones.get("NUROPE", 0.0))
    nursta_emp = float(empirical_zones.get("NURSTA", 0.0))
    nurope_emp = float(empirical_zones.get("NUROPE", 0.0))

    lines = [
        "# Part 1 Baseline Failure Diagnostic",
        "",
        "Status: provisional diagnostic, not a finalized study finding.",
        "",
        "## Run Context",
        f"- Validation target: `{validation_target_metadata.get('validation_target')}`",
        "- Variant: `traditional_rule` / `baseline_rule_model`",
        f"- Duration: {row.get('duration_seconds')} seconds",
        f"- Paired seeds / runs: {row.get('n_runs')}",
        f"- Validation status: {fit.get('overall_status', 'UNKNOWN')}",
        f"- Distance to empirical: {float(row.get('distance_to_empirical', 0.0)):.3f}",
        f"- Empirical F2F rate: {float(empirical_window.get('f2f_records_per_hour', 0.0)):.2f}/hour",
        f"- Simulated F2F rate: {float(fit.get('simulated_f2f_per_hour', 0.0)):.2f}/hour",
        "",
        "## HCW-HCW / Patient-Facing Shares",
        *_format_share_table([
            ("HCW-HCW", float(fit.get("empirical_hcw_hcw_share", 0.0)), float(outcomes.get("hcw_hcw_share", 0.0))),
            ("HCW-patient", float(fit.get("empirical_hcw_patient_share", 0.0)), float(outcomes.get("hcw_patient_share", 0.0))),
            ("patient-facing", float(fit.get("empirical_patient_facing_share", 0.0)), float(outcomes.get("patient_facing_share", 0.0))),
        ]),
        "",
        "## Empirical vs Simulated Role-Pair Shares",
        *_format_share_table(role_rows),
        "",
        "## Simulated Interaction Counts By Interaction Type",
        "| Interaction type | Count | Patient-facing count | Share of all interactions |",
        "| --- | ---: | ---: | ---: |",
    ]
    for interaction_type, count in sorted(interaction_type_counts.items(), key=lambda item: int(item[1]), reverse=True):
        count_int = int(count)
        lines.append(
            f"| {interaction_type} | {count_int} | {int(patient_facing_by_type.get(interaction_type, 0))} | {count_int / total_interactions:.3f} |"
        )
    lines.extend([
        "",
        "## Simulated Interaction Counts By Workflow Task",
        "| Task | Count | Patient-facing count |",
        "| --- | ---: | ---: |",
    ])
    for task_name, count in sorted(task_counts.items(), key=lambda item: int(item[1]), reverse=True):
        lines.append(f"| {task_name} | {int(count)} | {int(patient_facing_by_task.get(task_name, 0))} |")
    lines.extend([
        "",
        "## Workflow Task Counts",
        "| Workflow task | Count |",
        "| --- | ---: |",
    ])
    for task_name, count in sorted(workflow_task_counts.items(), key=lambda item: int(item[1]), reverse=True):
        lines.append(f"| {task_name} | {int(count)} |")
    lines.extend([
        "",
        "## Patient-Facing Task Attendance vs Validation-Counted Interactions",
        "| Role/task | Task attendance count | Validation-counted patient-facing count |",
        "| --- | ---: | ---: |",
    ])
    for role_task in sorted(set(patient_facing_attendance) | set(patient_facing_validation)):
        lines.append(
            f"| {role_task} | {int(patient_facing_attendance.get(role_task, 0))} | {int(patient_facing_validation.get(role_task, 0))} |"
        )
    lines.extend([
        "",
        "## Empirical vs Simulated Zone Shares",
        *_format_share_table(zone_rows),
        "",
        "## Diagnostic Interpretation",
        f"- Patient-facing interactions account for {patient_facing_count}/{total_interactions} simulated interactions ({patient_facing_count / total_interactions:.1%}), versus {float(fit.get('empirical_patient_facing_share', 0.0)):.1%} in the experimental target.",
        "- Patient-facing task attendance is now counted separately from validation-counted patient communication. Workflow task starts still execute and are logged as workflow events, but only selected explanation/check-in stages can enter the interaction log.",
        "- Bed assignment remains workflow-only. Placement can create a validation interaction only when the explicit patient communication gate passes.",
        "- Nursing and doctor task starts no longer automatically become validation-equivalent F2F events; routine task attendance is mostly excluded unless the task is in the validation-communication set and passes interval/probability gates.",
        "- Opportunistic station/corridor rules primarily generate HCW-HCW interactions and are not the main source of patient-facing excess.",
        "",
        "## Largest Role-Pair Overproduction",
        *_format_share_table(role_over),
        "",
        "## Largest Role-Pair Underproduction",
        *_format_share_table(role_under),
        "",
        "## Largest Zone Overproduction",
        *_format_share_table(zone_over),
        "",
        "## Largest Zone Underproduction",
        *_format_share_table(zone_under),
        "",
        "## NURSTA / NUROPE Use",
        f"- NURSTA empirical share: {nursta_emp:.3f}; simulated share: {nursta_sim:.3f}; delta: {nursta_sim - nursta_emp:+.3f}.",
        f"- NUROPE empirical share: {nurope_emp:.3f}; simulated share: {nurope_sim:.3f}; delta: {nurope_sim - nurope_emp:+.3f}.",
        "- Interpretation: the baseline experimental run underuses both NURSTA and NUROPE as interaction locations despite their inclusion in the target.",
        "",
        "## Ranked Likely Causes",
        "1. Patient-facing task-start communication is too frequent or too broadly counted as validation-equivalent F2F communication for the experimental target.",
        "2. Doctor-patient and coordination-nurse-patient interactions are overrepresented relative to the target, while nurse-nurse and coordination-nurse-nurse interactions are underrepresented.",
        "3. Bedside / outside-named-zone interactions are overrepresented, while NURSTA and NUROPE station interactions are nearly absent in the baseline aggregate.",
        "4. The current baseline still routes and logs many clinically plausible patient contacts, but the empirical experimental target appears more HCW-HCW coordination-heavy.",
        "",
        "## Smallest Structural Correction Candidates",
        "- Split patient-care task execution from validation-relevant patient-facing explanation/check-in events more strictly.",
        "- Add a task-stage or topic filter so routine task attendance does not automatically become an empirical F2F analogue.",
        "- Increase station-context coordination opportunities at NURSTA/NUROPE through existing staff station dwell/coordination mechanisms, not by adding new agents or scope.",
        "- Recheck placement handoff semantics: keep bed assignment workflow-only and only log placement communication when it corresponds to a face-to-face exchange.",
        "",
        "## Recommended Next Implementation Step",
        "Audit and revise the patient-facing task-start logging rule first. The goal is to reduce validation-counted patient-facing events without suppressing underlying workflow execution. Do not tune global probabilities until the logging semantics are tightened.",
        "",
    ])
    config.PART1_BASELINE_FAILURE_DIAGNOSTIC_PATH.write_text("\n".join(lines))


def _status_from_outcome_count(count: int) -> str:
    if count >= 2:
        return "READY"
    if count >= 1:
        return "PARTIAL"
    return "NOT READY"


def _readiness_from_summary(summary: Mapping[str, Mapping[str, object]]) -> dict:
    memory = summary.get("memory_rule", {})
    generative = summary.get("generative_interaction", {})
    memory_diag = memory.get("diagnostics", {}) if isinstance(memory, Mapping) else {}
    generative_diag = generative.get("diagnostics", {}) if isinstance(generative, Mapping) else {}
    mechanism_ok = (
        int(memory_diag.get("memory_effects_applied", 0)) > 0
        and int(generative_diag.get("structured_fallback_calls", 0)) > 0
    )
    mechanism_status = "READY" if mechanism_ok else "PARTIAL"

    strongest_count = 0
    for variant, payload in summary.items():
        if variant == "traditional_rule" or not isinstance(payload, Mapping):
            continue
        strongest_count = max(int(payload.get("outcome_difference_signal_count", 0)), strongest_count)
    max_runs = max(
        int(payload.get("n_runs", 1))
        for payload in summary.values()
        if isinstance(payload, Mapping)
    ) if summary else 1
    outcome_status = _status_from_outcome_count(strongest_count)
    if max_runs <= 1 and outcome_status == "READY":
        outcome_status = "PARTIAL"

    metric_audit = next(
        (
            payload.get("validation_metric_audit", {})
            for payload in summary.values()
            if isinstance(payload, Mapping) and payload.get("validation_metric_audit")
        ),
        {},
    )
    topic_ok = bool(metric_audit.get("empirical_topic_target_nonempty")) and bool(metric_audit.get("empirical_topic_by_zone_nonempty"))
    finite_components = all(
        np.isfinite(float(component_value))
        for payload in summary.values()
        if isinstance(payload, Mapping)
        for component_value in payload.get("distance_components", {}).values()
    )
    empirical_metric_status = "READY" if topic_ok and finite_components else "NOT READY"
    if finite_components and not topic_ok:
        empirical_metric_status = "PARTIAL"

    validation_status = (
        "PARTIAL"
        if mechanism_status in {"READY", "PARTIAL"}
        and outcome_status in {"READY", "PARTIAL"}
        and empirical_metric_status in {"READY", "PARTIAL"}
        else "NOT READY"
    )
    if empirical_metric_status == "NOT READY" or outcome_status == "NOT READY":
        validation_status = "NOT READY"

    return {
        "mechanism_differentiation": mechanism_status,
        "outcome_differentiation": outcome_status,
        "empirical_metric_interpretability": empirical_metric_status,
        "validation_readiness": validation_status,
        "outcome_difference_signal_count": strongest_count,
    }


def _paired_delta_noise_statement(deltas: list[Mapping[str, float]], metric: str) -> str:
    if len(deltas) < 2:
        return "Not enough paired seeds to compare differences with run-to-run noise."
    values = [float(delta.get(metric, 0.0)) for delta in deltas]
    mean_delta = float(mean(values))
    sd_delta = float(stdev(values))
    if abs(mean_delta) > sd_delta:
        return (
            f"For {metric}, the mean paired delta ({mean_delta:.3f}) is larger than "
            f"the paired-delta SD ({sd_delta:.3f}); inspect metric-level deltas before interpreting."
        )
    return (
        f"For {metric}, the mean paired delta ({mean_delta:.3f}) is not larger than "
        f"the paired-delta SD ({sd_delta:.3f}); differences look noisy at this ensemble size."
    )


def _plot_dashboard(rows: list[dict]) -> None:
    variants = [row["model_variant"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    components = ["kde_component", "zone_component", "role_pair_component", "topic_component", "duration_component"]
    x = np.arange(len(variants))
    width = 0.8 / len(components)
    for index, component in enumerate(components):
        axes[0].bar(
            x + (index * width),
            [float(row.get("distance_components", {}).get(component, 0.0)) for row in rows],
            width=width,
            label=component.replace("_component", ""),
        )
    axes[0].set_xticks(x + width * (len(components) - 1) / 2)
    axes[0].set_xticklabels(variants, rotation=25, ha="right")
    axes[0].set_title("Empirical distance components")
    axes[0].set_ylabel("Component distance")
    axes[0].legend(fontsize=7)

    delta_keys = [
        ("count", "interaction_count_delta"),
        ("missed", "missed_opportunity_episodes_delta"),
        ("HCW-HCW share", "hcw_hcw_share_delta"),
        ("patient share", "hcw_patient_share_delta"),
        ("mean duration", "mean_duration_delta"),
    ]
    x_delta = np.arange(len(delta_keys))
    width_delta = 0.8 / max(len(rows) - 1, 1)
    comparison_rows = [row for row in rows if row["model_variant"] != "traditional_rule"]
    for index, row in enumerate(comparison_rows):
        axes[1].bar(
            x_delta + (index * width_delta),
            [float(row.get("outcome_deltas_vs_traditional", {}).get(key, 0.0)) for _, key in delta_keys],
            width=width_delta,
            label=row["model_variant"],
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x_delta + width_delta * max(len(comparison_rows) - 1, 0) / 2)
    axes[1].set_xticklabels([label for label, _ in delta_keys], rotation=25, ha="right")
    axes[1].set_title("Outcome deltas vs traditional_rule")
    axes[1].set_ylabel("Delta")
    axes[1].legend(fontsize=7)

    divergence_keys = [
        ("role-pair", "role_pair_jsd_vs_traditional"),
        ("topic", "topic_jsd_vs_traditional"),
        ("zone", "zone_jsd_vs_traditional"),
    ]
    x_div = np.arange(len(divergence_keys))
    for index, row in enumerate(comparison_rows):
        axes[2].bar(
            x_div + (index * width_delta),
            [float(row.get(key, 0.0)) for _, key in divergence_keys],
            width=width_delta,
            label=row["model_variant"],
        )
    axes[2].set_xticks(x_div + width_delta * max(len(comparison_rows) - 1, 0) / 2)
    axes[2].set_xticklabels([label for label, _ in divergence_keys], rotation=25, ha="right")
    axes[2].set_title("Distributional divergence vs traditional_rule")
    axes[2].set_ylabel("JSD")
    axes[2].legend(fontsize=7)

    mechanism_keys = [
        ("mem effects", "memory_effects_applied"),
        ("mem flips", "decision_flipped_by_memory"),
        ("FOV hard", "rejected_fov_hard"),
        ("fallback", "backend_fallbacks"),
        ("topic delta", "topic_jsd_vs_traditional"),
    ]
    x_mech = np.arange(len(mechanism_keys))
    for index, row in enumerate(rows):
        axes[3].bar(
            x_mech + (index * (0.8 / max(len(rows), 1))),
            [float(row.get(key, row.get("diagnostics", {}).get(key, 0.0))) for _, key in mechanism_keys],
            width=0.8 / max(len(rows), 1),
            label=row["model_variant"],
        )
    axes[3].set_xticks(x_mech + (0.8 / max(len(rows), 1)) * max(len(rows) - 1, 0) / 2)
    axes[3].set_xticklabels([label for label, _ in mechanism_keys], rotation=25, ha="right")
    axes[3].set_title("Mechanism counters beside outcome movement")
    axes[3].set_ylabel("Count or JSD")
    axes[3].legend(fontsize=7)
    fig.suptitle("Ablation dashboard: mechanisms are not validation; inspect outcome movement", y=0.995)
    fig.tight_layout()
    fig.savefig(config.ABLATION_DASHBOARD_PATH, dpi=220)
    plt.close(fig)


def _plot_decision_counts(rows: list[dict]) -> None:
    panels = [
        (
            "High-volume gates",
            [
                ("eligible", "eligible_encounters_considered"),
                ("distance", "rejected_by_distance"),
                ("cooldown", "rejected_by_cooldown"),
            ],
        ),
        (
            "Low-volume gates",
            [
                ("visibility", "rejected_by_visibility"),
                ("FOV hard", "rejected_fov_hard"),
                ("FOV soft", "rejected_fov_soft_probability"),
                ("rule/prob", "rejected_by_probability_or_rule_decision"),
                ("accepted", "accepted_as_interactions"),
            ],
        ),
        (
            "Mechanism counts",
            [
                ("memory", "memory_effects_applied"),
                ("mem flips", "decision_flipped_by_memory"),
                ("duration", "duration_changed_by_memory"),
                ("topic", "topic_changed_by_memory"),
                ("fallback", "backend_fallbacks"),
            ],
        ),
        (
            "Outcome counts",
            [
                ("interactions", "accepted_as_interactions"),
                ("missed", "missed_opportunity_episodes"),
                ("interruptions", "interruption_events"),
                ("high-acuity", "high_acuity_interactions"),
            ],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for axis, (title, metrics) in zip(axes, panels):
        labels = [label for label, _ in metrics]
        x = np.arange(len(labels))
        width = 0.8 / max(len(rows), 1)
        for index, row in enumerate(rows):
            values = [float(row.get(key, 0.0)) for _, key in metrics]
            axis.bar(x + (index * width), values, width=width, label=row["model_variant"])
        axis.set_title(title)
        axis.set_xticks(x + width * max(len(rows) - 1, 0) / 2)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylabel("Count")
    axes[0].legend(fontsize=8)
    fig.suptitle("Ablation diagnostic counts, not empirical validation", y=0.995)
    fig.tight_layout()
    fig.savefig(config.ABLATION_DECISION_COUNTS_PATH, dpi=220)
    plt.close(fig)


def _plot_mechanism_audit(rows: list[dict]) -> None:
    panels = [
        ("Perception gates", ["rejected_by_visibility", "rejected_fov_hard", "accepted_shared_station_awareness", "accepted_patient_facing"]),
        ("Memory mechanisms", ["memory_retrieval_calls", "memory_effects_applied", "decision_flipped_by_memory", "topic_changed_by_memory", "duration_changed_by_memory"]),
        ("Generative fallback", ["structured_fallback_calls", "backend_fallbacks", "schema_parse_failures"]),
        ("Outcomes for context", ["accepted_as_interactions", "missed_opportunity_episodes", "high_acuity_interactions"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for axis, (title, keys) in zip(axes, panels):
        x = np.arange(len(keys))
        width = 0.8 / max(len(rows), 1)
        for index, row in enumerate(rows):
            diagnostics = row.get("diagnostics", row)
            axis.bar(
                x + (index * width),
                [float(row.get(key, diagnostics.get(key, 0.0))) for key in keys],
                width=width,
                label=row["model_variant"],
            )
        axis.set_title(title)
        axis.set_xticks(x + width * max(len(rows) - 1, 0) / 2)
        axis.set_xticklabels([key.replace("_", "\n") for key in keys], fontsize=8)
        axis.set_ylabel("Count")
    axes[0].legend(fontsize=7)
    fig.suptitle("Mechanism audit, not empirical validation", y=0.995)
    fig.tight_layout()
    fig.savefig(config.MECHANISM_AUDIT_DASHBOARD_PATH, dpi=220)
    plt.close(fig)


def _plot_validation_fit(rows: list[dict], empirical: Mapping[str, object]) -> None:
    best_rows = [row for row in rows if row.get("validation_fit")]
    if not best_rows:
        return
    variants = [row["model_variant"] for row in best_rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    empirical_rate = [float(row["validation_fit"].get("empirical_f2f_per_hour", 0.0)) for row in best_rows]
    simulated_rate = [float(row["validation_fit"].get("simulated_f2f_per_hour", 0.0)) for row in best_rows]
    x = np.arange(len(variants))
    axes[0].bar(x - 0.18, empirical_rate, width=0.36, label="empirical", color="#4C78A8")
    axes[0].bar(x + 0.18, simulated_rate, width=0.36, label="simulated", color="#F58518")
    axes[0].set_title("Observed vs simulated interaction rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=25, ha="right")
    axes[0].set_ylabel("Interactions/hour")
    axes[0].legend(fontsize=8)

    component_keys = [
        "f2f_rate_relative_error",
        "hcw_hcw_share_abs_error",
        "patient_facing_share_abs_error",
        "station_share_abs_error",
        "corridor_share_abs_error",
        "zone_jsd",
    ]
    width = 0.8 / max(len(best_rows), 1)
    x_comp = np.arange(len(component_keys))
    for index, row in enumerate(best_rows):
        metrics = row["validation_fit"]["metrics"]
        axes[1].bar(
            x_comp + (index * width),
            [float(metrics[key]["value"]) if metrics[key]["value"] is not None else 0.0 for key in component_keys],
            width=width,
            label=row["model_variant"],
        )
    axes[1].set_title("Key validation gates")
    axes[1].set_xticks(x_comp + width * max(len(best_rows) - 1, 0) / 2)
    axes[1].set_xticklabels([key.replace("_", "\n") for key in component_keys], fontsize=8)
    axes[1].set_ylabel("Distance/error")

    pass_order = {"PASS": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": 3}
    statuses = [row["validation_fit"]["overall_status"] for row in best_rows]
    axes[2].bar(variants, [pass_order.get(status, 3) for status in statuses], color="#E45756")
    axes[2].set_title("Overall threshold status")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(["PASS", "WARN", "FAIL", "UNKNOWN"])
    axes[2].tick_params(axis="x", rotation=25)

    empirical_groups = empirical.get("zone_supergroup_distribution", {})
    groups = ["station_or_desk", "corridor", "bedside_or_patient_room", "other_or_uncoded"]
    x_topic = np.arange(len(groups))
    width_topic = 0.8 / (len(best_rows) + 1)
    axes[3].bar(
        x_topic,
        [float(empirical_groups.get(group, 0.0)) for group in groups],
        width=width_topic,
        label="empirical",
        color="#4C78A8",
    )
    for index, row in enumerate(best_rows):
        axes[3].bar(
            x_topic + ((index + 1) * width_topic),
            [float(row.get("outcome_metrics", {}).get("zone_supergroup_distribution", {}).get(group, 0.0)) for group in groups],
            width=width_topic,
            label=row["model_variant"],
        )
    axes[3].set_title("Station/corridor/bedside ecology")
    axes[3].set_xticks(x_topic + width_topic * len(best_rows) / 2)
    axes[3].set_xticklabels([group.replace("_", "\n") for group in groups], rotation=25, ha="right", fontsize=8)
    axes[3].set_ylabel("Share")
    axes[3].legend(fontsize=6)
    fig.suptitle("Validation candidate: diagnostic mode off", y=0.995)
    fig.tight_layout()
    fig.savefig(config.VALIDATION_FIT_DASHBOARD_PATH, dpi=220)
    plt.close(fig)


def _plot_paired_seed_uncertainty(rows: list[dict]) -> None:
    paired_rows = [
        row for row in rows
        if row.get("ensemble_summary", {}).get("paired_delta_vs_traditional_by_seed")
    ]
    if not paired_rows:
        return
    variants = [row["model_variant"] for row in paired_rows]
    means = []
    errors = []
    for row in paired_rows:
        deltas = [
            float(delta.get("distance_to_empirical", 0.0))
            for delta in row["ensemble_summary"]["paired_delta_vs_traditional_by_seed"]
        ]
        means.append(float(mean(deltas)))
        errors.append(float(stdev(deltas)) if len(deltas) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(variants, means, yerr=errors, capsize=5, color="#72B7B2")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Paired-seed uncertainty, not validation")
    ax.set_ylabel("Distance delta vs traditional_rule")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(config.PAIRED_SEED_UNCERTAINTY_PATH, dpi=220)
    plt.close(fig)


def _model_sufficiency_summary(summary: Mapping[str, Mapping[str, object]]) -> dict:
    metric_keys = [
        "interaction_count",
        "distance_to_empirical",
        "hcw_hcw_share",
        "patient_facing_share",
        "station_or_desk_interaction_share",
        "corridor_interaction_share",
        "mean_duration",
    ]
    variants = {
        variant: payload
        for variant, payload in summary.items()
        if isinstance(payload, Mapping) and payload.get("per_run_metrics")
    }
    metric_summary = {}
    for metric_key in metric_keys:
        means = {}
        within_sds = {}
        for variant, payload in variants.items():
            values = [float(record.get(metric_key, 0.0)) for record in payload.get("per_run_metrics", [])]
            if not values:
                continue
            means[variant] = float(mean(values))
            within_sds[variant] = float(stdev(values)) if len(values) > 1 else 0.0
        between_sd = float(stdev(means.values())) if len(means) > 1 else 0.0
        typical_within_sd = float(mean(within_sds.values())) if within_sds else 0.0
        metric_summary[metric_key] = {
            "variant_means": means,
            "within_variant_seed_sd": within_sds,
            "between_variant_sd": between_sd,
            "typical_within_variant_sd": typical_within_sd,
            "between_larger_than_seed_noise": between_sd > typical_within_sd,
        }
    classifications = {}
    for variant, payload in variants.items():
        mechanism_active = bool(payload.get("variant_classification", {}).get("mechanism_active"))
        outcome_count = int(payload.get("outcome_difference_signal_count", 0))
        if mechanism_active and outcome_count >= 2:
            label = "mechanism active / outcome different"
        elif mechanism_active and outcome_count >= 1:
            label = "mechanism active / outcome similar"
        elif mechanism_active:
            label = "mechanism active / outcome similar"
        else:
            label = "mechanism inactive or ineffective"
        classifications[variant] = {
            "classification": label,
            "outcome_difference_signal_count": outcome_count,
            "mechanism_active": mechanism_active,
            "interpretation": payload.get("variant_classification", {}).get("interpretation", ""),
        }
    return {
        "purpose": "Model sufficiency audit: separates mechanism activity from aggregate outcome differentiation.",
        "plain_language": (
            "If between-variant differences are smaller than within-variant seed noise, "
            "the current workflow/spatial opportunity structure is dominating aggregate behavior."
        ),
        "metric_summary": metric_summary,
        "variant_classifications": classifications,
    }


def _plot_model_sufficiency(summary: Mapping[str, Mapping[str, object]]) -> None:
    sufficiency = _model_sufficiency_summary(summary)
    metrics = [
        "interaction_count",
        "hcw_hcw_share",
        "patient_facing_share",
        "station_or_desk_interaction_share",
        "corridor_interaction_share",
        "mean_duration",
    ]
    labels = ["count", "HCW-HCW", "patient-facing", "station", "corridor", "duration"]
    between = [
        sufficiency["metric_summary"].get(metric, {}).get("between_variant_sd", 0.0)
        for metric in metrics
    ]
    within = [
        sufficiency["metric_summary"].get(metric, {}).get("typical_within_variant_sd", 0.0)
        for metric in metrics
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, between, width=0.36, label="between variants", color="#4C78A8")
    ax.bar(x + 0.18, within, width=0.36, label="within variant seed noise", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("SD")
    ax.set_title("Model sufficiency: mechanism activity versus outcome separation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.MODEL_SUFFICIENCY_DASHBOARD_PATH, dpi=220)
    plt.close(fig)


def _status_line(label: str, status: str, note: str) -> str:
    return f"- {label}: {status} - {note}"


def write_model_status(
    *,
    source: str,
    duration_seconds: int,
    summary: Optional[Mapping[str, Mapping[str, object]]] = None,
    high_acuity_payload: Optional[Mapping[str, object]] = None,
) -> None:
    """Write the single plain-language readiness report for the latest standard run."""

    summary = summary or {}
    baseline = next(iter(summary.values()), {}) if summary else {}
    fit = baseline.get("validation_fit", {}) if isinstance(baseline, Mapping) else {}
    outcomes = baseline.get("outcome_metrics", {}) if isinstance(baseline, Mapping) else {}
    workflow = baseline.get("workflow_health", {}) if isinstance(baseline, Mapping) else {}
    perception = baseline.get("perception_diagnostics", baseline.get("diagnostics", {})) if isinstance(baseline, Mapping) else {}
    distance = baseline.get("distance_to_empirical", None) if isinstance(baseline, Mapping) else None
    lines = [
        "# Model Status",
        "",
        f"Latest run source: {source}",
        f"Latest duration: {duration_seconds} seconds",
        "",
        "## Architecture",
        "Part 1 uses one baseline model: a rule-based, workflow-constrained, movement-valid, spatially grounded perception model. Visibility, field of view, salience, attention capacity, and cooldowns are baseline mechanisms.",
        "",
        "Part 2 uses the same baseline under spatial/design conditions: baseline, cockpit_only, nursta_only, and both.",
        "",
        "Part 3 is reserved for a later LLM/generative-agent extension. It is not active in Part 1 or Part 2.",
        "",
        "## Current Baseline",
        "- Selected baseline: rule-based perception care-area model.",
        "- SeniorDoctor oversight: disabled by default; retained only as sensitivity code.",
        "- Workflow events remain separate from validation-counted F2F interaction logs.",
        "- Cooldowns act as a crude short-term memory substitute.",
        "",
        "## Latest Validation Snapshot",
        f"- Status: {fit.get('overall_status', 'UNKNOWN')}",
        f"- Distance to empirical: {float(distance):.3f}" if isinstance(distance, (int, float)) else "- Distance to empirical: unknown",
        f"- Simulated F2F/hour: {float(fit.get('simulated_f2f_per_hour', 0.0)):.2f}",
        f"- HCW-HCW share: {float(outcomes.get('hcw_hcw_share', 0.0)):.3f}",
        f"- Patient-facing share: {float(outcomes.get('patient_facing_share', 0.0)):.3f}",
        f"- Station/desk share: {float(outcomes.get('station_or_desk_interaction_share', 0.0)):.3f}",
        f"- Corridor share: {float(outcomes.get('corridor_interaction_share', 0.0)):.3f}",
        f"- Workflow health: {workflow.get('workflow_health_status', 'UNKNOWN')}",
        f"- Bed 8 ESI 3-5 violations: {workflow.get('reserved_bed_restriction_violations', 0)}",
        "",
        "## Perception Mechanism",
        f"- Perception baseline enabled: {perception.get('perception_based_baseline_enabled', config.PERCEPTION_BASED_BASELINE_ENABLED)}",
        f"- Raw visible staff percepts: {perception.get('raw_visible_staff_percepts_count', 0)}",
        f"- Eligible visible staff opportunities: {perception.get('eligible_visible_staff_opportunity_count', 0)}",
        f"- Selected visible staff opportunities: {perception.get('selected_visible_staff_opportunity_count', 0)}",
        f"- Perceived staff interactions: {perception.get('perceived_staff_interaction_count', 0)}",
        f"- Non-proximate logged staff interactions: {perception.get('nonproximate_logged_interaction_count', 0)}",
        f"- Protected-state suppressions: {perception.get('protected_state_suppression_count', 0)}",
        f"- Cooldown rejections: {perception.get('rejected_by_cooldown', 0)}",
        "",
        "## Readiness",
        "Part 2 can proceed only after the latest Part 1 validation has acceptable empirical fit, workflow health PASS, Bed 8 violations equal zero, and perception-mechanism diagnostics prove visible non-proximate interaction opportunities exist.",
        "",
    ]
    config.MODEL_STATUS_PATH.write_text("\n".join(lines))
    return

    if summary is None:
        for candidate in (
            config.ABLATION_SUMMARY_PATH.with_name("ablation_summary_4h.json"),
            config.ABLATION_SUMMARY_PATH.with_name("ablation_summary_1h.json"),
            config.ABLATION_SUMMARY_PATH.with_name("ablation_summary_30m.json"),
            config.ABLATION_SUMMARY_PATH.with_name("ablation_summary_latest_debug.json"),
        ):
            if candidate.exists():
                summary = json.loads(candidate.read_text())
                break
    summary = summary or {}
    generative = summary.get("generative_interaction", {})
    diagnostics = generative.get("diagnostics", {}) if isinstance(generative, Mapping) else {}
    topic_count = 0
    if isinstance(generative, Mapping):
        interaction_summary = generative.get("interaction_summary", {})
        if isinstance(interaction_summary, Mapping):
            topics = interaction_summary.get("interactions_by_topic", {})
            if isinstance(topics, Mapping):
                topic_count = len([topic for topic, count in topics.items() if int(count) > 0])
    missed = int(diagnostics.get("missed_opportunity_episodes", 0)) if isinstance(diagnostics, Mapping) else 0
    high_acuity_interactions = int(diagnostics.get("high_acuity_interactions", 0)) if isinstance(diagnostics, Mapping) else 0
    if high_acuity_payload is not None:
        high_acuity_interactions = int(high_acuity_payload.get("high_acuity_interactions", 0))
    topic_status = "PARTIAL" if topic_count >= 3 else "NOT READY"
    missed_status = "PARTIAL" if missed > 0 else "NOT READY"
    high_acuity_status = "PARTIAL" if high_acuity_interactions > 0 else "NOT READY"
    longer_status = "READY" if duration_seconds >= 3600 and topic_status == "PARTIAL" else "PARTIAL"
    output_status = "READY" if config.OUTPUTS_README_PATH.exists() else "PARTIAL"
    ablation_status = "PARTIAL" if isinstance(generative, Mapping) and generative.get("interaction_count", 0) else "NOT READY"
    readiness = _readiness_from_summary(summary)
    metric_audit = next(
        (
            payload.get("validation_metric_audit", {})
            for payload in summary.values()
            if isinstance(payload, Mapping) and payload.get("validation_metric_audit")
        ),
        {},
    )
    topic_note = (
        "Topic component has empirical targets and is no longer the empty-target 0.5 artifact."
        if metric_audit.get("empirical_topic_by_zone_nonempty")
        else "Topic component is not fully interpretable because empirical topic targets are missing."
    )

    paired_run = False
    if "n_runs=" in source:
        try:
            paired_run = int(source.split("n_runs=", 1)[1].split()[0].strip(",")) > 1
        except (IndexError, ValueError):
            paired_run = False
    validation_run = source.startswith("validation candidate")
    mechanism_run = source.startswith("mechanism audit")
    validation_recommendation = _select_baseline_recommendation(summary) if validation_run else {}
    if validation_run:
        if validation_recommendation.get("status") == "PASS":
            next_action = "Inspect `outputs/latest/validation_summary.json`; if provisional gates are acceptable, discuss whether the recommended baseline is defensible before any Part 2 intervention run."
        elif validation_recommendation.get("recommended_variant"):
            next_action = "Do not run Part 2 yet. Inspect WARN gates in `outputs/latest/validation_summary.json`, especially HCW-HCW share and visibility-sensitive baseline choice."
        else:
            next_action = "Do not run Part 2 yet. Inspect failed validation gates in `outputs/latest/validation_summary.json` and decide which empirical-fit issue to address first."
    elif mechanism_run:
        next_action = "Inspect `outputs/latest/mechanism_audit_summary.json`; treat this as mechanism traceability, not validation."
    elif source == "high-acuity smoke":
        next_action = "Inspect `outputs/latest/mechanism_audit_diagnostics.json` and scenario pressure counters."
    elif paired_run:
        next_action = "Inspect the `ensemble_summary` sections in `outputs/latest/ensemble_summary.json`; the next scientific step is deciding whether outcome differences are large enough to justify empirical-matched validation runs."
    elif duration_seconds >= 14400:
        next_action = "Inspect the latest mechanism audit and validation candidate outputs before any longer run."
    else:
        next_action = "Run `python3 main.py --mechanism-audit --duration 3600 --n-runs 10`, then run diagnostic-mode-off validation candidate."
    readiness_color = "ORANGE"
    if validation_run and validation_recommendation.get("status") == "PASS":
        readiness_color = "YELLOW"
    elif validation_run and not validation_recommendation.get("recommended_variant"):
        readiness_color = "ORANGE"
    empirical_validation_status = (
        "NOT READY"
        if validation_run and not validation_recommendation.get("recommended_variant")
        else readiness["validation_readiness"]
    )
    empirical_window_text = "Empirical window summary has not been generated yet."
    if config.EMPIRICAL_WINDOW_SUMMARY_PATH.exists():
        try:
            empirical_window = json.loads(config.EMPIRICAL_WINDOW_SUMMARY_PATH.read_text())
            empirical_window_text = (
                f"{empirical_window.get('validation_record_count', 0)} F2F records across "
                f"{float(empirical_window.get('total_observed_clock_span_hours', 0.0)):.2f} summed observed hours "
                f"({float(empirical_window.get('f2f_records_per_hour', 0.0)):.2f} records/hour). "
                f"The {float(empirical_window.get('continuous_calendar_span_hours', 0.0)):.2f}-hour calendar span is metadata only."
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            empirical_window_text = "Empirical window summary exists but could not be parsed."
    scope_alignment_text = "Scope alignment audit has not been generated yet."
    if config.SCOPE_ALIGNMENT_AUDIT_PATH.exists():
        try:
            scope_payload = json.loads(config.SCOPE_ALIGNMENT_AUDIT_PATH.read_text())
            alignment = scope_payload.get("scope_alignment", {})
            strategy = scope_payload.get("validation_strategy", {})
            scope_alignment_text = (
                f"{float(alignment.get('inside_active_simulated_scope_share', 0.0)):.1%} of empirical F2F records are inside active simulated scope; "
                f"{float(alignment.get('inactive_adjacent_outside_or_unclear_share', 0.0)):.1%} are adjacent, inactive, outside, or unclear. "
                f"Recommended strategy: {strategy.get('recommended_strategy', 'unknown')}."
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            scope_alignment_text = "Scope alignment audit exists but could not be parsed."
    candidate_ecology = ""
    if validation_recommendation.get("recommended_variant") in summary:
        candidate = summary[str(validation_recommendation["recommended_variant"])]
        outcomes = candidate.get("outcome_metrics", {}) if isinstance(candidate, Mapping) else {}
        fit = candidate.get("validation_fit", {}) if isinstance(candidate, Mapping) else {}
        candidate_ecology = (
            f"`{validation_recommendation['recommended_variant']}` currently has "
            f"{float(outcomes.get('hcw_hcw_share', 0.0)):.1%} HCW-HCW, "
            f"{float(outcomes.get('patient_facing_share', 0.0)):.1%} patient-facing, "
            f"{float(outcomes.get('station_or_desk_interaction_share', 0.0)):.1%} station/desk, "
            f"{float(outcomes.get('corridor_interaction_share', 0.0)):.1%} corridor, "
            f"and {float(fit.get('simulated_f2f_per_hour', 0.0)):.1f} interactions/hour."
        )
    lines = [
        "# Model Status",
        "",
        f"Latest run source: {source}",
        f"Latest duration: {duration_seconds} seconds",
        "",
        "## Part 1 Framing",
        "Part 1 is now framed as baseline model validation and mechanism audit, not as proof that generative agents outperform rule agents.",
        "Large differences in global movement or interaction counts are not expected from memory or generative variants alone because movement and clinical workflow are intentionally rule-constrained. Ablation should therefore be interpreted as a mechanism audit, while empirical validation should focus on whether the selected baseline variant reproduces observed F2F interaction distributions.",
        f"Current readiness color: {readiness_color}.",
        "",
        "## Traffic-Light Status",
        "| Area | Status | Reason |",
        "| --- | --- | --- |",
        "| Architecture | READY | Core logs, variants, personas, memory, and verification harness run. |",
        "| Code execution / verification | READY | `python3 main.py --verify` is the required gate before longer runs. |",
        f"| Output cleanliness | {output_status} | Durable Part 1 outputs are overwritten under `outputs/latest/`; stale duration-labelled outputs are not part of the current workflow. |",
        f"| Mechanism differentiation | {readiness['mechanism_differentiation']} | Memory and fallback counters separate expected variants. |",
        f"| Outcome differentiation | {readiness['outcome_differentiation']} | Outcome differences must appear in role-pair, topic, zone, duration, count, or missed-opportunity space. |",
        f"| Ablation differentiation | {ablation_status} | Variants leave diagnostics, but superiority is not established. |",
        f"| Empirical metric interpretability | {readiness['empirical_metric_interpretability']} | {topic_note} |",
        f"| Empirical validation | {empirical_validation_status} | Diagnostic-mode-off validation candidate is required before Part 2. |",
        f"| Topic realism | {topic_status} | Latest generative run used {topic_count} observed topic categories. |",
        f"| High-acuity behavior | {high_acuity_status} | Latest relevant run logged {high_acuity_interactions} high-acuity interactions. |",
        f"| Missed opportunities | {missed_status} | Latest generative run logged {missed} missed-opportunity episodes. |",
        "| Interviews | PARTIAL | Trace-grounded and leakage-checked, but still rule-narrative rather than true LLM interviewing. |",
        "| Design intervention experiments | NOT READY | Run paired baseline validation before comparing design interventions. |",
        "",
        "## What Works",
        "- Verification and standard ablation commands run on a laptop.",
        "- Perception, memory, and structured generative fallback now leave distinct mechanism diagnostics.",
        "- Durable Part 1 outputs are overwritten under `outputs/latest/` and current figures replace old misleading figures.",
        "- Empirical comparison uses the interaction log only; workflow events are excluded.",
        "",
        "## What Does Not Work Yet",
        "- A single run is still too fragile for empirical claims.",
        "- High-acuity behavior needs repeated smoke and paired-seed evidence.",
        "- Missed-opportunity logging needs calibration against a defensible target range.",
        "- Composite distance does not include interaction count, high-acuity behavior, missed opportunities, or HCW-HCW versus patient-facing proportions.",
        "- Mechanism activity is not evidence of empirical validity.",
        "- A visibility-sensitive Part 2 baseline is not selected yet; `traditional_rule` may fit current gates but cannot test transparency mechanisms.",
        "",
        "## Metric Audit",
        "- Composite formula: `1.5*KDE + 1.0*zone + 1.0*role_pair + 1.0*topic + 0.5*duration`.",
        "- Empirical targets: F2F shadowing KDE, zone histogram, role-pair distribution, inferred topic-by-zone distribution, and duration quantiles by role pair.",
        "- Simulated targets: simulated interactions only; workflow events are excluded.",
        f"- Diagnostic mode changes rates: {config.DIAGNOSTIC_ARRIVAL_MULTIPLIER}x arrivals and {config.DIAGNOSTIC_INTERACTION_PROBABILITY_MULTIPLIER}x interaction probability multiplier.",
        f"- Empirical window: {empirical_window_text}",
        "- Empirical observation-window timing is used for validation candidate count/rate gates; it does not change the legacy composite distance formula.",
        "",
        "## Interaction Ecology Audit",
        candidate_ecology or "No validation candidate ecology summary is available yet.",
        "The active refactor removes automatic patient-facing task logging, adds probabilistic task communication, and strengthens station/corridor HCW-HCW opportunities. This improves the ecology but does not make Part 2 ready by itself.",
        "",
        "## Scope Alignment Audit",
        scope_alignment_text,
        "",
        "## Ready For Longer Runs?",
        f"{longer_status}: 1-hour and 4-hour plausibility runs are appropriate after verification, but they are not validation.",
        "",
        "## Ready For Paired-Seed Validation?",
        (
            "PARTIAL: the paired-seed scaffold has run, but empirical-matched validation still needs stronger outcome differentiation and uncertainty reporting."
            if paired_run
            else "PARTIAL: use the paired-seed scaffold only after 1-hour diagnostics look stable."
        ),
        "",
        "## Why Paired Seeds Are Needed",
        "- A single run cannot distinguish stable differences from random variation.",
        "- Paired seeds compare variants under the same random arrivals and task stochasticity.",
        "- More seeds estimate variance and confidence intervals.",
        "- More seeds do not fix broken metrics.",
        "- More seeds do not create meaningful differences if mechanisms do not affect outcomes.",
        "- Therefore paired seeds should run only after metric interpretability is audited.",
        "",
        "## Ready For Design-Intervention Experiments?",
        "NOT READY: Part 2 should wait until baseline variant behavior is validated across paired seeds.",
        "",
        "## Baseline Recommendation",
        (
            f"Candidate baseline: `{validation_recommendation.get('recommended_variant')}` ({validation_recommendation.get('status')}). {validation_recommendation.get('rationale')}"
            if validation_recommendation.get("recommended_variant")
            else "No baseline variant is ready for Part 2 design interventions from the current validation candidate."
        ),
        "",
        "## Ready Checklist",
        "- Empirical validation: diagnostic mode OFF; empirical observation window matched or sampled; paired seeds completed; pass/warn/fail statuses reported; interaction count/rate, role-pair, zone, topic, duration, and HCW-patient shares reported; selected baseline stated; limitations stated.",
        "- Part 2 interventions: one baseline selected; baseline empirical fit at least YELLOW; interventions modify only intended spatial/design parameters; paired seeds and predeclared outcomes used.",
        "- Part 3 interviews: memories/retrievals/reflections present; interview answers cite event and memory IDs; unsupported claims flagged; outputs labelled synthetic and exploratory.",
        "",
        "## Single Next Action",
        next_action,
        "",
        "## Do Not Claim Yet",
        "- Do not claim empirical superiority from a single 1-hour or 4-hour run.",
        "- Do not treat no-LLM fallback interviews as real clinician experiences.",
        "- Do not use design-intervention results as Part 2 evidence until baseline behavior is stable across paired seeds.",
        "",
    ]
    config.MODEL_STATUS_PATH.write_text("\n".join(lines))


def _write_model_status(
    summary: Mapping[str, Mapping[str, object]],
    duration_seconds: int,
    n_runs: int,
    run_kind: str = "ablation_smoke",
) -> None:
    source = {
        "mechanism_audit": "mechanism audit, n_runs={n_runs}",
        "validation_candidate": "validation candidate, n_runs={n_runs}",
    }.get(run_kind, "ablation smoke, n_runs={n_runs}").format(n_runs=n_runs)
    write_model_status(
        source=source,
        duration_seconds=duration_seconds,
        summary=summary,
    )


def _write_outputs_readme() -> None:
    lines = [
        "# Outputs",
        "",
        "`outputs/latest/` holds the current imported reference result for local inspection. `outputs/batch/` is for local smoke/batch runs and can be cleared when a run is obsolete. `outputs/findings/` is reserved for durable presentation-ready findings only.",
        "",
        "## Current Latest Outputs",
        "- `latest/ABM_results/part1_validation_n100/`: canonical frozen Part 1 n100 reference import.",
        "",
        "## Durable Findings",
        "Write to `outputs/findings/` only after a result is good enough to show. Stale Part 1/Part 2 findings from older framing should be deleted or replaced when the baseline is reworked.",
        "",
        "## Normal Commands",
        "- `python3 main.py --verify`",
        "- `python3 main.py --mechanism-audit --duration 14400 --n-runs 1`",
        "- `python3 main.py --validation-candidate --validation-target care_area --duration 43200 --n-runs 3 --warmup-seconds 7200`",
        "- `python3 main.py --condition-smoke --duration 600`",
        "- `python3 main.py --part2-scenarios --duration 43200 --warmup-seconds 7200 --n-runs 3 --conditions baseline,cockpit_only,nursta_only,both`",
        "",
        "The normal workflow uses one rule-based perception baseline. SeniorDoctor and Part 3 generative cognition are disabled unless explicitly requested for sensitivity or future extension work.",
        "",
    ]
    config.OUTPUTS_README_PATH.write_text("\n".join(lines))


def _polygon_centroid(points: Iterable[Iterable[float]]) -> dict:
    vertices = [(float(point[0]), float(point[1])) for point in points]
    if not vertices:
        return {"x": 0.0, "y": 0.0}
    return {
        "x": float(sum(point[0] for point in vertices) / len(vertices)),
        "y": float(sum(point[1] for point in vertices) / len(vertices)),
    }


def _point_payload(point: Iterable[float]) -> dict[str, float]:
    values = list(point)
    return {"x": float(values[0]), "y": float(values[1])}


def _polygon_payload(points: Iterable[Iterable[float]]) -> list[dict[str, float]]:
    return [_point_payload(point) for point in points]


def _registry_polygon_points(record: Mapping[str, object]) -> list[tuple[float, float]]:
    points = []
    for point in record.get("polygon", []) or []:
        if isinstance(point, Mapping):
            points.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
        else:
            values = list(point)
            points.append((float(values[0]), float(values[1])))
    return points


def _normalize_care_area_registry_record(item: Mapping[str, object], environment: Environment) -> dict:
    record = dict(item)
    object_id = str(record["id"])
    if "polygon" not in record and object_id in environment.zones:
        record["polygon"] = list(environment.zones[object_id])
    polygon_points = _registry_polygon_points(record)
    centroid = record.get("centroid")
    if isinstance(centroid, Mapping):
        centroid_payload = {
            "x": float(centroid.get("x", 0.0)),
            "y": float(centroid.get("y", 0.0)),
        }
    elif centroid is not None:
        centroid_payload = _point_payload(centroid)
    else:
        centroid_payload = _polygon_centroid(polygon_points)
    return {
        "id": object_id,
        "display_label": str(record.get("display_label", object_id)),
        "object_type": str(record.get("object_type", "unknown")),
        "active_scope_modes": list(record.get("active_scope_modes", [])),
        "allowed_roles": list(record.get("allowed_roles", [])),
        "preferred_roles": list(record.get("preferred_roles", [])),
        "supported_task_types": list(record.get("supported_task_types", [])),
        "supports_dwell": bool(record.get("supports_dwell", False)),
        "supports_routing": bool(record.get("supports_routing", False)),
        "supports_patient_assignment": bool(record.get("supports_patient_assignment", False)),
        "supports_patient_facing_care": bool(record.get("supports_patient_facing_care", False)),
        "supports_hcw_hcw_coordination": bool(record.get("supports_hcw_hcw_coordination", False)),
        "linked_empirical_cluster_evidence": str(record.get("linked_empirical_cluster_evidence", "")),
        "polygon": _polygon_payload(polygon_points),
        "centroid": centroid_payload,
        "unity_export_note": str(
            record.get(
                "unity_export_note",
                "Can be translated to a Unity zone, POI, trigger volume, or navmesh target.",
            )
        ),
    }


def _find_xy_columns(dataframe) -> tuple[Optional[str], Optional[str]]:
    x_candidates = ["x_shadowing", "x", "X"]
    y_candidates = ["y_shadowing", "y", "Y"]
    x_column = next((column for column in x_candidates if column in dataframe.columns), None)
    y_column = next((column for column in y_candidates if column in dataframe.columns), None)
    return x_column, y_column


def _collect_empirical_scope_points(limit: int = 2000) -> list[dict]:
    dataframe = load_scope_filtered_f2f_dataframe("care_area")
    x_column, y_column = _find_xy_columns(dataframe)
    if x_column is None or y_column is None:
        return []
    points = []
    for row in dataframe[[x_column, y_column, "zone_id"]].dropna(subset=[x_column, y_column]).head(limit).to_dict("records"):
        points.append({
            "x": float(row[x_column]),
            "y": float(row[y_column]),
            "zone_id": str(row.get("zone_id", "unknown")),
        })
    return points


def _qa_geometry_issues(registry: Iterable[Mapping[str, object]]) -> tuple[list[str], list[str]]:
    malformed = []
    out_of_range = []
    for record in registry:
        object_id = str(record.get("id", "unknown"))
        polygon = _registry_polygon_points(record)
        centroid = record.get("centroid", {})
        if len(polygon) < 3:
            malformed.append(f"{object_id}: polygon has fewer than 3 vertices")
            continue
        values = [coordinate for point in polygon for coordinate in point]
        if isinstance(centroid, Mapping):
            values.extend([float(centroid.get("x", 0.0)), float(centroid.get("y", 0.0))])
        if any(not np.isfinite(value) for value in values):
            malformed.append(f"{object_id}: non-finite coordinate detected")
        if object_id.startswith("BED_"):
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            if min(xs) < -20 or max(xs) > 12 or min(ys) < -16 or max(ys) > 12:
                out_of_range.append(f"{object_id}: bed marker falls outside expected QA bounds")
    return malformed, out_of_range


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _bed_cluster_qa(
    registry: Iterable[Mapping[str, object]],
    empirical_points: list[dict],
) -> dict[str, dict]:
    intended_clusters = {
        "BED_6": "lower-right empirical cluster near CORR05",
        "BED_7": "lower-right empirical cluster near CORR05",
        "BED_8": "above/left-of-NUROPE high-acuity bed location",
        "BED_9": "lower-left NUROPE-side care-area cluster",
        "BED_10": "lower-left NUROPE-side care-area cluster",
    }
    registry_by_id = {str(record.get("id")): record for record in registry}
    qa = {}
    for object_id, intended_cluster in intended_clusters.items():
        record = registry_by_id.get(object_id, {})
        polygon = _registry_polygon_points(record)
        area = _polygon_area(polygon)
        centroid = record.get("centroid", {})
        centroid_x = float(centroid.get("x", 0.0)) if isinstance(centroid, Mapping) else 0.0
        centroid_y = float(centroid.get("y", 0.0)) if isinstance(centroid, Mapping) else 0.0
        empirical_count = sum(
            1
            for point in empirical_points
            if ((float(point["x"]) - centroid_x) ** 2 + (float(point["y"]) - centroid_y) ** 2)
            <= config.BED_PROXIMITY_METERS ** 2
        )
        notes = []
        if empirical_count > 0:
            notes.append("overlaps mapped empirical points for the intended cluster")
        else:
            notes.append("no empirical points are currently within the bed-proximity radius")
        if area > 25.0:
            notes.append("bed marker footprint is broad; visually review room boundaries separately")
        elif area < 6.0:
            notes.append("bed marker footprint is small; check that it does not miss nearby cluster mass")
        else:
            notes.append("bed marker size is within the current QA heuristic range")
        qa[object_id] = {
            "intended_cluster": intended_cluster,
            "empirical_points_within_bed_radius": int(empirical_count),
            "marker_area_square_coordinate_units": float(area),
            "visual_overlap_assessment": "present" if empirical_count > 0 else "not_detected_from_current_mapping",
            "qa_notes": notes,
        }
    return qa


def _draw_floorplan_walls(axis, environment: Environment) -> None:
    for wall_start, wall_end in environment.walls:
        axis.plot(
            [wall_start[0], wall_end[0]],
            [wall_start[1], wall_end[1]],
            color="#2F2F2F",
            linewidth=1.35,
            zorder=1,
        )


def _write_care_area_floorplan_overlay(
    registry: list[dict],
    payload: Mapping[str, object],
    simulated_points: Optional[list[dict]] = None,
) -> None:
    plotted_ids = [
        "COCPIT",
        "NURSTA",
        "NUROPE",
        "CORR01",
        "CORR02",
        "CORR03",
        "CORR04",
        "CORR05",
        "CORR06",
        "CORR07",
    ]
    environment = Environment(config.WALL_POSITIONS_PATH, config.ZONE_BOUNDARIES_PATH)
    registry_by_id = {str(record["id"]): record for record in registry}
    plotted_objects = []
    for object_id in plotted_ids:
        if object_id in registry_by_id:
            plotted_objects.append(registry_by_id[object_id])
        elif object_id in environment.zones:
            polygon = list(environment.zones[object_id])
            plotted_objects.append({
                "id": object_id,
                "display_label": object_id,
                "object_type": "named_zone",
                "active_scope_modes": [config.DEFAULT_SCOPE_MODE],
                "polygon": _polygon_payload(polygon),
                "centroid": _polygon_centroid(polygon),
            })
    for bed_index, (bed_x, bed_y) in enumerate(config.BED_POSITIONS, start=1):
        radius = 0.45
        plotted_objects.append({
            "id": f"BED_{bed_index}",
            "display_label": f"Bed {bed_index}",
            "object_type": "bed",
            "active_scope_modes": [config.DEFAULT_SCOPE_MODE],
            "polygon": _polygon_payload([
                (bed_x - radius, bed_y - radius),
                (bed_x + radius, bed_y - radius),
                (bed_x + radius, bed_y + radius),
                (bed_x - radius, bed_y + radius),
            ]),
            "centroid": {"x": float(bed_x), "y": float(bed_y)},
        })

    empirical_points = _collect_empirical_scope_points()
    simulated_points = simulated_points or []
    malformed, out_of_range = _qa_geometry_issues(plotted_objects)
    bed_qa = _bed_cluster_qa(plotted_objects, empirical_points)

    fig, ax = plt.subplots(figsize=(14, 9))
    _draw_floorplan_walls(ax, environment)
    wall_proxy = plt.Line2D([0], [0], color="#2F2F2F", linewidth=1.35, label="Floorplan walls")
    named_label_used = False
    care_label_used = False
    centroid_label_used = False
    for record in plotted_objects:
        polygon = _registry_polygon_points(record)
        if not polygon:
            continue
        xs = [point[0] for point in polygon] + [polygon[0][0]]
        ys = [point[1] for point in polygon] + [polygon[0][1]]
        is_bed = str(record.get("id")).startswith("BED_")
        color = "#111111" if is_bed else "#555555"
        label = "Bed marker" if is_bed and not care_label_used else (
            "Named zone" if not is_bed and not named_label_used else None
        )
        ax.plot(xs, ys, color=color, linewidth=1.8, linestyle="--" if not is_bed else "-", label=label, zorder=4)
        if is_bed:
            care_label_used = True
        else:
            named_label_used = True
        centroid = record.get("centroid", {})
        if isinstance(centroid, Mapping):
            cx = float(centroid.get("x", 0.0))
            cy = float(centroid.get("y", 0.0))
            ax.scatter([cx], [cy], marker="x", color=color, s=55, label="Centroid" if not centroid_label_used else None)
            centroid_label_used = True
            ax.text(cx, cy, str(record.get("id")), fontsize=8, ha="left", va="bottom")

    if empirical_points:
        ax.scatter(
            [point["x"] for point in empirical_points],
            [point["y"] for point in empirical_points],
            s=14,
            color="#6F6F6F",
            alpha=0.42,
            label="Empirical F2F points",
            zorder=3,
        )
    if simulated_points:
        ax.scatter(
            [point["x"] for point in simulated_points],
            [point["y"] for point in simulated_points],
            s=24,
            color="#CC79A7",
            alpha=0.75,
            marker="^",
            label="Simulated interaction points",
            zorder=5,
        )

    min_x, max_x, min_y, max_y = environment.plot_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title("Care-area floorplan overlay QA (model-development, not a finalized finding)")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([wall_proxy] + handles, ["Floorplan walls"] + labels, loc="upper left", fontsize=8)
    fig.tight_layout()
    config.CARE_AREA_FLOORPLAN_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.CARE_AREA_FLOORPLAN_OVERLAY_PATH, dpi=200)
    plt.close(fig)

    metadata_objects = [
        {
            "id": str(record.get("id")),
            "display_label": str(record.get("display_label", record.get("id"))),
            "object_type": str(record.get("object_type", "unknown")),
            "polygon": record.get("polygon", []),
            "centroid": record.get("centroid", {}),
            "active_scope_modes": list(record.get("active_scope_modes", [])),
        }
        for record in plotted_objects
    ]
    metadata = {
        "purpose": "Floorplan-wall spatial QA for the active care-area simulation; not a finalized finding.",
        "floorplan_geometry_source": str(config.WALL_POSITIONS_PATH),
        "zone_geometry_source": str(config.ZONE_BOUNDARIES_PATH),
        "uses_same_coordinate_system_as_empirical_map": True,
        "wall_segment_count": len(environment.walls),
        "plotted_object_ids": [record["id"] for record in metadata_objects],
        "plotted_objects": metadata_objects,
        "empirical_points_plotted": bool(empirical_points),
        "empirical_point_count": len(empirical_points),
        "simulated_points_plotted": bool(simulated_points),
        "simulated_point_count": len(simulated_points),
        "source_validation_target": payload.get("validation_target"),
        "source_scope_mode": payload.get("scope_mode"),
        "geometry_schema_issues": malformed,
        "bed_coordinate_range_warnings": out_of_range,
        "bed_cluster_qa": bed_qa,
    }
    config.CARE_AREA_FLOORPLAN_OVERLAY_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    lines = [
        "# Care-Area Floorplan Overlay QA",
        "",
        "Status: model-development QA only, not a finalized finding.",
        "",
        f"- Figure: `{config.CARE_AREA_FLOORPLAN_OVERLAY_PATH.name}`",
        f"- Wall geometry source: `{config.WALL_POSITIONS_PATH}`",
        "- Uses same coordinate system as empirical map: yes",
        f"- Objects plotted: {', '.join(metadata['plotted_object_ids'])}",
        f"- Empirical points included: {'yes' if empirical_points else 'no'} ({len(empirical_points)})",
        f"- Simulated points included: {'yes' if simulated_points else 'no'} ({len(simulated_points)})",
        f"- Malformed geometry: {'none detected' if not malformed else '; '.join(malformed)}",
        f"- Bed coordinate range warnings: {'none detected' if not out_of_range else '; '.join(out_of_range)}",
        "",
        "## Bed Location QA",
        "| Bed | Intended empirical cluster | Empirical points within bed-proximity radius | QA note |",
        "| --- | --- | ---: | --- |",
    ]
    for object_id, qa in bed_qa.items():
        lines.append(
            f"| {object_id} | {qa['intended_cluster']} | {qa['empirical_points_within_bed_radius']} | "
            f"{'; '.join(qa['qa_notes'])} |"
        )
    lines.extend([
        "",
        "Use this floorplan overlay to visually check whether new bed locations sit in the intended empirical cluster areas before any further behavior changes.",
        "If simulated points are absent, the latest saved validation summary did not retain per-event coordinates; rerunning the validation command regenerates this overlay with simulated points.",
        "",
    ])
    config.CARE_AREA_FLOORPLAN_OVERLAY_SUMMARY_PATH.write_text("\n".join(lines))


def _write_care_area_outputs(
    payload: Mapping[str, object],
    simulated_points: Optional[list[dict]] = None,
) -> None:
    registry: list[dict] = []
    _write_care_area_floorplan_overlay(registry, payload, simulated_points=simulated_points)

    model_results = payload.get("model_results", payload.get("variants", {}))
    row = payload.get("baseline_result", {})
    if not row and isinstance(model_results, Mapping):
        row = next(iter(model_results.values()), {})
    empirical_zones = payload.get("validation_target_metadata", {}).get("zone_distribution", {})
    empirical_roles = payload.get("validation_target_metadata", {}).get("role_pair_distribution", {})
    zone_distribution = row.get("zone_distribution", {}) if isinstance(row, Mapping) else {}
    role_distribution = row.get("role_pair_distribution", {}) if isinstance(row, Mapping) else {}
    fit = row.get("validation_fit", {}) if isinstance(row, Mapping) else {}
    station_ids = ["COCPIT", "NURSTA", "NUROPE"]

    summary = {
        "purpose": "Provisional care-area roster summary; not a finalized finding.",
        "scope_mode": payload.get("scope_mode"),
        "validation_target": payload.get("validation_target"),
        "duration_seconds": payload.get("duration_seconds"),
        "n_runs": payload.get("n_runs"),
        "staff_roster": {
            "staff_counts": config.STAFF_COUNTS,
            "nurse_home_zone_ids": config.NURSE_HOME_ZONE_IDS,
            "bed_positions": config.BED_POSITIONS,
            "nurse_bed_coverage": config.NURSE_BED_COVERAGE,
            "high_acuity_reserved_bed_indices": sorted(config.HIGH_ACUITY_RESERVED_BED_INDICES),
        },
        "validation_status": fit.get("overall_status"),
        "f2f_per_hour": {
            "empirical": fit.get("empirical_f2f_per_hour"),
            "simulated": fit.get("simulated_f2f_per_hour"),
        },
        "hcw_hcw_share": {
            "empirical": fit.get("empirical_hcw_hcw_share"),
            "simulated": row.get("outcome_metrics", {}).get("hcw_hcw_share", 0.0),
        },
        "patient_facing_share": {
            "empirical": fit.get("empirical_patient_facing_share"),
            "simulated": row.get("outcome_metrics", {}).get("patient_facing_share", 0.0),
        },
        "station_zone_shares": {
            zone_id: {
                "empirical": float(empirical_zones.get(zone_id, 0.0)),
                "simulated": float(zone_distribution.get(zone_id, 0.0)),
            }
            for zone_id in station_ids
        },
        "role_pair_shares": {
            role_pair: {
                "empirical": float(empirical_roles.get(role_pair, 0.0)),
                "simulated": float(role_distribution.get(role_pair, 0.0)),
            }
            for role_pair in ["Nurse|Nurse", "CoordinationNurse|Nurse", "Doctor|Nurse"]
        },
        "station_hcw_hcw_interaction_count": int(row.get("station_hcw_hcw_interaction_count", 0)),
        "interpretation": (
            "The simulation now uses the single 10-bed care-area setup. Bed 8 is reserved for rare ESI 1/2 cases; "
            "support/admin/triage agents and LLM destination choice remain out of scope."
        ),
    }
    config.CARE_AREA_ROSTER_SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Care-Area Roster Summary",
        "",
        "Status: provisional model-development output, not a finalized finding.",
        "",
        f"- Scope mode: `{summary['scope_mode']}`",
        f"- Validation target: `{summary['validation_target']}`",
        f"- Validation status: `{summary['validation_status']}`",
        f"- F2F/hour: empirical {float(summary['f2f_per_hour']['empirical'] or 0.0):.2f}, simulated {float(summary['f2f_per_hour']['simulated'] or 0.0):.2f}",
        f"- HCW-HCW share: empirical {float(summary['hcw_hcw_share']['empirical'] or 0.0):.3f}, simulated {float(summary['hcw_hcw_share']['simulated'] or 0.0):.3f}",
        f"- Patient-facing share: empirical {float(summary['patient_facing_share']['empirical'] or 0.0):.3f}, simulated {float(summary['patient_facing_share']['simulated'] or 0.0):.3f}",
        "",
        "## Station Shares",
        "| Zone | Empirical | Simulated |",
        "| --- | ---: | ---: |",
    ]
    for zone_id, shares in summary["station_zone_shares"].items():
        lines.append(f"| {zone_id} | {float(shares['empirical']):.3f} | {float(shares['simulated']):.3f} |")
    lines.extend([
        "",
        "## Role-Pair Shares",
        "| Role pair | Empirical | Simulated |",
        "| --- | ---: | ---: |",
    ])
    for role_pair, shares in summary["role_pair_shares"].items():
        lines.append(f"| {role_pair} | {float(shares['empirical']):.3f} | {float(shares['simulated']):.3f} |")
    lines.extend([
        "",
        "## Interpretation",
        summary["interpretation"],
        "",
    ])
    config.CARE_AREA_ROSTER_SUMMARY_MD_PATH.write_text("\n".join(lines))


def _write_perception_mechanism_outputs(summary: Mapping[str, object]) -> None:
    row = next(iter(summary.values()), {})
    diagnostics = dict(row.get("diagnostics", {}))
    diagnostics.update(dict(row.get("perception_diagnostics", {})))
    examples = list(diagnostics.get("example_event_traces", []))
    interaction_count = int(row.get("interaction_count", row.get("outcome_metrics", {}).get("interaction_count", 0)))
    outcome_metrics = row.get("outcome_metrics", {}) if isinstance(row.get("outcome_metrics", {}), Mapping) else {}
    interaction_type_counts = row.get("interaction_type_counts", {}) if isinstance(row.get("interaction_type_counts", {}), Mapping) else {}
    station_count = int(round(float(outcome_metrics.get("station_or_desk_interaction_share", 0.0)) * interaction_count))
    corridor_count = int(round(float(outcome_metrics.get("corridor_interaction_share", 0.0)) * interaction_count))
    bedside_count = int(round(float(outcome_metrics.get("bedside_or_patient_room_interaction_share", 0.0)) * interaction_count))
    handoff_count = sum(
        int(count)
        for interaction_type, count in interaction_type_counts.items()
        if "handoff" in str(interaction_type) or "alert" in str(interaction_type)
    )
    perception_checked_count = int(diagnostics.get("perception_checked_interaction_count", 0))
    unperceived_count = int(diagnostics.get("interactions_logged_without_perception_check", 0))
    perception_audit = {
        "all_logged_interactions_are_perception_checked": bool(
            interaction_count == perception_checked_count and unperceived_count == 0
        ),
        "interactions_logged_without_perception_check": unperceived_count,
        "interaction_count": interaction_count,
        "perception_checked_interaction_count": perception_checked_count,
        "staff_staff_interactions": int(diagnostics.get("hcw_hcw_interactions", 0)),
        "patient_facing_interactions": int(diagnostics.get("patient_facing_interactions", 0)),
        "station_interactions": station_count,
        "corridor_interactions": corridor_count,
        "bedside_interactions": bedside_count,
        "handoff_interactions": handoff_count,
        "nonproximate_logged_interaction_count": int(diagnostics.get("nonproximate_logged_interaction_count", 0)),
        "rejected_not_perceived": int(
            diagnostics.get("rejected_not_perceived", 0)
            + diagnostics.get("rejected_by_visibility", 0)
            + diagnostics.get("rejected_by_fov", 0)
        ),
        "rejected_protected_state": int(diagnostics.get("rejected_by_protected_state", 0)),
        "rejected_cooldown": int(diagnostics.get("rejected_by_cooldown", 0)),
        "rejected_salience": int(diagnostics.get("rejected_salience", 0) + diagnostics.get("rejected_by_salience_threshold", 0)),
    }
    config.PERCEPTION_AUDIT_JSON_PATH.write_text(json.dumps(perception_audit, indent=2))
    config.PERCEPTION_AUDIT_MD_PATH.write_text(
        "\n".join([
            "# Perception Audit",
            "",
            "Every validation-counted F2F interaction should be a perceived communication episode.",
            "",
            f"- All logged interactions perception-checked: {str(perception_audit['all_logged_interactions_are_perception_checked']).lower()}",
            f"- Interactions logged without perception check: {perception_audit['interactions_logged_without_perception_check']}",
            f"- Staff-staff interactions: {perception_audit['staff_staff_interactions']}",
            f"- Patient-facing interactions: {perception_audit['patient_facing_interactions']}",
            f"- Station interactions: {perception_audit['station_interactions']}",
            f"- Corridor interactions: {perception_audit['corridor_interactions']}",
            f"- Bedside interactions: {perception_audit['bedside_interactions']}",
            f"- Handoff interactions: {perception_audit['handoff_interactions']}",
            f"- Non-proximate logged staff interactions: {perception_audit['nonproximate_logged_interaction_count']}",
            f"- Rejected not perceived: {perception_audit['rejected_not_perceived']}",
            f"- Rejected protected state: {perception_audit['rejected_protected_state']}",
            f"- Rejected cooldown: {perception_audit['rejected_cooldown']}",
            f"- Rejected salience: {perception_audit['rejected_salience']}",
            "",
        ])
    )
    audit = {
        "perception_based_baseline_enabled": diagnostics.get("perception_based_baseline_enabled", False),
        "legacy_proximity_mode_enabled": diagnostics.get("legacy_proximity_mode_enabled", False),
        "raw_visible_staff_percepts_count": diagnostics.get("raw_visible_staff_percepts_count", 0),
        "eligible_visible_staff_opportunity_count": diagnostics.get("eligible_visible_staff_opportunity_count", 0),
        "selected_visible_staff_opportunity_count": diagnostics.get("selected_visible_staff_opportunity_count", 0),
        "perceived_staff_interaction_count": diagnostics.get("perceived_staff_interaction_count", 0),
        "nonproximate_logged_interaction_count": diagnostics.get("nonproximate_logged_interaction_count", 0),
        "approach_intent_count": diagnostics.get("approach_intent_count", 0),
        "approach_success_count": diagnostics.get("approach_success_count", 0),
        "approach_failure_count": diagnostics.get("approach_failure_count", 0),
        "approach_failure_reasons": diagnostics.get("approach_failure_reasons", {}),
        "route_override_count": diagnostics.get("route_override_count", 0),
        "route_resume_count": diagnostics.get("route_resume_count", 0),
        "target_route_override_count": diagnostics.get("target_route_override_count", 0),
        "target_receive_pause_count": diagnostics.get("target_receive_pause_count", 0),
        "target_continued_route_count": diagnostics.get("target_continued_route_count", 0),
        "post_task_station_check_count": diagnostics.get("post_task_station_check_count", 0),
        "post_task_station_check_completed_count": diagnostics.get("post_task_station_check_completed_count", 0),
        "station_check_route_failures": diagnostics.get("station_check_route_failures", 0),
        "post_task_station_check_by_role": diagnostics.get("post_task_station_check_by_role", {}),
        "post_task_station_check_by_station": diagnostics.get("post_task_station_check_by_station", {}),
        "initial_opportunity_distance_distribution": diagnostics.get("initial_opportunity_distance_distribution", {}),
        "final_logged_distance_distribution": diagnostics.get("final_logged_distance_distribution", {}),
        "interaction_locations_are_actual_contact_coordinates": diagnostics.get("interaction_locations_are_actual_contact_coordinates", False),
        "mean_initiation_distance_m": diagnostics.get("mean_initiation_distance_m", 0.0),
        "median_initiation_distance_m": diagnostics.get("median_initiation_distance_m", 0.0),
        "max_initiation_distance_m": diagnostics.get("max_initiation_distance_m", 0.0),
        "stationary_initiated_to_moving_count": diagnostics.get("stationary_initiated_to_moving_count", 0),
        "rejected_by_visibility": diagnostics.get("rejected_by_visibility", 0),
        "rejected_by_fov": diagnostics.get("rejected_by_fov", 0),
        "rejected_by_salience_threshold": diagnostics.get("rejected_by_salience_threshold", 0),
        "rejected_by_attention_capacity": diagnostics.get("rejected_by_attention_capacity", 0),
        "rejected_by_protected_state": diagnostics.get("rejected_by_protected_state", 0),
        "rejected_by_cooldown": diagnostics.get("rejected_by_cooldown", 0),
        "protected_state_suppression_count": diagnostics.get("protected_state_suppression_count", 0),
        "essential_questions": {
            "agents_perceive_visible_staff": int(diagnostics.get("raw_visible_staff_percepts_count", 0)) > 0,
            "eligible_visible_staff_opportunities_exist": int(diagnostics.get("eligible_visible_staff_opportunity_count", 0)) > 0,
            "staff_opportunities_selected": int(diagnostics.get("selected_visible_staff_opportunity_count", 0)) > 0,
            "nonproximate_interactions_logged": int(diagnostics.get("nonproximate_logged_interaction_count", 0)) > 0,
            "approach_intents_created": int(diagnostics.get("approach_intent_count", 0)) > 0,
            "approach_intents_completed_at_contact": int(diagnostics.get("approach_success_count", 0)) > 0,
            "all_opportunistic_logs_are_close_contact": int(diagnostics.get("nonproximate_logged_interaction_count", 0)) == 0,
            "stationary_to_moving_initiation_seen": int(diagnostics.get("stationary_initiated_to_moving_count", 0)) > 0,
            "protected_states_suppress_opportunities": int(diagnostics.get("protected_state_suppression_count", 0)) > 0
            or int(diagnostics.get("rejected_by_protected_state", 0)) > 0,
            "cooldowns_suppress_repeats": int(diagnostics.get("rejected_by_cooldown", 0)) > 0,
        },
        "example_event_traces": examples,
    }
    config.PERCEPTION_MECHANISM_AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2))

    lines = [
        "# Perception Mechanism Audit",
        "",
        "This is a development audit of the single rule-based perception baseline. It is not a Part 2 finding.",
        "",
        "| Question | Result |",
        "|---|---|",
    ]
    for question, result in audit["essential_questions"].items():
        lines.append(f"| {question.replace('_', ' ')} | {'yes' if result else 'no'} |")
    lines.extend([
        "",
        "## Counters",
        f"- Raw visible staff percepts: {audit['raw_visible_staff_percepts_count']}",
        f"- Eligible visible staff opportunities: {audit['eligible_visible_staff_opportunity_count']}",
        f"- Selected visible staff opportunities: {audit['selected_visible_staff_opportunity_count']}",
        f"- Perceived staff interactions: {audit['perceived_staff_interaction_count']}",
        f"- Approach intents: {audit['approach_intent_count']}",
        f"- Approach successes: {audit['approach_success_count']}",
        f"- Approach failures: {audit['approach_failure_count']}",
        f"- Non-proximate logged interactions: {audit['nonproximate_logged_interaction_count']}",
        f"- Route overrides: {audit['route_override_count']}",
        f"- Target route overrides: {audit['target_route_override_count']}",
        f"- Targets continuing prior route: {audit['target_continued_route_count']}",
        f"- Route resumes: {audit['route_resume_count']}",
        f"- Mean initiation distance: {float(audit['mean_initiation_distance_m']):.2f} m",
        f"- Max initiation distance: {float(audit['max_initiation_distance_m']):.2f} m",
        f"- Protected-state suppressions: {audit['protected_state_suppression_count']}",
        f"- Cooldown rejections: {audit['rejected_by_cooldown']}",
        "",
        "## Example Traces",
    ])
    for example in examples[:8]:
        lines.append(
            f"- t={example.get('timestep')}: {example.get('initiator')} ({example.get('initiator_role')}, "
            f"{example.get('initiator_mode')}) saw {example.get('target')} ({example.get('target_role')}, "
            f"{example.get('target_mode')}) at {float(example.get('initial_distance_m', 0.0)):.2f} m; "
            f"salience={example.get('salience_score')}; type={example.get('interaction_type')}; "
            f"topic={example.get('topic')}; location={example.get('location')}."
        )
    config.PERCEPTION_MECHANISM_AUDIT_MD_PATH.write_text("\n".join(lines))

    example_lines = ["# Perception Event Examples", ""]
    for example in examples[: max(5, min(len(examples), 10))]:
        example_lines.extend([
            f"## t={example.get('timestep')} {example.get('interaction_type')}",
            f"- Initiator: {example.get('initiator')} ({example.get('initiator_role')}, {example.get('initiator_mode')})",
            f"- Target: {example.get('target')} ({example.get('target_role')}, {example.get('target_mode')})",
            f"- Initial distance: {float(example.get('initial_distance_m', 0.0)):.2f} m",
            f"- Visible / mutual visible: {example.get('visible')} / {example.get('mutual_visible')}",
            f"- Salience score: {example.get('salience_score')}",
            f"- Reason selected: {example.get('reason_selected')}",
            f"- Approach/pause: {example.get('approach_or_pause_occurred')}",
            f"- Topic: {example.get('topic')}",
            f"- Location: {example.get('location')}",
            f"- Cooldown state: {example.get('cooldown_state')}",
            "",
        ])
    config.PERCEPTION_EVENT_EXAMPLES_PATH.write_text("\n".join(example_lines))


def _public_baseline_payload(payload: Mapping[str, object]) -> dict:
    """Remove internal compatibility labels from user-facing run outputs."""

    legacy_values = {
        "traditional_rule",
        "perception_rule",
        "memory_rule",
        "generative_interaction",
        "generative_rule",
    }
    legacy_keys = {
        "model_variant",
        "variant_classification",
        "outcome_deltas_vs_traditional",
        "role_pair_jsd_vs_traditional",
        "topic_jsd_vs_traditional",
        "zone_jsd_vs_traditional",
        "outcome_difference_signal_count",
    }

    def clean(value):
        if isinstance(value, Mapping):
            return {
                str(key): clean(item)
                for key, item in value.items()
                if str(key) not in legacy_keys
                and "variant" not in str(key).lower()
                and "traditional" not in str(key).lower()
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str) and value in legacy_values:
            return config.BASELINE_MODEL_ID
        return value

    public = clean(payload)
    public["model_id"] = config.BASELINE_MODEL_ID
    public["display_label"] = "rule-based perception baseline"
    return public


def run_ablation_smoke(
    duration_seconds: Optional[int] = None,
    n_runs: Optional[int] = None,
    condition_name: str = "baseline",
    variants: Optional[Iterable[str]] = None,
    *,
    run_kind: str = "ablation_smoke",
    diagnostic_mode: Optional[bool] = None,
    validation_target: str = "care_area",
    scope_mode: str = config.DEFAULT_SCOPE_MODE,
    warmup_seconds: int = 0,
    matched_empirical_windows: bool = False,
    enable_senior_doctor_oversight: Optional[bool] = None,
    scenario_mode: str = config.DEFAULT_SCENARIO_MODE,
    scenario_start_hour: int = config.DEFAULT_SCENARIO_START_HOUR,
) -> dict:
    if validation_target not in VALIDATION_TARGETS:
        raise ValueError(f"validation_target must be one of {sorted(VALIDATION_TARGETS)}")
    if scope_mode not in config.SCOPE_MODES:
        raise ValueError(f"scope_mode must be one of {config.SCOPE_MODES}")
    duration_seconds = config.SMOKE_DURATION_SECONDS if duration_seconds is None else duration_seconds
    n_runs = config.SMOKE_N_RUNS if n_runs is None else n_runs
    warmup_seconds = max(0, min(int(warmup_seconds), int(duration_seconds)))
    variants = list(config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS if variants is None else variants)
    run_purpose = "baseline_validation" if run_kind == "validation_candidate" else run_kind
    empirical = compute_empirical_summary(validation_target=validation_target).as_dict()
    empirical_window = describe_empirical_observation_window(write_output=(validation_target == "care_area"), validation_target=validation_target)
    validation_target_metadata = _validation_target_metadata(validation_target, empirical, empirical_window)
    summary = {}
    metric_rows = []
    diagnostic_rows = []
    plot_simulations = {}
    per_run_records_by_variant = {}
    qa_simulated_points_by_variant = {}
    validation_audit = _validation_metric_audit(empirical)
    diagnostic_updates = {}
    active_diagnostic_mode = config.ABLATION_DIAGNOSTIC_MODE if diagnostic_mode is None else bool(diagnostic_mode)
    if active_diagnostic_mode:
        diagnostic_updates = {
            "ARRIVAL_RATE_PER_SECOND": config.ARRIVAL_RATE_PER_SECOND * config.DIAGNOSTIC_ARRIVAL_MULTIPLIER,
            "INITIAL_PATIENT_COUNT": config.DIAGNOSTIC_INITIAL_PATIENT_COUNT,
        }

    for variant in variants:
        run_summaries = []
        all_interactions = []
        all_full_run_interactions = []
        all_missed_opportunities = []
        full_run_interactions_by_run = []
        variant_distances = []
        full_run_distance_records = []
        workflow_event_count = 0
        missed_opportunity_count = 0
        diagnostic_accumulator: Counter = Counter()
        llm_metric_accumulator: Counter = Counter()
        workflow_task_counter: Counter = Counter()
        patient_facing_task_attendance_accumulator: Counter = Counter()
        patient_facing_validation_interaction_accumulator: Counter = Counter()
        perception_event_examples: list[dict] = []
        per_run_records = []
        distance_component_records = []
        workflow_health_records = []
        bed_care_flow_records = []
        for run_index in range(n_runs):
            with _temporary_config_updates(diagnostic_updates):
                simulation = Simulation(
                    random_seed=config.RUN_BASE_SEED + run_index,
                    condition_name=condition_name,
                    interaction_backend=(
                        "structured_generative_fallback"
                        if variant == "generative_interaction"
                        else "rule_stub"
                    ),
                    persona_source=config.PERSONA_SOURCE,
                    model_variant=variant,
                    scope_mode=scope_mode,
                    enable_senior_doctor_oversight=enable_senior_doctor_oversight,
                    scenario_mode=scenario_mode,
                    scenario_start_hour=scenario_start_hour,
                )
                simulation.interaction_probability_multiplier = (
                    config.DIAGNOSTIC_INTERACTION_PROBABILITY_MULTIPLIER
                    if active_diagnostic_mode
                    else 1.0
                )
                while simulation.timestep < duration_seconds:
                    simulation.step()
                if n_runs == 1:
                    simulation.save_trace_outputs()
            run_full_interactions = list(simulation.interaction_log)
            run_interactions = [
                event for event in run_full_interactions
                if int(event.get("timestamp", event.get("timestep", 0))) >= warmup_seconds
            ]
            run_missed_opportunities = [
                event for event in simulation.missed_opportunity_log
                if int(event.get("timestamp", event.get("timestep", 0))) >= warmup_seconds
            ]
            run_summary = compute_summary_statistics(run_interactions, simulation)
            full_run_summary = compute_summary_statistics(run_full_interactions, simulation)
            workflow_health = simulation.workflow_health_summary(warmup_seconds=warmup_seconds)
            workflow_health_records.append(workflow_health)
            bed_care_flow_records.append(_bed_care_flow_summary(simulation))
            distance_components = compute_distance_debug(run_summary, empirical)
            full_run_distance_components = compute_distance_debug(full_run_summary, empirical)
            run_summary["distance_components"] = distance_components
            full_run_summary["distance_components"] = full_run_distance_components
            distance_component_records.append(distance_components)
            full_run_distance_records.append(full_run_distance_components)
            run_summaries.append(run_summary)
            all_interactions.extend(run_interactions)
            all_missed_opportunities.extend(run_missed_opportunities)
            all_full_run_interactions.extend(run_full_interactions)
            full_run_interactions_by_run.append(run_full_interactions)
            workflow_event_count += len(simulation.workflow_event_log)
            workflow_task_counter.update(
                str(event.get("task_name", "none")) for event in simulation.workflow_event_log
            )
            patient_facing_task_attendance_accumulator.update(
                simulation.patient_facing_task_attendance_counts
            )
            patient_facing_validation_interaction_accumulator.update(
                simulation.patient_facing_validation_interaction_counts
            )
            run_missed_count = len(simulation.missed_opportunity_log)
            missed_opportunity_count += run_missed_count
            diagnostic_accumulator.update(simulation.ablation_diagnostics)
            for role, zone_counter in simulation.zone_dwell_by_role.items():
                for zone_id, seconds in zone_counter.items():
                    diagnostic_accumulator[f"zone_dwell_role_{role}_zone_{zone_id}"] += int(seconds)
            perception_event_examples.extend(simulation.perception_event_examples)
            run_llm_metrics = {
                key: int(value)
                for key, value in simulation.llm_metrics().items()
                if isinstance(value, (int, float))
            }
            llm_metric_accumulator.update(run_llm_metrics)
            variant_distances.append(distance_components["weighted_total"])
            run_outcomes = _outcome_metrics(run_interactions, run_missed_count)
            per_run_records.append({
                "seed": config.RUN_BASE_SEED + run_index,
                "distance_to_empirical": float(distance_components["weighted_total"]),
                "interaction_count": int(run_outcomes["interaction_count"]),
                "missed_opportunity_episodes": int(run_outcomes["missed_opportunity_episodes"]),
                "memory_effects_applied": int(run_llm_metrics.get("memory_effects_applied", 0)),
                "decision_flipped_by_memory": int(run_llm_metrics.get("decision_flipped_by_memory", 0)),
                "topic_changed_by_memory": int(run_llm_metrics.get("topic_changed_by_memory", 0)),
                "backend_fallbacks": int(run_llm_metrics.get("llm_fallbacks", 0)),
                "mean_duration": float(run_outcomes["mean_duration"]),
                "median_duration": float(run_outcomes["median_duration"]),
                "hcw_hcw_share": float(run_outcomes["hcw_hcw_share"]),
                "hcw_patient_share": float(run_outcomes["hcw_patient_share"]),
                "patient_facing_share": float(run_outcomes["patient_facing_share"]),
                "station_or_desk_interaction_share": float(run_outcomes["station_or_desk_interaction_share"]),
                "corridor_interaction_share": float(run_outcomes["corridor_interaction_share"]),
                "bedside_or_patient_room_interaction_share": float(run_outcomes["bedside_or_patient_room_interaction_share"]),
                "other_or_uncoded_interaction_share": float(run_outcomes["other_or_uncoded_interaction_share"]),
                "workflow_health_status": str(workflow_health.get("workflow_health_status", "UNKNOWN")),
                "arrival_attempts": int(workflow_health.get("arrival_attempts", workflow_health.get("patient_arrivals", 0))),
                "patient_arrivals": int(workflow_health.get("patient_arrivals", 0)),
                "admitted_arrivals": int(workflow_health.get("admitted_arrivals", workflow_health.get("patient_arrivals", 0))),
                "deferred_arrivals": int(workflow_health.get("deferred_arrivals", 0)),
                "bed_assignments": int(workflow_health.get("bed_assignments", 0)),
                "successful_escorts_to_bed": int(workflow_health.get("successful_escorts_to_bed", 0)),
                "placement_tasks_completed": int(workflow_health.get("placement_tasks_completed", 0)),
                "workflow_tasks_completed": int(workflow_health.get("workflow_tasks_completed", 0)),
                "discharges": int(workflow_health.get("discharges", 0)),
                "stuck_agent_events": int(workflow_health.get("stuck_agent_events", 0)),
                "oscillation_warnings": int(workflow_health.get("oscillation_warnings", 0)),
            })
            if run_index == 0:
                plot_simulations[variant] = simulation
        per_run_records_by_variant[variant] = per_run_records

        role_pair_distribution = _normalized(
            Counter(
                "|".join(sorted((str(event["role_1"]), str(event["role_2"]))))
                for event in all_interactions
            )
        )
        topic_distribution = _normalized(Counter(str(event.get("topic", "unknown")) for event in all_interactions))
        zone_distribution = _normalized(Counter(str(event.get("zone_id", "unknown")) for event in all_interactions))
        interaction_type_counts = Counter(str(event.get("interaction_type", "unknown")) for event in all_interactions)
        interaction_task_counts = Counter(str(event.get("task_name", "none")) for event in all_interactions)
        patient_facing_events = [
            event for event in all_interactions
            if "Patient" in {event.get("role_1"), event.get("role_2")}
        ]
        patient_facing_interaction_type_counts = Counter(
            str(event.get("interaction_type", "unknown")) for event in patient_facing_events
        )
        patient_facing_task_counts = Counter(
            str(event.get("task_name", "none")) for event in patient_facing_events
        )
        station_ids = {"COCPIT", "NURSTA", "NUROPE"}
        added_bed_indices = set(range(5, len(config.BED_POSITIONS)))
        high_acuity_bed_indices = set(config.HIGH_ACUITY_RESERVED_BED_INDICES)
        patient_facing_bed_count = sum(1 for event in all_interactions if event.get("patient_id") is not None)
        station_hcw_hcw_count = sum(
            1
            for event in all_interactions
            if str(event.get("zone_id", "")) in station_ids
            and "Patient" not in {event.get("role_1"), event.get("role_2")}
        )
        qa_simulated_points_by_variant[variant] = [
            {
                "x": float(event.get("x")),
                "y": float(event.get("y")),
                "zone_id": str(event.get("zone_id", "unknown")),
                "role_pair": "|".join(sorted((str(event.get("role_1")), str(event.get("role_2"))))),
                "interaction_type": str(event.get("interaction_type", "unknown")),
            }
            for event in all_interactions
            if event.get("x") is not None
            and event.get("y") is not None
            and np.isfinite(float(event.get("x")))
            and np.isfinite(float(event.get("y")))
        ]
        outcome_metrics = _outcome_metrics(all_interactions, missed_opportunity_count)
        perceived_staff_events = [
            event for event in all_interactions
            if event.get("role_1") != "Patient"
            and event.get("role_2") != "Patient"
            and bool(event.get("perception_checked", False))
        ]
        initiation_distances = [
            float(event.get("initial_opportunity_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in perceived_staff_events
            if event.get("initial_opportunity_distance_m", event.get("initiation_distance_m")) is not None
        ]
        if not any(distance > 0.0 for distance in initiation_distances):
            initiation_distances = [
                float(example.get("initial_distance_m", 0.0))
                for example in perception_event_examples
                if example.get("initial_distance_m") is not None
            ]
        final_logged_distances = [
            float(event.get("final_logged_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in perceived_staff_events
            if event.get("interaction_source") == "opportunistic_perception_interruption"
            and event.get("final_logged_distance_m", event.get("initiation_distance_m")) is not None
        ]
        initial_opportunity_distances = [
            float(event.get("initial_opportunity_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in perceived_staff_events
            if event.get("interaction_source") == "opportunistic_perception_interruption"
            and event.get("initial_opportunity_distance_m", event.get("initiation_distance_m")) is not None
        ]
        nonproximate_logged_interaction_count = sum(
            1
            for distance in final_logged_distances
            if distance > config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS
        )
        close_threshold = float(config.PERCEPTION_CLOSE_PROXIMITY_THRESHOLD_METERS)
        validation_logged_distances = [
            float(event.get("final_logged_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in all_interactions
            if event.get("final_logged_distance_m", event.get("initiation_distance_m")) is not None
        ]
        staff_staff_logged_distances = [
            float(event.get("final_logged_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in all_interactions
            if event.get("final_logged_distance_m", event.get("initiation_distance_m")) is not None
            and event.get("role_1") != "Patient"
            and event.get("role_2") != "Patient"
        ]
        patient_facing_logged_distances = [
            float(event.get("final_logged_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in all_interactions
            if event.get("final_logged_distance_m", event.get("initiation_distance_m")) is not None
            and (event.get("role_1") == "Patient" or event.get("role_2") == "Patient")
        ]
        task_transition_logged_distances = [
            float(event.get("final_logged_distance_m", event.get("initiation_distance_m", 0.0)))
            for event in all_interactions
            if event.get("interaction_type") == "task_transition_update"
            and event.get("final_logged_distance_m", event.get("initiation_distance_m")) is not None
        ]
        validation_beyond_close_count = sum(
            1 for distance in validation_logged_distances if distance > close_threshold
        )
        synthetic_or_relocated_count = sum(
            1 for event in all_interactions if bool(event.get("synthetic_or_relocated_coordinate", False))
        )
        midpoint_logged_count = sum(
            1 for event in all_interactions if bool(event.get("midpoint_logged_validation_interaction", False))
        )
        perception_diagnostics = {
            "perception_based_baseline_enabled": bool(config.PERCEPTION_BASED_BASELINE_ENABLED),
            "legacy_proximity_mode_enabled": bool(config.LEGACY_PROXIMITY_MODE_ENABLED),
            "perception_checked_interaction_count": int(
                sum(1 for event in all_interactions if bool(event.get("perception_checked", False)))
            ),
            "interactions_logged_without_perception_check": int(
                sum(1 for event in all_interactions if not bool(event.get("perception_checked", False)))
            ),
            "raw_visible_staff_percepts_count": int(diagnostic_accumulator.get("raw_visible_staff_percepts_count", 0)),
            "eligible_visible_staff_opportunity_count": int(diagnostic_accumulator.get("eligible_visible_staff_opportunity_count", 0)),
            "selected_visible_staff_opportunity_count": int(diagnostic_accumulator.get("selected_visible_staff_opportunity_count", 0)),
            "perceived_staff_interaction_count": int(diagnostic_accumulator.get("perceived_staff_interaction_count", 0)),
            "nonproximate_logged_interaction_count": int(nonproximate_logged_interaction_count),
            "validation_counted_interactions_beyond_close_threshold_count": int(validation_beyond_close_count),
            "synthetic_or_relocated_interaction_coordinate_count": int(synthetic_or_relocated_count),
            "midpoint_logged_validation_interaction_count": int(midpoint_logged_count),
            "task_transition_update_logged_count": int(len(task_transition_logged_distances)),
            "task_transition_update_not_close_count": int(
                diagnostic_accumulator.get("task_transition_update_not_close_count", 0)
            ),
            "task_transition_update_max_logged_distance": float(max(task_transition_logged_distances))
            if task_transition_logged_distances else 0.0,
            "staff_staff_logged_max_distance": float(max(staff_staff_logged_distances))
            if staff_staff_logged_distances else 0.0,
            "patient_facing_logged_max_distance": float(max(patient_facing_logged_distances))
            if patient_facing_logged_distances else 0.0,
            "immediate_close_interaction_count": int(
                diagnostic_accumulator.get("immediate_close_interaction_count", 0)
            ),
            "pending_intent_completed_interaction_count": int(
                diagnostic_accumulator.get("pending_intent_completed_interaction_count", 0)
            ),
            "missed_not_close_opportunity_count": int(
                diagnostic_accumulator.get("missed_not_close_opportunity_count", 0)
            ),
            "externally_interruptible_rejections_count": int(
                diagnostic_accumulator.get("externally_interruptible_rejections_count", 0)
            ),
            "locally_communicative_allowed_count": int(
                diagnostic_accumulator.get("locally_communicative_allowed_count", 0)
            ),
            "cotask_interaction_allowed_count": int(
                diagnostic_accumulator.get("cotask_interaction_allowed_count", 0)
            ),
            "escort_patient_corridor_interaction_count": int(
                diagnostic_accumulator.get("escort_patient_corridor_interaction_count", 0)
            ),
            "walking_update_corridor_interaction_count": int(
                diagnostic_accumulator.get("walking_update_corridor_interaction_count", 0)
            ),
            "corridor_micro_interaction_count": int(
                diagnostic_accumulator.get("corridor_micro_interaction_count", 0)
            ),
            "corridor_pending_intent_count": int(
                diagnostic_accumulator.get("corridor_pending_intent_count", 0)
            ),
            "corridor_pending_intent_completed_in_corridor_count": int(
                diagnostic_accumulator.get("corridor_pending_intent_completed_in_corridor_count", 0)
            ),
            "corridor_pending_intent_completed_in_station_count": int(
                diagnostic_accumulator.get("corridor_pending_intent_completed_in_station_count", 0)
            ),
            "corridor_close_copresence_opportunity_count": int(
                diagnostic_accumulator.get("corridor_close_copresence_opportunity_count", 0)
            ),
            "corridor_close_copresence_logged_count": int(
                diagnostic_accumulator.get("corridor_close_copresence_logged_count", 0)
            ),
            "corridor_close_copresence_rejected_count": int(
                diagnostic_accumulator.get("corridor_close_copresence_rejected_count", 0)
            ),
            "corridor_rejection_reasons": {
                str(key).replace("corridor_rejection_reason_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("corridor_rejection_reason_")
            },
            "repeated_station_pair_suppressed_count": int(
                diagnostic_accumulator.get("repeated_station_pair_suppressed_count", 0)
            ),
            "station_same_pair_repeat_count": int(
                diagnostic_accumulator.get("station_same_pair_repeat_count", 0)
            ),
            "station_same_dwell_repeat_suppressed_count": int(
                diagnostic_accumulator.get("station_same_dwell_repeat_suppressed_count", 0)
            ),
            "station_same_pair_episode_count": int(
                diagnostic_accumulator.get("station_same_pair_episode_count", 0)
            ),
            "station_episode_new_encounter_count": int(
                diagnostic_accumulator.get("station_episode_new_encounter_count", 0)
            ),
            "station_episode_started_count": int(
                diagnostic_accumulator.get("station_episode_started_count", 0)
            ),
            "station_episode_ended_count": int(
                diagnostic_accumulator.get("station_episode_ended_count", 0)
            ),
            "station_same_episode_repeat_suppressed_count": int(
                diagnostic_accumulator.get("station_same_episode_repeat_suppressed_count", 0)
            ),
            "station_same_pair_same_episode_opportunity_count": int(
                diagnostic_accumulator.get("station_same_pair_same_episode_opportunity_count", 0)
            ),
            "station_episode_reset_reasons": {
                str(key).replace("station_episode_reset_reason_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("station_episode_reset_reason_")
            },
            "task_transition_update_episode_suppressed_count": int(
                diagnostic_accumulator.get("task_transition_update_episode_suppressed_count", 0)
            ),
            "task_transition_update_new_episode_count": int(
                diagnostic_accumulator.get("task_transition_update_new_episode_count", 0)
            ),
            "task_update_episode_started_count": int(
                diagnostic_accumulator.get("task_update_episode_started_count", 0)
            ),
            "task_update_episode_completed_count": int(
                diagnostic_accumulator.get("task_update_episode_completed_count", 0)
            ),
            "task_update_episode_suppressed_repeat_count": int(
                diagnostic_accumulator.get("task_update_episode_suppressed_repeat_count", 0)
            ),
            "task_update_episode_expired_count": int(
                diagnostic_accumulator.get("task_update_episode_expired_count", 0)
            ),
            "task_transition_update_by_role_pair": {
                str(key).replace("task_transition_update_role_pair_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("task_transition_update_role_pair_")
            },
            "task_transition_update_by_patient_context": {
                str(key).replace("task_transition_update_patient_context_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("task_transition_update_patient_context_")
            },
            "task_update_episode_by_role_pair": {
                str(key).replace("task_update_episode_role_pair_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("task_update_episode_role_pair_")
            },
            "task_update_episode_by_patient_context": {
                str(key).replace("task_update_episode_patient_context_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("task_update_episode_patient_context_")
            },
            "visible_staff_percepts_count": int(diagnostic_accumulator.get("visible_staff_percepts_count", 0)),
            "visible_staff_with_reason_count": int(diagnostic_accumulator.get("visible_staff_with_reason_count", 0)),
            "visible_staff_no_reason_count": int(diagnostic_accumulator.get("visible_staff_no_reason_count", 0)),
            "reason_type_counts": {
                str(key).replace("reason_type_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("reason_type_")
            },
            "reason_gate_rejected_count": int(diagnostic_accumulator.get("reason_gate_rejected_count", 0)),
            "reason_gate_rejection_reasons": {
                str(key).replace("reason_gate_rejection_reason_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("reason_gate_rejection_reason_")
            },
            "actionable_reason_opportunity_count": int(diagnostic_accumulator.get("actionable_reason_opportunity_count", 0)),
            "actionable_reason_to_approach_count": int(diagnostic_accumulator.get("actionable_reason_to_approach_count", 0)),
            "reason_to_completed_interaction_count": int(diagnostic_accumulator.get("reason_to_completed_interaction_count", 0)),
            "pressure_suppressed_unrelated_interactions": int(
                diagnostic_accumulator.get("pressure_suppressed_unrelated_interactions", 0)
            ),
            "pressure_created_bed_flow_reasons": int(
                diagnostic_accumulator.get("pressure_created_bed_flow_reasons", 0)
            ),
            "pressure_created_high_acuity_reasons": int(
                diagnostic_accumulator.get("pressure_created_high_acuity_reasons", 0)
            ),
            "high_acuity_escalation_count": int(diagnostic_accumulator.get("high_acuity_escalation_count", 0)),
            "high_acuity_event_arrival_count": int(diagnostic_accumulator.get("high_acuity_event_arrival_count", 0)),
            "high_acuity_shock_event_count": int(diagnostic_accumulator.get("high_acuity_shock_event_count", 0)),
            "high_acuity_shock_event_by_esi": {
                f"ESI {str(key).replace('high_acuity_shock_event_ESI_', '')}": int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("high_acuity_shock_event_ESI_")
            },
            "high_acuity_preemption_attempt_count": int(
                diagnostic_accumulator.get("high_acuity_preemption_attempt_count", 0)
            ),
            "high_acuity_preemption_success_count": int(
                diagnostic_accumulator.get("high_acuity_preemption_success_count", 0)
            ),
            "high_acuity_preemption_rejected_count": int(
                diagnostic_accumulator.get("high_acuity_preemption_rejected_count", 0)
            ),
            "high_acuity_preemption_rejection_reasons": {
                str(key).replace("high_acuity_preemption_rejection_reason_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("high_acuity_preemption_rejection_reason_")
            },
            "high_acuity_soft_preemption_count": int(
                diagnostic_accumulator.get("high_acuity_soft_preemption_count", 0)
            ),
            "high_acuity_conditional_preemption_count": int(
                diagnostic_accumulator.get("high_acuity_conditional_preemption_count", 0)
            ),
            "high_acuity_active_task_preemption_deferred_count": int(
                diagnostic_accumulator.get("high_acuity_active_task_preemption_deferred_count", 0)
            ),
            "high_acuity_soft_preemption_by_role": {
                str(key).replace("high_acuity_soft_preemption_role_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("high_acuity_soft_preemption_role_")
            },
            "high_acuity_preempted_from_states": {
                str(key).replace("high_acuity_preempted_from_state_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("high_acuity_preempted_from_state_")
            },
            "high_acuity_preemption_destination_tasks": {
                str(key).replace("high_acuity_preemption_destination_task_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("high_acuity_preemption_destination_task_")
            },
            "low_acuity_task_deferred_for_high_acuity_count": int(
                diagnostic_accumulator.get("low_acuity_task_deferred_for_high_acuity_count", 0)
            ),
            "high_acuity_reason_completed_count": int(
                diagnostic_accumulator.get("high_acuity_reason_completed_count", 0)
            ),
            "high_acuity_reason_missed_count": int(
                diagnostic_accumulator.get("high_acuity_reason_missed_count", 0)
            ),
            "coordination_nurse_patient_escort_interactions": int(
                diagnostic_accumulator.get("coordination_nurse_patient_escort_interactions", 0)
            ),
            "coordination_nurse_patient_interactions_by_zone": {
                str(key).replace("coordination_nurse_patient_interactions_zone_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("coordination_nurse_patient_interactions_zone_")
            },
            "patient_facing_task_start_interaction_count": int(
                diagnostic_accumulator.get("patient_facing_task_start_interaction_count", 0)
            ),
            "patient_facing_in_task_interaction_count": int(
                diagnostic_accumulator.get("patient_facing_in_task_interaction_count", 0)
            ),
            "patient_facing_in_task_by_role": {
                str(key).replace("patient_facing_in_task_role_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("patient_facing_in_task_role_")
            },
            "patient_facing_in_task_by_zone": {
                str(key).replace("patient_facing_in_task_zone_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("patient_facing_in_task_zone_")
            },
            "patient_facing_suppressed_by_interval_count": int(
                diagnostic_accumulator.get("patient_facing_suppressed_by_interval_count", 0)
            ),
            "bedside_cotask_staff_interaction_count": int(
                diagnostic_accumulator.get("bedside_cotask_staff_interaction_count", 0)
            ),
            "bedside_cotask_by_role_pair": {
                str(key).replace("bedside_cotask_role_pair_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("bedside_cotask_role_pair_")
            },
            "bedside_cotask_by_zone": {
                str(key).replace("bedside_cotask_zone_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("bedside_cotask_zone_")
            },
            "cotask_interaction_suppressed_by_cooldown": int(
                diagnostic_accumulator.get("cotask_interaction_suppressed_by_cooldown", 0)
            ),
            "cotask_interaction_suppressed_not_same_patient": int(
                diagnostic_accumulator.get("cotask_interaction_suppressed_not_same_patient", 0)
            ),
            "approach_intent_count": int(diagnostic_accumulator.get("approach_intent_count", 0)),
            "approach_success_count": int(diagnostic_accumulator.get("approach_success_count", 0)),
            "approach_failure_count": int(diagnostic_accumulator.get("approach_failure_count", 0)),
            "approach_failure_reasons": {
                str(key).replace("approach_failure_reason_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("approach_failure_reason_")
            },
            "immediate_close_logged_count": int(diagnostic_accumulator.get("immediate_close_logged_count", 0)),
            "raw_visible_staff_cocpit_related_count": int(diagnostic_accumulator.get("raw_visible_staff_cocpit_related_count", 0)),
            "eligible_cocpit_related_opportunity_count": int(diagnostic_accumulator.get("eligible_cocpit_related_opportunity_count", 0)),
            "approach_intent_cocpit_related_count": int(diagnostic_accumulator.get("approach_intent_cocpit_related_count", 0)),
            "completed_cocpit_related_count": int(diagnostic_accumulator.get("completed_cocpit_related_count", 0)),
            "raw_visible_staff_nursta_related_count": int(diagnostic_accumulator.get("raw_visible_staff_nursta_related_count", 0)),
            "eligible_nursta_related_opportunity_count": int(diagnostic_accumulator.get("eligible_nursta_related_opportunity_count", 0)),
            "approach_intent_nursta_related_count": int(diagnostic_accumulator.get("approach_intent_nursta_related_count", 0)),
            "completed_nursta_related_count": int(diagnostic_accumulator.get("completed_nursta_related_count", 0)),
            "route_override_count": int(diagnostic_accumulator.get("route_override_count", 0)),
            "route_resume_count": int(diagnostic_accumulator.get("route_resume_count", 0)),
            "post_task_station_check_count": int(diagnostic_accumulator.get("post_task_station_check_count", 0)),
            "post_task_station_check_completed_count": int(
                diagnostic_accumulator.get("post_task_station_check_completed_count", 0)
            ),
            "route_failures": int(diagnostic_accumulator.get("route_failure_count", 0)),
            "station_check_route_failures": int(diagnostic_accumulator.get("station_check_route_failures", 0)),
            "post_task_station_check_by_role": {
                str(key).replace("post_task_station_check_role_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("post_task_station_check_role_")
            },
            "post_task_station_check_by_station": {
                str(key).replace("post_task_station_check_station_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("post_task_station_check_station_")
            },
            "idle_heading_by_zone": {
                str(key).replace("idle_heading_zone_", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("idle_heading_zone_")
            },
            "cocpit_idle_heading_north_count": int(
                diagnostic_accumulator.get("cocpit_idle_heading_north_count", 0)
            ),
            "cocpit_idle_heading_south_count": int(
                diagnostic_accumulator.get("cocpit_idle_heading_south_count", 0)
            ),
            "cocpit_workstation_region_1_assignment_count": int(
                diagnostic_accumulator.get("cocpit_workstation_region_1_assignment_count", 0)
            ),
            "cocpit_workstation_region_2_assignment_count": int(
                diagnostic_accumulator.get("cocpit_workstation_region_2_assignment_count", 0)
            ),
            "cocpit_idle_region_1_occupancy_count": int(
                diagnostic_accumulator.get("cocpit_idle_region_1_occupancy_count", 0)
            ),
            "cocpit_idle_region_2_occupancy_count": int(
                diagnostic_accumulator.get("cocpit_idle_region_2_occupancy_count", 0)
            ),
            "nurope_dwell_by_role": {
                str(key).replace("zone_dwell_role_", "").replace("_zone_NUROPE", ""): int(value)
                for key, value in diagnostic_accumulator.items()
                if str(key).startswith("zone_dwell_role_") and str(key).endswith("_zone_NUROPE")
            },
            "initial_opportunity_distance_distribution": {
                "count": len(initial_opportunity_distances),
                "mean": float(mean(initial_opportunity_distances)) if initial_opportunity_distances else 0.0,
                "max": float(max(initial_opportunity_distances)) if initial_opportunity_distances else 0.0,
            },
            "final_logged_distance_distribution": {
                "count": len(final_logged_distances),
                "mean": float(mean(final_logged_distances)) if final_logged_distances else 0.0,
                "max": float(max(final_logged_distances)) if final_logged_distances else 0.0,
            },
            "interaction_locations_are_actual_contact_coordinates": bool(nonproximate_logged_interaction_count == 0),
            "mean_initiation_distance_m": float(mean(initiation_distances)) if initiation_distances else 0.0,
            "median_initiation_distance_m": float(median(initiation_distances)) if initiation_distances else 0.0,
            "max_initiation_distance_m": float(max(initiation_distances)) if initiation_distances else 0.0,
            "stationary_initiated_to_moving_count": int(diagnostic_accumulator.get("stationary_initiated_to_moving_count", 0)),
            "rejected_by_visibility": int(diagnostic_accumulator.get("rejected_by_visibility", 0)),
            "rejected_by_fov": int(
                diagnostic_accumulator.get("rejected_by_field_of_view", 0)
                + diagnostic_accumulator.get("rejected_fov_hard", 0)
                + diagnostic_accumulator.get("rejected_fov_soft_probability", 0)
            ),
            "rejected_by_salience_threshold": int(diagnostic_accumulator.get("rejected_by_salience_threshold", 0)),
            "rejected_by_attention_capacity": int(diagnostic_accumulator.get("rejected_by_attention_capacity", 0)),
            "rejected_by_protected_state": int(diagnostic_accumulator.get("rejected_by_protected_state", 0)),
            "rejected_by_cooldown": int(diagnostic_accumulator.get("rejected_by_cooldown", 0)),
            "protected_state_suppression_count": int(diagnostic_accumulator.get("protected_state_suppression_count", 0)),
            "example_event_traces": perception_event_examples[: max(int(config.PERCEPTION_MAX_EXAMPLES), 5)],
            "interaction_source_audit": _interaction_source_audit(all_interactions, all_missed_opportunities),
            "target_route_override_count": int(diagnostic_accumulator.get("target_route_override_count", 0)),
            "target_receive_pause_count": int(diagnostic_accumulator.get("target_receive_pause_count", 0)),
            "target_continued_route_count": int(diagnostic_accumulator.get("target_continued_route_count", 0)),
        }
        perception_diagnostics.update(_static_visibility_diagnostics())
        full_run_outcome_metrics = _outcome_metrics(all_full_run_interactions, missed_opportunity_count)
        workflow_health_summary = _aggregate_workflow_health(workflow_health_records)
        bed_care_flow_summary = _aggregate_bed_care_flow(bed_care_flow_records)
        representative = run_summaries[0] if run_summaries else {}
        mean_distance_components = {
            key: float(mean(record.get(key, 0.0) for record in distance_component_records))
            for key in ("kde_component", "zone_component", "role_pair_component", "topic_component", "duration_component", "weighted_total")
        } if distance_component_records else {}
        full_run_distance_components = {
            key: float(mean(record.get(key, 0.0) for record in full_run_distance_records))
            for key in ("kde_component", "zone_component", "role_pair_component", "topic_component", "duration_component", "weighted_total")
        } if full_run_distance_records else {}
        distance = float(mean(variant_distances)) if variant_distances else 0.0
        full_run_row_for_fit = {
            "duration_seconds": duration_seconds,
            "n_runs": n_runs,
            "interaction_count": len(all_full_run_interactions),
            "distance_components": full_run_distance_components,
            "outcome_metrics": full_run_outcome_metrics,
            "topic_divergence": _jsd(
                _normalized(Counter(str(event.get("topic", "unknown")) for event in all_full_run_interactions)),
                empirical.get("topic_distribution", {}),
            ),
        }
        full_run_validation_fit = (
            _validation_fit_for_row(full_run_row_for_fit, empirical, empirical_window)
            if run_kind == "validation_candidate"
            else {}
        )
        time_sliced_summary = _time_sliced_validation_summary(
            all_full_run_interactions,
            duration_seconds=duration_seconds,
            n_runs=n_runs,
        )
        matched_window_summary = (
            _matched_window_validation_summary(
                full_run_interactions_by_run,
                empirical_window,
                duration_seconds=duration_seconds,
                warmup_seconds=warmup_seconds,
            )
            if matched_empirical_windows
            else {"enabled": False, "reason": "Matched empirical-window sampling was not requested."}
        )
        row = {
            "model_variant": variant,
            "display_label": _display_variant_label(variant),
            "condition_name": condition_name,
            "scenario_mode": scenario_mode,
            "scenario_definition": dict(config.SCENARIO_MODES.get(scenario_mode, {})),
            "duration_seconds": duration_seconds,
            "warmup_seconds": warmup_seconds,
            "evaluation_duration_seconds": max(duration_seconds - warmup_seconds, 0),
            "n_runs": n_runs,
            "distance_to_empirical": distance,
            "distance_std": float(stdev(variant_distances)) if len(variant_distances) > 1 else 0.0,
            "spatial_cosine_similarity": cosine_similarity(
                np.asarray(representative.get("kde_grid", []), dtype=float),
                np.asarray(empirical.get("kde_grid", []), dtype=float),
            ),
            "interaction_count": len(all_interactions),
            "workflow_event_count": workflow_event_count,
            "missed_opportunity_count": missed_opportunity_count,
            "role_pair_divergence": _jsd(role_pair_distribution, empirical.get("role_pair_matrix", {})),
            "topic_divergence": _jsd(topic_distribution, empirical.get("topic_distribution", {})),
            "zone_divergence": _jsd(zone_distribution, empirical.get("zone_histogram", {})),
            "role_pair_distribution": role_pair_distribution,
            "topic_distribution": topic_distribution,
            "zone_distribution": zone_distribution,
            "interaction_type_counts": dict(interaction_type_counts),
            "interaction_task_counts": dict(interaction_task_counts),
            "workflow_task_counts": dict(workflow_task_counter),
            "patient_facing_interaction_type_counts": dict(patient_facing_interaction_type_counts),
            "patient_facing_task_counts": dict(patient_facing_task_counts),
            "patient_facing_task_attendance_counts": dict(patient_facing_task_attendance_accumulator),
            "patient_facing_validation_interaction_counts": dict(patient_facing_validation_interaction_accumulator),
            "patient_facing_task_attendance_count": int(sum(patient_facing_task_attendance_accumulator.values())),
            "patient_facing_validation_interaction_count": int(sum(patient_facing_validation_interaction_accumulator.values())),
            "nursta_interaction_count": int(Counter(str(event.get("zone_id", "unknown")) for event in all_interactions).get("NURSTA", 0)),
            "nurope_interaction_count": int(Counter(str(event.get("zone_id", "unknown")) for event in all_interactions).get("NUROPE", 0)),
            "nursta_interaction_share": float(zone_distribution.get("NURSTA", 0.0)),
            "nurope_interaction_share": float(zone_distribution.get("NUROPE", 0.0)),
            "patient_facing_bed_interaction_count": int(patient_facing_bed_count),
            "station_hcw_hcw_interaction_count": int(station_hcw_hcw_count),
            "staff_roster_used": {
                "staff_counts": config.STAFF_COUNTS,
                "nurse_home_zone_ids": config.NURSE_HOME_ZONE_IDS,
                "bed_positions": config.BED_POSITIONS,
                "nurse_bed_coverage": config.NURSE_BED_COVERAGE,
                "high_acuity_reserved_bed_indices": sorted(high_acuity_bed_indices),
            },
            "new_bed_indices": sorted(added_bed_indices | high_acuity_bed_indices),
            "outcome_metrics": outcome_metrics,
            "full_run_outcome_metrics": full_run_outcome_metrics,
            "senior_doctor_summary": _senior_doctor_summary(all_interactions),
            "perception_diagnostics": perception_diagnostics,
            "full_run_validation_fit": full_run_validation_fit,
            "hourly_time_slices": time_sliced_summary,
            "matched_window_summary": matched_window_summary,
            "workflow_health": workflow_health_summary,
            "bed_care_flow_summary": bed_care_flow_summary,
            "diagnostic_mode": active_diagnostic_mode,
            "scope_mode": scope_mode,
            "senior_doctor_oversight_enabled": bool(
                config.ENABLE_SENIOR_DOCTOR_OVERSIGHT
                if enable_senior_doctor_oversight is None
                else enable_senior_doctor_oversight
            ),
            "run_kind": run_kind,
            "run_purpose": run_purpose,
            "validation_target": validation_target,
            "distance_components": mean_distance_components,
            "full_run_distance_components": full_run_distance_components,
            "full_run_distance_to_empirical": float(full_run_distance_components.get("weighted_total", 0.0)),
        }
        diagnostics = {
            "model_variant": variant,
            "duration_seconds": duration_seconds,
            "warmup_seconds": warmup_seconds,
            "evaluation_duration_seconds": max(duration_seconds - warmup_seconds, 0),
            "n_runs": n_runs,
            "diagnostic_mode": active_diagnostic_mode,
            "scope_mode": scope_mode,
            "senior_doctor_oversight_enabled": bool(
                config.ENABLE_SENIOR_DOCTOR_OVERSIGHT
                if enable_senior_doctor_oversight is None
                else enable_senior_doctor_oversight
            ),
            "run_kind": run_kind,
            "run_purpose": run_purpose,
            "validation_target": validation_target,
            "eligible_encounters_considered": int(diagnostic_accumulator.get("eligible_encounters_considered", 0)),
            "rejected_by_distance": int(diagnostic_accumulator.get("rejected_by_distance", 0)),
            "rejected_by_visibility": int(diagnostic_accumulator.get("rejected_by_visibility", 0)),
            "rejected_by_field_of_view": int(
                diagnostic_accumulator.get("rejected_by_field_of_view", 0)
                + diagnostic_accumulator.get("rejected_fov_hard", 0)
                + diagnostic_accumulator.get("rejected_fov_soft_probability", 0)
            ),
            "rejected_fov_hard": int(diagnostic_accumulator.get("rejected_fov_hard", 0)),
            "rejected_fov_soft_probability": int(diagnostic_accumulator.get("rejected_fov_soft_probability", 0)),
            "rejected_by_attention_capacity": int(diagnostic_accumulator.get("rejected_by_attention_capacity", 0)),
            "rejected_by_cooldown": int(diagnostic_accumulator.get("rejected_by_cooldown", 0)),
            "rejected_by_probability_or_rule_decision": int(diagnostic_accumulator.get("rejected_by_probability_or_rule_decision", 0)),
            "accepted_direct_fov": int(diagnostic_accumulator.get("accepted_direct_fov", 0)),
            "accepted_shared_station_awareness": int(diagnostic_accumulator.get("accepted_shared_station_awareness", 0)),
            "accepted_task_driven": int(diagnostic_accumulator.get("accepted_task_driven", 0)),
            "accepted_patient_facing": int(diagnostic_accumulator.get("accepted_patient_facing", 0)),
            "accepted_corridor_opportunistic": int(diagnostic_accumulator.get("accepted_corridor_opportunistic", 0)),
            "accepted_as_interactions": len(all_interactions),
            "perception_checked_interaction_count": int(diagnostic_accumulator.get("perception_checked_interaction_count", 0)),
            "interactions_logged_without_perception_check": int(diagnostic_accumulator.get("interactions_logged_without_perception_check", 0)),
            **perception_diagnostics,
            "memory_retrieval_calls": int(llm_metric_accumulator.get("memory_retrieval_calls", 0)),
            "memories_retrieved": int(llm_metric_accumulator.get("memories_retrieved", 0)),
            "reflection_events_generated": sum(
                1
                for agent in plot_simulations[variant].staff_agents
                for event in plot_simulations[variant].interaction_engine.stream_for(agent.gid).events
                if event.event_type == "reflection"
            ),
            "backend_calls": int(llm_metric_accumulator.get("backend_calls", 0)),
            "backend_fallbacks": int(llm_metric_accumulator.get("llm_fallbacks", 0)),
            "llm_calls": int(llm_metric_accumulator.get("llm_calls", 0)),
            "rule_stub_calls": int(llm_metric_accumulator.get("rule_stub_calls", 0)),
            "schema_parse_failures": int(llm_metric_accumulator.get("schema_parse_failures", 0)),
            "patient_facing_interactions": sum(1 for event in all_interactions if "Patient" in {event["role_1"], event["role_2"]}),
            "hcw_hcw_interactions": sum(1 for event in all_interactions if "Patient" not in {event["role_1"], event["role_2"]}),
            "high_acuity_interactions": sum(1 for event in all_interactions if event.get("esi_level") in {1, 2}),
            "interruption_events": sum(len(plot_simulations[variant].interruption_log) for _ in [0]),
            "missed_opportunity_episodes": missed_opportunity_count,
            "memory_effects_applied": int(llm_metric_accumulator.get("memory_effects_applied", 0)),
            "accepted_because_of_memory": int(llm_metric_accumulator.get("accepted_because_of_memory", 0)),
            "rejected_because_of_memory": int(llm_metric_accumulator.get("rejected_because_of_memory", 0)),
            "decision_flipped_by_memory": int(llm_metric_accumulator.get("decision_flipped_by_memory", 0)),
            "duration_changed_by_memory": int(llm_metric_accumulator.get("duration_changed_by_memory", 0)),
            "topic_changed_by_memory": int(llm_metric_accumulator.get("topic_changed_by_memory", 0)),
            "structured_fallback_calls": int(llm_metric_accumulator.get("structured_fallback_calls", 0)),
            "satellite_station_checkin_opportunities": int(diagnostic_accumulator.get("satellite_station_checkin_opportunities", 0)),
            "patient_facing_task_not_validation_semantic": int(diagnostic_accumulator.get("patient_facing_task_not_validation_semantic", 0)),
            "patient_facing_suppressed_by_patient_interval": int(diagnostic_accumulator.get("patient_facing_suppressed_by_patient_interval", 0)),
            "patient_facing_task_no_explicit_communication": int(diagnostic_accumulator.get("patient_facing_task_no_explicit_communication", 0)),
            "workflow_health_status": workflow_health_summary.get("workflow_health_status", "UNKNOWN"),
            "arrival_attempts": int(workflow_health_summary.get("arrival_attempts", workflow_health_summary.get("patient_arrivals", 0))),
            "patient_arrivals": int(workflow_health_summary.get("patient_arrivals", 0)),
            "admitted_arrivals": int(workflow_health_summary.get("admitted_arrivals", workflow_health_summary.get("patient_arrivals", 0))),
            "deferred_arrivals": int(workflow_health_summary.get("deferred_arrivals", 0)),
            "bed_assignments": int(workflow_health_summary.get("bed_assignments", 0)),
            "successful_escorts_to_bed": int(workflow_health_summary.get("successful_escorts_to_bed", 0)),
            "placement_tasks_completed": int(workflow_health_summary.get("placement_tasks_completed", 0)),
            "workflow_tasks_completed": int(workflow_health_summary.get("workflow_tasks_completed", 0)),
            "discharges": int(workflow_health_summary.get("discharges", 0)),
            "stuck_agent_events": int(workflow_health_summary.get("stuck_agent_events", 0)),
            "oscillation_warnings": int(workflow_health_summary.get("oscillation_warnings", 0)),
        }
        diagnostic_rows.append(diagnostics)
        row.update({
            "missed_opportunity_episodes": missed_opportunity_count,
            "backend_calls": diagnostics["backend_calls"],
            "memory_retrieval_calls": diagnostics["memory_retrieval_calls"],
            "memory_effects_applied": diagnostics["memory_effects_applied"],
            "backend_fallbacks": diagnostics["backend_fallbacks"],
            "decision_flipped_by_memory": diagnostics["decision_flipped_by_memory"],
            "duration_changed_by_memory": diagnostics["duration_changed_by_memory"],
            "topic_changed_by_memory": diagnostics["topic_changed_by_memory"],
            "rejected_fov_hard": diagnostics["rejected_fov_hard"],
            "per_run_metrics": per_run_records,
        })
        metric_rows.append(row)
        summary[variant] = {
            **row,
            "distance_components": mean_distance_components or representative.get("distance_components", {}),
            "interaction_summary": plot_simulations[variant].interaction_summary(),
            "diagnostics": diagnostics,
            "validation_metric_audit": validation_audit,
        }

    if metric_rows:
        traditional_row = metric_rows[0]
        for row in metric_rows:
            variant = row["model_variant"]
            deltas = _delta_metrics(row["outcome_metrics"], traditional_row["outcome_metrics"])
            row["outcome_deltas_vs_traditional"] = deltas
            row["role_pair_jsd_vs_traditional"] = _jsd(row["role_pair_distribution"], traditional_row["role_pair_distribution"])
            row["topic_jsd_vs_traditional"] = _jsd(row["topic_distribution"], traditional_row["topic_distribution"])
            row["zone_jsd_vs_traditional"] = _jsd(row["zone_distribution"], traditional_row["zone_distribution"])
            outcome_signals = [
                abs(deltas["interaction_count_delta"]) >= max(5.0, 0.1 * max(float(traditional_row["outcome_metrics"]["interaction_count"]), 1.0)),
                abs(deltas["missed_opportunity_episodes_delta"]) >= 3.0,
                row["role_pair_jsd_vs_traditional"] >= 0.05,
                row["topic_jsd_vs_traditional"] >= 0.05,
                row["zone_jsd_vs_traditional"] >= 0.05,
                abs(deltas["mean_duration_delta"]) >= 10.0,
            ]
            row["outcome_difference_signal_count"] = int(sum(outcome_signals))
            if variant in summary:
                summary[variant].update({
                    "outcome_metrics": row["outcome_metrics"],
                    "outcome_deltas_vs_traditional": row["outcome_deltas_vs_traditional"],
                    "role_pair_jsd_vs_traditional": row["role_pair_jsd_vs_traditional"],
                    "topic_jsd_vs_traditional": row["topic_jsd_vs_traditional"],
                    "zone_jsd_vs_traditional": row["zone_jsd_vs_traditional"],
                    "outcome_difference_signal_count": row["outcome_difference_signal_count"],
                    "per_run_metrics": row["per_run_metrics"],
                })
        for row in metric_rows:
            row_with_diagnostics = {
                **row,
                "diagnostics": summary[row["model_variant"]].get("diagnostics", {}),
            }
            row["variant_classification"] = _variant_interpretation(row["model_variant"], row_with_diagnostics, traditional_row)
            if run_kind == "validation_candidate":
                row["validation_fit"] = _validation_fit_for_row(row, empirical, empirical_window)
            if row["model_variant"] in summary:
                summary[row["model_variant"]]["variant_classification"] = row["variant_classification"]
                if run_kind == "validation_candidate":
                    summary[row["model_variant"]]["validation_fit"] = row["validation_fit"]
        if n_runs > 1:
            ensemble_metric_keys = [
                "distance_to_empirical",
                "interaction_count",
                "missed_opportunity_episodes",
                "memory_effects_applied",
                "decision_flipped_by_memory",
                "topic_changed_by_memory",
                "backend_fallbacks",
                "hcw_hcw_share",
                "patient_facing_share",
                "station_or_desk_interaction_share",
                "corridor_interaction_share",
                "mean_duration",
            ]
            traditional_runs = per_run_records_by_variant.get("traditional_rule", [])
            perception_runs = per_run_records_by_variant.get("perception_rule", [])
            for row in metric_rows:
                variant = row["model_variant"]
                records = per_run_records_by_variant.get(variant, [])
                ensemble_summary = {
                    "metric_mean": {
                        key: float(mean(record[key] for record in records)) if records else 0.0
                        for key in ensemble_metric_keys
                    },
                    "metric_std": {
                        key: float(stdev(record[key] for record in records)) if len(records) > 1 else 0.0
                        for key in ensemble_metric_keys
                    },
                    "paired_delta_vs_traditional_by_seed": [],
                    "paired_delta_vs_perception_by_seed": [],
                    "plain_language": "Paired-seed deltas estimate stability; they are not empirical validation.",
                }
                if traditional_runs and variant != "traditional_rule":
                    for record, reference in zip(records, traditional_runs):
                        ensemble_summary["paired_delta_vs_traditional_by_seed"].append({
                            "seed": record["seed"],
                            **{
                                key: float(record[key]) - float(reference[key])
                                for key in ensemble_metric_keys
                            },
                        })
                if perception_runs and variant in {"memory_rule", "generative_interaction"}:
                    for record, reference in zip(records, perception_runs):
                        ensemble_summary["paired_delta_vs_perception_by_seed"].append({
                            "seed": record["seed"],
                            **{
                                key: float(record[key]) - float(reference[key])
                                for key in ensemble_metric_keys
                            },
                        })
                if ensemble_summary["paired_delta_vs_traditional_by_seed"]:
                    ensemble_summary["plain_language"] = _paired_delta_noise_statement(
                        ensemble_summary["paired_delta_vs_traditional_by_seed"],
                        "distance_to_empirical",
                    )
                elif variant == "traditional_rule":
                    ensemble_summary["plain_language"] = "Traditional_rule is the paired-delta reference variant."
                if variant in summary:
                    summary[variant]["ensemble_summary"] = ensemble_summary
        for row in metric_rows:
            row.update(summary[row["model_variant"]])
        diagnostics_payload = {row["model_variant"]: row for row in diagnostic_rows}
        sufficiency_payload = _model_sufficiency_summary(summary)
        config.LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if run_kind == "mechanism_audit":
            baseline_key = variants[0] if variants else next(iter(summary), "")
            baseline_summary = summary.get(baseline_key, {})
            config.MECHANISM_AUDIT_SUMMARY_PATH.write_text(json.dumps({
                "run_kind": "mechanism_audit",
                "baseline_model_id": config.BASELINE_MODEL_ID,
                "duration_seconds": duration_seconds,
                "n_runs": n_runs,
                "baseline_result": _public_baseline_payload(baseline_summary),
            }, indent=2))
            config.MECHANISM_AUDIT_DIAGNOSTICS_PATH.write_text(json.dumps({
                config.BASELINE_MODEL_ID: _public_baseline_payload(
                    diagnostics_payload.get(baseline_key, {})
                )
            }, indent=2))
            _write_perception_mechanism_outputs(summary)
            if n_runs > 1:
                config.ENSEMBLE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        elif run_kind == "validation_candidate":
            baseline_key = variants[0] if variants else next(iter(summary), "")
            baseline_summary = summary.get(baseline_key, {})
            public_baseline_summary = _public_baseline_payload(baseline_summary)
            baseline_validation_fit = baseline_summary.get("validation_fit", {})
            baseline_full_fit = baseline_summary.get("full_run_validation_fit", {})
            baseline_time_slices = baseline_summary.get("hourly_time_slices", [])
            baseline_matched_windows = baseline_summary.get("matched_window_summary", {})
            payload = {
                "run_kind": run_kind,
                "run_purpose": run_purpose,
                "diagnostic_mode": active_diagnostic_mode,
                "scope_mode": scope_mode,
                "scenario_mode": scenario_mode,
                "scenario_definition": dict(config.SCENARIO_MODES.get(scenario_mode, {})),
                "scenario_clock": _scenario_clock_metadata(
                    scenario_mode=scenario_mode,
                    scenario_start_hour=scenario_start_hour,
                    duration_seconds=duration_seconds,
                    warmup_seconds=warmup_seconds,
                ),
                "validation_target": validation_target,
                "validation_target_metadata": validation_target_metadata,
                "senior_doctor_oversight_enabled": bool(
                    config.ENABLE_SENIOR_DOCTOR_OVERSIGHT
                    if enable_senior_doctor_oversight is None
                    else enable_senior_doctor_oversight
                ),
                "baseline_model_id": config.BASELINE_MODEL_ID,
                "full_empirical_is_mismatch_reference": False,
                "duration_seconds": duration_seconds,
                "warmup_seconds": warmup_seconds,
                "evaluation_duration_seconds": max(duration_seconds - warmup_seconds, 0),
                "validation_protocol": {
                    "model_behavior_changed": False,
                    "full_run_duration_seconds": duration_seconds,
                    "warmup_seconds": warmup_seconds,
                    "evaluation_window_start_second": warmup_seconds,
                    "evaluation_window_end_second": duration_seconds,
                    "matched_empirical_windows": bool(matched_empirical_windows),
                    "purpose": (
                        "Separate startup transient behavior from steady-state behavior; "
                        "workflow events remain excluded from validation metrics."
                    ),
                },
                "standard_validation_recommendation": _standard_validation_protocol_recommendation(
                    full_fit=baseline_full_fit or baseline_validation_fit,
                    warmup_fit=baseline_validation_fit if warmup_seconds > 0 else None,
                    time_slices=baseline_time_slices,
                    matched_summary=baseline_matched_windows,
                    warmup_seconds=warmup_seconds,
                ),
                "n_runs": n_runs,
                "empirical_window": empirical_window,
                "validation_thresholds": config.VALIDATION_THRESHOLDS,
                "workflow_health_by_model": {
                    config.BASELINE_MODEL_ID: baseline_summary.get("workflow_health", {})
                },
                "baseline_result": public_baseline_summary,
                "model_results": {config.BASELINE_MODEL_ID: public_baseline_summary},
                "baseline_recommendation": _select_baseline_recommendation(summary),
            }
            config.VALIDATION_SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
            scenario_summary_path = config.LATEST_OUTPUT_DIR / f"validation_summary_{scenario_mode}.json"
            scenario_summary_path.write_text(json.dumps(payload, indent=2))
            comparison_payload = {}
            if config.SCENARIO_SANITY_COMPARISON_PATH.exists():
                try:
                    comparison_payload = json.loads(config.SCENARIO_SANITY_COMPARISON_PATH.read_text())
                except json.JSONDecodeError:
                    comparison_payload = {}
            comparison_payload = {
                key: value
                for key, value in comparison_payload.items()
                if key in {"normal_load", "high_load_high_acuity"}
            }
            comparison_payload[scenario_mode] = {
                "scenario_mode": scenario_mode,
                "scenario_definition": dict(config.SCENARIO_MODES.get(scenario_mode, {})),
                "scenario_clock": payload["scenario_clock"],
                "summary_path": str(scenario_summary_path),
                "distance_to_empirical": public_baseline_summary.get("distance_to_empirical"),
                "validation_fit": public_baseline_summary.get("validation_fit", {}),
                "workflow_health": public_baseline_summary.get("workflow_health", {}),
                "outcome_metrics": public_baseline_summary.get("outcome_metrics", {}),
                "perception_diagnostics": public_baseline_summary.get("perception_diagnostics", {}),
            }
            config.SCENARIO_SANITY_COMPARISON_PATH.write_text(json.dumps(comparison_payload, indent=2))
            _write_perception_mechanism_outputs(summary)
            _write_care_area_outputs(
                payload,
                simulated_points=qa_simulated_points_by_variant.get(baseline_key, []),
            )
            _plot_validation_fit([{**row, **summary[row["model_variant"]]} for row in metric_rows], empirical)
            _plot_paired_seed_uncertainty(metric_rows)
        else:
            latest_summary = config.LATEST_OUTPUT_DIR / "ablation_summary.json"
            latest_diagnostics = config.LATEST_OUTPUT_DIR / "ablation_diagnostics.json"
            latest_summary.write_text(json.dumps(summary, indent=2))
            latest_diagnostics.write_text(json.dumps(diagnostics_payload, indent=2))
            _write_perception_mechanism_outputs(summary)
            _plot_mechanism_audit([{**row, "diagnostics": diagnostics_payload[row["model_variant"]]} for row in metric_rows])
            _plot_paired_seed_uncertainty(metric_rows)
            config.MODEL_SUFFICIENCY_SUMMARY_PATH.write_text(json.dumps(sufficiency_payload, indent=2))
            _plot_model_sufficiency(summary)
        _write_model_status(summary, duration_seconds, n_runs, run_kind)
        _write_outputs_readme()

    return summary


def run_mechanism_audit(
    duration_seconds: int = 3600,
    n_runs: int = 10,
    condition_name: str = "baseline",
) -> dict:
    """Run diagnostic-mode-on mechanism audit. This is not validation."""

    return run_ablation_smoke(
        duration_seconds=duration_seconds,
        n_runs=n_runs,
        condition_name=condition_name,
        run_kind="mechanism_audit",
        diagnostic_mode=True,
    )


def run_validation_candidate(
    duration_seconds: int,
    n_runs: int = 10,
    condition_name: str = "baseline",
    validation_target: str = "care_area",
    variants: Optional[Iterable[str]] = None,
    scope_mode: str = config.DEFAULT_SCOPE_MODE,
    warmup_seconds: int = 0,
    matched_empirical_windows: bool = False,
    enable_senior_doctor_oversight: Optional[bool] = None,
    scenario_mode: str = config.DEFAULT_SCENARIO_MODE,
    scenario_start_hour: int = config.DEFAULT_SCENARIO_START_HOUR,
) -> dict:
    """Run diagnostic-mode-off validation candidate against empirical F2F targets."""

    selected_variants = list(config.DEFAULT_BEHAVIORAL_ABLATION_VARIANTS) if variants is None else list(variants)
    return run_ablation_smoke(
        duration_seconds=duration_seconds,
        n_runs=n_runs,
        condition_name=condition_name,
        variants=selected_variants,
        run_kind="validation_candidate",
        diagnostic_mode=False,
        validation_target=validation_target,
        scope_mode=scope_mode,
        warmup_seconds=warmup_seconds,
        matched_empirical_windows=matched_empirical_windows,
        enable_senior_doctor_oversight=enable_senior_doctor_oversight,
        scenario_mode=scenario_mode,
        scenario_start_hour=scenario_start_hour,
    )
