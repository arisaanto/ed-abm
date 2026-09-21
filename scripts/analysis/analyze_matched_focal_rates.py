#!/usr/bin/env python3
"""Compare empirical and simulated interaction rates under matched focal sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROLE_MAP = {
    "coordination": "CoordinationNurse",
    "nurse": "Nurse",
    "assistant_doctor": "Doctor",
}


def empirical_sessions(path: Path) -> pd.DataFrame:
    records = pd.read_csv(path)
    records["start"] = pd.to_datetime(records["start"], errors="coerce")
    records["end"] = pd.to_datetime(records["end"], errors="coerce")
    records = records.dropna(subset=["participantSensorId", "start", "end"]).copy()
    records["observation_date"] = records["start"].dt.date.astype(str)
    records["is_f2f"] = records["interfaceType"].fillna("").astype(str).str.contains(
        "Person", case=False
    )
    grouped = records.groupby(
        ["participantSensorId", "observation_date"], sort=True, dropna=False
    )
    sessions = grouped.agg(
        first_record=("start", "min"),
        last_record=("end", "max"),
        empirical_role=("role_survey", "first"),
        all_record_count=("start", "size"),
        f2f_count=("is_f2f", "sum"),
    ).reset_index()
    sessions["duration_seconds"] = (
        sessions["last_record"] - sessions["first_record"]
    ).dt.total_seconds()
    sessions = sessions.loc[sessions["duration_seconds"] > 0].copy()
    sessions["duration_hours"] = sessions["duration_seconds"] / 3600.0
    sessions["observed_f2f_per_hour"] = (
        sessions["f2f_count"] / sessions["duration_hours"]
    )
    sessions["simulation_role"] = sessions["empirical_role"].map(ROLE_MAP)
    sessions["included_in_role_match"] = sessions["simulation_role"].notna()
    sessions["session_id"] = (
        sessions["participantSensorId"].astype(str)
        + "_"
        + sessions["observation_date"].astype(str)
    )
    return sessions


def load_simulated_focal_events(batch_dir: Path) -> pd.DataFrame:
    files = sorted(batch_dir.rglob("interaction_events.csv"))
    if not files:
        raise FileNotFoundError(f"No interaction_events.csv files under {batch_dir}")
    rows: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        required = {
            "seed",
            "condition",
            "scenario",
            "time_seconds",
            "agent_a_id",
            "agent_b_id",
            "role_a",
            "role_b",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks focal-agent columns: {sorted(missing)}")
        frame = frame.loc[
            (frame["condition"].astype(str) == "baseline")
            & (frame["scenario"].astype(str) == "normal_load")
        ].copy()
        if frame.empty:
            continue
        for side in ("a", "b"):
            endpoint = frame[
                ["seed", "time_seconds", f"agent_{side}_id", f"role_{side}"]
            ].copy()
            endpoint.columns = ["seed", "time_seconds", "agent_id", "role"]
            endpoint = endpoint.loc[endpoint["role"].astype(str) != "Patient"]
            rows.append(endpoint)
    if not rows:
        raise ValueError("No normal-load baseline staff interaction endpoints found")
    endpoints = pd.concat(rows, ignore_index=True)
    endpoints["seed"] = pd.to_numeric(endpoints["seed"], errors="raise").astype(int)
    endpoints["agent_id"] = pd.to_numeric(
        endpoints["agent_id"], errors="raise"
    ).astype(int)
    endpoints["time_seconds"] = pd.to_numeric(
        endpoints["time_seconds"], errors="raise"
    ).astype(float)
    return endpoints.sort_values(["seed", "agent_id", "time_seconds"])


def direct_first_to_last(endpoints: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    focal = endpoints.groupby(["seed", "agent_id", "role"], sort=True).agg(
        first_event=("time_seconds", "min"),
        last_event=("time_seconds", "max"),
        f2f_count=("time_seconds", "size"),
    ).reset_index()
    focal["duration_seconds"] = focal["last_event"] - focal["first_event"]
    focal = focal.loc[focal["duration_seconds"] > 0].copy()
    focal["duration_hours"] = focal["duration_seconds"] / 3600.0
    focal["f2f_per_hour"] = focal["f2f_count"] / focal["duration_hours"]
    seed_rates = focal.groupby("seed").apply(
        lambda x: x["f2f_count"].sum() / x["duration_hours"].sum(),
        include_groups=False,
    )
    summary = {
        "focal_agent_seed_windows": int(len(focal)),
        "pooled_f2f_per_hour": float(
            focal["f2f_count"].sum() / focal["duration_hours"].sum()
        ),
        "mean_seed_pooled_f2f_per_hour": float(seed_rates.mean()),
        "seed_pooled_95_interval": [
            float(seed_rates.quantile(0.025)),
            float(seed_rates.quantile(0.975)),
        ],
        "median_focal_window_rate": float(focal["f2f_per_hour"].median()),
    }
    return focal, summary


def duration_and_role_match(
    sessions: pd.DataFrame, endpoints: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    included = sessions.loc[sessions["included_in_role_match"]].copy()
    agent_events = {
        (int(seed), int(agent_id), str(role)): group["time_seconds"].to_numpy()
        for (seed, agent_id, role), group in endpoints.groupby(
            ["seed", "agent_id", "role"], sort=True
        )
    }
    seeds = sorted(endpoints["seed"].unique())
    matched_rows: list[dict] = []
    for seed in seeds:
        role_agents: dict[str, list[int]] = {}
        for candidate_seed, agent_id, role in agent_events:
            if candidate_seed == seed:
                role_agents.setdefault(role, []).append(agent_id)
        for session in included.itertuples(index=False):
            counts = []
            for agent_id in role_agents.get(str(session.simulation_role), []):
                times = agent_events[(seed, agent_id, str(session.simulation_role))]
                if len(times) == 0:
                    continue
                cutoff = times[0] + float(session.duration_seconds)
                counts.append(int(np.searchsorted(times, cutoff, side="right")))
            if not counts:
                raise ValueError(
                    f"No simulated {session.simulation_role} focal agents for seed {seed}"
                )
            matched_rows.append(
                {
                    "seed": seed,
                    "session_id": session.session_id,
                    "simulation_role": session.simulation_role,
                    "duration_hours": float(session.duration_hours),
                    "observed_f2f_count": int(session.f2f_count),
                    "mean_simulated_f2f_count": float(np.mean(counts)),
                    "simulated_agent_count": len(counts),
                }
            )
    matched = pd.DataFrame(matched_rows)
    session_summary = matched.groupby(
        ["session_id", "simulation_role", "duration_hours", "observed_f2f_count"],
        as_index=False,
    ).agg(
        mean_simulated_f2f_count=("mean_simulated_f2f_count", "mean"),
        lower_simulated_f2f_count=(
            "mean_simulated_f2f_count",
            lambda x: x.quantile(0.025),
        ),
        upper_simulated_f2f_count=(
            "mean_simulated_f2f_count",
            lambda x: x.quantile(0.975),
        ),
    )
    seed_totals = matched.groupby("seed", as_index=False).agg(
        simulated_count=("mean_simulated_f2f_count", "sum"),
        duration_hours=("duration_hours", "sum"),
    )
    seed_totals["simulated_f2f_per_hour"] = (
        seed_totals["simulated_count"] / seed_totals["duration_hours"]
    )
    observed_count = int(included["f2f_count"].sum())
    observed_hours = float(included["duration_hours"].sum())
    rates = seed_totals["simulated_f2f_per_hour"]
    observed_rate = observed_count / observed_hours
    poisson_se = np.sqrt(observed_count) / observed_hours
    rng = np.random.default_rng(20260823)
    bootstrap_rates = []
    for _ in range(20_000):
        sample = included.iloc[
            rng.integers(0, len(included), size=len(included))
        ]
        bootstrap_rates.append(
            float(sample["f2f_count"].sum() / sample["duration_hours"].sum())
        )
    role_rows = []
    for role, role_sessions in included.groupby("simulation_role", sort=True):
        role_match = session_summary.loc[
            session_summary["simulation_role"] == role
        ]
        role_hours = float(role_sessions["duration_hours"].sum())
        role_observed = int(role_sessions["f2f_count"].sum())
        role_simulated = float(role_match["mean_simulated_f2f_count"].sum())
        role_rows.append(
            {
                "role": str(role),
                "session_count": int(len(role_sessions)),
                "observed_f2f_per_hour": role_observed / role_hours,
                "mean_simulated_f2f_per_hour": role_simulated / role_hours,
            }
        )
    duration_sensitivity = []
    for minimum_minutes in (0, 10, 30, 60):
        retained = included.loc[
            included["duration_seconds"] >= minimum_minutes * 60
        ]
        retained_ids = set(retained["session_id"])
        retained_matches = matched.loc[matched["session_id"].isin(retained_ids)]
        retained_seed_totals = retained_matches.groupby("seed", as_index=False).agg(
            simulated_count=("mean_simulated_f2f_count", "sum"),
            duration_hours=("duration_hours", "sum"),
        )
        retained_seed_rates = (
            retained_seed_totals["simulated_count"]
            / retained_seed_totals["duration_hours"]
        )
        duration_sensitivity.append(
            {
                "minimum_session_minutes": minimum_minutes,
                "session_count": int(len(retained)),
                "observed_f2f_per_hour": float(
                    retained["f2f_count"].sum()
                    / retained["duration_hours"].sum()
                ),
                "mean_simulated_f2f_per_matched_focal_hour": float(
                    retained_seed_rates.mean()
                ),
                "simulated_seed_95_interval": [
                    float(retained_seed_rates.quantile(0.025)),
                    float(retained_seed_rates.quantile(0.975)),
                ],
            }
        )
    summary = {
        "empirical_sessions_included": int(len(included)),
        "empirical_sessions_excluded_unmodelled_role": int(
            (~sessions["included_in_role_match"]).sum()
        ),
        "observed_f2f_count": observed_count,
        "observed_focal_hours": observed_hours,
        "observed_f2f_per_focal_hour": float(observed_rate),
        "observed_poisson_approximate_95_interval": [
            float(observed_rate - 1.96 * poisson_se),
            float(observed_rate + 1.96 * poisson_se),
        ],
        "observed_session_bootstrap_95_interval": [
            float(np.quantile(bootstrap_rates, 0.025)),
            float(np.quantile(bootstrap_rates, 0.975)),
        ],
        "mean_simulated_f2f_per_matched_focal_hour": float(rates.mean()),
        "simulated_seed_95_interval": [
            float(rates.quantile(0.025)),
            float(rates.quantile(0.975)),
        ],
        "observed_to_simulated_rate_ratio": float(
            observed_rate / rates.mean()
        ),
        "role_specific_rates": role_rows,
        "observed_duration_threshold_sensitivity": duration_sensitivity,
    }
    return matched, session_summary, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-csv", type=Path, required=True)
    parser.add_argument("--simulation-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sessions = empirical_sessions(args.empirical_csv)
    endpoints = load_simulated_focal_events(args.simulation_batch_dir)
    focal_windows, direct_summary = direct_first_to_last(endpoints)
    matched, session_summary, matched_summary = duration_and_role_match(
        sessions, endpoints
    )

    empirical_pooled = {
        "participant_date_sessions": int(len(sessions)),
        "f2f_count": int(sessions["f2f_count"].sum()),
        "focal_hours": float(sessions["duration_hours"].sum()),
        "f2f_per_focal_hour": float(
            sessions["f2f_count"].sum() / sessions["duration_hours"].sum()
        ),
    }
    payload = {
        "analysis_definition": {
            "empirical_session": "participantSensorId by observation date",
            "empirical_window": "first to last recorded event of any interface type",
            "empirical_numerator": "records whose interfaceType contains Person",
            "simulated_focal_event": "one staff endpoint of any simulated F2F event",
            "matched_window": "same role and empirical session duration, beginning at the simulated focal agent's first event",
        },
        "empirical_direct_first_to_last": empirical_pooled,
        "simulation_direct_first_to_last": direct_summary,
        "duration_and_role_matched": matched_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(args.output_dir / "empirical_focal_sessions.csv", index=False)
    focal_windows.to_csv(
        args.output_dir / "simulated_focal_first_to_last_windows.csv", index=False
    )
    matched.to_csv(args.output_dir / "matched_seed_session_counts.csv", index=False)
    session_summary.to_csv(
        args.output_dir / "matched_session_summary.csv", index=False
    )
    (args.output_dir / "matched_focal_rate_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
