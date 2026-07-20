"""Readable supervisor-facing smoke-test outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import config

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
config.THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_simulation_overview(simulation, path: Path | None = None) -> Path:
    path = config.SIMULATION_OVERVIEW_PATH if path is None else path
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    simulation.condition_manager.draw_floorplan(ax)
    simulation.condition_manager.draw_zone_overlays(ax)
    for agent in simulation.staff_agents:
        xs = [point[0] for point in agent.trail]
        ys = [point[1] for point in agent.trail]
        ax.plot(xs, ys, color=config.ROLE_COLORS[agent.role], alpha=0.18, linewidth=0.9)
        ax.scatter(agent.position[0], agent.position[1], color=config.ROLE_COLORS[agent.role], s=45, zorder=5)
        ax.text(agent.position[0], agent.position[1], getattr(agent, "name", str(agent.gid)).split()[0], fontsize=7)
    for bed_index, bed in enumerate(config.BED_POSITIONS, start=1):
        ax.scatter(bed[0], bed[1], marker="s", color="#555555", s=35, zorder=4)
        ax.text(bed[0], bed[1] + 0.25, f"B{bed_index}", fontsize=7, ha="center")
    if simulation.interaction_log:
        xs = [event["x"] for event in simulation.interaction_log]
        ys = [event["y"] for event in simulation.interaction_log]
        ax.scatter(xs, ys, color="#E45756", s=18, alpha=0.65, zorder=6, label="Interactions")
    high_acuity = [patient for patient in simulation.active_patients + simulation.completed_patients if getattr(patient, "esi_level", 5) in {1, 2}]
    for patient in high_acuity:
        ax.scatter(patient.position[0], patient.position[1], marker="*", color="#F58518", s=80, zorder=7)
    ax.set_xlim(*simulation.environment.plot_bounds[:2])
    ax.set_ylim(*simulation.environment.plot_bounds[2:])
    ax.set_aspect("equal")
    ax.set_title(f"{simulation.condition_spec.name} | {simulation.model_variant} | seed {simulation.random_seed}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_interaction_heatmap(simulation, path: Path | None = None) -> Path:
    path = config.THESIS_FIGURES_DIR / f"interaction_heatmap_{simulation.condition_spec.name}_{simulation.model_variant}.png" if path is None else path
    fig, ax = plt.subplots(figsize=(9, 7))
    simulation.condition_manager.draw_floorplan(ax)
    simulation.condition_manager.draw_zone_overlays(ax)
    grid = np.asarray(simulation.compute_kde_grid(), dtype=float)
    if grid.sum() > 0:
        min_x, max_x, min_y, max_y = simulation.environment.plot_bounds
        ax.imshow(grid, extent=(min_x, max_x, min_y, max_y), origin="lower", cmap=config.HEATMAP_CMAP, alpha=0.45)
    ax.set_xlim(*simulation.environment.plot_bounds[:2])
    ax.set_ylim(*simulation.environment.plot_bounds[2:])
    ax.set_aspect("equal")
    ax.set_title("Interaction heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_agent_experience_cards(simulation, path: Path | None = None) -> Path:
    path = config.AGENT_EXPERIENCE_CARDS_PATH if path is None else path
    lines = [
        f"# Agent Experience Cards: {simulation.condition_spec.name} / {simulation.model_variant}",
        "",
        "Synthetic trace summaries for inspection; not human testimony.",
        "",
    ]
    for agent in simulation.staff_agents:
        name = getattr(agent, "name", f"{agent.role} {agent.gid}")
        events = [
            event for event in simulation.interaction_log
            if event["agent_1_id"] == agent.gid or event["agent_2_id"] == agent.gid
        ]
        topic_counts = Counter(str(event.get("topic", "unknown")) for event in events)
        partner_counts = Counter(
            str(event["agent_2_name"] if event["agent_1_id"] == agent.gid else event["agent_1_name"])
            for event in events
        )
        stream = simulation.interaction_engine.stream_for(agent.gid)
        top_memories = sorted(stream.events, key=lambda event: (event.importance, event.timestamp), reverse=True)[:3]
        missed = [
            event for event in simulation.missed_opportunity_log
            if event.get("agent_id") == agent.gid
        ]
        lines.extend([
            f"## {name} ({agent.role})",
            f"- Common partners: {', '.join(name for name, _ in partner_counts.most_common(3)) or 'none'}",
            f"- Common topics: {', '.join(topic for topic, _ in topic_counts.most_common(3)) or 'none'}",
            f"- Missed opportunities: {len(missed)}",
            "- Top memories:",
        ])
        if top_memories:
            lines.extend([f"  - {memory.memory_id}: {memory.summary}" for memory in top_memories])
        else:
            lines.append("  - none in this smoke run")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def write_timeline(simulation, path: Path | None = None) -> Path:
    path = config.TIMELINE_PATH if path is None else path
    items = []
    for event in simulation.workflow_event_log:
        if event["event_type"] in {"patient_arrival", "bed_assigned", "task_completed"}:
            items.append((int(event["timestep"]), f"workflow: {event['event_type']} patient={event.get('patient_id')} {event.get('task_name')}"))
    for event in simulation.interaction_log:
        items.append((int(event["timestep"]), f"communication: {event['agent_1_name']} - {event['agent_2_name']} {event['topic']} in {event['zone_id']}"))
    items.sort(key=lambda item: item[0])
    lines = [f"# Timeline Seed {simulation.random_seed}", ""]
    for timestep, text in items[:200]:
        lines.append(f"- {timestep // 60:02d}:{timestep % 60:02d} {text}")
    path.write_text("\n".join(lines))
    return path


def write_supervisor_outputs(simulation) -> dict[str, str]:
    paths = {
        "simulation_overview": str(write_simulation_overview(simulation)),
        "interaction_heatmap": str(write_interaction_heatmap(simulation)),
        "agent_experience_cards": str(write_agent_experience_cards(simulation)),
        "timeline": str(write_timeline(simulation)),
    }
    simulation.save_trace_outputs()
    return paths
