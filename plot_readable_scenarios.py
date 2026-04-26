from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import run_unified as unified
from src import project_globals
from src.model.agent_handler import Driver
from src.experiment.scenarios import (
    base_complete_scenarios_6_cars,
    conflict_base_scenarios,
    roundabout_base_scenarios,
    roundabout_conflict_base_scenarios,
    double_intersection_base_scenarios,
    double_intersection_conflict_base_scenarios,
)


BASE_LONG = 40.0
AGENT_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]


def _unwrap_road_env(env):
    while hasattr(env, "env") and not hasattr(env, "road"):
        env = env.env
    return env


def _make_road_network(env_id: str, out_dir: str):
    project_globals.after_is_arrived_flags = [False] * 6
    exp = unified._make_experiment(out_dir, 10, env_id=env_id)
    exp.CONFIG = unified._make_env_config(env_id)
    driver = Driver(exp)
    driver.highway_env.reset()
    inner = _unwrap_road_env(driver.highway_env)
    return driver, inner.road.network


def _bfs_node_path(graph: dict, start: str, goal: str) -> list[str]:
    if start == goal:
        return [start]
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        node = path[-1]
        for nxt in graph.get(node, {}):
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt == goal:
                return new_path
            seen.add(nxt)
            q.append(new_path)
    return []


def _lane_key_for_edge(road_net, from_node: str, to_node: str, preferred_idx: int = 0):
    lanes = road_net.graph.get(from_node, {}).get(to_node, [])
    if not lanes:
        return None
    idx = min(preferred_idx, len(lanes) - 1)
    return (from_node, to_node, idx)


def _route_lane_keys(road_net, start_lane_key: tuple, destination: str) -> list[tuple]:
    route = [start_lane_key]
    node_path = _bfs_node_path(road_net.graph, start_lane_key[1], destination)
    if len(node_path) < 2:
        return route
    for a, b in zip(node_path[:-1], node_path[1:]):
        lk = _lane_key_for_edge(road_net, a, b, start_lane_key[2])
        if lk is not None:
            route.append(lk)
    return route


def _sample_route_points(road_net, start_lane_key: tuple, destination: str, offset: float) -> np.ndarray:
    lane_keys = _route_lane_keys(road_net, start_lane_key, destination)
    points: list[np.ndarray] = []
    for idx, lane_key in enumerate(lane_keys):
        try:
            lane = road_net.get_lane(lane_key)
        except Exception:
            continue
        start_s = max(0.0, min(lane.length, BASE_LONG + offset)) if idx == 0 else 0.0
        n = max(8, int(lane.length / 4))
        for s in np.linspace(start_s, lane.length, n):
            points.append(np.asarray(lane.position(float(s), 0), dtype=float))
    if not points:
        lane = road_net.get_lane(start_lane_key)
        points = [np.asarray(lane.position(BASE_LONG + offset, 0), dtype=float)]
    return np.vstack(points)


def _draw_road_network(ax, road_net) -> None:
    for from_node in road_net.graph:
        for to_node in road_net.graph[from_node]:
            for lane in road_net.graph[from_node][to_node]:
                pts = np.asarray([
                    lane.position(float(s), 0)
                    for s in np.linspace(0.0, lane.length, 32)
                ])
                ax.plot(pts[:, 0], pts[:, 1], color="#D0D0D0", linewidth=1.2, zorder=1)


def _draw_route_arrow(ax, pts: np.ndarray, color: str, zorder: int) -> None:
    if len(pts) < 2:
        return
    mid = max(1, min(len(pts) - 2, len(pts) // 2))
    ax.annotate(
        "",
        xy=(pts[mid + 1, 0], pts[mid + 1, 1]),
        xytext=(pts[mid - 1, 0], pts[mid - 1, 1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=12),
        zorder=zorder,
    )


def _origin_label(lane_key: tuple) -> str:
    return str(lane_key[0])


def _turn_label(origin: str, destination: str) -> str:
    if origin == destination:
        return "same"
    return f"{origin}->{destination}"


def _plot_one_scenario(
    ax,
    road_net,
    scenario: dict[str, Any],
    title: str,
    *,
    show_legend: bool,
) -> list[dict[str, Any]]:
    _draw_road_network(ax, road_net)
    manifest_rows = []

    for agent_idx, (lane_key, destination, offset) in enumerate(scenario.get("agents", [])):
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        pts = _sample_route_points(road_net, lane_key, destination, offset)
        origin = _origin_label(lane_key)
        label = _turn_label(origin, destination)
        group = "LM1" if agent_idx < 3 else "LM2"

        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.6, alpha=0.82, zorder=3)
        _draw_route_arrow(ax, pts, color, zorder=4)
        ax.scatter(pts[0, 0], pts[0, 1], s=110, c=color, edgecolors="black", linewidths=0.8, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=70, marker="X", c=color, edgecolors="black", linewidths=0.6, zorder=5)
        ax.text(
            pts[0, 0],
            pts[0, 1],
            f" A{agent_idx}",
            color="black",
            fontsize=8,
            weight="bold",
            ha="left",
            va="bottom",
            zorder=6,
        )
        label_idx = min(len(pts) - 1, max(0, len(pts) // 3))
        ax.text(
            pts[label_idx, 0],
            pts[label_idx, 1],
            f"A{agent_idx} {label}",
            color=color,
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.2),
            zorder=6,
        )
        manifest_rows.append({
            "agent": f"A{agent_idx}",
            "local_master": group,
            "origin": origin,
            "destination": destination,
            "route": label,
            "offset": offset,
        })

    for static_idx, (lane_key, destination, offset) in enumerate(scenario.get("static", [])):
        pts = _sample_route_points(road_net, lane_key, destination, offset)
        ax.plot(pts[:, 0], pts[:, 1], color="#616161", linewidth=1.8, linestyle="--", alpha=0.65, zorder=2)
        ax.scatter(pts[0, 0], pts[0, 1], s=70, c="#9E9E9E", marker="s", edgecolors="black", linewidths=0.6, zorder=5)
        ax.text(pts[0, 0], pts[0, 1], f" S{static_idx}", fontsize=7, weight="bold", zorder=6)

    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w", label=f"A{i} ({'LM1' if i < 3 else 'LM2'})",
                   markerfacecolor=AGENT_COLORS[i], markeredgecolor="black", markersize=7)
            for i in range(6)
        ]
        handles += [
            Line2D([0], [0], marker="o", color="black", label="start", markerfacecolor="white", markersize=6),
            Line2D([0], [0], marker="X", color="black", label="destination", markerfacecolor="white", markersize=6),
            Line2D([0], [0], color="#616161", linestyle="--", label="static/NPC route"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)

    return manifest_rows


def _scenario_sets() -> list[tuple[str, str, list[tuple[int, str, dict[str, Any]]]]]:
    return [
        (
            "intersection",
            "RELintersection-v0",
            [(i, "regular", s) for i, s in enumerate(base_complete_scenarios_6_cars[:5])]
            + [(i, "conflict", s) for i, s in enumerate(conflict_base_scenarios[:3])],
        ),
        (
            "roundabout",
            "RELroundabout-v0",
            [(i, "regular", s) for i, s in enumerate(roundabout_base_scenarios[:5])]
            + [(i, "conflict", s) for i, s in enumerate(roundabout_conflict_base_scenarios[:3])],
        ),
        (
            "double_intersection",
            "RELdouble-intersection-v0",
            [(i, "regular", s) for i, s in enumerate(double_intersection_base_scenarios[:5])]
            + [(i, "conflict", s) for i, s in enumerate(double_intersection_conflict_base_scenarios[:3])],
        ),
    ]


def generate_readable_layouts(output_root: str) -> str:
    out_dir = os.path.join(output_root, "scenario_layouts", "route_readable")
    individual_dir = os.path.join(out_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for env_label, env_id, scenarios in _scenario_sets():
        driver, road_net = _make_road_network(env_id, out_dir)
        try:
            cols = 4
            rows = int(np.ceil(len(scenarios) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.4 * rows), squeeze=False)

            for plot_idx, (scenario_idx, scenario_type, scenario) in enumerate(scenarios):
                r, c = divmod(plot_idx, cols)
                title = f"{env_label} {scenario_type} #{scenario_idx}"
                rows_for_scenario = _plot_one_scenario(
                    axes[r][c],
                    road_net,
                    scenario,
                    title,
                    show_legend=(plot_idx == 0),
                )
                for row in rows_for_scenario:
                    manifest.append({
                        "env": env_label,
                        "scenario_type": scenario_type,
                        "scenario_index_in_subset": scenario_idx,
                        **row,
                    })

                fig_single, ax_single = plt.subplots(figsize=(8, 7))
                _plot_one_scenario(ax_single, road_net, scenario, title, show_legend=True)
                fig_single.tight_layout()
                fig_single.savefig(
                    os.path.join(individual_dir, f"{env_label}_{scenario_type}_{scenario_idx:03d}.png"),
                    dpi=220,
                )
                plt.close(fig_single)

            for idx in range(len(scenarios), rows * cols):
                r, c = divmod(idx, cols)
                axes[r][c].set_visible(False)

            fig.suptitle(
                f"Readable Scenario Routes - {env_label}\n"
                "Circle=start, X=destination, line/arrow=planned route, A0-A2=LM1, A3-A5=LM2",
                fontsize=15,
                weight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(out_dir, f"readable_routes_{env_label}.png"), dpi=200)
            plt.close(fig)
        finally:
            try:
                driver.highway_env.close()
            except Exception:
                pass

    with open(os.path.join(out_dir, "route_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "scenario_type",
                "scenario_index_in_subset",
                "agent",
                "local_master",
                "origin",
                "destination",
                "route",
                "offset",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate readable route-based scenario diagrams.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("experiment_runs", "full_26_04_2026-11_40_39"),
        help="Experiment root that will receive scenario_layouts/route_readable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    output = generate_readable_layouts(parse_args().output_root)
    print(f"Saved readable scenario layouts to: {output}")
