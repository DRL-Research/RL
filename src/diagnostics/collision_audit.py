# Code and comments only in English.

"""
Post-step snapshots and heuristics to sanity-check whether collisions come from
HighwayEnv physics (vehicle.crashed) and whether they occur after meaningful agent-driven motion.

This does not prove causality in the legal sense; it flags suspicious patterns
(e.g. crash on step 1–2) that may indicate spawn/scenario issues vs agent-induced contacts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Typical vehicle length scale in highway-env intersection (meters, order-of-magnitude).
_CONTACT_DIST_THRESHOLD_M = 8.0


def _active_controlled_indices(inner) -> List[int]:
    out = []
    for i, v in enumerate(inner.controlled_vehicles):
        if hasattr(v, "is_arrived") and getattr(v, "is_arrived", False):
            continue
        out.append(i)
    return out


def _xy(vehicle) -> Tuple[float, float]:
    p = vehicle.position
    return float(p[0]), float(p[1])


def min_pairwise_distance_xy(inner, indices: List[int]) -> Optional[float]:
    if len(indices) < 2:
        return None
    pts = [_xy(inner.controlled_vehicles[i]) for i in indices]
    best = None
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            d = float(np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
            if best is None or d < best:
                best = d
    return best


def min_dist_active_to_others(inner) -> Tuple[Optional[float], int]:
    """
    Minimum distance from any *active* controlled vehicle to any other road vehicle
    that is not in the controlled set (static / NPC).
    """
    controlled = inner.controlled_vehicles
    c_set = set(id(v) for v in controlled)
    active_idx = _active_controlled_indices(inner)
    others = [v for v in inner.road.vehicles if id(v) not in c_set]
    if not others or not active_idx:
        return None, len(others)

    best = None
    for i in active_idx:
        px, py = _xy(controlled[i])
        for ov in others:
            ox, oy = _xy(ov)
            d = float(np.hypot(px - ox, py - oy))
            if best is None or d < best:
                best = d
    return best, len(others)


def build_step_trace_row(
    step: int,
    actions_scalar: List[Any],
    step_reward: float,
    info: Dict[str, Any],
    inner,
    experiment,
) -> Dict[str, Any]:
    """One dict per env step (call after env.step), for JSON-serializable logs."""
    n = len(inner.controlled_vehicles)
    crashed_flags = [bool(v.crashed) for v in inner.controlled_vehicles]
    arrived_flags = [bool(getattr(v, "is_arrived", False)) for v in inner.controlled_vehicles]
    positions = [[float(v.position[0]), float(v.position[1])] for v in inner.controlled_vehicles]
    speeds = [float(np.linalg.norm(np.asarray(v.velocity, dtype=np.float64))) for v in inner.controlled_vehicles]

    active_ix = _active_controlled_indices(inner)
    min_pair = min_pairwise_distance_xy(inner, active_ix)
    min_other, n_other = min_dist_active_to_others(inner)

    agents_rewards = info.get("agents_rewards")
    if agents_rewards is not None:
        agents_rewards_list = [float(x) for x in agents_rewards]
    else:
        agents_rewards_list = []

    cr = float(getattr(experiment, "COLLISION_REWARD", -50.0))
    min_agent_r = min(agents_rewards_list) if agents_rewards_list else None

    return {
        "step": int(step),
        "actions_scalar": [int(x) for x in actions_scalar],
        "step_reward": float(step_reward),
        "agents_rewards": agents_rewards_list,
        "crashed_flags": crashed_flags,
        "arrived_flags": arrived_flags,
        "positions_xy": positions,
        "speeds": speeds,
        "n_active_controlled": len(active_ix),
        "min_pairwise_dist_active_m": min_pair,
        "min_dist_active_to_uncontrolled_m": min_other,
        "n_uncontrolled_vehicles": int(n_other),
        "collision_reward_config": cr,
        "min_agent_reward_matches_collision_penalty": (
            min_agent_r is not None and abs(min_agent_r - cr) < 1e-3
        ),
    }


def classify_collision_episode(
    trace: List[Dict[str, Any]],
    crashed_episode: bool,
    episode_sum_reward: float,
    collision_reward: float,
) -> Dict[str, Any]:
    """
    Heuristic labels for one episode that ended with crashed_episode=True
    (from episode_utils: any v.crashed at terminal done).
    """
    if not crashed_episode:
        return {"outcome": "success_or_truncated_no_flag", "crashed": False}

    first_crash_step: Optional[int] = None
    which_at_first: List[int] = []
    for row in trace:
        cf = row.get("crashed_flags") or []
        if any(cf):
            first_crash_step = int(row["step"])
            which_at_first = [i for i, c in enumerate(cf) if c]
            break

    pre_crash_min_pair: Optional[float] = None
    if first_crash_step is not None and first_crash_step >= 2:
        prev = [r for r in trace if r["step"] == first_crash_step - 1]
        if prev:
            pre_crash_min_pair = prev[0].get("min_pairwise_dist_active_m")

    # Displacement of first active car from first recorded step to step before crash
    path_len = None
    if first_crash_step is not None and len(trace) >= 2 and first_crash_step >= 2:
        r0 = trace[0]
        r1 = [r for r in trace if r["step"] == first_crash_step - 1]
        if r0.get("positions_xy") and r1:
            p0 = np.asarray(r0["positions_xy"][0], dtype=float)
            p1 = np.asarray(r1[0]["positions_xy"][0], dtype=float)
            path_len = float(np.linalg.norm(p1 - p0))

    labels: List[str] = []
    if first_crash_step is None:
        labels.append("CRASH_FLAG_BUT_NO_STEP_WITH_CRASHED_TRUE")
    elif first_crash_step <= 2:
        labels.append("VERY_EARLY_CRASH_SUSPECT_SPAWN_OR_NUMERICS")
    else:
        labels.append("CRASH_AFTER_MULTIPLE_STEPS_AGENT_IN_THE_LOOP")

    if pre_crash_min_pair is not None and pre_crash_min_pair < _CONTACT_DIST_THRESHOLD_M:
        labels.append("TIGHT_PAIRWISE_SPACING_BEFORE_CRASH")

    reward_matches = abs(float(episode_sum_reward) - float(collision_reward)) < 1.0
    # Episode return is sum of step rewards; often not exactly collision_reward — weak check
    if reward_matches:
        labels.append("EPISODE_RETURN_NEAR_SINGLE_COLLISION_REWARD")

    return {
        "crashed": True,
        "first_crash_step": first_crash_step,
        "crashed_vehicle_indices_first": which_at_first,
        "min_pairwise_dist_step_before_crash_m": pre_crash_min_pair,
        "approx_displacement_car0_to_pre_crash_m": path_len,
        "episode_sum_reward": float(episode_sum_reward),
        "collision_reward": float(collision_reward),
        "heuristic_labels": labels,
    }


def aggregate_audit(episode_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    crashed_eps = [e for e in episode_reports if e.get("classification", {}).get("crashed")]
    n_c = len(crashed_eps)
    n_total = len(episode_reports)
    early = sum(
        1
        for e in crashed_eps
        if any(
            "VERY_EARLY_CRASH" in x
            for x in e.get("classification", {}).get("heuristic_labels", [])
        )
    )
    agent_loop = sum(
        1
        for e in crashed_eps
        for x in e.get("classification", {}).get("heuristic_labels", [])
        if "CRASH_AFTER_MULTIPLE_STEPS_AGENT_IN_THE_LOOP" in x
    )
    return {
        "episodes": n_total,
        "collision_episodes": n_c,
        "fraction_collision": (n_c / n_total) if n_total else 0.0,
        "collision_episodes_very_early_heuristic": early,
        "collision_episodes_after_steps_heuristic": agent_loop,
        "interpretation": (
            "If 'very_early' dominates, inspect scenarios/spawn overlap before blaming policy. "
            "If 'after_steps' dominates, crashes follow agent-chosen actions over several steps "
            "(consistent with real collisions in sim)."
        ),
    }


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
