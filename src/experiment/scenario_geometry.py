# Code and comments only in English.

"""
Semantic geometry tags for RELintersection 6-controlled-car scenarios.

The physical graph is always a single 4-way junction with curved inner arcs
(mini roundabout arms). We summarise *scenario intent* rather than CAD topology:

  - Crossing / straight-heavy: mostly straight crossings through the junction.
  - Turn-arc-heavy: many left/right turns that use CircularLane arcs.
  - Mixed-maneuvers: no single manoeuvre dominates.
  - Conflict-with-static-npc: sampled from the conflict/static-heavy pool.

This is computed from `base_complete_scenarios_6_cars` manoeuvre counts.
"""

from __future__ import annotations

from typing import Any, Dict


def _turn_label(origin_corner: int, dest_outer: str) -> str:
    d = int(dest_outer[1])
    diff = (d - origin_corner) % 4
    if diff == 2:
        return "straight"
    if diff == 1:
        return "right"
    if diff == 3:
        return "left"
    return "same_arm"


def _count_turns_regular_base(scenario_entry: Dict[str, Any]) -> Dict[str, int]:
    ct = {"straight": 0, "right": 0, "left": 0, "same_arm": 0}
    for triplet in scenario_entry["agents"]:
        lane_key = triplet[0]
        outer = lane_key[0]
        oc = int(outer[-1])
        dest = triplet[1]
        lbl = _turn_label(oc, dest)
        if lbl != "same_arm":
            ct[lbl] += 1
        else:
            ct["same_arm"] += 1
    return ct


def manoeuvre_geometry_label_from_counts(ct: Dict[str, int]) -> str:
    n = ct["straight"] + ct["right"] + ct["left"]
    if n <= 0:
        return "mixed_maneuvers"
    frac_s = ct["straight"] / n
    frac_r = ct["right"] / n
    frac_l = ct["left"] / n
    if frac_s >= 0.55:
        return "crossing_straight_heavy"
    if frac_r + frac_l >= 0.55:
        return "turn_arc_heavy"
    return "mixed_maneuvers"


def regular_base_geometry_bucket(base_idx: int, base_scenarios_table) -> str:
    if base_idx < 0 or base_idx >= len(base_scenarios_table):
        return "mixed_maneuvers"
    entry = base_scenarios_table[base_idx]
    ct = _count_turns_regular_base(entry)
    return manoeuvre_geometry_label_from_counts(ct)


def episode_geometry_fields(
    *,
    scenario_pool: str,
    scenario_index: int,
    scenario_base: int,
    base_scenarios_table,
) -> Dict[str, Any]:
    pool = (scenario_pool or "unknown").lower()
    if pool == "conflict":
        bucket = "static_conflict_heavy"
    else:
        bucket = regular_base_geometry_bucket(int(scenario_base), base_scenarios_table)
    return {
        "scenario_pool": pool,
        "scenario_index": int(scenario_index),
        "scenario_base": int(scenario_base),
        "geometry_bucket": bucket,
    }
