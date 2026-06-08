"""
100 double-intersection scenarios for OOD-style G03 eval (not seen during training).

Composition:
  • 5 exact ``DOUBLE_INTERSECTION_HELD_OUT_INDICES`` layouts (regular held-out).
  • 95 deterministic jittered copies: same routes/destinations, small longitudinal-offset
    perturbations per agent (similar flavour to training layouts, not 1:1).

Training uses all regular indices except (held-out ∪ excluded), so these 100 are strictly
outside the training set for the 5 exact bases; the 95 are additional continuations that do
not match any discrete training scenario index.
"""

from __future__ import annotations

import copy
import random
from typing import Any

# Noise steps along lane (meters-ish; matches stagger values used in base scenarios, ±25).
_JITTER_CHOICES: tuple[int, ...] = (
    -25, -22, -18, -15, -12, -10, -8, -6, -5, -4, -3, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 25
)


def _jitter_one_scenario(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    agents: list = []
    for ag in base["agents"]:
        (pose, dest, off) = ag
        delta = int(rng.choice(_JITTER_CHOICES))
        if delta == 0:
            delta = 4
        agents.append((pose, dest, int(off) + delta))
    return {"agents": agents, "static": copy.deepcopy(base.get("static", []))}


def build_double_intersection_100_ood_eval_scenarios() -> list[dict[str, Any]]:
    from src.experiment.scenarios import (
        DOUBLE_INTERSECTION_HELD_OUT_INDICES,
        double_intersection_base_scenarios,
    )

    idx = sorted(DOUBLE_INTERSECTION_HELD_OUT_INDICES)
    bases = [copy.deepcopy(double_intersection_base_scenarios[i]) for i in idx]
    out: list[dict[str, Any]] = [copy.deepcopy(b) for b in bases]

    for j in range(95):
        rng = random.Random(9_001_003 + j * 17)
        bi = j % len(bases)
        out.append(_jitter_one_scenario(bases[bi], rng))

    if len(out) != 100:
        raise RuntimeError(f"expected 100 scenarios, got {len(out)}")
    return out


# Built once; scenarios list depends on fully-populated double_intersection_base_scenarios.
DOUBLE_INTERSECTION_100_OOD_EVAL_SCENARIOS: list[dict[str, Any]] = build_double_intersection_100_ood_eval_scenarios()
