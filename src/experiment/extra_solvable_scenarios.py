# Code and comments only in English.
"""
Fifty additional *conservative* (stagger-heavy) base scenarios per layout, appended to the
pools in `scenarios.py`. Each pattern keeps strong temporal separation on shared approaches
(≥25–28 m between same-lane followers) to stay in the "solvable with modest coordination" regime.

Training ideas so agents lean on master (LM) broadcasts instead of kinematics-only shortcuts:
  - Avoid auxiliary losses that reconstruct master embeddings from local observations alone.
  - Optional short scheduled ``zero_master`` / embedding-dropout on the agent side during
    training (with main task reward) so policies cannot assume the vector is always present.
  - Track the eval gap between ``normal`` and ``zero_master``; if gap vanishes early, tighten
    agent-side regularization toward using the fused master features.

Smoke verification: ``py -3.11 verify_extra_scenarios.py`` after import of merged ``scenarios``.
"""

from __future__ import annotations

EXTRA_INTERSECTION_6CAR_COUNT = 50
EXTRA_ROUNDABOUT_COUNT = 50
EXTRA_DOUBLE_INTERSECTION_COUNT = 50


def build_extra_intersection_6car() -> list[dict]:
    """
    Strictly sequential staggering on all four approaches (no pack of four at offset 0).
    Same-lane followers keep >= 25 m gap; spacing varies slowly with k for diversity.
    """
    out: list[dict] = []
    for k in range(EXTRA_INTERSECTION_6CAR_COUNT):
        d = 14 + (k % 4) * 2  # 14,16,18,20
        tail = 5 * d + 8 + (k // 4) * 2
        out.append({
            "agents": [
                (('o0', 'ir0', 0), "o3", 0),
                (('o1', 'ir1', 0), "o0", -d),
                (('o2', 'ir2', 0), "o1", -2 * d),
                (('o3', 'ir3', 0), "o2", -3 * d),
                (('o0', 'ir0', 0), "o2", -4 * d - (k % 11)),
                (('o1', 'ir1', 0), "o3", -tail),
            ],
            "static": [],
        })
    return out


def build_extra_roundabout() -> list[dict]:
    """Mostly sequential ring entry — opposites active; others stagger back with mild k-spread."""
    out: list[dict] = []
    for k in range(EXTRA_ROUNDABOUT_COUNT):
        s = -18 - (k % 7) * 7
        out.append({
            "agents": [
                (('o0', 'ir0', 0), "o3", 0),
                (('o2', 'ir2', 0), "o1", 0),
                (('o1', 'ir1', 0), "o0", max(-88, s - 12 - k // 3)),
                (('o3', 'ir3', 0), "o2", max(-88, s - 18 - k // 3)),
                (('o0', 'ir0', 0), "o2", max(-90, -48 - k)),
                (('o2', 'ir2', 0), "o0", max(-90, -52 - k)),
            ],
            "static": [],
        })
    return out


def build_extra_double_intersection() -> list[dict]:
    """
    Variations on mild within-intersection flow + one convoy gap on A — avoids full A↔B swap storms.
    """
    out: list[dict] = []
    for k in range(EXTRA_DOUBLE_INTERSECTION_COUNT):
        gap = -22 - (k % 6) * 4
        tail = max(-78, gap - (k // 6) * 3)
        out.append({
            "agents": [
                (('A_o0', 'A_ir0', 0), "A_o2", 0),
                (('A_o0', 'A_ir0', 0), "A_o1", tail),
                (('A_o2', 'A_ir2', 0), "A_o0", 0),
                (('B_o0', 'B_ir0', 0), "B_o2", 0),
                (('B_o2', 'B_ir2', 0), "B_o0", 0),
                (('B_o3', 'B_ir3', 0), "B_o2", max(-76, -20 - k // 4)),
            ],
            "static": [],
        })
    return out


EXTRA_INTERSECTION_6CAR_50 = build_extra_intersection_6car()
EXTRA_ROUNDABOUT_50 = build_extra_roundabout()
EXTRA_DOUBLE_INTERSECTION_50 = build_extra_double_intersection()
