"""
Collision audit — IDE Run or: python run_collision_audit.py

Edit the CONFIG block below (no CLI). Repo root is set from this file’s path.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

# ── CONFIG: change these only ───────────────────────────────────────────────
EPISODES = 50
OUT_JSON = "experiments/collision_audit_run.json"
# Prefix for load_models(); loads PREFIX_agent.pth and PREFIX_master.pth — or "" for random init
LOAD_PREFIX = ""
# e.g. r"experiments/final_12_04_2026-12_44_13/trained_model"
STOCHASTIC_AGENT = False
DUMP_FIRST_CRASH_TRACE = "experiments/first_crash_trace.json"  # set to "" to skip
SEED = -1  # >= 0 for reproducibility
PRINT_EVERY_N_EPISODES = 10
# ───────────────────────────────────────────────────────────────────────────

from analyze_collisions import run_collision_audit


def main() -> None:
    run_collision_audit(
        episodes=EPISODES,
        out=OUT_JSON,
        load=LOAD_PREFIX,
        stochastic=STOCHASTIC_AGENT,
        dump_trace=DUMP_FIRST_CRASH_TRACE or "",
        seed=SEED,
        verbose_every=PRINT_EVERY_N_EPISODES,
    )


if __name__ == "__main__":
    main()
