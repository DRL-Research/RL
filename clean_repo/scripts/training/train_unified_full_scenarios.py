"""
Unified hierarchical training — same pipeline as ``continue_unified_training_from_ckpt.py``
and ``run_unified._run_one_condition`` (three environments, equal sampling weights).

Scenario pools (regular + conflict sampling inside each env) come entirely from
``src.experiment.scenarios`` after it merges hand-authored lists with
``extra_solvable_scenarios`` — **no manual scenario list here**. Each episode calls
``env.reset()``, which draws from that merged pool.

IDE usage
---------
1. Optionally set ``AGENT_PATH`` and ``MASTER_PATH`` below (both nonempty for warm-start).
   Leave both ``""`` to use ``run_unified.PRETRAINED_CHECKPOINT`` / random fallback logic.
2. Press Run on this file.

Output root defaults to ``experiment_runs/UNIFIED_FULL_SCENARIOS_<timestamp>/``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# repo root + every scripts/ sub-folder go on sys.path so cross-script imports
# (e.g. ``import run_scalability_suite``) keep working from any category folder.
for _p in (
    _REPO,
    os.path.join(_REPO, "scripts", "training"),
    os.path.join(_REPO, "scripts", "evaluation"),
    os.path.join(_REPO, "scripts", "visualization"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO)

# ── Edit these for IDE runs ───────────────────────────────────────────────────
# Use forward slashes or raw strings (r"..."); plain backslashes break paths (\a = bell).
AGENT_PATH = "models_to_check/agent/ckpt_agent6.pth"
MASTER_PATH = "models_to_check/master/ckpt_master6.pth"
SEED = 123
CONFIG_NAME = "A_base"
CONDITION = "W_MASTER"
EPISODES_OVERRIDE = 7500
EXPERIMENT_HOME = "experiment_runs/fine_tune_id6"

import run_unified as ru
from src.experiment import scenarios as sc


def _print_scenario_inventory() -> None:
    try:
        from src.experiment import extra_solvable_scenarios as ex
        extra_note = (
            f"+ {ex.EXTRA_INTERSECTION_6CAR_COUNT} intersection / "
            f"{ex.EXTRA_ROUNDABOUT_COUNT} roundabout / "
            f"{ex.EXTRA_DOUBLE_INTERSECTION_COUNT} double bases appended"
        )
    except Exception:
        extra_note = "(extra_solvable_scenarios not imported)"

    # Conflict pools unchanged by the procedural append (still sampled via conflict_ratio).
    print("\n--- Scenario pools loaded by environments (merged scenarios.py) ---")
    print(f"  Intersection regular bases: {len(sc.base_complete_scenarios_6_cars)}  ({extra_note})")
    print(f"  Intersection conflict bases: {len(sc.conflict_base_scenarios)}")
    print(f"  Roundabout regular bases: {len(sc.roundabout_base_scenarios)}")
    print(f"  Roundabout conflict bases: {len(sc.roundabout_conflict_base_scenarios)}")
    print(f"  Double regular bases: {len(sc.double_intersection_base_scenarios)}")
    print(f"  Double conflict bases: {len(sc.double_intersection_conflict_base_scenarios)}")
    print("Training visits all three envs; reset() samples from these pools automatically.\n")


def main() -> None:
    _print_scenario_inventory()

    wa = (AGENT_PATH or "").strip()
    wm = (MASTER_PATH or "").strip()
    if (wa or wm) and not (wa and wm):
        sys.exit("Set both AGENT_PATH and MASTER_PATH, or leave both empty for default init.")

    base_cfg = None
    for c in ru.ALL_CONFIGS:
        if c.name == CONFIG_NAME:
            base_cfg = c
            break
    if base_cfg is None:
        sys.exit(f"No RunConfig named {CONFIG_NAME!r} in run_unified.ALL_CONFIGS")

    cfg = (
        replace(base_cfg, total_episodes=int(EPISODES_OVERRIDE))
        if int(EPISODES_OVERRIDE) > 0
        else base_cfg
    )

    if (EXPERIMENT_HOME or "").strip():
        home = os.path.normpath(os.path.abspath(os.path.expanduser(EXPERIMENT_HOME.strip())))
    else:
        home = os.path.join(
            _REPO,
            "experiment_runs",
            f"UNIFIED_FULL_SCENARIOS_{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}",
        )
    os.makedirs(home, exist_ok=True)

    cond_u = CONDITION.strip().upper()
    if cond_u == "W_MASTER":
        ablation = False
    elif cond_u == "NO_MASTER":
        ablation = True
    else:
        sys.exit("CONDITION must be W_MASTER or NO_MASTER")

    ws_kw = {}
    if wa and wm:
        ws_kw["warmstart_agent_path"] = os.path.abspath(os.path.expanduser(wa))
        ws_kw["warmstart_master_path"] = os.path.abspath(os.path.expanduser(wm))

    print(f"Output root: {home}")
    print(f"Config={cfg.name} episodes={cfg.total_episodes} condition={CONDITION} seed={SEED}")

    ru._run_one_condition(cfg, CONDITION, ablation, home, int(SEED), **ws_kw)

    print("\nDone.")
    print(f"Artifacts: {os.path.join(home, cfg.name, CONDITION, f's{int(SEED)}')}")


if __name__ == "__main__":
    main()
