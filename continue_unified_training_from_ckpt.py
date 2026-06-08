"""
Continue `run_unified`-style hierarchical training exactly like Phase-1 **A_base / W_MASTER**
(same `_BASE_HP`, same three-env mixture, same `unified_training_loop`), but **warm-start**
from YOUR `*_agent.pth` / `*_master.pth` pair instead of the baked-in Q02 path.

Reference layout (same as `python run_unified.py` Phase 1):
    <experiment_home>/A_base/W_MASTER/s123/
        episode_metrics.csv
        trained_model_* | best/ckpt_*

This script writes a **new** tree (no overwrite), e.g.
    experiment_runs/CONTINUE_unified_<datetime>/A_base/W_MASTER/s123/

The **zero_master** test in `run_models_evaluation_suite` is *eval-time* (embeddings zeroed).
Unified training always feeds real master outputs during rollouts — continuing this run
strengthens behaviour under the **full hierarchy**; fixing “bad policy when embeddings are 0”
is not automatic and should be checked with the eval suite after training.

Example
-------
  py -3.11 continue_unified_training_from_ckpt.py ^
    --agent models_to_check/agent/ckpt_agent6.pth ^
    --master models_to_check/master/ckpt_master6.pth ^
    --experiment-home experiment_runs/full_26_04_2026-11_40_39_CONT ^
    --seed 123

For IDE “Run current file”, see ``train_unified_full_scenarios.py`` (same ``_run_one_condition``
recipe; scenario pools come from merged ``scenarios.py`` automatically).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import run_unified as ru


def _find_config(name: str) -> ru.RunConfig:
    for c in ru.ALL_CONFIGS:
        if c.name == name:
            return c
    raise SystemExit(f"No RunConfig named {name!r} in run_unified.ALL_CONFIGS")


def _ablation_for_condition(label: str) -> bool:
    u = label.upper()
    if u == "W_MASTER":
        return False
    if u == "NO_MASTER":
        return True
    raise SystemExit("--condition must be W_MASTER or NO_MASTER")


def main() -> None:
    default_ref = os.path.join(
        "experiment_runs", "full_26_04_2026-11_40_39", "A_base", "W_MASTER", "s123"
    )
    parser = argparse.ArgumentParser(
        description="Warm-start unified A_base/W_MASTER training from two .pth files."
    )
    parser.add_argument("--agent", required=True, help="Path to *_agent.pth")
    parser.add_argument("--master", required=True, help="Path to *_master.pth")
    parser.add_argument("--config-name", type=str, default="A_base", help="Must match run_unified RunConfig name")
    parser.add_argument("--condition", type=str, default="W_MASTER")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Override total_episodes (0 = use RunConfig default, e.g. 2500 for A_base).",
    )
    parser.add_argument(
        "--experiment-home",
        type=str,
        default="",
        help=(
            "Directory that will contain <config>/<condition>/s<seed> subfolders. "
            "Default: experiment_runs/CONTINUE_unified_<timestamp>."
        ),
    )
    parser.add_argument(
        "--reference-run",
        type=str,
        default=default_ref,
        help="Only checked for existence (info). Does not read hyperparameters from disk.",
    )
    args = parser.parse_args()

    ag = os.path.normpath(os.path.abspath(os.path.expanduser(args.agent.strip())))
    ms = os.path.normpath(os.path.abspath(os.path.expanduser(args.master.strip())))
    if not os.path.isfile(ag) or not os.path.isfile(ms):
        sys.exit("Provide valid --agent and --master .pth files.")

    ref = os.path.normpath(os.path.abspath(os.path.expanduser(args.reference_run.strip())))
    if os.path.isdir(ref):
        print(f"Reference run folder exists (recipe match): {ref}")
    else:
        print(f"(Note) Reference path not found locally: {ref}")

    base_cfg = _find_config(args.config_name)
    if int(args.episodes) > 0:
        cfg = replace(base_cfg, total_episodes=int(args.episodes))
    else:
        cfg = base_cfg

    if (args.experiment_home or "").strip():
        home = os.path.normpath(os.path.abspath(os.path.expanduser(args.experiment_home.strip())))
    else:
        home = os.path.join(_REPO, "experiment_runs", f"CONTINUE_unified_{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}")
    os.makedirs(home, exist_ok=True)

    ablation = _ablation_for_condition(args.condition)
    print(f"Output root: {home}")
    print(f"Config: {cfg.name} | episodes={cfg.total_episodes} | conflict_schedule={cfg.conflict_schedule}")

    ru._run_one_condition(
        cfg,
        args.condition,
        ablation,
        home,
        int(args.seed),
        warmstart_agent_path=ag,
        warmstart_master_path=ms,
    )
    print("\nDone.")
    out = os.path.join(home, cfg.name, args.condition, f"s{int(args.seed)}")
    print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
