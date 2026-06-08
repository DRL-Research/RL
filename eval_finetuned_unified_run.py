"""
Post fine-tune evaluation for a unified hierarchical run directory.

**Ready to run from repo root** with no arguments if this folder exists::

    experiment_runs/fine_tune_id6/A_base/W_MASTER/s123

Override with ``--run-dir``, or env ``EVAL_RUN_DIR``.

Steps:
  1) Summarize ``episode_metrics.csv`` (unless ``--eval-only``).
  2) Load ``best/ckpt_*`` (or ``trained_model_*`` / ``auto``).
  3) Test 3 envs × (regular | held-out | conflict-only).

Examples::

  py -3 eval_finetuned_unified_run.py
  py -3 eval_finetuned_unified_run.py --analyze-only
  py -3 eval_finetuned_unified_run.py --checkpoint final --episodes 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import run_unified as ru
from src.model.model_handler import load_models, load_models_from_paths
from src.model.master_model import MasterModel
from src.model.agent_handler import Driver
from src.training.general_utils import initialize_models, setup_experiment_dirs


# Matches ``train_unified_full_scenarios.EXPERIMENT_HOME`` + config/condition/seed layout.
DEFAULT_RUN_DIR = os.path.join(
    _REPO, "experiment_runs", "fine_tune_id6", "A_base", "W_MASTER", "s123"
)


def resolve_run_dir(explicit: str) -> str:
    """Prefer CLI, then ``EVAL_RUN_DIR``, then DEFAULT_RUN_DIR if present."""
    s = (explicit or "").strip()
    if s:
        return os.path.abspath(os.path.expanduser(s))
    env_s = (os.environ.get("EVAL_RUN_DIR") or "").strip()
    if env_s:
        return os.path.abspath(os.path.expanduser(env_s))
    if os.path.isdir(DEFAULT_RUN_DIR):
        print(f"[eval] Using default run dir:\n  {DEFAULT_RUN_DIR}")
        return DEFAULT_RUN_DIR
    sys.exit(
        "No --run-dir and default path missing:\n"
        f"  {DEFAULT_RUN_DIR}\n"
        "Pass --run-dir <.../sNNN> or set EVAL_RUN_DIR."
    )


def infer_total_episodes_ref(run_dir: str, fallback: int) -> int:
    metrics_csv = os.path.join(run_dir, "episode_metrics.csv")
    if not os.path.isfile(metrics_csv):
        return fallback
    try:
        with open(metrics_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return fallback
        return max(int(float(rows[-1]["episode"])), fallback)
    except Exception:
        return fallback


def _resolve_checkpoint_prefix(run_dir: str, mode: str) -> str:
    run_dir = os.path.abspath(run_dir)
    p_best = os.path.join(run_dir, "best", "ckpt")
    p_fin = os.path.join(run_dir, "trained_model")
    if mode == "best":
        cand = p_best
    elif mode == "final":
        cand = p_fin
    else:
        hb = os.path.isfile(p_best + "_agent.pth") and os.path.isfile(p_best + "_master.pth")
        hf = os.path.isfile(p_fin + "_agent.pth") and os.path.isfile(p_fin + "_master.pth")
        if hb and hf:
            tb = os.path.getmtime(p_best + "_agent.pth")
            tf = os.path.getmtime(p_fin + "_agent.pth")
            cand = p_best if tb >= tf else p_fin  # noqa: simplicity
        elif hb:
            cand = p_best
        elif hf:
            cand = p_fin
        else:
            raise FileNotFoundError(
                f"No checkpoints under {run_dir!r}: need {p_best}_agent.pth (+_master) "
                f"OR {p_fin}_agent.pth (+_master)"
            )

    if not (
        os.path.isfile(cand + "_agent.pth")
        and os.path.isfile(cand + "_master.pth")
    ):
        raise FileNotFoundError(f"Incomplete ckpt prefix {cand!r} (need *_agent.pth and *_master.pth)")
    return cand


def summarize_training_csv(run_dir: str) -> dict | None:
    metrics_csv = os.path.join(run_dir, "episode_metrics.csv")
    if not os.path.isfile(metrics_csv):
        print(f"[analyze] missing {metrics_csv}")
        return None
    try:
        import pandas as pd
    except ImportError:
        print("[analyze] install pandas for CSV summary (`pip install pandas`)")
        return None

    df = pd.read_csv(metrics_csv)
    n = len(df)
    tail = min(500, n)
    by_env = df.groupby("env")["arrival_pct"].mean().to_dict() if "env" in df.columns else {}
    summary = {
        "n_episodes": int(n),
        "arrival_mean_all": float(df["arrival_pct"].mean()),
        "arrival_mean_last_500": float(df["arrival_pct"].iloc[-tail:].mean()),
        "collision_episode_frac_pct": float(df["collision"].mean() * 100),
        "collision_last_500_frac_pct": float(df["collision"].iloc[-tail:].mean() * 100),
        "reward_mean_last_500": float(df["reward"].iloc[-tail:].mean()),
        "arrival_mean_by_env": {str(k): float(v) for k, v in by_env.items()},
    }
    sj = os.path.join(run_dir, "summary.json")
    if os.path.isfile(sj):
        with open(sj, encoding="utf-8") as f:
            summary["summary_json"] = json.load(f)

    print("\n=== Training metrics (episode_metrics.csv) ===")
    print(json.dumps(summary, indent=2))
    out = os.path.join(run_dir, "eval_training_digest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[wrote] {out}")
    return summary


def run_evaluation(
    *,
    run_dir: str,
    ckpt_prefix: str | None,
    agent_pth: str | None,
    master_pth: str | None,
    n_episodes: int,
    total_episodes_ref: int,
) -> dict:
    run_dir = os.path.abspath(run_dir)
    setup_experiment_dirs(run_dir)

    MasterModel.NORMALIZE_INPUTS = ru.NORMALIZE_MASTER_INPUTS
    Driver.NORMALIZE_AGENT_OBS = ru.NORMALIZE_AGENT_OBS

    ru._set_all_seeds(0)

    exp = ru._make_experiment(run_dir, max(500, total_episodes_ref))
    inter_cfg = ru._make_env_config("RELintersection-v0", conflict_ratio=0.0)
    master_model, agent_model, _ = initialize_models(exp, inter_cfg)

    if agent_pth and master_pth:
        ok = load_models_from_paths(agent_model, master_model, agent_pth, master_pth)
    else:
        assert ckpt_prefix is not None
        ok = load_models(agent_model, master_model, ckpt_prefix)

    if not ok:
        raise RuntimeError("Failed to load checkpoints.")

    modes = [
        ("regular_pipeline", False, False),
        ("held_out", True, False),
        ("conflict_only", False, True),
    ]

    grid: dict[tuple[str, str], dict] = {}
    for env_id, env_label in ru.ENV_DEFS:
        for label, uh, uc in modes:
            key = (env_label, label)
            grid[key] = ru._run_test(
                label,
                env_id,
                env_label,
                exp,
                agent_model,
                master_model,
                total_episodes_ref,
                use_held_out=uh,
                use_conflict_only=uc,
                n_episodes=n_episodes,
            )

    flat = []
    for (env_l, mode_l), r in grid.items():
        flat.append({"env": env_l, "test_mode": mode_l, **r})

    report = {
        "run_dir": run_dir,
        "checkpoint_mode": "explicit_paths" if agent_pth else ckpt_prefix,
        "n_episodes_per_cell": n_episodes,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "cells": flat,
    }
    out_json = os.path.join(run_dir, "eval_post_finetune_grid.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[wrote] {out_json}")

    # compact console table
    print("\n=== Post fine-tune eval (mean arrival %, crash % of episodes) ===")
    for row in flat:
        print(
            f"  {row['env']:<20} {row['test_mode']:<18}  "
            f"arr={row['arrival_rate_avg']:.1f}%  crash={row['crash_rate_pct']:.1f}%"
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help=f"Unified condition dir (default: {DEFAULT_RUN_DIR} if it exists).",
    )
    ap.add_argument(
        "--checkpoint",
        choices=("auto", "best", "final"),
        default="best",
        help="Which saved prefix to load when not using explicit pth paths.",
    )
    ap.add_argument("--agent-pth", type=str, default="", help="Override agent .pth")
    ap.add_argument("--master-pth", type=str, default="", help="Override master .pth")
    ap.add_argument("--episodes", type=int, default=ru.N_TEST_EPISODES, help="Episodes per test cell")
    ap.add_argument(
        "--total-episodes-ref",
        type=int,
        default=0,
        help="Hyperparam ref only; 0 = infer last episode from episode_metrics.csv when possible.",
    )
    ap.add_argument("--analyze-only", action="store_true", help="Only CSV/JSON digest, no rollout")
    ap.add_argument("--eval-only", action="store_true", help="Skip CSV digest")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)

    fallback_ep = 7500 if int(args.total_episodes_ref or 0) <= 0 else int(args.total_episodes_ref)
    total_episodes_ref = infer_total_episodes_ref(run_dir, fallback_ep)

    if not args.eval_only:
        summarize_training_csv(run_dir)

    if args.analyze_only:
        return

    agent_pth = (args.agent_pth or "").strip() or None
    master_pth = (args.master_pth or "").strip() or None
    if bool(agent_pth) ^ bool(master_pth):
        ap.error("Provide both --agent-pth and --master-pth, or neither (use --checkpoint).")

    ckpt_prefix: str | None
    if agent_pth and master_pth:
        ckpt_prefix = None
    else:
        mode = args.checkpoint if args.checkpoint != "auto" else "auto"
        ckpt_prefix = _resolve_checkpoint_prefix(run_dir, mode)

    run_evaluation(
        run_dir=run_dir,
        ckpt_prefix=ckpt_prefix,
        agent_pth=agent_pth,
        master_pth=master_pth,
        n_episodes=int(args.episodes),
        total_episodes_ref=total_episodes_ref,
    )


if __name__ == "__main__":
    main()
