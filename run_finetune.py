"""
Fine-tune the Q02 checkpoint on new environments.

Loads Q02 weights (trained on the single intersection) and continues training
on RELroundabout-v0 and RELdouble-intersection-v0 sequentially in a single run.

Usage:
    python run_finetune.py
"""

from __future__ import annotations

import logging
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import csv
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

# Which environments to fine-tune on (runs in order)
ENVS_TO_RUN = ["roundabout", "double_intersection"]

# Q02 best checkpoint (pre-trained on intersection)
PRETRAINED_CHECKPOINT = os.path.join(
    "experiment_runs",
    "grid_23_04_2026-13_51_18",
    "Q02_ep1_ent0005_PL75",
    "best_model",
    "checkpoint",
)

SHORT_RUN            = False
EPISODES_PER_CYCLE   = 100 if SHORT_RUN else 500
CYCLES               = 2   if SHORT_RUN else 3      # 3 × 500 = 1500 episodes
ROLLING_CRASH_WINDOW = 50
SMOOTH_COMPARE       = 50
# ──────────────────────────────────────────────────────────────────────────────

# ── Register all environments upfront ─────────────────────────────────────────
from highwayenv.utils import (
    patch_intersection_env,
    register_roundabout_env,
    register_double_intersection_env,
)
patch_intersection_env()
register_roundabout_env()
register_double_intersection_env()

from src.experiment.new_envs_config import (
    make_roundabout_env_config,
    make_double_intersection_env_config,
)
from src import project_globals
from src.experiment.experiment_config import Experiment
from src.model.model_handler import load_models, save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs
from src.training.training_handler import training_loop
from stable_baselines3.common.logger import configure as _sb3_configure
from run_learning_experiment import save_json, save_plots

logging.basicConfig(level=logging.WARNING)

# ── Per-environment metadata ──────────────────────────────────────────────────
_ENV_META = {
    "roundabout": {
        "env_id":    "RELroundabout-v0",
        "make_cfg":  make_roundabout_env_config,
        "prefix":    "FR",
    },
    "double_intersection": {
        "env_id":    "RELdouble-intersection-v0",
        "make_cfg":  make_double_intersection_env_config,
        "prefix":    "FD",
    },
}

# ── Fine-tune base config (Q02 formula, adjusted for transfer) ─────────────────
_BASE = dict(
    collision_reward=-50,
    arrived_reward=50,
    starvation_reward=0,
    high_speed_reward=5,
    reward_mode="global",
    master_lr=3e-4,
    gamma=0.9,
    gae_lambda=0.9,
    agent_net_arch="wide",
    ep_for_train=1,
    vf_coef=1.0,
    n_ppo_epochs=5,
    ent_coef=0.05,           # keep exploration high — new env needs it
    ent_coef_final=0.05,
    agent_lr=3e-3,
    clip_range=0.2,
    n_steps=384,
    warmup_episodes=200,     # longer warmup for new env (was 100)
    peak_arrival_threshold=60.0,  # achievable on new env (was 75 — never triggered)
    n_value_epochs=0,
    target_speeds=[5, 10],
)


def _merged_cfg(overrides: dict) -> dict:
    return {**_BASE, **overrides}


def _grid_configs_for(prefix: str) -> list[dict]:
    return [
        # Q02 weights, adapted for new env: ent=0.05, PL60, warmup=200
        _merged_cfg({"label": f"{prefix}01_ent05_PL60"}),
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_loggers_csv(base_path: str):
    agent_logger  = _sb3_configure(os.path.join(base_path, "agent_logs"),  ["stdout", "csv"])
    master_logger = _sb3_configure(os.path.join(base_path, "master_logs"), ["stdout", "csv"])
    return agent_logger, master_logger


def _nanmean_last(lst: list, n: int) -> float:
    vals = [v for v in lst[-n:] if v is not None]
    return float(np.mean(vals)) if vals else 0.0


# ── Single-config runner ───────────────────────────────────────────────────────

def run_one_config(cfg: dict, grid_root: str, env_type: str) -> tuple[int, dict]:
    meta     = _ENV_META[env_type]
    env_id   = meta["env_id"]
    make_cfg = meta["make_cfg"]

    label    = cfg["label"]
    total_ep = CYCLES * EPISODES_PER_CYCLE
    exp_path = os.path.join(grid_root, label)

    print(f"\n{'='*60}")
    print(f"  FINE-TUNE [{env_type.upper()}] — {label}  ({total_ep} ep)")
    print(f"  Starting from: {PRETRAINED_CHECKPOINT}")
    print(f"{'='*60}")

    project_globals.after_is_arrived_flags.clear()
    for _ in range(6):
        project_globals.after_is_arrived_flags.append(False)

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID=f"ft_{label}",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=CYCLES,
        ENT_COEF=cfg["ent_coef"],
        ENT_COEF_FINAL=cfg.get("ent_coef_final", 0.0),
        WARMUP_EPISODES=cfg.get("warmup_episodes", 0),
        PEAK_ARRIVAL_THRESHOLD=cfg.get("peak_arrival_threshold", 0.0),
        N_VALUE_EPOCHS=cfg.get("n_value_epochs", 0),
        COLLISION_REWARD=cfg["collision_reward"],
        REACHED_TARGET_REWARD=cfg["arrived_reward"],
        STARVATION_REWARD=cfg["starvation_reward"],
        HIGH_SPEED_REWARD=cfg["high_speed_reward"],
        AGENT_REWARD_MODE=cfg["reward_mode"],
        FULL_JOINT_TRAINING=False,
        COTRAIN_CYCLES=False,
        AGENT_LR=cfg["agent_lr"],
        MASTER_LR=cfg["master_lr"],
        CLIP_RANGE=cfg["clip_range"],
        GAMMA=cfg["gamma"],
        GAE_LAMBDA=cfg["gae_lambda"],
        AGENT_NET_ARCH=cfg["agent_net_arch"],
        EPISODE_AMOUNT_FOR_TRAIN=cfg["ep_for_train"],
        VF_COEF=cfg["vf_coef"],
        N_PPO_EPOCHS=cfg["n_ppo_epochs"],
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
        N_STEPS=int(cfg["n_steps"]),
    )
    exp.EXPERIMENT_PATH      = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, "trained_model")
    exp.ENV_ID               = env_id

    env_config = make_cfg(
        collision_reward=cfg["collision_reward"],
        arrived_reward=cfg["arrived_reward"],
        starvation_reward=cfg["starvation_reward"],
        high_speed_reward=cfg["high_speed_reward"],
        target_speeds=cfg.get("target_speeds"),
    )

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, "agent_logs"),  exist_ok=True)
    os.makedirs(os.path.join(exp_path, "master_logs"), exist_ok=True)
    setup_experiment_dirs(exp_path)

    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)

    # ── Load Q02 pre-trained weights ──────────────────────────────────────────
    ckpt_agent = PRETRAINED_CHECKPOINT + "_agent.pth"
    if os.path.exists(ckpt_agent):
        loaded = load_models(agent_model, master_model, PRETRAINED_CHECKPOINT)
        status = "loaded" if loaded else "FAILED (using random init)"
    else:
        status = f"checkpoint not found at {ckpt_agent!r} — using random init"
    print(f"  Pre-trained weights: {status}")

    agent_logger, master_logger = _setup_loggers_csv(exp_path)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    agent_model, master_model, collisions, _, _, results, best_model_dir = training_loop(
        experiment=exp,
        env=wrapped_env,
        agent_model=agent_model,
        master_model=master_model,
    )

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)

    metrics_csv = os.path.join(exp_path, "episode_metrics.csv")
    cfg_snapshot = {k: v for k, v in cfg.items() if k != "label"}
    save_plots(
        results, exp_path, metrics_csv,
        suptitle=f"FineTune {env_type} — {label} ({total_ep} ep)",
        total_episodes=total_ep,
        rolling_crash_window=ROLLING_CRASH_WINDOW,
    )
    save_json(
        results, collisions, exp_path, metrics_csv,
        extra_fields={
            "config_label":    label,
            "env_type":        env_type,
            "pretrained_from": PRETRAINED_CHECKPOINT,
            "hyperparameters": cfg_snapshot,
        },
    )

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    arr_last20 = _nanmean_last(results.get("arrival_rates", []), 20)
    print(f"  [{label}] collisions={collisions}  arrival_last20≈{arr_last20:.1f}%")
    return collisions, results


# ── Grid summary ───────────────────────────────────────────────────────────────

def _write_grid_outputs(grid_root: str, all_results: dict, grid_configs: list, env_type: str):
    summary = {}
    for cfg in grid_configs:
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        coll, res = all_results[lbl]
        arr  = [v for v in res.get("arrival_rates",   []) if v is not None]
        cr   = [v for v in res.get("collision_rates", []) if v is not None]
        q1_a = arr[:len(arr)//4]  if arr else []
        q4_a = arr[-len(arr)//4:] if arr else []
        q1_c = cr[:len(cr)//4]    if cr  else []
        q4_c = cr[-len(cr)//4:]   if cr  else []
        summary[lbl] = {
            "total_collisions":        coll,
            "arrival_rate_avg_pct":    round(float(np.mean(arr)),       2) if arr else 0,
            "arrival_rate_last20_pct": round(float(np.mean(arr[-20:])), 2) if arr else 0,
            "arrival_rate_last50_pct": round(float(np.mean(arr[-50:])), 2) if arr else 0,
            "arrival_delta_q1_q4":
                round(float(np.mean(q4_a) - np.mean(q1_a)), 2) if (q1_a and q4_a) else 0,
            "crash_delta_q1_q4":
                round(float(np.mean(q4_c) - np.mean(q1_c)), 2) if (q1_c and q4_c) else 0,
        }

    csv_path = os.path.join(grid_root, "grid_summary.csv")
    if summary:
        fields = list(next(iter(summary.values())).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label"] + fields)
            w.writeheader()
            for lbl, row in summary.items():
                w.writerow({"label": lbl, **row})
        print(f"  Saved: {csv_path}")

    _comparison_plot(grid_root, all_results, grid_configs, env_type)


def _comparison_plot(grid_root: str, all_results: dict, grid_configs: list, env_type: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Fine-tune {env_type.upper()} from Q02  (smooth={SMOOTH_COMPARE})", fontsize=12
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, cfg in enumerate(grid_configs):
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        _, res = all_results[lbl]
        col = colors[i % len(colors)]

        def _smooth(vals):
            vals = [v for v in vals if v is not None]
            if len(vals) < SMOOTH_COMPARE:
                return vals
            return np.convolve(vals, np.ones(SMOOTH_COMPARE) / SMOOTH_COMPARE,
                               mode="valid").tolist()

        axes[0].plot(_smooth(res.get("arrival_rates",   [])), label=lbl, color=col)
        axes[1].plot(_smooth(res.get("collision_rates", [])), label=lbl, color=col)

    for ax, title, ylabel in [
        (axes[0], "Arrival rate (smoothed)", "%"),
        (axes[1], "Crash rate (smoothed)",   "%"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out = os.path.join(grid_root, "grid_comparison.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Run a full grid for one environment ───────────────────────────────────────

def run_env_grid(env_type: str, parent_root: str):
    meta         = _ENV_META[env_type]
    grid_configs = _grid_configs_for(meta["prefix"])
    total_ep     = CYCLES * EPISODES_PER_CYCLE

    grid_root = os.path.join(parent_root, env_type)
    os.makedirs(grid_root, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  FINE-TUNE: {env_type.upper()}")
    print(f"  Pre-trained: {PRETRAINED_CHECKPOINT}")
    print(f"  {len(grid_configs)} configs × {CYCLES}×{EPISODES_PER_CYCLE} = {total_ep} ep each")
    print(f"  Output: {grid_root}")
    print(f"{'#'*60}\n")

    all_results: dict = {}
    for cfg in grid_configs:
        try:
            coll, res = run_one_config(cfg, grid_root, env_type)
            all_results[cfg["label"]] = (coll, res)
        except Exception as exc:
            print(f"  ERROR in {cfg['label']}: {exc}")
            import traceback; traceback.print_exc()

    _write_grid_outputs(grid_root, all_results, grid_configs, env_type)
    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ckpt_agent = PRETRAINED_CHECKPOINT + "_agent.pth"
    if not os.path.exists(ckpt_agent):
        print(f"WARNING: checkpoint not found at {ckpt_agent}")
        print("  Will train from random init for all configs.")

    ts          = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    parent_root = os.path.join("experiment_runs", f"finetune_{ts}")
    os.makedirs(parent_root, exist_ok=True)

    print(f"\nRun root: {parent_root}")
    print(f"Environments: {ENVS_TO_RUN}")

    for env_type in ENVS_TO_RUN:
        run_env_grid(env_type, parent_root)

    print(f"\n{'='*60}")
    print(f"  All fine-tune runs complete.")
    print(f"  Results: {parent_root}/")
    print(f"    roundabout/          → FR01_ent05_PL60")
    print(f"    double_intersection/ → FD01_ent05_PL60")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
