"""
New-environment training experiments.
Tests the Q02-winning hyperparameters on:
  - RELroundabout-v0           (circular ring with 4 arms, same node naming as intersection)
  - RELdouble-intersection-v0  (two 4-way junctions connected by a bidirectional road)

Both use the same 3+3 hierarchical architecture (LM1→agents 0-2, LM2→agents 3-5, GM on top).
No changes to model or training code — only the environment and scenarios differ.

Usage:
    python run_new_envs.py

Set ENV_TYPE below to "roundabout" or "double_intersection" before running.

Outputs: experiment_runs/new_envs_<timestamp>/...
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

# ── Choose environment ────────────────────────────────────────────────────────
# "roundabout"          → RELroundabout-v0  (25 base × 4 rotations = 100 scenarios)
# "double_intersection" → RELdouble-intersection-v0  (20 base scenarios, no rotation)
ENV_TYPE = "roundabout"   # ← change this line to switch environment
# ─────────────────────────────────────────────────────────────────────────────

# Register selected environment before any gym.make calls
from highwayenv.utils import patch_intersection_env
patch_intersection_env()

if ENV_TYPE == "roundabout":
    from highwayenv.utils import register_roundabout_env
    register_roundabout_env()
    ENV_ID = "RELroundabout-v0"
    from src.experiment.new_envs_config import make_roundabout_env_config as _make_env_cfg
elif ENV_TYPE == "double_intersection":
    from highwayenv.utils import register_double_intersection_env
    register_double_intersection_env()
    ENV_ID = "RELdouble-intersection-v0"
    from src.experiment.new_envs_config import make_double_intersection_env_config as _make_env_cfg
else:
    raise ValueError(f"Unknown ENV_TYPE: {ENV_TYPE!r}. Choose 'roundabout' or 'double_intersection'.")

from src import project_globals
from src.experiment.experiment_config import Experiment
from src.model.model_handler import save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs
from stable_baselines3.common.logger import configure as _sb3_configure


def _setup_loggers_csv(base_path):
    """Loggers with stdout + CSV only (no tensorboard — avoids Windows path issues)."""
    agent_logger  = _sb3_configure(os.path.join(base_path, "agent_logs"),  ["stdout", "csv"])
    master_logger = _sb3_configure(os.path.join(base_path, "master_logs"), ["stdout", "csv"])
    return agent_logger, master_logger
from src.training.training_handler import training_loop

from run_learning_experiment import save_json, save_plots

logging.basicConfig(level=logging.WARNING)

# ── Run length ────────────────────────────────────────────────────────────────
SHORT_GRID = False
EPISODES_PER_CYCLE = 100 if SHORT_GRID else 500
CYCLES = 2 if SHORT_GRID else 3   # 3 × 500 = 1500 episodes per config

SMOOTH_COMPARE = 50
ROLLING_CRASH_WINDOW = 50

# ── Base config (Q02 winning formula) ─────────────────────────────────────────
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
    ent_coef=0.005,
    ent_coef_final=0.005,
    agent_lr=3e-3,
    clip_range=0.2,
    n_steps=384,
    warmup_episodes=400,
    peak_arrival_threshold=75.0,
    n_value_epochs=0,
    target_speeds=[5, 10],
)


def _merged_cfg(overrides: dict) -> dict:
    return {**_BASE, **overrides}


# ── Configs ────────────────────────────────────────────────────────────────────
# R = Roundabout, D = Double-intersection (label prefix changes automatically)
_PREFIX = "R" if ENV_TYPE == "roundabout" else "D"

GRID_CONFIGS = [
    # Config 1: Q02 exact reference — does it transfer to the new env?
    _merged_cfg({"label": f"{_PREFIX}01_ref_ent0005_PL75"}),

    # Config 2: higher entropy — more exploration for the new harder topology
    _merged_cfg({"label": f"{_PREFIX}02_ent05_PL75",
                 "ent_coef": 0.05,
                 "ent_coef_final": 0.05}),

    # Config 3: lower peak-lock threshold — commit to exploitation earlier
    _merged_cfg({"label": f"{_PREFIX}03_ent0005_PL65",
                 "peak_arrival_threshold": 65.0}),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nanmean_last(lst, n):
    vals = [v for v in lst[-n:] if v is not None]
    return float(np.mean(vals)) if vals else 0.0


# ── Single-config runner ───────────────────────────────────────────────────────

def run_one_config(cfg: dict, grid_root: str) -> tuple[int, dict]:
    label    = cfg["label"]
    total_ep = CYCLES * EPISODES_PER_CYCLE
    exp_path = os.path.join(grid_root, label)

    print(f"\n{'='*60}")
    print(f"  {ENV_TYPE.upper()} — {label}  ({total_ep} episodes)")
    print(f"{'='*60}")

    # Reset global flags for this fresh experiment
    project_globals.after_is_arrived_flags.clear()
    for _ in range(6):
        project_globals.after_is_arrived_flags.append(False)

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID=f"newenv_{label}",
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
    exp.EXPERIMENT_PATH = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, "trained_model")
    exp.ENV_ID = ENV_ID  # tells Driver which gym env to create

    env_config = _make_env_cfg(
        collision_reward=cfg["collision_reward"],
        arrived_reward=cfg["arrived_reward"],
        starvation_reward=cfg["starvation_reward"],
        high_speed_reward=cfg["high_speed_reward"],
        target_speeds=cfg.get("target_speeds", None),
    )

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, "agent_logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_path, "master_logs"), exist_ok=True)
    setup_experiment_dirs(exp_path)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)
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
        results,
        exp_path,
        metrics_csv,
        suptitle=f"{ENV_TYPE} — {label} ({total_ep} ep)",
        total_episodes=total_ep,
        rolling_crash_window=ROLLING_CRASH_WINDOW,
    )
    save_json(
        results,
        collisions,
        exp_path,
        metrics_csv,
        extra_fields={
            "config_label": label,
            "env_type": ENV_TYPE,
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

def _write_grid_outputs(grid_root: str, all_results: dict):
    summary = {}
    for cfg in GRID_CONFIGS:
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        coll, res = all_results[lbl]
        arr_vals  = [v for v in res.get("arrival_rates", []) if v is not None]
        last20    = arr_vals[-20:] if len(arr_vals) >= 20 else arr_vals
        last50    = arr_vals[-50:] if len(arr_vals) >= 50 else arr_vals
        q1        = arr_vals[:len(arr_vals)//4] if arr_vals else []
        q4        = arr_vals[-len(arr_vals)//4:] if arr_vals else []
        cr_vals   = [v for v in res.get("collision_rates", []) if v is not None]
        crq1      = cr_vals[:len(cr_vals)//4] if cr_vals else []
        crq4      = cr_vals[-len(cr_vals)//4:] if cr_vals else []
        summary[lbl] = {
            "total_collisions":        coll,
            "arrival_rate_avg_pct":    float(np.mean(arr_vals)) if arr_vals else 0,
            "arrival_rate_last20_pct": float(np.mean(last20))   if last20   else 0,
            "arrival_rate_last50_pct": float(np.mean(last50))   if last50   else 0,
            "arrival_delta_q1_q4":
                float(np.mean(q4) - np.mean(q1)) if (q1 and q4) else 0,
            "crash_delta_q1_q4":
                float(np.mean(crq4) - np.mean(crq1)) if (crq1 and crq4) else 0,
        }

    csv_path = os.path.join(grid_root, "grid_summary.csv")
    if summary:
        fields = list(next(iter(summary.values())).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label"] + fields)
            w.writeheader()
            for lbl, row in summary.items():
                w.writerow({"label": lbl, **row})
        print(f"  grid_summary.csv → {csv_path}")

    _comparison_plot(grid_root, all_results, SMOOTH_COMPARE)


def _comparison_plot(grid_root: str, all_results: dict, smooth: int = 50):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ENV_TYPE.upper()} — Grid comparison  (smooth={smooth})", fontsize=12)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, cfg in enumerate(GRID_CONFIGS):
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        _, res = all_results[lbl]
        col = colors[i % len(colors)]

        arr = [v for v in res.get("arrival_rates", []) if v is not None]
        cr  = [v for v in res.get("collision_rates", []) if v is not None]

        def _smooth(vals):
            if len(vals) < smooth:
                return vals
            return np.convolve(vals, np.ones(smooth) / smooth, mode="valid").tolist()

        axes[0].plot(_smooth(arr), label=lbl, color=col)
        axes[1].plot(_smooth(cr),  label=lbl, color=col)

    axes[0].set_title("Arrival rate (smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 105)

    axes[1].set_title("Crash rate (smoothed)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("%")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    out = os.path.join(grid_root, "grid_comparison.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  grid_comparison.png saved: {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ts        = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    grid_root = os.path.join("experiment_runs", f"new_envs_{ENV_TYPE}_{ts}")
    os.makedirs(grid_root, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  NEW ENV GRID: {ENV_TYPE.upper()}")
    print(f"  {len(GRID_CONFIGS)} configs × {CYCLES}×{EPISODES_PER_CYCLE}={CYCLES*EPISODES_PER_CYCLE} ep")
    print(f"  Output: {grid_root}")
    print(f"{'#'*60}\n")

    all_results: dict[str, tuple] = {}
    for cfg in GRID_CONFIGS:
        try:
            coll, res = run_one_config(cfg, grid_root)
            all_results[cfg["label"]] = (coll, res)
        except Exception as exc:
            print(f"  ERROR in {cfg['label']}: {exc}")
            import traceback
            traceback.print_exc()

    _write_grid_outputs(grid_root, all_results)
    print(f"\nDone. Results in: {grid_root}")


if __name__ == "__main__":
    main()
