"""
Master-network architecture search (5 variants).

All 5 configs use the Q02 formula (ent=0.005, PL75).
Only the master network size / depth changes across configs.

Architectures tested:
  M01_tiny      → ResNet extractor fd=64,  net=[64]
  M02_small     → ResNet extractor fd=128, net=[128,128]
  M03_default   → ResNet extractor fd=128, net=[128,256,128]  (current)
  M04_deep      → ResNet extractor fd=128, net=[128,128,128,128]
  M05_wide      → ResNet extractor fd=256, net=[256,256]

Training: 3×500 = 1500 episodes on the intersection only (same as Q02).
Pre-trained Q02 agent weights are loaded so that only the master architecture
effect is isolated.

Usage:
    python run_master_arch.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

logging.basicConfig(level=logging.WARNING)

from highwayenv.utils import patch_intersection_env
patch_intersection_env()

from src.experiment.experiment_config import Experiment
from src.experiment.scenarios_config import make_env_config_exp7
from src.model.model_handler import load_models, save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs
from src.training.training_handler import training_loop
from stable_baselines3.common.logger import configure as _sb3_configure
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

PRETRAINED_CHECKPOINT = os.path.join(
    "experiment_runs",
    "grid_23_04_2026-13_51_18",
    "Q02_ep1_ent0005_PL75",
    "best_model",
    "checkpoint",
)

SHORT_RUN          = False
EPISODES_PER_CYCLE = 100 if SHORT_RUN else 500
CYCLES             = 2   if SHORT_RUN else 3    # 3×500 = 1500

# Base Q02 hyperparameters — only master_net_arch varies
_BASE = dict(
    collision_reward      = -50,
    arrived_reward        = 50,
    starvation_reward     = 0,
    high_speed_reward     = 5,
    reward_mode           = "global",
    master_lr             = 3e-4,
    gamma                 = 0.9,
    gae_lambda            = 0.9,
    agent_net_arch        = "wide",
    ep_for_train          = 1,
    vf_coef               = 1.0,
    n_ppo_epochs          = 5,
    ent_coef              = 0.005,
    ent_coef_final        = 0.005,
    agent_lr              = 3e-3,
    clip_range            = 0.2,
    n_steps               = 384,
    warmup_episodes       = 400,
    peak_arrival_threshold= 75.0,
    n_value_epochs        = 0,
    target_speeds         = [5, 10],
)

def _cfg(label: str, master_arch: str) -> dict:
    d = deepcopy(_BASE)
    d["label"]            = label
    d["master_net_arch"]  = master_arch
    return d

GRID_CONFIGS = [
    _cfg("M01_tiny",    "tiny"),
    _cfg("M02_small",   "small"),
    _cfg("M03_default", "default"),   # same as current Q02 master
    _cfg("M04_deep",    "deep"),
    _cfg("M05_wide",    "wide"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_loggers_csv(base_path: str):
    agent_logger  = _sb3_configure(os.path.join(base_path, "agent_logs"),  ["stdout", "csv"])
    master_logger = _sb3_configure(os.path.join(base_path, "master_logs"), ["stdout", "csv"])
    return agent_logger, master_logger


def _make_experiment(cfg: dict, exp_path: str) -> Experiment:
    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID=cfg["label"],
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=CYCLES,
        ENT_COEF=cfg["ent_coef"],
        ENT_COEF_FINAL=cfg["ent_coef_final"],
        WARMUP_EPISODES=cfg["warmup_episodes"],
        PEAK_ARRIVAL_THRESHOLD=cfg["peak_arrival_threshold"],
        N_VALUE_EPOCHS=cfg["n_value_epochs"],
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
    # Attach master arch key
    exp.MASTER_NET_ARCH = cfg["master_net_arch"]
    return exp


def run_one_config(cfg: dict, grid_root: str) -> tuple[int, dict]:
    label    = cfg["label"]
    exp_path = os.path.join(grid_root, label)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, "agent_logs"),  exist_ok=True)
    os.makedirs(os.path.join(exp_path, "master_logs"), exist_ok=True)

    env_config = make_env_config_exp7(
        collision_reward  = cfg["collision_reward"],
        arrived_reward    = cfg["arrived_reward"],
        starvation_reward = cfg["starvation_reward"],
        high_speed_reward = cfg["high_speed_reward"],
        target_speeds     = cfg["target_speeds"],
    )

    exp = _make_experiment(cfg, exp_path)
    setup_experiment_dirs(exp_path)

    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)

    # Load Q02 agent weights; master starts fresh (different architecture)
    ckpt = PRETRAINED_CHECKPOINT + "_agent.pth"
    if os.path.exists(ckpt):
        # Only load agent weights — master arch differs so master starts fresh
        import torch
        try:
            state_dict = torch.load(PRETRAINED_CHECKPOINT + "_agent.pth", map_location="cpu")
            agent_model.policy.load_state_dict(state_dict)
            print(f"  [{label}] Agent weights loaded from Q02 checkpoint.")
        except Exception as e:
            print(f"  [{label}] Could not load agent weights: {e}")
    else:
        print(f"  [{label}] No checkpoint found — using random init for both.")

    agent_logger, master_logger = _setup_loggers_csv(exp_path)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    print(f"\n{'='*55}")
    print(f"  {label}  (master_arch={cfg['master_net_arch']})")
    print(f"  Episodes: {CYCLES}×{EPISODES_PER_CYCLE}={CYCLES*EPISODES_PER_CYCLE}")
    print(f"{'='*55}")

    collision_counter, results = training_loop(exp, wrapped_env, agent_model, master_model)

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)

    arr = [v for v in results["arrival_rates"] if v is not None]
    summary = {
        "label":                   label,
        "master_net_arch":         cfg["master_net_arch"],
        "total_collisions":        collision_counter,
        "arrival_rate_avg_pct":    round(float(np.mean(arr)), 2) if arr else 0,
        "arrival_rate_last50_pct": round(float(np.mean(arr[-50:])), 2) if arr else 0,
    }
    with open(os.path.join(exp_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  [{label}] avg={summary['arrival_rate_avg_pct']:.1f}%  last50={summary['arrival_rate_last50_pct']:.1f}%")

    try:
        wrapped_env.close()
    except Exception:
        pass

    return collision_counter, summary


def main():
    ts         = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    grid_root  = os.path.join("experiment_runs", f"master_arch_{ts}")
    os.makedirs(grid_root, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  MASTER ARCHITECTURE SEARCH — 5 configs")
    print(f"  Base: Q02 formula  |  {CYCLES}×{EPISODES_PER_CYCLE} episodes each")
    print(f"  Output: {grid_root}")
    print(f"{'#'*60}\n")

    all_summaries = []
    for cfg in GRID_CONFIGS:
        _, summary = run_one_config(cfg, grid_root)
        all_summaries.append(summary)

    # ── Grid comparison ────────────────────────────────────────────────────────
    with open(os.path.join(grid_root, "grid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    labels    = [s["label"] for s in all_summaries]
    avg_arr   = [s["arrival_rate_avg_pct"]    for s in all_summaries]
    last50    = [s["arrival_rate_last50_pct"] for s in all_summaries]
    x         = np.arange(len(labels))
    w         = 0.35
    ax.bar(x - w/2, avg_arr, w, label="Avg arrival %",   color="#2196F3")
    ax.bar(x + w/2, last50,  w, label="Last-50 arrival %", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([s["master_net_arch"] for s in all_summaries], rotation=15)
    ax.set_ylabel("Arrival %")
    ax.set_title("Master network architecture comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(grid_root, "arch_comparison.png"), dpi=120)
    plt.close(fig)

    print(f"\n{'#'*60}")
    print(f"  GRID COMPLETE")
    for s in all_summaries:
        print(f"  {s['label']:25s} arch={s['master_net_arch']:12s}  last50={s['arrival_rate_last50_pct']:.1f}%")
    print(f"  Results: {grid_root}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
