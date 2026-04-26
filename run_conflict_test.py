"""
Conflict-scenario test: measures master contribution on forced-conflict scenarios.

Runs the SAME pre-trained model twice:
  1. WITH master embeddings (normal)
  2. WITHOUT master (zero embeddings — ablation)

Only conflict scenarios are used (use_conflict_scenarios_only=True).
100 episodes per environment × 3 environments × 2 conditions = 600 total episodes.

Usage:
    python run_conflict_test.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.model.agent_handler import DummyVecEnv
from highwayenv.utils import (
    patch_intersection_env,
    register_intersection_env,
    register_roundabout_env,
    register_double_intersection_env,
)
patch_intersection_env()
register_intersection_env()
register_roundabout_env()
register_double_intersection_env()

from src import project_globals
from src.experiment.experiment_config import Experiment
from src.experiment.scenarios_config import make_env_config_exp7
from src.experiment.new_envs_config import (
    make_roundabout_env_config,
    make_double_intersection_env_config,
)
from src.model.model_handler import load_models
from src.model.agent_handler import Driver
from src.training.general_utils import initialize_models
from src.training.episode_utils import process_episode
from src.model.master_model import MasterModel

# ── Where to find the best unified model ──────────────────────────────────────
BEST_UNIFIED_CHECKPOINT = os.path.join(
    "experiment_runs",
    "unified_25_04_2026-19_21_45",
    "best_model",
    "checkpoint",
)

N_TEST_EPISODES = 100

_CFG = dict(
    collision_reward     = -50,
    arrived_reward       = 50,
    starvation_reward    = 0,
    high_speed_reward    = 5,
    reward_mode          = "global",
    master_lr            = 3e-4,
    gamma                = 0.9,
    gae_lambda           = 0.9,
    agent_net_arch       = "wide",
    ep_for_train         = 1,
    vf_coef              = 1.0,
    n_ppo_epochs         = 5,
    ent_coef             = 0.005,
    ent_coef_final       = 0.005,
    agent_lr             = 3e-3,
    clip_range           = 0.2,
    n_steps              = 384,
    warmup_episodes      = 0,
    peak_arrival_threshold = 0.0,
    n_value_epochs       = 0,
    target_speeds        = [5, 10],
)


def _make_experiment(exp_path: str, env_id: str = "RELintersection-v0") -> Experiment:
    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID="conflict_test",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=1,
        ENT_COEF=_CFG["ent_coef"],
        ENT_COEF_FINAL=_CFG["ent_coef_final"],
        WARMUP_EPISODES=0,
        PEAK_ARRIVAL_THRESHOLD=0.0,
        N_VALUE_EPOCHS=0,
        COLLISION_REWARD=_CFG["collision_reward"],
        REACHED_TARGET_REWARD=_CFG["arrived_reward"],
        STARVATION_REWARD=_CFG["starvation_reward"],
        HIGH_SPEED_REWARD=_CFG["high_speed_reward"],
        AGENT_REWARD_MODE=_CFG["reward_mode"],
        FULL_JOINT_TRAINING=False,
        COTRAIN_CYCLES=False,
        AGENT_LR=_CFG["agent_lr"],
        MASTER_LR=_CFG["master_lr"],
        CLIP_RANGE=_CFG["clip_range"],
        GAMMA=_CFG["gamma"],
        GAE_LAMBDA=_CFG["gae_lambda"],
        AGENT_NET_ARCH=_CFG["agent_net_arch"],
        EPISODE_AMOUNT_FOR_TRAIN=1,
        VF_COEF=_CFG["vf_coef"],
        N_PPO_EPOCHS=_CFG["n_ppo_epochs"],
        EPISODES_PER_CYCLE=N_TEST_EPISODES,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
        N_STEPS=int(_CFG["n_steps"]),
    )
    exp.EXPERIMENT_PATH      = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, "trained_model")
    exp.ENV_ID = env_id
    return exp


def _make_env_config(env_id: str, conflict_only: bool = True) -> dict:
    kw = dict(
        collision_reward  = _CFG["collision_reward"],
        arrived_reward    = _CFG["arrived_reward"],
        starvation_reward = _CFG["starvation_reward"],
        high_speed_reward = _CFG["high_speed_reward"],
        target_speeds     = _CFG["target_speeds"],
    )
    if env_id == "RELintersection-v0":
        cfg = make_env_config_exp7(**kw)
    elif env_id == "RELroundabout-v0":
        cfg = make_roundabout_env_config(**kw)
    elif env_id == "RELdouble-intersection-v0":
        cfg = make_double_intersection_env_config(**kw)
    else:
        raise ValueError(f"Unknown env_id: {env_id!r}")
    if conflict_only:
        cfg["use_conflict_scenarios_only"] = True
    return cfg


def test_condition(
    label: str,
    env_defs: list,
    exp_path: str,
    agent_model,
    master_model,
) -> dict:
    """Run N_TEST_EPISODES on conflict scenarios for each environment."""
    results = {}
    for env_id, env_label in env_defs:
        print(f"\n  [{label}] Testing {env_label} ({N_TEST_EPISODES} episodes, conflict-only) ...")
        project_globals.after_is_arrived_flags = [False] * 6

        exp_test = _make_experiment(exp_path, env_id=env_id)
        env_cfg = _make_env_config(env_id, conflict_only=True)
        exp_test.CONFIG = env_cfg

        def _env_fn(ec=exp_test, eid=env_id):
            d = Driver(ec)
            d.highway_env.unwrapped.config["use_conflict_scenarios_only"] = True
            return d

        test_wrapped = DummyVecEnv([_env_fn])
        arrivals, crashes = [], []

        for ep in range(1, N_TEST_EPISODES + 1):
            _, _, _, crashed, arrival_rate = process_episode(
                ep, 0, test_wrapped, master_model, agent_model,
                exp_test,
                train_both=False,
                training_local_master=False,
                training_agent=False,
                training_global_master=False,
            )
            arrivals.append(arrival_rate)
            crashes.append(1 if crashed else 0)

        try:
            test_wrapped.close()
        except Exception:
            pass

        r = {
            "arrival_avg": round(float(np.mean(arrivals)), 2),
            "crash_rate_pct": round(100.0 * sum(crashes) / len(crashes), 2),
            "arrivals": arrivals,
            "crashes": crashes,
        }
        results[env_label] = r
        print(f"    arrival={r['arrival_avg']:.1f}%  crash={r['crash_rate_pct']:.1f}%")

    return results


def main():
    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    exp_path = os.path.join("experiment_runs", f"conflict_test_{ts}")
    os.makedirs(exp_path, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  CONFLICT SCENARIO TEST")
    print(f"  {N_TEST_EPISODES} episodes × 3 envs × 2 conditions")
    print(f"  Checkpoint: {BEST_UNIFIED_CHECKPOINT}")
    print(f"  Output: {exp_path}")
    print(f"{'#'*60}\n")

    env_defs = [
        ("RELintersection-v0",       "intersection"),
        ("RELroundabout-v0",          "roundabout"),
        ("RELdouble-intersection-v0", "double_intersection"),
    ]

    inter_cfg = _make_env_config("RELintersection-v0", conflict_only=False)
    exp = _make_experiment(exp_path, env_id="RELintersection-v0")
    exp.CONFIG = inter_cfg
    master_model, agent_model, _ = initialize_models(exp, inter_cfg)

    ckpt = BEST_UNIFIED_CHECKPOINT + "_agent.pth"
    if os.path.exists(ckpt):
        loaded = load_models(agent_model, master_model, BEST_UNIFIED_CHECKPOINT)
        print(f"  Checkpoint: {'loaded' if loaded else 'FAILED'}")
    else:
        print(f"  WARNING: no checkpoint at {ckpt}, using random init")

    # ── Condition 1: WITH master ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONDITION 1: WITH MASTER")
    print(f"{'='*60}")
    results_with = test_condition("WITH_MASTER", env_defs, exp_path,
                                  agent_model, master_model)

    # ── Condition 2: WITHOUT master (zero embeddings) ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONDITION 2: WITHOUT MASTER (ablation)")
    print(f"{'='*60}")
    _original = MasterModel.get_proto_action
    def _zero_proto_action(self, master_input):
        emb = np.zeros(self.embedding_size, dtype=np.float32)
        val = torch.tensor([0.0])
        lp  = torch.tensor([0.0])
        return emb, val, lp
    MasterModel.get_proto_action = _zero_proto_action

    results_without = test_condition("NO_MASTER", env_defs, exp_path,
                                     agent_model, master_model)

    MasterModel.get_proto_action = _original

    # ── Comparison plot ───────────────────────────────────────────────────────
    env_labels = [label for _, label in env_defs]
    with_arrivals = [results_with[l]["arrival_avg"] for l in env_labels]
    without_arrivals = [results_without[l]["arrival_avg"] for l in env_labels]
    with_crashes = [results_with[l]["crash_rate_pct"] for l in env_labels]
    without_crashes = [results_without[l]["crash_rate_pct"] for l in env_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(env_labels))
    w = 0.35

    ax1 = axes[0]
    bars1 = ax1.bar(x - w/2, with_arrivals, w, label="With Master", color="#2196F3")
    bars2 = ax1.bar(x + w/2, without_arrivals, w, label="Without Master", color="#FF5722")
    ax1.set_ylabel("Arrival Rate (%)")
    ax1.set_title("Conflict Scenarios: Arrival Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_labels, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    bars3 = ax2.bar(x - w/2, with_crashes, w, label="With Master", color="#2196F3")
    bars4 = ax2.bar(x + w/2, without_crashes, w, label="Without Master", color="#FF5722")
    ax2.set_ylabel("Crash Rate (%)")
    ax2.set_title("Conflict Scenarios: Crash Rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(env_labels, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Master Contribution on Forced-Conflict Scenarios", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, "conflict_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "description": "Conflict-only scenario test: WITH vs WITHOUT master",
        "n_episodes_per_env": N_TEST_EPISODES,
        "checkpoint": BEST_UNIFIED_CHECKPOINT,
        "with_master": {l: {"arrival": results_with[l]["arrival_avg"],
                            "crash": results_with[l]["crash_rate_pct"]}
                        for l in env_labels},
        "without_master": {l: {"arrival": results_without[l]["arrival_avg"],
                               "crash": results_without[l]["crash_rate_pct"]}
                           for l in env_labels},
        "delta": {l: {
            "arrival_improvement": round(results_with[l]["arrival_avg"] - results_without[l]["arrival_avg"], 2),
            "crash_reduction": round(results_without[l]["crash_rate_pct"] - results_with[l]["crash_rate_pct"], 2),
        } for l in env_labels},
    }
    with open(os.path.join(exp_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  CONFLICT TEST COMPLETE")
    print(f"{'#'*60}")
    print(f"\n  {'Environment':<25s} {'With Master':>12s} {'No Master':>12s} {'Δ Arrival':>12s}")
    print(f"  {'-'*61}")
    for l in env_labels:
        wa = results_with[l]["arrival_avg"]
        na = results_without[l]["arrival_avg"]
        delta = wa - na
        print(f"  {l:<25s} {wa:>11.1f}% {na:>11.1f}% {delta:>+11.1f}%")
    print(f"\n  Results: {exp_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
