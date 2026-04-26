"""
Zero-shot transfer test.

Loads the best Q02 checkpoint (trained on the single intersection)
and runs it WITHOUT ANY TRAINING on:
  1. RELroundabout-v0          (circular ring, 4 arms)
  2. RELdouble-intersection-v0 (two 4-way junctions connected)
  3. RELintersection-v0        (original environment — sanity check)

No gradient updates. Pure inference.

Usage:
    python run_transfer_test.py

Set Q02_BEST_MODEL_DIR below if the path differs.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import json
from datetime import datetime

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

# ── Path to Q02 best model ────────────────────────────────────────────────────
Q02_BEST_MODEL_DIR = os.path.join(
    "experiment_runs",
    "grid_23_04_2026-13_51_18",
    "Q02_ep1_ent0005_PL75",
    "best_model",
)
N_TEST_EPISODES = 200   # episodes per environment

# ── Register all environments ─────────────────────────────────────────────────
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
from src.model.model_handler import load_models
from src.training.general_utils import initialize_models
from src.training.episode_utils import process_episode
from src.training.training_handler import _make_master_buffer
from src.project_globals import rollout_buffers

# ── Q02 hyperparameters (must match checkpoint architecture exactly) ──────────
_Q02_PARAMS = dict(
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
    agent_lr=3e-3,
    clip_range=0.2,
    n_steps=384,
    target_speeds=[5, 10],
)


def _make_experiment(env_id: str) -> Experiment:
    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID="transfer_test",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=1,
        ENT_COEF=_Q02_PARAMS["ent_coef"],
        ENT_COEF_FINAL=_Q02_PARAMS["ent_coef"],
        WARMUP_EPISODES=0,           # no warmup during test
        PEAK_ARRIVAL_THRESHOLD=0.0,  # no peak-lock during test
        N_VALUE_EPOCHS=0,
        COLLISION_REWARD=_Q02_PARAMS["collision_reward"],
        REACHED_TARGET_REWARD=_Q02_PARAMS["arrived_reward"],
        STARVATION_REWARD=_Q02_PARAMS["starvation_reward"],
        HIGH_SPEED_REWARD=_Q02_PARAMS["high_speed_reward"],
        AGENT_REWARD_MODE=_Q02_PARAMS["reward_mode"],
        FULL_JOINT_TRAINING=False,
        COTRAIN_CYCLES=False,
        AGENT_LR=_Q02_PARAMS["agent_lr"],
        MASTER_LR=_Q02_PARAMS["master_lr"],
        CLIP_RANGE=_Q02_PARAMS["clip_range"],
        GAMMA=_Q02_PARAMS["gamma"],
        GAE_LAMBDA=_Q02_PARAMS["gae_lambda"],
        AGENT_NET_ARCH=_Q02_PARAMS["agent_net_arch"],
        EPISODE_AMOUNT_FOR_TRAIN=1,
        VF_COEF=_Q02_PARAMS["vf_coef"],
        N_PPO_EPOCHS=_Q02_PARAMS["n_ppo_epochs"],
        EPISODES_PER_CYCLE=N_TEST_EPISODES,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
        N_STEPS=int(_Q02_PARAMS["n_steps"]),
    )
    exp.ENV_ID = env_id
    exp.EXPERIMENT_PATH = "transfer_test_tmp"
    exp.SAVE_MODEL_DIRECTORY = "transfer_test_tmp/model"
    return exp


def _make_env_cfg(env_id: str) -> dict:
    if env_id == "RELroundabout-v0":
        from src.experiment.new_envs_config import make_roundabout_env_config
        return make_roundabout_env_config(
            collision_reward=_Q02_PARAMS["collision_reward"],
            arrived_reward=_Q02_PARAMS["arrived_reward"],
            starvation_reward=_Q02_PARAMS["starvation_reward"],
            high_speed_reward=_Q02_PARAMS["high_speed_reward"],
            target_speeds=_Q02_PARAMS["target_speeds"],
        )
    elif env_id == "RELdouble-intersection-v0":
        from src.experiment.new_envs_config import make_double_intersection_env_config
        return make_double_intersection_env_config(
            collision_reward=_Q02_PARAMS["collision_reward"],
            arrived_reward=_Q02_PARAMS["arrived_reward"],
            starvation_reward=_Q02_PARAMS["starvation_reward"],
            high_speed_reward=_Q02_PARAMS["high_speed_reward"],
            target_speeds=_Q02_PARAMS["target_speeds"],
        )
    else:  # intersection
        from src.experiment.scenarios_config import make_env_config_exp7
        return make_env_config_exp7(
            collision_reward=_Q02_PARAMS["collision_reward"],
            arrived_reward=_Q02_PARAMS["arrived_reward"],
            starvation_reward=_Q02_PARAMS["starvation_reward"],
            high_speed_reward=_Q02_PARAMS["high_speed_reward"],
            target_speeds=_Q02_PARAMS["target_speeds"],
        )


def _init_buffers(exp: Experiment):
    rollout_buffers.clear()
    project_globals.local_master_rollout_buffers.clear()
    for _ in range(exp.CARS_AMOUNT):
        rollout_buffers.append(RolloutBuffer(
            buffer_size=exp.N_STEPS,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(exp.STATE_INPUT_SIZE,), dtype=np.float32,
            ),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            gamma=exp.GAMMA,
            gae_lambda=exp.GAE_LAMBDA,
            n_envs=1,
        ))
    for _ in range(exp.NUM_LOCAL_MASTERS):
        project_globals.local_master_rollout_buffers.append(_make_master_buffer(exp))
    project_globals.global_master_rollout_buffer = _make_master_buffer(exp)


def test_on_env(env_id: str, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  TRANSFER TEST: {label}")
    print(f"  Model: {Q02_BEST_MODEL_DIR}")
    print(f"  Episodes: {N_TEST_EPISODES}")
    print(f"{'='*60}")

    project_globals.after_is_arrived_flags.clear()
    for _ in range(6):
        project_globals.after_is_arrived_flags.append(False)

    exp       = _make_experiment(env_id)
    env_cfg   = _make_env_cfg(env_id)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_cfg)

    # ── Load Q02 best checkpoint ──────────────────────────────────────────────
    ckpt_path = os.path.join(Q02_BEST_MODEL_DIR, "checkpoint")
    loaded    = load_models(agent_model, master_model, ckpt_path)
    if loaded:
        print(f"  Checkpoint loaded from: {ckpt_path}")
    else:
        print(f"  WARNING: could not load checkpoint — using random weights!")

    _init_buffers(exp)

    arrivals, crashes = [], []
    for ep in range(1, N_TEST_EPISODES + 1):
        _, _, _, crashed, arrival_rate = process_episode(
            ep, 0, wrapped_env, master_model, agent_model,
            exp,
            train_both=False,
            training_local_master=False,
            training_agent=False,
            training_global_master=False,
        )
        arrivals.append(float(arrival_rate))
        crashes.append(1 if crashed else 0)

        if ep % 50 == 0:
            recent_arr = arrivals[-50:]
            recent_cr  = crashes[-50:]
            print(f"  ep {ep:4d}/{N_TEST_EPISODES}  "
                  f"arrival(last 50)={np.mean(recent_arr):.1f}%  "
                  f"crash(last 50)={np.mean(recent_cr)*100:.1f}%")

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    result = {
        "env":              label,
        "env_id":           env_id,
        "n_episodes":       N_TEST_EPISODES,
        "arrival_pct":      round(float(np.mean(arrivals)), 2),
        "crash_rate_pct":   round(float(np.mean(crashes)) * 100.0, 2),
        "total_crashes":    int(sum(crashes)),
        "arrival_last50":   round(float(np.mean(arrivals[-50:])), 2),
    }
    print(f"\n  RESULT [{label}]: arrival={result['arrival_pct']}%  "
          f"crash={result['crash_rate_pct']}%  "
          f"(last-50 arrival={result['arrival_last50']}%)")
    return result


def main():
    if not os.path.exists(Q02_BEST_MODEL_DIR):
        print(f"ERROR: Q02 best model not found at: {Q02_BEST_MODEL_DIR}")
        sys.exit(1)

    results = []

    # 1. Sanity check — original intersection (should be ~95-100%)
    results.append(test_on_env("RELintersection-v0",      "Intersection (original)"))

    # 2. Roundabout — zero-shot transfer
    results.append(test_on_env("RELroundabout-v0",        "Roundabout (new)"))

    # 3. Double intersection — zero-shot transfer
    results.append(test_on_env("RELdouble-intersection-v0", "Double Intersection (new)"))

    # ── Save results ──────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    out_dir  = os.path.join("experiment_runs", f"transfer_test_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "transfer_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"model": Q02_BEST_MODEL_DIR, "results": results}, f, indent=2)

    print(f"\n{'#'*60}")
    print("  ZERO-SHOT TRANSFER SUMMARY")
    print(f"  Model: Q02 (trained on intersection, {N_TEST_EPISODES} ep each)")
    print(f"{'#'*60}")
    print(f"  {'Environment':<30}  {'Arrival':>8}  {'Crash':>8}  {'Last-50':>8}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['env']:<30}  {r['arrival_pct']:>7.1f}%  "
              f"{r['crash_rate_pct']:>7.1f}%  {r['arrival_last50']:>7.1f}%")
    print(f"\n  Results saved: {out_file}")


if __name__ == "__main__":
    main()
