"""
Fine-tune the shared agent + master checkpoint on **all implemented REL environments**:

  * RELintersection-v0        — single 4-way junction (6 agents, 2×LM)
  * RELroundabout-v0          — traffic circle (6 agents, 2×LM)
  * RELdouble-intersection-v0 — two linked junctions (6 agents, 2×LM)
  * RELchain-intersection-v0  — connected corridor (6 agents, 2 zones × 1 regional LM)

Note: triangular / pentagonal junctions are **not** in this highway-env fork; adding them
requires new env classes. This script covers every topology that exists in ``highwayenv/``.

Each phase loads the checkpoint produced by the previous phase (warm chain).
Chain uses the same scenario generator + local-frame normalization as eval.

Usage:
  py -3 train_all_environments_finetune.py
  py -3 train_all_environments_finetune.py --short   # 100 ep per env (smoke)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from datetime import datetime

import numpy as np

from highway_env.road.lane import AbstractLane
from highwayenv.utils import (
    patch_intersection_env,
    register_intersection_env,
    register_roundabout_env,
    register_double_intersection_env,
    register_chain_intersection_env,
)
from src import project_globals
from src.experiment_run_paths import EXPERIMENT_RUNS_ROOT
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.experiment.new_envs_config import (
    make_roundabout_env_config,
    make_double_intersection_env_config,
)
from src.experiment.scenarios_config import make_env_config_exp7
from src.training.training_handler import training_loop
from src.model.model_handler import save_models, load_models_from_paths
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers
from run_chain_scalability import generate_chain_scenario

logging.basicConfig(level=logging.WARNING)

N_AGENTS = 6
N_INTERSECTIONS_CHAIN = 2
EPISODES_PER_ENV = 400


def _patch_vehicles_count(cfg: dict) -> dict:
    """Ensure observation returns rows for all N_AGENTS vehicles."""
    cfg.setdefault("observation", {})["vehicles_count"] = N_AGENTS
    return cfg
CYCLES = 4

_LANE_W = AbstractLane.DEFAULT_WIDTH
_OUTER = (_LANE_W + 5) + _LANE_W / 2
CHAIN_SPACING = 2 * _OUTER + 80

CKPT_AGENT = os.path.join(_REPO, "models_to_check", "agent", "ckpt_agent6.pth")
CKPT_MASTER = os.path.join(_REPO, "models_to_check", "master", "ckpt_master6.pth")

PHASES = [
    {
        "name": "intersection",
        "env_id": "RELintersection-v0",
        "make_config": lambda: _make_intersection_cfg(),
    },
    {
        "name": "roundabout",
        "env_id": "RELroundabout-v0",
        "make_config": lambda: _patch_vehicles_count(make_roundabout_env_config(
            collision_reward=-50, arrived_reward=50, starvation_reward=0, high_speed_reward=5,
        )),
    },
    {
        "name": "double_intersection",
        "env_id": "RELdouble-intersection-v0",
        "make_config": lambda: _patch_vehicles_count(make_double_intersection_env_config(
            collision_reward=-50, arrived_reward=50, starvation_reward=0, high_speed_reward=5,
        )),
    },
    {
        "name": "chain",
        "env_id": "RELchain-intersection-v0",
        "make_config": lambda: _make_chain_cfg(),
    },
]


def _base_rewards() -> dict:
    return dict(
        collision_reward=-50, arrived_reward=50, starvation_reward=0, high_speed_reward=5,
    )


def _make_intersection_cfg() -> dict:
    cfg = make_env_config_exp7(
        collision_reward=-50, arrived_reward=50, starvation_reward=0, high_speed_reward=5,
    )
    cfg["observation"]["vehicles_count"] = N_AGENTS
    return cfg


def _make_chain_cfg() -> dict:
    scenarios = [
        generate_chain_scenario(N_INTERSECTIONS_CHAIN, N_AGENTS, np.random.default_rng(12345 + i))
        for i in range(200)
    ]
    controlled = {}
    defaults = [
        (("I0_o0", "I0_ir0", 0), "I1_o2"),
        (("I0_o2", "I0_ir2", 0), "I1_o0"),
        (("I0_o1", "I0_ir1", 0), "I1_o2"),
        (("I1_o0", "I1_ir0", 0), "I0_o2"),
        (("I1_o2", "I1_ir2", 0), "I0_o0"),
        (("I1_o3", "I1_ir3", 0), "I0_o2"),
    ]
    colors = [(0, 204, 0), (0, 0, 204), (204, 0, 0), (204, 204, 0), (0, 204, 204), (204, 0, 204)]
    for i, name in enumerate([f"car{j+1}" for j in range(N_AGENTS)]):
        lk, dest = defaults[i]
        controlled[name] = {
            "start_lane": lk, "destination": dest, "speed": 5,
            "init_location": {"longitudinal": 40 - i * 20, "lateral": 0},
            "color": colors[i],
        }
    cfg = sc.create_full_environment_config({"controlled_cars": controlled, "static_cars": {}, **_base_rewards()})
    cfg["observation"]["vehicles_count"] = N_AGENTS
    max_x = 100 + (N_INTERSECTIONS_CHAIN - 1) * 102 + 100
    cfg["observation"]["features_range"] = {
        "x": [-100, max_x], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20],
    }
    cfg.update({
        "n_intersections": N_INTERSECTIONS_CHAIN,
        "connector_length": 80,
        "chain_scenarios": scenarios,
        "chain_scenarios_only": True,
        "initial_vehicle_count": 0,
        "duration": 120,
        "policy_frequency": 1,
    })
    return cfg


def _attach_chain_overrides(wrapped_env) -> None:
    """Local-frame master states + bootstrap obs (same as train_chain.py)."""
    def _chain_prepare_state(_self_or_obs):
        drv = wrapped_env.env
        inner = drv._get_unwrapped_env()
        out = []
        for v in inner.controlled_vehicles:
            if getattr(v, "is_arrived", False):
                out.append([0.0, 0.0, 0.0, 0.0])
            else:
                vel = v.velocity if hasattr(v, "velocity") else np.zeros(2)
                x = float(v.position[0])
                zone = max(0, min(N_INTERSECTIONS_CHAIN - 1, round(x / CHAIN_SPACING)))
                out.append([x - zone * CHAIN_SPACING, float(v.position[1]),
                            float(vel[0]), float(vel[1])])
        return np.asarray(out, dtype=np.float32)

    def _build_full_obs(local_embeddings):
        drv = wrapped_env.env
        states = drv.current_state
        obs_list = []
        for i in range(N_AGENTS):
            if states is not None and len(states.shape) == 2 and i < states.shape[0]:
                car_state = states[i]
            elif states is not None and len(states.shape) == 1:
                car_state = states[i * 4:i * 4 + 4]
            else:
                car_state = np.zeros(4, dtype=np.float32)
            emb = np.asarray(local_embeddings[i], dtype=np.float32).reshape(-1)
            obs_list.append(np.concatenate([car_state, emb]))
        return obs_list

    wrapped_env.env._prepare_state_for_master = _chain_prepare_state
    wrapped_env.env.build_full_obs = _build_full_obs


def _make_experiment(exp_path: str, env_id: str, episodes: int) -> Experiment:
    return Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID="all_env_finetune",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=CYCLES,
        ENT_COEF=0.02,
        COLLISION_REWARD=-50,
        REACHED_TARGET_REWARD=50,
        STARVATION_REWARD=0,
        HIGH_SPEED_REWARD=5,
        AGENT_REWARD_MODE="global",
        FULL_JOINT_TRAINING=True,
        COTRAIN_CYCLES=True,
        AGENT_LR=1e-3,
        MASTER_LR=1e-4,
        CLIP_RANGE=0.2,
        GAMMA=0.90,
        GAE_LAMBDA=0.90,
        AGENT_NET_ARCH="wide",
        EPISODE_AMOUNT_FOR_TRAIN=3,
        VF_COEF=1.0,
        N_PPO_EPOCHS=5,
        EPISODES_PER_CYCLE=max(1, episodes // CYCLES),
    )


def run_phase(phase: dict, exp_root: str, agent_pth: str, master_pth: str, episodes: int) -> tuple[str, str]:
    phase_dir = os.path.join(exp_root, phase["name"])
    os.makedirs(phase_dir, exist_ok=True)

    project_globals.reset_globals()
    exp = _make_experiment(phase_dir, phase["env_id"], episodes)
    exp.ENV_ID = phase["env_id"]
    exp.EXPERIMENT_PATH = phase_dir
    exp.SAVE_MODEL_DIRECTORY = os.path.join(phase_dir, "trained_model")
    exp.CONFIG = phase["make_config"]()

    setup_experiment_dirs(phase_dir)
    master_model, agent_model, wrapped_env = initialize_models(exp, exp.CONFIG)

    if not load_models_from_paths(agent_model, master_model, agent_pth, master_pth):
        print(f"  WARNING: could not load {agent_pth}")

    wrapped_env.env.master_model = master_model

    # build_full_obs is called by training_handler for the bootstrap obs after each PPO batch.
    # It concatenates each car's 4-D state with its assigned local-master embedding.
    def _generic_build_full_obs(local_embeddings):
        drv = wrapped_env.env
        states = drv.current_state
        obs_list = []
        for i in range(N_AGENTS):
            if states is not None and len(states.shape) == 2 and i < states.shape[0]:
                car_state = states[i]
            elif states is not None and len(states.shape) == 1:
                car_state = states[i * 4:i * 4 + 4]
            else:
                car_state = np.zeros(4, dtype=np.float32)
            emb = np.asarray(local_embeddings[i], dtype=np.float32).reshape(-1)
            obs_list.append(np.concatenate([car_state, emb]))
        return obs_list
    wrapped_env.env.build_full_obs = _generic_build_full_obs

    if phase["name"] == "chain":
        _attach_chain_overrides(wrapped_env)

    os.makedirs(os.path.join(phase_dir, "agent_logs"), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, "master_logs"), exist_ok=True)
    from stable_baselines3.common.logger import configure as sb3_configure
    agent_logger = sb3_configure(os.path.join(phase_dir, "agent_logs"), ["stdout", "csv"])
    master_logger = sb3_configure(os.path.join(phase_dir, "master_logs"), ["stdout", "csv"])
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    print(f"\n{'='*60}\n  Phase: {phase['name']}  ({phase['env_id']})  {episodes} episodes\n{'='*60}")
    agent_model, master_model, collisions, _, _, results, _ = training_loop(
        experiment=exp, env=wrapped_env,
        agent_model=agent_model, master_model=master_model,
    )
    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)

    arr = [v for v in results.get("arrival_rates", []) if v is not None]
    summary = {
        "env": phase["name"],
        "env_id": phase["env_id"],
        "episodes": episodes,
        "arrival_avg_pct": float(np.mean(arr)) if arr else None,
        "arrival_last50_pct": float(np.mean(arr[-50:])) if len(arr) >= 50 else None,
        "collisions": int(collisions),
    }
    with open(os.path.join(phase_dir, "phase_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    return (
        os.path.join(exp.SAVE_MODEL_DIRECTORY + "_agent.pth"),
        os.path.join(exp.SAVE_MODEL_DIRECTORY + "_master.pth"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short", action="store_true", help="100 episodes per env (smoke)")
    ap.add_argument("--episodes", type=int, default=0, help="Override episodes per env")
    args = ap.parse_args()

    patch_intersection_env()
    register_intersection_env()
    register_roundabout_env()
    register_double_intersection_env()
    register_chain_intersection_env()

    episodes = args.episodes or (100 if args.short else EPISODES_PER_ENV)
    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    exp_root = os.path.join(EXPERIMENT_RUNS_ROOT, f"all_env_finetune_{ts}")
    os.makedirs(exp_root, exist_ok=True)

    agent_pth, master_pth = CKPT_AGENT, CKPT_MASTER
    phase_summaries = []

    for phase in PHASES:
        agent_pth, master_pth = run_phase(phase, exp_root, agent_pth, master_pth, episodes)
        with open(os.path.join(exp_root, phase["name"], "phase_summary.json")) as f:
            phase_summaries.append(json.load(f))

    final_agent = os.path.join(exp_root, "final_agent.pth")
    final_master = os.path.join(exp_root, "final_master.pth")
    import shutil
    shutil.copy2(agent_pth, final_agent)
    shutil.copy2(master_pth, final_master)

    manifest = {
        "run": exp_root,
        "episodes_per_env": episodes,
        "n_agents": N_AGENTS,
        "environments": [p["name"] for p in PHASES],
        "note_triangular_pentagonal": (
            "Not implemented in highwayenv/. Available: intersection, roundabout, "
            "double_intersection, chain."
        ),
        "phases": phase_summaries,
        "final_checkpoints": {"agent": final_agent, "master": final_master},
    }
    with open(os.path.join(exp_root, "finetune_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Final weights:\n  {final_agent}\n  {final_master}")
    print(f"Manifest: {os.path.join(exp_root, 'finetune_manifest.json')}")


if __name__ == "__main__":
    main()
