"""
Fine-tune master + agents on the CONNECTED chain topology (2 intersections).

The existing checkpoint was trained on perpendicular crossing conflicts.
This script fine-tunes it on the sequential corridor conflicts that arise
in connected intersections — agents must cross intersection boundaries.

Architecture identical to main_final.py W01:
  2 Local Masters x 3 agents, shared weights, Global Master on top.

Key difference: environment is RELchain-intersection-v0 with multi-hop routing.
Initializes from the best existing checkpoint (ckpt6).
"""

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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from highway_env.road.lane import AbstractLane
from highwayenv.utils import patch_intersection_env, register_intersection_env, register_chain_intersection_env
from src import project_globals
from src.experiment_run_paths import EXPERIMENT_RUNS_ROOT
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import training_loop
from src.model.model_handler import save_models, load_models_from_paths
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers
# Train on the EXACT scenario distribution the evaluation uses (windowed perpendicular
# crossings + corridor traverses), so the fine-tuned model is in-distribution at eval.
from run_chain_scalability import generate_chain_scenario

logging.basicConfig(level=logging.WARNING)

EPISODES_PER_CYCLE = 250
CYCLES = 6
N_INTERSECTIONS = 2
AGENTS_PER_INTERSECTION = 3
N_AGENTS = N_INTERSECTIONS * AGENTS_PER_INTERSECTION

# Intersection spacing (identical formula to ChainCell._get_spacing / env _make_road),
# used for local-frame normalization so each zone looks like an isolated intersection.
_LANE_W = AbstractLane.DEFAULT_WIDTH
_OUTER = (_LANE_W + 5) + _LANE_W / 2
SPACING = 2 * _OUTER + 80

CKPT_AGENT = os.path.join(_REPO, "models_to_check", "agent", "ckpt_agent6.pth")
CKPT_MASTER = os.path.join(_REPO, "models_to_check", "master", "ckpt_master6.pth")


def generate_chain_training_scenarios(n_int: int, n_agents: int, n_scenarios: int = 200):
    """Pool of training scenarios drawn from the evaluation generator (jittered, so
    every scenario is slightly different) — matching train and eval distributions."""
    rng = np.random.default_rng(12345)
    return [generate_chain_scenario(n_int, n_agents, rng) for _ in range(n_scenarios)]


def make_chain_env_config(scenarios: list) -> dict:
    """Build env config for chain training."""
    controlled_cars = {}
    car_names = [f"car{i+1}" for i in range(N_AGENTS)]

    # Initial positions — will be overridden by chain_scenarios on each reset.
    # LM1: cars 0-2 start at I0, LM2: cars 3-5 start at I1
    default_lanes = [
        (("I0_o0", "I0_ir0", 0), "I1_o2"),
        (("I0_o2", "I0_ir2", 0), "I1_o0"),
        (("I0_o1", "I0_ir1", 0), "I1_o2"),
        (("I1_o0", "I1_ir0", 0), "I0_o2"),
        (("I1_o2", "I1_ir2", 0), "I0_o0"),
        (("I1_o3", "I1_ir3", 0), "I0_o2"),
    ]

    for i, name in enumerate(car_names):
        lane_key, dest = default_lanes[i]
        controlled_cars[name] = {
            "start_lane": lane_key,
            "destination": dest,
            "speed": 5,
            "init_location": {"longitudinal": 40 - i * 20, "lateral": 0},
            "color": [(0, 204, 0), (0, 0, 204), (204, 0, 0),
                      (204, 204, 0), (0, 204, 204), (204, 0, 204)][i],
        }

    base_config = {
        "controlled_cars": controlled_cars,
        "static_cars": {},
        "collision_reward": -50,
        "arrived_reward": 50,
        "starvation_reward": 0,
        "high_speed_reward": 5,
    }

    cfg = sc.create_full_environment_config(base_config)

    # Ensure observation returns all 6 vehicles (not default 5)
    cfg["observation"]["vehicles_count"] = N_AGENTS

    # Expand features_range to cover the full chain extent (I1 center is at x≈102)
    max_x = 100 + (N_INTERSECTIONS - 1) * 102 + 100
    cfg["observation"]["features_range"] = {
        "x": [-100, max_x],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    }

    cfg.update({
        "n_intersections": N_INTERSECTIONS,
        "connector_length": 80,
        "chain_scenarios": scenarios,
        "chain_scenarios_only": True,
        "initial_vehicle_count": 0,
        "duration": 120,
        "policy_frequency": 1,
    })
    return cfg


def _smooth(values, window=40):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, np.nan), arr, np.full(pad, np.nan)])
    s = np.convolve(np.where(np.isnan(padded), 0, padded), kernel, mode='valid')
    c = np.convolve((~np.isnan(padded)).astype(float), kernel, mode='valid')
    s = s / np.where(c > 0, c, 1)
    s[c == 0] = np.nan
    return s[:len(arr)]


def save_plots(results: dict, exp_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    total_ep = EPISODES_PER_CYCLE * CYCLES
    fig.suptitle(f'Chain Fine-Tune ({total_ep} ep, {N_INTERSECTIONS} intersections)', fontsize=13)

    vals = results.get('arrival_rates', [])
    if vals:
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x = np.arange(1, len(raw) + 1)
        axes[0].plot(x, raw, color='#2196F3', alpha=0.12, linewidth=0.5)
        axes[0].plot(x, _smooth(vals), color='#2196F3', linewidth=2.2, label='Arrival %')
    axes[0].set_title('Arrival Rate'); axes[0].set_ylabel('%')
    axes[0].set_ylim(-5, 105); axes[0].set_xlabel('Episode')
    axes[0].axhline(80, color='green', linestyle='--', alpha=0.4)
    axes[0].legend()

    vals = results.get('episode_rewards', [])
    if vals:
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x = np.arange(1, len(raw) + 1)
        axes[1].plot(x, raw, color='#4CAF50', alpha=0.12, linewidth=0.5)
        axes[1].plot(x, _smooth(vals), color='#4CAF50', linewidth=2.2, label='Reward')
    axes[1].set_title('Episode Reward'); axes[1].set_ylabel('Reward')
    axes[1].set_xlabel('Episode'); axes[1].legend()

    plt.tight_layout()
    out = os.path.join(exp_path, 'chain_training.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [plot] {out}")


if __name__ == '__main__':
    patch_intersection_env()
    register_intersection_env()
    register_chain_intersection_env()

    ts = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
    exp_path = os.path.join(EXPERIMENT_RUNS_ROOT, f'chain_finetune_{ts}')
    os.makedirs(exp_path, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  CHAIN FINE-TUNE  |  {EPISODES_PER_CYCLE * CYCLES} episodes")
    print(f"  {N_INTERSECTIONS} intersections, {N_AGENTS} agents, multi_hop=1.0")
    print(f"  Init from: {CKPT_AGENT}")
    print(f"  Output: {exp_path}")
    print(f"{'#'*70}\n")

    scenarios = generate_chain_training_scenarios(N_INTERSECTIONS, N_AGENTS, n_scenarios=200)
    print(f"  Generated {len(scenarios)} training scenarios")

    project_globals.reset_globals()

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID='chain_finetune',
        LOAD_MODEL_DIRECTORY='',
        EPOCHS=1,
        CYCLES=CYCLES,
        ENT_COEF=0.02,
        COLLISION_REWARD=-50,
        REACHED_TARGET_REWARD=50,
        STARVATION_REWARD=0,
        HIGH_SPEED_REWARD=5,
        AGENT_REWARD_MODE='global',
        FULL_JOINT_TRAINING=True,
        COTRAIN_CYCLES=True,
        AGENT_LR=1e-3,
        MASTER_LR=1e-4,
        CLIP_RANGE=0.2,
        GAMMA=0.90,
        GAE_LAMBDA=0.90,
        AGENT_NET_ARCH='wide',
        EPISODE_AMOUNT_FOR_TRAIN=3,
        VF_COEF=1.0,
        N_PPO_EPOCHS=5,
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
    )
    exp.ENV_ID = "RELchain-intersection-v0"
    exp.EXPERIMENT_PATH = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, 'trained_model')

    env_config = make_chain_env_config(scenarios)
    exp.CONFIG = env_config

    setup_experiment_dirs(exp_path)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)

    # Load pretrained weights (fine-tune, not from scratch)
    loaded = load_models_from_paths(agent_model, master_model, CKPT_AGENT, CKPT_MASTER)
    if loaded:
        print("  Loaded pretrained checkpoint for fine-tuning")
    else:
        print("  WARNING: Could not load checkpoint — training from scratch")

    # Driver.reset() needs a master_model reference for bootstrap obs
    wrapped_env.env.master_model = master_model

    # Override _prepare_state_for_master to read DIRECTLY from vehicles (bypasses the
    # distance-limited Kinematics observation) AND apply local-frame normalization:
    # each car's x is shifted by its current intersection-zone centre, exactly like the
    # evaluation (run_chain_scalability). This keeps every zone in the master's trained
    # coordinate range and removes the train/eval mismatch.
    _orig_prepare = wrapped_env.env._prepare_state_for_master
    def _chain_prepare_state(self_or_obs):
        drv = wrapped_env.env
        inner = drv._get_unwrapped_env()
        out = []
        for v in inner.controlled_vehicles:
            if getattr(v, "is_arrived", False):
                out.append([0.0, 0.0, 0.0, 0.0])
            else:
                vel = v.velocity if hasattr(v, "velocity") else np.zeros(2)
                x = float(v.position[0])
                zone = max(0, min(N_INTERSECTIONS - 1, round(x / SPACING)))
                out.append([x - zone * SPACING, float(v.position[1]),
                            float(vel[0]), float(vel[1])])
        return np.asarray(out, dtype=np.float32)
    wrapped_env.env._prepare_state_for_master = _chain_prepare_state

    # training_handler calls env.env.build_full_obs — add it to Driver
    def _build_full_obs(local_embeddings):
        drv = wrapped_env.env
        states = drv.current_state
        obs_list = []
        for i in range(N_AGENTS):
            if states is not None and len(states.shape) == 2 and i < states.shape[0]:
                car_state = states[i]
            elif states is not None and len(states.shape) == 1:
                car_state = states[i*4:i*4+4]
            else:
                car_state = np.zeros(4, dtype=np.float32)
            emb = np.asarray(local_embeddings[i], dtype=np.float32).reshape(-1)
            obs_list.append(np.concatenate([car_state, emb]))
        return obs_list
    wrapped_env.env.build_full_obs = _build_full_obs

    agent_logger, master_logger = setup_loggers(exp_path)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    agent_model, master_model, collisions, _, _, results, _ = training_loop(
        experiment=exp, env=wrapped_env,
        agent_model=agent_model, master_model=master_model,
    )

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)
    save_plots(results, exp_path)

    arr_vals = [v for v in results.get('arrival_rates', []) if v is not None]
    last50 = arr_vals[-50:] if len(arr_vals) >= 50 else arr_vals
    summary = {
        'total_episodes': EPISODES_PER_CYCLE * CYCLES,
        'total_collisions': collisions,
        'arrival_rate_avg_pct': float(np.mean(arr_vals)) if arr_vals else None,
        'arrival_rate_last50_pct': float(np.mean(last50)) if last50 else None,
        'topology': 'chain',
        'n_intersections': N_INTERSECTIONS,
        'n_agents': N_AGENTS,
        'init_checkpoint': CKPT_AGENT,
    }
    with open(os.path.join(exp_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  arrival avg={summary['arrival_rate_avg_pct']:.1f}%  "
          f"last50={summary['arrival_rate_last50_pct']:.1f}%  "
          f"crashes={collisions}")

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    print(f"\n{'#'*70}")
    print(f"  DONE.  {exp_path}")
    print(f"{'#'*70}")
