"""
CURRICULUM fine-tune on the connected chain.

The chain env serves scenarios at random, so a curriculum cannot be expressed by
ordering one pool — instead we train in two warm-started stages of increasing
difficulty:

  Stage A  (easy)  : single-hop adjacent crossings, generously staggered. Teaches
                     basic connector turning + zone hand-off without the early
                     collisions that stall learning. Warm-starts from ckpt6.
  Stage B  (hard)  : the FULL evaluation distribution (windowed corridor traverses
                     from run_chain_scalability.generate_chain_scenario). Warm-starts
                     from Stage A's best checkpoint.

Both stages use local-frame normalization (each car's x shifted to its zone centre),
identical to the evaluation, so the fine-tuned model is in-distribution at eval time.

After Stage B finishes, evaluate it against the baseline with:

  py -3 run_chain_scalability.py --n-intersections 2,5,15 --n-scenarios 30 --no-plots \
      --agent  experiment_runs/<run>/stageB/trained_model_agent.pth \
      --master experiment_runs/<run>/stageB/trained_model_master.pth
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

from highwayenv.utils import (patch_intersection_env, register_intersection_env,
                              register_chain_intersection_env)
from src import project_globals
from src.experiment_run_paths import EXPERIMENT_RUNS_ROOT
from src.experiment.experiment_config import Experiment
from src.training.training_handler import training_loop
from src.model.model_handler import save_models, load_models_from_paths
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers

import train_chain as tc  # reuse env-config, dims, local-frame SPACING, plotting

logging.basicConfig(level=logging.WARNING)

EPISODES_PER_CYCLE = 250
CYCLES_PER_STAGE = 3              # 2 stages x 3 cycles x 250 = 1500 episodes total
N_INTERSECTIONS = tc.N_INTERSECTIONS
N_AGENTS = tc.N_AGENTS

CKPT_AGENT = os.path.join(_REPO, "models_to_check", "agent", "ckpt_agent6.pth")
CKPT_MASTER = os.path.join(_REPO, "models_to_check", "master", "ckpt_master6.pth")


def generate_singlehop_scenarios(n_int: int, n_agents: int, n_scenarios: int = 200, seed: int = 12345):
    """EASY stage: every car crosses to an ADJACENT intersection, staggered 20 m apart
    on its approach lane so spawns are collision-free and conflicts are mild."""
    rng = np.random.default_rng(seed)
    approaches = []
    for i in range(n_int):
        for corner in range(4):
            if corner == 3 and i < n_int - 1:
                continue
            if corner == 1 and i > 0:
                continue
            approaches.append(((f"I{i}_o{corner}", f"I{i}_ir{corner}", 0), i, corner))
    exits = []
    for i in range(n_int):
        exits += [f"I{i}_o0", f"I{i}_o2"]
    exits += ["I0_o1", f"I{n_int - 1}_o3"]

    scenarios = []
    for _ in range(n_scenarios):
        agents, lane_count = [], {}
        for _ in range(n_agents):
            counts = [(lane_count.get(j, 0), j) for j in range(len(approaches))]
            mn = min(c for c, _ in counts)
            ap_idx = int(rng.choice([j for c, j in counts if c == mn]))
            n_on_lane = lane_count.get(ap_idx, 0)
            lane_count[ap_idx] = n_on_lane + 1
            lane_key, src_int, _c = approaches[ap_idx]
            lo, hi = max(0, src_int - 1), min(n_int - 1, src_int + 1)
            targets = [t for t in range(lo, hi + 1) if t != src_int] or [src_int]
            dst = int(rng.choice(targets))
            int_exits = [e for e in exits if e.startswith(f"I{dst}_")] or exits
            agents.append((lane_key, str(rng.choice(int_exits)), -n_on_lane * 20.0))
        scenarios.append({"agents": agents, "static": []})
    return scenarios


def _install_overrides(wrapped_env, master_model):
    """Local-frame state prep + build_full_obs (identical to train_chain)."""
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
                zone = max(0, min(N_INTERSECTIONS - 1, round(x / tc.SPACING)))
                out.append([x - zone * tc.SPACING, float(v.position[1]),
                            float(vel[0]), float(vel[1])])
        return np.asarray(out, dtype=np.float32)
    wrapped_env.env._prepare_state_for_master = _chain_prepare_state

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
    wrapped_env.env.build_full_obs = _build_full_obs
    wrapped_env.env.master_model = master_model


def run_stage(stage_name, scenarios, load_agent, load_master, stage_dir):
    """One warm-started training stage. Returns paths to its best checkpoint."""
    os.makedirs(stage_dir, exist_ok=True)
    project_globals.reset_globals()

    exp = Experiment(
        RENDER_MODE=None, EXPERIMENT_ID=f"chain_curriculum_{stage_name}",
        LOAD_MODEL_DIRECTORY='', EPOCHS=1, CYCLES=CYCLES_PER_STAGE,
        ENT_COEF=0.02, COLLISION_REWARD=-50, REACHED_TARGET_REWARD=50,
        STARVATION_REWARD=0, HIGH_SPEED_REWARD=5, AGENT_REWARD_MODE='global',
        FULL_JOINT_TRAINING=True, COTRAIN_CYCLES=True,
        AGENT_LR=1e-3, MASTER_LR=1e-4, CLIP_RANGE=0.2, GAMMA=0.90, GAE_LAMBDA=0.90,
        AGENT_NET_ARCH='wide', EPISODE_AMOUNT_FOR_TRAIN=3, VF_COEF=1.0, N_PPO_EPOCHS=5,
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
    )
    exp.ENV_ID = "RELchain-intersection-v0"
    exp.EXPERIMENT_PATH = stage_dir
    exp.SAVE_MODEL_DIRECTORY = os.path.join(stage_dir, 'trained_model')

    env_config = tc.make_chain_env_config(scenarios)
    exp.CONFIG = env_config

    setup_experiment_dirs(stage_dir)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)

    if not load_models_from_paths(agent_model, master_model, load_agent, load_master):
        print(f"  [WARN] {stage_name}: could not load {load_agent}")
    else:
        print(f"  [{stage_name}] warm-started from {load_agent}")

    _install_overrides(wrapped_env, master_model)
    a_log, m_log = setup_loggers(stage_dir)
    agent_model.set_logger(a_log)
    master_model.set_logger(m_log)

    agent_model, master_model, collisions, _, _, results, _ = training_loop(
        experiment=exp, env=wrapped_env, agent_model=agent_model, master_model=master_model)

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)
    tc.EPISODES_PER_CYCLE, tc.CYCLES = EPISODES_PER_CYCLE, CYCLES_PER_STAGE
    try:
        tc.save_plots(results, stage_dir)
    except Exception as e:
        print(f"  [plot warn] {e}")

    arr = [v for v in results.get('arrival_rates', []) if v is not None]
    last50 = arr[-50:] if len(arr) >= 50 else arr
    summary = {'stage': stage_name, 'collisions': collisions,
               'arrival_avg_pct': float(np.mean(arr)) if arr else None,
               'arrival_last50_pct': float(np.mean(last50)) if last50 else None}
    with open(os.path.join(stage_dir, 'stage_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [{stage_name}] arrival avg={summary['arrival_avg_pct']}, "
          f"last50={summary['arrival_last50_pct']}, crashes={collisions}")

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    # Chain stages via the FINAL trained model: the training-time arrival metric is
    # unreliable on the chain (multi-hop arrival is under-counted), so the rolling
    # "best_model" selection cannot be trusted — the end-of-stage model is the fully
    # trained one and the correct warm-start for the next stage.
    final_agent = exp.SAVE_MODEL_DIRECTORY + '_agent.pth'
    final_master = exp.SAVE_MODEL_DIRECTORY + '_master.pth'
    return final_agent, final_master


if __name__ == '__main__':
    patch_intersection_env()
    register_intersection_env()
    register_chain_intersection_env()

    ts = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
    root = os.path.join(EXPERIMENT_RUNS_ROOT, f'chain_curriculum_{ts}')
    os.makedirs(root, exist_ok=True)
    print(f"\n{'#'*70}\n  CHAIN CURRICULUM FINE-TUNE  |  2 stages x {CYCLES_PER_STAGE*EPISODES_PER_CYCLE} ep")
    print(f"  Output: {root}\n{'#'*70}\n")

    # Stage A — easy single-hop crossings, warm-start from ckpt6.
    easy = generate_singlehop_scenarios(N_INTERSECTIONS, N_AGENTS, n_scenarios=200)
    print(f"  Stage A: {len(easy)} single-hop scenarios")
    a_ag, a_ma = run_stage("stageA_easy", easy, CKPT_AGENT, CKPT_MASTER,
                           os.path.join(root, "stageA"))

    # Stage B — hard eval distribution, warm-start from Stage A's best.
    hard = tc.generate_chain_training_scenarios(N_INTERSECTIONS, N_AGENTS, n_scenarios=200)
    print(f"  Stage B: {len(hard)} traverse scenarios (eval distribution)")
    b_ag, b_ma = run_stage("stageB_hard", hard, a_ag, a_ma,
                           os.path.join(root, "stageB"))

    with open(os.path.join(root, 'curriculum.json'), 'w') as f:
        json.dump({'stageA_best': [a_ag, a_ma], 'stageB_best': [b_ag, b_ma],
                   'episodes_total': 2 * CYCLES_PER_STAGE * EPISODES_PER_CYCLE}, f, indent=2)

    print(f"\n{'#'*70}\n  DONE. Final model (Stage B best):\n   agent={b_ag}\n   master={b_ma}")
    print(f"\n  Evaluate vs baseline:")
    print(f"   py -3 run_chain_scalability.py --n-intersections 2,5,15 --n-scenarios 30 "
          f"--no-plots --agent \"{b_ag}\" --master \"{b_ma}\"\n{'#'*70}")
