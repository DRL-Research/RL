"""
Final production run — exact W01 config from sweep 3 (the one that reached ~82% arrival).

  1_500 episodes = 375 × 4 cycles  |  FULL_JOINT_TRAINING=True
  (master + agents every cycle — avoids alternating-freeze cycle jumps).

  agent_lr=3e-3,  master_lr=3e-4
  ent_coef=0.05,  clip_range=0.2
  gamma=0.90,     gae_lambda=0.90
  arch='wide',    ep_for_train=3
  cotrain=False,  starvation=0,  high_speed=5
  vf_coef=1.0,    n_ppo_epochs=5
  collision=-50,  arrived=50

  Run from repo root or IDE — cwd is normalized to this file’s directory.
"""

import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# repo root + every scripts/ sub-folder go on sys.path so cross-script imports
# (e.g. ``import run_scalability_suite``) keep working from any category folder.
for _p in (
    _REPO,
    os.path.join(_REPO, "scripts", "training"),
    os.path.join(_REPO, "scripts", "evaluation"),
    os.path.join(_REPO, "scripts", "visualization"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO)

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src import project_globals
from src.experiment_run_paths import EXPERIMENT_RUNS_ROOT
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import training_loop
from src.model.model_handler import save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers

logging.basicConfig(level=logging.WARNING)

EPISODES_PER_CYCLE = 375    # × 4 cycles = 1_500 episodes
CYCLES             = 4

# ── helpers ────────────────────────────────────────────────────────────────────

def _smooth(values, window=60):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    pad    = window // 2
    padded = np.concatenate([np.full(pad, np.nan), arr, np.full(pad, np.nan)])
    s = np.convolve(np.where(np.isnan(padded), 0, padded), kernel, mode='valid')
    c = np.convolve((~np.isnan(padded)).astype(float), kernel, mode='valid')
    s = s / np.where(c > 0, c, 1)
    s[c == 0] = np.nan
    return s[:len(arr)]


def save_plots(results: dict, exp_path: str):
    fig = plt.figure(figsize=(18, 11))
    total_ep = EPISODES_PER_CYCLE * CYCLES
    fig.suptitle(
        f'Final Run — W01 ({total_ep} ep, {CYCLES} cycles, FULL_JOINT master+agents)',
        fontsize=15, fontweight='bold',
    )
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def _ps(ax, vals, color, lbl, alpha_raw=0.12):
        if not vals:
            return
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x   = np.arange(1, len(raw) + 1)
        ax.plot(x, raw, color=color, alpha=alpha_raw, linewidth=0.5)
        ax.plot(x, _smooth(vals), color=color, linewidth=2.2, label=lbl)

    _ps(axes[0], results.get('arrival_rates',   []), '#2196F3', 'Arrival %')
    axes[0].set_title('Arrival Rate', fontsize=12); axes[0].set_ylabel('%')
    axes[0].set_ylim(-5, 105); axes[0].axhline(80, color='green', linestyle='--', alpha=0.4, label='80% target')
    axes[0].grid(True, alpha=0.25); axes[0].legend(fontsize=9)

    _ps(axes[1], results.get('episode_rewards', []), '#4CAF50', 'Reward')
    axes[1].set_title('Episode Reward', fontsize=12); axes[1].set_ylabel('Reward')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.4)
    axes[1].grid(True, alpha=0.25); axes[1].legend(fontsize=9)

    for key, col, lbl_s in [
        ('master_policy_losses', '#E53935', 'Policy'),
        ('master_value_losses',  '#FB8C00', 'Value'),
        ('master_total_losses',  '#8E24AA', 'Total'),
    ]:
        _ps(axes[2], results.get(key, []), col, lbl_s, 0.2)
    axes[2].set_title('Master Losses', fontsize=12); axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.25); axes[2].legend(fontsize=9)

    for key, col, lbl_s in [
        ('agent_group0_total_losses', '#00ACC1', 'Agent G0 (cars 0-2)'),
        ('agent_group1_total_losses', '#43A047', 'Agent G1 (cars 3-5)'),
    ]:
        _ps(axes[3], results.get(key, []), col, lbl_s, 0.2)
    axes[3].set_title('Agent Losses', fontsize=12); axes[3].set_ylabel('Loss')
    axes[3].grid(True, alpha=0.25); axes[3].legend(fontsize=9)

    axes[0].set_xlabel('Episode'); axes[1].set_xlabel('Episode')
    axes[2].set_xlabel('Training call'); axes[3].set_xlabel('Training call')

    out = os.path.join(exp_path, 'results.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [plot saved] {out}")


def save_json(results: dict, collisions: int, exp_path: str):
    arr_vals = [v for v in results.get('arrival_rates', []) if v is not None]
    last50   = arr_vals[-50:] if len(arr_vals) >= 50 else arr_vals
    summary  = {
        'total_collisions':        collisions,
        'arrival_rate_avg_pct':    float(np.mean(arr_vals)) if arr_vals else None,
        'arrival_rate_last50_pct': float(np.mean(last50))   if last50   else None,
    }
    with open(os.path.join(exp_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  arrival avg={summary['arrival_rate_avg_pct']:.1f}%  "
          f"last50={summary['arrival_rate_last50_pct']:.1f}%  "
          f"crashes={collisions}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    patch_intersection_env()
    register_intersection_env()

    ts       = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
    os.makedirs(EXPERIMENT_RUNS_ROOT, exist_ok=True)
    exp_path = os.path.join(EXPERIMENT_RUNS_ROOT, f'final_{ts}')
    os.makedirs(exp_path, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  FINAL RUN  —  {EPISODES_PER_CYCLE * CYCLES} episodes  "
          f"({EPISODES_PER_CYCLE}/cycle × {CYCLES} cycles, FULL_JOINT)")
    print(f"  W01: ent=0.05  clip=0.2  agent_lr=3e-3  master_lr=3e-4  arch=wide  ep=3")
    print(f"  Path: {exp_path}")
    print(f"{'#'*70}\n")

    project_globals.reset_globals()

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID='final_W01',
        LOAD_MODEL_DIRECTORY='',
        EPOCHS=1,
        CYCLES=CYCLES,
        ENT_COEF=0.05,
        COLLISION_REWARD=-50,
        REACHED_TARGET_REWARD=50,
        STARVATION_REWARD=0,
        HIGH_SPEED_REWARD=5,
        AGENT_REWARD_MODE='global',
        # Alternating cycles (False) cause sharp plot jumps + frozen-peer non-stationarity.
        FULL_JOINT_TRAINING=True,   # master+agents every cycle; 4×375 episodes
        COTRAIN_CYCLES=True,        # ignored when FULL_JOINT_TRAINING is True
        AGENT_LR=3e-3,
        MASTER_LR=3e-4,
        CLIP_RANGE=0.2,
        GAMMA=0.90,
        GAE_LAMBDA=0.90,
        AGENT_NET_ARCH='wide',
        EPISODE_AMOUNT_FOR_TRAIN=3,
        VF_COEF=1.0,
        N_PPO_EPOCHS=5,
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
    )
    exp.EXPERIMENT_PATH      = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, 'trained_model')

    env_config = sc.make_env_config_exp7(
        collision_reward=-50,
        arrived_reward=50,
        starvation_reward=0,
        high_speed_reward=5,
    )

    setup_experiment_dirs(exp_path)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)
    agent_logger, master_logger = setup_loggers(exp_path)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    agent_model, master_model, collisions, _, _, results = training_loop(
        experiment=exp, env=wrapped_env,
        agent_model=agent_model, master_model=master_model,
    )

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)
    save_plots(results, exp_path)
    save_json(results, collisions, exp_path)

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    print(f"\n{'#'*70}")
    print(f"  DONE.  Results saved to {exp_path}")
    print(f"{'#'*70}")
