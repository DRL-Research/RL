"""
Unified training across ALL THREE environments simultaneously.

Multi-seed experiment with statistical reporting (mean ± std).

Configs A-C: master-contribution proof (baseline / gentle conflict / curriculum)
Configs D-F: cross-environment generalization (train on ONE env, test on ALL)
Config  G  : scalability test (mask agents at test time: 2/4/6 active)

Each training config runs WITH master then WITHOUT master (ablation).
Each condition is repeated across N_SEEDS random seeds.

After training, models are tested on:
  1. Held-out regular scenarios (100 ep per env)
  2. Conflict-only scenarios   (100 ep per env)

Additionally (using the best seed):
  - Master embedding PCA visualizations
  - Scenario layout plots for all environments

Usage:
    python run_unified.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure as _sb3_configure
from src.model.agent_handler import DummyVecEnv

logging.basicConfig(level=logging.ERROR)

import torch
from src.model.master_model import MasterModel

def _set_all_seeds(seed: int):
    """Set all random seeds for reproducibility across a single run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Register all environments ──────────────────────────────────────────────────
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
from src.model.model_handler import load_models, save_models
from src.model.agent_handler import Driver
from src.training.general_utils import initialize_models, setup_experiment_dirs
from src.training.training_handler import _make_master_buffer
from src.training.training_loop_utils import (
    init_training_results,
    prepare_models_for_cycle,
    perform_training_phase,
)
from src.training.episode_utils import process_episode
from src.project_globals import rollout_buffers
from run_learning_experiment import save_plots

# ── Global constants ──────────────────────────────────────────────────────────

PRETRAINED_CHECKPOINT = os.path.join(
    "experiment_runs",
    "grid_23_04_2026-13_51_18",
    "Q02_ep1_ent0005_PL75",
    "best_model",
    "checkpoint",
)

N_TEST_EPISODES      = 30
ROLLING_CRASH_WINDOW = 50
SMOOTH_EP            = 50
ENV_WEIGHTS          = [1/3, 1/3, 1/3]
N_SEEDS              = 5
SEEDS                = [42, 123, 7, 2024, 314]
RUN_CONDITIONS       = [("W_MASTER", False)]
NORMALIZE_MASTER_INPUTS = False
NORMALIZE_AGENT_OBS = False
LOAD_PRETRAINED_CHECKPOINT = not (NORMALIZE_MASTER_INPUTS or NORMALIZE_AGENT_OBS)

ENV_DEFS = [
    ("RELintersection-v0",       "intersection"),
    ("RELroundabout-v0",          "roundabout"),
    ("RELdouble-intersection-v0", "double_intersection"),
]

# ── Q02 base hyperparameters ─────────────────────────────────────────────────
_BASE_HP = dict(
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
    warmup_episodes      = 200,
    peak_arrival_threshold = 75.0,
    n_value_epochs       = 0,
    target_speeds        = [5, 10],
    normalize_master_inputs = NORMALIZE_MASTER_INPUTS,
    normalize_agent_obs    = NORMALIZE_AGENT_OBS,
    load_pretrained_checkpoint = LOAD_PRETRAINED_CHECKPOINT,
)

# ── Experiment configurations ─────────────────────────────────────────────────

@dataclass
class RunConfig:
    name: str
    total_episodes: int
    conflict_schedule: list = field(default_factory=list)
    train_env_only: str | None = None   # if set, train on this env_id only
    test_only: bool = False             # skip training, just test

# ── Essential configs (Phase 1): master-contribution proof ────────────────────

CONFIGS_ABC = [
    RunConfig(
        name="A_base",
        total_episodes=2500,
        conflict_schedule=[],
    ),
]

# ── Optional configs: uncomment to add conflict / curriculum / generalization ─
# CONFIGS_ABC += [
#     RunConfig(name="B_gentle", total_episodes=4500, conflict_schedule=[(0, 0.2)]),
#     RunConfig(name="C_curric", total_episodes=5000, conflict_schedule=[(0, 0.0), (2500, 0.5)]),
# ]

CONFIGS_DEF = [
    # RunConfig(name="D_inter", total_episodes=4500, conflict_schedule=[(0, 0.0)],
    #           train_env_only="RELintersection-v0"),
    # RunConfig(name="E_round", total_episodes=4500, conflict_schedule=[(0, 0.0)],
    #           train_env_only="RELroundabout-v0"),
    # RunConfig(name="F_dbl",   total_episodes=4500, conflict_schedule=[(0, 0.0)],
    #           train_env_only="RELdouble-intersection-v0"),
]

ALL_CONFIGS = CONFIGS_ABC + CONFIGS_DEF


def _get_conflict_ratio(schedule: list, episode: int) -> float:
    ratio = 0.0
    for threshold, r in schedule:
        if episode >= threshold:
            ratio = r
    return ratio


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_experiment(exp_path: str, total_episodes: int, env_id: str | None = None) -> Experiment:
    cycles = 1
    eps_per_cycle = total_episodes
    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID="unified",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=cycles,
        ENT_COEF=_BASE_HP["ent_coef"],
        ENT_COEF_FINAL=_BASE_HP["ent_coef_final"],
        WARMUP_EPISODES=_BASE_HP["warmup_episodes"],
        PEAK_ARRIVAL_THRESHOLD=_BASE_HP["peak_arrival_threshold"],
        N_VALUE_EPOCHS=_BASE_HP["n_value_epochs"],
        COLLISION_REWARD=_BASE_HP["collision_reward"],
        REACHED_TARGET_REWARD=_BASE_HP["arrived_reward"],
        STARVATION_REWARD=_BASE_HP["starvation_reward"],
        HIGH_SPEED_REWARD=_BASE_HP["high_speed_reward"],
        AGENT_REWARD_MODE=_BASE_HP["reward_mode"],
        FULL_JOINT_TRAINING=False,
        COTRAIN_CYCLES=False,
        AGENT_LR=_BASE_HP["agent_lr"],
        MASTER_LR=_BASE_HP["master_lr"],
        CLIP_RANGE=_BASE_HP["clip_range"],
        GAMMA=_BASE_HP["gamma"],
        GAE_LAMBDA=_BASE_HP["gae_lambda"],
        AGENT_NET_ARCH=_BASE_HP["agent_net_arch"],
        EPISODE_AMOUNT_FOR_TRAIN=_BASE_HP["ep_for_train"],
        VF_COEF=_BASE_HP["vf_coef"],
        N_PPO_EPOCHS=_BASE_HP["n_ppo_epochs"],
        EPISODES_PER_CYCLE=eps_per_cycle,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
        N_STEPS=int(_BASE_HP["n_steps"]),
    )
    exp.EXPERIMENT_PATH      = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, "trained_model")
    exp.NORMALIZE_MASTER_INPUTS = NORMALIZE_MASTER_INPUTS
    exp.NORMALIZE_AGENT_OBS = NORMALIZE_AGENT_OBS
    if env_id:
        exp.ENV_ID = env_id
    return exp


def _make_env_config(env_id: str, conflict_ratio: float = 0.0,
                     use_held_out: bool = False,
                     use_conflict_only: bool = False) -> dict:
    kw = dict(
        collision_reward  = _BASE_HP["collision_reward"],
        arrived_reward    = _BASE_HP["arrived_reward"],
        starvation_reward = _BASE_HP["starvation_reward"],
        high_speed_reward = _BASE_HP["high_speed_reward"],
        target_speeds     = _BASE_HP["target_speeds"],
    )
    if env_id == "RELintersection-v0":
        cfg = make_env_config_exp7(**kw)
    elif env_id == "RELroundabout-v0":
        cfg = make_roundabout_env_config(**kw)
    elif env_id == "RELdouble-intersection-v0":
        cfg = make_double_intersection_env_config(**kw)
    else:
        raise ValueError(f"Unknown env_id: {env_id!r}")
    cfg["conflict_ratio"] = conflict_ratio
    cfg["use_held_out_scenarios"] = use_held_out
    cfg["use_conflict_scenarios_only"] = use_conflict_only
    return cfg


def _setup_loggers_csv(base_path: str):
    agent_logger  = _sb3_configure(os.path.join(base_path, "agent_logs"),  ["csv"])
    master_logger = _sb3_configure(os.path.join(base_path, "master_logs"), ["csv"])
    return agent_logger, master_logger


def _rolling_mean(lst: list, w: int) -> list:
    out = []
    for i in range(len(lst)):
        window = [v for v in lst[max(0, i - w + 1): i + 1] if v is not None]
        out.append(float(np.mean(window)) if window else 0.0)
    return out


def _env_label_to_id(label: str) -> str:
    for eid, lbl in ENV_DEFS:
        if lbl == label:
            return eid
    raise ValueError(f"Unknown label: {label}")


# ── Multi-environment wrapper ──────────────────────────────────────────────────

class MultiEnvWrapper:
    def __init__(self, wrapped_envs: list, env_names: list, weights: list | None = None):
        self.wrapped_envs = wrapped_envs
        self.env_names    = env_names
        self.weights      = weights
        self.current      = wrapped_envs[0]
        self.current_name = env_names[0]

    def pick_env(self):
        idx = random.choices(range(len(self.wrapped_envs)), weights=self.weights)[0]
        self.current      = self.wrapped_envs[idx]
        self.current_name = self.env_names[idx]

    def reset(self):
        return self.current.reset()

    def step(self, actions):
        return self.current.step(actions)

    def close(self):
        for e in self.wrapped_envs:
            try:
                e.close()
            except Exception:
                pass

    @property
    def env(self):
        return self.current.env

    @property
    def observation_space(self):
        return self.current.observation_space

    @property
    def action_space(self):
        return self.current.action_space

    @property
    def num_envs(self):
        return 1

    def env_method(self, method_name, *args, **kwargs):
        return self.current.env_method(method_name, *args, **kwargs)

    def get_attr(self, attr_name, indices=None):
        return self.current.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.current.set_attr(attr_name, value, indices)

    def render(self, *args, **kwargs):
        return self.current.render(*args, **kwargs)

    def seed(self, seed=None):
        return self.current.seed(seed)

    def update_conflict_ratio(self, new_ratio: float):
        for wrapped in self.wrapped_envs:
            try:
                wrapped.envs[0].highway_env.unwrapped.config["conflict_ratio"] = new_ratio
            except Exception:
                pass


# ── Unified training loop ──────────────────────────────────────────────────────

def unified_training_loop(
    exp: Experiment,
    multi_env: MultiEnvWrapper,
    agent_model,
    master_model,
    exp_path: str,
    total_episodes: int,
    conflict_schedule: list,
    skip_master_training: bool = False,
) -> tuple:
    from src.training.episode_utils import (
        _build_local_master_input,
        _build_global_master_input,
    )

    rollout_buffers.clear()
    project_globals.local_master_rollout_buffers.clear()
    project_globals.global_master_rollout_buffer = None

    from src.training.training_loop_utils import agent_value_normalizer, master_value_normalizer
    agent_value_normalizer.__init__()
    master_value_normalizer.__init__()

    for _ in exp.CONFIG["controlled_cars"]:
        rollout_buffers.append(RolloutBuffer(
            buffer_size=exp.N_STEPS,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(exp.STATE_INPUT_SIZE,)),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            gamma=exp.GAMMA,
            gae_lambda=exp.GAE_LAMBDA,
            n_envs=1,
        ))
    for _ in range(exp.NUM_LOCAL_MASTERS):
        project_globals.local_master_rollout_buffers.append(_make_master_buffer(exp))
    project_globals.global_master_rollout_buffer = _make_master_buffer(exp)

    collision_counter = 0
    episode_counter   = 0
    total_steps       = 0
    results           = init_training_results()
    results["collision_rates"] = []
    env_arrivals = {n: [] for n in multi_env.env_names}

    peak_threshold  = exp.PEAK_ARRIVAL_THRESHOLD
    peak_locked     = False
    best_arrival    = -1.0
    best_model_dir  = os.path.join(exp_path, "best")
    os.makedirs(best_model_dir, exist_ok=True)

    metrics_csv = os.path.join(exp_path, "episode_metrics.csv")
    current_conflict_ratio = _get_conflict_ratio(conflict_schedule, 0)

    cotrain    = getattr(exp, 'COTRAIN_CYCLES', True)
    full_joint = getattr(exp, 'FULL_JOINT_TRAINING', False)
    (train_both, training_lm, training_agent,
     training_gm) = prepare_models_for_cycle(
        1, 1, master_model, agent_model,
        cotrain_cycles=cotrain, full_joint=full_joint,
    )

    for ep in range(1, total_episodes + 1):
        episode_counter = ep

        new_ratio = _get_conflict_ratio(conflict_schedule, ep)
        if new_ratio != current_conflict_ratio:
            current_conflict_ratio = new_ratio
            multi_env.update_conflict_ratio(new_ratio)
            print(f"  [Schedule] Ep {ep}: conflict_ratio -> {new_ratio:.2f}")

        multi_env.pick_env()
        chosen_env_name = multi_env.current_name

        _ep_lm = training_lm and not skip_master_training
        _ep_gm = training_gm and not skip_master_training
        _ep_tb = train_both  and not skip_master_training
        episode_rewards, actions, steps, crashed, arrival_rate, bootstrap = process_episode(
            episode_counter, total_steps, multi_env, master_model, agent_model,
            exp,
            train_both=_ep_tb,
            training_local_master=_ep_lm,
            training_agent=training_agent,
            training_global_master=_ep_gm,
            collect_bootstrap=True,
        )
        total_steps += steps

        if crashed:
            collision_counter += 1
        env_arrivals[chosen_env_name].append(arrival_rate)

        unwrapped = multi_env.env._get_unwrapped_env()
        scenario_idx = getattr(unwrapped, 'last_scenario_index', -1)

        results["arrival_rates"].append(arrival_rate)
        results["episode_rewards"].append(episode_rewards)
        results["collision_rates"].append(100.0 if crashed else 0.0)
        results["all_actions"].append(actions)

        if ep == 1:
            with open(metrics_csv, "w", encoding="utf-8") as f:
                f.write("episode,env,scenario_idx,reward,arrival_pct,collision\n")
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{chosen_env_name},{scenario_idx},{float(episode_rewards):.6f},"
                f"{float(arrival_rate):.6f},{1 if crashed else 0}\n"
            )

        if ep >= 50:
            recent_arr = [v for v in results["arrival_rates"][-50:] if v is not None]
            rolling50 = float(np.mean(recent_arr)) if recent_arr else 0.0
            if rolling50 > best_arrival:
                best_arrival = rolling50
                save_models(agent_model, master_model,
                            os.path.join(best_model_dir, "ckpt"))

        tag = "CRASH" if crashed else "ok"
        print(f"  Ep {ep}/{total_episodes}  {tag}  arrival={arrival_rate:.0f}%")

        # ── Bootstrap observations for GAE ────────────────────────────────
        # Use the actual post-step simulator state returned by run_episode.
        # Resetting here creates an unrelated state and corrupts value targets.
        lm1_input = bootstrap["last_lm1_obs"]
        lm2_input = bootstrap["last_lm2_obs"]
        gm_input = bootstrap["last_gm_obs"]
        last_agent_obs_list = bootstrap["last_agent_obs"]
        last_done = bool(bootstrap["last_done"])

        # ── PPO training ──────────────────────────────────────────────────
        if ep % exp.EPISODE_AMOUNT_FOR_TRAIN == 0:
            ent_start = getattr(exp, 'ENT_COEF', 0.01)
            ent_end   = getattr(exp, 'ENT_COEF_FINAL', ent_start)

            if peak_locked:
                current_ent_coef = 0.0
            else:
                if peak_threshold > 0.0:
                    recent = [v for v in results["arrival_rates"][-20:] if v is not None]
                    rolling_arr = float(np.mean(recent)) if recent else 0.0
                    if rolling_arr >= peak_threshold:
                        peak_locked = True
                        current_ent_coef = 0.0
                    else:
                        frac = min(1.0, ep / max(1, total_episodes))
                        current_ent_coef = ent_start + (ent_end - ent_start) * frac
                else:
                    frac = min(1.0, ep / max(1, total_episodes))
                    current_ent_coef = ent_start + (ent_end - ent_start) * frac

            _lm  = training_lm  and not skip_master_training
            _gm  = training_gm  and not skip_master_training
            _tb  = train_both   and not skip_master_training

            perform_training_phase(
                train_both=_tb,
                training_local_master=_lm,
                training_agent=training_agent,
                training_global_master=_gm,
                master_model=master_model,
                agent_model=agent_model,
                last_agent_obs_8d=last_agent_obs_list,
                last_lm_obs_25d=lm1_input,
                last_lm2_obs_25d=lm2_input,
                last_gm_obs_25d=gm_input,
                results=results,
                ent_coef=current_ent_coef,
                clip_range=getattr(exp, 'CLIP_RANGE', 0.2),
                vf_coef=getattr(exp, 'VF_COEF', 0.5),
                n_ppo_epochs=getattr(exp, 'N_PPO_EPOCHS', 1),
                n_value_epochs=getattr(exp, 'N_VALUE_EPOCHS', 0),
                last_done=last_done,
            )

    return collision_counter, results, best_model_dir, env_arrivals


def _analyze_scenario_difficulty(metrics_csv_path: str, exp_path: str) -> None:
    """Analyze per-scenario crash rates from episode_metrics.csv.

    Produces scenario_analysis.csv and a bar-chart PNG showing crash rates
    per (env, scenario_idx).  Scenarios that always crash in the last quarter
    of training are flagged as "unsolvable".
    """
    import pandas as pd

    if not os.path.exists(metrics_csv_path):
        return
    df = pd.read_csv(metrics_csv_path)
    if "scenario_idx" not in df.columns:
        return

    total_eps = len(df)
    last_quarter = df[df["episode"] > total_eps * 0.75].copy()
    if last_quarter.empty:
        return

    stats = (
        last_quarter.groupby(["env", "scenario_idx"])
        .agg(n_episodes=("collision", "count"),
             n_crashes=("collision", "sum"),
             avg_arrival=("arrival_pct", "mean"))
        .reset_index()
    )
    stats["crash_rate_pct"] = (stats["n_crashes"] / stats["n_episodes"] * 100).round(1)
    stats = stats.sort_values("crash_rate_pct", ascending=False)

    csv_out = os.path.join(exp_path, "scenario_analysis.csv")
    stats.to_csv(csv_out, index=False)

    unsolvable = stats[(stats["crash_rate_pct"] >= 80) & (stats["n_episodes"] >= 3)]
    if not unsolvable.empty:
        print(f"  [Scenario Analysis] {len(unsolvable)} potentially unsolvable scenarios:")
        for _, row in unsolvable.iterrows():
            print(f"    {row['env']} #{int(row['scenario_idx'])}: "
                  f"crash={row['crash_rate_pct']:.0f}% "
                  f"({int(row['n_crashes'])}/{int(row['n_episodes'])} eps)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, env_name in zip(axes, ["intersection", "roundabout", "double_intersection"]):
            env_stats = stats[stats["env"] == env_name].sort_values("scenario_idx")
            if env_stats.empty:
                ax.set_title(env_name)
                continue
            colors = ["red" if cr >= 80 else "orange" if cr >= 50 else "green"
                      for cr in env_stats["crash_rate_pct"]]
            ax.bar(env_stats["scenario_idx"].astype(str), env_stats["crash_rate_pct"],
                   color=colors)
            ax.set_title(f"{env_name} (last 25% of training)")
            ax.set_xlabel("Scenario Index")
            ax.tick_params(axis='x', rotation=90, labelsize=6)
        axes[0].set_ylabel("Crash Rate %")
        fig.suptitle("Per-Scenario Crash Rates (last quarter of training)")
        fig.tight_layout()
        fig.savefig(os.path.join(exp_path, "scenario_difficulty.png"), dpi=120)
        plt.close(fig)
    except Exception:
        pass


# ── Test runner (held-out or conflict-only) ───────────────────────────────────

def _run_test(
    label: str,
    env_id: str,
    env_label: str,
    exp_base: Experiment,
    agent_model,
    master_model,
    total_episodes_cfg: int,
    use_held_out: bool = False,
    use_conflict_only: bool = False,
    n_episodes: int = N_TEST_EPISODES,
) -> dict:
    print(f"  [{label}] {env_label} ({n_episodes} eps) ...")

    project_globals.after_is_arrived_flags.clear()
    for _ in range(6):
        project_globals.after_is_arrived_flags.append(False)

    exp_test = _make_experiment(exp_base.EXPERIMENT_PATH, total_episodes_cfg, env_id=env_id)
    env_config = _make_env_config(
        env_id,
        conflict_ratio=0.0,
        use_held_out=use_held_out,
        use_conflict_only=use_conflict_only,
    )
    exp_test.WARMUP_EPISODES = 0
    exp_test.CONFIG = env_config

    def _env_fn():
        d = Driver(exp_test)
        if use_held_out:
            d.highway_env.unwrapped.config["use_held_out_scenarios"] = True
        if use_conflict_only:
            d.highway_env.unwrapped.config["use_conflict_scenarios_only"] = True
        return d

    test_wrapped = DummyVecEnv([_env_fn])
    arrivals, crashes = [], []
    for ep in range(1, n_episodes + 1):
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

    result = {
        "env_id":           env_id,
        "n_episodes":       n_episodes,
        "arrival_rate_avg": float(np.mean(arrivals)),
        "crash_rate_pct":   100.0 * sum(crashes) / max(1, len(crashes)),
    }
    print(f"    arrival={result['arrival_rate_avg']:.1f}%  crash={result['crash_rate_pct']:.1f}%")
    return result


# ── Single condition (train + test) ───────────────────────────────────────────

def _run_one_condition(
    config: RunConfig,
    condition_label: str,
    ablation: bool,
    base_exp_path: str,
    seed: int = 42,
) -> dict:
    _set_all_seeds(seed)
    MasterModel.NORMALIZE_INPUTS = NORMALIZE_MASTER_INPUTS
    Driver.NORMALIZE_AGENT_OBS = NORMALIZE_AGENT_OBS

    sub_dir = os.path.join(base_exp_path, config.name, condition_label, f"s{seed}")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(os.path.join(sub_dir, "agent_logs"),  exist_ok=True)
    os.makedirs(os.path.join(sub_dir, "master_logs"), exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  CONFIG: {config.name}  |  COND: {condition_label}  |  SEED: {seed}")
    if ablation:
        print(f"  Master = ZEROS, master training = SKIPPED")
    else:
        print(f"  Normal hierarchical training")
    train_env_desc = config.train_env_only or "ALL 3 environments"
    print(f"  Training env: {train_env_desc}")
    print(f"  Episodes: {config.total_episodes}")
    print(f"  Conflict schedule: {config.conflict_schedule}")
    print(f"  Output: {sub_dir}")
    print(f"{'#'*60}\n")

    # ── Ablation monkey-patch ─────────────────────────────────────────────────
    _original_gpa = MasterModel.get_proto_action
    if ablation:
        def _zero_proto_action(self, master_input):
            emb = np.zeros(self.embedding_size, dtype=np.float32)
            val = torch.tensor([0.0])
            lp  = torch.tensor([0.0])
            return emb, val, lp
        MasterModel.get_proto_action = _zero_proto_action

    # ── Build experiment + models ─────────────────────────────────────────────
    initial_ratio = _get_conflict_ratio(config.conflict_schedule, 0)
    exp = _make_experiment(sub_dir, config.total_episodes, env_id="RELintersection-v0")
    setup_experiment_dirs(sub_dir)
    inter_cfg = _make_env_config("RELintersection-v0", conflict_ratio=initial_ratio)
    master_model_inst, agent_model_inst, _ = initialize_models(exp, inter_cfg)

    ckpt = PRETRAINED_CHECKPOINT + "_agent.pth"
    if LOAD_PRETRAINED_CHECKPOINT and os.path.exists(ckpt):
        loaded = load_models(agent_model_inst, master_model_inst, PRETRAINED_CHECKPOINT)
        print(f"  Checkpoint: {'loaded' if loaded else 'FAILED -- random init'}")
    elif not LOAD_PRETRAINED_CHECKPOINT:
        print("  Checkpoint: skipped (normalization changed observation scale)")
    else:
        print(f"  WARNING: no checkpoint at {ckpt}, using random init")

    agent_logger, master_logger = _setup_loggers_csv(sub_dir)
    agent_model_inst.set_logger(agent_logger)
    master_model_inst.set_logger(master_logger)

    # ── Environments ──────────────────────────────────────────────────────────
    project_globals.after_is_arrived_flags = [False] * 6

    if config.train_env_only:
        # Single-env training (configs D/E/F)
        train_env_defs = [(eid, lbl) for eid, lbl in ENV_DEFS if eid == config.train_env_only]
        train_weights = [1.0]
    else:
        train_env_defs = list(ENV_DEFS)
        train_weights = list(ENV_WEIGHTS)

    wrapped_envs = []
    for env_id, label in train_env_defs:
        env_cfg = _make_env_config(env_id, conflict_ratio=initial_ratio)
        exp_copy = _make_experiment(sub_dir, config.total_episodes, env_id=env_id)
        exp_copy.CONFIG = env_cfg
        wrapped = DummyVecEnv([lambda ec=exp_copy: Driver(ec)])
        wrapped_envs.append(wrapped)

    multi_env = MultiEnvWrapper(
        wrapped_envs, [label for _, label in train_env_defs], weights=train_weights,
    )
    exp.CONFIG = inter_cfg

    # ── Training ──────────────────────────────────────────────────────────────
    collision_counter, results, best_model_dir, env_arrivals = unified_training_loop(
        exp, multi_env, agent_model_inst, master_model_inst, sub_dir,
        total_episodes=config.total_episodes,
        conflict_schedule=config.conflict_schedule,
        skip_master_training=ablation,
    )
    save_models(agent_model_inst, master_model_inst, exp.SAVE_MODEL_DIRECTORY)

    # ── Per-scenario difficulty analysis ──────────────────────────────────────
    metrics_csv = os.path.join(sub_dir, "episode_metrics.csv")
    _analyze_scenario_difficulty(metrics_csv, sub_dir)

    # ── Per-condition training plots ──────────────────────────────────────────
    save_plots(
        results, sub_dir, metrics_csv,
        suptitle=f"{config.name} -- {condition_label} -- seed={seed} -- {config.total_episodes} ep",
        total_episodes=config.total_episodes,
        rolling_crash_window=ROLLING_CRASH_WINDOW,
    )

    # ── Load best model for testing ───────────────────────────────────────────
    best_ckpt = os.path.join(best_model_dir, "ckpt")
    if os.path.exists(best_ckpt + "_agent.pth"):
        load_models(agent_model_inst, master_model_inst, best_ckpt)

    # ── Held-out regular test (ALL envs, even if trained on one) ──────────────
    print(f"\n{'='*60}")
    print(f"  HELD-OUT REGULAR TEST -- {config.name} / {condition_label}")
    print(f"{'='*60}")
    held_out_results = {}
    for env_id, label in ENV_DEFS:
        r = _run_test("Held-out", env_id, label, exp, agent_model_inst,
                       master_model_inst, config.total_episodes, use_held_out=True)
        held_out_results[label] = r

    # ── Conflict-only test (ALL envs) ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONFLICT-ONLY TEST -- {config.name} / {condition_label}")
    print(f"{'='*60}")
    conflict_results = {}
    for env_id, label in ENV_DEFS:
        r = _run_test("Conflict", env_id, label, exp, agent_model_inst,
                       master_model_inst, config.total_episodes, use_conflict_only=True)
        conflict_results[label] = r

    multi_env.close()
    MasterModel.get_proto_action = _original_gpa

    arr_list = [v for v in results["arrival_rates"] if v is not None]
    return {
        "config_name":        config.name,
        "condition":          condition_label,
        "seed":               seed,
        "best_model_dir":     best_model_dir,
        "results":            results,
        "env_arrivals":       env_arrivals,
        "held_out":           held_out_results,
        "conflict_test":      conflict_results,
        "total_collisions":   collision_counter,
        "arrival_avg":        round(float(np.mean(arr_list)), 2) if arr_list else 0,
        "arrival_last50":     round(float(np.mean(arr_list[-50:])), 2) if arr_list else 0,
        "per_env_last50": {
            name: round(float(np.mean(arr[-50:])), 2) if len(arr) >= 50
                  else round(float(np.mean(arr)), 2) if arr else 0
            for name, arr in env_arrivals.items()
        },
    }


# ── Scalability test (Config G) ──────────────────────────────────────────────

def _run_scalability_test(
    best_model_path: str,
    exp_path: str,
) -> dict:
    """
    Load the best model from Config A WITH_MASTER and test with different
    numbers of active agents (2, 4, 6) by masking extra agents as 'arrived'.
    """
    import torch
    from src.model.master_model import MasterModel

    scale_dir = os.path.join(exp_path, "scalability_test")
    os.makedirs(scale_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  SCALABILITY TEST (Config G)")
    print(f"  Model: {best_model_path}")
    print(f"  Agent counts: 2, 4, 6")
    print(f"{'#'*60}\n")

    # Load model
    exp = _make_experiment(scale_dir, 100, env_id="RELintersection-v0")
    setup_experiment_dirs(scale_dir)
    inter_cfg = _make_env_config("RELintersection-v0")
    master_model, agent_model, _ = initialize_models(exp, inter_cfg)

    if os.path.exists(best_model_path + "_agent.pth"):
        load_models(agent_model, master_model, best_model_path)
        print(f"  Loaded model from {best_model_path}")
    else:
        print(f"  WARNING: model not found at {best_model_path}, using random init")

    # Agent masks: which agents are active for each count
    # 2 agents: 0,3 (1 per LM); 4 agents: 0,1,3,4 (2 per LM); 6: all
    agent_masks = {
        2: [0, 3],
        4: [0, 1, 3, 4],
        6: [0, 1, 2, 3, 4, 5],
    }

    all_scale_results = {}

    for n_active, active_indices in agent_masks.items():
        masked_indices = [i for i in range(6) if i not in active_indices]
        label = f"G_{n_active}agents"
        print(f"\n  --- {label}: active={active_indices}, masked={masked_indices} ---")

        scale_results = {"held_out": {}, "conflict_test": {}}

        for test_type, use_ho, use_co in [
            ("held_out", True, False),
            ("conflict_test", False, True),
        ]:
            for env_id, env_label in ENV_DEFS:
                project_globals.after_is_arrived_flags = [False] * 6
                exp_test = _make_experiment(scale_dir, 100, env_id=env_id)
                env_config = _make_env_config(env_id, use_held_out=use_ho, use_conflict_only=use_co)
                exp_test.WARMUP_EPISODES = 0
                exp_test.CONFIG = env_config

                def _env_fn(ec=exp_test, _eid=env_id, _uho=use_ho, _uco=use_co):
                    d = Driver(ec)
                    if _uho:
                        d.highway_env.unwrapped.config["use_held_out_scenarios"] = True
                    if _uco:
                        d.highway_env.unwrapped.config["use_conflict_scenarios_only"] = True
                    return d

                test_wrapped = DummyVecEnv([_env_fn])
                arrivals, crashes = [], []

                for ep in range(1, N_TEST_EPISODES + 1):
                    # Mask agents by setting them as arrived before episode
                    project_globals.after_is_arrived_flags = [False] * 6
                    for mi in masked_indices:
                        project_globals.after_is_arrived_flags[mi] = True

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
                    "arrival_rate_avg": float(np.mean(arrivals)),
                    "crash_rate_pct": 100.0 * sum(crashes) / max(1, len(crashes)),
                }
                scale_results[test_type][env_label] = r
                print(f"    [{test_type}] {env_label}: arrival={r['arrival_rate_avg']:.1f}%  crash={r['crash_rate_pct']:.1f}%")

        all_scale_results[n_active] = scale_results

    # ── Scalability bar charts ────────────────────────────────────────────────
    _plot_scalability(all_scale_results, scale_dir)

    return all_scale_results


def _plot_scalability(scale_results: dict, out_dir: str):
    env_labels = ["intersection", "roundabout", "double_intersection"]
    agent_counts = sorted(scale_results.keys())

    for test_type, test_title in [
        ("held_out", "Held-Out Scenarios"),
        ("conflict_test", "Conflict-Only Scenarios"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(env_labels))
        w = 0.25

        for metric_idx, (metric_key, metric_title, ax) in enumerate([
            ("arrival_rate_avg", f"{test_title}: Arrival Rate (%)", axes[0]),
            ("crash_rate_pct", f"{test_title}: Crash Rate (%)", axes[1]),
        ]):
            colors = ["#4CAF50", "#2196F3", "#FF9800"]
            for i, n_agents in enumerate(agent_counts):
                vals = [scale_results[n_agents][test_type].get(el, {}).get(metric_key, 0)
                        for el in env_labels]
                bars = ax.bar(x + i * w, vals, w, label=f"{n_agents} agents", color=colors[i])
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                                f"{h:.0f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(metric_title, fontsize=12)
            ax.set_xticks(x + w)
            ax.set_xticklabels(env_labels, rotation=15)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"Scalability Test -- {test_title}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scalability_{test_type}.png"), dpi=150)
        plt.close(fig)


# ── Embedding PCA visualization ──────────────────────────────────────────────

def _collect_and_plot_embeddings(
    best_model_path: str,
    exp_path: str,
) -> None:
    """
    Run inference episodes while intercepting master embeddings.
    Produce PCA scatter plots, cosine similarity analysis, and histograms.
    """
    from sklearn.decomposition import PCA
    from src.model.master_model import MasterModel
    from src.training.episode_utils import (
        _build_local_master_input,
        _build_global_master_input,
    )

    emb_dir = os.path.join(exp_path, "embeddings_pca")
    os.makedirs(emb_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  MASTER EMBEDDING ANALYSIS")
    print(f"  Model: {best_model_path}")
    print(f"{'#'*60}\n")

    # Load model
    exp = _make_experiment(emb_dir, 100, env_id="RELintersection-v0")
    setup_experiment_dirs(emb_dir)
    inter_cfg = _make_env_config("RELintersection-v0")
    master_model, agent_model, _ = initialize_models(exp, inter_cfg)

    if os.path.exists(best_model_path + "_agent.pth"):
        load_models(agent_model, master_model, best_model_path)

    # Collector storage
    collected = defaultdict(list)  # role -> list of 4D embeddings
    collected_env = []             # env label per call
    collected_role = []            # "LM1", "LM2", "GM" per call
    collected_inputs = []          # 25D inputs
    episode_outcomes = []          # (env_label, crashed, arrival_rate) per episode

    _call_counter = [0]  # mutable for closure
    _current_env = [""]
    _original_gpa = MasterModel.get_proto_action

    def _intercepting_gpa(self, master_input):
        emb, val, lp = _original_gpa(self, master_input)
        role_idx = _call_counter[0] % 3
        role = ["LM1", "LM2", "GM"][role_idx]
        collected[role].append(emb.copy())
        collected_env.append(_current_env[0])
        collected_role.append(role)
        collected_inputs.append(np.asarray(master_input, dtype=np.float32).flatten().copy())
        _call_counter[0] += 1
        return emb, val, lp

    MasterModel.get_proto_action = _intercepting_gpa

    # Run test episodes across all envs
    n_eps_per_env = 50
    for env_id, env_label in ENV_DEFS:
        _current_env[0] = env_label
        project_globals.after_is_arrived_flags = [False] * 6
        exp_test = _make_experiment(emb_dir, 100, env_id=env_id)
        env_config = _make_env_config(env_id, use_held_out=True)
        exp_test.WARMUP_EPISODES = 0
        exp_test.CONFIG = env_config

        def _env_fn(ec=exp_test):
            d = Driver(ec)
            d.highway_env.unwrapped.config["use_held_out_scenarios"] = True
            return d

        test_wrapped = DummyVecEnv([_env_fn])
        for ep in range(1, n_eps_per_env + 1):
            _, _, _, crashed, arrival_rate = process_episode(
                ep, 0, test_wrapped, master_model, agent_model,
                exp_test,
                train_both=False,
                training_local_master=False,
                training_agent=False,
                training_global_master=False,
            )
            episode_outcomes.append((env_label, crashed, arrival_rate))

        try:
            test_wrapped.close()
        except Exception:
            pass

    MasterModel.get_proto_action = _original_gpa

    # ── Process collected data ────────────────────────────────────────────────
    all_embs = np.array(collected_env)  # just labels
    lm1_embs = np.array(collected["LM1"]) if collected["LM1"] else np.zeros((0, 4))
    lm2_embs = np.array(collected["LM2"]) if collected["LM2"] else np.zeros((0, 4))
    gm_embs  = np.array(collected["GM"])  if collected["GM"]  else np.zeros((0, 4))

    # Assign env labels to each role call
    # Every episode step produces 3 calls: LM1, LM2, GM. So env labels repeat in groups of 3.
    lm1_envs = [collected_env[i] for i in range(len(collected_env)) if collected_role[i] == "LM1"]
    lm2_envs = [collected_env[i] for i in range(len(collected_env)) if collected_role[i] == "LM2"]
    gm_envs  = [collected_env[i] for i in range(len(collected_env)) if collected_role[i] == "GM"]

    if len(lm1_embs) < 2 or len(lm2_embs) < 2:
        print("  Not enough embeddings collected for PCA")
        return

    # ── 1. PCA scatter: LM1 + LM2 colored by env ─────────────────────────────
    all_lm = np.vstack([lm1_embs, lm2_embs])
    all_lm_labels = (["LM1"] * len(lm1_embs)) + (["LM2"] * len(lm2_embs))
    all_lm_env_labels = lm1_envs + lm2_envs

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(all_lm)

    env_colors = {"intersection": "#2196F3", "roundabout": "#4CAF50", "double_intersection": "#FF9800"}
    role_markers = {"LM1": "o", "LM2": "^"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for env_l in ["intersection", "roundabout", "double_intersection"]:
        for role in ["LM1", "LM2"]:
            mask = [(e == env_l and r == role)
                    for e, r in zip(all_lm_env_labels, all_lm_labels)]
            if any(mask):
                pts = pca_2d[mask]
                ax.scatter(pts[:, 0], pts[:, 1], c=env_colors.get(env_l, "gray"),
                           marker=role_markers.get(role, "o"), alpha=0.4, s=15,
                           label=f"{env_l} {role}")
    ax.set_title(f"Master Embeddings PCA (explained var: {pca.explained_variance_ratio_.sum():.2f})",
                 fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(emb_dir, "pca_by_env_and_role.png"), dpi=150)
    plt.close(fig)

    # ── 2. PCA scatter: GM embeddings colored by env ──────────────────────────
    if len(gm_embs) >= 2:
        pca_gm = PCA(n_components=2)
        gm_2d = pca_gm.fit_transform(gm_embs)

        fig, ax = plt.subplots(figsize=(10, 8))
        for env_l in ["intersection", "roundabout", "double_intersection"]:
            mask = [e == env_l for e in gm_envs]
            if any(mask):
                pts = gm_2d[mask]
                ax.scatter(pts[:, 0], pts[:, 1], c=env_colors.get(env_l, "gray"),
                           alpha=0.4, s=15, label=env_l)
        ax.set_title(f"Global Master Embeddings PCA (explained var: {pca_gm.explained_variance_ratio_.sum():.2f})",
                     fontsize=13)
        ax.set_xlabel(f"PC1 ({pca_gm.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca_gm.explained_variance_ratio_[1]:.2%})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(emb_dir, "pca_gm_by_env.png"), dpi=150)
        plt.close(fig)

    # ── 3. Embedding dimension histograms ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for dim_idx, ax in enumerate(axes.flat):
        for role, embs, color in [
            ("LM1", lm1_embs, "#2196F3"),
            ("LM2", lm2_embs, "#FF5722"),
            ("GM",  gm_embs,  "#4CAF50"),
        ]:
            if len(embs) > 0:
                ax.hist(embs[:, dim_idx], bins=50, alpha=0.5, label=role, color=color, density=True)
        ax.set_title(f"Embedding Dimension {dim_idx}", fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Master Embedding Distributions by Dimension", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(emb_dir, "embedding_histograms.png"), dpi=150)
    plt.close(fig)

    # ── 4. LM1 vs LM2 cosine similarity per step ─────────────────────────────
    n_pairs = min(len(lm1_embs), len(lm2_embs))
    if n_pairs > 0:
        cos_sims = []
        for i in range(n_pairs):
            a, b = lm1_embs[i], lm2_embs[i]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a > 1e-8 and norm_b > 1e-8:
                cos_sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
            else:
                cos_sims.append(0.0)

        cos_envs = lm1_envs[:n_pairs]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Overall distribution
        ax = axes[0]
        ax.hist(cos_sims, bins=50, color="#673AB7", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(cos_sims), color="red", linestyle="--", label=f"mean={np.mean(cos_sims):.3f}")
        ax.set_title("LM1 vs LM2 Cosine Similarity Distribution", fontsize=12)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Per-env boxplot
        ax = axes[1]
        env_cos = defaultdict(list)
        for cs, el in zip(cos_sims, cos_envs):
            env_cos[el].append(cs)
        box_data = [env_cos.get(el, []) for el in ["intersection", "roundabout", "double_intersection"]]
        bp = ax.boxplot(box_data, labels=["intersection", "roundabout", "double_int"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#2196F3", "#4CAF50", "#FF9800"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title("LM1 vs LM2 Cosine Similarity by Environment", fontsize=12)
        ax.set_ylabel("Cosine Similarity")
        ax.grid(True, alpha=0.3)

        plt.suptitle("Local Master Embedding Differentiation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(emb_dir, "cosine_similarity.png"), dpi=150)
        plt.close(fig)

    # Save raw data
    np.savez_compressed(
        os.path.join(emb_dir, "embeddings_raw.npz"),
        lm1=lm1_embs, lm2=lm2_embs, gm=gm_embs,
        lm1_envs=np.array(lm1_envs), lm2_envs=np.array(lm2_envs), gm_envs=np.array(gm_envs),
    )
    print(f"  Saved embeddings to {emb_dir}")


# ── Scenario layout visualization ─────────────────────────────────────────────

def _plot_scenario_layouts(exp_path: str) -> None:
    """Draw road geometry + vehicle positions for representative scenarios."""
    import gymnasium as gym
    from src.experiment.scenarios import (
        base_complete_scenarios_6_cars,
        conflict_base_scenarios,
        roundabout_base_scenarios,
        roundabout_conflict_base_scenarios,
        double_intersection_base_scenarios,
        double_intersection_conflict_base_scenarios,
    )

    layout_dir = os.path.join(exp_path, "scenario_layouts")
    os.makedirs(layout_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  SCENARIO LAYOUT VISUALIZATION")
    print(f"{'#'*60}\n")

    BASE_LONG = 40

    env_scenario_sets = [
        ("intersection", "RELintersection-v0",
         base_complete_scenarios_6_cars[:5], conflict_base_scenarios[:3]),
        ("roundabout", "RELroundabout-v0",
         roundabout_base_scenarios[:5], roundabout_conflict_base_scenarios[:3]),
        ("double_intersection", "RELdouble-intersection-v0",
         double_intersection_base_scenarios[:5], double_intersection_conflict_base_scenarios[:3]),
    ]

    agent_colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]

    for env_label, env_id, regular_scenarios, conflict_scenarios in env_scenario_sets:
        all_scens = [(s, "regular") for s in regular_scenarios] + [(s, "conflict") for s in conflict_scenarios]
        n_scens = len(all_scens)
        if n_scens == 0:
            continue

        cols = min(4, n_scens)
        rows = (n_scens + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

        # Create a temporary env to get road geometry
        project_globals.after_is_arrived_flags = [False] * 6
        exp_tmp = _make_experiment(layout_dir, 10, env_id=env_id)
        env_cfg = _make_env_config(env_id)
        exp_tmp.CONFIG = env_cfg
        try:
            d = Driver(exp_tmp)
            d.highway_env.reset()
            unwrapped = d.highway_env
            while hasattr(unwrapped, 'env') and not hasattr(unwrapped, 'road'):
                unwrapped = unwrapped.env
            road_net = unwrapped.road.network
        except Exception as e:
            print(f"  Could not create env for {env_label}: {e}")
            plt.close(fig)
            continue

        for scen_idx, (scenario, scen_type) in enumerate(all_scens):
            r, c_idx = divmod(scen_idx, cols)
            ax = axes[r][c_idx]

            # Draw road network lanes
            drawn_lanes = set()
            for from_node in road_net.graph:
                for to_node in road_net.graph[from_node]:
                    for lane_idx, lane in enumerate(road_net.graph[from_node][to_node]):
                        lane_key = (from_node, to_node, lane_idx)
                        if lane_key in drawn_lanes:
                            continue
                        drawn_lanes.add(lane_key)
                        pts = []
                        n_samples = 20
                        for si in range(n_samples + 1):
                            s = lane.length * si / n_samples
                            pos = lane.position(s, 0)
                            pts.append(pos)
                        pts = np.array(pts)
                        ax.plot(pts[:, 0], pts[:, 1], color="#BDBDBD", linewidth=1.5, zorder=1)

            # Draw vehicle start positions
            for agent_idx, (lane_key, destination, offset) in enumerate(scenario["agents"]):
                try:
                    lane = road_net.get_lane(lane_key)
                    pos = lane.position(BASE_LONG + offset, 0)
                    heading = lane.heading_at(pos)
                    color = agent_colors[agent_idx % len(agent_colors)]

                    ax.scatter(pos[0], pos[1], c=color, s=80, zorder=3, edgecolors="black", linewidth=0.5)
                    dx = 3 * np.cos(heading)
                    dy = 3 * np.sin(heading)
                    ax.annotate("", xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                                arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=4)
                    ax.annotate(f"A{agent_idx}->{destination}", (pos[0] + 1, pos[1] + 1),
                                fontsize=6, color=color, zorder=5)
                except Exception:
                    pass

            title_prefix = "CONFLICT" if scen_type == "conflict" else f"Regular"
            ax.set_title(f"{title_prefix} #{scen_idx + 1}", fontsize=10,
                         color="red" if scen_type == "conflict" else "black")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for idx in range(n_scens, rows * cols):
            r, c_idx = divmod(idx, cols)
            axes[r][c_idx].set_visible(False)

        plt.suptitle(f"Scenario Layouts -- {env_label}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(layout_dir, f"scenarios_{env_label}.png"), dpi=150)
        plt.close(fig)

        try:
            d.highway_env.close()
        except Exception:
            pass

    print(f"  Saved scenario layouts to {layout_dir}")


# ── Multi-seed comparison plots ───────────────────────────────────────────────

def _smooth_arr(arr: list, window: int) -> np.ndarray:
    """Return rolling-mean as numpy array, handling Nones."""
    return np.array(_rolling_mean(arr, window), dtype=np.float64)


def _compute_seed_bands(seed_data_list: list, key: str, window: int):
    """Given list of per-seed data dicts, compute mean ± std of a rolling curve."""
    curves = []
    for d in seed_data_list:
        raw = d["results"].get(key, [])
        if raw:
            curves.append(_smooth_arr(raw, window))
    if not curves:
        return None, None, None
    min_len = min(len(c) for c in curves)
    stacked = np.array([c[:min_len] for c in curves])
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)
    return mean, std, min_len


def _plot_multi_seed_comparisons(all_seeds_data: dict, exp_path: str):
    """
    all_seeds_data: { config_name: { cond_label: [data_seed0, data_seed1, ...] } }
    """
    config_names = list(all_seeds_data.keys())
    env_labels = ["intersection", "roundabout", "double_intersection"]
    n_configs = len(config_names)
    if n_configs == 0:
        return

    # ── 1. Arrival curves: mean ± std  ────────────────────────────────────────
    fig, axes = plt.subplots(1, n_configs, figsize=(7 * min(n_configs, 4), 5),
                             squeeze=False)
    for i, cfg_name in enumerate(config_names):
        ax = axes[0][i]
        for cond, color, ls in [("W_MASTER", "#2196F3", "-"),
                                ("NO_MASTER", "#FF5722", "--")]:
            seed_list = all_seeds_data[cfg_name].get(cond, [])
            mean, std, n = _compute_seed_bands(seed_list, "arrival_rates", SMOOTH_EP)
            if mean is None:
                continue
            x = np.arange(n)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=2, label=cond)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(cfg_name, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Arrival %")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Training Arrival (mean ± std, {N_SEEDS} seeds)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, "arrival_mean_std.png"), dpi=150)
    plt.close(fig)

    # ── 2. Loss curves: mean ± std ────────────────────────────────────────────
    for loss_key, loss_title in [("agent_total_losses", "Agent Loss"),
                                 ("master_total_losses", "Master Loss")]:
        fig, axes = plt.subplots(1, n_configs,
                                 figsize=(7 * min(n_configs, 4), 5), squeeze=False)
        for i, cfg_name in enumerate(config_names):
            ax = axes[0][i]
            for cond, color, ls in [("W_MASTER", "#2196F3", "-"),
                                    ("NO_MASTER", "#FF5722", "--")]:
                seed_list = all_seeds_data[cfg_name].get(cond, [])
                mean, std, n = _compute_seed_bands(seed_list, loss_key, 20)
                if mean is None:
                    continue
                x = np.arange(n)
                ax.plot(x, mean, color=color, linestyle=ls, linewidth=2, label=cond)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_title(f"{cfg_name} -- {loss_title}", fontsize=10)
            ax.set_xlabel("Training step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"{loss_title} (mean ± std, {N_SEEDS} seeds)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        safe_name = loss_key.replace("_losses", "")
        plt.savefig(os.path.join(exp_path, f"{safe_name}_loss_mean_std.png"), dpi=150)
        plt.close(fig)

    # ── 3. Test bars with error bars (mean ± std across seeds) ────────────────
    for test_key, test_title, fname in [
        ("held_out",     "Held-Out Regular Test", "held_out_bars.png"),
        ("conflict_test", "Conflict-Only Test",   "conflict_test_bars.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        x = np.arange(len(env_labels))
        bar_width = max(0.06, 0.8 / (n_configs * 2))
        for metric_idx, (mk, mt, ax) in enumerate([
            ("arrival_rate_avg", f"{test_title}: Arrival %", axes[0]),
            ("crash_rate_pct",   f"{test_title}: Crash %",   axes[1]),
        ]):
            bar_offset = 0
            for cfg_name in config_names:
                for cond, hatch, alpha in [("W_MASTER", None, 1.0),
                                           ("NO_MASTER", "//", 0.7)]:
                    seed_list = all_seeds_data[cfg_name].get(cond, [])
                    means, stds = [], []
                    for env_l in env_labels:
                        vals = [d[test_key].get(env_l, {}).get(mk, 0)
                                for d in seed_list if test_key in d]
                        means.append(float(np.mean(vals)) if vals else 0)
                        stds.append(float(np.std(vals)) if vals else 0)
                    label = f"{cfg_name[:8]} {'W' if cond == 'W_MASTER' else 'WO'}"
                    color = "#2196F3" if cond == "W_MASTER" else "#FF5722"
                    ax.bar(x + bar_offset * bar_width, means, bar_width,
                           yerr=stds, capsize=3,
                           label=label, color=color, alpha=alpha, hatch=hatch,
                           edgecolor="black", linewidth=0.5)
                    bar_offset += 1
            ax.set_title(mt, fontsize=11)
            ax.set_xticks(x + bar_width * (n_configs * 2 - 1) / 2)
            ax.set_xticklabels(env_labels, rotation=15)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=6, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle(f"{test_title} (mean ± std, {N_SEEDS} seeds)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_path, fname), dpi=150)
        plt.close(fig)

    # ── 4. Delta table (mean across seeds) ────────────────────────────────────
    rows = []
    headers = ["Config", "Env", "HeldOut ΔArr", "Conflict ΔArr",
               "HeldOut ΔCrash", "Conflict ΔCrash"]
    for cfg_name in config_names:
        w_list  = all_seeds_data[cfg_name].get("W_MASTER", [])
        wo_list = all_seeds_data[cfg_name].get("NO_MASTER", [])
        if not w_list or not wo_list:
            continue
        for env_l in env_labels:
            w_ho  = np.mean([d["held_out"].get(env_l, {}).get("arrival_rate_avg", 0)
                             for d in w_list])
            wo_ho = np.mean([d["held_out"].get(env_l, {}).get("arrival_rate_avg", 0)
                             for d in wo_list])
            w_ct  = np.mean([d["conflict_test"].get(env_l, {}).get("arrival_rate_avg", 0)
                             for d in w_list])
            wo_ct = np.mean([d["conflict_test"].get(env_l, {}).get("arrival_rate_avg", 0)
                             for d in wo_list])
            w_hoc = np.mean([d["held_out"].get(env_l, {}).get("crash_rate_pct", 0)
                             for d in w_list])
            wo_hoc = np.mean([d["held_out"].get(env_l, {}).get("crash_rate_pct", 0)
                              for d in wo_list])
            w_ctc = np.mean([d["conflict_test"].get(env_l, {}).get("crash_rate_pct", 0)
                             for d in w_list])
            wo_ctc = np.mean([d["conflict_test"].get(env_l, {}).get("crash_rate_pct", 0)
                              for d in wo_list])
            rows.append([
                cfg_name, env_l,
                f"{w_ho - wo_ho:+.1f}%", f"{w_ct - wo_ct:+.1f}%",
                f"{w_hoc - wo_hoc:+.1f}%", f"{w_ctc - wo_ctc:+.1f}%",
            ])
    if rows:
        fig, ax = plt.subplots(figsize=(16, 1 + 0.5 * len(rows)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f2f2f2")
            if r > 0 and c in (2, 3):
                text = cell.get_text().get_text()
                if text.startswith("+") and text != "+0.0%":
                    cell.set_facecolor("#C8E6C9")
                elif text.startswith("-") and text != "-0.0%":
                    cell.set_facecolor("#FFCDD2")
            if r > 0 and c in (4, 5):
                text = cell.get_text().get_text()
                if text.startswith("-") and text != "-0.0%":
                    cell.set_facecolor("#C8E6C9")
                elif text.startswith("+") and text != "+0.0%":
                    cell.set_facecolor("#FFCDD2")
        plt.suptitle(f"Master Delta Table (mean of {N_SEEDS} seeds)",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_path, "delta_table.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def _plot_generalization_heatmap(all_seeds_data: dict, exp_path: str):
    gen_configs = ["D_inter", "E_round", "F_dbl"]
    env_labels = ["intersection", "roundabout", "double_intersection"]
    available = [c for c in gen_configs if c in all_seeds_data]
    if not available:
        return

    for test_type, test_title in [("held_out", "Held-Out Regular"),
                                  ("conflict_test", "Conflict-Only")]:
        for cond in ["W_MASTER", "NO_MASTER"]:
            matrix = np.zeros((len(available), len(env_labels)))
            std_matrix = np.zeros_like(matrix)
            row_labels = []
            for ri, cfg_name in enumerate(available):
                seed_list = all_seeds_data[cfg_name].get(cond, [])
                row_labels.append(cfg_name.replace("_", " "))
                for ci, env_l in enumerate(env_labels):
                    vals = [d[test_type].get(env_l, {}).get("arrival_rate_avg", 0)
                            for d in seed_list if test_type in d]
                    matrix[ri, ci] = float(np.mean(vals)) if vals else 0
                    std_matrix[ri, ci] = float(np.std(vals)) if vals else 0

            fig, ax = plt.subplots(figsize=(10, 4 + 0.5 * len(available)))
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(len(env_labels)))
            ax.set_xticklabels(env_labels, rotation=15)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            for ri in range(matrix.shape[0]):
                for ci in range(matrix.shape[1]):
                    val = matrix[ri, ci]
                    sd  = std_matrix[ri, ci]
                    tc  = "white" if val < 40 or val > 80 else "black"
                    ax.text(ci, ri, f"{val:.1f}±{sd:.1f}%", ha="center",
                            va="center", fontsize=10, fontweight="bold", color=tc)
            plt.colorbar(im, ax=ax, label="Arrival Rate (%)")
            ax.set_title(f"Generalization: {test_title} -- {cond} (mean±std)",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(
                exp_path,
                f"gen_heatmap_{test_type}_{cond.lower()}.png"), dpi=150)
            plt.close(fig)


def _find_best_seed(seed_data_list: list) -> dict:
    """Return the seed-run data dict with the highest avg arrival rate."""
    if not seed_data_list:
        return {}
    return max(seed_data_list, key=lambda d: d.get("arrival_avg", 0))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ts       = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    exp_path = os.path.join("experiment_runs", f"full_{ts}")
    os.makedirs(exp_path, exist_ok=True)

    conditions_str = ", ".join(c for c, _ in RUN_CONDITIONS)
    print(f"\n{'#'*60}")
    print(f"  MULTI-SEED EXPERIMENT ({N_SEEDS} seeds: {SEEDS})")
    print(f"  Conditions: {conditions_str}")
    print(f"  Configs: {[c.name for c in CONFIGS_ABC]}")
    print(f"  Checkpoint: {PRETRAINED_CHECKPOINT}")
    print(f"  Normalize master inputs: {NORMALIZE_MASTER_INPUTS}")
    print(f"  Normalize agent obs: {NORMALIZE_AGENT_OBS}")
    print(f"  Load pretrained checkpoint: {LOAD_PRETRAINED_CHECKPOINT}")
    print(f"  Output: {exp_path}")
    print(f"{'#'*60}\n")

    # all_seeds_data: { cfg_name: { cond: [data_s0, data_s1, ...] } }
    all_seeds_data = {}

    # ── Phase 1: Configs A/B/C × seeds ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1: CONFIGS A/B/C ({N_SEEDS} seeds × {len(RUN_CONDITIONS)} cond)")
    print(f"{'='*60}")
    for config in CONFIGS_ABC:
        all_seeds_data[config.name] = {}
        for cond, ablation in RUN_CONDITIONS:
            all_seeds_data[config.name][cond] = []
            for seed in SEEDS:
                data = _run_one_condition(config, cond, ablation, exp_path,
                                          seed=seed)
                all_seeds_data[config.name][cond].append(data)

    # ── Phase 2: Configs D/E/F × seeds ────────────────────────────────────────
    if CONFIGS_DEF:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: CONFIGS D/E/F ({N_SEEDS} seeds × {len(RUN_CONDITIONS)} cond)")
        print(f"{'='*60}")
        for config in CONFIGS_DEF:
            all_seeds_data[config.name] = {}
            for cond, ablation in RUN_CONDITIONS:
                all_seeds_data[config.name][cond] = []
                for seed in SEEDS:
                    data = _run_one_condition(config, cond, ablation, exp_path,
                                              seed=seed)
                    all_seeds_data[config.name][cond].append(data)
    else:
        print(f"\n  PHASE 2: SKIPPED (no D/E/F configs defined)")

    # ── Phase 3: Scalability (best A_base W_MASTER seed) ──────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3: CONFIG G (Scalability Test)")
    print(f"{'='*60}")
    best_a = _find_best_seed(all_seeds_data.get("A_base", {}).get("W_MASTER", []))
    best_a_dir = best_a.get("best_model_dir", "")
    best_a_ckpt = os.path.join(best_a_dir, "ckpt") if best_a_dir else ""
    if best_a_ckpt and os.path.exists(best_a_ckpt + "_agent.pth"):
        scale_results = _run_scalability_test(best_a_ckpt, exp_path)
    else:
        print("  WARNING: best A_base model not found, skipping scalability test")
        scale_results = {}

    # ── Phase 4: Embedding PCA (best A_base W_MASTER seed) ────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 4: MASTER EMBEDDING PCA ANALYSIS")
    print(f"{'='*60}")
    try:
        if best_a_ckpt and os.path.exists(best_a_ckpt + "_agent.pth"):
            _collect_and_plot_embeddings(best_a_ckpt, exp_path)
        else:
            print("  Skipped — no best model available")
    except ImportError:
        print("  WARNING: sklearn not available, skipping PCA analysis")
    except Exception as e:
        print(f"  WARNING: PCA analysis failed: {e}")

    # ── Phase 5: Scenario layouts ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 5: SCENARIO LAYOUT VISUALIZATION")
    print(f"{'='*60}")
    try:
        _plot_scenario_layouts(exp_path)
    except Exception as e:
        print(f"  WARNING: Scenario layout plotting failed: {e}")

    # ── Phase 6: Multi-seed comparison plots ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 6: GENERATING MULTI-SEED COMPARISON PLOTS")
    print(f"{'='*60}")
    _plot_multi_seed_comparisons(all_seeds_data, exp_path)
    _plot_generalization_heatmap(all_seeds_data, exp_path)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "configs": [c.name for c in ALL_CONFIGS],
        "hyperparameters": _BASE_HP,
    }

    for cfg_name, conditions in all_seeds_data.items():
        summary[cfg_name] = {}
        for cond, seed_list in conditions.items():
            arrivals_avg = [d["arrival_avg"] for d in seed_list]
            arrivals_l50 = [d["arrival_last50"] for d in seed_list]
            collisions   = [d["total_collisions"] for d in seed_list]
            summary[cfg_name][cond] = {
                "per_seed": [
                    {
                        "seed": SEEDS[si],
                        "arrival_avg": round(d["arrival_avg"], 2),
                        "arrival_last50": round(d["arrival_last50"], 2),
                        "total_collisions": d["total_collisions"],
                    }
                    for si, d in enumerate(seed_list)
                ],
                "mean_arrival_avg":  round(float(np.mean(arrivals_avg)), 2),
                "std_arrival_avg":   round(float(np.std(arrivals_avg)), 2),
                "mean_arrival_l50":  round(float(np.mean(arrivals_l50)), 2),
                "std_arrival_l50":   round(float(np.std(arrivals_l50)), 2),
                "mean_collisions":   round(float(np.mean(collisions)), 1),
                "held_out_mean": {},
                "conflict_test_mean": {},
            }
            for test_key in ["held_out", "conflict_test"]:
                out_key = f"{test_key}_mean"
                for env_l in ["intersection", "roundabout", "double_intersection"]:
                    arr_vals = [d[test_key].get(env_l, {}).get("arrival_rate_avg", 0)
                                for d in seed_list if test_key in d]
                    cr_vals  = [d[test_key].get(env_l, {}).get("crash_rate_pct", 0)
                                for d in seed_list if test_key in d]
                    summary[cfg_name][cond][out_key][env_l] = {
                        "arrival_mean": round(float(np.mean(arr_vals)), 2) if arr_vals else 0,
                        "arrival_std":  round(float(np.std(arr_vals)), 2)  if arr_vals else 0,
                        "crash_mean":   round(float(np.mean(cr_vals)), 2)  if cr_vals else 0,
                        "crash_std":    round(float(np.std(cr_vals)), 2)   if cr_vals else 0,
                    }

    if scale_results:
        summary["scalability"] = {}
        for n_agents, res in scale_results.items():
            summary["scalability"][f"{n_agents}_agents"] = {
                tt: {el: {k: round(v, 2) for k, v in env_res.items()}
                     for el, env_res in res[tt].items()}
                for tt in ["held_out", "conflict_test"]
            }

    with open(os.path.join(exp_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print final table ─────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  RESULTS SUMMARY (mean ± std over {N_SEEDS} seeds)")
    print(f"{'#'*60}")

    for cfg_name in all_seeds_data:
        w_list  = all_seeds_data[cfg_name].get("W_MASTER", [])
        wo_list = all_seeds_data[cfg_name].get("NO_MASTER", [])

        print(f"\n  == {cfg_name} {'='*40}")

        if w_list and wo_list:
            w_avg  = np.mean([d["arrival_avg"] for d in w_list])
            w_std  = np.std([d["arrival_avg"] for d in w_list])
            wo_avg = np.mean([d["arrival_avg"] for d in wo_list])
            wo_std = np.std([d["arrival_avg"] for d in wo_list])
            w_l50  = np.mean([d["arrival_last50"] for d in w_list])
            wo_l50 = np.mean([d["arrival_last50"] for d in wo_list])
            w_col  = np.mean([d["total_collisions"] for d in w_list])
            wo_col = np.mean([d["total_collisions"] for d in wo_list])

            print(f"  {'Metric':<35s} {'With Master':>16s} {'No Master':>16s} {'Delta':>10s}")
            print(f"  {'-'*80}")
            print(f"  {'Arrival (avg) %':<35s} {w_avg:>7.1f}±{w_std:<5.1f}%  {wo_avg:>7.1f}±{wo_std:<5.1f}%  {w_avg - wo_avg:>+8.1f}%")
            print(f"  {'Arrival (last-50) %':<35s} {w_l50:>13.1f}%  {wo_l50:>13.1f}%  {w_l50 - wo_l50:>+8.1f}%")
            print(f"  {'Collisions (mean)':<35s} {w_col:>14.0f}  {wo_col:>14.0f}  {w_col - wo_col:>+9.0f}")
            print()
            for env_l in ["intersection", "roundabout", "double_intersection"]:
                w_ho  = np.mean([d["held_out"].get(env_l, {}).get("arrival_rate_avg", 0) for d in w_list])
                wo_ho = np.mean([d["held_out"].get(env_l, {}).get("arrival_rate_avg", 0) for d in wo_list])
                print(f"  {f'Held-out {env_l}':<35s} {w_ho:>13.1f}%  {wo_ho:>13.1f}%  {w_ho - wo_ho:>+8.1f}%")
        else:
            for cond, seed_list in all_seeds_data[cfg_name].items():
                if not seed_list:
                    continue
                avg  = np.mean([d["arrival_avg"] for d in seed_list])
                std  = np.std([d["arrival_avg"] for d in seed_list])
                l50  = np.mean([d["arrival_last50"] for d in seed_list])
                l50s = np.std([d["arrival_last50"] for d in seed_list])
                col  = np.mean([d["total_collisions"] for d in seed_list])
                print(f"  [{cond}] Arrival avg: {avg:.1f}±{std:.1f}%  "
                      f"Last-50: {l50:.1f}±{l50s:.1f}%  "
                      f"Collisions: {col:.0f}")
                for env_l in ["intersection", "roundabout", "double_intersection"]:
                    ho = np.mean([d["held_out"].get(env_l, {}).get("arrival_rate_avg", 0)
                                  for d in seed_list])
                    print(f"    Held-out {env_l}: {ho:.1f}%")

    if scale_results:
        print(f"\n  == SCALABILITY (Config G, best seed) {'='*22}")
        for n_agents in sorted(scale_results.keys()):
            print(f"\n  {n_agents} agents:")
            for test_type in ["held_out", "conflict_test"]:
                for env_l in ["intersection", "roundabout", "double_intersection"]:
                    r = scale_results[n_agents][test_type].get(env_l, {})
                    print(f"    [{test_type}] {env_l}: "
                          f"arrival={r.get('arrival_rate_avg',0):.1f}%  "
                          f"crash={r.get('crash_rate_pct',0):.1f}%")

    print(f"\n  Results: {exp_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
