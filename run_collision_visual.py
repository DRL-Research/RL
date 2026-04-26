"""
Watch a few episodes in a pygame window (same control stack as the collision dataset).

From project root or IDE Run.  Edit NUM_EPISODES, RENDER_DELAY_SEC, and LOAD_PREFIX below.
"""

from __future__ import annotations

import json
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from highwayenv.utils import patch_intersection_env, register_intersection_env

from src import project_globals
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.model.model_handler import load_models
from src.training.general_utils import initialize_models, setup_experiment_dirs
from src.diagnostics.full_collision_dataset import collect_episodes_to_csv, analyze_steps_csv
from src.diagnostics.collision_step_analysis import analyze_collision_step_pattern

NUM_EPISODES = 5
RENDER_MODE = "human"
RENDER_DELAY_SEC = 0.02
CSV_OUT = "experiments/collision_visual_steps.csv"
EPISODE_SUMMARY_JSON = "experiments/collision_visual_episode_summary.json"
STEPS_ANALYSIS_JSON = "experiments/collision_visual_steps_analysis.json"
STEP14_ANALYSIS_JSON = "experiments/collision_visual_step14_analysis.json"
LOAD_PREFIX = ""
STOCHASTIC_AGENT = False
VERBOSE_EVERY = 1


def _make_experiment() -> Experiment:
    return Experiment(
        RENDER_MODE=RENDER_MODE,
        EXPERIMENT_ID="collision_visual",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=1,
        ENT_COEF=0.05,
        COLLISION_REWARD=-50,
        REACHED_TARGET_REWARD=50,
        STARVATION_REWARD=0,
        HIGH_SPEED_REWARD=5,
        AGENT_REWARD_MODE="global",
        FULL_JOINT_TRAINING=True,
        COTRAIN_CYCLES=True,
        AGENT_LR=3e-3,
        MASTER_LR=3e-4,
        CLIP_RANGE=0.2,
        GAMMA=0.90,
        GAE_LAMBDA=0.90,
        AGENT_NET_ARCH="wide",
        EPISODE_AMOUNT_FOR_TRAIN=3,
        VF_COEF=1.0,
        N_PPO_EPOCHS=5,
        EPISODES_PER_CYCLE=300,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
    )


def main() -> None:
    patch_intersection_env()
    register_intersection_env()

    exp = _make_experiment()
    scratch = os.path.join("experiments", "collision_visual_scratch")
    os.makedirs(scratch, exist_ok=True)
    exp.EXPERIMENT_PATH = scratch
    setup_experiment_dirs(scratch)

    env_config = sc.make_env_config_exp7(
        collision_reward=exp.COLLISION_REWARD,
        arrived_reward=exp.REACHED_TARGET_REWARD,
        starvation_reward=exp.STARVATION_REWARD,
        high_speed_reward=exp.HIGH_SPEED_REWARD,
    )

    project_globals.reset_globals()
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)

    if LOAD_PREFIX:
        load_models(agent_model, master_model, LOAD_PREFIX)

    os.makedirs(os.path.dirname(CSV_OUT) or ".", exist_ok=True)

    collect_episodes_to_csv(
        exp,
        wrapped_env,
        master_model,
        agent_model,
        num_episodes=NUM_EPISODES,
        csv_path=CSV_OUT,
        episode_summary_json=EPISODE_SUMMARY_JSON,
        stochastic_agent=STOCHASTIC_AGENT,
        verbose_every=VERBOSE_EVERY,
        render_delay_sec=RENDER_DELAY_SEC,
    )

    analyze_steps_csv(CSV_OUT, STEPS_ANALYSIS_JSON)
    step14 = analyze_collision_step_pattern(
        CSV_OUT,
        EPISODE_SUMMARY_JSON,
        highlight_steps=(12, 13, 14, 15, 16),
    )
    with open(STEP14_ANALYSIS_JSON, "w", encoding="utf-8") as jf:
        json.dump(step14, jf, indent=2)

    print(f"Wrote {CSV_OUT} and summaries. Step-14 snapshot → {STEP14_ANALYSIS_JSON}")

    try:
        wrapped_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
