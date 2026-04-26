from __future__ import annotations

import argparse
import os
from datetime import datetime

import run_unified as unified
from master_embedding_deep_dive import run_deep_dive
from src.model.agent_handler import Driver
from src.model.master_model import MasterModel


def run_probe(args: argparse.Namespace) -> str:
    MasterModel.NORMALIZE_INPUTS = True
    Driver.NORMALIZE_AGENT_OBS = True

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    exp_path = os.path.join("experiment_runs", f"nm_{ts}")
    os.makedirs(exp_path, exist_ok=True)

    config = unified.RunConfig(
        name="A",
        total_episodes=args.episodes,
        conflict_schedule=[(0, 0.0)],
    )
    condition = "W"
    data = unified._run_one_condition(
        config,
        condition,
        False,
        exp_path,
        seed=args.seed,
    )

    source_run = os.path.join(exp_path, config.name, condition, f"s{args.seed}")
    analysis_dir = os.path.join(exp_path, "deep")
    deep_args = argparse.Namespace(
        source_run=source_run,
        seed=args.seed,
        episodes_per_env=args.analysis_episodes_per_env,
        output_dir=analysis_dir,
        normalize_master_inputs=True,
        normalize_agent_obs=True,
    )
    run_deep_dive(deep_args)

    print("\nNormalized master probe complete")
    print(f"Run: {source_run}")
    print(f"Deep dive: {analysis_dir}")
    print(f"Arrival avg: {data['arrival_avg']} | last50: {data['arrival_last50']} | collisions: {data['total_collisions']}")
    return exp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short probe run with normalized master inputs.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--analysis-episodes-per-env", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_probe(parse_args())
