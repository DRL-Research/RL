#!/usr/bin/env python3
"""
Run N episodes with full hierarchical control, record per-step traces, and summarize
whether collisions align with HighwayEnv vehicle.crashed and typical agent-motion patterns.

Usage:
  python analyze_collisions.py --episodes 80 --out experiments/audit_report.json
  python analyze_collisions.py --load experiments/final_.../trained_model --episodes 40

  # Optional: save one full trace (first collision episode) for inspection
  python analyze_collisions.py --episodes 20 --dump-trace collision_ep_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from highwayenv.utils import patch_intersection_env, register_intersection_env

from src import project_globals
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.model.model_handler import load_models
from src.training.episode_utils import run_episode
from src.training.general_utils import initialize_models, setup_experiment_dirs
from src.diagnostics.collision_audit import (
    aggregate_audit,
    classify_collision_episode,
    save_json,
)


def _build_experiment(episodes_placeholder: int = 300) -> Experiment:
    """Match main_final W01-style knobs; EPISODES_PER_CYCLE unused for audit loop."""
    return Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID="collision_audit",
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
        EPISODES_PER_CYCLE=episodes_placeholder,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
    )


def run_collision_audit(
    episodes: int = 50,
    out: str = "collision_audit_report.json",
    load: str = "",
    *,
    stochastic: bool = False,
    dump_trace: str = "",
    seed: int = -1,
    verbose_every: int = 10,
) -> dict:
    """
    Run the audit and write ``out`` JSON. Returns the full summary dict (includes ``episodes_detail``).

    ``load``: prefix path; loads ``{load}_agent.pth`` and ``{load}_master.pth`` (see ``load_models``).
    """
    if seed >= 0:
        import numpy as np

        np.random.seed(seed)

    patch_intersection_env()
    register_intersection_env()

    exp = _build_experiment()
    out_dir = os.path.join("experiments", "collision_audit_scratch")
    os.makedirs(out_dir, exist_ok=True)
    exp.EXPERIMENT_PATH = out_dir
    setup_experiment_dirs(out_dir)

    env_config = sc.make_env_config_exp7(
        collision_reward=exp.COLLISION_REWARD,
        arrived_reward=exp.REACHED_TARGET_REWARD,
        starvation_reward=exp.STARVATION_REWARD,
        high_speed_reward=exp.HIGH_SPEED_REWARD,
    )

    project_globals.reset_globals()
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)

    if load:
        ok = load_models(agent_model, master_model, load)
        if not ok:
            print("Warning: load failed; continuing with fresh weights.", file=sys.stderr)

    master_model.freeze()
    train_both = False
    training_local_master = False
    training_agent = bool(stochastic)
    training_global_master = False

    episode_reports = []
    dumped = False
    global_step = 0

    for ep in range(1, episodes + 1):
        trace: list = []
        total_r, _, steps, crashed, arrival = run_episode(
            exp,
            global_step,
            wrapped_env,
            master_model,
            agent_model,
            train_both,
            training_local_master,
            training_agent,
            training_global_master,
            step_trace=trace,
        )
        global_step += steps

        classification = classify_collision_episode(
            trace,
            crashed,
            total_r,
            float(exp.COLLISION_REWARD),
        )
        row = {
            "episode": ep,
            "steps": steps,
            "episode_return": total_r,
            "crashed": crashed,
            "arrival_pct": arrival,
            "classification": classification,
        }
        episode_reports.append(row)

        if dump_trace and crashed and not dumped:
            save_json(dump_trace, {"episode": ep, "trace": trace, "classification": classification})
            dumped = True
            print(f"Wrote first collision trace to {dump_trace}")

        if verbose_every > 0 and (ep % verbose_every == 0 or ep == 1):
            print(
                f"Episode {ep}/{episodes}  steps={steps}  crashed={crashed}  "
                f"return={total_r:.1f}  arrival={arrival:.1f}%"
            )

    summary = aggregate_audit(episode_reports)
    summary["episodes_detail"] = episode_reports
    summary["settings"] = {
        "load": load or None,
        "stochastic_agent": bool(stochastic),
        "master_frozen": True,
    }
    save_json(out, summary)
    print(f"\nWrote report to {out}")
    print(json.dumps({k: summary[k] for k in summary if k != "episodes_detail"}, indent=2))

    try:
        wrapped_env.close()
    except Exception:
        pass
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit collision episodes vs agent actions")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Prefix path to saved weights (loads PREFIX_agent.pth and PREFIX_master.pth)",
    )
    parser.add_argument("--out", type=str, default="collision_audit_report.json")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample agent policy; default uses deterministic agent actions (evaluation mode)",
    )
    parser.add_argument(
        "--dump-trace",
        type=str,
        default="",
        help="If set, write full step trace of the first collision episode to this JSON path",
    )
    parser.add_argument("--seed", type=int, default=-1, help="If >=0, call np.random.seed")
    args = parser.parse_args()

    run_collision_audit(
        episodes=args.episodes,
        out=args.out,
        load=args.load,
        stochastic=args.stochastic,
        dump_trace=args.dump_trace,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
