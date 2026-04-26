"""
Single end-to-end training run with clear metrics: loss, arrival %, collision rate.

Uses the same W01 hyperparameters as main_final.py (wide agents, FULL_JOINT, ent=0.05, …).
Edit EPISODES_PER_CYCLE / CYCLES below if you want a shorter smoke test.

From project root or IDE (Run): cwd is forced to this file’s directory.

  py -3 run_learning_experiment.py

Outputs under experiments/learning_<timestamp>/:
  results.png, summary.json, trained_model/, episode_metrics.csv
"""

from __future__ import annotations

import os
import sys

# IDE “Run Python File” often uses a different cwd than the repo root.
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import csv
import json
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from highwayenv.utils import patch_intersection_env, register_intersection_env

from src import project_globals
from src.experiment_run_paths import EXPERIMENT_RUNS_ROOT
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.model.model_handler import save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers
from src.training.training_handler import training_loop


# ── length (375×4 = 1500 matches main_final; reduce for a quicker check) ─────
EPISODES_PER_CYCLE = 375
CYCLES = 4

SMOOTH_EP = 60
ROLLING_CRASH_WINDOW = 50


def _smooth(values, window: int = 60):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, np.nan), arr, np.full(pad, np.nan)])
    s = np.convolve(np.where(np.isnan(padded), 0, padded), kernel, mode="valid")
    c = np.convolve((~np.isnan(padded)).astype(float), kernel, mode="valid")
    s = s / np.where(c > 0, c, 1)
    s[c == 0] = np.nan
    return s[: len(arr)]


def _rolling_crash_pct(collisions_01: list[int], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(collisions_01)):
        start = max(0, i - window + 1)
        seg = collisions_01[start : i + 1]
        out.append(100.0 * sum(seg) / len(seg) if seg else 0.0)
    return out


def _load_episode_metrics(path: str):
    arrivals: list[float] = []
    rewards: list[float] = []
    collisions: list[int] = []
    if not os.path.isfile(path):
        return arrivals, rewards, collisions
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rewards.append(float(row["reward"]))
                arrivals.append(float(row["arrival_pct"]))
                collisions.append(int(float(row["collision"])))
            except (KeyError, ValueError):
                continue
    return arrivals, rewards, collisions


def _quarter_stats(arrivals: list[float], collisions: list[int], n: int):
    if n < 8:
        return None
    q = n // 4
    a0 = float(np.mean(arrivals[:q]))
    a1 = float(np.mean(arrivals[-q:]))
    c0 = 100.0 * sum(collisions[:q]) / q
    c1 = 100.0 * sum(collisions[-q:]) / q
    return {"arrival_first_q_pct": a0, "arrival_last_q_pct": a1, "crash_rate_first_q_pct": c0, "crash_rate_last_q_pct": c1}


def save_plots(
    results: dict,
    exp_path: str,
    metrics_csv: str,
    *,
    suptitle: str | None = None,
    total_episodes: int | None = None,
    smooth_ep: int | None = None,
    rolling_crash_window: int | None = None,
):
    total_ep = total_episodes if total_episodes is not None else (EPISODES_PER_CYCLE * CYCLES)
    sw = smooth_ep if smooth_ep is not None else SMOOTH_EP
    rw = rolling_crash_window if rolling_crash_window is not None else ROLLING_CRASH_WINDOW
    if suptitle is None:
        suptitle = f"Learning run — W01 ({total_ep} ep, {CYCLES} cycles, FULL_JOINT)"
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.28)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    def _ps(ax, vals, color, lbl, alpha_raw=0.12):
        if not vals:
            return
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x = np.arange(1, len(raw) + 1)
        ax.plot(x, raw, color=color, alpha=alpha_raw, linewidth=0.5)
        ax.plot(x, _smooth(vals, sw), color=color, linewidth=2.2, label=lbl)

    _ps(axes[0], results.get("arrival_rates", []), "#2196F3", "Arrival %")
    axes[0].set_title("Arrival rate", fontsize=12)
    axes[0].set_ylabel("%")
    axes[0].set_ylim(-5, 105)
    axes[0].axhline(80, color="green", linestyle="--", alpha=0.4)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    _ps(axes[1], results.get("episode_rewards", []), "#4CAF50", "Reward")
    axes[1].set_title("Episode reward", fontsize=12)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.4)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)

    _, _, coll = _load_episode_metrics(metrics_csv)
    if coll:
        roll = _rolling_crash_pct(coll, rw)
        x = np.arange(1, len(roll) + 1)
        axes[2].plot(x, roll, color="#E53935", linewidth=2.0, label=f"{rw}-ep %")
        axes[2].set_title("Collision episodes (rolling)", fontsize=12)
        axes[2].set_ylabel("% episodes w/ crash")
        axes[2].set_ylim(-5, 105)
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(fontsize=9)
    else:
        axes[2].set_title("Collision episodes (rolling)", fontsize=12)
        axes[2].text(0.5, 0.5, "No episode_metrics.csv", ha="center", va="center", transform=axes[2].transAxes)

    for key, col, lbl_s in [
        ("master_policy_losses", "#E53935", "Policy"),
        ("master_value_losses", "#FB8C00", "Value"),
        ("master_total_losses", "#8E24AA", "Total"),
    ]:
        _ps(axes[3], results.get(key, []), col, lbl_s, 0.2)
    axes[3].set_title("Master losses", fontsize=12)
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(fontsize=8)

    for key, col, lbl_s in [
        ("agent_group0_total_losses", "#00ACC1", "Agent G0"),
        ("agent_group1_total_losses", "#43A047", "Agent G1"),
    ]:
        _ps(axes[4], results.get(key, []), col, lbl_s, 0.2)
    axes[4].set_title("Agent total losses", fontsize=12)
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(fontsize=9)

    axes[5].axis("off")
    arr = [v for v in results.get("arrival_rates", []) if v is not None]
    _, _, coll2 = _load_episode_metrics(metrics_csv)
    qs = _quarter_stats(arr, coll2, min(len(arr), len(coll2))) if arr and coll2 else None
    lines = [
        f"Episodes: {total_ep}",
        f"Total collision episodes: {sum(coll2)}" if coll2 else "",
        "",
    ]
    if qs:
        lines += [
            f"Arrival %  first quarter: {qs['arrival_first_q_pct']:.1f}",
            f"Arrival %  last quarter:  {qs['arrival_last_q_pct']:.1f}",
            f"Crash ep % first quarter: {qs['crash_rate_first_q_pct']:.1f}",
            f"Crash ep % last quarter:  {qs['crash_rate_last_q_pct']:.1f}",
        ]
    axes[5].text(0.05, 0.95, "\n".join(l for l in lines if l is not None), transform=axes[5].transAxes, fontsize=11, va="top", family="monospace")

    for ax in (axes[0], axes[1], axes[2]):
        ax.set_xlabel("Episode")
    axes[3].set_xlabel("Training call")
    axes[4].set_xlabel("Training call")

    out = os.path.join(exp_path, "results.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot saved] {out}")


def save_json(
    results: dict,
    collisions: int,
    exp_path: str,
    metrics_csv: str,
    *,
    extra_fields: dict | None = None,
):
    arr_vals = [v for v in results.get("arrival_rates", []) if v is not None]
    last50 = arr_vals[-50:] if len(arr_vals) >= 50 else arr_vals
    _, _, coll = _load_episode_metrics(metrics_csv)
    qs = _quarter_stats(arr_vals, coll, min(len(arr_vals), len(coll))) if arr_vals and coll else None
    summary = {
        "total_collision_episodes": collisions,
        "arrival_rate_avg_pct": float(np.mean(arr_vals)) if arr_vals else None,
        "arrival_rate_last50_pct": float(np.mean(last50)) if last50 else None,
        "quarter_comparison": qs,
    }
    if extra_fields:
        summary = {**summary, **extra_fields}
    with open(os.path.join(exp_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {os.path.join(exp_path, 'summary.json')}")
    if qs:
        print(
            f"  arrival  Q1→Q4: {qs['arrival_first_q_pct']:.1f}% → {qs['arrival_last_q_pct']:.1f}%  |  "
            f"crash ep% Q1→Q4: {qs['crash_rate_first_q_pct']:.1f}% → {qs['crash_rate_last_q_pct']:.1f}%"
        )


def main() -> None:
    patch_intersection_env()
    register_intersection_env()

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    os.makedirs(EXPERIMENT_RUNS_ROOT, exist_ok=True)
    exp_path = os.path.join(EXPERIMENT_RUNS_ROOT, f"learning_{ts}")
    os.makedirs(exp_path, exist_ok=True)
    metrics_csv = os.path.join(exp_path, "episode_metrics.csv")

    n_ep = EPISODES_PER_CYCLE * CYCLES
    print(f"\n{'#' * 70}")
    print(f"  LEARNING RUN  —  {n_ep} episodes  ({EPISODES_PER_CYCLE}/cycle × {CYCLES} cycles)")
    print("  W01: ent=0.05  clip=0.2  agent_lr=3e-3  master_lr=3e-4  arch=wide  ep_train=3")
    print(f"  Path: {exp_path}")
    print(f"{'#' * 70}\n")

    project_globals.reset_globals()

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID="learning_W01",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=CYCLES,
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
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
    )
    exp.EXPERIMENT_PATH = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, "trained_model")

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
        experiment=exp,
        env=wrapped_env,
        agent_model=agent_model,
        master_model=master_model,
    )

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)
    save_plots(results, exp_path, metrics_csv)
    save_json(results, collisions, exp_path, metrics_csv)

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    print(f"\n{'#' * 70}")
    print(f"  DONE. Open: {os.path.join(exp_path, 'results.png')}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
