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

import math
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

    # Hierarchical / two-group runs use group keys; unified loop uses agent_* only.
    _group_keys = ("agent_group0_total_losses", "agent_group1_total_losses")
    if any(results.get(k) for k in _group_keys):
        for key, col, lbl_s in [
            ("agent_group0_total_losses", "#00ACC1", "Agent G0"),
            ("agent_group1_total_losses", "#43A047", "Agent G1"),
        ]:
            _ps(axes[4], results.get(key, []), col, lbl_s, 0.2)
    else:
        for key, col, lbl_s in [
            ("agent_policy_losses", "#00ACC1", "Agent policy"),
            ("agent_value_losses", "#FB8C00", "Agent value"),
            ("agent_total_losses", "#43A047", "Agent total"),
        ]:
            _ps(axes[4], results.get(key, []), col, lbl_s, 0.2)
    axes[4].set_title("Agent losses", fontsize=12)
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


def _fmt_csv_float(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    try:
        return f"{float(x):.8f}"
    except (TypeError, ValueError):
        return ""


def _write_csv_rows(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_unified_metric_exports(
    results: dict,
    exp_path: str,
    metrics_csv: str,
    *,
    env_arrivals: dict[str, list] | None = None,
    smooth_ep: int | None = None,
    rolling_crash_window: int | None = None,
    total_episodes: int | None = None,
) -> None:
    """
    Write extra CSV/JSON and auxiliary PNGs for unified hierarchical runs:
    PPO update log (per LM1/LM2/GM + weight norms), aligned time series, snapshot JSON.
    """
    sw = smooth_ep if smooth_ep is not None else SMOOTH_EP
    rw = rolling_crash_window if rolling_crash_window is not None else ROLLING_CRASH_WINDOW

    ppo_rows = results.get("ppo_training_log") or []
    if ppo_rows:
        ppo_fields = sorted({k for row in ppo_rows for k in row.keys()})
        _write_csv_rows(os.path.join(exp_path, "ppo_training_log.csv"), ppo_fields, ppo_rows)

    n_ep_hint = total_episodes or 0
    arrivals = results.get("arrival_rates", [])
    rewards = results.get("episode_rewards", [])
    coll_rates = results.get("collision_rates", [])
    n_ts = max(len(arrivals), len(rewards), len(coll_rates), n_ep_hint)
    ts_rows: list[dict[str, object]] = []
    for i in range(n_ts):
        ep = i + 1
        row: dict[str, object] = {"episode": ep}
        row["arrival_pct"] = arrivals[i] if i < len(arrivals) else ""
        row["episode_reward"] = rewards[i] if i < len(rewards) else ""
        row["collision_pct"] = coll_rates[i] if i < len(coll_rates) else ""
        seg_a = arrivals[max(0, i - 49) : i + 1] if i < len(arrivals) else []
        seg_ok = [x for x in seg_a if x is not None]
        row["rolling_mean_arrival_50"] = float(np.mean(seg_ok)) if seg_ok else ""
        ts_rows.append(row)
    if ts_rows:
        _write_csv_rows(
            os.path.join(exp_path, "timeseries_training.csv"),
            ["episode", "arrival_pct", "episode_reward", "collision_pct", "rolling_mean_arrival_50"],
            ts_rows,
        )

    if env_arrivals:
        max_len = max(len(v) for v in env_arrivals.values()) if env_arrivals else 0
        env_rows: list[dict[str, object]] = []
        env_names = sorted(env_arrivals.keys())
        for i in range(max_len):
            er: dict[str, object] = {"episode_index_1based": i + 1}
            for name in env_names:
                col = f"arrival_pct_{name.replace(' ', '_').replace('-', '_')}"
                seq = env_arrivals[name]
                er[col] = seq[i] if i < len(seq) else ""
            env_rows.append(er)
        if env_rows:
            hdr = ["episode_index_1based"] + [
                f"arrival_pct_{name.replace(' ', '_').replace('-', '_')}" for name in env_names
            ]
            _write_csv_rows(os.path.join(exp_path, "arrivals_by_env_episode.csv"), hdr, env_rows)

    snapshot = {
        "n_episodes": len(results.get("arrival_rates", [])),
        "n_ppo_updates": len(ppo_rows),
        "mean_arrival_pct": float(np.mean([x for x in arrivals if x is not None]))
        if any(v is not None for v in arrivals)
        else None,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "ppo_training_log_tail": ppo_rows[-10:] if ppo_rows else [],
    }
    with open(os.path.join(exp_path, "training_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(
        "  [metrics] ppo_training_log.csv, timeseries_training.csv,"
        " arrivals_by_env_episode.csv (multi-env), training_snapshot.json, summary.json,"
        " plots/*.png"
    )

    save_unified_auxiliary_plots(
        results, exp_path, metrics_csv,
        smooth_ep=sw, rolling_crash_window=rw,
        total_episodes=total_episodes,
    )


def save_unified_auxiliary_plots(
    results: dict,
    exp_path: str,
    metrics_csv: str,
    *,
    smooth_ep: int = SMOOTH_EP,
    rolling_crash_window: int = ROLLING_CRASH_WINDOW,
    total_episodes: int | None = None,
) -> None:
    """Additional figures from PPO logs and episodic collisions (Unified runs)."""
    plots_dir = os.path.join(exp_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ppo_rows = results.get("ppo_training_log") or []

    # ── 1) LM1 / LM2 / GM total losses per PPO update ───────────────────────
    if ppo_rows:
        xs = np.arange(1, len(ppo_rows) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, lbl, col in [
            ("lm1_total_loss", "LM1 total", "#C62828"),
            ("lm2_total_loss", "LM2 total", "#1565C0"),
            ("gm_total_loss", "GM total", "#2E7D32"),
        ]:
            ys = [r.get(key) for r in ppo_rows]
            ys = [float(y) if y is not None else np.nan for y in ys]
            ax.plot(xs, ys, "o-", ms=3, lw=1.2, label=lbl, color=col, alpha=0.85)
        ax.set_title("Master auxiliary buffers — total loss per PPO update")
        ax.set_xlabel("PPO update #")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        p1 = os.path.join(plots_dir, "master_buffer_total_losses.png")
        plt.savefig(p1, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  [plot saved] {p1}")

        # ── 2) Parameter ‖θ‖₂ before/after PPO ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        def _finite(x) -> bool:
            try:
                v = float(x)
                return not (math.isnan(v) or math.isinf(v))
            except (TypeError, ValueError):
                return False

        d_m = [float(r["delta_master_param_l2norm"]) for r in ppo_rows if _finite(r.get("delta_master_param_l2norm"))]
        d_a = [float(r["delta_agent_param_l2norm"]) for r in ppo_rows if _finite(r.get("delta_agent_param_l2norm"))]
        if d_m:
            axes[0].bar(np.arange(1, len(d_m) + 1), d_m, color="#5C6BC0", alpha=0.85)
            axes[0].set_title("Δ‖θ‖₂ Master (concat policy params)")
            axes[0].set_xlabel("PPO update #")
            axes[0].grid(True, axis="y", alpha=0.25)
        if d_a:
            axes[1].bar(np.arange(1, len(d_a) + 1), d_a, color="#00897B", alpha=0.85)
            axes[1].set_title("Δ‖θ‖₂ Agent (concat policy params)")
            axes[1].set_xlabel("PPO update #")
            axes[1].grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        p2 = os.path.join(plots_dir, "parameter_norm_deltas.png")
        plt.savefig(p2, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  [plot saved] {p2}")

        # ── Absolute norms trajectories ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))

        def _nf(v):
            try:
                z = float(v)
                return z if z == z else np.nan
            except (TypeError, ValueError):
                return np.nan

        ym = [_nf(r.get("master_param_l2norm_post")) for r in ppo_rows]
        ya = [_nf(r.get("agent_param_l2norm_post")) for r in ppo_rows]
        ax.plot(xs, ym, "-", label="Master ‖θ‖₂ post-update", color="#3949AB")
        ax.plot(xs, ya, "-", label="Agent ‖θ‖₂ post-update", color="#00796B")
        ax.set_xlabel("PPO update #")
        ax.legend()
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        p3 = os.path.join(plots_dir, "parameter_l2norms_post_update.png")
        plt.savefig(p3, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  [plot saved] {p3}")

    # ── 3) Larger 2×2: arrival / reward / rolling crash / collision pct ─────────
    total_ep = total_episodes or len(results.get("arrival_rates", []))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Unified-run episode metrics (dense)", fontsize=14, fontweight="bold")

    def _psp(ax, vals, color, lbl):
        if not vals:
            return
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x = np.arange(1, len(raw) + 1)
        ax.plot(x, raw, color=color, alpha=0.15, lw=0.6)
        ax.plot(x, _smooth(list(vals), smooth_ep), color=color, lw=2.0, label=lbl)

    arr = results.get("arrival_rates", [])
    rew = results.get("episode_rewards", [])
    crp = results.get("collision_rates", [])
    _psp(axes[0, 0], arr, "#1565C0", "Arrival %")
    axes[0, 0].set_ylim(-5, 105)
    axes[0, 0].set_title("Arrival rate (smoothed)")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.22)

    _psp(axes[0, 1], rew, "#2E7D32", "Reward")
    axes[0, 1].set_title("Episode reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.22)

    _, _, coll01 = _load_episode_metrics(metrics_csv)
    if coll01:
        roll = _rolling_crash_pct(coll01, rolling_crash_window)
        x = np.arange(1, len(roll) + 1)
        axes[1, 0].plot(x, roll, color="#C62828", lw=1.8, label=f"Rolling {rolling_crash_window} ep")
        axes[1, 0].set_ylim(-5, 105)
        axes[1, 0].set_title("Crash episode rate (rolling)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.22)

    if crp:
        x2 = np.arange(1, len(crp) + 1)
        axes[1, 1].plot(x2, crp, color="#6A1B9A", alpha=0.2, lw=0.7)
        axes[1, 1].plot(x2, _smooth(crp, smooth_ep), color="#4527A0", lw=2.0)
    axes[1, 1].set_title("Collision rate from results (% per ep)")
    axes[1, 1].grid(True, alpha=0.22)

    for ax in axes.flat:
        ax.set_xlabel("Episode")
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "episode_dashboard_extended.png")
    plt.savefig(p4, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot saved] {p4}")


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
