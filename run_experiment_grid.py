"""
Hyperparameter grid — P-series.
O-series analysis (3000 ep):
  O02 (ep1, ent=0.05, PL85): 552 crashes, 87.6% avg, 100% last-20  ← BEST
  O03 (ep3, ent=0.005, PL85): 946 crashes, 74.2% avg, 85% last-20
                               crash 47%→15% (-31.7%) — still improving at ep 3000!
  O01 (ep3, ent=0, N02base):  833 crashes, 82.6% avg, 84% last-20

P-series goal: fine-tune around O02 and O03.
  - Does ep1 + tiny entropy (0.005) beat ep1 + ent=0.05?
  - Does ep2 (middle ground) outperform ep1 and ep3?
  - Is PL75 or PL90 better than PL85?
  - Does O03 (ep3, PL85) surpass O02 with 5000 episodes?

NEW: held-out test — 100 episodes on 5 unseen scenarios after each run.

  py run_experiment_grid.py

Outputs: experiment_runs/grid_<timestamp>/...
"""

from __future__ import annotations

import logging
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import csv
import json
from datetime import datetime

import matplotlib.cm as _cm
import matplotlib.pyplot as plt
import numpy as np

from highwayenv.utils import patch_intersection_env, register_intersection_env

from src import project_globals
from src.experiment_run_paths import EXPERIMENT_RUNS_ROOT
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.model.model_handler import save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers
from src.training.training_handler import training_loop, run_held_out_test

from run_learning_experiment import save_json, save_plots

logging.basicConfig(level=logging.WARNING)

# ── run length ────────────────────────────────────────────────────────────────
# True → 100×2 = 200 episodes per config (quick sanity). False → full run like learning.
SHORT_GRID = False
EPISODES_PER_CYCLE = 100 if SHORT_GRID else 500
CYCLES = 2 if SHORT_GRID else 3    # 3 × 500 = 1500 episodes

SMOOTH_COMPARE = 50
ROLLING_CRASH_WINDOW = 50

# ── Q-series base = P05 formula (ep1, ent=0.05, PL75) — most consistent winner ─
# P05: 538 crashes, 93.8% avg, 97.5% last-20, crash 22.5%→7.2%
# PL75 is reliably reachable regardless of random init (unlike PL85 which is fragile)
# 1500 episodes (3×500) — enough to see convergence + held-out test
_BASE = dict(
    collision_reward=-50,
    arrived_reward=50,
    starvation_reward=0,
    high_speed_reward=5,
    reward_mode="global",
    master_lr=3e-4,
    gamma=0.9,
    gae_lambda=0.9,
    agent_net_arch="wide",
    ep_for_train=1,
    vf_coef=1.0,
    n_ppo_epochs=5,
    ent_coef=0.05,
    ent_coef_final=0.05,
    agent_lr=3e-3,
    clip_range=0.2,
    n_steps=384,
    warmup_episodes=400,
    peak_arrival_threshold=75.0,  # PL75 — consistent, proven reliable
    n_value_epochs=0,
    target_speeds=[5, 10],
)

_CFG_KEYS = frozenset(k for k in _BASE if k != "label")


def _merged_cfg(overrides: dict) -> dict:
    return {**_BASE, **overrides}


# ── Q-series: 3 configs, PL75 only, 1500 episodes + held-out test ────────────
#
#  Q01 — P05 exact (reference): confirm P05 is consistently good
#  Q02 — ep1 + ent=0.005 + PL75: tiny entropy, PL75 — best generalization candidate
#  Q03 — ep1 + ent=0.01  + PL75: middle entropy
GRID_CONFIGS = [
    # Q01: P05 exact — confirm consistency of PL75 formula
    _merged_cfg({"label": "Q01_ep1_ent05_PL75_ref"}),

    # Q02: tiny entropy + PL75 — does low entropy generalize better?
    _merged_cfg({"label": "Q02_ep1_ent0005_PL75",
                 "ent_coef": 0.005,
                 "ent_coef_final": 0.005}),

    # Q03: mid entropy + PL75
    _merged_cfg({"label": "Q03_ep1_ent001_PL75",
                 "ent_coef": 0.01,
                 "ent_coef_final": 0.01}),
]


def validate_grid_configs() -> None:
    """Fail fast with a clear message if GRID_CONFIGS is inconsistent."""
    labels: list[str] = []
    for i, raw in enumerate(GRID_CONFIGS):
        if "label" not in raw:
            raise ValueError(f"GRID_CONFIGS[{i}] missing 'label'")
        label = raw["label"]
        if label in labels:
            raise ValueError(f"Duplicate label: {label!r}")
        labels.append(label)
        cfg = {**_BASE, **{k: v for k, v in raw.items() if k != "label"}}
        cfg["label"] = label
        missing = _CFG_KEYS - frozenset(k for k in cfg if k != "label")
        if missing:
            raise ValueError(f"{label}: missing keys {sorted(missing)}")
        if not (0.0 <= cfg["ent_coef"] <= 1.0):
            raise ValueError(f"{label}: ent_coef out of range")
        if not (0.0 <= cfg.get("ent_coef_final", 0.0) <= 1.0):
            raise ValueError(f"{label}: ent_coef_final out of range")
        if cfg.get("warmup_episodes", 0) < 0:
            raise ValueError(f"{label}: warmup_episodes must be >= 0")
        if not (0.0 <= cfg.get("peak_arrival_threshold", 0.0) <= 100.0):
            raise ValueError(f"{label}: peak_arrival_threshold must be in [0, 100]")
        if not (0.0 < cfg["clip_range"] <= 1.0):
            raise ValueError(f"{label}: clip_range invalid")
        if cfg["agent_lr"] <= 0 or cfg["master_lr"] <= 0:
            raise ValueError(f"{label}: learning rates must be > 0")
        if cfg["ep_for_train"] < 1:
            raise ValueError(f"{label}: ep_for_train >= 1")
        speeds = cfg.get("target_speeds", [5, 10])
        if len(speeds) != 2 or speeds[0] <= 0 or speeds[1] <= speeds[0]:
            raise ValueError(f"{label}: target_speeds must be [slow, fast] with 0 < slow < fast")
        # Rollout buffer must hold one full training batch (rough lower bound).
        min_steps = int(cfg["ep_for_train"]) * 50
        if int(cfg["n_steps"]) < min_steps:
            raise ValueError(
                f"{label}: n_steps={cfg['n_steps']} < ep_for_train*50={min_steps} "
                f"(increase n_steps or lower ep_for_train)"
            )


def _smooth(values, window: int = 50):
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


def _nanmean_all(lst):
    arr = np.array([v for v in (lst or []) if v is not None and not np.isnan(v)], dtype=float)
    return float(np.nanmean(arr)) if len(arr) else None


def _nanmean_last(lst, n=20):
    tail = [v for v in (lst or [])[-n:] if v is not None and not np.isnan(v)]
    return float(np.nanmean(np.array(tail, dtype=float))) if tail else None


def run_one_config(cfg_raw: dict, exp_path: str, total_ep: int) -> tuple[int, dict]:
    label = cfg_raw["label"]
    cfg = {**_BASE, **{k: v for k, v in cfg_raw.items() if k != "label"}}
    cfg["label"] = label
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(
        f"  ent={cfg['ent_coef']}  clip={cfg['clip_range']}  "
        f"agent_lr={cfg['agent_lr']:.2e}  master_lr={cfg['master_lr']:.2e}  "
        f"ppo_epochs={cfg['n_ppo_epochs']}  γ={cfg['gamma']}  λ={cfg['gae_lambda']}  "
        f"ep_train={cfg['ep_for_train']}  vf={cfg['vf_coef']}  n_steps={cfg['n_steps']}"
    )
    print(f"  → {exp_path}")
    print(f"{'=' * 70}\n")

    project_globals.reset_globals()

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID=f"grid_{label}",
        LOAD_MODEL_DIRECTORY="",
        EPOCHS=1,
        CYCLES=CYCLES,
        ENT_COEF=cfg["ent_coef"],
        ENT_COEF_FINAL=cfg.get("ent_coef_final", 0.0),
        WARMUP_EPISODES=cfg.get("warmup_episodes", 0),
        PEAK_ARRIVAL_THRESHOLD=cfg.get("peak_arrival_threshold", 0.0),
        N_VALUE_EPOCHS=cfg.get("n_value_epochs", 0),
        COLLISION_REWARD=cfg["collision_reward"],
        REACHED_TARGET_REWARD=cfg["arrived_reward"],
        STARVATION_REWARD=cfg["starvation_reward"],
        HIGH_SPEED_REWARD=cfg["high_speed_reward"],
        AGENT_REWARD_MODE=cfg["reward_mode"],
        FULL_JOINT_TRAINING=False,
        COTRAIN_CYCLES=False,
        AGENT_LR=cfg["agent_lr"],
        MASTER_LR=cfg["master_lr"],
        CLIP_RANGE=cfg["clip_range"],
        GAMMA=cfg["gamma"],
        GAE_LAMBDA=cfg["gae_lambda"],
        AGENT_NET_ARCH=cfg["agent_net_arch"],
        EPISODE_AMOUNT_FOR_TRAIN=cfg["ep_for_train"],
        VF_COEF=cfg["vf_coef"],
        N_PPO_EPOCHS=cfg["n_ppo_epochs"],
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
        EXPLORATION_EXPLOITATION_THRESHOLD=0,
        N_STEPS=int(cfg["n_steps"]),
    )
    exp.EXPERIMENT_PATH = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, "trained_model")

    env_config = sc.make_env_config_exp7(
        collision_reward=cfg["collision_reward"],
        arrived_reward=cfg["arrived_reward"],
        starvation_reward=cfg["starvation_reward"],
        high_speed_reward=cfg["high_speed_reward"],
        target_speeds=cfg.get("target_speeds", None),
    )

    os.makedirs(exp_path, exist_ok=True)
    setup_experiment_dirs(exp_path)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)
    agent_logger, master_logger = setup_loggers(exp_path)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    agent_model, master_model, collisions, _, _, results, best_model_dir = training_loop(
        experiment=exp,
        env=wrapped_env,
        agent_model=agent_model,
        master_model=master_model,
    )

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)

    # ── Held-out evaluation (best checkpoint vs 5 unseen scenarios) ──────────
    held_out_metrics = run_held_out_test(
        exp, wrapped_env, agent_model, master_model,
        best_model_dir=best_model_dir,
        n_episodes=100,
    )

    metrics_csv = os.path.join(exp_path, "episode_metrics.csv")
    cfg_snapshot = {k: v for k, v in cfg.items() if k != "label"}
    save_plots(
        results,
        exp_path,
        metrics_csv,
        suptitle=f"Grid — {label} ({total_ep} ep, FULL_JOINT)",
        total_episodes=total_ep,
        rolling_crash_window=ROLLING_CRASH_WINDOW,
    )
    save_json(
        results,
        collisions,
        exp_path,
        metrics_csv,
        extra_fields={
            "config_label": label,
            "hyperparameters": cfg_snapshot,
            "held_out_test": held_out_metrics,
        },
    )

    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    arr_last20 = _nanmean_last(results.get("arrival_rates", []), 20)
    print(
        f"  [{label}] collisions={collisions}  arrival_last20≈{arr_last20}%"
        f"  held_out_arrival={held_out_metrics.get('held_out_arrival_pct', 0):.1f}%"
    )
    return collisions, results, held_out_metrics


def _write_grid_outputs(grid_root: str, all_results: dict[str, dict]):
    summary = {}
    for cfg in GRID_CONFIGS:
        lbl = cfg["label"]
        if lbl not in all_results:
            continue
        res = all_results[lbl]
        arr_vals = [v for v in res.get("arrival_rates", []) if v is not None]
        last20 = arr_vals[-20:] if len(arr_vals) >= 20 else arr_vals
        last50 = arr_vals[-50:] if len(arr_vals) >= 50 else arr_vals
        qs = (res.get("quarter_comparison") or {}) if isinstance(res.get("quarter_comparison"), dict) else {}
        held = res.get("held_out_metrics") or {}
        summary[lbl] = {
            "total_collisions": res.get("total_collisions"),
            "arrival_rate_avg_pct": float(np.mean(arr_vals)) if arr_vals else None,
            "arrival_rate_last20_pct": float(np.mean(last20)) if last20 else None,
            "arrival_rate_last50_pct": float(np.mean(last50)) if last50 else None,
            "arrival_delta_q1_q4": (
                (qs.get("arrival_last_q_pct") or 0) - (qs.get("arrival_first_q_pct") or 0)
                if qs
                else None
            ),
            "crash_delta_q1_q4": (
                (qs.get("crash_rate_last_q_pct") or 0) - (qs.get("crash_rate_first_q_pct") or 0)
                if qs
                else None
            ),
            "held_out_arrival_pct": held.get("held_out_arrival_pct"),
            "held_out_crash_rate_pct": held.get("held_out_crash_rate_pct"),
            "held_out_n_crashes": held.get("held_out_n_crashes"),
            "master_policy_loss_avg": _nanmean_all(res.get("master_policy_losses", [])),
            "master_value_loss_avg": _nanmean_all(res.get("master_value_losses", [])),
            "agent_g0_loss_avg": _nanmean_all(res.get("agent_group0_total_losses", [])),
            "agent_g1_loss_avg": _nanmean_all(res.get("agent_group1_total_losses", [])),
            "cfg": {k: v for k, v in cfg.items() if k != "label"},
        }

    with open(os.path.join(grid_root, "grid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    stat_cols = [
        "total_collisions",
        "arrival_rate_avg_pct",
        "arrival_rate_last20_pct",
        "arrival_rate_last50_pct",
        "arrival_delta_q1_q4",
        "crash_delta_q1_q4",
        "held_out_arrival_pct",
        "held_out_crash_rate_pct",
        "held_out_n_crashes",
        "master_policy_loss_avg",
        "master_value_loss_avg",
        "agent_g0_loss_avg",
        "agent_g1_loss_avg",
    ]
    all_keys = sorted({k for c in GRID_CONFIGS for k in c if k != "label"})
    with open(os.path.join(grid_root, "grid_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label"] + stat_cols + all_keys)
        for lbl, d in summary.items():
            row = [lbl] + [d.get(c) for c in stat_cols]
            for k in all_keys:
                row.append(d["cfg"].get(k, ""))
            w.writerow(row)

    print(f"  → {os.path.join(grid_root, 'grid_summary.csv')}")


def _comparison_plot(grid_root: str, all_results: dict[str, dict]):
    done = [c for c in GRID_CONFIGS if c["label"] in all_results]
    if len(done) < 2:
        return
    n_done = len(done)
    cmap = plt.get_cmap("tab20" if n_done > 10 else "tab10")
    n_colors = getattr(cmap, "N", 20)
    colors = [cmap(i % n_colors) for i in range(n_done)]

    # Check if any config has held-out results to show
    has_held_out = any(
        (all_results[c["label"]].get("held_out_metrics") or {}).get("held_out_arrival_pct") is not None
        for c in done
    )
    n_cols = 3 if has_held_out else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 5.2))
    if n_cols == 2:
        axes = list(axes)
    total_ep = EPISODES_PER_CYCLE * CYCLES
    fig.suptitle(f"Grid comparison — {total_ep} ep / config", fontweight="bold", fontsize=13)

    for i, c in enumerate(done):
        lbl = c["label"]
        res = all_results[lbl]
        col = colors[i % len(colors)]
        ncol = res.get("total_collisions", "?")
        tag = f"{lbl} ({ncol} crash-ep)"
        arr = res.get("arrival_rates", [])
        rew = res.get("episode_rewards", [])
        for ax, vals in [(axes[0], arr), (axes[1], rew)]:
            if vals:
                raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
                ax.plot(raw, color=col, alpha=0.08, linewidth=0.5)
                ax.plot(_smooth(vals, SMOOTH_COMPARE), color=col, linewidth=2.0, label=tag)

    axes[0].set_title("Arrival rate — training (%)")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(-5, 105)
    axes[0].set_xlabel("Episode")
    axes[1].set_title("Episode reward — training")
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Episode")
    for ax in axes[:2]:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    # ── Held-out bar chart ────────────────────────────────────────────────────
    if has_held_out:
        ax_ho = axes[2]
        labels_ho, arrivals_ho, crashes_ho = [], [], []
        for i, c in enumerate(done):
            lbl = c["label"]
            hm = (all_results[lbl].get("held_out_metrics") or {})
            arr_val = hm.get("held_out_arrival_pct")
            cr_val  = hm.get("held_out_crash_rate_pct")
            if arr_val is not None:
                labels_ho.append(lbl)
                arrivals_ho.append(float(arr_val))
                crashes_ho.append(float(cr_val) if cr_val is not None else 0.0)

        x = np.arange(len(labels_ho))
        w = 0.35
        bars_arr = ax_ho.bar(x - w / 2, arrivals_ho, w, label="Arrival %",
                             color=[colors[done.index(next(c for c in done if c["label"] == l))] for l in labels_ho],
                             alpha=0.85)
        bars_cr  = ax_ho.bar(x + w / 2, crashes_ho,  w, label="Crash %",
                             color=[colors[done.index(next(c for c in done if c["label"] == l))] for l in labels_ho],
                             alpha=0.45, hatch="//")
        ax_ho.set_xticks(x)
        ax_ho.set_xticklabels([l.split("_")[0] for l in labels_ho], fontsize=9)
        ax_ho.set_ylim(0, 110)
        ax_ho.set_ylabel("%")
        ax_ho.set_title("Held-out test (100 ep, best checkpoint)\n5 unseen scenarios")
        ax_ho.axhline(y=90, color="green", linestyle="--", alpha=0.4, linewidth=1)
        ax_ho.legend(fontsize=8)
        ax_ho.grid(True, alpha=0.25, axis="y")
        # annotate values
        for bar in bars_arr:
            ax_ho.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                       f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=7)
        for bar in bars_cr:
            ax_ho.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                       f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = os.path.join(grid_root, "grid_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


def main() -> None:
    patch_intersection_env()
    register_intersection_env()

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    os.makedirs(EXPERIMENT_RUNS_ROOT, exist_ok=True)
    grid_root = os.path.join(EXPERIMENT_RUNS_ROOT, f"grid_{ts}")
    os.makedirs(grid_root, exist_ok=True)
    total_ep = EPISODES_PER_CYCLE * CYCLES

    print(f"\n{'#' * 70}")
    print(f"  EXPERIMENT GRID  root: {grid_root}")
    print(f"  {len(GRID_CONFIGS)} configs × {total_ep} episodes  (SHORT_GRID={SHORT_GRID})")
    print(f"{'#' * 70}\n")

    validate_grid_configs()

    all_results: dict[str, dict] = {}

    for i, cfg in enumerate(GRID_CONFIGS, 1):
        label = cfg["label"]
        exp_path = os.path.join(grid_root, label)
        print(f"[{i}/{len(GRID_CONFIGS)}] {label}")
        try:
            collisions, results, held_out_metrics = run_one_config(cfg, exp_path, total_ep)
        except Exception as exc:
            import traceback

            print(f"\n[ERROR] {label}: {exc}")
            traceback.print_exc()
            results = {k: [] for k in ["arrival_rates", "episode_rewards"]}
            collisions = -1
            held_out_metrics = {}

        results["total_collisions"] = collisions
        results["held_out_metrics"] = held_out_metrics
        # merge quarter stats from per-run summary.json for grid CSV deltas
        summ_path = os.path.join(exp_path, "summary.json")
        if os.path.isfile(summ_path):
            try:
                with open(summ_path, encoding="utf-8") as sf:
                    sj = json.load(sf)
                results["quarter_comparison"] = sj.get("quarter_comparison")
            except Exception:
                results["quarter_comparison"] = None
        else:
            results["quarter_comparison"] = None

        all_results[label] = results
        _write_grid_outputs(grid_root, all_results)
        _comparison_plot(grid_root, all_results)

    print(f"\n{'#' * 70}")
    print(f"  GRID complete: {grid_root}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
