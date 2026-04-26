# hier_main.py
"""
Hierarchical Architecture - Scalability POC
============================================
Run: python hier_main.py

Trains for 500 episodes and produces:
  results/hier_<timestamp>/
    ├── models/          saved .pt files
    ├── episodes.csv     per-episode metrics
    ├── losses.csv       per-update loss values
    └── plots/
        ├── reward_curve.png
        ├── collision_rate.png
        ├── arrival_rate.png
        ├── loss_curves.png
        └── summary.png   (combined dashboard)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make sure we can import from new_arch/
sys.path.insert(0, os.path.dirname(__file__))

from hier_train import HierLogger, train_hier


# ── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Hierarchy
    "n_local_masters":    2,
    "agents_per_master":  3,
    # n_agents is derived automatically: n_local_masters * agents_per_master

    # Embedding dimensions
    "global_embedding_dim": 4,   # GlobalMaster -> LocalMasters
    "local_embedding_dim":  4,   # LocalMasters -> Agents

    # Environment - calibrated for 6 agents at intersection
    "duration":           40,          # more time to cross
    "target_speeds":      [0, 10, 20],
    "collision_reward":   -5,          # small: avoids gradient explosion
    "arrived_reward":     20,          # positive signal when agent exits
    "high_speed_reward":  1.0,         # strong: encourages movement
    "reward_speed_range": [0, 9],

    # Coordination rewards (light - don't overshadow env signal)
    "coord_velocity_bonus":  0.5,
    "coord_safe_bonus":      1.0,
    "coord_danger_penalty":  2.0,
    "coord_diversity_bonus": 0.3,

    # Networks
    "master_hidden_dim": 64,
    "agent_hidden_dim":  48,
    "master_lr":         3e-4,
    "agent_lr":          3e-4,

    # PPO
    "gamma":              0.99,
    "gae_lambda":         0.95,
    "clip_eps":           0.2,
    "ppo_epochs":         4,
    "mini_batch_size":    64,
    "value_loss_coef":    0.5,
    "max_grad_norm":      0.5,
    "master_entropy_coef": 0.02,
    "agent_entropy_coef":  0.01,

    # Training schedule
    "total_episodes":          500,
    "train_every_n_episodes":   5,
    "print_every_n_episodes":  50,
}


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def smooth(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _x_for_smooth(n, w=20):
    if n < w:
        return np.arange(n)
    return np.arange(w - 1, n)


def plot_reward(ep_df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    eps  = ep_df["episode"].values
    rews = ep_df["total_reward"].values
    ax.plot(eps, rews, alpha=0.25, color="steelblue", lw=0.8)
    ax.plot(_x_for_smooth(len(rews)), smooth(rews), color="steelblue", lw=2, label="Total reward")

    rg0 = ep_df["reward_g0"].values
    rg1 = ep_df["reward_g1"].values
    ax.plot(_x_for_smooth(len(rg0)), smooth(rg0), color="tomato",    lw=1.5, ls="--", label="Group 0")
    ax.plot(_x_for_smooth(len(rg1)), smooth(rg1), color="seagreen",  lw=1.5, ls="--", label="Group 1")

    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.set_title("Reward Curve – Hierarchical Architecture (500 episodes)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_curve.png"), dpi=120)
    plt.close()


def plot_collision_arrival(ep_df: pd.DataFrame, plots_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    crashes = ep_df["collisions"].values.astype(float)
    ax1.plot(ep_df["episode"], crashes, alpha=0.25, color="firebrick", lw=0.8)
    ax1.plot(_x_for_smooth(len(crashes)), smooth(crashes), color="firebrick", lw=2)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Collisions per episode")
    ax1.set_title("Collision Rate"); ax1.grid(alpha=0.3)

    arrivals = ep_df["arrivals"].values.astype(float)
    ax2.plot(ep_df["episode"], arrivals, alpha=0.25, color="seagreen", lw=0.8)
    ax2.plot(_x_for_smooth(len(arrivals)), smooth(arrivals), color="seagreen", lw=2)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Arrivals per episode")
    ax2.set_title("Arrival Rate"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "collision_arrival.png"), dpi=120)
    plt.close()


def plot_losses(loss_df: pd.DataFrame, plots_dir: str):
    if loss_df.empty:
        return

    # Identify loss columns by keyword
    loss_cols = [c for c in loss_df.columns if "total_loss" in c]
    if not loss_cols:
        loss_cols = [c for c in loss_df.columns if c != "episode"]

    n = len(loss_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    colors = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]
    for i, col in enumerate(loss_cols):
        vals = loss_df[col].dropna().values
        eps  = loss_df.loc[loss_df[col].notna(), "episode"].values
        label = col.replace("/total_loss", "").replace("_", " ")
        c = colors[i % len(colors)]
        axes[i].plot(eps, vals, alpha=0.35, color=c, lw=0.8)
        if len(vals) >= 5:
            axes[i].plot(eps[4:], smooth(vals, 5), color=c, lw=2)
        axes[i].set_title(label.title())
        axes[i].set_xlabel("Episode"); axes[i].set_ylabel("Loss")
        axes[i].grid(alpha=0.3)

    plt.suptitle("Training Loss – All Network Levels", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=120)
    plt.close()


def plot_summary_dashboard(ep_df: pd.DataFrame, loss_df: pd.DataFrame, plots_dir: str):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Total reward
    ax1 = fig.add_subplot(gs[0, :2])
    rews = ep_df["total_reward"].values
    ax1.plot(ep_df["episode"], rews, alpha=0.2, color="steelblue", lw=0.7)
    ax1.plot(_x_for_smooth(len(rews)), smooth(rews), color="steelblue", lw=2)
    ax1.set_title("Total Reward"); ax1.set_xlabel("Episode"); ax1.grid(alpha=0.3)

    # 2. Collisions
    ax2 = fig.add_subplot(gs[0, 2])
    crashes = ep_df["collisions"].values.astype(float)
    ax2.plot(ep_df["episode"], crashes, alpha=0.2, color="firebrick", lw=0.7)
    ax2.plot(_x_for_smooth(len(crashes)), smooth(crashes), color="firebrick", lw=2)
    ax2.set_title("Collisions"); ax2.set_xlabel("Episode"); ax2.grid(alpha=0.3)

    # 3. Arrivals
    ax3 = fig.add_subplot(gs[1, 0])
    arrivals = ep_df["arrivals"].values.astype(float)
    ax3.plot(ep_df["episode"], arrivals, alpha=0.2, color="seagreen", lw=0.7)
    ax3.plot(_x_for_smooth(len(arrivals)), smooth(arrivals), color="seagreen", lw=2)
    ax3.set_title("Arrivals"); ax3.set_xlabel("Episode"); ax3.grid(alpha=0.3)

    # 4. Mean speed
    ax4 = fig.add_subplot(gs[1, 1])
    spd = ep_df["mean_speed"].values
    ax4.plot(ep_df["episode"], spd, alpha=0.2, color="darkorange", lw=0.7)
    ax4.plot(_x_for_smooth(len(spd)), smooth(spd), color="darkorange", lw=2)
    ax4.set_title("Mean Speed"); ax4.set_xlabel("Episode"); ax4.grid(alpha=0.3)

    # 5. Min distance
    ax5 = fig.add_subplot(gs[1, 2])
    mdist = ep_df["min_dist"].values
    ax5.plot(ep_df["episode"], mdist, alpha=0.2, color="mediumpurple", lw=0.7)
    ax5.plot(_x_for_smooth(len(mdist)), smooth(mdist), color="mediumpurple", lw=2)
    ax5.set_title("Min Distance"); ax5.set_xlabel("Episode"); ax5.grid(alpha=0.3)

    # 6. Per-group reward
    ax6 = fig.add_subplot(gs[2, :2])
    rg0 = ep_df["reward_g0"].values
    rg1 = ep_df["reward_g1"].values
    ax6.plot(_x_for_smooth(len(rg0)), smooth(rg0), lw=2, color="tomato",   label="Group 0 (LM_0)")
    ax6.plot(_x_for_smooth(len(rg1)), smooth(rg1), lw=2, color="seagreen", label="Group 1 (LM_1)")
    ax6.set_title("Per-Group Reward"); ax6.set_xlabel("Episode"); ax6.legend(); ax6.grid(alpha=0.3)

    # 7. Architecture diagram (text box)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    arch_text = (
        "Hierarchy\n"
        "─────────────\n"
        "GlobalMaster\n"
        "  emb_dim=4\n"
        "     │\n"
        " ┌───┴───┐\n"
        "LM_0   LM_1\n"
        " emb=4  emb=4\n"
        " │       │\n"
        "A0,A1,A2 A3,A4,A5\n\n"
        "Input: 5 slots×5\n"
        "+ 5-bit mask = 30"
    )
    ax7.text(0.05, 0.95, arch_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Hierarchical Architecture – Scalability POC (500 episodes)",
                 fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(plots_dir, "summary_dashboard.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Plots] summary_dashboard.png saved")


def generate_plots(results_dir: str):
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ep_path   = os.path.join(results_dir, "episodes.csv")
    loss_path = os.path.join(results_dir, "losses.csv")

    if not os.path.exists(ep_path):
        print("  [Plots] episodes.csv not found, skipping plots.")
        return

    ep_df   = pd.read_csv(ep_path)
    loss_df = pd.read_csv(loss_path) if os.path.exists(loss_path) else pd.DataFrame()

    plot_reward(ep_df, plots_dir)
    plot_collision_arrival(ep_df, plots_dir)
    plot_losses(loss_df, plots_dir)
    plot_summary_dashboard(ep_df, loss_df, plots_dir)
    print(f"  [Plots] All plots saved to {plots_dir}/")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def run():
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"hier_{ts}")
    models_dir  = os.path.join(results_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    # Derive n_agents
    config = CONFIG.copy()
    config["n_agents"] = config["n_local_masters"] * config["agents_per_master"]

    # Save config
    with open(os.path.join(results_dir, "config.txt"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    print(f"\nOutput directory: {results_dir}")

    logger     = HierLogger(results_dir)
    t0         = time.time()
    train_hier(config, models_dir, logger)
    elapsed    = (time.time() - t0) / 60.0

    print(f"\nTraining done in {elapsed:.1f} min")
    print("Generating plots...")
    generate_plots(results_dir)

    # Final summary
    ep_df = pd.read_csv(os.path.join(results_dir, "episodes.csv"))
    last50 = ep_df.tail(50)
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS (last 50 episodes)")
    print(f"{'='*55}")
    print(f"  Avg Reward   : {last50['total_reward'].mean():.1f}")
    print(f"  Avg Crashes  : {last50['collisions'].mean():.2f}")
    print(f"  Avg Arrivals : {last50['arrivals'].mean():.2f}")
    print(f"  Avg Speed    : {last50['mean_speed'].mean():.2f}")
    print(f"  Avg Min Dist : {last50['min_dist'].mean():.2f}")
    print(f"{'='*55}")
    print(f"\n  Results: {results_dir}/")


if __name__ == "__main__":
    run()
