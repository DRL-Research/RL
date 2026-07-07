"""
╔══════════════════════════════════════════════════════════════╗
║   VISUALIZATION & EVALUATION — Ego-Attention on highway-v0   ║
╚══════════════════════════════════════════════════════════════╝

Generates presentation figures from a trained checkpoint:
  figure1_training_curves.png      reward / crash / epsilon / loss
  figure2_attention_heatmap.png    who the ego watches over one episode
  figure3_single_frame.png         scene + attention bars + Q-values
  figure4_action_distribution.png  greedy action mix + per-episode reward
  figure5_diagnostics.png          7 extra diagnostic panels

RUN WITH:  python inference.py   (needs checkpoints/best_model.pt)
"""

import torch
import numpy as np
import gymnasium as gym
import highway_env
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import json
import os

from model import EgoAttentionNetwork, N_FEATURES, EMBED_DIM, N_ACTIONS, N_VEHICLES

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR   = "checkpoints"
OUTPUT_DIR = "logs/plots"
MAX_STEPS  = 80     # highway-v0: duration 40 s x policy_frequency 2 Hz

# highway-v0 DiscreteMetaAction order: 0..4
ACTION_NAMES  = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
ACTION_COLORS = ["#16a085", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  GENERATING FIGURES — Ego-Attention DQN on highway-v0")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at '{path}'. Run train_model.py first.")
    model = EgoAttentionNetwork(
        n_vehicles=N_VEHICLES, n_features=N_FEATURES,
        embed_dim=EMBED_DIM, n_actions=N_ACTIONS,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"\n  Loaded {path}  (episode {ckpt.get('episode', '?')})")
    return model


def make_env():
    """highway-v0 — must match train_model.py exactly (env_obs_attention.json)."""
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "lanes_count":      3,
        "vehicles_count":   15,
        "policy_frequency": 2,
        "duration":         40,
        "observation": {
            "type":           "Kinematics",
            "vehicles_count": N_VEHICLES,
            "features":       ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute":       False,
        },
    })
    return env


# ─────────────────────────────────────────────────────────────────
# FIGURE 1 — TRAINING CURVES
# ─────────────────────────────────────────────────────────────────

def plot_training_curves(metrics_path):
    print("\n  Figure 1: Training curves...")
    with open(metrics_path) as f:
        m = json.load(f)
    rewards    = np.array(m["episode_rewards"])
    collisions = np.array(m["collision_rate"])
    epsilon    = np.array(m["epsilon_history"])
    losses     = np.array(m["loss_history"])
    episodes   = np.arange(1, len(rewards) + 1)

    W = 20
    smooth = lambda a: np.convolve(a, np.ones(W) / W, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Social Attention DQN — Training Progress  (highway-v0)",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.25, color="#2980b9", lw=0.8, label="Raw")
    if len(rewards) >= W:
        ax.plot(episodes[W-1:], smooth(rewards), color="#2980b9", lw=2.2, label=f"{W}-ep avg")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Episode Reward", fontweight="bold"); ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward"); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    if len(collisions) >= W:
        cr = smooth(collisions) * 100
        ax.plot(episodes[W-1:], cr, color="#e74c3c", lw=2.2)
        ax.fill_between(episodes[W-1:], cr, alpha=0.15, color="#e74c3c")
    ax.set_title("Collision Rate (%)", fontweight="bold"); ax.set_xlabel("Episode")
    ax.set_ylabel("Crash %"); ax.set_ylim(0, 105); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(episodes, epsilon, color="#e67e22", lw=2.2)
    ax.fill_between(episodes, epsilon, alpha=0.15, color="#e67e22")
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1, label="min eps = 0.05")
    ax.set_title("Exploration Rate (epsilon)", fontweight="bold"); ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon"); ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    nz = [(i, l) for i, l in enumerate(losses) if l > 1e-8]
    if nz:
        idxs, vals = map(np.array, zip(*nz))
        ax.plot(idxs + 1, vals, alpha=0.25, color="#8e44ad", lw=0.8)
        if len(vals) >= W:
            ax.plot(idxs[W-1:] + 1, smooth(vals), color="#8e44ad", lw=2.2, label=f"{W}-ep avg")
    ax.set_title("Bellman MSE Loss", fontweight="bold"); ax.set_xlabel("Episode")
    ax.set_ylabel("Loss"); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure1_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# FIGURE 2 — ATTENTION HEATMAP OVER ONE EPISODE
# ─────────────────────────────────────────────────────────────────

def plot_attention_heatmap(model, env):
    print("\n  Figure 2: Attention heatmap over one episode...")
    obs, _ = env.reset(seed=7)
    all_attn, all_present, all_actions = [], [], []

    for _ in range(MAX_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals, attn = model(obs_t, return_attention=True)
        all_attn.append(attn[0, 1:].cpu().numpy())     # neighbours only
        all_present.append(obs[1:, 0].copy())
        all_actions.append(q_vals[0].argmax().item())
        obs, _, terminated, truncated, _ = env.step(all_actions[-1])
        if terminated or truncated:
            break

    attn_matrix    = np.array(all_attn)
    present_matrix = np.array(all_present)
    T = attn_matrix.shape[0]

    ever = present_matrix.max(axis=0) > 0.5
    show = np.where(ever)[0]
    if len(show) == 0:
        show = np.arange(min(5, N_VEHICLES - 1))
    attn_show, present_show = attn_matrix[:, show], present_matrix[:, show]
    labels = [f"V{i+1}" for i in show]
    K = len(show)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(attn_show.T, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=max(attn_show.max(), 1e-6), interpolation="nearest")
    fig.colorbar(im, ax=ax1, label="Attention Weight", shrink=0.8)
    ax1.set_title("Social Attention Weights — Who Does the Ego Watch Over Time?  (highway-v0)",
                  fontweight="bold")
    ax1.set_xlabel("Timestep"); ax1.set_ylabel("Neighbor Vehicle")
    ax1.set_yticks(range(K)); ax1.set_yticklabels(labels, fontsize=8); ax1.set_xlim(-0.5, T-0.5)

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(present_show.T, aspect="auto", cmap="Blues",
                     vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im2, ax=ax2, label="Presence", shrink=0.8)
    ax2.set_title("Vehicle Presence  (white = empty, blue = present)", fontweight="bold")
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("Neighbor")
    ax2.set_yticks(range(K)); ax2.set_yticklabels(labels, fontsize=8); ax2.set_xlim(-0.5, T-0.5)

    ax3 = fig.add_subplot(gs[2])
    cmap_a = mcolors.ListedColormap(ACTION_COLORS)
    ax3.imshow([all_actions], aspect="auto", cmap=cmap_a,
               vmin=0, vmax=N_ACTIONS-1, interpolation="nearest")
    ax3.set_title("Action Taken at Each Timestep", fontweight="bold")
    ax3.set_xlabel("Timestep"); ax3.set_yticks([]); ax3.set_xlim(-0.5, T-0.5)
    patches = [mpatches.Patch(color=c, label=a) for c, a in zip(ACTION_COLORS, ACTION_NAMES)]
    ax3.legend(handles=patches, loc="upper right", fontsize=8, ncol=5, bbox_to_anchor=(1.0, 2.3))

    path = f"{OUTPUT_DIR}/figure2_attention_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# FIGURE 3 — SINGLE DECISION STEP
# ─────────────────────────────────────────────────────────────────

def plot_single_frame(model, env):
    print("\n  Figure 3: Single-frame attention...")
    obs, _ = env.reset(seed=21)
    for _ in range(8):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = model(obs_t)
        obs, _, terminated, truncated, _ = env.step(q_vals.argmax().item())
        if terminated or truncated:
            obs, _ = env.reset(seed=21)

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_vals, attn = model(obs_t, return_attention=True)
    attn_np  = attn[0, 1:].cpu().numpy()
    q_np     = q_vals[0].cpu().numpy()
    presence = obs[1:, 0]
    best_a   = int(q_np.argmax())

    print(f"\n     Ego (relative frame): x={obs[0,1]:.3f} y={obs[0,2]:.3f} "
          f"vx={obs[0,3]:.3f} vy={obs[0,4]:.3f}")
    print("     ATTENTION (present neighbours):")
    for i in [j for j in range(N_VEHICLES-1) if presence[j] > 0.5][:8]:
        print(f"       V{i+1}: {attn_np[i]:.4f}  " + "#" * int(attn_np[i]*40))
    print("     Q-VALUES:")
    for i, (n, q) in enumerate(zip(ACTION_NAMES, q_np)):
        print(f"       {n:<11}: {q:+.4f}{'  <- CHOSEN' if i == best_a else ''}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Social Attention — Single Decision Step  (highway-v0, trained agent)",
                 fontsize=14, fontweight="bold")
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=max(attn_np.max(), 1e-6))

    # LEFT: relative scene (ego at origin; x ahead/behind, y lateral)
    ax = axes[0]
    ax.set_facecolor("#444444")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 0.6)
    ax.set_title("Scene (ego frame; color = attention)", fontweight="bold")
    for yl in (-0.4, 0.0, 0.4):
        ax.axhline(yl, color="white", ls="--", lw=0.8, alpha=0.4)
    es = 0.05
    ax.add_patch(mpatches.FancyBboxPatch((-es, -es*0.6), es*2, es*1.2,
                 boxstyle="round,pad=0.004", facecolor="#2980b9",
                 edgecolor="white", lw=2, zorder=6))
    ax.text(0, 0, "EGO", ha="center", va="center", fontsize=7, color="white",
            fontweight="bold", zorder=7)
    plotted = 0
    for i in range(N_VEHICLES - 1):
        if presence[i] < 0.5:
            continue
        x, y = float(obs[i+1, 1]), float(obs[i+1, 2])
        if abs(x) > 1.05 or abs(y) > 0.55:
            continue
        ax.add_patch(mpatches.FancyBboxPatch((x-es, y-es*0.6), es*2, es*1.2,
                     boxstyle="round,pad=0.004", facecolor=cmap(norm(attn_np[i])),
                     edgecolor="white", lw=1.2, alpha=0.92, zorder=5))
        ax.text(x, y, f"V{i+1}", ha="center", va="center", fontsize=6,
                color="black", fontweight="bold", zorder=6)
        plotted += 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Attention Weight", shrink=0.7, pad=0.02)
    ax.set_xlabel("X relative to ego (ahead ->)"); ax.set_ylabel("Y relative (lane offset)")
    ax.text(0.02, 0.97, f"{plotted} neighbours visible", transform=ax.transAxes,
            fontsize=8, va="top", color="white", alpha=0.8)

    # CENTER: attention bars
    ax = axes[1]
    idx = np.where(presence > 0.5)[0][:12]
    if len(idx) == 0:
        idx = np.arange(min(5, N_VEHICLES - 1))
    vals = attn_np[idx]
    ax.bar(range(len(idx)), vals, color=[cmap(norm(a)) for a in vals],
           edgecolor="#555555", lw=0.7)
    ax.set_title("Attention Distribution (present neighbours)", fontweight="bold")
    ax.set_xlabel("Neighbor Vehicle"); ax.set_ylabel("Attention Weight")
    ax.set_xticks(range(len(idx))); ax.set_xticklabels([f"V{i+1}" for i in idx],
                                                       fontsize=8, rotation=45)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0 / max(int((presence > 0.5).sum()), 1), color="gray", ls="--",
               lw=1, alpha=0.6, label="Uniform baseline"); ax.legend(fontsize=8)

    # RIGHT: Q-values
    ax = axes[2]
    cols = [ACTION_COLORS[i] if i == best_a else "#aaaaaa" for i in range(N_ACTIONS)]
    ax.bar(ACTION_NAMES, q_np, color=cols, edgecolor="#555555", lw=0.7)
    ax.set_title(f"Q-Values per Action (chosen: {ACTION_NAMES[best_a]})", fontweight="bold")
    ax.set_ylabel("Q-Value"); ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="black", lw=0.8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure3_single_frame.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# FIGURE 4 — GREEDY EVALUATION
# ─────────────────────────────────────────────────────────────────

def plot_action_distribution(model, env, n_eval=100):
    """Greedy (eps=0) rollouts. highway-v0 has no destination: success = surviving
    the full episode (truncation) without a crash."""
    print(f"\n  Figure 4: Greedy evaluation over {n_eval} episodes...")
    action_counts = np.zeros(N_ACTIONS, dtype=int)
    episode_rewards, episode_crashes, episode_survived = [], [], []

    for ep in range(n_eval):
        obs, _ = env.reset(seed=ep * 13)
        ep_rew, crashed, truncated = 0.0, False, False
        for _ in range(MAX_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_vals = model(obs_t)
            action = q_vals.argmax().item()
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            if info.get("crashed", False):
                crashed = True
            if terminated or truncated:
                break
        # highway: only terminal cause is a crash; reaching duration -> truncated.
        survived = bool(truncated and not crashed)
        episode_rewards.append(ep_rew)
        episode_crashes.append(crashed)
        episode_survived.append(survived)
        print(f"     Ep {ep+1:>3}: reward={ep_rew:+.3f}  "
              f"[{'CRASHED' if crashed else 'SURVIVED'}]")

    mean_rew    = float(np.mean(episode_rewards))
    crash_rate  = float(np.mean(episode_crashes)) * 100
    surv_rate   = float(np.mean(episode_survived)) * 100
    print(f"\n     Mean reward   : {mean_rew:.4f}")
    print(f"     Crash rate    : {crash_rate:.1f}%")
    print(f"     Survival rate : {surv_rate:.1f}%")
    print(f"     Action breakdown:")
    for n, c in zip(ACTION_NAMES, action_counts):
        print(f"       {n:<11}: {c:>5}  ({c/max(action_counts.sum(),1)*100:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Trained Agent Evaluation  ({n_eval} episodes, greedy eps=0)\n"
                 f"Mean reward: {mean_rew:.3f}   Crash: {crash_rate:.1f}%   "
                 f"Survival: {surv_rate:.1f}%", fontsize=12, fontweight="bold")

    wl = [f"{n}\n({c/max(action_counts.sum(),1)*100:.1f}%)"
          for n, c in zip(ACTION_NAMES, action_counts)]
    ax1.pie(action_counts + 1e-9, labels=wl, colors=ACTION_COLORS, startangle=90,
            textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax1.set_title("Action Distribution (greedy)", fontweight="bold")

    bar_colors = ["#e74c3c" if c else "#2ecc71" for c in episode_crashes]
    ax2.bar(range(1, n_eval+1), episode_rewards, color=bar_colors,
            edgecolor="white", lw=0.5)
    ax2.axhline(mean_rew, color="#2980b9", ls="--", lw=2, label=f"Mean = {mean_rew:.3f}")
    ax2.set_title("Reward per Episode  (red=crashed | green=survived)", fontweight="bold")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Total Reward")
    ax2.legend(fontsize=10); ax2.grid(axis="y", alpha=0.3)
    if n_eval <= 20:
        ax2.set_xticks(range(1, n_eval+1))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure4_action_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")

    return {
        "action_counts":    action_counts,
        "episode_rewards":  episode_rewards,
        "episode_crashes":  episode_crashes,
        "episode_survived": episode_survived,
    }


# ─────────────────────────────────────────────────────────────────
# FIGURE 5 — EXTRA TRAINING DIAGNOSTICS (7 panels)
# ─────────────────────────────────────────────────────────────────

def plot_training_diagnostics(metrics_path, eval_results, W=10):
    print("\n  Figure 5: Training diagnostics...")
    with open(metrics_path) as f:
        m = json.load(f)
    rewards = np.array(m["episode_rewards"], dtype=float)
    lengths = np.array(m["episode_lengths"], dtype=float)
    crashes = np.array(m["collision_rate"], dtype=float)
    epsilon = np.array(m["epsilon_history"], dtype=float)
    n  = len(rewards)
    ep = np.arange(1, n + 1)
    ma = lambda a, w: (a.copy() if len(a) < w else np.convolve(a, np.ones(w)/w, mode="valid"))

    fig, axes = plt.subplots(4, 2, figsize=(15, 19))
    fig.suptitle("Ego-Attention DQN — Training Diagnostics  (highway-v0)",
                 fontsize=14, fontweight="bold", y=0.997)

    # 1. Epsilon
    ax = axes[0, 0]
    ax.plot(ep, epsilon, color="#e67e22", lw=1.8)
    ax.fill_between(ep, epsilon, alpha=0.15, color="#e67e22")
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1, label="min eps = 0.05")
    ax.set_title("1. Exploration Rate (epsilon) over Episodes", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("epsilon"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # 2. Explore vs exploit steps (stacked)
    ax = axes[0, 1]
    Wm = max(n // 300, 1)
    roll = lambda a: a if Wm == 1 else np.convolve(a, np.ones(Wm)/Wm, mode="valid")
    xs = ep if Wm == 1 else ep[Wm-1:]
    ax.stackplot(xs, roll(lengths*epsilon), roll(lengths*(1.0-epsilon)),
                 labels=["Explore  (~ len*eps)", "Exploit  (~ len*(1-eps))"],
                 colors=["#f39c12", "#2980b9"], alpha=0.85)
    ax.set_title("2. Steps per Episode: Explore vs Exploit (stacked)", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps (smoothed)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)

    # 3. MA reward
    ax = axes[1, 0]
    ax.plot(ep, rewards, color="#2980b9", alpha=0.18, lw=0.6, label="Raw")
    ax.plot(ep[W-1:], ma(rewards, W), color="#2980b9", lw=2, label=f"MA (window={W})")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(f"3. Moving-Average Reward (window {W})", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # 4. Crash rate raw + MA
    ax = axes[1, 1]
    ax.plot(ep, crashes*100, color="#e74c3c", alpha=0.12, lw=0.6, label="Raw (0/100)")
    ax.plot(ep[W-1:], ma(crashes, W)*100, color="#e74c3c", lw=2, label=f"MA (window={W})")
    ax.set_title(f"4. Crash Rate per Episode (raw + MA {W})", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Crash %"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # 5. Outcome distribution (eval): Survived vs Crashed
    ax = axes[2, 0]
    cr = np.array(eval_results["episode_crashes"], dtype=bool)
    sv = np.array(eval_results["episode_survived"], dtype=bool)
    n_crash, n_surv = int(cr.sum()), int(sv.sum())
    bars = ax.bar(["Survived\n(full episode)", "Crashed"], [n_surv, n_crash],
                  color=["#2ecc71", "#e74c3c"], edgecolor="white")
    for b, v in zip(bars, [n_surv, n_crash]):
        ax.text(b.get_x()+b.get_width()/2, v, str(v), ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title(f"5. Episode Outcome Distribution (eval, greedy, {len(cr)} ep)",
                 fontweight="bold")
    ax.set_ylabel("Episodes"); ax.grid(axis="y", alpha=0.25)

    # 6. Cumulative collisions
    ax = axes[2, 1]
    ax.plot(ep, np.cumsum(crashes), color="#8e44ad", lw=2)
    ax.fill_between(ep, np.cumsum(crashes), alpha=0.12, color="#8e44ad")
    ax.set_title("6. Cumulative Collisions (training)", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total crashes"); ax.grid(alpha=0.25)

    # 7. Greedy action counts
    ax = axes[3, 0]
    counts = np.array(eval_results["action_counts"], dtype=int)
    bars = ax.bar(ACTION_NAMES, counts, color=ACTION_COLORS, edgecolor="white")
    tot = max(int(counts.sum()), 1)
    for b, v in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v}\n({v/tot*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("7. Action Counts — Greedy (exploitation only)", fontweight="bold")
    ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.25)

    axes[3, 1].axis("off")
    summary = (f"Training episodes : {n:,}\n"
               f"Final epsilon     : {epsilon[-1]:.3f}\n"
               f"Total collisions  : {int(crashes.sum()):,}\n\n"
               f"Eval (greedy, {len(cr)} ep):\n"
               f"  Survived : {n_surv}\n"
               f"  Crashed  : {n_crash}\n"
               f"  Mean reward : {np.mean(eval_results['episode_rewards']):.3f}")
    axes[3, 1].text(0.03, 0.97, summary, va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure5_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model(f"{SAVE_DIR}/best_model.pt")
    env   = make_env()

    cfg = env.unwrapped.config
    print("\n  ENV REWARD CONFIG:")
    for k in ["collision_reward", "high_speed_reward", "right_lane_reward",
              "lane_change_reward", "reward_speed_range", "normalize_reward",
              "duration", "policy_frequency", "lanes_count", "vehicles_count"]:
        if k in cfg:
            print(f"     {k:<20}: {cfg[k]}")

    metrics_path = f"{SAVE_DIR}/metrics.json"
    if os.path.exists(metrics_path):
        plot_training_curves(metrics_path)
    else:
        print("\n  No metrics.json — skipping Figure 1.")

    plot_attention_heatmap(model, env)
    plot_single_frame(model, env)
    eval_results = plot_action_distribution(model, env, n_eval=100)

    if os.path.exists(metrics_path):
        plot_training_diagnostics(metrics_path, eval_results, W=10)

    print("\n" + "=" * 65)
    print("  ALL FIGURES SAVED to logs/plots/")
    print("=" * 65)