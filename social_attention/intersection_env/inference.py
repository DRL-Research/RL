"""
VISUALIZATION & EVALUATION — Ego-Attention DQN on intersection-v0

Generates presentation figures from a trained checkpoint:
  figure1_training_curves.png      reward / crash / epsilon / loss
  figure2_attention_heatmap.png    who the ego watches over one episode
  figure3_single_frame.png         scene + attention bars + Q-values
  figure4_action_distribution.png  greedy action mix + per-episode reward
  figure5_diagnostics.png          7 extra diagnostic panels
  
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
# CONFIG — intersection-v0 (3 actions: SLOWER / IDLE / FASTER)
# ─────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR   = "checkpoints"
OUTPUT_DIR = "logs/plots"
MAX_STEPS  = 13     # intersection-v0 default duration

# DiscreteMetaAction with longitudinal=True, lateral=False → 3 actions
ACTION_NAMES  = ["SLOWER", "IDLE", "FASTER"]
ACTION_COLORS = ["#e74c3c", "#3498db", "#f39c12"]   # red / blue / orange

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  GENERATING FIGURES — Ego-Attention DQN on intersection-v0")
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
    """intersection-v0 — must match train_model.py exactly."""
    env = gym.make("intersection-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "observation": {
            "type":           "Kinematics",
            "vehicles_count": N_VEHICLES,
            "features":       ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x":  [-100, 100], "y":  [-100, 100],
                "vx": [-20,   20], "vy": [-20,   20],
            },
            "absolute": True,
            "order":    "shuffled",
        },
        "destination": "o1",
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
    collisions = np.array(m["crash_rate"])
    epsilon    = np.array(m["epsilon_history"])
    losses     = np.array(m["loss_history"])
    episodes   = np.arange(1, len(rewards) + 1)

    W = 50
    smooth = lambda a: np.convolve(a, np.ones(W) / W, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Social Attention DQN — Training Progress  (intersection-v0)\n"
        "Paper: Social Attention for Autonomous Decision-Making in Dense Traffic",
        fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.2, color="#2980b9", lw=0.6, label="Raw")
    if len(rewards) >= W:
        ax.plot(episodes[W-1:], smooth(rewards), color="#2980b9", lw=2.2,
                label=f"{W}-ep avg")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Episode Reward", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    if len(collisions) >= W:
        cr = smooth(collisions) * 100
        ax.plot(episodes[W-1:], cr, color="#e74c3c", lw=2.2)
        ax.fill_between(episodes[W-1:], cr, alpha=0.15, color="#e74c3c")
    ax.set_title("Collision Rate (%)", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Crash %")
    ax.set_ylim(0, 105); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(episodes, epsilon, color="#e67e22", lw=2.2)
    ax.fill_between(episodes, epsilon, alpha=0.15, color="#e67e22")
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1, label="min eps = 0.05")
    ax.set_title("Exploration Rate (epsilon)", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    nonzero = [(i, l) for i, l in enumerate(losses) if l > 1e-8]
    if nonzero:
        idxs, vals = zip(*nonzero)
        idxs, vals = np.array(idxs), np.array(vals)
        ax.plot(idxs + 1, vals, alpha=0.2, color="#8e44ad", lw=0.6)
        if len(vals) >= W:
            ax.plot(idxs[W-1:] + 1,
                    np.convolve(vals, np.ones(W)/W, mode="valid"),
                    color="#8e44ad", lw=2.2, label=f"{W}-ep avg")
    ax.set_title("Bellman MSE Loss", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure1_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# FIGURE 2 — ATTENTION HEATMAP (over a full episode)
# ─────────────────────────────────────────────────────────────────

def plot_attention_heatmap(model, env):
    print("\n  Figure 2: Attention heatmap over one episode...")
    obs, _ = env.reset(seed=7)
    all_attn, all_present, all_actions = [], [], []

    for _ in range(MAX_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals, attn = model(obs_t, return_attention=True)
        all_attn.append(attn[0, 1:].cpu().numpy())   # [N-1] neighbors only
        all_present.append(obs[1:, 0].copy())
        all_actions.append(q_vals[0].argmax().item())
        obs, _, terminated, truncated, _ = env.step(all_actions[-1])
        if terminated or truncated:
            break

    attn_matrix    = np.array(all_attn)     # [T, N-1]
    present_matrix = np.array(all_present)  # [T, N-1]
    T = attn_matrix.shape[0]
    N_neighbors = N_VEHICLES - 1

    ever_present = present_matrix.max(axis=0) > 0.5
    show = np.where(ever_present)[0]
    if len(show) == 0:
        show = np.arange(min(5, N_neighbors))

    attn_show   = attn_matrix[:, show]
    present_show = present_matrix[:, show]
    labels = [f"V{i+1}" for i in show]
    K = len(show)

    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(attn_show.T, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=max(attn_show.max(), 1e-6),
                    interpolation="nearest")
    fig.colorbar(im, ax=ax1, label="Attention Weight", shrink=0.8)
    ax1.set_title("Social Attention Weights — Who Does the Ego Watch?",
                  fontweight="bold")
    ax1.set_xlabel("Timestep"); ax1.set_ylabel("Neighbor Vehicle")
    ax1.set_yticks(range(K)); ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlim(-0.5, T - 0.5)

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(present_show.T, aspect="auto", cmap="Blues",
                     vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im2, ax=ax2, label="Presence", shrink=0.8)
    ax2.set_title("Vehicle Presence  (white = empty, blue = present)",
                  fontweight="bold")
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("Neighbor")
    ax2.set_yticks(range(K)); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlim(-0.5, T - 0.5)

    ax3 = fig.add_subplot(gs[2])
    cmap_a = matplotlib.colors.ListedColormap(ACTION_COLORS)
    ax3.imshow([all_actions], aspect="auto", cmap=cmap_a,
               vmin=0, vmax=N_ACTIONS - 1, interpolation="nearest")
    ax3.set_title("Action at Each Timestep", fontweight="bold")
    ax3.set_xlabel("Timestep"); ax3.set_yticks([])
    ax3.set_xlim(-0.5, T - 0.5)
    patches = [mpatches.Patch(color=c, label=a)
               for c, a in zip(ACTION_COLORS, ACTION_NAMES)]
    ax3.legend(handles=patches, loc="upper right", fontsize=8,
               ncol=3, bbox_to_anchor=(1.0, 2.5))

    path = f"{OUTPUT_DIR}/figure2_attention_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# FIGURE 3 — SINGLE FRAME (best presentation slide)
# ─────────────────────────────────────────────────────────────────

def plot_single_frame(model, env):
    print("\n  Figure 3: Single-frame attention visualization...")

    obs, _ = env.reset(seed=21)
    for _ in range(6):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = model(obs_t)
        obs, _, terminated, truncated, _ = env.step(q_vals.argmax().item())
        if terminated or truncated:
            obs, _ = env.reset(seed=21)

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_vals, attn = model(obs_t, return_attention=True)

    attn_np  = attn[0, 1:].cpu().numpy()   # [N-1] neighbors
    q_np     = q_vals[0].cpu().numpy()     # [N_ACTIONS]
    presence = obs[1:, 0]
    best_a   = int(q_np.argmax())

    print(f"\n     Q-VALUES:")
    for i, (name, q) in enumerate(zip(ACTION_NAMES, q_np)):
        mark = " <- CHOSEN" if i == best_a else ""
        print(f"       {name:<10}: {q:+.4f}{mark}")

    cmap     = plt.cm.YlOrRd
    attn_max = max(attn_np.max(), 1e-6)
    norm     = mcolors.Normalize(vmin=0, vmax=attn_max)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Social Attention — Single Decision Step  (intersection-v0)",
                 fontsize=14, fontweight="bold")

    # LEFT: intersection scene schematic
    ax = axes[0]
    ax.set_facecolor("#3d5a3d")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_title("Intersection Scene (color = attention)", fontweight="bold")
    road_w, road_color = 0.45, "#555555"
    ax.fill_between([-1.1, 1.1], [-road_w]*2, [road_w]*2, color=road_color, zorder=1)
    ax.fill_betweenx([-1.1, 1.1], [-road_w]*2, [road_w]*2, color=road_color, zorder=1)
    ax.fill_between([-road_w, road_w], [-road_w]*2, [road_w]*2,
                    color="#666666", zorder=2)
    for xy in [(-1.1, -road_w), (road_w, 1.1)]:
        ax.plot(list(xy), [0, 0], "w--", lw=1.0, alpha=0.5, zorder=3)
    for xy in [(-1.1, -road_w), (road_w, 1.1)]:
        ax.plot([0, 0], list(xy), "w--", lw=1.0, alpha=0.5, zorder=3)

    es = 0.08
    ax.add_patch(mpatches.FancyBboxPatch((-es, -es*0.6), es*2, es*1.2,
                 boxstyle="round,pad=0.006", facecolor="#2980b9",
                 edgecolor="white", lw=2, zorder=6))
    ax.text(0, 0, "EGO", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", zorder=7)

    plotted = 0
    for i in range(N_VEHICLES - 1):
        if presence[i] < 0.5:
            continue
        x_raw, y_raw = float(obs[i+1, 1]), float(obs[i+1, 2])
        if abs(x_raw) > 1.0 or abs(y_raw) > 1.0:
            continue
        ax.add_patch(mpatches.FancyBboxPatch((x_raw-es, y_raw-es*0.6), es*2, es*1.2,
                     boxstyle="round,pad=0.005",
                     facecolor=cmap(norm(attn_np[i])), edgecolor="white",
                     lw=1.2, alpha=0.92, zorder=5))
        ax.text(x_raw, y_raw, f"V{i+1}", ha="center", va="center",
                fontsize=6, color="black", fontweight="bold", zorder=6)
        plotted += 1

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Attention Weight", shrink=0.7, pad=0.02)
    ax.set_xlabel("X (absolute)"); ax.set_ylabel("Y (absolute)")
    ax.text(0.02, 0.97, f"{plotted} neighbours visible",
            transform=ax.transAxes, fontsize=8, va="top", color="white", alpha=0.8)

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
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([f"V{i+1}" for i in idx], fontsize=8, rotation=45)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    n_present = max(int((presence > 0.5).sum()), 1)
    ax.axhline(1.0 / n_present, color="gray", ls="--", lw=1,
               alpha=0.6, label="Uniform baseline")
    ax.legend(fontsize=8)

    # RIGHT: Q-values — use actual q_np length to avoid shape mismatch
    ax = axes[2]
    n_act = len(q_np)
    act_names  = ACTION_NAMES[:n_act] if n_act <= len(ACTION_NAMES) else \
                 ACTION_NAMES + [f"A{i}" for i in range(len(ACTION_NAMES), n_act)]
    act_colors = ACTION_COLORS[:n_act] if n_act <= len(ACTION_COLORS) else \
                 ACTION_COLORS + ["#888888"] * (n_act - len(ACTION_COLORS))
    cols = [act_colors[i] if i == best_a else "#aaaaaa" for i in range(n_act)]
    ax.bar(act_names, q_np, color=cols, edgecolor="#555555", lw=0.7)
    ax.set_title(f"Q-Values (chosen: {act_names[best_a]})", fontweight="bold")
    ax.set_ylabel("Q-Value"); ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="black", lw=0.8)
    ax.text(best_a, q_np[best_a] + abs(q_np).max() * 0.05, "CHOSEN",
            ha="center", fontsize=10, color="#c0392b", fontweight="bold")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure3_single_frame.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# FIGURE 4 — GREEDY EVALUATION
# ─────────────────────────────────────────────────────────────────

def plot_action_distribution(model, env, n_eval=100):
    print(f"\n  Figure 4: Greedy evaluation over {n_eval} episodes...")

    action_counts   = np.zeros(N_ACTIONS, dtype=int)
    episode_rewards = []
    episode_crashes = []
    episode_arrived = []

    for ep in range(n_eval):
        obs, _ = env.reset(seed=ep * 13)
        ep_rew  = 0.0
        crashed = False

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

        arrived = bool(terminated and not crashed)
        episode_rewards.append(ep_rew)
        episode_crashes.append(crashed)
        episode_arrived.append(arrived)
        status = "CRASHED" if crashed else ("ARRIVED" if arrived else "timeout")
        print(f"     Ep {ep+1:>3}: reward={ep_rew:+.3f}  [{status}]")

    mean_rew    = float(np.mean(episode_rewards))
    crash_rate  = float(np.mean(episode_crashes)) * 100
    arrive_rate = float(np.mean(episode_arrived)) * 100
    print(f"\n     Mean reward  : {mean_rew:.3f}")
    print(f"     Crash rate   : {crash_rate:.1f}%")
    print(f"     Arrival rate : {arrive_rate:.1f}%")
    for name, cnt in zip(ACTION_NAMES, action_counts):
        pct = cnt / max(action_counts.sum(), 1) * 100
        print(f"     {name:<10}: {cnt:>4}  ({pct:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Trained Agent Evaluation  ({n_eval} greedy episodes)\n"
        f"Mean reward: {mean_rew:.3f}   Crash: {crash_rate:.1f}%   "
        f"Arrival: {arrive_rate:.1f}%",
        fontsize=12, fontweight="bold")

    wedge_labels = [
        f"{n}\n({c / max(action_counts.sum(), 1) * 100:.1f}%)"
        for n, c in zip(ACTION_NAMES, action_counts)]
    ax1.pie(action_counts + 1e-9, labels=wedge_labels, colors=ACTION_COLORS,
            startangle=90, textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax1.set_title("Action Distribution (greedy)", fontweight="bold")

    bar_colors = ["#e74c3c" if c else ("#2ecc71" if a else "#f39c12")
                  for c, a in zip(episode_crashes, episode_arrived)]
    ax2.bar(range(1, n_eval + 1), episode_rewards,
            color=bar_colors, edgecolor="white", lw=0.8)
    ax2.axhline(mean_rew, color="#2980b9", ls="--", lw=2,
                label=f"Mean = {mean_rew:.3f}")
    ax2.set_title("Reward per Episode\n(red=crash | green=arrived | orange=timeout)",
                  fontweight="bold")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Total Reward")
    ax2.legend(fontsize=10); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/figure4_action_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"     Saved -> {path}")

    return {
        "action_counts":   action_counts,
        "episode_rewards": episode_rewards,
        "episode_crashes": episode_crashes,
        "episode_arrived": episode_arrived,
    }


# ─────────────────────────────────────────────────────────────────
# FIGURE 5 — TRAINING DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────

def plot_training_diagnostics(metrics_path, eval_results, W=50):
    print("\n  Figure 5: Training diagnostics...")
    with open(metrics_path) as f:
        m = json.load(f)
    rewards = np.array(m["episode_rewards"], dtype=float)
    lengths = np.array(m["episode_lengths"], dtype=float)
    crashes = np.array(m["crash_rate"],      dtype=float)
    epsilon = np.array(m["epsilon_history"], dtype=float)
    n  = len(rewards)
    ep = np.arange(1, n + 1)
    ma = lambda a, w: (a.copy() if len(a) < w
                       else np.convolve(a, np.ones(w)/w, mode="valid"))

    fig, axes = plt.subplots(4, 2, figsize=(15, 19))
    fig.suptitle("Social Attention DQN — Training Diagnostics  (intersection-v0)",
                 fontsize=14, fontweight="bold", y=0.997)

    ax = axes[0, 0]
    ax.plot(ep, epsilon, color="#e67e22", lw=1.8)
    ax.fill_between(ep, epsilon, alpha=0.15, color="#e67e22")
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1, label="min eps = 0.05")
    ax.set_title("1. Exploration Rate (epsilon)", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("epsilon")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    Wm = max(n // 300, 1)
    roll = lambda a: (a if Wm == 1
                      else np.convolve(a, np.ones(Wm)/Wm, mode="valid"))
    xs = ep if Wm == 1 else ep[Wm-1:]
    ax.stackplot(xs, roll(lengths * epsilon), roll(lengths * (1 - epsilon)),
                 labels=["Explore (len*eps)", "Exploit (len*(1-eps))"],
                 colors=["#f39c12", "#2980b9"], alpha=0.85)
    ax.set_title("2. Steps per Episode: Explore vs Exploit", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps (smoothed)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(ep, rewards, color="#2980b9", alpha=0.15, lw=0.5, label="Raw")
    ax.plot(ep[W-1:], ma(rewards, W), color="#2980b9", lw=2,
            label=f"MA (window={W})")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(f"3. Moving-Average Reward (window {W})", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(ep, crashes * 100, color="#e74c3c", alpha=0.12, lw=0.5,
            label="Raw (0/100)")
    ax.plot(ep[W-1:], ma(crashes, W) * 100, color="#e74c3c", lw=2,
            label=f"MA (window={W})")
    ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5, label="50%")
    ax.set_title(f"4. Crash Rate (raw + MA {W})", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Crash %")
    ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2, 0]
    cr  = np.array(eval_results["episode_crashes"], dtype=bool)
    arr = np.array(eval_results["episode_arrived"], dtype=bool)
    n_crash = int(cr.sum())
    n_arr   = int((arr & ~cr).sum())
    n_to    = int(len(cr) - n_crash - n_arr)
    bars = ax.bar(["Arrived", "Crashed", "Timeout"],
                  [n_arr, n_crash, n_to],
                  color=["#2ecc71", "#e74c3c", "#f39c12"], edgecolor="white")
    for b, v in zip(bars, [n_arr, n_crash, n_to]):
        ax.text(b.get_x() + b.get_width()/2, v, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"5. Episode Outcomes (greedy, {len(cr)} ep)", fontweight="bold")
    ax.set_ylabel("Episodes"); ax.grid(axis="y", alpha=0.25)

    ax = axes[2, 1]
    ax.plot(ep, np.cumsum(crashes), color="#8e44ad", lw=2)
    ax.fill_between(ep, np.cumsum(crashes), alpha=0.12, color="#8e44ad")
    ax.set_title("6. Cumulative Collisions (training)", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total crashes")
    ax.grid(alpha=0.25)

    ax = axes[3, 0]
    counts = np.array(eval_results["action_counts"], dtype=int)
    bars = ax.bar(ACTION_NAMES, counts, color=ACTION_COLORS, edgecolor="white")
    tot = max(int(counts.sum()), 1)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v}\n({v/tot*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("7. Action Counts (greedy policy)", fontweight="bold")
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.25)

    axes[3, 1].axis("off")
    summary = (
        f"Training episodes : {n:,}\n"
        f"Final epsilon     : {epsilon[-1]:.3f}\n"
        f"Total collisions  : {int(crashes.sum()):,}\n\n"
        f"Eval ({len(cr)} greedy ep):\n"
        f"  Arrived  : {n_arr}\n"
        f"  Crashed  : {n_crash}\n"
        f"  Timeout  : {n_to}\n"
        f"  Mean rew : {np.mean(eval_results['episode_rewards']):.3f}"
    )
    axes[3, 1].text(0.03, 0.97, summary, va="top", ha="left",
                    fontsize=11, family="monospace")

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

    metrics_path = f"{SAVE_DIR}/metrics.json"
    if os.path.exists(metrics_path):
        plot_training_curves(metrics_path)
    else:
        print("\n  No metrics.json — skipping Figure 1.")

    plot_attention_heatmap(model, env)
    plot_single_frame(model, env)
    eval_results = plot_action_distribution(model, env, n_eval=100)

    if os.path.exists(metrics_path):
        plot_training_diagnostics(metrics_path, eval_results, W=50)

    print("\n" + "=" * 65)
    print("  ALL FIGURES SAVED to", OUTPUT_DIR)
    print("=" * 65)