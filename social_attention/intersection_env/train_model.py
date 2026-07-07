"""
DQN Training — Social Attention on intersection-v0
This script trains a DQN agent with ego-attention on the intersection-v0 environment,

OUTPUT:
  checkpoints/best_model.pt      <- best episode-average reward seen
  checkpoints/final_model.pt     <- weights after episode 4000
  checkpoints/metrics.json       <- full per-episode logs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import highway_env
import random
import math
import json
import os
import time
from collections import deque

import wandb
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for training scripts
import matplotlib.pyplot as plt

from model import EgoAttentionNetwork, N_FEATURES, EMBED_DIM, N_ACTIONS, N_VEHICLES

# Action labels from the paper (Section 4): SLOWER=0, NO-OP=1, FASTER=2
# intersection-v0 default action space has 3 discrete actions in this order.
ACTION_LABELS = {0: "SLOWER", 1: "NO-OP", 2: "FASTER"}
# Extend with generic labels if N_ACTIONS > 3
for _a in range(3, N_ACTIONS):
    ACTION_LABELS[_a] = "A{}".format(_a)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  (matched to rl-agents baseline.json + ego_attention_2h.json)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES  = 4_000          # matches rl-agents --episodes 4000

BUFFER_SIZE   = 15_000         # baseline.json: memory_capacity
BATCH_SIZE    = 64             # baseline.json: batch_size
MIN_BUFFER    = BATCH_SIZE     # rl-agents starts training as soon as buffer >= batch_size

GAMMA         = 0.95           # baseline.json: gamma
LEARNING_RATE = 5e-4           # baseline.json: learning_rate (Adam)
TARGET_UPDATE = 512            # baseline.json: target_update

EPS_START     = 1.0            # baseline.json: temperature
EPS_END       = 0.05           # baseline.json: final_temperature
EPS_TAU       = 15_000         # baseline.json: tau  (step-based exponential decay)

LOG_EVERY     = 50             # print a line every N episodes
SAVE_DIR      = "checkpoints"

SEED          = 42

WANDB_PROJECT  = "phases_wandb"
WANDB_RUN_NAME = "intersection-v0"          # set to a string to give the run a fixed name
WANDB_GROUP_NAME = "social_attention"  # set to a string to group runs together



# ─────────────────────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples (state, action, reward, next_state, done).'''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch  = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    '''DQN agent with ego-attention architecture, Double DQN updates, and epsilon-greedy exploration.'''
    def __init__(self):
        self.epsilon     = EPS_START
        self.total_steps = 0

        self.policy_net = EgoAttentionNetwork().to(DEVICE)
        self.target_net = EgoAttentionNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)

    # ------------------------------------------------------------------
    def select_action(self, obs):
        """Returns (action, is_exploit).
        is_exploit=True  → greedy action from policy network
        is_exploit=False → random exploration action
        """
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS), False
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            return self.policy_net(obs_t)[0].argmax().item(), True

    # ------------------------------------------------------------------
    def learn(self):
        """Sample a minibatch and do one gradient step (Double DQN).
        Returns a dict of step-level metrics, or None if buffer not ready."""
        if len(self.buffer) < MIN_BUFFER:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Current Q-values
        q_values  = self.policy_net(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            q_next = self.target_net(next_states).gather(
                1, best_actions.unsqueeze(1)).squeeze(1)
            # dones=1 only for true terminal states (crash / arrival),
            # NOT for time-limit truncations — matching rl-agents behaviour.
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = F.mse_loss(q_current, q_target)

        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return None

        self.optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm BEFORE clipping (detects exploding gradients)
        grad_norm = 0.0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(grad_norm)

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)   # gradient clipping (matches rl-agents)
        self.optimizer.step()

        return {
            "loss":         loss.item(),
            "q_current":    q_current.mean().item(),
            "q_target":     q_target.mean().item(),
            "td_error":     (q_current - q_target).abs().mean().item(),
            "grad_norm":    grad_norm,
            "reward_batch": rewards.mean().item(),
        }

    # ------------------------------------------------------------------
    def update_target_network(self):
        '''Copy policy network weights to target network.'''
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        '''Update epsilon using exponential decay based on total steps taken.'''
        # Exponential decay over steps — identical to rl-agents epsilon_greedy.py
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.total_steps / EPS_TAU)

    # ------------------------------------------------------------------
    def save(self, path, episode, metrics):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "episode":         episode,
            "model_state":     self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon":         self.epsilon,
            "total_steps":     self.total_steps,
            "metrics":         metrics,
        }, path)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    """
    Build intersection-v0 with the exact config from rl-agents env.json.
    """
    env = gym.make("intersection-v0", render_mode=None)
    env.unwrapped.configure({
        "observation": {
            "type":           "Kinematics",
            "vehicles_count": N_VEHICLES,          # 15
            "features":       ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x":  [-100, 100],
                "y":  [-100, 100],
                "vx": [-20,   20],
                "vy": [-20,   20],
            },
            "absolute": True,
            "order":    "shuffled",
            # normalize=True is the KinematicObservation default; rl-agents relies on it
        },
        "destination": "o1",
        # duration=13 is the intersection-v0 default; rl-agents does not override it
    })
    return env


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD FIGURE  (logged to wandb)
# ─────────────────────────────────────────────────────────────────────────────

def log_dashboard_figure(metrics, cumulative_action_counts, cumulative_exploit_action_counts,
                         episode, global_step):
    """
    Two-figure dashboard logged to wandb:

    Figure A — Paper Figure 4 replica (3 panels, matching Leurent & Mercat 2019):
      1. Total reward per episode
      2. Episode length per episode
      3. Average velocity per episode

    Figure B — Extended results (4 panels):
      1. Reward stability with MA
      2. Crash rate + cumulative collisions inset
      3. Episode outcomes bar chart (Success vs Crash)
      4. Action distribution: all steps vs exploit-only
    """
    window_ma = 50
    rewards    = np.array(metrics["episode_rewards"])
    lengths    = np.array(metrics["episode_lengths"])
    velocities = np.array(metrics["velocity_history"])
    crashes    = np.array(metrics["crash_rate"])
    episodes   = np.arange(1, len(rewards) + 1)

    def ma(arr, w):
        if len(arr) < w:
            return np.array([]), np.array([])
        smoothed = np.convolve(arr, np.ones(w) / w, mode="valid")
        return episodes[w - 1:], smoothed

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE A — Paper Figure 4 replica
    # ══════════════════════════════════════════════════════════════════════════
    fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 4))
    fig_a.suptitle(
        "Paper Figure 4 Replica — Episode {}  (Leurent & Mercat, 2019)".format(episode),
        fontsize=12, fontweight="bold", color="#1a2744"
    )
    fig_a.patch.set_facecolor("#f7f9fc")

    panel_data = [
        (rewards,    "total reward",  "Total Reward",    "#1f77b4"),
        (lengths,    "length",        "Episode Length",  "#ff7f0e"),
        (velocities, "velocity",      "Avg Velocity",    "#2ca02c"),
    ]
    for ax, (data, ylabel, title, color) in zip(axes_a, panel_data):
        ax.set_facecolor("#ffffff")
        ax.plot(episodes, data, color=color, alpha=0.25, linewidth=0.6)
        ep_ma, sm = ma(data, window_ma)
        if len(sm) > 0:
            ax.plot(ep_ma, sm, color=color, linewidth=2.2,
                    label="MA (w={})".format(window_ma))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    plt.tight_layout()
    wandb.log({"charts/paper_figure4": wandb.Image(fig_a)}, step=global_step)
    plt.close(fig_a)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B — Extended 4-panel dashboard
    # ══════════════════════════════════════════════════════════════════════════
    cum_collisions = np.cumsum(crashes)
    n_success = int(np.sum(1 - crashes))
    n_crash   = int(np.sum(crashes))

    fig_b, axes_b = plt.subplots(2, 2, figsize=(14, 9))
    fig_b.suptitle(
        "intersection-v0 — Extended Results (Episode {})".format(episode),
        fontsize=13, fontweight="bold", color="#1a2744"
    )
    fig_b.patch.set_facecolor("#f7f9fc")
    for ax in axes_b.flat:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    # Panel 1: Reward stability
    ax = axes_b[0, 0]
    ax.plot(episodes, rewards, color="#4a90d9", alpha=0.3, linewidth=0.7, label="Raw")
    ep_ma, sm = ma(rewards, window_ma)
    if len(sm) > 0:
        ax.plot(ep_ma, sm, color="#1a2744", linewidth=2.0,
                label="MA (w={})".format(window_ma))
    ax.set_title("Reward Stability", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Crash rate + cumulative collisions inset
    ax = axes_b[0, 1]
    ax.plot(episodes, crashes, color="#4a90d9", alpha=0.3, linewidth=0.7, label="Raw")
    ep_ma, sm = ma(crashes, window_ma)
    if len(sm) > 0:
        ax.plot(ep_ma, sm, color="#e07b39", linewidth=2.0,
                label="MA (w={})".format(window_ma))
    ax.set_title("Crash Rate over Episodes", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Crash Rate (0–1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = ax.inset_axes([0.55, 0.05, 0.42, 0.45])
    ax2.plot(episodes, cum_collisions, color="#c0392b", linewidth=1.5)
    ax2.set_title("Cum. Collisions", fontsize=8)
    ax2.set_xlabel("Ep", fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.annotate("{:,}".format(int(cum_collisions[-1])),
                 xy=(episodes[-1], cum_collisions[-1]),
                 xytext=(-30, -12), textcoords="offset points",
                 fontsize=8, color="#c0392b", fontweight="bold")

    # Panel 3: Episode outcomes bar chart
    ax = axes_b[1, 0]
    bars = ax.bar(["Success", "Crash"], [n_success, n_crash],
                  color=["#2ecc71", "#e74c3c"], width=0.5, edgecolor="white")
    for bar, val in zip(bars, [n_success, n_crash]):
        pct = 100 * val / max(len(rewards), 1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(n_success, n_crash) * 0.02,
                "{:,}\n({:.1f}%)".format(val, pct),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Episode Outcome Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(n_success, n_crash) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: Action distribution — all vs exploit-only
    ax = axes_b[1, 1]
    action_ids  = sorted(cumulative_action_counts.keys())
    labels      = [ACTION_LABELS.get(a, str(a)) for a in action_ids]
    all_counts  = [cumulative_action_counts.get(a, 0)         for a in action_ids]
    exp_counts  = [cumulative_exploit_action_counts.get(a, 0) for a in action_ids]
    x = np.arange(len(action_ids))
    bw = 0.35
    b1 = ax.bar(x - bw / 2, all_counts, bw, label="All steps",    color="#4a90d9")
    b2 = ax.bar(x + bw / 2, exp_counts, bw, label="Exploit only", color="#e07b39")
    max_count = max(all_counts + exp_counts) if (all_counts + exp_counts) else 1
    for bar, val in zip(list(b1) + list(b2), all_counts + exp_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_count * 0.01,
                    "{:,}".format(val),
                    ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Action Counts (SLOWER / NO-OP / FASTER)", fontweight="bold")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    wandb.log({"charts/dashboard": wandb.Image(fig_b)}, step=global_step)
    plt.close(fig_b)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── wandb init ───────────────────────────────────────────────────────────
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group=WANDB_GROUP_NAME,
        config={
            "num_episodes":  NUM_EPISODES,
            "buffer_size":   BUFFER_SIZE,
            "batch_size":    BATCH_SIZE,
            "gamma":         GAMMA,
            "learning_rate": LEARNING_RATE,
            "target_update": TARGET_UPDATE,
            "eps_start":     EPS_START,
            "eps_end":       EPS_END,
            "eps_tau":       EPS_TAU,
            "n_vehicles":    N_VEHICLES,
            "n_features":    N_FEATURES,
            "embed_dim":     EMBED_DIM,
            "n_actions":     N_ACTIONS,
            "seed":          seed,
            "device":        str(DEVICE),
        },
    )

    env   = make_env()
    agent = DQNAgent()

    # Log gradients + parameter histograms every 500 optimizer steps
    wandb.watch(agent.policy_net, log="all", log_freq=500)

    # Seed the env once at the start (rl-agents does not re-seed each episode)
    env.reset(seed=seed)

    metrics = {
        "episode_rewards":    [],
        "episode_lengths":    [],
        "crash_rate":         [],
        "arrival_rate":       [],
        "epsilon_history":    [],
        "loss_history":       [],
        "velocity_history":   [],   # paper Figure 4: avg velocity per episode
    }

    best_window = float("-inf")   # smoothed over last 50 episodes (matches rl-agents)
    t_start     = time.time()

    # global_step used as the wandb x-axis so step-level and episode-level
    # logs share the same axis (total environment steps taken)
    global_step = 0

    # Cumulative action counters across all episodes (for dashboard)
    cumulative_action_counts        = {a: 0 for a in range(N_ACTIONS)}
    cumulative_exploit_action_counts = {a: 0 for a in range(N_ACTIONS)}

    print("Training on {}".format(DEVICE))
    print("Episodes: {}  |  buffer: {}  |  batch: {}  |  tau: {}".format(
        NUM_EPISODES, BUFFER_SIZE, BATCH_SIZE, EPS_TAU))
    print()
    print("  {:>5}  {:>10}  {:>8}  {:>6}  {:>8}  {:>7}  {:>7}".format(
        "Ep", "AvgRew(50)", "TotSteps", "eps", "Loss", "Crash%", "Arr%"))
    print("  " + "-" * 62)

    for episode in range(1, NUM_EPISODES + 1):

        obs, _    = env.reset()   # no seed — rl-agents does not re-seed each episode
        ep_reward = 0.0
        ep_losses      = []
        ep_q_currents  = []
        ep_q_targets   = []
        ep_td_errors   = []
        ep_grad_norms  = []
        crashed   = False
        arrived   = False

        # Per-episode action tracking
        ep_explore_steps = 0
        ep_exploit_steps = 0
        ep_action_counts        = {a: 0 for a in range(N_ACTIONS)}
        ep_exploit_action_counts = {a: 0 for a in range(N_ACTIONS)}

        # Paper metrics (Figure 4)
        ep_steps      = 0          # actual env steps (episode length in the paper)
        ep_velocities = []         # ego vehicle speed each step → average velocity
        ep_speed_reward    = 0.0   # accumulated +1 rewards (driving at max speed)
        ep_collision_penalty = 0.0 # accumulated −5 rewards (collisions)

        while True:
            action, is_exploit = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Track explore / exploit split and action distribution
            if is_exploit:
                ep_exploit_steps += 1
                ep_exploit_action_counts[action] += 1
            else:
                ep_explore_steps += 1
            ep_action_counts[action] += 1
            cumulative_action_counts[action]         += 1
            if is_exploit:
                cumulative_exploit_action_counts[action] += 1

            # KEY FIX: store only `terminated` as the done flag.
            # Truncated episodes (time limit) are NOT true terminal states;
            # their next-state value should still be bootstrapped.
            agent.buffer.push(obs, action, reward, next_obs, terminated)

            agent.total_steps += 1
            global_step       += 1
            agent.update_epsilon()

            target_updated = False
            if agent.total_steps % TARGET_UPDATE == 0:
                agent.update_target_network()
                target_updated = True

            step_metrics = agent.learn()

            if step_metrics is not None:
                ep_losses.append(step_metrics["loss"])
                ep_q_currents.append(step_metrics["q_current"])
                ep_q_targets.append(step_metrics["q_target"])
                ep_td_errors.append(step_metrics["td_error"])
                ep_grad_norms.append(step_metrics["grad_norm"])

                # ── per-step wandb log ────────────────────────────────────────
                step_log = {
                    "train/loss":         step_metrics["loss"],
                    "train/q_current":    step_metrics["q_current"],
                    "train/q_target":     step_metrics["q_target"],
                    "train/td_error":     step_metrics["td_error"],
                    "train/grad_norm":    step_metrics["grad_norm"],
                    "train/reward_batch": step_metrics["reward_batch"],
                    "train/epsilon":      agent.epsilon,
                    "train/buffer_fill":  len(agent.buffer),
                }
                if target_updated:
                    step_log["event/target_network_update"] = 1
                wandb.log(step_log, step=global_step)

            ep_reward += reward
            ep_steps  += 1
            obs        = next_obs

            # Paper Figure 4 — average velocity metric
            # highway-env stores the controlled vehicle on the unwrapped env
            try:
                ep_velocities.append(env.unwrapped.vehicle.speed)
            except AttributeError:
                pass   # safety fallback if env layout changes

            # Reward decomposition (+1 for speed, -5 for collision)
            if reward <= -4.0:
                ep_collision_penalty += reward   # -5 collision hit
            else:
                ep_speed_reward += reward        # +1 (or 0) speed reward

            if info.get("crashed", False):
                crashed = True
            if terminated and not info.get("crashed", False):
                arrived = True

            if terminated or truncated:
                break

        # ── per-episode metrics ──────────────────────────────────────────────
        # ep_steps is the actual number of env steps (= episode length in paper)
        ep_length    = ep_steps
        avg_velocity = float(np.mean(ep_velocities)) if ep_velocities else 0.0
        avg_loss  = float(np.mean(ep_losses))     if ep_losses     else 0.0
        avg_qc    = float(np.mean(ep_q_currents)) if ep_q_currents else 0.0
        avg_qt    = float(np.mean(ep_q_targets))  if ep_q_targets  else 0.0
        avg_tde   = float(np.mean(ep_td_errors))  if ep_td_errors  else 0.0
        avg_gnorm = float(np.mean(ep_grad_norms)) if ep_grad_norms else 0.0

        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_lengths"].append(ep_length)
        metrics["crash_rate"].append(1 if crashed else 0)
        metrics["arrival_rate"].append(1 if arrived else 0)
        metrics["epsilon_history"].append(agent.epsilon)
        metrics["loss_history"].append(avg_loss)
        metrics["velocity_history"].append(avg_velocity)

        # ── rolling window stats ─────────────────────────────────────────────
        window = 50
        w      = min(window, episode)
        avg50_reward    = float(np.mean(metrics["episode_rewards"][-w:]))
        avg50_crash     = float(np.mean(metrics["crash_rate"][-w:]))
        avg50_arrival   = float(np.mean(metrics["arrival_rate"][-w:]))
        avg50_loss      = float(np.mean(metrics["loss_history"][-w:]))
        avg50_length    = float(np.mean(metrics["episode_lengths"][-w:]))
        avg50_velocity  = float(np.mean(metrics["velocity_history"][-w:]))

        # ── save best model (smoothed over last 50 episodes) ─────────────────
        saved_best = False
        if episode >= window:
            if avg50_reward > best_window:
                best_window = avg50_reward
                agent.save(os.path.join(SAVE_DIR, "best_model.pt"), episode, metrics)
                saved_best = True

        # ── per-episode wandb log ────────────────────────────────────────────
        ep_total_steps = ep_explore_steps + ep_exploit_steps
        ep_log = {
            # ── Paper Figure 4 metrics ────────────────────────────────────────
            # 1. Total reward (paper calls this "total reward")
            "paper/total_reward":      ep_reward,
            # 2. Episode length (paper calls this "length")
            "paper/episode_length":    ep_length,
            # 3. Average velocity (paper calls this "velocity")
            "paper/avg_velocity":      avg_velocity,
            # Rolling 50-ep versions of the paper's three main metrics
            "paper/rolling_reward_50":   avg50_reward,
            "paper/rolling_length_50":   avg50_length,
            "paper/rolling_velocity_50": avg50_velocity,

            # ── Reward decomposition (paper: +1 speed, -5 collision) ──────────
            "reward/speed_reward":       ep_speed_reward,
            "reward/collision_penalty":  ep_collision_penalty,
            "reward/total":              ep_reward,

            # ── Raw episode outcome ───────────────────────────────────────────
            "episode/reward":          ep_reward,
            "episode/length":          ep_length,
            "episode/crashed":         int(crashed),
            "episode/arrived":         int(arrived),
            # Per-episode training diagnostics
            "episode/avg_loss":        avg_loss,
            "episode/epsilon":         agent.epsilon,
            "episode/avg_q_current":   avg_qc,
            "episode/avg_q_target":    avg_qt,
            "episode/avg_td_error":    avg_tde,
            "episode/avg_grad_norm":   avg_gnorm,
            # Explore vs exploit breakdown
            "episode/explore_steps":   ep_explore_steps,
            "episode/exploit_steps":   ep_exploit_steps,
            "episode/exploit_ratio":   ep_exploit_steps / max(ep_total_steps, 1),
            # Per-action counts this episode (all steps)
            **{"episode/action_{}_count".format(a): ep_action_counts[a]
               for a in range(N_ACTIONS)},
            # Per-action counts this episode (exploit steps only)
            **{"episode/exploit_action_{}_count".format(a): ep_exploit_action_counts[a]
               for a in range(N_ACTIONS)},
            # Rolling 50-episode window
            "rolling/reward_50":         avg50_reward,
            "rolling/length_50":         avg50_length,
            "rolling/crash_rate_50":     avg50_crash   * 100,
            "rolling/arrival_rate_50":   avg50_arrival * 100,
            "rolling/avg_loss_50":       avg50_loss,
            # Cumulative collisions
            "rolling/cumulative_collisions": int(np.sum(metrics["crash_rate"])),
            # Infrastructure
            "train/total_steps":       agent.total_steps,
            "train/buffer_fill":       len(agent.buffer),
        }
        if saved_best:
            ep_log["event/best_model_saved"] = avg50_reward

        wandb.log(ep_log, step=global_step)

        # ── dashboard figure every 500 episodes and at the end ───────────────
        if episode % 500 == 0 or episode == NUM_EPISODES:
            log_dashboard_figure(metrics, cumulative_action_counts,
                                 cumulative_exploit_action_counts,
                                 episode, global_step)

        # ── logging ─────────────────────────────────────────────────────────
        if episode % LOG_EVERY == 0 or episode <= 5:
            crash_pct  = avg50_crash   * 100
            arrive_pct = avg50_arrival * 100
            print("  {:>5}  {:>+10.3f}  {:>8,}  {:>6.3f}  {:>8.5f}  {:>6.1f}%  {:>6.1f}%".format(
                episode, avg50_reward, agent.total_steps, agent.epsilon,
                avg_loss, crash_pct, arrive_pct))

        # ── periodic checkpoint ──────────────────────────────────────────────
        if episode % 500 == 0:
            agent.save(
                os.path.join(SAVE_DIR, "checkpoint_ep{}.pt".format(episode)),
                episode, metrics)
            with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    # ── final save ───────────────────────────────────────────────────────────
    agent.save(os.path.join(SAVE_DIR, "final_model.pt"), episode, metrics)
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    w = min(200, NUM_EPISODES)
    last200_crash = float(np.mean(metrics["crash_rate"][-w:]))   * 100
    last200_arr   = float(np.mean(metrics["arrival_rate"][-w:])) * 100
    last200_rew   = float(np.mean(metrics["episode_rewards"][-w:]))

    # ── wandb run summary (shown in the run overview panel) ──────────────────
    last200_vel = float(np.mean(metrics["velocity_history"][-w:]))
    wandb.summary["summary/last200_reward"]       = last200_rew
    wandb.summary["summary/last200_crash_rate"]   = last200_crash
    wandb.summary["summary/last200_arrival_rate"] = last200_arr
    wandb.summary["summary/last200_avg_velocity"] = last200_vel
    wandb.summary["summary/best_window_reward"]   = best_window
    wandb.summary["summary/total_steps"]          = agent.total_steps
    wandb.summary["summary/training_minutes"]     = total_time / 60

    print()
    print("=" * 50)
    print("Training complete in {:.1f} min".format(total_time / 60))
    print("Last 200 episodes:")
    print("  Crash rate   : {:.1f}%".format(last200_crash))
    print("  Arrival rate : {:.1f}%".format(last200_arr))
    print("  Avg reward   : {:+.3f}".format(last200_rew))
    print("  Total steps  : {:,}".format(agent.total_steps))
    print("=" * 50)

    wandb.finish()
    env.close()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(seed=SEED)