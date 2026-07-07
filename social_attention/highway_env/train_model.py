"""
╔══════════════════════════════════════════════════════════════╗
║        DQN TRAINING LOOP — Ego-Attention on highway-v0        ║
╚══════════════════════════════════════════════════════════════╝

Reproduces the rl-agents HighwayEnv "ego_attention" experiment.

Config sources (verified @ master):
  scripts/configs/HighwayEnv/env_obs_attention.json   (env + 7-feature obs)
  scripts/configs/HighwayEnv/agents/DQNAgent/ego_attention.json
      ← ddqn.json (double=True) ← dqn.json (loss l2, lr 5e-4, n_steps 1, tau 6000)
  highway-env HighwayEnv.default_config()             (rewards, action, etc.)

Merged ego_attention hyperparameters:
  gamma 0.99 | batch_size 64 | memory_capacity 15000 | target_update 512 (steps)
  double True | loss "l2" (MSE) | optimizer ADAM lr 5e-4 | n_steps 1
  exploration EpsilonGreedy: tau 6000, temperature 1.0, final_temperature 0.05
  model EgoAttentionNetwork: embeddings [64,64] in=7, attention feature_size 64 heads 2,
        output [64,64] -> 5 actions

RUN WITH:  python train_model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import highway_env
import random
import math
import json
import os
import time
import torch.nn.functional as F
from collections import deque

from model import EgoAttentionNetwork, N_FEATURES, EMBED_DIM, N_ACTIONS, N_VEHICLES


# ─────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  (HighwayEnv ego_attention.json ← ddqn.json ← dqn.json)
# ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Replay buffer ---
BUFFER_SIZE  = 15_000   # memory_capacity
BATCH_SIZE   = 64       # batch_size  (dqn.json=32, overridden to 64 by ego_attention.json)
MIN_BUFFER   = 1_000    # warmup before learning (rl-agents starts at batch_size)

# --- Training ---
GAMMA         = 0.99    # ego_attention.json gamma (dqn.json=0.8, overridden)
LEARNING_RATE = 5e-4    # dqn.json optimizer lr
TARGET_UPDATE = 512     # ego_attention.json target_update — in STEPS

# Step-based budget. Exploration anneals over STEPS (tau), so we train on a step
# budget rather than a fixed episode count. highway episodes run up to 80 steps.
# With tau=6000, eps≈0.05 by ~30k steps; 60k steps gives a long exploitation phase.
TARGET_STEPS  = 60_000
MAX_EPISODES  = 100_000  # safety cap on the episode loop (step budget ends first)

# duration 40 s × policy_frequency 2 Hz = 80 policy steps per episode
MAX_STEPS     = 80

# --- Seed ---
SIM_SEED = 42

# --- Epsilon-greedy (dqn.json exploration; step-based exponential decay) ---
# eps(t) = final + (temp - final) * exp(-t / tau),  t = cumulative env steps
EPS_START = 1.0          # temperature
EPS_END   = 0.05         # final_temperature
EPS_TAU   = 6000         # tau (HighwayEnv dqn.json)

SAVE_DIR = "checkpoints"


# ─────────────────────────────────────────────────────────────────
# BLOCK 1 — REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Stores (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),       # [B, N, F]
            torch.LongTensor(actions).to(DEVICE),                  # [B]
            torch.FloatTensor(rewards).to(DEVICE),                 # [B]
            torch.FloatTensor(np.array(next_states)).to(DEVICE),  # [B, N, F]
            torch.FloatTensor(dones).to(DEVICE),                   # [B]
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────
# BLOCK 2 — DQN AGENT  (Double DQN, matches rl-agents pytorch.py)
# ─────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self):
        self.epsilon     = EPS_START
        self.total_steps = 0      # cumulative env steps — drives epsilon & target update

        self.policy_net = EgoAttentionNetwork().to(DEVICE)   # Xavier init in __init__
        self.target_net = EgoAttentionNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)
            return self.policy_net(obs_t)[0].argmax().item()

    def learn(self):
        if len(self.buffer) < MIN_BUFFER:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.nan_to_num(states,      nan=0.0, posinf=0.0, neginf=0.0)
        next_states = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        rewards     = torch.nan_to_num(rewards,     nan=0.0, posinf=0.0, neginf=0.0)

        # Current Q(s, a)
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double-DQN target: policy_net selects, target_net evaluates
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            q_next = self.target_net(next_states).gather(
                1, best_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        # rl-agents DQNAgent loss_function "l2" -> MSE
        loss = F.mse_loss(q_current, q_target)
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return None

        self.optimizer.zero_grad()
        loss.backward()
        # rl-agents step_optimizer: param.grad.data.clamp_(-1, 1)
        for p in self.policy_net.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.total_steps / EPS_TAU)

    def save_checkpoint(self, path, episode, metrics):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "episode":         episode,
            "model_state":     self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon":         self.epsilon,
            "metrics":         metrics,
        }, path)


# ─────────────────────────────────────────────────────────────────
# BLOCK 3 — ENVIRONMENT FACTORY  (highway-v0, env_obs_attention.json)
# ─────────────────────────────────────────────────────────────────

def make_env():
    """
    highway-v0 configured per rl-agents env_obs_attention.json.

    Overrides: lanes_count 3, vehicles_count 15, policy_frequency 2, duration 40.
    Observation: Kinematics, 15 vehicles, 7 features, absolute=False (relative),
                 order/normalize left at highway-env defaults (sorted / True).
    Action: highway-v0 default DiscreteMetaAction -> 5 actions (lane changes enabled).
    Reward: highway-v0 defaults (collision -1, high_speed 0.4, right_lane 0.1,
            normalize_reward True). No destination/arrival — success = surviving.
    """
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure({
        "lanes_count":      3,
        "vehicles_count":   15,
        "policy_frequency": 2,
        "duration":         40,   # seconds; 40 * 2Hz = 80 policy steps = MAX_STEPS
        "observation": {
            "type":           "Kinematics",
            "vehicles_count": N_VEHICLES,   # 15
            "features":       ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute":       False,        # env_obs_attention.json: relative coords
        },
        # action left at highway-v0 default DiscreteMetaAction (5 actions).
    })
    return env


# ─────────────────────────────────────────────────────────────────
# BLOCK 4 — TRAINING LOOP  (step-budget based)
# ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 65)
    print("  TRAINING THE SOCIAL ATTENTION DQN")
    print("  Task: highway-v0  (rl-agents ego_attention config)")
    print("=" * 65)
    print(f"\n  Device       : {DEVICE}")
    print(f"  Step budget  : {TARGET_STEPS:,}  (episode cap {MAX_EPISODES:,})")
    print(f"  Steps/episode: {MAX_STEPS}  (duration 40s x 2Hz)")
    print(f"  Actions      : {N_ACTIONS}  (LANE_LEFT/IDLE/LANE_RIGHT/FASTER/SLOWER)")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Buffer size  : {BUFFER_SIZE:,}")
    print(f"  Gamma        : {GAMMA}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epsilon      : {EPS_START} -> {EPS_END}  (tau={EPS_TAU} steps, exponential)")
    print(f"  Target update: every {TARGET_UPDATE} steps")
    print(f"\n  Saved files -> {SAVE_DIR}/")

    env   = make_env()
    agent = DQNAgent()
    os.makedirs(SAVE_DIR, exist_ok=True)

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "collision_rate":  [],
        "epsilon_history": [],
        "loss_history":    [],
    }
    best_reward = float("-inf")

    print(f"\n  {'Ep':>6}  {'AvgRew(10)':>11}  {'Steps':>6}  {'eps':>6}  {'Loss':>8}  {'Crash%':>7}  {'TotSteps':>9}")
    print("  " + "─" * 72)

    t_start = time.time()
    episode = 0

    while agent.total_steps < TARGET_STEPS and episode < MAX_EPISODES:
        episode += 1
        obs, _   = env.reset(seed=SIM_SEED + episode)
        ep_reward = 0.0
        ep_losses = []
        crashed   = False
        step      = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.total_steps += 1
            agent.update_epsilon()

            if agent.total_steps % TARGET_UPDATE == 0:
                agent.update_target_network()

            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            obs = next_obs
            if info.get("crashed", False):
                crashed = True

            if done or agent.total_steps >= TARGET_STEPS:
                break

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_lengths"].append(step + 1)
        metrics["collision_rate"].append(1 if crashed else 0)
        metrics["epsilon_history"].append(agent.epsilon)
        metrics["loss_history"].append(avg_loss)

        if episode % 50 == 0 or episode <= 5:
            avg_reward = float(np.mean(metrics["episode_rewards"][-10:]))
            crash_pct  = float(np.mean(metrics["collision_rate"][-10:])) * 100
            print(f"  {episode:>6}  {avg_reward:>+11.4f}  {step+1:>6}  {agent.epsilon:>6.3f}  "
                  f"{avg_loss:>8.5f}  {crash_pct:>6.1f}%  {agent.total_steps:>9,}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save_checkpoint(f"{SAVE_DIR}/best_model.pt", episode, metrics)

        if episode % 500 == 0:
            agent.save_checkpoint(f"{SAVE_DIR}/checkpoint_ep{episode}.pt", episode, metrics)
            with open(f"{SAVE_DIR}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            recent_crash = float(np.mean(metrics["collision_rate"][-50:])) * 100
            recent_rew   = float(np.mean(metrics["episode_rewards"][-50:]))
            elapsed      = time.time() - t_start
            print(f"\n  ── CHECKPOINT ep {episode} ───────────────────────────")
            print(f"     Best reward : {best_reward:.4f}")
            print(f"     Avg rew/50  : {recent_rew:.4f}")
            print(f"     Crash%/50   : {recent_crash:.1f}%")
            print(f"     Total steps : {agent.total_steps:,} / {TARGET_STEPS:,}")
            print(f"     Elapsed     : {elapsed/60:.1f} min\n")

    print(f"\n  Reached budget: {agent.total_steps:,} steps at episode {episode} "
          f"(final epsilon {agent.epsilon:.3f}).")

    agent.save_checkpoint(f"{SAVE_DIR}/final_model.pt", episode, metrics)
    with open(f"{SAVE_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE!")
    print("=" * 65)
    print(f"  Total time          : {total_time/60:.1f} min")
    print(f"  Best episode reward : {best_reward:.4f}")
    print(f"  Final epsilon       : {agent.epsilon:.4f}")
    print(f"  Crash rate (last 50): {np.mean(metrics['collision_rate'][-50:])*100:.1f}%")
    print(f"  Avg reward (last 50): {np.mean(metrics['episode_rewards'][-50:]):.4f}")
    print(f"\n  Saved: {SAVE_DIR}/best_model.pt, final_model.pt, metrics.json")
    print(f"  -> Run: python inference.py")

    env.close()
    return agent, metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    agent, metrics = train()