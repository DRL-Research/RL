"""
VN-MADDPG for Unsignalized Intersection Navigation (With Logging)
=================================================================
Implementation of VN-MADDPG with granular logging inside algorithm stages.

This file contains:
1. Scenario Management
2. VN-MADDPG Networks (Actor + Critic)
3. Prioritized Replay Buffer (PER)
4. VN-MADDPG Agent
5. Environment Wrapper
6. Training Loop (with stage-based logging)
7. Main Execution

Adapted by: Gemini
Date: 2025
"""
import csv  #
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import gymnasium as gym
import highway_env
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json
from copy import deepcopy
import warnings
import math
import logging  # Added logging module

from experiment.experiment_config import Experiment

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Create a custom logger
logger = logging.getLogger("VN_MADDPG")
logger.setLevel(logging.INFO)  # Change to DEBUG for extremely verbose output

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('vn_maddpg_training.log', mode='w')

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ============================================================================
# SECTION 1: SCENARIO MANAGEMENT
# ============================================================================

def rotate_lane_id(lane_id, rotation):
    """Rotate a lane ID clockwise by rotation steps (1-3)"""
    if len(lane_id) < 2 or not lane_id[-1].isdigit():
        return lane_id
    prefix = lane_id[:-1]
    direction = int(lane_id[-1])
    new_direction = (direction + rotation) % 4
    return prefix + str(new_direction)

def rotate_scenario_clockwise(scenario, rotation):
    """Rotate an entire scenario clockwise by rotation steps"""
    rotated_scenario = []
    for lane_key, destination, offset in scenario:
        if isinstance(lane_key, tuple):
            rotated_lane = tuple(
                rotate_lane_id(part, rotation) if isinstance(part, str) else part
                for part in lane_key
            )
        else:
            rotated_lane = rotate_lane_id(lane_key, rotation)
        rotated_dest = rotate_lane_id(destination, rotation)
        rotated_scenario.append((rotated_lane, rotated_dest, offset))
    return rotated_scenario

class ScenarioManager:
    """Manages the exact 100 scenarios (25 base × 4 rotations)"""

    def __init__(self):
        self.all_scenarios = self._generate_all_scenarios()
        logger.info(f"[ScenarioManager] Generated {len(self.all_scenarios)} scenarios")

    def _get_base_scenarios(self):
        """Returns the exact 25 base scenarios from the original code."""
        base_scenarios = [
            # Scenario 0: Heavy north-south flow
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", 0)],
             "static": [(('o0', 'ir0', 0), "o2", -50), (('o2', 'ir2', 0), "o0", -35), (('o0', 'ir0', 0), "o1", -70)]},
            # Scenario 1: All cars turning right
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -25), (('o3', 'ir3', 0), "o0", -35), (('o0', 'ir0', 0), "o1", -15)]},
            # Scenario 2: Crossing patterns
            {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o2', 'ir2', 0), "o1", 0)],
             "static": [(('o1', 'ir1', 0), "o0", -20), (('o3', 'ir3', 0), "o2", -30), (('o0', 'ir0', 0), "o2", -40)]},
            # Scenario 3: East-west corridor
            {"agents": [(('o1', 'ir1', 0), "o3", 0), (('o3', 'ir3', 0), "o1", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -25), (('o2', 'ir2', 0), "o1", -35)]},
            # Scenario 4: Complex multi-direction
            {"agents": [(('o2', 'ir2', 0), "o0", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o0', 'ir0', 0), "o3", -45), (('o1', 'ir1', 0), "o2", -30), (('o2', 'ir2', 0), "o0", -20)]},
            # Scenario 5: Same origin dispersal
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o0', 'ir0', 0), "o1", 20)],
             "static": [(('o0', 'ir0', 0), "o2", -50), (('o1', 'ir1', 0), "o3", -30), (('o2', 'ir2', 0), "o0", -25)]},
            # Scenario 6: Convergence to same destination
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o3', 'ir3', 0), "o2", -40), (('o2', 'ir2', 0), "o1", -30), (('o0', 'ir0', 0), "o3", -50)]},
            # Scenario 7: Minimum conflict scenario
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o2', 'ir2', 0), "o0", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -40), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -60)]},
            # Scenario 8: Rush hour challenge
            {"agents": [(('o3', 'ir3', 0), "o1", 0), (('o0', 'ir0', 0), "o2", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -20), (('o2', 'ir2', 0), "o0", -15), (('o3', 'ir3', 0), "o2", -45)]},
            # Scenario 9: Complex intersection
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o3', 'ir3', 0), "o0", 0)],
             "static": [(('o1', 'ir1', 0), "o2", -25), (('o2', 'ir2', 0), "o3", -30), (('o0', 'ir0', 0), "o3", -60)]},
            # Scenario 10: Parallel lanes
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o0', 'ir0', 0), "o2", -30)],
             "static": [(('o2', 'ir2', 0), "o0", -60), (('o1', 'ir1', 0), "o3", -55), (('o3', 'ir3', 0), "o1", -65)]},
            # Scenario 11: Cross traffic
            {"agents": [(('o1', 'ir1', 0), "o3", 0), (('o2', 'ir2', 0), "o0", 0)],
             "static": [(('o0', 'ir0', 0), "o2", -30), (('o3', 'ir3', 0), "o1", -25), (('o1', 'ir1', 0), "o0", -50)]},
            # Scenario 12: Left turn conflict
            {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o1', 'ir1', 0), "o0", 0)],
             "static": [(('o2', 'ir2', 0), "o1", -35), (('o3', 'ir3', 0), "o2", -40), (('o0', 'ir0', 0), "o1", -55)]},
            # Scenario 13: Right turn priority
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o2', 'ir2', 0), "o3", 0)],
             "static": [(('o1', 'ir1', 0), "o2", -50), (('o3', 'ir3', 0), "o0", -55), (('o0', 'ir0', 0), "o2", -55)]},
            # Scenario 14: Staggered timing
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", -40)],
             "static": [(('o2', 'ir2', 0), "o0", -50), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -20)]},
            # Scenario 15: Opposite directions
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o2', 'ir2', 0), "o0", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -10)]},
            # Scenario 16: Diagonal crossing
            {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o1", -25), (('o3', 'ir3', 0), "o0", -30), (('o0', 'ir0', 0), "o2", -35)]},
            # Scenario 17: Sequential turns
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -50), (('o3', 'ir3', 0), "o0", -55), (('o0', 'ir0', 0), "o3", -65)]},
            # Scenario 18: Wide spacing
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", 50)],
             "static": [(('o2', 'ir2', 0), "o0", -60), (('o3', 'ir3', 0), "o1", -45), (('o0', 'ir0', 0), "o1", -75)]},
            # Scenario 19: Mixed patterns
            {"agents": [(('o2', 'ir2', 0), "o3", 0), (('o3', 'ir3', 0), "o2", 15)],
             "static": [(('o0', 'ir0', 0), "o1", -30), (('o1', 'ir1', 0), "o0", -35), (('o2', 'ir2', 0), "o0", -50)]},
            # Scenario 20: Heavy traffic mix
            {"agents": [(('o1', 'ir1', 0), "o2", 0), (('o0', 'ir0', 0), "o1", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -25), (('o3', 'ir3', 0), "o0", -30), (('o1', 'ir1', 0), "o3", -40)]},
            # Scenario 21: Priority conflict
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o3', 'ir3', 0), "o1", 0)],
             "static": [(('o1', 'ir1', 0), "o0", -35), (('o2', 'ir2', 0), "o3", -40), (('o0', 'ir0', 0), "o3", -50)]},
            # Scenario 22: Roundabout-like flow
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -30), (('o3', 'ir3', 0), "o0", -35), (('o0', 'ir0', 0), "o2", -60)]},
            # Scenario 23: High-speed challenge
            {"agents": [(('o2', 'ir2', 0), "o0", 0), (('o1', 'ir1', 0), "o3", 0)],
             "static": [(('o0', 'ir0', 0), "o2", -20), (('o3', 'ir3', 0), "o1", -25), (('o2', 'ir2', 0), "o1", -45)]},
            # Scenario 24: Final complex scenario
            {"agents": [(('o3', 'ir3', 0), "o2", 0), (('o0', 'ir0', 0), "o3", 0)],
             "static": [(('o1', 'ir1', 0), "o0", -30), (('o2', 'ir2', 0), "o1", -35), (('o3', 'ir3', 0), "o1", -55)]}
        ]
        return base_scenarios

    def _generate_all_scenarios(self):
        """Generate all 100 scenarios (25 base × 4 rotations)"""
        base_scenarios = self._get_base_scenarios()
        all_scenarios = []
        for base_scenario in base_scenarios:
            all_scenarios.append(base_scenario)
            for rotation in [1, 2, 3]:
                rotated_agents = rotate_scenario_clockwise(base_scenario["agents"], rotation)
                rotated_static = rotate_scenario_clockwise(base_scenario["static"], rotation)
                all_scenarios.append({"agents": rotated_agents, "static": rotated_static})
        return all_scenarios

    def get_scenario(self, idx):
        return self.all_scenarios[idx]

    def get_random_scenario(self):
        return random.choice(self.all_scenarios)


# ============================================================================
# SECTION 2: VN-MADDPG NETWORKS
# ============================================================================

def gumbel_sigmoid(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = F.softmax(gumbels, dim=-1)

    if hard:
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    return y_soft

class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs, temperature=1.0, hard=False):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        action_probs = gumbel_sigmoid(logits, temperature, hard)
        return action_probs, logits

class Critic(nn.Module):
    def __init__(self, state_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, all_actions):
        x = torch.cat([state, all_actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# ============================================================================
# SECTION 3: PRIORITIZED REPLAY BUFFER
# ============================================================================

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 0.01

    def add(self, obs, state, actions, rewards, next_obs, next_state, dones):
        transition = (obs, state, actions, rewards, next_obs, next_state, dones)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            if data is None:
                idx = random.randint(self.capacity - 1, self.capacity + self.tree.n_entries - 2)
                p = self.tree.tree[idx]
                data = self.tree.data[idx - self.capacity + 1]

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        obs_batch, state_batch, act_batch, rew_batch, next_obs_batch, next_state_batch, done_batch = zip(*batch)
        return (obs_batch, state_batch, act_batch, rew_batch, next_obs_batch, next_state_batch, done_batch, idxs, is_weights)

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# ============================================================================
# SECTION 4: VN-MADDPG AGENT
# ============================================================================

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

class VN_MADDPG:
    """VN-MADDPG Agent with Logging Hooks"""
    def __init__(
            self,
            n_agents=2,
            obs_dim=8,
            n_actions=2,
            state_dim=20,
            hidden_dim=64,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.95,
            tau=0.01,
            buffer_capacity=100000,
            batch_size=64,
            device='cpu'
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        logger.info(f"[VN_MADDPG] Initializing {n_agents} agents on {device}")

        # Initialize actors and critics
        self.actors = [Actor(obs_dim, n_actions, hidden_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dim, n_actions, hidden_dim).to(device) for _ in range(n_agents)]

        total_action_dim = n_agents * n_actions
        self.critics = [Critic(state_dim, total_action_dim, hidden_dim).to(device) for _ in range(n_agents)]
        self.target_critics = [Critic(state_dim, total_action_dim, hidden_dim).to(device) for _ in range(n_agents)]

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=lr_critic) for c in self.critics]

        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.memory = PrioritizedReplayBuffer(capacity=buffer_capacity)
        self.noise = [OUNoise(n_actions, scale=1.0) for _ in range(n_agents)]
        self.noise_scale = 1.0
        self.training_steps = 0

    def update_noise(self, current_episode, max_episodes):
        prev_scale = self.noise_scale
        self.noise_scale = max(0.05, 1.0 - (current_episode / max_episodes))
        for n in self.noise:
            n.scale = self.noise_scale
        if current_episode % 100 == 0:
            logger.info(f"[Stage: Noise Update] Decay {prev_scale:.3f} -> {self.noise_scale:.3f}")

    def select_action(self, obs, explore=True):
        """Action Selection Stage with Logging"""
        actions = []
        action_indices = []

        with torch.no_grad():
            for i, agent_obs in enumerate(obs):
                obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                _, logits = self.actors[i](obs_tensor)
                logits = logits.cpu().detach().numpy()[0]

                if explore:
                    noise = self.noise[i].noise()
                    logits += noise

                probs = np.exp(logits) / np.sum(np.exp(logits))

                if explore:
                    action = np.random.choice(self.n_actions, p=probs)
                else:
                    action = np.argmax(probs)

                actions.append(action)

        # Verbose logging (Debug level)
        logger.debug(f"[Stage: Select Action] Actions: {actions} | Explore: {explore}")
        return actions

    def update(self, beta=0.4):
        """Learning Stage with Internal Logging"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample from PER
        samples = self.memory.sample(self.batch_size, beta)
        obs_batch, state_batch, act_batch, rew_batch, next_obs_batch, next_state_batch, done_batch, idxs, is_weights = samples

        obs = [torch.FloatTensor(np.stack([o[i] for o in obs_batch])).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.FloatTensor(np.stack([o[i] for o in next_obs_batch])).to(self.device) for i in range(self.n_agents)]
        state = torch.FloatTensor(np.stack(state_batch)).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state_batch)).to(self.device)
        actions = torch.LongTensor(np.stack(act_batch)).to(self.device)
        rewards = torch.FloatTensor(np.stack(rew_batch)).to(self.device)
        dones = torch.FloatTensor(np.stack(done_batch)).to(self.device)
        weights = torch.FloatTensor(is_weights).to(self.device)

        actions_onehot = []
        for i in range(self.n_agents):
            a = F.one_hot(actions[:, i], num_classes=self.n_actions).float()
            actions_onehot.append(a)
        all_actions = torch.cat(actions_onehot, dim=1)

        total_critic_loss = 0
        total_actor_loss = 0
        new_priorities = np.zeros(self.batch_size)

        # Iterate over agents
        for agent_i in range(self.n_agents):
            # --- CRITIC UPDATE STAGE ---
            with torch.no_grad():
                next_actions_onehot = []
                for j in range(self.n_agents):
                    next_act_prob, _ = self.target_actors[j](next_obs[j], hard=False)
                    next_actions_onehot.append(next_act_prob)
                all_next_actions = torch.cat(next_actions_onehot, dim=1)
                target_q = self.target_critics[agent_i](next_state, all_next_actions)
                y = rewards[:, agent_i].unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

            current_q = self.critics[agent_i](state, all_actions)
            td_error = torch.abs(y - current_q).detach().cpu().numpy()
            new_priorities += td_error.flatten()

            critic_loss = (weights.unsqueeze(1) * F.mse_loss(current_q, y, reduction='none')).mean()

            self.critic_optimizers[agent_i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_i].parameters(), 0.5)
            self.critic_optimizers[agent_i].step()

            total_critic_loss += critic_loss.item()

            # --- ACTOR UPDATE STAGE ---
            curr_act_probs, _ = self.actors[agent_i](obs[agent_i], hard=False)
            all_pol_actions = []
            for j in range(self.n_agents):
                if j == agent_i:
                    all_pol_actions.append(curr_act_probs)
                else:
                    other_act, _ = self.actors[j](obs[j], hard=False)
                    all_pol_actions.append(other_act.detach())
            all_pol_actions_concat = torch.cat(all_pol_actions, dim=1)

            actor_loss = -self.critics[agent_i](state, all_pol_actions_concat).mean()

            self.actor_optimizers[agent_i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_i].parameters(), 0.5)
            self.actor_optimizers[agent_i].step()

            total_actor_loss += actor_loss.item()

            # Log internal update details occasionally (Debug level)
            if self.training_steps % 100 == 0:
                logger.debug(f"[Stage: Update] Agent {agent_i} | Critic Loss: {critic_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f}")

        # --- SOFT UPDATE STAGE ---
        for i in range(self.n_agents):
            self._soft_update(self.actors[i], self.target_actors[i], self.tau)
            self._soft_update(self.critics[i], self.target_critics[i], self.tau)

        new_priorities /= self.n_agents
        self.memory.update_priorities(idxs, new_priorities)

        self.training_steps += 1

        return {
            "critic_loss": total_critic_loss / self.n_agents,
            "actor_loss": total_actor_loss / self.n_agents,
            "noise_scale": self.noise_scale
        }

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'training_steps': self.training_steps,
            'noise_scale': self.noise_scale
        }
        for i in range(self.n_agents):
            checkpoint[f'actor_{i}'] = self.actors[i].state_dict()
            checkpoint[f'critic_{i}'] = self.critics[i].state_dict()
            checkpoint[f'actor_opt_{i}'] = self.actor_optimizers[i].state_dict()
            checkpoint[f'critic_opt_{i}'] = self.critic_optimizers[i].state_dict()
        torch.save(checkpoint, path)
        logger.info(f"[IO] Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.training_steps = checkpoint['training_steps']
        self.noise_scale = checkpoint['noise_scale']
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint[f'actor_{i}'])
            self.critics[i].load_state_dict(checkpoint[f'critic_{i}'])
            self.target_actors[i].load_state_dict(checkpoint[f'actor_{i}'])
            self.target_critics[i].load_state_dict(checkpoint[f'critic_{i}'])
            self.actor_optimizers[i].load_state_dict(checkpoint[f'actor_opt_{i}'])
            self.critic_optimizers[i].load_state_dict(checkpoint[f'critic_opt_{i}'])
        logger.info(f"[IO] Model loaded from {path}")


# ============================================================================
# SECTION 5: ENVIRONMENT WRAPPER
# ============================================================================

class IntersectionWrapper:
    """Wrapper for highway-env intersection with logging"""

    # CHANGE: Added render_mode parameter
    def __init__(self, scenario_manager, n_agents=2, render_mode=None):
        self.scenario_manager = scenario_manager
        self.n_agents = n_agents

        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20],
                },
                "absolute": True, "flatten": False, "observe_intentions": False, "vehicles_count": 10,
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": False, "longitudinal": True, "target_speeds": [5, 10],
                },
            },
            "collision_reward": Experiment.COLLISION_REWARD,
            "arrived_reward": Experiment.REACHED_TARGET_REWARD,  # reward for arriving at the destination
            "normalize_reward": False,
            "starvation_reward": Experiment.STARVATION_REWARD,  # reward for every step taken
            "offroad_terminal": False,
            "duration": Experiment.EPISODE_MAX_TIME,
            "initial_vehicle_count": Experiment.CARS_AMOUNT,
            "spawn_probability": Experiment.SPAWN_PROBABILITY,
            "screen_width": Experiment.SCREEN_WIDTH,
            "screen_height": Experiment.SCREEN_HEIGHT,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5,
            "show_trajectories": True,  # Helpful for visualization
            "render_agent": True
        }

        # CHANGE: Passed render_mode to gym.make
        self.env = gym.make('intersection-v0', render_mode=render_mode, config=config)
        logger.info(f"[Env] IntersectionWrapper Initialized (Mode: {render_mode})")

    def reset(self, scenario_idx=None):
        obs_raw, info = self.env.reset()
        if scenario_idx is None:
            scenario = self.scenario_manager.get_random_scenario()
        else:
            scenario = self.scenario_manager.get_scenario(scenario_idx)

        # Handling observation reshaping
        if isinstance(obs_raw, np.ndarray) and len(obs_raw.shape) == 1:
            n_features = 4
            n_vehicles = len(obs_raw) // n_features
            obs_raw = obs_raw.reshape(n_vehicles, n_features)

        obs = []
        for i in range(self.n_agents):
            if i < len(obs_raw):
                obs.append(obs_raw[i].flatten())
            else:
                obs.append(np.zeros(4))
        state = obs_raw.flatten()
        return obs, state

    def step(self, actions):
        if isinstance(actions, list):
            actions = tuple(actions)
        obs_raw, reward, done, truncated, info = self.env.step(actions)

        if isinstance(obs_raw, np.ndarray) and len(obs_raw.shape) == 1:
            n_features = 4
            n_vehicles = len(obs_raw) // n_features
            obs_raw = obs_raw.reshape(n_vehicles, n_features)

        next_obs = []
        for i in range(self.n_agents):
            if i < len(obs_raw):
                next_obs.append(obs_raw[i].flatten())
            else:
                next_obs.append(np.zeros(4))
        next_state = obs_raw.flatten()
        return next_obs, next_state, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================================
# SECTION 6: TRAINING LOOP
# ============================================================================

def train_vn_maddpg(
        agent,
        env,
        n_episodes=5000,
        max_steps=100,
        update_freq=1,
        save_interval=500,
        save_dir='results'
):
    """Main training loop for VN-MADDPG with Stage Logging and CSV Output."""
    results = {
        'rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'noise_scales': [],
        'success_rates': []
    }

    logger.info("=" * 40)
    logger.info("STARTING VN-MADDPG TRAINING")
    logger.info("=" * 40)

    # ---------------------------------------------------------
    # CSV LOGGING SETUP
    # ---------------------------------------------------------
    csv_file_path = os.path.join(save_dir, 'training_log.csv')

    # Initialize file with headers
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['episode', 'result', 'reward', 'steps'])

    logger.info(f"[CSV] Logging episode data to: {csv_file_path}")
    # ---------------------------------------------------------

    start_time = datetime.now()

    for episode in range(n_episodes):
        # Stage: Noise Update
        agent.update_noise(episode, n_episodes)

        beta = 0.4 + (1.0 - 0.4) * (episode / n_episodes)

        obs, state = env.reset()
        episode_reward = 0
        crashed = False
        step_count = 0

        for step in range(max_steps):
            step_count = step + 1

            # Stage: Action Selection
            actions = agent.select_action(obs, explore=True)

            # Stage: Environment Step
            next_obs, next_state, reward, done, truncated, info = env.step(actions)

            # Stage: Memory Storage
            rewards_list = [reward] * agent.n_agents
            agent.memory.add(obs, state, actions, rewards_list, next_obs, next_state, done)

            # Stage: Network Update
            loss_info = None
            if len(agent.memory) > agent.batch_size and step % update_freq == 0:
                loss_info = agent.update(beta)

            obs = next_obs
            state = next_state
            episode_reward += reward

            if info.get('crashed', False):
                crashed = True

            if done or truncated:
                logger.debug(f"[Episode {episode}] Finished at step {step}. Total Reward: {episode_reward}")
                break

        # ---------------------------------------------------------
        # DETERMINE RESULT & LOG TO CSV
        # ---------------------------------------------------------
        if crashed:
            outcome = "Collision"
        elif done and not crashed:
            outcome = "Success"
        else:
            outcome = "Timeout"  # Truncated or max steps reached

        # Append row to CSV
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # episode, result, reward, steps
            writer.writerow([episode + 1, outcome, round(episode_reward, 2), step_count])
        # ---------------------------------------------------------

        # Logging aggregation
        results['rewards'].append(episode_reward)
        results['success_rates'].append(0 if crashed else 1)
        results['noise_scales'].append(agent.noise_scale)
        if loss_info:
            results['critic_losses'].append(loss_info['critic_loss'])
            results['actor_losses'].append(loss_info['actor_loss'])

        # High-level logging
        if (episode + 1) % 100 == 0:
            avg_rew = np.mean(results['rewards'][-100:])
            avg_suc = np.mean(results['success_rates'][-100:])
            elapsed = datetime.now() - start_time
            logger.info(f"Episode {episode + 1}/{n_episodes} | Rew: {avg_rew:.1f} | "
                        f"Success: {avg_suc:.1%} | Noise: {agent.noise_scale:.2f} | Time: {elapsed}")

        # Checkpoint Saving
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(save_dir, f'vn_maddpg_ep{episode + 1}.pt'))

    return agent, results

# ============================================================================
# SECTION 7: EVALUATION & VISUALIZATION
# ============================================================================

def evaluate(agent, env, n_episodes=100, render=False):
    """Evaluate trained agent on scenarios"""
    logger.info("=" * 40)
    logger.info(f"EVALUATION START ({n_episodes} episodes)")
    logger.info("=" * 40)

    success_count = 0
    total_rewards = []

    for i in range(n_episodes):
        obs, state = env.reset(scenario_idx=i)
        ep_reward = 0
        done = False
        truncated = False
        crashed = False

        while not done and not truncated:
            actions = agent.select_action(obs, explore=False)
            obs, state, reward, done, truncated, info = env.step(actions)
            ep_reward += reward
            if info.get('crashed', False):
                crashed = True
            if render:
                env.render()

        total_rewards.append(ep_reward)
        if not crashed:
            success_count += 1

        if (i+1) % 10 == 0:
            logger.info(f"[Eval] Scenarios {i+1}/{n_episodes} complete. Current Success Rate: {success_count/(i+1):.2%}")

    success_rate = success_count / n_episodes
    avg_reward = np.mean(total_rewards)

    logger.info(f"Evaluation Results: Success Rate: {success_rate:.2%} | Avg Reward: {avg_reward:.2f}")

    return {'success_rate': success_rate, 'avg_reward': avg_reward}

def save_results(train_results, eval_results, save_dir):
    """Save plots and data"""
    os.makedirs(save_dir, exist_ok=True)

    np.savez(os.path.join(save_dir, 'results.npz'),
             rewards=train_results['rewards'],
             critic_losses=train_results['critic_losses'],
             actor_losses=train_results['actor_losses'])

    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(train_results['rewards'], np.ones(100)/100, mode='valid'))
    plt.title("Training Reward (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, 'reward_curve.png'))
    plt.close()

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)

    logger.info(f"Results and plots saved to {save_dir}")


# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/vn_maddpg_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(save_dir, 'experiment.log'))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    scenario_manager = ScenarioManager()
    n_agents = 2

    # ==========================================
    # PHASE 1: TRAINING (No Render - Fast)
    # ==========================================
    logger.info("Initializing Training Environment (Headless)...")
    env = IntersectionWrapper(scenario_manager, n_agents=n_agents, render_mode=None)

    agent = VN_MADDPG(
        n_agents=n_agents,
        obs_dim=4,
        n_actions=2,
        state_dim=40,
        hidden_dim=64,
        lr_actor=1e-3,
        lr_critic=1e-2,
        buffer_capacity=50000,
        batch_size=256,
        device=device
    )

    # Train (Using a smaller number for testing seeing the render quickly)
    agent, train_results = train_vn_maddpg(
        agent, env,
        n_episodes=900,
        save_dir=save_dir
    )

    # Close the training environment to free resources
    env.close()

    # ==========================================
    # PHASE 2: VISUALIZATION (Render - Human)
    # ==========================================
    logger.info("Initializing Evaluation Environment (Visual)...")

    visual_env = IntersectionWrapper(scenario_manager, n_agents=n_agents, render_mode='rgb_array')

    # Evaluate with rendering enabled
    # Note: select_action(explore=False) uses the learned policy
    eval_results = evaluate(agent, visual_env, n_episodes=100, render=True)

    save_results(train_results, eval_results, save_dir)
    visual_env.close()


if __name__ == "__main__":
    main()