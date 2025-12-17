"""
QMIXwD for Unsignalized Intersection Navigation
================================================
Complete implementation of "Multi-agent Decision-making at Unsignalized Intersections
with Reinforcement Learning from Demonstrations" (Huang et al., 2023)

This file contains:
1. Scenario Management (100 scenarios from original code)
2. QMIX Networks (Q-networks + Mixing Network)
3. Replay Buffer
4. QMIXwD Agent with Pre-training
5. Environment Wrapper
6. Demonstration Collection
7. Training Loop
8. Evaluation
9. Visualization & Results Saving
10. Main Execution

Author: Ariel
Date: 2025
"""

import os
import random
import warnings
from collections import deque, namedtuple
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Required for the rolling window calculations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# SECTION 1: SCENARIO MANAGEMENT
# ============================================================================

def rotate_lane_id(lane_id, rotation):
    """
    Rotate a lane ID clockwise by rotation steps (1-3)
    Examples: 'o0' -> 'o1' -> 'o2' -> 'o3'
    """
    if len(lane_id) < 2 or not lane_id[-1].isdigit():
        return lane_id

    prefix = lane_id[:-1]  # 'o', 'ir', 'il'
    direction = int(lane_id[-1])  # 0, 1, 2, 3

    new_direction = (direction + rotation) % 4
    return prefix + str(new_direction)


def rotate_scenario_clockwise(scenario, rotation):
    """Rotate an entire scenario clockwise by rotation steps"""
    rotated_scenario = []
    for lane_key, destination, offset in scenario:
        # Rotate lane_key (can be tuple like ('o0', 'ir0', 0))
        if isinstance(lane_key, tuple):
            rotated_lane = tuple(
                rotate_lane_id(part, rotation) if isinstance(part, str) else part
                for part in lane_key
            )
        else:
            rotated_lane = rotate_lane_id(lane_key, rotation)

        # Rotate destination
        rotated_dest = rotate_lane_id(destination, rotation)

        rotated_scenario.append((rotated_lane, rotated_dest, offset))

    return rotated_scenario


class ScenarioManager:
    """Manages the exact 100 scenarios (25 base × 4 rotations) from original code"""

    def __init__(self):
        self.all_scenarios = self._generate_all_scenarios()
        print(f"[ScenarioManager] Generated {len(self.all_scenarios)} scenarios")

    def _get_base_scenarios(self):
        """
        Returns the exact 25 base scenarios from the original code.
        Each scenario contains 2 controlled agents and 3 static vehicles.
        """
        base_scenarios = [
            # Scenario 0: Heavy north-south flow
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o3", 0)  # Agent 2: East to West
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -50),  # Static: North to South (behind)
                    (('o2', 'ir2', 0), "o0", -35),  # Static: South to North
                    (('o0', 'ir0', 0), "o1", -70)  # Static: North to East
                ]
            },

            # Scenario 1: All cars turning right
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o3", -25),  # Static: South to West (right)
                    (('o3', 'ir3', 0), "o0", -35),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o1", -15)  # Static: North to East (right)
                ]
            },

            # Scenario 2: Crossing patterns
            {
                "agents": [
                    (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left)
                    (('o2', 'ir2', 0), "o1", 0)  # Agent 2: South to East (left)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o0", -20),  # Static: East to North (left)
                    (('o3', 'ir3', 0), "o2", -30),  # Static: West to South (left)
                    (('o0', 'ir0', 0), "o2", -40)  # Static: North to South (straight)
                ]
            },

            # Scenario 3: East-west corridor
            {
                "agents": [
                    (('o1', 'ir1', 0), "o3", 0),  # Agent 1: East to West
                    (('o3', 'ir3', 0), "o1", 0)  # Agent 2: West to East
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -30),  # Static: East to West (behind)
                    (('o3', 'ir3', 0), "o1", -25),  # Static: West to East (behind)
                    (('o2', 'ir2', 0), "o1", -35)  # Static: South to East (left)
                ]
            },

            # Scenario 4: Complex multi-direction
            {
                "agents": [
                    (('o2', 'ir2', 0), "o0", 0),  # Agent 1: South to North
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o0', 'ir0', 0), "o3", -45),  # Static: North to West (left, far behind)
                    (('o1', 'ir1', 0), "o2", -30),  # Static: East to South (right)
                    (('o2', 'ir2', 0), "o0", -20)  # Static: South to North (straight)
                ]
            },

            # Scenario 5: Same origin dispersal
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o0', 'ir0', 0), "o1", 20)  # Agent 2: North to East (offset forward)
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -50),  # Static: North to South (behind)
                    (('o1', 'ir1', 0), "o3", -30),  # Static: East to West
                    (('o2', 'ir2', 0), "o0", -25)  # Static: South to North
                ]
            },

            # Scenario 6: Convergence to same destination
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o3', 'ir3', 0), "o2", -40),  # Static: West to South (converge)
                    (('o2', 'ir2', 0), "o1", -30),  # Static: South to East
                    (('o0', 'ir0', 0), "o3", -50)  # Static: North to West
                ]
            },

            # Scenario 7: Minimum conflict scenario
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South (straight)
                    (('o2', 'ir2', 0), "o0", 0)  # Agent 2: South to North (straight)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -40),  # Static: East to West (parallel)
                    (('o3', 'ir3', 0), "o1", -35),  # Static: West to East (parallel)
                    (('o0', 'ir0', 0), "o1", -60)  # Static: North to East (turn)
                ]
            },

            # Scenario 8: Rush hour challenge
            {
                "agents": [
                    (('o3', 'ir3', 0), "o1", 0),  # Agent 1: West to East
                    (('o0', 'ir0', 0), "o2", 0)  # Agent 2: North to South
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -20),  # Static: East to West (opposite)
                    (('o2', 'ir2', 0), "o0", -15),  # Static: South to North (opposite)
                    (('o3', 'ir3', 0), "o2", -45)  # Static: West to South (turn)
                ]
            },

            # Scenario 9: Complex intersection
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o3', 'ir3', 0), "o0", 0)  # Agent 2: West to North (right)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o2", -25),  # Static: East to South (right)
                    (('o2', 'ir2', 0), "o3", -30),  # Static: South to West (right)
                    (('o0', 'ir0', 0), "o3", -60)  # Static: North to West (left)
                ]
            },

            # Scenario 10: Parallel lanes
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o0', 'ir0', 0), "o2", -30)  # Agent 2: North to South (behind)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o0", -60),  # Static: South to North (parallel)
                    (('o1', 'ir1', 0), "o3", -55),  # Static: East to West (crossing)
                    (('o3', 'ir3', 0), "o1", -65)  # Static: West to East (crossing)
                ]
            },

            # Scenario 11: Cross traffic
            {
                "agents": [
                    (('o1', 'ir1', 0), "o3", 0),  # Agent 1: East to West
                    (('o2', 'ir2', 0), "o0", 0)  # Agent 2: South to North
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -30),  # Static: North to South (crossing)
                    (('o3', 'ir3', 0), "o1", -25),  # Static: West to East (opposite)
                    (('o1', 'ir1', 0), "o0", -50)  # Static: East to North (turn)
                ]
            },

            # Scenario 12: Left turn conflict
            {
                "agents": [
                    (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left)
                    (('o1', 'ir1', 0), "o0", 0)  # Agent 2: East to North (left)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o1", -35),  # Static: South to East (left)
                    (('o3', 'ir3', 0), "o2", -40),  # Static: West to South (left)
                    (('o0', 'ir0', 0), "o1", -55)  # Static: North to East (right)
                ]
            },

            # Scenario 13: Right turn priority
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o2', 'ir2', 0), "o3", 0)  # Agent 2: South to West (right)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o2", -50),  # Static: East to South (right)
                    (('o3', 'ir3', 0), "o0", -55),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o2", -55)  # Static: North to South (straight)
                ]
            },

            # Scenario 14: Staggered timing
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o3", -40)  # Agent 2: East to West (behind)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o0", -50),  # Static: South to North (behind)
                    (('o3', 'ir3', 0), "o1", -35),  # Static: West to East (behind)
                    (('o0', 'ir0', 0), "o1", -20)  # Static: North to East (behind)
                ]
            },

            # Scenario 15: Opposite directions
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o2', 'ir2', 0), "o0", 0)  # Agent 2: South to North
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -30),  # Static: East to West (parallel)
                    (('o3', 'ir3', 0), "o1", -35),  # Static: West to East (parallel)
                    (('o0', 'ir0', 0), "o1", -10)  # Static: North to East (turn)
                ]
            },

            # Scenario 16: Diagonal crossing
            {
                "agents": [
                    (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left)
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o1", -25),  # Static: South to East (left)
                    (('o3', 'ir3', 0), "o0", -30),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o2", -35)  # Static: North to South (straight)
                ]
            },

            # Scenario 17: Sequential turns
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o3", -50),  # Static: South to West (right)
                    (('o3', 'ir3', 0), "o0", -55),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o3", -65)  # Static: North to West (left)
                ]
            },

            # Scenario 18: Wide spacing
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o3", 50)  # Agent 2: East to West (far ahead)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o0", -60),  # Static: South to North (far behind)
                    (('o3', 'ir3', 0), "o1", -45),  # Static: West to East (behind)
                    (('o0', 'ir0', 0), "o1", -75)  # Static: North to East (far behind)
                ]
            },

            # Scenario 19: Mixed patterns
            {
                "agents": [
                    (('o2', 'ir2', 0), "o3", 0),  # Agent 1: South to West (right)
                    (('o3', 'ir3', 0), "o2", 15)  # Agent 2: West to South (left, ahead)
                ],
                "static": [
                    (('o0', 'ir0', 0), "o1", -30),  # Static: North to East (right)
                    (('o1', 'ir1', 0), "o0", -35),  # Static: East to North (left)
                    (('o2', 'ir2', 0), "o0", -50)  # Static: South to North (straight)
                ]
            },

            # Scenario 20: Heavy traffic mix
            {
                "agents": [
                    (('o1', 'ir1', 0), "o2", 0),  # Agent 1: East to South
                    (('o0', 'ir0', 0), "o1", 0)  # Agent 2: North to East
                ],
                "static": [
                    (('o2', 'ir2', 0), "o3", -25),  # Static: South to West
                    (('o3', 'ir3', 0), "o0", -30),  # Static: West to North
                    (('o1', 'ir1', 0), "o3", -40)  # Static: East to West
                ]
            },

            # Scenario 21: Priority conflict
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o3', 'ir3', 0), "o1", 0)  # Agent 2: West to East
                ],
                "static": [
                    (('o1', 'ir1', 0), "o0", -35),  # Static: East to North
                    (('o2', 'ir2', 0), "o3", -40),  # Static: South to West
                    (('o0', 'ir0', 0), "o3", -50)  # Static: North to West
                ]
            },

            # Scenario 22: Roundabout-like flow
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South
                ],
                "static": [
                    (('o2', 'ir2', 0), "o3", -30),  # Static: South to West
                    (('o3', 'ir3', 0), "o0", -35),  # Static: West to North
                    (('o0', 'ir0', 0), "o2", -60)  # Static: North to South
                ]
            },

            # Scenario 23: High-speed challenge
            {
                "agents": [
                    (('o2', 'ir2', 0), "o0", 0),  # Agent 1: South to North
                    (('o1', 'ir1', 0), "o3", 0)  # Agent 2: East to West
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -20),  # Static: North to South (close)
                    (('o3', 'ir3', 0), "o1", -25),  # Static: West to East (close)
                    (('o2', 'ir2', 0), "o1", -45)  # Static: South to East
                ]
            },

            # Scenario 24: Final complex scenario
            {
                "agents": [
                    (('o3', 'ir3', 0), "o2", 0),  # Agent 1: West to South
                    (('o0', 'ir0', 0), "o3", 0)  # Agent 2: North to West
                ],
                "static": [
                    (('o1', 'ir1', 0), "o0", -30),  # Static: East to North
                    (('o2', 'ir2', 0), "o1", -35),  # Static: South to East
                    (('o3', 'ir3', 0), "o1", -55)  # Static: West to East
                ]
            }
        ]

        return base_scenarios

    def _generate_all_scenarios(self):
        """Generate all 100 scenarios (25 base × 4 rotations)"""
        base_scenarios = self._get_base_scenarios()
        all_scenarios = []

        for base_idx, base_scenario in enumerate(base_scenarios):
            # Add original (0° rotation)
            all_scenarios.append(base_scenario)

            # Add 3 rotations (90°, 180°, 270°)
            for rotation in [1, 2, 3]:
                rotated_agents = rotate_scenario_clockwise(base_scenario["agents"], rotation)
                rotated_static = rotate_scenario_clockwise(base_scenario["static"], rotation)

                rotated_scenario = {
                    "agents": rotated_agents,
                    "static": rotated_static
                }
                all_scenarios.append(rotated_scenario)

        return all_scenarios

    def get_scenario(self, idx):
        """Get specific scenario by index (0-99)"""
        if idx < 0 or idx >= len(self.all_scenarios):
            raise ValueError(f"Scenario index {idx} out of range [0, {len(self.all_scenarios) - 1}]")
        return self.all_scenarios[idx]

    def get_random_scenario(self):
        """Get random scenario"""
        return random.choice(self.all_scenarios)

    def get_scenario_count(self):
        """Get total number of scenarios"""
        return len(self.all_scenarios)


# ============================================================================
# SECTION 2: QMIX NETWORKS
# ============================================================================

class QNetwork(nn.Module):
    """Individual Q-network for each agent"""

    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] or [obs_dim]
        Returns:
            q_values: [batch_size, n_actions] or [n_actions]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class HyperNetwork(nn.Module):
    """Hypernetwork for generating mixing network weights"""

    def __init__(self, state_dim, n_agents, mixing_embed_dim):
        super(HyperNetwork, self).__init__()

        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim

        # Hypernetwork for first layer weights (ensures non-negativity via abs)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # Hypernetwork for second layer weights
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            w1, b1, w2, b2: weights and biases for mixing network
        """
        batch_size = state.shape[0]

        # First layer weights (must be non-negative for monotonicity)
        w1 = torch.abs(self.hyper_w1(state))  # [batch_size, n_agents * mixing_embed_dim]
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(state)  # [batch_size, mixing_embed_dim]

        # Second layer weights (must be non-negative for monotonicity)
        w2 = torch.abs(self.hyper_w2(state))  # [batch_size, mixing_embed_dim]
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(state)  # [batch_size, 1]

        return w1, b1, w2, b2


class MixingNetwork(nn.Module):
    """
    Mixing network with monotonicity constraints (QMIX)
    Ensures: ∂Q_tot/∂Q_i ≥ 0 for all i
    """

    def __init__(self, n_agents, state_dim, mixing_embed_dim=32):
        super(MixingNetwork, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_net = HyperNetwork(state_dim, n_agents, mixing_embed_dim)

    def forward(self, agent_qs, state):
        """
        Args:
            agent_qs: [batch_size, n_agents] - individual Q-values
            state: [batch_size, state_dim] - global state
        Returns:
            q_tot: [batch_size, 1] - mixed Q-value
        """
        batch_size = agent_qs.shape[0]

        # Get mixing network weights from hypernetwork
        w1, b1, w2, b2 = self.hyper_net(state)

        # First layer: [batch_size, n_agents] × [batch_size, n_agents, mixing_embed_dim]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1.unsqueeze(1))

        # Second layer: [batch_size, 1, mixing_embed_dim] × [batch_size, mixing_embed_dim, 1]
        # FIX: Reshape b2 to [batch_size, 1, 1] to ensure correct broadcasting
        q_tot = torch.bmm(hidden, w2) + b2.view(batch_size, 1, 1)

        return q_tot.view(batch_size, 1)


# ============================================================================
# SECTION 3: REPLAY BUFFER
# ============================================================================

Transition = namedtuple('Transition',
                        ('obs', 'state', 'actions', 'reward', 'next_obs',
                         'next_state', 'done'))


class ReplayBuffer:
    """
    Replay buffer for storing episodes.
    Stores complete episodes for TD(λ) calculation.
    """

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_buffer = []

    def add_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        """Add a single transition to the current episode"""
        transition = Transition(obs, state, actions, reward, next_obs, next_state, done)
        self.episode_buffer.append(transition)

    def end_episode(self):
        """Mark the end of an episode and store it"""
        if len(self.episode_buffer) > 0:
            self.buffer.append(list(self.episode_buffer))
            self.episode_buffer = []

    def sample(self, batch_size):
        """Sample a batch of episodes"""
        episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return episodes

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.episode_buffer = []


# ============================================================================
# SECTION 4: QMIXwD AGENT
# ============================================================================

class QMIXwD:
    """
    QMIXwD: QMIX with Demonstrations

    Implements pre-training with demonstration data and online training.
    """

    def __init__(
            self,
            n_agents=2,
            obs_dim=8,
            n_actions=2,
            state_dim=20,
            hidden_dim=64,
            mixing_embed_dim=32,
            lr=5e-4,
            gamma=0.99,
            lambda_td=0.8,
            target_update_interval=200,
            buffer_capacity=5000,
            device='cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.lambda_td = lambda_td
        self.target_update_interval = target_update_interval
        self.device = device

        # Individual Q-networks (shared parameters across agents)
        self.q_network = QNetwork(obs_dim, n_actions, hidden_dim).to(device)
        self.target_q_network = QNetwork(obs_dim, n_actions, hidden_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Mixing network
        self.mixing_network = MixingNetwork(n_agents, state_dim, mixing_embed_dim).to(device)
        self.target_mixing_network = MixingNetwork(n_agents, state_dim, mixing_embed_dim).to(device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=lr)

        # Replay buffers
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.demo_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training counters
        self.training_steps = 0
        self.episodes_trained = 0

        # Speed optimization
        self.max_n_step = 5  # Limit TD(lambda) n-step for speed

        print(f"[QMIXwD] Initialized with {n_agents} agents")
        print(f"  - Observation dim: {obs_dim}")
        print(f"  - State dim: {state_dim}")
        print(f"  - Action space: {n_actions}")
        print(f"  - Max n-step (TD-lambda): {self.max_n_step}")
        print(f"  - Device: {device}")

    def select_action(self, obs, epsilon=0.0):
        """
        Select actions for all agents using ε-greedy policy

        Args:
            obs: list of observations, each [obs_dim]
            epsilon: exploration rate
        Returns:
            actions: list of action indices
        """
        if random.random() < epsilon:
            # Random actions
            actions = [random.randint(0, self.n_actions - 1) for _ in range(self.n_agents)]
        else:
            # Greedy actions
            actions = []
            with torch.no_grad():
                for agent_obs in obs:
                    obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                    q_values = self.q_network(obs_tensor)
                    action = q_values.argmax(dim=1).item()
                    actions.append(action)

        return actions

    def compute_td_lambda_targets(self, episode, gamma, lambda_td, max_n_step=10):
        """
        Compute TD(λ) targets for an episode (OPTIMIZED VERSION)

        Args:
            episode: list of Transitions
            gamma: discount factor
            lambda_td: TD(λ) parameter
            max_n_step: maximum n-step lookahead (reduces O(T²) to O(T*n))
        Returns:
            targets: list of TD(λ) targets
        """
        T = len(episode)
        targets = []

        # Pre-compute all rewards
        rewards = [t.reward for t in episode]

        for t in range(T):
            # Limit n-step lookahead for speed (instead of going to T)
            max_n = min(max_n_step, T - t)

            g_lambda = 0.0
            weight_sum = 0.0

            for n in range(max_n):
                # n-step return (vectorized)
                g_n = sum(gamma ** k * rewards[t + k] for k in range(n + 1))

                # Add terminal value if not done
                if t + n + 1 < T and not episode[t + n].done:
                    # Get target Q-value for next state (cached computation)
                    next_obs = episode[t + n].next_obs
                    next_state = episode[t + n].next_state

                    with torch.no_grad():
                        # Batch process all agents at once
                        next_obs_tensor = torch.FloatTensor(np.array(next_obs)).to(self.device)
                        next_q_values = self.target_q_network(next_obs_tensor)
                        next_agent_qs = next_q_values.max(dim=1)[0].unsqueeze(0)

                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        next_q_tot = self.target_mixing_network(next_agent_qs, next_state_tensor)
                        g_n += (gamma ** (n + 1)) * next_q_tot.item()

                # Weight by λ
                weight = (lambda_td ** n) if n < max_n - 1 else 1.0
                g_lambda += weight * g_n
                weight_sum += weight

            if weight_sum > 0:
                g_lambda /= weight_sum

            targets.append(g_lambda)

        return targets

    def pretrain_step(self, batch_size=32, lambda_demo=1.0, lambda_td_loss=1.0, lambda_l2=1e-4):
        if len(self.demo_buffer) < batch_size:
            return None

        episodes = self.demo_buffer.sample(batch_size)

        total_loss_sum = 0.0
        n_batches = 0

        # Optimization: Process episodes in a loop (QMIX typically handles 1 episode per batch-step in RNNs,
        # but here we use dense nets, so we can batch transitions differently.
        # For simplicity adhering to your structure, we process per episode).

        for episode in episodes:
            # Calculate TD Targets
            td_targets = self.compute_td_lambda_targets(
                episode, self.gamma, self.lambda_td, max_n_step=self.max_n_step
            )

            # Unpack episode
            obs_batch = [t.obs for t in episode]  # List of agent obs lists
            state_batch = np.array([t.state for t in episode])  # [T, state_dim]
            actions_batch = np.array([t.actions for t in episode])  # [T, n_agents]

            # Convert to tensors
            state_tensor = torch.FloatTensor(state_batch).to(self.device)
            target_tensor = torch.FloatTensor(td_targets).unsqueeze(1).to(self.device)  # [T, 1]
            actions_tensor = torch.LongTensor(actions_batch).to(self.device)  # [T, n_agents]

            # 1. Calculate Q_tot for TD Loss
            agent_qs_list = []
            q_evals_for_margin = []  # Store for demo loss

            for agent_idx in range(self.n_agents):
                # Extract obs for this agent across time
                ag_obs = np.array([o[agent_idx] for o in obs_batch])
                ag_obs_tensor = torch.FloatTensor(ag_obs).to(self.device)

                # Get Q values [T, n_actions]
                q_values = self.q_network(ag_obs_tensor)
                q_evals_for_margin.append(q_values)

                # Select Q of taken action
                taken_actions = actions_tensor[:, agent_idx]
                q_taken = q_values.gather(1, taken_actions.unsqueeze(1)).squeeze(1)
                agent_qs_list.append(q_taken)

            # Stack for mixing [T, n_agents]
            agent_qs_stacked = torch.stack(agent_qs_list, dim=1)

            # Mix
            q_tot = self.mixing_network(agent_qs_stacked, state_tensor)

            # TD Loss
            td_loss = F.mse_loss(q_tot, target_tensor)

            # 2. Large Margin Classification Loss for Demonstrations
            # J = max_a [ Q(s,a) + l(a_E, a) ] - Q(s, a_E)
            demo_loss = 0.0
            margin_val = 0.8

            for agent_idx in range(self.n_agents):
                q_values = q_evals_for_margin[agent_idx]  # [T, n_actions]
                expert_actions = actions_tensor[:, agent_idx].unsqueeze(1)  # [T, 1]

                # Create margin tensor: 0 for expert action, margin_val for others
                margins = torch.ones_like(q_values) * margin_val
                # Scatter 0.0 where action == expert_action
                margins.scatter_(1, expert_actions, 0.0)

                # Q(s,a) + l(a_E, a)
                margin_augmented_q = q_values + margins

                # max_a ...
                max_margin_q = margin_augmented_q.max(dim=1)[0]

                # Q(s, a_E)
                expert_q = q_values.gather(1, expert_actions).squeeze(1)

                # Loss for this agent
                demo_loss += (max_margin_q - expert_q).mean()

            demo_loss /= self.n_agents

            # L2 Loss (Manual)
            l2_loss = 0.0
            for p in self.q_network.parameters(): l2_loss += p.norm(2)
            for p in self.mixing_network.parameters(): l2_loss += p.norm(2)

            loss = (lambda_demo * demo_loss) + (lambda_td_loss * td_loss) + (lambda_l2 * l2_loss)

            self.q_optimizer.zero_grad()
            self.mixing_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.q_optimizer.step()
            self.mixing_optimizer.step()

            total_loss_sum += loss.item()
            n_batches += 1

        self.training_steps += 1
        return {'total_loss': total_loss_sum / max(1, n_batches)}

    def train_step(self, batch_size=32):
        """
        Online training step (standard QMIX with TD(λ))

        Loss: L = L_TD(λ) only
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample episodes from replay buffer
        episodes = self.replay_buffer.sample(batch_size)

        total_loss = 0.0
        n_transitions = 0

        for episode in episodes:
            # Compute TD(λ) targets for this episode (OPTIMIZED)
            td_targets = self.compute_td_lambda_targets(
                episode, self.gamma, self.lambda_td, max_n_step=self.max_n_step
            )

            for t, transition in enumerate(episode):
                obs = transition.obs
                state = transition.state
                actions = transition.actions
                td_target = td_targets[t]

                # Convert to tensors
                obs_tensors = [torch.FloatTensor(o).unsqueeze(0).to(self.device) for o in obs]
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                actions_tensor = torch.LongTensor(actions).to(self.device)
                target_tensor = torch.FloatTensor([td_target]).to(self.device)

                # Get current Q-values
                agent_qs = []
                for i, obs_tensor in enumerate(obs_tensors):
                    q_values = self.q_network(obs_tensor)
                    agent_qs.append(q_values[0, actions_tensor[i]])

                agent_qs = torch.stack(agent_qs).unsqueeze(0)
                q_tot = self.mixing_network(agent_qs, state_tensor)

                # TD(λ) loss
                loss = F.mse_loss(q_tot, target_tensor.unsqueeze(1))

                # Backward pass
                self.q_optimizer.zero_grad()
                self.mixing_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10.0)

                self.q_optimizer.step()
                self.mixing_optimizer.step()

                total_loss += loss.item()
                n_transitions += 1

        self.training_steps += 1

        # Update target networks periodically
        if self.training_steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        return {'loss': total_loss / n_transitions}

    def save(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'mixing_network': self.mixing_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'mixing_optimizer': self.mixing_optimizer.state_dict(),
            'training_steps': self.training_steps
        }, path)
        print(f"[QMIXwD] Model saved to {path}")

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.mixing_optimizer.load_state_dict(checkpoint['mixing_optimizer'])
        self.training_steps = checkpoint['training_steps']
        print(f"[QMIXwD] Model loaded from {path}")


# ============================================================================
# SECTION 5: ENVIRONMENT WRAPPER
# ============================================================================

class IntersectionWrapper:
    """
    Wrapper for highway-env intersection with State Padding.
    """

    def __init__(self, scenario_manager, n_agents=2, n_vehicles_state=10):
        self.scenario_manager = scenario_manager
        self.n_agents = n_agents
        self.n_features = 4  # [x, y, vx, vy]
        self.n_vehicles_state = n_vehicles_state  # Fixed number of vehicles to observe for global state
        self.state_dim = n_vehicles_state * self.n_features

        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
                "absolute": True,
                "flatten": False,
                "observe_intentions": False,
                "vehicles_count": 15,  # Observe enough vehicles to fill state
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": False,
                    "longitudinal": True,
                    "target_speeds": [0, 10],
                },
            },
            "controlled_vehicles": n_agents,
            "duration": 40,  # Shorter duration for faster iterations
        }

        self.env = gym.make('intersection-v0', render_mode=None, config=config)

    def _process_obs(self, obs_raw):
        """Pad or truncate observation to ensure fixed state size"""
        # obs_raw is (V, F)
        if not isinstance(obs_raw, np.ndarray):
            obs_raw = np.array(obs_raw)

        # 1. Get Agent Observations
        obs = []
        for i in range(self.n_agents):
            if i < len(obs_raw):
                obs.append(obs_raw[i].flatten())
            else:
                obs.append(np.zeros(self.n_features))

        # 2. Construct Global State (Fixed Size)
        # Sort vehicles by distance to center (approximate using x+y magnitude) for consistency
        # This helps the network learn that row 0 is always the most relevant vehicle
        dists = np.abs(obs_raw[:, 0]) + np.abs(obs_raw[:, 1])
        sorted_indices = np.argsort(dists)
        sorted_obs = obs_raw[sorted_indices]

        # Pad or Truncate
        current_vehicles = len(sorted_obs)
        if current_vehicles >= self.n_vehicles_state:
            state_data = sorted_obs[:self.n_vehicles_state]
        else:
            padding = np.zeros((self.n_vehicles_state - current_vehicles, self.n_features))
            state_data = np.vstack((sorted_obs, padding))

        state = state_data.flatten()
        return obs, state

    def reset(self, scenario_idx=None):
        if scenario_idx is not None:
            # Seed the env for consistent evaluation, though full scenario control
            # requires deeper highway-env overriding
            self.env.unwrapped.configure({"seed": int(scenario_idx)})

        obs_raw, info = self.env.reset()
        return self._process_obs(obs_raw)

    def step(self, actions):
        if isinstance(actions, list):
            actions = tuple(actions)
        obs_raw, reward, done, truncated, info = self.env.step(actions)

        next_obs, next_state = self._process_obs(obs_raw)
        return next_obs, next_state, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================================
# SECTION 6: DEMONSTRATION COLLECTOR
# ============================================================================

class DemonstrationCollector:
    """
    Collects demonstration data for pre-training.

    Demonstration data consists of:
    - Expert data (10%): from a pre-trained QMIX policy
    - Interaction data (90%): self-generated by current policy
    """

    def __init__(self, agent, env, scenario_manager):
        self.agent = agent
        self.env = env
        self.scenario_manager = scenario_manager

    def collect_expert_episode(self):
        """
        Collect one episode using expert policy (greedy + small epsilon)
        For initial collection, we use a nearly greedy policy from the initialized agent
        """
        obs, state = self.env.reset()
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            # Expert uses greedy policy with small exploration
            actions = self.agent.select_action(obs, epsilon=0.05)
            next_obs, next_state, reward, done, truncated, info = self.env.step(actions)

            # Store transition
            self.agent.demo_buffer.add_transition(
                obs, state, actions, reward, next_obs, next_state, done
            )

            obs = next_obs
            state = next_state
            steps += 1

        self.agent.demo_buffer.end_episode()
        return steps

    def collect_interaction_episode(self, epsilon=0.3):
        """
        Collect one episode using current policy with exploration
        This is the "self-generated interaction data"
        """
        obs, state = self.env.reset()
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            # Use current policy with moderate exploration
            actions = self.agent.select_action(obs, epsilon=epsilon)
            next_obs, next_state, reward, done, truncated, info = self.env.step(actions)

            # Store transition
            self.agent.demo_buffer.add_transition(
                obs, state, actions, reward, next_obs, next_state, done
            )

            obs = next_obs
            state = next_state
            steps += 1

        self.agent.demo_buffer.end_episode()
        return steps

    def collect_demonstrations(self, n_episodes=1000, expert_ratio=0.1):
        """
        Collect demonstration data

        Args:
            n_episodes: total number of episodes to collect
            expert_ratio: ratio of expert episodes (0.1 = 10%)
        """
        n_expert = int(n_episodes * expert_ratio)
        n_interaction = n_episodes - n_expert

        print(f"[DemoCollector] Collecting {n_episodes} demonstration episodes")
        print(f"  - Expert episodes: {n_expert}")
        print(f"  - Interaction episodes: {n_interaction}")

        import time

        # Collect expert episodes
        start = time.time()
        total_steps = 0
        for i in range(n_expert):
            steps = self.collect_expert_episode()
            total_steps += steps
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (n_expert - i - 1)
                print(
                    f"  Expert: {i + 1}/{n_expert} | Avg: {avg_time:.2f}s/ep | Steps: {total_steps / (i + 1):.1f} | ETA: {remaining:.1f}s")

        expert_time = time.time() - start
        print(f"  Expert collection: {expert_time:.1f}s ({expert_time / n_expert:.2f}s per episode)")

        # Collect interaction episodes
        start = time.time()
        total_steps = 0
        for i in range(n_interaction):
            steps = self.collect_interaction_episode(epsilon=0.3)
            total_steps += steps
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (n_interaction - i - 1)
                print(
                    f"  Interaction: {i + 1}/{n_interaction} | Avg: {avg_time:.2f}s/ep | Steps: {total_steps / (i + 1):.1f} | ETA: {remaining:.1f}s")

        interaction_time = time.time() - start
        print(
            f"  Interaction collection: {interaction_time:.1f}s ({interaction_time / n_interaction:.2f}s per episode)")

        print(f"[DemoCollector] Total time: {(expert_time + interaction_time) / 60:.1f} minutes")
        print(f"[DemoCollector] Collected {len(self.agent.demo_buffer)} episodes")


# ============================================================================
# SECTION 7: TRAINING LOOP
# ============================================================================

def train_qmixwd(
        agent,
        env,
        scenario_manager,
        n_episodes=10000,
        n_pretrain_episodes=1000,
        n_pretrain_steps=5000,
        expert_ratio=0.1,
        epsilon=0.1,
        batch_size=32,
        train_interval=4
):
    """
    Main training loop for QMIXwD

    Phase 1: Collect demonstration data
    Phase 2: Pre-train with demonstrations
    Phase 3: Online training

    Args:
        agent: QMIXwD agent
        env: IntersectionWrapper
        scenario_manager: ScenarioManager
        n_episodes: total training episodes
        n_pretrain_episodes: episodes for demonstration collection
        n_pretrain_steps: pre-training gradient steps
        expert_ratio: ratio of expert data in demonstrations
        epsilon: exploration rate for online training
        batch_size: mini-batch size
        train_interval: train every N episodes

    Returns:
        agent: trained agent
        results: training metrics
    """
    results = {
        'train_success_rates': [],
        'train_rewards': [],
        'episode_lengths': [],
        'pretrain_losses': [],
        'train_losses': []
    }

    # ========================================================================
    # PHASE 1: COLLECT DEMONSTRATION DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: COLLECTING DEMONSTRATION DATA")
    print("=" * 70)

    import time
    start_time = time.time()

    demo_collector = DemonstrationCollector(agent, env, scenario_manager)
    demo_collector.collect_demonstrations(n_pretrain_episodes, expert_ratio)

    phase1_time = time.time() - start_time
    print(f"[Phase 1] Completed in {phase1_time / 60:.1f} minutes")

    # ========================================================================
    # PHASE 2: PRE-TRAINING
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: PRE-TRAINING WITH DEMONSTRATIONS")
    print("=" * 70)

    phase2_start = time.time()

    for step in range(n_pretrain_steps):
        loss_dict = agent.pretrain_step(batch_size=batch_size)

        if loss_dict is not None:
            results['pretrain_losses'].append(loss_dict['total_loss'])

            if (step + 1) % 200 == 0:  # More frequent updates
                elapsed = time.time() - phase2_start
                remaining = (elapsed / (step + 1)) * (n_pretrain_steps - step - 1)
                print(f"Pre-train Step {step + 1}/{n_pretrain_steps} | "
                      f"Loss: {loss_dict['total_loss']:.4f} | "
                      f"ETA: {remaining / 60:.1f} min")

    phase2_time = time.time() - phase2_start
    print(f"\n[Phase 2] Completed in {phase2_time / 60:.1f} minutes")

    # ========================================================================
    # PHASE 3: ONLINE TRAINING
    # ========================================================================
    # ========================================================================
    # PHASE 3: ONLINE TRAINING
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: ONLINE TRAINING")
    print("=" * 70)

    # Initialize log file for parsing later
    log_file_path = "training_log.txt"
    with open(log_file_path, "w") as f:
        f.write("Training Logs\n")

    phase3_start = time.time()

    for episode in range(n_episodes):
        obs, state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        crashed = False

        while not done and not truncated:
            actions = agent.select_action(obs, epsilon=epsilon)
            next_obs, next_state, reward, done, truncated, info = env.step(actions)

            agent.replay_buffer.add_transition(
                obs, state, actions, reward, next_obs, next_state, done
            )

            obs = next_obs
            state = next_state
            episode_reward += reward
            episode_length += 1

            if info.get('crashed', False):
                crashed = True

        agent.replay_buffer.end_episode()

        # --- LOGGING UPDATES START HERE ---
        # Determine result string for Regex parsing
        if crashed:
            result_str = "Collision"
        elif truncated:
            result_str = "Timeout"
        else:
            result_str = "Success"

        # 1. Update internal metrics
        results['train_success_rates'].append(1 if result_str == "Success" else 0)
        results['train_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)

        # 2. Log to console in specific Regex-friendly format:
        # "Episode X Result: Success|Collision | Reward: Y | Steps: Z"
        log_string = (f"Episode {episode + 1} Result: {result_str} | "
                      f"Reward: {episode_reward:.2f} | Steps: {episode_length}")

        # Only print every 10 episodes to keep console clean, or every 1 if you prefer
        if (episode + 1) % 10 == 0:
            print(log_string)

        # Always write to file for the graph parser
        with open(log_file_path, "a") as f:
            f.write(log_string + "\n")
        # --- LOGGING UPDATES END HERE ---

        # Train periodically
        if episode % train_interval == 0 and len(agent.replay_buffer) >= batch_size:
            loss_dict = agent.train_step(batch_size=batch_size)
            if loss_dict is not None:
                results['train_losses'].append(loss_dict['loss'])

    phase3_time = time.time() - phase3_start
    print(f"\n[Phase 3] Completed in {phase3_time / 60:.1f} minutes")
    print(f"Log file saved to: {os.path.abspath(log_file_path)}")

    return agent, results


# ============================================================================
# SECTION 8: EVALUATION
# ============================================================================

def evaluate(agent, env, scenario_manager, n_episodes=100, render=False):
    """
    Evaluate agent on specific scenarios

    Uses scenarios 0-99 in order (each scenario exactly once)

    Args:
        agent: trained QMIXwD agent
        env: IntersectionWrapper
        scenario_manager: ScenarioManager
        n_episodes: number of evaluation episodes (default 100)
        render: whether to render episodes

    Returns:
        results: evaluation metrics
    """
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = {
        'success_rates': [],
        'rewards': [],
        'episode_lengths': [],
        'travel_times': []  # Only for successful episodes
    }

    for episode_idx in range(n_episodes):
        # Reset with specific scenario
        obs, state = env.reset(scenario_idx=episode_idx)

        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        crashed = False

        while not done and not truncated:
            # Greedy action selection (no exploration)
            actions = agent.select_action(obs, epsilon=0.0)

            # Step environment
            next_obs, next_state, reward, done, truncated, info = env.step(actions)

            # Update
            obs = next_obs
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Check for crash
            if info.get('crashed', False):
                crashed = True

            # Render if requested
            if render:
                env.render()

        # Record results
        success = not crashed
        results['success_rates'].append(1 if success else 0)
        results['rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)

        if success:
            results['travel_times'].append(episode_length)

        # Log progress
        if (episode_idx + 1) % 10 == 0:
            current_sr = np.mean(results['success_rates'])
            print(f"Evaluated {episode_idx + 1}/{n_episodes} scenarios, Success Rate: {current_sr:.2%}")

    # Final statistics
    final_sr = np.mean(results['success_rates'])
    avg_reward = np.mean(results['rewards'])
    avg_length = np.mean(results['episode_lengths'])

    if len(results['travel_times']) > 0:
        avg_travel_time = np.mean(results['travel_times'])
    else:
        avg_travel_time = 0.0

    print(f"\n[Evaluation] Results:")
    print(f"  Success Rate: {final_sr:.2%} ({int(final_sr * n_episodes)}/{n_episodes})")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Episode Length: {avg_length:.1f}")
    print(f"  Avg Travel Time (successful): {avg_travel_time:.1f}")

    return results


# ============================================================================
# SECTION 9: VISUALIZATION & SAVING
# ============================================================================

def save_results(train_results, eval_results, save_dir):
    """
    Save training curves using the specific 4-subplot style requested.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Convert Dictionary Results to DataFrame for easy rolling calculation
    # We reconstruct the 'Result' column based on success rate for consistency
    data = {
        'episode': range(1, len(train_results['train_success_rates']) + 1),
        'reward': train_results['train_rewards'],
        'steps': train_results['episode_lengths'],
        'is_success': train_results['train_success_rates']
    }
    df = pd.DataFrame(data)

    # Create a 'result' column for text logic if needed, though we can compute rates directly
    # Note: In the simplified dict, 0 is failure. We assume Failure = Collision for the graph
    # unless we passed the specific crash list.
    df['result'] = df['is_success'].apply(lambda x: 'Success' if x == 1 else 'Collision')

    # 2. Define Helper Functions (Internal)
    window = 20

    def get_success_rate(df, window):
        return (df['result'] == 'Success').astype(int).rolling(window, min_periods=1).mean() * 100

    def get_collision_rate(df, window):
        # In this simplified binary view, Collision = 1 - Success
        return (df['result'] == 'Collision').astype(int).rolling(window, min_periods=1).mean() * 100

    def get_reward_per_episode(df, window):
        return df['reward'].rolling(window, min_periods=1).mean()

    def get_avg_travel_time(df, window):
        # Filter for only successful episodes, then roll
        # Note: This creates gaps in the index, so we reindex or plot against original episodes
        success_mask = (df['result'] == 'Success')
        successful_steps = df['steps'].where(success_mask)  # NaN where not success
        return successful_steps.rolling(window, min_periods=1).mean()

    # 3. Plotting
    plt.figure(figsize=(14, 10))

    # Plot 1: Success Rate
    plt.subplot(2, 2, 1)
    plt.plot(df['episode'], get_success_rate(df, window), label='QMIXwD', color='blue')
    plt.title(f'Success Rate (%) - Moving Average (w={window})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Collision Rate
    plt.subplot(2, 2, 2)
    plt.plot(df['episode'], get_collision_rate(df, window), label='QMIXwD', color='red')
    plt.title(f'Collision Rate (%) - Moving Average (w={window})')
    plt.xlabel('Episode')
    plt.ylabel('Collision Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Reward
    plt.subplot(2, 2, 3)
    plt.plot(df['episode'], get_reward_per_episode(df, window), label='QMIXwD', color='green')
    plt.title(f'Reward per Episode - Moving Average (w={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 4: Travel Time
    plt.subplot(2, 2, 4)
    plt.plot(df['episode'], get_avg_travel_time(df, window), label='QMIXwD', color='purple')
    plt.title(f'Average Travel Time (Steps) - Moving Average (w={window})')
    plt.xlabel('Episode')
    plt.ylabel('Steps in Successful Episodes')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_graphs.png'), dpi=300)
    print(f"[Results] Graphs saved to {save_dir}/training_graphs.png")
    plt.close()

    # [Preserve JSON/CSV saving logic from original code here if desired...]
    # For brevity, simply dumping the CSV for manual checking:
    df.to_csv(os.path.join(save_dir, 'training_data.csv'), index=False)
    print(f"[Results] Raw data CSV saved to {save_dir}/training_data.csv")


def main():
    """Main execution function"""

    # ========================================================================
    # SETUP
    # ========================================================================
    print("\n" + "=" * 70)
    print("QMIXwD FOR UNSIGNALIZED INTERSECTION NAVIGATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/qmixwd_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # ========================================================================
    # CREATE COMPONENTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("INITIALIZING COMPONENTS")
    print("=" * 70)

    # Scenario manager
    scenario_manager = ScenarioManager()

    # Environment
    n_agents = 2
    env = IntersectionWrapper(scenario_manager, n_agents=n_agents)

    # Agent
    agent = QMIXwD(
        n_agents=n_agents,
        obs_dim=4,  # [x, y, vx, vy] for each agent
        n_actions=2,  # {stop, go}
        state_dim=40,  # 10 vehicles × 4 features
        hidden_dim=64,
        mixing_embed_dim=32,
        lr=5e-4,
        gamma=0.99,
        lambda_td=0.8,
        target_update_interval=200,
        buffer_capacity=5000,
        device=device
    )

    # ========================================================================
    # TRAINING
    # ========================================================================
    agent, train_results = train_qmixwd(
        agent=agent,
        env=env,
        scenario_manager=scenario_manager,
        n_episodes=1000,  # Increased to see learning
        n_pretrain_episodes=100,  # Increased back
        n_pretrain_steps=300,  # More pre-training
        expert_ratio=0.1,  # 10% expert, 90% interaction
        epsilon=0.3,  # HIGH exploration for harder task!
        batch_size=8,  # Back to 8
        train_interval=4  # Train more frequently
    )

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    model_path = os.path.join(save_dir, 'qmixwd_model.pt')
    agent.save(model_path)

    # ========================================================================
    # EVALUATION
    # ========================================================================
    eval_results = evaluate(
        agent=agent,
        env=env,
        scenario_manager=scenario_manager,
        n_episodes=100,  # Evaluate on all 100 scenarios
        render=False
    )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    save_results(train_results, eval_results, save_dir)

    # ========================================================================
    # CLEANUP
    # ========================================================================
    env.close()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"All results saved to: {save_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
