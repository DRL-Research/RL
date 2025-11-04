"""
QMIX Network Architecture
Implements individual Q-networks and mixing network with monotonicity constraints
Based on: "Monotonic value function factorisation for deep multi-agent reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    Individual Q-network for each agent.
    All agents share the same network parameters.
    
    Input: Agent observation (partial state)
    Output: Q-values for each action
    """
    
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] agent observations
        Returns:
            q_values: [batch_size, n_actions]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class HyperNetwork(nn.Module):
    """
    Hypernetwork that generates mixing network weights from global state.
    Ensures monotonicity by using absolute values of generated weights.
    """
    
    def __init__(self, state_dim, n_agents, hidden_dim=64, mixing_embed_dim=32):
        super(HyperNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim
        
        # Generate weights for first layer of mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * mixing_embed_dim)
        )
        
        # Generate bias for first layer
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Generate weights for second layer of mixing network
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mixing_embed_dim)
        )
        
        # Generate bias for second layer (scalar output)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] global state
        Returns:
            w1, b1, w2, b2: weights and biases for mixing network
        """
        batch_size = state.shape[0]
        
        # Generate weights - use absolute value to ensure monotonicity
        w1 = torch.abs(self.hyper_w1(state))  # [batch, n_agents * mixing_embed_dim]
        w1 = w1.reshape(batch_size, self.n_agents, self.mixing_embed_dim)

        b1 = self.hyper_b1(state)  # [batch, mixing_embed_dim]

        w2 = torch.abs(self.hyper_w2(state))  # [batch, mixing_embed_dim]
        w2 = w2.reshape(batch_size, self.mixing_embed_dim, 1)

        b2 = self.hyper_b2(state)  # [batch, 1]

        return w1, b1, w2, b2


class MixingNetwork(nn.Module):
    """
    Mixing network that combines individual Q-values into joint Q-value.
    Uses hypernetwork-generated weights to ensure monotonicity constraint.
    """

    def __init__(self, state_dim, n_agents, mixing_embed_dim=32):
        super(MixingNetwork, self).__init__()

        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim
        self.state_dim = state_dim

        self.hyper = HyperNetwork(state_dim, n_agents, mixing_embed_dim=mixing_embed_dim)

    def forward(self, q_values, state):
        """
        Args:
            q_values: [batch_size, n_agents] individual Q-values
            state: [batch_size, state_dim] global state
        Returns:
            q_total: [batch_size, 1] mixed Q-value
        """
        batch_size = q_values.shape[0]

        # Get hypernetwork weights
        w1, b1, w2, b2 = self.hyper(state)

        # First layer: [batch, n_agents] @ [batch, n_agents, mixing_embed_dim] -> [batch, mixing_embed_dim]
        q_values = q_values.reshape(batch_size, 1, self.n_agents)
        hidden = torch.bmm(q_values, w1).squeeze(1) + b1  # [batch, mixing_embed_dim]
        hidden = F.elu(hidden)

        # Second layer: [batch, mixing_embed_dim] @ [batch, mixing_embed_dim, 1] -> [batch, 1]
        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2

        return q_total


class QMIXNetwork(nn.Module):
    """
    Complete QMIX architecture combining individual Q-networks and mixing network.
    """

    def __init__(self, obs_dim, state_dim, n_agents, n_actions,
                 hidden_dim=64, mixing_embed_dim=32):
        super(QMIXNetwork, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        # Shared Q-network for all agents
        self.q_network = QNetwork(obs_dim, n_actions, hidden_dim)

        # Mixing network
        self.mixing_network = MixingNetwork(state_dim, n_agents, mixing_embed_dim)

    def forward(self, obs, state=None):
        """
        Forward pass for all agents.

        Args:
            obs: [batch_size, n_agents, obs_dim] agent observations
            state: [batch_size, state_dim] global state (optional for greedy actions)
        Returns:
            q_values: [batch_size, n_agents, n_actions] individual Q-values
        """
        batch_size = obs.shape[0]

        # Reshape for batch processing - use reshape instead of view for safety
        obs_flat = obs.reshape(-1, self.obs_dim)  # [batch * n_agents, obs_dim]

        # Get individual Q-values
        q_values = self.q_network(obs_flat)  # [batch * n_agents, n_actions]
        q_values = q_values.reshape(batch_size, self.n_agents, self.n_actions)

        return q_values

    def get_q_total(self, q_values, state, actions):
        """
        Get mixed Q-value for given actions.

        Args:
            q_values: [batch_size, n_agents, n_actions] individual Q-values
            state: [batch_size, state_dim] global state
            actions: [batch_size, n_agents] actions taken
        Returns:
            q_total: [batch_size, 1] mixed Q-value
        """
        batch_size = q_values.shape[0]

        # Select Q-values for taken actions
        actions = actions.long()
        q_selected = torch.gather(q_values, dim=2, index=actions.unsqueeze(2)).squeeze(2)
        # [batch_size, n_agents]

        # Mix Q-values
        q_total = self.mixing_network(q_selected, state)

        return q_total

    def get_greedy_actions(self, obs):
        """
        Get greedy actions for all agents (for decentralized execution).

        Args:
            obs: [n_agents, obs_dim] current observations
        Returns:
            actions: [n_agents] greedy actions
        """
        with torch.no_grad():
            # Add batch dimension
            obs = obs.unsqueeze(0) if len(obs.shape) == 2 else obs

            # Get Q-values
            q_values = self.forward(obs, state=None)  # Don't need state for greedy

            # Select greedy actions
            actions = q_values.argmax(dim=2).squeeze(0)

        return actions