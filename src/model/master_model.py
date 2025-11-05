# Code and comments only in English.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_numpy(data: torch.Tensor | np.ndarray | list | tuple) -> np.ndarray:
    """Utility that converts inputs to float32 numpy arrays."""
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _flatten_state(state: torch.Tensor | np.ndarray) -> np.ndarray:
    """Flatten arbitrary shaped observations into (obs_dim,) arrays."""
    array = _to_numpy(state)
    if array.ndim > 1:
        return array.reshape(-1)
    return array


class AttentionPolicyNetwork(nn.Module):
    """Attention based policy network used by every agent actor."""

    def __init__(
        self,
        per_agent_obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.per_agent_obs_dim = per_agent_obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim

        self.entity_encoder = nn.Sequential(
            nn.Linear(per_agent_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self._last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-agent continuous actions and the attention weights."""

        batch_size = observations.shape[0]
        reshaped = observations.view(batch_size, self.num_agents, self.per_agent_obs_dim)
        encoded = self.entity_encoder(reshaped)

        # Each agent attends to every other agent and itself
        attention_context, attention_weights = self.attention(encoded, encoded, encoded, need_weights=True)
        self._last_attention_weights = attention_weights.detach()

        actions = self.decoder(attention_context)
        return actions, attention_weights

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attention_weights


class CentralCriticNetwork(nn.Module):
    """Centralized critic that evaluates the joint observation-action pair."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs_flat = observations.view(observations.size(0), -1)
        actions_flat = actions.view(actions.size(0), -1)
        critic_input = torch.cat([obs_flat, actions_flat], dim=-1)
        return self.net(critic_input)


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Replay buffer tailored for MADDPG style updates."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        num_agents: int,
    ) -> None:
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self._pos] = observation
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_observation
        self.dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        obs = torch.tensor(self.observations[indices], dtype=torch.float32, device=device)
        acts = torch.tensor(self.actions[indices], dtype=torch.float32, device=device)
        rews = torch.tensor(self.rewards[indices], dtype=torch.float32, device=device)
        next_obs = torch.tensor(self.next_observations[indices], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=device)
        return ReplayBatch(obs, acts, rews, next_obs, dones)

    def clear(self) -> None:
        self._size = 0
        self._pos = 0


class MasterModel:
    """MADDPG style master model with attention based actors."""

    def __init__(
        self,
        observation_dim: int,
        embedding_dim: int,
        num_agents: Optional[int] = None,
        *,
        per_agent_obs_dim: Optional[int] = None,
        attention_hidden_dim: int = 128,
        attention_heads: int = 4,
        critic_hidden_dim: int = 256,
        replay_capacity: int = 50_000,
        batch_size: int = 128,
        gamma: float = 0.99,
        tau: float = 0.01,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        if num_agents is None:
            inferred_agents = observation_dim // 4
            num_agents = max(1, inferred_agents)
        if per_agent_obs_dim is None:
            per_agent_obs_dim = observation_dim // num_agents

        self.observation_dim = observation_dim
        self.embedding_dim = embedding_dim
        self.num_agents = num_agents
        self.per_agent_obs_dim = per_agent_obs_dim

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.training_enabled = True

        self.actor = AttentionPolicyNetwork(
            per_agent_obs_dim=per_agent_obs_dim,
            action_dim=embedding_dim,
            num_agents=num_agents,
            hidden_dim=attention_hidden_dim,
            num_heads=attention_heads,
        ).to(self.device)
        self.critic = CentralCriticNetwork(
            observation_dim=observation_dim,
            action_dim=embedding_dim * num_agents,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)

        self.actor_target = AttentionPolicyNetwork(
            per_agent_obs_dim=per_agent_obs_dim,
            action_dim=embedding_dim,
            num_agents=num_agents,
            hidden_dim=attention_hidden_dim,
            num_heads=attention_heads,
        ).to(self.device)
        self.critic_target = CentralCriticNetwork(
            observation_dim=observation_dim,
            action_dim=embedding_dim * num_agents,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            observation_dim=observation_dim,
            action_dim=embedding_dim,
            num_agents=num_agents,
        )

        self.noise_std = 0.1
        self.logger = None
        self._last_joint_action: Optional[np.ndarray] = None
        self._last_state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API used by the training loop
    # ------------------------------------------------------------------
    def unfreeze(self) -> None:
        self.training_enabled = True
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True
        self.actor.train(True)
        self.critic.train(True)

    def freeze(self) -> None:
        self.training_enabled = False
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        self.actor.eval()
        self.critic.eval()

    def set_logger(self, logger) -> None:  # pragma: no cover - simple setter
        self.logger = logger

    def set_noise_std(self, noise_std: float) -> None:
        self.noise_std = max(0.0, float(noise_std))

    def get_proto_action(
        self, state_tensor: torch.Tensor | np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        state_np = _flatten_state(state_tensor)
        if state_np.shape[0] < self.observation_dim:
            padded = np.zeros(self.observation_dim, dtype=np.float32)
            padded[: state_np.shape[0]] = state_np
            state_np = padded
        elif state_np.shape[0] > self.observation_dim:
            state_np = state_np[: self.observation_dim]
        self._last_state = state_np

        state_batch = torch.tensor(state_np, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            actions, _ = self.actor(state_batch)
        actions = actions.cpu().numpy()[0]

        if self.training_enabled and self.noise_std > 0.0:
            actions = actions + np.random.normal(0.0, self.noise_std, size=actions.shape)
            actions = np.clip(actions, -1.0, 1.0)

        self._last_joint_action = actions.astype(np.float32)

        # Return the ego agent embedding (index 0) for compatibility
        embedding = self._last_joint_action[0]
        dummy_tensor = torch.zeros(1, dtype=torch.float32)
        return embedding, dummy_tensor, dummy_tensor

    def store_transition(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_flat = _flatten_state(state)
        if state_flat.shape[0] < self.observation_dim:
            padded_state = np.zeros(self.observation_dim, dtype=np.float32)
            padded_state[: state_flat.shape[0]] = state_flat
            state_flat = padded_state
        elif state_flat.shape[0] > self.observation_dim:
            state_flat = state_flat[: self.observation_dim]
        next_state_flat = _flatten_state(next_state)
        if next_state_flat.shape[0] < self.observation_dim:
            padded_next = np.zeros(self.observation_dim, dtype=np.float32)
            padded_next[: next_state_flat.shape[0]] = next_state_flat
            next_state_flat = padded_next
        elif next_state_flat.shape[0] > self.observation_dim:
            next_state_flat = next_state_flat[: self.observation_dim]
        actions_arr = _to_numpy(actions).reshape(self.num_agents, self.embedding_dim)
        self.replay_buffer.add(state_flat, actions_arr, float(reward), next_state_flat, bool(done))

    def train_step(self) -> Optional[Tuple[float, float]]:
        if not self.training_enabled:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_actions, _ = self.actor_target(batch.next_observations)
            target_q = self.critic_target(batch.next_observations, next_actions)
            target_values = batch.rewards + self.gamma * (1.0 - batch.dones) * target_q

        current_q = self.critic(batch.observations, batch.actions)
        critic_loss = F.mse_loss(current_q, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actions_pred, _ = self.actor(batch.observations)
        actor_loss = -self.critic(batch.observations, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "config": {
                "observation_dim": self.observation_dim,
                "embedding_dim": self.embedding_dim,
                "num_agents": self.num_agents,
                "per_agent_obs_dim": self.per_agent_obs_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "batch_size": self.batch_size,
                "noise_std": self.noise_std,
            },
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No master model found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])

        config = checkpoint.get("config", {})
        self.gamma = config.get("gamma", self.gamma)
        self.tau = config.get("tau", self.tau)
        self.batch_size = config.get("batch_size", self.batch_size)
        self.noise_std = config.get("noise_std", self.noise_std)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def last_joint_action(self) -> Optional[np.ndarray]:
        return self._last_joint_action

    @property
    def last_state(self) -> Optional[np.ndarray]:
        return self._last_state

    @property
    def last_attention_weights(self) -> Optional[np.ndarray]:
        weights = self.actor.last_attention_weights
        if weights is None:
            return None
        return weights.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

