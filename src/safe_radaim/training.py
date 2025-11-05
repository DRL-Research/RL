"""High level training loop for SafeR-ADAIM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .algorithm import PolicyUpdateInfo, SafeRADAIMAgent
from .config import SafeRADAIMConfig, SafeRADAIMTrainSettings
from .env_wrapper import SafeIntersectionEnv
from .utils import TrajectoryBatch


@dataclass
class TrainingStats:
    epoch: int
    total_reward: float
    total_cost: float
    policy_info: PolicyUpdateInfo
    value_losses: Dict[str, float]


class SafeRADAIMTrainer:
    """Coordinate trajectory collection and model updates."""

    def __init__(
        self,
        env: SafeIntersectionEnv,
        agent: SafeRADAIMAgent,
        config: SafeRADAIMConfig,
        settings: SafeRADAIMTrainSettings | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.settings = settings or SafeRADAIMTrainSettings(total_epochs=config.epochs)

    def train(self) -> List[TrainingStats]:
        stats: List[TrainingStats] = []
        for epoch in range(1, self.settings.total_epochs + 1):
            batch = self._collect_trajectories()
            batch = batch.to(self.agent.device)
            value_losses = self.agent.update_values(batch)
            policy_info = self.agent.update_policy(batch)
            total_reward = float(batch.rewards.sum().item())
            total_cost = float(batch.costs.sum().item())
            stats.append(
                TrainingStats(
                    epoch=epoch,
                    total_reward=total_reward,
                    total_cost=total_cost,
                    policy_info=policy_info,
                    value_losses=value_losses,
                )
            )
        return stats

    def _collect_trajectories(self) -> TrajectoryBatch:
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        logp_list: List[float] = []
        rew_list: List[float] = []
        cost_list: List[float] = []
        done_list: List[float] = []

        obs, _ = self.env.reset()

        for step in range(self.config.steps_per_epoch):
            action, logp = self.agent.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            cost = info.get("risk_cost", 0.0)

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp)
            rew_list.append(reward)
            cost_list.append(cost)
            done_list.append(float(terminated or truncated))

            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset()

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
        act_tensor = torch.tensor(np.array(act_list), dtype=torch.float32)
        logp_tensor = torch.tensor(np.array(logp_list), dtype=torch.float32)
        rew_tensor = torch.tensor(np.array(rew_list), dtype=torch.float32)
        cost_tensor = torch.tensor(np.array(cost_list), dtype=torch.float32)
        done_tensor = torch.tensor(np.array(done_list), dtype=torch.float32)

        reward_returns, reward_advantages = self._estimate_advantages(
            obs_tensor,
            rew_tensor,
            done_tensor,
            self.agent.networks.value_fn,
            self.config.gamma,
            self.config.gae_lambda,
        )
        cost_returns, cost_advantages = self._estimate_advantages(
            obs_tensor,
            cost_tensor,
            done_tensor,
            self.agent.networks.cost_value_fn,
            self.config.cost_gamma,
            self.config.cost_gae_lambda,
        )

        return TrajectoryBatch(
            observations=obs_tensor,
            actions=act_tensor,
            rewards=rew_tensor,
            costs=cost_tensor,
            dones=done_tensor,
            log_probs=logp_tensor,
            reward_advantages=reward_advantages,
            reward_targets=reward_returns,
            cost_advantages=cost_advantages,
            cost_targets=cost_returns,
        )

    def _estimate_advantages(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        value_net: nn.Module,
        gamma: float,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = value_net(observations).detach()
        values = torch.cat([values, torch.zeros(1, device=values.device)])
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages[step] = gae
        targets = advantages + values[:-1]
        return targets.detach(), advantages.detach()

