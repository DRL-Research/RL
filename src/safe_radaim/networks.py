"""Neural network modules used by SafeR-ADAIM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple fully-connected network with Tanh activations."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.Tanh())
            last_dim = size
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy with state-independent log standard deviation."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], action_dim: int) -> None:
        super().__init__()
        self.net = MLP(input_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):  # noqa: D401
        mean = self.net(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def distribution(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

    def sample(self, x: torch.Tensor):
        dist = self.distribution(x)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist

    def log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(x)
        return dist.log_prob(action).sum(dim=-1)


class ValueFunction(nn.Module):
    """State-value estimator."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.net = MLP(input_dim, hidden_sizes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


@dataclass
class SafeRADAIMNetworks:
    """Container bundling the policy, reward value and cost value networks."""

    policy: GaussianPolicy
    value_fn: ValueFunction
    cost_value_fn: ValueFunction

    def to(self, device: torch.device | str) -> "SafeRADAIMNetworks":
        device = torch.device(device)
        self.policy.to(device)
        self.value_fn.to(device)
        self.cost_value_fn.to(device)
        return self
