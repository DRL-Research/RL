"""Utility helpers for SafeR-ADAIM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def unflatten_tensors(flat: torch.Tensor, template: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    outputs = []
    offset = 0
    for tensor in template:
        numel = tensor.numel()
        outputs.append(flat[offset : offset + numel].view_as(tensor))
        offset += numel
    return tuple(outputs)


def conjugate_gradients(Avp_fn, b: torch.Tensor, nsteps: int, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def compute_gae(returns: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(returns)
    lastgaelam = 0
    for t in reversed(range(len(returns))):
        delta = returns[t] + gamma * values[t + 1] - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * lastgaelam
    targets = advantages + values[:-1]
    return advantages, targets


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if torch.isclose(var_y, torch.tensor(0.0)):
        return 0.0
    return 1 - torch.var(y_true - y_pred) / var_y


@dataclass
class TrajectoryBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    reward_advantages: torch.Tensor
    reward_targets: torch.Tensor
    cost_advantages: torch.Tensor
    cost_targets: torch.Tensor

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        device = torch.device(device)
        return TrajectoryBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            costs=self.costs.to(device),
            dones=self.dones.to(device),
            log_probs=self.log_probs.to(device),
            reward_advantages=self.reward_advantages.to(device),
            reward_targets=self.reward_targets.to(device),
            cost_advantages=self.cost_advantages.to(device),
            cost_targets=self.cost_targets.to(device),
        )

