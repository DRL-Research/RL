"""SafeR-ADAIM algorithm implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import SafeRADAIMConfig
from .networks import GaussianPolicy, SafeRADAIMNetworks, ValueFunction
from .utils import TrajectoryBatch, conjugate_gradients, flatten_tensors, unflatten_tensors


@dataclass
class PolicyUpdateInfo:
    """Diagnostics from a policy update step."""

    surrogate_reward: float
    surrogate_cost: float
    kl_divergence: float
    case: str
    backtracks: int


class SafeRADAIMAgent:
    """Risk Situation-aware Constrained Policy Optimization agent."""

    def __init__(self, obs_dim: int, action_dim: int, config: SafeRADAIMConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.networks = SafeRADAIMNetworks(
            policy=GaussianPolicy(obs_dim, config.policy_hidden_sizes, action_dim),
            value_fn=ValueFunction(obs_dim, config.value_hidden_sizes),
            cost_value_fn=ValueFunction(obs_dim, config.cost_value_hidden_sizes),
        ).to(self.device)
        self.value_optimizer = optim.Adam(self.networks.value_fn.parameters(), lr=config.value_lr)
        self.cost_value_optimizer = optim.Adam(
            self.networks.cost_value_fn.parameters(), lr=config.cost_value_lr
        )

    # ------------------------------------------------------------------
    # trajectory utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        observation = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action, log_prob, _ = self.networks.policy.sample(observation.unsqueeze(0))
        return action.squeeze(0).cpu().numpy(), float(log_prob.item())

    def evaluate_policy(self, batch: TrajectoryBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.networks.policy.distribution(batch.observations)
        log_probs = dist.log_prob(batch.actions).sum(dim=-1)
        return log_probs, dist

    # ------------------------------------------------------------------
    # training steps
    # ------------------------------------------------------------------
    def update_values(self, batch: TrajectoryBatch) -> Dict[str, float]:
        results: Dict[str, float] = {}
        value_loss_fn = nn.MSELoss()

        value_pred = self.networks.value_fn(batch.observations)
        value_loss = value_loss_fn(value_pred, batch.reward_targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        results["value_loss"] = float(value_loss.item())

        cost_pred = self.networks.cost_value_fn(batch.observations)
        cost_loss = value_loss_fn(cost_pred, batch.cost_targets)
        self.cost_value_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_value_optimizer.step()
        results["cost_value_loss"] = float(cost_loss.item())

        return results

    def update_policy(self, batch: TrajectoryBatch) -> PolicyUpdateInfo:
        observations = batch.observations
        actions = batch.actions
        old_log_probs = batch.log_probs
        reward_adv = batch.reward_advantages
        cost_adv = batch.cost_advantages

        reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-8)
        cost_adv = cost_adv - cost_adv.mean()

        dist = self.networks.policy.distribution(observations)
        dist_old = torch.distributions.Normal(dist.loc.detach(), dist.scale.detach())
        log_probs = dist.log_prob(actions).sum(dim=-1)

        def policy_loss_fn():
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * reward_adv).mean()

        def cost_surrogate():
            ratio = torch.exp(log_probs - old_log_probs)
            return (ratio * cost_adv).mean()

        policy_loss = policy_loss_fn()
        g = flatten_tensors(torch.autograd.grad(policy_loss, self.networks.policy.parameters(), create_graph=True))

        cost_loss = cost_surrogate()
        b = flatten_tensors(torch.autograd.grad(cost_loss, self.networks.policy.parameters(), create_graph=True))

        with torch.no_grad():
            mean_cost = float(batch.costs.mean().item())
        c_hat = mean_cost - self.config.cost_limit

        def hessian_vector_product(v: torch.Tensor) -> torch.Tensor:
            kl = self._kl_divergence(observations, dist_old)
            grads = torch.autograd.grad(kl, self.networks.policy.parameters(), create_graph=True)
            flat_grad = flatten_tensors(grads)
            kl_v = (flat_grad * v).sum()
            hvp = torch.autograd.grad(kl_v, self.networks.policy.parameters())
            return flatten_tensors(hvp) + self.config.damping_coeff * v

        x = conjugate_gradients(hessian_vector_product, g, self.config.cg_iters)
        xHx = torch.dot(x, hessian_vector_product(x))

        y = conjugate_gradients(hessian_vector_product, b, self.config.cg_iters)
        bHb = torch.dot(b, y)
        gHy = torch.dot(g, y)

        c_hat = float(c_hat)
        bHb = float(bHb.item())
        gHx = float(xHx.item())
        gHy = float(gHy.item())

        F = self.config.max_kl - (c_hat**2) / (bHb + 1e-8)

        if c_hat < 0 and F > 0:
            step_dir = torch.sqrt(torch.tensor(2 * self.config.max_kl / (gHx + 1e-8))) * x
            case = "risk_free"
        elif c_hat > 0 and F > 0:
            A = gHx
            B = gHy
            C = bHb
            denom = max(self.config.max_kl - c_hat**2 / (C + 1e-8), 1e-8)
            lam = torch.sqrt(torch.tensor(max(A - (B**2) / (C + 1e-8), 0.0) / denom))
            nu = (B + c_hat * lam) / (C + 1e-8)
            step_dir = lam * (x - nu * y)
            case = "moderate_risk"
        else:
            step_dir = -torch.sqrt(torch.tensor(2 * self.config.max_kl / (bHb + 1e-8))) * y
            case = "high_risk"

        step_dir = step_dir.detach()

        old_params = flatten_tensors(self.networks.policy.parameters()).detach()
        new_params, backtracks, surrogate_reward, surrogate_cost, kl = self._line_search(
            observations,
            actions,
            old_log_probs,
            reward_adv,
            cost_adv,
            dist_old,
            mean_cost,
            old_params,
            step_dir,
            case,
        )

        self._assign_params(new_params)
        return PolicyUpdateInfo(
            surrogate_reward=surrogate_reward,
            surrogate_cost=surrogate_cost,
            kl_divergence=kl,
            case=case,
            backtracks=backtracks,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _assign_params(self, flat: torch.Tensor) -> None:
        params = list(self.networks.policy.parameters())
        for param, value in zip(params, unflatten_tensors(flat, params)):
            param.data.copy_(value.data)

    def _kl_divergence(self, observations: torch.Tensor, dist_old) -> torch.Tensor:
        dist_new = self.networks.policy.distribution(observations)
        kl = torch.distributions.kl_divergence(dist_old, dist_new)
        return kl.mean()

    def _line_search(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        reward_adv: torch.Tensor,
        cost_adv: torch.Tensor,
        dist_old,
        mean_cost: float,
        old_params: torch.Tensor,
        step_dir: torch.Tensor,
        case: str,
    ) -> Tuple[torch.Tensor, int, float, float, float]:
        step = 1.0
        backtracks = 0
        best_reward = -np.inf
        best_cost = np.inf
        best_kl = np.inf
        params = list(self.networks.policy.parameters())
        for backtracks in range(self.config.max_line_search_steps):
            new_params = old_params + step * step_dir
            for param, value in zip(params, unflatten_tensors(new_params, params)):
                param.data.copy_(value.data)

            dist_new = self.networks.policy.distribution(observations)
            new_log_probs = dist_new.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate_reward = float((ratio * reward_adv).mean().item())
            surrogate_cost = float((ratio * cost_adv).mean().item())
            kl = float(torch.distributions.kl_divergence(dist_old, dist_new).mean().item())

            new_cost_estimate = mean_cost + surrogate_cost
            cost_ok = case != "moderate_risk" or new_cost_estimate <= self.config.cost_limit
            if kl <= self.config.max_kl and cost_ok:
                best_reward, best_cost, best_kl = surrogate_reward, surrogate_cost, kl
                return new_params, backtracks, best_reward, best_cost, best_kl
            step *= self.config.line_search_coeff
        return old_params, backtracks + 1, best_reward, best_cost, best_kl

