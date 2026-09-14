from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from hdmaps.protocol import EMBEDDING_DIM, MASTER_OBS_DIM, TRAIN_ENT_COEF, TRAIN_GAMMA, TRAIN_VF_COEF


def _buffer() -> RolloutBuffer:
    return RolloutBuffer(
        buffer_size=256,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(MASTER_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(EMBEDDING_DIM,), dtype=np.float32),
        gamma=TRAIN_GAMMA,
        gae_lambda=0.90,
        n_envs=1,
    )


def update_master(master, stream: list[dict]) -> dict | None:
    if len(stream) < 2:
        return None
    buf = _buffer()
    last_obs = None
    last_done = np.array([1.0], dtype=np.float32)
    for i, tr in enumerate(stream):
        last_obs = tr["obs"]
        episode_start = np.array([1.0 if i == 0 else 0.0], dtype=np.float32)
        buf.add(
            obs=tr["obs"].reshape(1, -1),
            action=tr["action"].reshape(1, -1),
            reward=np.array([tr["reward"]], dtype=np.float32),
            episode_start=episode_start,
            value=torch.as_tensor([[tr["value"]]], dtype=torch.float32),
            log_prob=torch.as_tensor([tr["log_prob"]], dtype=torch.float32),
        )
        if buf.full:
            break
    if buf.pos == 0:
        return None
    policy = master.model.policy
    with torch.no_grad():
        last_value = policy.predict_values(torch.as_tensor(last_obs.reshape(1, -1), dtype=torch.float32))
    buf.compute_returns_and_advantage(last_values=last_value, dones=last_done)
    orig_full = buf.full
    buf.full = True
    try:
        data = buf._get_samples(np.arange(buf.pos))
    finally:
        buf.full = orig_full
    obs = data.observations if torch.is_tensor(data.observations) else torch.as_tensor(data.observations, dtype=torch.float32)
    act = data.actions if torch.is_tensor(data.actions) else torch.as_tensor(data.actions, dtype=torch.float32)
    if obs.ndim > 2:
        obs = obs.reshape(obs.shape[0], -1)
    if act.ndim > 2:
        act = act.reshape(act.shape[0], -1)
    values, log_probs, entropy = policy.evaluate_actions(obs, act)
    n = int(obs.shape[0])
    returns = torch.as_tensor(data.returns, dtype=torch.float32).reshape(-1)[:n]
    adv = torch.as_tensor(data.advantages, dtype=torch.float32).reshape(-1)[:n]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    value_loss = ((values.reshape(-1)[:n] - returns) ** 2).mean()
    policy_loss = -(log_probs.reshape(-1)[:n] * adv).mean()
    entropy_loss = -entropy.mean() if entropy is not None else obs.new_tensor(0.0)
    loss = policy_loss + TRAIN_VF_COEF * value_loss + TRAIN_ENT_COEF * entropy_loss
    policy.optimizer.zero_grad()
    if torch.isfinite(loss).all():
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()
    return {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "total_loss": float(loss.detach()),
    }
