from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO

from hdmaps.protocol import AGENT_OBS_DIM


class _StaticEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, True, {}


def load_agent(path: str) -> PPO:
    env = _StaticEnv(AGENT_OBS_DIM)
    return PPO.load(path, env=env, device="cpu")


def agent_actions(agent_model: PPO, observations: list[np.ndarray], deterministic: bool) -> list[int]:
    if not observations:
        return []
    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    with torch.no_grad():
        actions, _, _ = agent_model.policy.forward(obs_t, deterministic=deterministic)
    return actions.detach().cpu().numpy().reshape(-1).astype(int).tolist()
