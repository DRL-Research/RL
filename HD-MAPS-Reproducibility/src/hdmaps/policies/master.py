from __future__ import annotations

import os
import sys
import types

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from hdmaps.protocol import EMBEDDING_DIM, MASTER_OBS_DIM


class SimpleResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.input_layer = nn.Linear(input_dim, 128)
        self.res_block1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.res_block2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.output_layer = nn.Linear(128, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        x = self.relu(self.input_layer(observations))
        residual1 = x
        x = self.relu(self.res_block1(x) + residual1)
        residual2 = x
        x = self.relu(self.res_block2(x) + residual2)
        return self.output_layer(x)


class _MasterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, emb_dim: int, max_steps: int = 128):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(emb_dim,), dtype=np.float32)
        self._obs_dim = obs_dim
        self._emb_dim = emb_dim
        self._max_steps = max_steps
        self._rng = np.random.default_rng()
        self._state = np.zeros(obs_dim, dtype=np.float32)
        self._steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._state = self._rng.standard_normal(self._obs_dim).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        self._steps += 1
        terminated = False
        truncated = self._steps >= self._max_steps
        return self._state.copy(), 0.0, terminated, truncated, {}


def install_legacy_module() -> None:
    mod = types.ModuleType("src.model.master_model")
    mod.SimpleResNetExtractor = SimpleResNetExtractor
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.model", types.ModuleType("src.model"))
    sys.modules["src.model.master_model"] = mod
    sys.modules["src.model.master_model"].SimpleResNetExtractor = SimpleResNetExtractor


class MasterModel:
    NORMALIZE_INPUTS = False
    DETERMINISTIC_PROTO = False

    def __init__(self, embedding_size=EMBEDDING_DIM, observation_dim=MASTER_OBS_DIM, **kwargs):
        self.embedding_size = int(embedding_size)
        self.observation_dim = int(observation_dim)
        env = _MasterEnv(self.observation_dim, self.embedding_size)
        policy_kwargs = dict(
            features_extractor_class=SimpleResNetExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[128, 256, 128],
        )
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=64,
            gamma=0.9,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu",
        )

    def load(self, path: str) -> None:
        install_legacy_module()
        self.model = PPO.load(path, device="cpu")
        parent = os.path.dirname(os.path.abspath(path))
        sidecar = os.path.join(parent, "master_custom_params.pt")
        if os.path.exists(sidecar):
            params = torch.load(sidecar, map_location="cpu", weights_only=False)
            if "observation_dim" in params:
                self.observation_dim = int(params["observation_dim"])
            if "embedding_size" in params:
                self.embedding_size = int(params["embedding_size"])

    def save(self, path: str) -> None:
        self.model.save(path)

    def embeddings(self, inputs: list[np.ndarray], deterministic: bool) -> list[np.ndarray]:
        actions, _, _ = self.act(inputs, deterministic)
        return [actions[i].reshape(-1)[: self.embedding_size] for i in range(actions.shape[0])]

    def act(self, inputs: list[np.ndarray], deterministic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not inputs:
            empty = np.zeros((0, self.embedding_size), dtype=np.float32)
            return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        obs_t = torch.as_tensor(np.asarray(inputs, dtype=np.float32), device=self.model.device)
        with torch.no_grad():
            actions, values, log_probs = self.model.policy.forward(obs_t, deterministic=deterministic)
        act = actions.detach().cpu().numpy().astype(np.float32)
        val = values.detach().cpu().numpy().reshape(-1).astype(np.float32)
        lp = log_probs.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return act, val, lp


def load_models(agent_path: str, master_path: str):
    from hdmaps.policies.agent import load_agent

    master = MasterModel()
    master.load(master_path)
    agent = load_agent(agent_path)
    return master, agent
