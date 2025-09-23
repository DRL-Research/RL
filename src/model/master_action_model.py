import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from src.model.master_model import SimpleResNetExtractor


class ActionMasterModel:
    """Master network that outputs acceleration/deceleration decisions."""

    def __init__(self, action_dim=2, experiment=None, observation_dim=None, **kwargs):
        self.action_dim = action_dim
        self.experiment = experiment
        self.is_frozen = False

        if observation_dim is not None:
            self.observation_dim = observation_dim
        elif experiment is not None:
            max_cars = getattr(experiment, "CARS_AMOUNT", 5)
            self.observation_dim = max_cars * 4
        else:
            self.observation_dim = 20

        dummy_env = gym.Env()
        dummy_env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        dummy_env.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        try:
            n_steps = experiment.N_STEPS if experiment and hasattr(experiment, "N_STEPS") else 64
        except Exception:
            n_steps = 64

        self.model = PPO(
            "MlpPolicy",
            dummy_env,
            learning_rate=1e-2,
            n_steps=n_steps,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.9,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=dict(
                features_extractor_class=SimpleResNetExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[64, 32],
            ),
            verbose=1,
            device="cpu",
        )

        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=1,
        )

    def unfreeze(self):
        self.is_frozen = False
        for param in self.model.policy.parameters():
            param.requires_grad = True
        self.model.policy.set_training_mode(True)
        print("[ActionMaster] UNFROZEN - Training enabled")

    def freeze(self):
        self.is_frozen = True
        for param in self.model.policy.parameters():
            param.requires_grad = False
        self.model.policy.set_training_mode(False)
        print("[ActionMaster] FROZEN - Training disabled")

    def get_proto_action(self, state_tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        if isinstance(state_tensor, np.ndarray) and state_tensor.shape == (5, 4):
            obs = state_tensor.flatten()
        else:
            obs = np.array(state_tensor).flatten()

        if len(obs) < self.observation_dim:
            padded_obs = np.zeros(self.observation_dim)
            padded_obs[: len(obs)] = obs
            obs = padded_obs
        elif len(obs) > self.observation_dim:
            obs = obs[: self.observation_dim]

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, value, log_prob = self.model.policy.forward(obs_tensor)
            action_np = action.cpu().numpy()[0]
            value_tensor = value.cpu()
            log_prob_tensor = log_prob.cpu()

        return action_np, value_tensor, log_prob_tensor

    @staticmethod
    def to_discrete_actions(action_values, threshold=0.0, action_dim=None):
        arr = np.array(action_values).flatten()
        if action_dim is not None and arr.size < action_dim:
            arr = np.pad(arr, (0, action_dim - arr.size))
        if action_dim is not None and arr.size > action_dim:
            arr = arr[:action_dim]
        discrete = (arr >= threshold).astype(int)
        return discrete

    def set_logger(self, logger):
        self.model.set_logger(logger)

    def save(self, path):
        self.model.save(path)
        custom_params = {
            "action_dim": self.action_dim,
        }
        torch.save(custom_params, f"{path}_custom_params.pt")

    def load(self, path):
        self.model = PPO.load(path)
        if os.path.exists(f"{path}_custom_params.pt"):
            custom_params = torch.load(f"{path}_custom_params.pt")
            self.action_dim = custom_params.get("action_dim", self.action_dim)
