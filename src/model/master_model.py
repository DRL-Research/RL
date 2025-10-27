# Code and comments only in English.

import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from experiment.experiment_config import Experiment
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

###############################################
# MASTER MODEL - PURE SB3 PPO
###############################################
class SimpleResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        # Input layer
        self.input_layer = nn.Linear(input_dim, 128)

        # ResNet blocks
        self.res_block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Output
        self.output_layer = nn.Linear(128, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        x = self.relu(self.input_layer(observations))

        # ResNet skip connections
        residual1 = x
        x = self.res_block1(x) + residual1  # Skip connection
        x = self.relu(x)

        residual2 = x
        x = self.res_block2(x) + residual2  # Skip connection
        x = self.relu(x)

        return self.output_layer(x)


###############################################
# Minimal real Gymnasium Env (replaces the dummy)
###############################################
class _MasterEnv(gym.Env):
    """
    Minimal working environment so PPO can actually collect rollouts.
    Replace the synthetic dynamics/reward with your simulator when ready.
    """
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, emb_dim: int, max_steps: int = 128, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(emb_dim,), dtype=np.float32)

        self._obs_dim = obs_dim
        self._emb_dim = emb_dim
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(self._obs_dim, dtype=np.float32)
        self._steps = 0

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        # synthetic initial state
        self._state = self._rng.standard_normal(self._obs_dim).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32).reshape(self._emb_dim)

        # synthetic reward: penalize large embeddings (placeholder)
        reward = -float(np.linalg.norm(action))

        # simple controllable dynamics: blend action (padded) into state + noise
        pad = np.zeros(self._obs_dim, dtype=np.float32)
        pad[: self._emb_dim] = action
        noise = self._rng.standard_normal(self._obs_dim).astype(np.float32) * 0.05
        self._state = (0.9 * self._state + 0.1 * pad + noise).astype(np.float32)

        terminated = False
        truncated = self._steps >= self._max_steps
        return self._state.copy(), reward, terminated, truncated, {}


class MasterModel:
    """
    PURE SB3 PPO - NO CUSTOM ANYTHING
    """

    def __init__(self, embedding_size=4, experiment=None, observation_dim=None, **kwargs):
        # Accept all possible arguments for compatibility

        if "embedding_dim" in kwargs and kwargs["embedding_dim"] is not None:
                   embedding_size = int(kwargs.pop("embedding_dim"))
        self.embedding_size = embedding_size
        self.experiment = experiment
        self.is_frozen = False

        # Calculate observation dimension
        if observation_dim is not None:
            self.observation_dim = observation_dim
        elif experiment is not None:
            max_cars = experiment.CARS_AMOUNT if hasattr(experiment, 'CARS_AMOUNT') else 5
            self.observation_dim = max_cars * 4
        else:
            self.observation_dim = 20  # Default 5 cars * 4 features

        # Create a real minimal environment (replaces the dummy)
        env = _MasterEnv(obs_dim=self.observation_dim, emb_dim=embedding_size)

        # Get n_steps safely
        try:
            n_steps = experiment.N_STEPS if experiment and hasattr(experiment, 'N_STEPS') else 64
        except:
            n_steps = 64

        # PURE SB3 PPO - NOTHING CUSTOM
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=n_steps,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=dict(
                features_extractor_class=SimpleResNetExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[128,256,128]
            ),
            verbose=1,
            device="cpu"
        )

        # Rollout buffer for compatibility (kept as-is, just uses env spaces)
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=1
        )

    def unfreeze(self):
        """Enable training for master network"""
        self.is_frozen = False
        for param in self.model.policy.parameters():
            param.requires_grad = True
        self.model.policy.set_training_mode(True)
        print("[PureSB3Master] UNFROZEN - Training enabled")

    def freeze(self):
        """Disable training for master network"""
        self.is_frozen = True
        for param in self.model.policy.parameters():
            param.requires_grad = False
        self.model.policy.set_training_mode(False)
        print("[PureSB3Master] FROZEN - Training disabled")

    def get_proto_action(self, state_tensor):
        """
        Get embedding using standard SB3 predict with proper tensor handling
        """
        # Convert to the format SB3 expects
        if isinstance(state_tensor, np.ndarray) and state_tensor.shape == (5, 4):
            obs = state_tensor.flatten()
        else:
            obs = np.array(state_tensor).flatten()

        # Ensure correct size
        if len(obs) < self.observation_dim:
            padded_obs = np.zeros(self.observation_dim)
            padded_obs[:len(obs)] = obs
            obs = padded_obs
        elif len(obs) > self.observation_dim:
            obs = obs[:self.observation_dim]

        # Convert to tensor for policy evaluation
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Use the policy directly to get action, value, and log_prob
            action, value, log_prob = self.model.policy.forward(obs_tensor)

            # Convert to numpy for the action (embedding)
            action_np, _ = self.model.predict(obs, deterministic=False)
            action_np = np.asarray(action_np).reshape(-1)

            value_tensor = self.model.policy.predict_values(obs_tensor)
            dist = self.model.policy.get_distribution(obs_tensor)
            log_prob_tensor = dist.log_prob(torch.tensor(action_np, dtype=torch.float32).unsqueeze(0))

        return action_np, value_tensor, log_prob_tensor

    def set_logger(self, logger):
        """Set logger for master network"""
        self.model.set_logger(logger)

    def save(self, path):
        """Save master network"""
        self.model.save(path)
        custom_params = {
            "embedding_size": self.embedding_size,
        }
        torch.save(custom_params, f"{path}_custom_params.pt")

    def load(self, path):
        """Load master network"""
        self.model = PPO.load(path)
        if os.path.exists(f"{path}_custom_params.pt"):
            custom_params = torch.load(f"{path}_custom_params.pt")
            self.embedding_size = custom_params["embedding_size"]
