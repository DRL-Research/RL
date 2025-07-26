import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Normal

from experiment.experiment_config import Experiment


###############################################
# Master Network Components
###############################################

class CustomMasterNetwork(nn.Module):
    def __init__(self, observation_dim, embedding_dim=4):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Simple, stable architecture - no fancy stuff
        self.shared_net = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            # No activation - let it be unbounded
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        # Simple reshape handling
        if obs.dim() == 2 and obs.shape[0] > 1 and obs.shape[1] == 4:
            obs = obs.reshape(1, -1)

        shared = self.shared_net(obs)
        embedding = self.embedding_head(shared)
        value = self.value_head(shared)
        return embedding, value


class CustomMasterPolicy(ActorCriticPolicy):
    """
    GUARANTEED LEARNING Gaussian policy for master model.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            embedding_dim: int = 4,
            *args,
            **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        obs_dim = observation_space.shape[0]
        self.custom_network = CustomMasterNetwork(obs_dim, embedding_dim)

        # Learnable log-std for REAL Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

        # Override optimizer to include log_std
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def forward(self, obs, deterministic: bool = False):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.ndim == 2 and obs.shape[1] == 4 and obs.shape[0] > 1:
            obs = obs.reshape(1, -1)

        # Get mean embedding and value
        mean, value = self.custom_network(obs)

        # Build REAL Gaussian distribution
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        # Sample or take mean
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, value, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        if obs.ndim == 2 and obs.shape[1] == 4 and obs.shape[0] > 1:
            obs = obs.reshape(1, -1)

        mean, value = self.custom_network(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value, log_prob, entropy

    def get_distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 2 and obs.shape[1] == 4 and obs.shape[0] > 1:
            obs = obs.reshape(1, -1)

        mean, _ = self.custom_network(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def predict_values(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 2 and obs.shape[1] == 4 and obs.shape[0] > 1:
            obs = obs.reshape(1, -1)

        _, value = self.custom_network(obs)
        return value


###############################################
# Master Model Class - GUARANTEED LEARNING
###############################################
class MasterModel:
    """
    Master model with GUARANTEED LEARNING SETTINGS.
    No EMA, no complicated stuff, just solid learning.
    """

    def __init__(self, embedding_size, experiment=None):
        self.embedding_size = embedding_size
        self.experiment = experiment
        self.is_frozen = False

        max_cars = self.experiment.CARS_AMOUNT if experiment else 5
        self.observation_dim = max_cars * 4

        dummy_env = gym.Env()
        dummy_env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,)
        )
        dummy_env.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(embedding_size,), dtype=np.float32
        )

        # SOLID LEARNING HYPERPARAMETERS
        learning_rate = 3e-4  # FIXED: was 0.01 (insane)
        n_steps = Experiment.N_STEPS
        batch_size = 128  # Larger for stability

        self.model = PPO(
            policy=CustomMasterPolicy,
            env=dummy_env,
            verbose=1,
            policy_kwargs={"embedding_dim": embedding_size},
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,  # FIXED: High entropy for strong policy gradients
            vf_coef=0.5,
            max_grad_norm=0.5,  # FIXED: Reasonable gradient clipping
            device="cpu"
        )

        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=1
        )

        # REMOVED ALL EMA - It was interfering with learning

    def unfreeze(self):
        """Enable training for master network"""
        self.is_frozen = False
        for param in self.model.policy.parameters():
            param.requires_grad = True
        self.model.policy.set_training_mode(True)
        print("[MasterModel] UNFROZEN - Training enabled")

    def freeze(self):
        """Disable training for master network"""
        self.is_frozen = True
        for param in self.model.policy.parameters():
            param.requires_grad = False
        self.model.policy.set_training_mode(False)
        print("[MasterModel] FROZEN - Training disabled")

    def get_proto_action(self, state_tensor):
        """
        CLEAN inference - NO EMA interference.
        """
        with torch.no_grad():
            # NO MORE: self.apply_ema() - This was destroying learning!

            # Simple reshaping
            if isinstance(state_tensor, np.ndarray) and state_tensor.shape == (5, 4):
                reshaped_tensor = torch.tensor(
                    state_tensor.reshape(1, -1), dtype=torch.float32
                )
            else:
                reshaped_tensor = torch.tensor(state_tensor, dtype=torch.float32)
                if len(reshaped_tensor.shape) == 1:
                    reshaped_tensor = reshaped_tensor.unsqueeze(0)

            # Ensure correct dimensions
            if reshaped_tensor.shape[1] != self.observation_dim:
                if reshaped_tensor.shape[1] < self.observation_dim:
                    padding = torch.zeros(
                        reshaped_tensor.shape[0],
                        self.observation_dim - reshaped_tensor.shape[1]
                    )
                    reshaped_tensor = torch.cat([reshaped_tensor, padding], dim=1)
                else:
                    reshaped_tensor = reshaped_tensor[:, :self.observation_dim]

            embedding, value, log_prob = self.model.policy.forward(reshaped_tensor)
            return embedding.cpu().numpy()[0], value, log_prob

    def debug_gradients(self):
        """Debug gradient flow - call after loss.backward()"""
        print("\n=== MASTER GRADIENT DEBUG ===")
        total_norm = 0
        param_count = 0
        for name, param in self.model.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                param_count += 1
                print(f"  {name}: grad_norm={grad_norm:.2e}")
            else:
                print(f"  {name}: grad=None (NO GRADIENT!)")

        total_norm = total_norm ** 0.5
        print(f"  TOTAL GRAD NORM: {total_norm:.2e}")
        print(f"  PARAMS WITH GRADIENTS: {param_count}")
        print("==============================\n")

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