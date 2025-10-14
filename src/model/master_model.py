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


class MasterModel:
    """
    PURE SB3 PPO - NO CUSTOM ANYTHING
    """

    def __init__(self, action_dim=None, experiment=None, observation_dim=None, ppo_hyperparams=None, **kwargs):
        # Accept all possible arguments for compatibility
        self.experiment = experiment
        self.is_frozen = False

        if action_dim is not None:
            self.action_dim = action_dim
        elif experiment is not None and hasattr(experiment, 'CARS_AMOUNT'):
            self.action_dim = experiment.CARS_AMOUNT
        else:
            self.action_dim = 4

        # Calculate observation dimension
        if observation_dim is not None:
            self.observation_dim = observation_dim
        elif experiment is not None:
            max_cars = experiment.CARS_AMOUNT if hasattr(experiment, 'CARS_AMOUNT') else 5
            self.observation_dim = max_cars * 4
        else:
            self.observation_dim = 20  # Default 5 cars * 4 features

        # Create dummy environment for PPO
        dummy_env = gym.Env()
        dummy_env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        dummy_env.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Get n_steps safely
        try:
            n_steps = experiment.N_STEPS if experiment and hasattr(experiment, 'N_STEPS') else 64
        except:
            n_steps = 64

        # PURE SB3 PPO - NOTHING CUSTOM
        default_policy_kwargs = dict(
            features_extractor_class=SimpleResNetExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[64, 32]
        )

        provided_params = dict(ppo_hyperparams or {})
        net_arch_override = provided_params.pop("net_arch", None)

        resolved_n_steps = provided_params.get("n_steps", n_steps)

        default_hyperparams = {
            "learning_rate": 1e-2,
            "n_steps": resolved_n_steps,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.9,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        }

        default_hyperparams.update(provided_params)

        policy_kwargs = default_policy_kwargs.copy()
        if net_arch_override is not None:
            policy_kwargs = policy_kwargs.copy()
            policy_kwargs["net_arch"] = net_arch_override

        default_hyperparams["policy_kwargs"] = policy_kwargs

        self.ppo_hyperparams = default_hyperparams

        self.model = PPO(
            "MlpPolicy",
            dummy_env,
            verbose=1,
            device="cpu",
            **self.ppo_hyperparams,
        )



        # self.model = PPO(
        #     "MlpPolicy",  # Standard MLP policy
        #     dummy_env,
        #     learning_rate=1e-2,
        #     n_steps=n_steps,
        #     batch_size=128,
        #     gamma=0.99,
        #     gae_lambda=0.95,
        #     clip_range=0.6,
        #     ent_coef=0.01,
        #     vf_coef=0.5,
        #     policy_kwargs=dict(
        #         net_arch=[128, 256,128,32]  # Simple 2-layer network
        #     ),
        #     verbose=1,
        #     device="cpu"
        # )

        # Rollout buffer for compatibility
        self.rollout_buffer = RolloutBuffer(
            buffer_size=int(self.ppo_hyperparams["n_steps"]),
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            gamma=self.ppo_hyperparams["gamma"],
            gae_lambda=self.ppo_hyperparams["gae_lambda"],
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
        Get master control signals (acceleration or deceleration suggestions)
        using standard SB3 predict with proper tensor handling.
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
            action_np = action.cpu().numpy()[0]

            if action_np.shape[0] < self.action_dim:
                padded_action = np.zeros(self.action_dim, dtype=action_np.dtype)
                padded_len = action_np.shape[0]
                padded_action[:padded_len] = action_np
                action_np = padded_action
            elif action_np.shape[0] > self.action_dim:
                action_np = action_np[:self.action_dim]

            # Keep value and log_prob as tensors
            value_tensor = value.cpu()
            log_prob_tensor = log_prob.cpu()

        return action_np, value_tensor, log_prob_tensor

    def set_logger(self, logger):
        """Set logger for master network"""
        self.model.set_logger(logger)

    def save(self, path):
        """Save master network"""
        self.model.save(path)
        custom_params = {
            "action_dim": self.action_dim,
        }
        torch.save(custom_params, f"{path}_custom_params.pt")

    def load(self, path):
        """Load master network"""
        self.model = PPO.load(path)
        if os.path.exists(f"{path}_custom_params.pt"):
            custom_params = torch.load(f"{path}_custom_params.pt")
            self.action_dim = custom_params.get("action_dim", self.action_dim)
