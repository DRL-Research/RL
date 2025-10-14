import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.model.master_model import MasterModel


def ensure_tensor(obs, target_dim: Optional[int] = None) -> torch.Tensor:
    """Convert obs to a tensor and ensure the correct shape."""
    if isinstance(obs, np.ndarray):
        tensor = torch.tensor(obs, dtype=torch.float32)
        if tensor.ndim > 1:
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.unsqueeze(0)
    elif isinstance(obs, torch.Tensor):
        tensor = obs.float()
        if tensor.ndim > 1 and tensor.shape[0] > 1:
            tensor = tensor.reshape(1, -1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
    else:
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    if target_dim is not None and tensor.shape[1] != target_dim:
        if tensor.shape[1] > target_dim:
            tensor = tensor[:, :target_dim]
        else:
            pad = torch.zeros((tensor.shape[0], target_dim - tensor.shape[1]), dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=1)
    return tensor


def setup_experiment_dirs(experiment_path: str) -> None:
    """Create experiment directory if it doesn't exist."""
    os.makedirs(experiment_path, exist_ok=True)


def initialize_models(experiment_config, env_config):
    """Initialize the master model and the Highway environment."""
    experiment_config.CONFIG = env_config

    obs_dim = experiment_config.CARS_AMOUNT * 4
    action_dim = experiment_config.CARS_AMOUNT

    master_model = MasterModel(
        observation_dim=obs_dim,
        action_dim=action_dim,
        experiment=experiment_config,
    )

    env = gym.make(
        'RELintersection-v0',
        render_mode=experiment_config.RENDER_MODE,
        config=env_config,
    )

    return master_model, env


def setup_loggers(base_path: str):
    """Setup and return the master logger."""
    master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
    return master_logger


def close_everything(env, master_logger) -> None:
    """Close environment and logger."""
    env.close()
    master_logger.close()
