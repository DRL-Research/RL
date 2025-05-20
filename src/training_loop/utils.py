import os

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.model.agent_handler import Agent, DummyVecEnv
from src.model.master_model import MasterModel
from src.model.model_handler import Model


def ensure_tensor(obs, target_dim=None):
    """Convert obs to a tensor and ensure the correct shape."""
    if isinstance(obs, np.ndarray):
        tensor = torch.tensor(obs, dtype=torch.float32)
        if len(obs.shape) == 3 or (len(obs.shape) == 2 and obs.shape[0] > 1):
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.unsqueeze(0)
    elif isinstance(obs, torch.Tensor):
        tensor = obs
        if len(obs.shape) == 3 or (len(obs.shape) == 2 and obs.shape[0] > 1):
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.unsqueeze(0) if len(obs.shape) == 1 else obs
    else:
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    if target_dim is not None and tensor.shape[1] != target_dim:
        if tensor.shape[1] > target_dim:
            tensor = tensor[:, :target_dim]
        else:
            pad = torch.zeros((1, target_dim - tensor.shape[1]), dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=1)
    return tensor


def flatten_obs(obs, length=4):
    """Flatten observation, taking the first car if multidimensional."""
    if isinstance(obs, np.ndarray):
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            return obs[0].flatten()[:length]
        return obs[:length].flatten()
    elif isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            return obs[0].flatten()[:length]
        return obs[:length].flatten()
    return np.zeros(length)


def combine_agent_obs(car_state, embedding, expected_dim):
    """Concatenate car state and embedding, ensuring correct dim."""
    car_state = np.array(car_state).flatten()
    embedding = np.array(embedding).flatten()
    agent_obs = np.concatenate((car_state, embedding))
    if agent_obs.shape[0] != expected_dim:
        if agent_obs.shape[0] > expected_dim:
            agent_obs = agent_obs[:expected_dim]
        else:
            pad = np.zeros(expected_dim - agent_obs.shape[0])
            agent_obs = np.concatenate([agent_obs, pad])
    return agent_obs


def get_action_array(action):
    """Get action in array format for logging/buffer."""
    if hasattr(action, 'shape') and len(action.shape) > 0:
        return np.array([[action[0]]])
    elif isinstance(action, list) and len(action) > 0:
        return np.array([[action[0]]])
    else:
        return np.array([[action]])


def setup_experiment_dirs(experiment_path):
    """Create experiment directory if it doesn't exist."""
    os.makedirs(experiment_path, exist_ok=True)


def initialize_models(experiment_config, env_config):
    """Initialize master and agent models, and environment."""
    # Attach environment configuration for easy access
    experiment_config.CONFIG = env_config
    # Create master model
    master_model = MasterModel(
        embedding_size=experiment_config.EMBEDDING_SIZE,
        experiment=experiment_config
    )
    # Create env wrapper with Agent handler
    env_fn = lambda: Agent(experiment_config, master_model=master_model)
    wrapped_env = DummyVecEnv([env_fn])
    # Create agent model
    agent_model = Model(wrapped_env, experiment_config).model
    return master_model, agent_model, wrapped_env


def setup_loggers(base_path):
    """Setup and return agent and master loggers."""
    agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
    master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
    return agent_logger, master_logger


def close_everything(env, agent_logger, master_logger):
    """Close environment and loggers."""
    env.close()
    agent_logger.close()
    master_logger.close()


def monitor_episode_results(master_model, agent_model):
    """Prints minimal statistics about the rollout buffers"""
    master_buffer_size = master_model.rollout_buffer.pos
    agent_buffer_size = agent_model.rollout_buffer.pos

    print(f"Current buffer sizes - Master: {master_buffer_size}, Agent: {agent_buffer_size}")

    if master_buffer_size > 0 and hasattr(master_model.rollout_buffer, 'rewards'):
        rewards = master_model.rollout_buffer.rewards[:master_buffer_size]
        print(f"Episode rewards - Min: {rewards.min():.2f}, Max: {rewards.max():.2f}, Avg: {rewards.mean():.2f}")


def monitor_rollout_buffers(master_model, agent_model):
    """Prints statistics about the rollout buffers for debugging"""
    print("\n--- Rollout Buffer Statistics ---")

    # Master buffer stats
    print("Master rollout buffer:")
    print(f"  Buffer size: {master_model.rollout_buffer.buffer_size}")
    print(f"  Current position: {master_model.rollout_buffer.pos}")
    print(f"  Is full: {master_model.rollout_buffer.full}")

    if hasattr(master_model.rollout_buffer, 'observations') and master_model.rollout_buffer.observations is not None:
        print(f"  Observations shape: {master_model.rollout_buffer.observations.shape}")

    if hasattr(master_model.rollout_buffer, 'actions') and master_model.rollout_buffer.actions is not None:
        print(f"  Actions shape: {master_model.rollout_buffer.actions.shape}")

    if hasattr(master_model.rollout_buffer, 'rewards') and master_model.rollout_buffer.rewards is not None:
        rewards = master_model.rollout_buffer.rewards[:master_model.rollout_buffer.pos]
        if len(rewards) > 0:
            print(f"  Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
            print(f"  Rewards: {rewards}")

    # Agent buffer stats
    print("\nAgent rollout buffer:")
    print(f"  Buffer size: {agent_model.rollout_buffer.buffer_size}")
    print(f"  Current position: {agent_model.rollout_buffer.pos}")
    print(f"  Is full: {agent_model.rollout_buffer.full}")

    if hasattr(agent_model.rollout_buffer, 'observations') and agent_model.rollout_buffer.observations is not None:
        print(f"  Observations shape: {agent_model.rollout_buffer.observations.shape}")

    if hasattr(agent_model.rollout_buffer, 'actions') and agent_model.rollout_buffer.actions is not None:
        print(f"  Actions shape: {agent_model.rollout_buffer.actions.shape}")

    if hasattr(agent_model.rollout_buffer, 'rewards') and agent_model.rollout_buffer.rewards is not None:
        rewards = agent_model.rollout_buffer.rewards[:agent_model.rollout_buffer.pos]
        if len(rewards) > 0:
            print(f"  Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
            print(f"  Rewards: {rewards}")

    print("-------------------------------\n")
