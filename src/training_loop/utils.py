import os

from stable_baselines3.common.logger import configure

from src.model.agent_handler import Agent, DummyVecEnv
from src.model.master_model import MasterModel
from src.model.model_handler import Model


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
