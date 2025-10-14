import os

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.model.agent_handler import Driver, DummyVecEnv
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


def get_obs_of_agent(all_drivers_states, car_index):
    """Flatten observation, taking the first car if multidimensional."""
    if isinstance(all_drivers_states, np.ndarray):
        if len(all_drivers_states.shape) == 2 and all_drivers_states.shape[0] > 1:
            return all_drivers_states[car_index]


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


def combine_agent_obs(car_state, control_signal, expected_dim):
    """Concatenate car state and master control signal, ensuring correct dim."""
    car_state = np.array(car_state).flatten()
    control = np.array(control_signal).flatten()
    agent_obs = np.concatenate((car_state, control))
    if agent_obs.shape[0] != expected_dim:
        if agent_obs.shape[0] > expected_dim:
            agent_obs = agent_obs[:expected_dim]
        else:
            pad = np.zeros(expected_dim - agent_obs.shape[0])
            agent_obs = np.concatenate([agent_obs, pad])
    return agent_obs


def get_scaler_action_and_action_array(car_action):
    # scaler action
    scaler_action =  car_action[0] if (hasattr(car_action, 'shape') and len(car_action.shape) > 0) else \
        (car_action[0] if isinstance(car_action, list) and len(car_action) > 0 else car_action)

    # action array
    if hasattr(car_action, 'shape') and len(car_action.shape) > 0:
        action_array = np.array([[car_action[0]]])
    elif isinstance(car_action, list) and len(car_action) > 0:
        action_array = np.array([[car_action[0]]])
    else:
        action_array = np.array([[car_action]])

    return scaler_action, action_array



def get_agent_values_from_observation(car_observation, car_action_array, agent_model):
    with torch.no_grad():
        obs_tensor = torch.tensor(car_observation, dtype=torch.float32).unsqueeze(0)
        car_values = agent_model.policy.predict_values(obs_tensor)
        dist = agent_model.policy.get_distribution(obs_tensor)
        log_prob = dist.log_prob(torch.tensor(car_action_array, dtype=torch.long))
        return car_values, log_prob


def setup_experiment_dirs(experiment_path):
    """Create experiment directory if it doesn't exist."""
    os.makedirs(experiment_path, exist_ok=True)

def initialize_models(experiment_config, env_config):
    """
    Initialize master and agent models, each with tuned architectures and hyperparameters.
    """
    experiment_config.CONFIG = env_config

    # === MASTER MODEL CONFIGURATION ===
    obs_dim = experiment_config.CARS_AMOUNT * 4
    control_dim = experiment_config.CARS_AMOUNT

    master_model = MasterModel(
        observation_dim=obs_dim,
        action_dim=control_dim,
    )

    # === AGENT MODEL CONFIGsURATION ===
    AGENT_NETWORK_ARCH = {
        'pi': [64, 256],
        'vf': [64, 256]
    }
    AGENT_LR = 1e-3
    AGENT_BATCH_SIZE = 32
    env_fn = lambda: Driver(experiment_config, master_model=master_model)
    wrapped_env = DummyVecEnv([env_fn])

    agent_additional_model_params = {
        'gamma': 0.8,
        'gae_lambda': 0.99,
        'ent_coef': 0.1,
        'clip_range': 0.9,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
    }

    original_define_model_params = Model.define_model_params

    @staticmethod
    def improved_define_model_params(experiment):
        params = original_define_model_params(experiment)
        params.update(agent_additional_model_params)
        params['learning_rate'] = AGENT_LR
        params['batch_size']    = AGENT_BATCH_SIZE
        params['policy_kwargs'] = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=AGENT_NETWORK_ARCH['pi'],
                           vf=AGENT_NETWORK_ARCH['vf'])]
        )
        return params

    Model.define_model_params = improved_define_model_params
    agent_model = Model(wrapped_env, experiment_config).model
    Model.define_model_params = original_define_model_params

    return master_model, agent_model, wrapped_env


# def initialize_models(experiment_config, env_config):
#     """
#     Initialize master and agent models, each with tuned architectures and hyperparameters.
#     """
#     experiment_config.CONFIG = env_config
#
#     # === MASTER MODEL CONFIGURATION ===
#     # - Slightly smaller network to avoid overfitting on the value function.
#     # - Lower learning rate for more stable global value estimation.
#     # - Higher batch size to smooth out the loss curve.
#     MASTER_NETWORK_ARCH = [128, 128]
#     MASTER_LR = 2e-5
#     MASTER_BATCH_SIZE = 128
#
#     master_policy_kwargs = dict(
#         activation_fn=torch.nn.ReLU,
#         net_arch=MASTER_NETWORK_ARCH
#     )
#
#     master_model = MasterModel(
#         embedding_size=experiment_config.EMBEDDING_SIZE,
#         experiment=experiment_config
#     )
#
#     # === AGENT MODEL CONFIGURATION ===
#     # - Larger network to capture complex state/action mapping.
#     # - Slightly higher entropy for exploration.
#     # - Lower learning rate for stability.
#     # - Conservative PPO clipping.
#     AGENT_NETWORK_ARCH = {
#         'pi': [16,32,64,32,16],
#         'vf': [16,32,64,32,16]
#     }
#     AGENT_LR = 0.05
#     AGENT_BATCH_SIZE = 64
#
#     # Environment wrapper using the custom Driver class
#     env_fn = lambda: Driver(experiment_config, master_model=master_model)
#     wrapped_env = DummyVecEnv([env_fn])
#
#     agent_additional_model_params = {
#         'gamma': 0.99,
#         'gae_lambda': 0.95,
#         'ent_coef': 0.2,
#         'clip_range': 0.05
#     }
#
#     # Patch model params definition for agent model
#     original_define_model_params = Model.define_model_params
#
#     @staticmethod
#     def improved_define_model_params(experiment):
#         params = original_define_model_params(experiment)
#         params.update(agent_additional_model_params)
#         params['learning_rate'] = AGENT_LR
#         params['batch_size'] = AGENT_BATCH_SIZE
#         params['policy_kwargs'] = dict(
#             activation_fn=torch.nn.ReLU,
#             net_arch=[dict(
#                 pi=AGENT_NETWORK_ARCH['pi'],
#                 vf=AGENT_NETWORK_ARCH['vf']
#             )]
#         )
#         return params
#
#     # Temporarily override static method for agent initialization
#     Model.define_model_params = improved_define_model_params
#
#     # Initialize agent model with improved parameters
#     agent_model = Model(wrapped_env, experiment_config).model
#
#     # Restore the original static method to avoid side effects
#     Model.define_model_params = original_define_model_params
#
#     return master_model, agent_model, wrapped_env


# def initialize_models(experiment_config, env_config):
#     """Initialize master and agent models, and environment."""
#     # Attach environment configuration for easy access
#     experiment_config.CONFIG = env_config
#     # Create master model
#     master_model = MasterModel(
#         embedding_size=experiment_config.EMBEDDING_SIZE,
#         experiment=experiment_config
#     )
#     # Create env wrapper with Agent handler
#     env_fn = lambda: Driver(experiment_config, master_model=master_model)
#     wrapped_env = DummyVecEnv([env_fn])
#     # Create agent model
#

#     return master_model, agent_model, wrapped_env


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
