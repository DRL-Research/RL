import os

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.model.agent_handler import AttentionObservationEncoder, Driver, DummyVecEnv
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
        # action must be float for continuous distributions
        log_prob = dist.log_prob(torch.tensor(car_action_array, dtype=torch.float32))
        return car_values, log_prob



def setup_experiment_dirs(experiment_path):
    """Create experiment directory if it doesn't exist."""
    os.makedirs(experiment_path, exist_ok=True)

def initialize_models(experiment_config, env_config):
    """Initialize the agent model and wrapped environment."""
    experiment_config.CONFIG = env_config

    env_fn = lambda: Driver(experiment_config)
    wrapped_env = DummyVecEnv([env_fn])

    per_agent_obs_dim = experiment_config.AGENT_STATE_SIZE
    observation_dim = int(np.prod(wrapped_env.observation_space.shape))
    if observation_dim % per_agent_obs_dim != 0:
        raise ValueError(
            "Observation dimension is not divisible by per-agent observation size. "
            f"Got observation_dim={observation_dim} and per_agent_obs_dim={per_agent_obs_dim}."
        )
    num_agents = observation_dim // per_agent_obs_dim

    agent_additional_model_params = {
        'n_steps': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.0,
        'clip_range': 0.1,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'n_epochs': 10
    }

    attention_policy_kwargs = dict(
        features_extractor_class=AttentionObservationEncoder,
        features_extractor_kwargs=dict(
            num_agents=num_agents,
            per_agent_obs_dim=per_agent_obs_dim,
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=torch.nn.ReLU,
    )

    original_define_model_params = Model.define_model_params

    @staticmethod
    def improved_define_model_params(experiment):
        params = original_define_model_params(experiment)
        params.update(agent_additional_model_params)
        params['learning_rate'] = 7e-4
        params['batch_size'] = 128
        params['policy_kwargs'] = attention_policy_kwargs
        return params

    Model.define_model_params = improved_define_model_params
    agent_model = Model(wrapped_env, experiment_config).model
    Model.define_model_params = original_define_model_params

    return agent_model, wrapped_env


def setup_loggers(base_path):
    """Setup and return the agent logger."""
    agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
    return agent_logger


def close_everything(env, agent_logger):
    """Close environment and loggers."""
    env.close()
    agent_logger.close()
