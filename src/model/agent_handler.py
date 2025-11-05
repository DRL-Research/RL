import random

import gymnasium as gym
import numpy as np
import torch
from gym import spaces as gym_spaces
import warnings

from gymnasium import spaces

warnings.filterwarnings("ignore")

class Driver(gym.Env):
    """
    Agent environment wrapper for Highway intersection.

    This wrapper interfaces with the Highway environment and integrates
    the master model's embedding into the agent's observation.

    The agent controls only one car (car1) and receives information about
    all cars in the environment through the master embedding.
    """

    def __init__(self, experiment, master_model=None):
        super().__init__()
        self.experiment = experiment
        self.master_model = master_model

        # Load environment configuration
        self.config = experiment.CONFIG if hasattr(experiment, 'CONFIG') else None

        # Define action and observation spaces. The agent now outputs a joint
        # action for all controlled vehicles at each step, therefore the action
        # space becomes MultiDiscrete with one discrete action per vehicle.
        if self.config and "controlled_cars" in self.config:
            self.num_cars = len(self.config["controlled_cars"])
        else:
            self.num_cars = getattr(self.experiment, "CARS_AMOUNT", 1)

        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_cars, experiment.ACTION_SPACE_SIZE, dtype=np.int64)
        )

        # Observation space: concatenated per-vehicle observation (4-dim state +
        # shared embedding) for all controlled vehicles.
        per_vehicle_obs_dim = experiment.STATE_INPUT_SIZE
        observation_dim = self.num_cars * per_vehicle_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )

        # Track episode information
        self.episode_step = 0
        self.total_episode_reward = 0

        # Environment state
        self.current_state = None
        self.current_embedding = None

        # Create the underlying Highway environment
        self.highway_env = gym.make('RELintersection-v0', render_mode=experiment.RENDER_MODE, config=self.config)

    def _get_unwrapped_env(self):
        env = self.highway_env
        while hasattr(env, 'env') and not hasattr(env, 'controlled_vehicles'):
            env = env.env
        return env


    @staticmethod
    def get_action(model, observation, step_counter, exploration_threshold):
        """Return the joint action for all vehicles."""
        if step_counter < exploration_threshold:
            # Random exploration: sample once from the joint action space
            joint_action = model.action_space.sample()
        else:
            # Policy exploitation: predict deterministically from the current observation
            action, _ = model.predict(observation, deterministic=True)
            joint_action = np.asarray(action)

        joint_action = np.asarray(joint_action).astype(np.int64).flatten()
        return joint_action


    def _prepare_state_for_master(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        elif not isinstance(state, np.ndarray):
            state = np.array(state)

        if len(state.shape) == 1:
            if state.shape[0] % 4 == 0:
                num_cars = state.shape[0] // 4
                state = state.reshape(num_cars, 4)
            else:
                target_size = ((state.shape[0] + 3) // 4) * 4
                padded_state = np.zeros(target_size)
                padded_state[:state.shape[0]] = state
                num_cars = target_size // 4
                state = padded_state.reshape(num_cars, 4)

        # Replace arrived vehicle states with zeros
        env = self._get_unwrapped_env()
        if hasattr(env, 'controlled_vehicles'):
            for i, vehicle in enumerate(env.controlled_vehicles):
                if i < state.shape[0]:
                    if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                        state[i] = [0.0, 0.0, 0.0, 0.0]
                        #print(f"Master: Sending zeros for arrived vehicle {i}")

        return state

    def reset(self, **kwargs):
        self.episode_step, self.total_episode_reward = 0, 0

        current_state, info = self.highway_env.reset(**kwargs)
        self.current_state = current_state

        env = self._get_unwrapped_env()

        # Clean arrival flags for new episode
        if hasattr(env, 'controlled_vehicles'):
            for vehicle in env.controlled_vehicles:
                if hasattr(vehicle, 'has_been_parked'):
                    delattr(vehicle, 'has_been_parked')
                if hasattr(vehicle, 'is_arrived'):
                    delattr(vehicle, 'is_arrived')

        current_state = self._prepare_state_for_master(current_state)

        if self.master_model is not None:
            master_input = torch.tensor(current_state.reshape(1, -1), dtype=torch.float32)
            embedding, _, _ = self.master_model.get_proto_action(master_input)
            self.current_embedding = embedding
        else:
            raise ValueError("master not available")

        # Build agent_observations
        self.num_cars = len(env.controlled_vehicles)
        embedding_array = np.asarray(self.current_embedding, dtype=np.float32)
        agent_observations = []  # TODO: duplicate code, move to utils

        for car_index in range(self.num_cars):
            if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                    hasattr(env.controlled_vehicles[car_index], 'is_arrived') and env.controlled_vehicles[car_index].is_arrived):
                car_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                print('The', env.controlled_vehicles[car_index], 'Arrived and sending : ', car_state)
            else:
                if len(current_state.shape) == 1:
                    car_state = current_state[car_index * 4:car_index * 4 + 4]
                else:
                    car_state = current_state[car_index]
                car_state = np.asarray(car_state, dtype=np.float32)

            agent_observations.append(np.concatenate((car_state, embedding_array)))

        stacked_obs = np.stack(agent_observations, axis=0).astype(np.float32)
        flat_obs = stacked_obs.reshape(-1)

        return flat_obs, info

    def step(self, action_tuple):
        """Execute action and return observations for both agents."""
        self.episode_step += 1

        next_state, reward, done, truncated, info = self.highway_env.step(action_tuple)
        next_state = self._prepare_state_for_master(next_state)

        if self.master_model is not None:
            master_input = torch.tensor(next_state.reshape(1, -1), dtype=torch.float32)
            embedding, _, _ = self.master_model.get_proto_action(master_input)
            self.current_embedding = embedding
        else:
            self.current_embedding = np.zeros(4)

        env = self._get_unwrapped_env()

        # Build agent_observations
        agent_next_observations = []  # TODO: duplicate code, move to utils
        embedding_array = np.asarray(self.current_embedding, dtype=np.float32)

        for car_index in range(len(env.controlled_vehicles)):

            if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                    hasattr(env.controlled_vehicles[car_index], 'is_arrived') and env.controlled_vehicles[
                        car_index].is_arrived):
                car_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                print('The', env.controlled_vehicles[car_index], 'Arrived and sending : ', car_state)
            else:
                if len(next_state.shape) == 1:
                    car_state = next_state[car_index * 4:car_index * 4 + 4]
                else:
                    car_state = next_state[car_index]
                car_state = np.asarray(car_state, dtype=np.float32)

            agent_next_observations.append(np.concatenate((car_state, embedding_array)))

        stacked_next_obs = np.stack(agent_next_observations, axis=0).astype(np.float32)
        flat_next_obs = stacked_next_obs.reshape(-1)

        self.current_state = next_state
        self.total_episode_reward += reward

        # Return same format as reset() - both observations
        return flat_next_obs, reward, done, truncated, info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        return self.highway_env.render()

    def close(self):
        """
        Close the environment.
        """
        return self.highway_env.close()

    def seed(self, seed=None):
        """
        Set random seed.
        """
        if hasattr(self.highway_env, 'seed'):
            return self.highway_env.seed(seed)
        return None


class DummyVecEnv(gym.Wrapper):
    """
    A simplified version of stable_baselines3.common.vec_env.DummyVecEnv
    that works with our Agent wrapper.
    """

    def __init__(self, env_fns):
        """
        env_fns: list of functions that create environments to run in parallel
        """
        self.envs = [env_fn() for env_fn in env_fns]
        self.env = self.envs[0]
        super().__init__(self.env)

        # Set up attributes for compatibility
        self.num_envs = len(self.envs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        """
        Step the environments with the given actions
        """
        return self.env.step(action)

    def reset(self, **kwargs):
        """
        Reset all environments
        """
        return self.env.reset(**kwargs)

    def close(self):
        """
        Close all environments
        """
        for env in self.envs:
            env.close()
