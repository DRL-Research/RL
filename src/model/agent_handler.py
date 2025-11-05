import random

import gymnasium as gym
import numpy as np
import torch
from gym import spaces as gym_spaces
import warnings
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

        # Define action and observation spaces
        # Highway environment uses discrete actions
        # Stable-Baselines3 (v1.6) expects classic gym space instances for
        # compatibility checks.  When we rely solely on gymnasium's spaces,
        # SB3 raises an assertion error because the objects are not instances
        # of ``gym.spaces.Space``.  To keep the environment gymnasium-based
        # while staying compatible with SB3 we instantiate equivalent spaces
        # from the legacy ``gym`` package.
        self.action_space = gym_spaces.Discrete(experiment.ACTION_SPACE_SIZE)

        # Observation space: combined car state and embedding
        # Car state is 4-dimensional (x, y, vx, vy) and embedding is 4-dimensional
        # Stable-Baselines3 requires finite bounds for Box spaces.  While the
        # underlying observation does not have hard physical limits, we can use
        # the maximum finite value representable in float32 to approximate
        # unbounded ranges while satisfying the API contract.
        obs_low = np.full(
            experiment.STATE_INPUT_SIZE,
            -np.finfo(np.float32).max,
            dtype=np.float32,
        )
        obs_high = np.full(
            experiment.STATE_INPUT_SIZE,
            np.finfo(np.float32).max,
            dtype=np.float32,
        )
        self.observation_space = gym_spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
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
    def get_action(model, car_observations, step_counter, exploration_threshold):
        actions = []  # will collect one action per car (same order as observations)

        if step_counter < exploration_threshold:
            # --- RANDOM PHASE: before the threshold, pick 0/1 at random for each car ---
            for _ in car_observations:
                # Choose a discrete action 0 or 1 uniformly at random
                a = random.choice([0, 1])  # new: replaces the old exp-decay exploration rule
                # Ensure the action has the same shape/type as model.predict output (e.g., [0] / [1])
                actions.append(np.array([a], dtype=np.int64))  # new: keep 1-D, length-1 action
        else:
            # --- POLICY PHASE: after the threshold, use the model deterministically ---
            for obs in car_observations:
                car_action, _ = model.predict(obs, deterministic=True)  # unchanged: use policy
                actions.append(car_action)

        return actions


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
        agent_observations = []  # TODO: duplicate code, move to utils

        for car_index in range(len(env.controlled_vehicles)):

            if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                    hasattr(env.controlled_vehicles[car_index], 'is_arrived') and env.controlled_vehicles[car_index].is_arrived):
                car_state = np.array([0.0, 0.0, 0.0, 0.0])
                print('The',env.controlled_vehicles[car_index],'Arrived and sending : ', car_state)
            else:
                car_state = current_state[car_index*4:car_index*4+4] if len(current_state.shape) == 1 else current_state[car_index]

            agent_observations.append(np.concatenate((car_state, self.current_embedding)))


        return agent_observations, info

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

        for car_index in range(len(env.controlled_vehicles)):

            if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                    hasattr(env.controlled_vehicles[car_index], 'is_arrived') and env.controlled_vehicles[
                        car_index].is_arrived):
                car_state = np.array([0.0, 0.0, 0.0, 0.0])
                print('The', env.controlled_vehicles[car_index], 'Arrived and sending : ', car_state)
            else:
                car_state = next_state[car_index * 4:car_index * 4 + 4] if len(next_state.shape) == 1 else next_state[car_index]

            agent_next_observations.append(np.concatenate((car_state, self.current_embedding)))

        self.current_state = next_state
        self.total_episode_reward += reward

        # Return same format as reset() - both observations
        return agent_next_observations, reward, done, truncated, info

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
