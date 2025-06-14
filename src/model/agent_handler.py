import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


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
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)

        # Observation space: combined car state and embedding
        # Car state is 4-dimensional (x, y, vx, vy) and embedding is 4-dimensional
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(experiment.STATE_INPUT_SIZE,),  # Default is 8 (4 car state + 4 embedding)
            dtype=np.float32
        )

        # Track episode information
        self.episode_step = 0
        self.total_episode_reward = 0

        # Environment state
        self.current_state = None
        self.current_embedding = None

        # Create the underlying Highway environment
        self.highway_env = gym.make('RELintersection-v0', render_mode="rgb_array", config=self.config)

    def _get_unwrapped_env(self):
        env = self.highway_env
        while hasattr(env, 'env') and not hasattr(env, 'controlled_vehicles'):
            env = env.env
        return env


    @staticmethod
    def get_action(model, car1_observation, car2_observation, step_counter, exploration_threshold):
        if random.random() < max(0.05, exploration_threshold * np.exp(-0.01 * step_counter)):
            # Bias towards movement during exploration
            car1_action = [1 if random.random() < 0.7 else 0]  # 70% chance to move
            car2_action = [1 if random.random() < 0.7 else 0]
        else:
            car1_action, _ = model.predict(car1_observation, deterministic=True)
            car2_action, _ = model.predict(car2_observation, deterministic=True)
        #print("Car 1 obs", car1_observation)
        #print("Car 2 obs", car2_observation)
        return car1_action, car2_action
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
                        print(f"Master: Sending zeros for arrived vehicle {i}")

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

        # Check if car1 has arrived
        if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                hasattr(env.controlled_vehicles[0], 'is_arrived') and env.controlled_vehicles[0].is_arrived):
            car1_state = np.array([0.0, 0.0, 0.0, 0.0])
            print('The',env.controlled_vehicles[0],'Arrived and sending : ',car1_state)
        else:
            car1_state = current_state[:4] if len(current_state.shape) == 1 else current_state[0]

        # Check if car2 has arrived
        if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 1 and
                hasattr(env.controlled_vehicles[1], 'is_arrived') and env.controlled_vehicles[1].is_arrived):
            car2_state = np.array([0.0, 0.0, 0.0, 0.0])
            print('The', env.controlled_vehicles[0], 'Arrived and sending : ', car1_state)
        else:
            car2_state = current_state[4:8] if len(current_state.shape) == 1 else current_state[1]

        agent_observation_car1 = np.concatenate((car1_state, self.current_embedding))
        agent_observation_car2 = np.concatenate((car2_state, self.current_embedding))

        return agent_observation_car1, agent_observation_car2, info

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

        # Car1 observation
        if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                hasattr(env.controlled_vehicles[0], 'is_arrived') and env.controlled_vehicles[0].is_arrived):
            car1_state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            car1_state = next_state[:4] if len(next_state.shape) == 1 else next_state[0]

        # Car2 observation
        if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 1 and
                hasattr(env.controlled_vehicles[1], 'is_arrived') and env.controlled_vehicles[1].is_arrived):
            car2_state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            car2_state = next_state[4:8] if len(next_state.shape) == 1 else next_state[1]

        agent_observation_car1 = np.concatenate((car1_state, self.current_embedding))
        agent_observation_car2 = np.concatenate((car2_state, self.current_embedding))

        self.current_state = next_state
        self.total_episode_reward += reward

        # Return same format as reset() - both observations
        return agent_observation_car1, agent_observation_car2, reward, done, truncated, info
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
