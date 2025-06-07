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

    @staticmethod
    def get_action(model, car1_observation, car2_observation, step_counter, exploration_threshold):
        """
        Get an action from the agent model, with optional exploration.
        """
        # Epsilon-greedy exploration
        if random.random() < max(0.05, exploration_threshold * np.exp(-0.01 * step_counter)):
            # Random action
            return [random.randint(0, model.action_space.n - 1)], [random.randint(0, model.action_space.n - 1)]
        else:
            # Model prediction
            car1_action, _ = model.predict(car1_observation, deterministic=True)
            car2_action, _ = model.predict(car2_observation, deterministic=True)
            return car1_action, car2_action

    def reset(self, **kwargs):
        """
        Reset the environment for a new episode.
        """
        self.episode_step, self.total_episode_reward = 0, 0

        # Reset Highway environment
        current_state, info = self.highway_env.reset(**kwargs)
        self.current_state = current_state  # gets the current_state of both controlled and static vehicles

        # Get initial embedding from master if available
        if self.master_model is not None:
            if len(current_state.shape) == 2 and current_state.shape[0] == 5 and current_state.shape[1] == 4:  # TODO: hardcoded sizes?
                master_input = torch.tensor(current_state.reshape(1, -1), dtype=torch.float32)
            else:
                master_input = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

            embedding, _, _ = self.master_model.get_proto_action(master_input)
            self.current_embedding = embedding
        else:
            # Use zeros if master not available
            raise ValueError("master not available")
            # self.current_embedding = np.zeros(4)

        # Combine each driver current_state with embedding from master
        car1_state = current_state[:4] if len(current_state.shape) == 1 else current_state[0]  # First 4 values are the ego car current_state
        car2_state = current_state[4:8] if len(current_state.shape) == 1 else current_state[1]
        agent_observation_car1 = np.concatenate((car1_state, self.current_embedding))
        agent_observation_car2 = np.concatenate((car2_state, self.current_embedding))

        return agent_observation_car1, agent_observation_car2, info  # car return embedding also

    def step(self, action_tuple):
        """
        Execute action in the environment.
        """
        self.episode_step += 1

        # Take action in Highway environment
        # next_state, reward, done, truncated, info = self.highway_env.step(car1_action)
        next_state, reward, done, truncated, info = self.highway_env.step(action_tuple)

        # Get embedding from master if available
        if self.master_model is not None:
            if len(next_state.shape) == 2 and next_state.shape[0] == 5 and next_state.shape[1] == 4:
                master_input = torch.tensor(next_state.reshape(1, -1), dtype=torch.float32)
            else:
                master_input = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            embedding, _, _ = self.master_model.get_proto_action(master_input)
            self.current_embedding = embedding
        else:
            # Use zeros if master not available
            self.current_embedding = np.zeros(4)

        # Combine car1 state with embedding for agent observation
        car1_state = next_state[:4] if len(next_state.shape) == 1 else next_state[0]
        agent_observation = np.concatenate((car1_state, self.current_embedding))

        # Update internal state
        self.current_state = next_state
        self.total_episode_reward += reward

        # Return the standard Gym interface
        return agent_observation, reward, done, truncated, info

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
