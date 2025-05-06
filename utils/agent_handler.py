import numpy as np
import random
import torch
import gymnasium as gym
from gymnasium import spaces


class Agent(gym.Env):
    """
    Agent environment wrapper for Highway.

    This wrapper interfaces with the Highway environment and integrates
    the master model's embedding into the agent's observation.
    """

    def __init__(self, experiment, master_model=None):
        super().__init__()
        self.experiment = experiment
        self.master_model = master_model

        # Load environment configuration
        self.config = experiment.CONFIG

        # Define action and observation spaces
        # Highway environment typically uses discrete actions
        self.action_space = spaces.Discrete(5)  # Example: 5 possible actions (NOOP, LEFT, RIGHT, FASTER, SLOWER)

        # Observation space: combined car state and embedding
        # Assuming car state is 4-dimensional and embedding is 4-dimensional
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),  # 4 (car state) + 4 (embedding)
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
    def get_action(model, observation, step_counter, exploration_threshold):
        """
        Get an action from the agent model, with optional exploration.
        """
        # Epsilon-greedy exploration
        if random.random() < max(0.05, exploration_threshold * np.exp(-0.01 * step_counter)):
            # Random action
            return [random.randint(0, model.action_space.n - 1)]
        else:
            # Model prediction
            action, _ = model.predict(observation, deterministic=True)
            return action

    def step(self, action):
        """
        Execute action in the environment.
        """
        self.episode_step += 1

        # Take action in Highway environment
        next_state, reward, done, truncated, info = self.highway_env.step(action)

        # Get embedding from master if available
        if self.master_model is not None:
            master_input = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            embedding = self.master_model.get_proto_action(master_input)
            self.current_embedding = embedding
        else:
            # Use zeros if master not available (for debugging/testing)
            self.current_embedding = np.zeros(4)

        # Combine highway state with embedding for agent observation
        # Adjust indices based on your specific implementation
        car_state = next_state[:4]  # Assuming first 4 values are the ego car state
        agent_observation = np.concatenate((car_state, self.current_embedding))

        # Update internal state
        self.current_state = next_state
        self.total_episode_reward += reward

        # Return the standard Gym interface
        return agent_observation, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment for a new episode.
        """
        self.episode_step = 0
        self.total_episode_reward = 0

        # Reset Highway environment
        state, info = self.highway_env.reset(**kwargs)
        self.current_state = state

        # Get initial embedding from master if available
        if self.master_model is not None:
            master_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            embedding = self.master_model.get_proto_action(master_input)
            self.current_embedding = embedding
        else:
            # Use zeros if master not available
            self.current_embedding = np.zeros(4)

        # Combine highway state with embedding for agent observation
        car_state = state[:4]  # Assuming first 4 values are the ego car state
        agent_observation = np.concatenate((car_state, self.current_embedding))

        return agent_observation, info

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
        return self.highway_env.seed(seed)

    def get_car_states(self):
        """
        Get the states of all cars in the environment.
        Adapt this to how Highway represents multiple vehicles.
        """
        # This is a placeholder - implement based on your Highway environment
        # For example, Highway might store vehicle states differently than AirSim

        # Get the full observation which might contain all vehicle states
        full_obs = self.current_state

        # Extract individual car states (adapt to your Highway implementation)
        # Example: if the state is [ego_x, ego_y, ego_vx, ego_vy, car1_x, car1_y, ...]
        car1_state = full_obs[:4]  # Ego vehicle
        car2_state = full_obs[4:8]  # First other vehicle
        car3_state = full_obs[8:12]  # Second other vehicle
        car4_state = full_obs[12:16]  # Third other vehicle
        car5_state = full_obs[16:20]  # Fourth other vehicle

        return car1_state, car2_state, car3_state, car4_state, car5_state


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