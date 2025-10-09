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
        self.master_output_mode = getattr(experiment, "MASTER_OUTPUT_MODE", "embedding")
        self.use_master_actions = getattr(experiment, "USE_MASTER_ACTIONS", False)
        self.master_action_threshold = getattr(experiment, "MASTER_ACTION_THRESHOLD", 0.0)
        self.master_action_dim = getattr(experiment, "MASTER_ACTION_DIM", 2)

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
        self.current_master_actions = np.zeros(self.master_action_dim, dtype=np.float32)
        self.last_master_output = None
        self.last_master_value = None
        self.last_master_log_prob = None
        self.last_applied_action_tuple = tuple()

        # Create the underlying Highway environment
        self.highway_env = gym.make('RELintersection-v0', render_mode="rgb_array", config=self.config)

    def _get_unwrapped_env(self):
        env = self.highway_env
        while hasattr(env, 'env') and not hasattr(env, 'controlled_vehicles'):
            env = env.env
        return env


    @staticmethod
    def get_action(model, car1_observation, car2_observation, step_counter, exploration_threshold):
        if step_counter < exploration_threshold:
            car1_action = random.choice([0,1])
            car2_action = random.choice([0,1])
            #print('random action',car1_action,car2_action)
        # if random.random() < max(0.05, exploration_threshold * np.exp(-0.01 * step_counter)):
        #     # Bias towards movement during exploration
        #     car1_action = [1 if random.random() < 0.7 else 0]  # 70% chance to move
        #     car2_action = [1 if random.random() < 0.7 else 0]
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
            self._update_master_outputs(current_state)
            self.last_applied_action_tuple = tuple()
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

        if self.use_master_actions:
            agent_observation_car1 = np.concatenate((car1_state, self.current_master_actions))
            agent_observation_car2 = np.concatenate((car2_state, self.current_master_actions))
        else:
            agent_observation_car1 = np.concatenate((car1_state, self.current_embedding))
            agent_observation_car2 = np.concatenate((car2_state, self.current_embedding))

        return agent_observation_car1, agent_observation_car2, info

    def step(self, action_tuple):
        """Execute action and return observations for both agents."""
        self.episode_step += 1

        if self.use_master_actions:
            master_action_tuple = self._build_master_action_tuple()
            self.last_applied_action_tuple = master_action_tuple
            for idx, action in enumerate(master_action_tuple, start=1):
                print(f"[MasterAction] Step {self.episode_step}: Car {idx} -> {action}")
            next_state, reward, done, truncated, info = self.highway_env.step(master_action_tuple)
        else:
            next_state, reward, done, truncated, info = self.highway_env.step(action_tuple)
        next_state = self._prepare_state_for_master(next_state)

        if self.master_model is not None:
            self._update_master_outputs(next_state)
        else:
            print('master is none')
            if self.use_master_actions:
                self.current_master_actions = np.zeros(self.master_action_dim, dtype=np.float32)
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

        if self.use_master_actions:
            agent_observation_car1 = np.concatenate((car1_state, self.current_master_actions))
            agent_observation_car2 = np.concatenate((car2_state, self.current_master_actions))
        else:
            agent_observation_car1 = np.concatenate((car1_state, self.current_embedding))
            agent_observation_car2 = np.concatenate((car2_state, self.current_embedding))

        self.current_state = next_state
        self.total_episode_reward += reward

        # Return same format as reset() - both observations
        return agent_observation_car1, agent_observation_car2, reward, done, truncated, info

    def _update_master_outputs(self, state):
        flat_state = np.array(state, dtype=np.float32).reshape(1, -1)
        master_input = torch.tensor(flat_state, dtype=torch.float32)
        master_output, value, log_prob = self.master_model.get_proto_action(master_input)
        self.last_master_output = master_output
        self.last_master_value = value
        self.last_master_log_prob = log_prob
        if self.use_master_actions:
            self.current_master_actions = self.convert_master_output_to_actions(
                master_output,
                self.master_action_dim,
                self.master_action_threshold,
            ).astype(np.float32)
        else:
            self.current_embedding = master_output

    def _build_master_action_tuple(self):
        env = self._get_unwrapped_env()
        num_cars = 0
        if hasattr(env, 'controlled_vehicles'):
            num_cars = len(env.controlled_vehicles)
        if num_cars == 0:
            num_cars = len(self.current_master_actions)
        actions = []
        for idx in range(num_cars):
            if idx < len(self.current_master_actions):
                actions.append(int(self.current_master_actions[idx]))
            else:
                actions.append(0)
        return tuple(actions)

    def get_current_master_action_tuple(self):
        if not self.use_master_actions:
            raise RuntimeError("Master actions are not enabled in this experiment")
        return self._build_master_action_tuple()

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

    @staticmethod
    def convert_master_output_to_actions(master_output, action_dim, threshold=0.0):
        values = np.array(master_output).flatten()
        if values.size < action_dim:
            values = np.pad(values, (0, action_dim - values.size))
        elif values.size > action_dim:
            values = values[:action_dim]
        discrete = (values >= threshold).astype(int)
        return discrete


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
