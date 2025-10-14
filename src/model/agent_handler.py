import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")

class Driver(gym.Env):
    """
    Agent environment wrapper for Highway intersection.

    This wrapper interfaces with the Highway environment and injects the
    master model's acceleration/deceleration guidance into each agent
    observation.

    The agent controls only one car (car1) and receives the master control
    signal alongside its own kinematic state.
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

        # Observation space: combined car state and master control signal
        # Car state is 4-dimensional (x, y, vx, vy) and master control is scalar per car
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(experiment.STATE_INPUT_SIZE,),  # Default is 5 (4 car state + 1 master control)
            dtype=np.float32
        )

        # Track episode information
        self.episode_step = 0
        self.total_episode_reward = 0

        # Environment state
        self.current_state = None
        self.current_controls = None
        self.current_value = None
        self.current_log_prob = None
        self.last_applied_controls = None
        self.last_value = None
        self.last_log_prob = None
        self.last_discrete_actions = None

        # Create the underlying Highway environment
        self.highway_env = gym.make('RELintersection-v0', render_mode=experiment.RENDER_MODE, config=self.config)

    def _get_unwrapped_env(self):
        env = self.highway_env
        while hasattr(env, 'env') and not hasattr(env, 'controlled_vehicles'):
            env = env.env
        return env

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

    def _align_controls_to_cars(self, controls, env):
        controls = np.array(controls, dtype=np.float32).flatten()
        num_cars = len(env.controlled_vehicles) if hasattr(env, 'controlled_vehicles') else 0
        if num_cars == 0:
            return controls

        if controls.shape[0] < num_cars:
            padded = np.zeros(num_cars, dtype=np.float32)
            padded[:controls.shape[0]] = controls
            controls = padded
        elif controls.shape[0] > num_cars:
            controls = controls[:num_cars]

        for i, vehicle in enumerate(env.controlled_vehicles):
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                controls[i] = 0.0

        return controls

    def _convert_controls_to_actions(self, controls, env):
        discrete_actions = []
        for i, vehicle in enumerate(env.controlled_vehicles):
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                discrete_actions.append(0)
                continue
            control_value = controls[i] if i < len(controls) else 0.0
            discrete_actions.append(1 if control_value >= 0 else 0)
        return tuple(discrete_actions)

    def get_discrete_master_actions(self):
        if self.last_discrete_actions is not None:
            return self.last_discrete_actions
        env = self._get_unwrapped_env()
        if self.current_controls is None:
            return tuple()
        return self._convert_controls_to_actions(self.current_controls, env)

    def _update_master_controls(self, state):
        env = self._get_unwrapped_env()
        if self.master_model is not None:
            master_input = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            controls, value, log_prob = self.master_model.get_proto_action(master_input)
            controls = self._align_controls_to_cars(controls, env)
            self.current_value = value
            self.current_log_prob = log_prob
        else:
            controls = np.zeros(len(env.controlled_vehicles), dtype=np.float32)
            self.current_value = torch.zeros((1, 1))
            self.current_log_prob = torch.zeros((1, 1))

        self.current_controls = controls

    def _build_agent_observations(self, state):
        env = self._get_unwrapped_env()
        agent_observations = []
        for car_index in range(len(env.controlled_vehicles)):
            if (hasattr(env, 'controlled_vehicles') and len(env.controlled_vehicles) > 0 and
                    hasattr(env.controlled_vehicles[car_index], 'is_arrived') and env.controlled_vehicles[car_index].is_arrived):
                car_state = np.array([0.0, 0.0, 0.0, 0.0])
                control_signal = 0.0
            else:
                car_state = state[car_index * 4:car_index * 4 + 4] if len(state.shape) == 1 else state[car_index]
                control_signal = self.current_controls[car_index] if self.current_controls is not None and car_index < len(self.current_controls) else 0.0

            agent_observations.append(np.concatenate((car_state, [control_signal])))

        return agent_observations

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

        self._update_master_controls(current_state)
        self.last_applied_controls = np.zeros_like(self.current_controls)
        self.last_value = self.current_value
        self.last_log_prob = self.current_log_prob
        self.last_discrete_actions = self._convert_controls_to_actions(self.current_controls, env)

        agent_observations = self._build_agent_observations(current_state)

        return agent_observations, info

    def step(self, action_tuple=None):
        """Execute action and return observations for both agents."""
        self.episode_step += 1

        env = self._get_unwrapped_env()

        if self.current_controls is None:
            raise ValueError("Master controls not initialized. Call reset() before step().")

        actions_to_apply = self._convert_controls_to_actions(self.current_controls, env)
        self.last_applied_controls = np.array(self.current_controls, copy=True)
        self.last_value = self.current_value
        self.last_log_prob = self.current_log_prob
        self.last_discrete_actions = actions_to_apply

        next_state, reward, done, truncated, info = self.highway_env.step(actions_to_apply)
        next_state = self._prepare_state_for_master(next_state)

        self.current_state = next_state
        self.total_episode_reward += reward

        self._update_master_controls(next_state)

        agent_next_observations = self._build_agent_observations(next_state)

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
