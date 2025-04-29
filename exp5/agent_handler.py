import time
import airsim
import gym
import numpy as np
from gym import spaces


class Agent(gym.Env):
    """
    The Agent environment builds an 8-dimensional observation by concatenating:
      - 4 dimensions from the local state of the agent vehicle.
      - 4 dimensions from the proto embedding provided by the Master network.

    The agent's discrete action is applied to Car1 while all other cars (Car2, Car3, Car4, Car5)
    receive appropriate throttle settings based on their role.
    """

    def __init__(self, experiment, airsim_manager, master_model):
        super(Agent, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.master_model = master_model
        # Define the discrete action space (e.g., 0 = FAST, 1 = SLOW)
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        # Define the observation space: continuous vector of length STATE_INPUT_SIZE.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(experiment.STATE_INPUT_SIZE,),
                                            dtype=np.float32)
        self.state = None

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets the agent's internal state for a new episode.
        If the simulation is paused (e.g. during weight updates), do not reset the simulation.
        Instead, return the current state.
        """
        if self.airsim_manager.is_simulation_paused():
            # If already paused, return the current state (or a default state)
            return self.state if self.state is not None else np.zeros(self.observation_space.shape)

        # Otherwise, perform a full reset:
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            car_state = self.airsim_manager.get_car1_state()
        else:
            car_state = self.airsim_manager.get_car2_state()
        proto_state = self.airsim_manager.get_proto_state(self.master_model)
        self.state = np.concatenate((car_state, proto_state))
        return self.state

    def step(self, action):
        """
        Executes one step in the environment and returns (state, reward, done, info).
        """
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, True, {}

        action_value = action
        throttle1 = self.experiment.THROTTLE_FAST if action_value == 0 else self.experiment.THROTTLE_SLOW
        throttle_non_agent = self.experiment.FIXED_THROTTLE

        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle1), self.experiment.CAR1_NAME)
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent+0.1), self.experiment.CAR2_NAME)
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent), self.experiment.CAR3_NAME)
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent), self.experiment.CAR4_NAME)
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.experiment.THROTTLE_FAST), self.experiment.CAR5_NAME)

        car1_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR1_NAME)
        car2_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR2_NAME)
        init_pos1 = self.airsim_manager.get_car1_initial_position()
        init_pos2 = self.airsim_manager.get_car2_initial_position()

        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            current_pos = car1_state["x"]
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos1[0] > 0 else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos1[0])
            traveled = abs(current_pos - init_pos1[0])
            local_state = self.airsim_manager.get_car1_state()
        elif self.experiment.ROLE == self.experiment.CAR2_NAME:
            current_pos = car2_state["y"]
            desired_global = (self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
                              if init_pos2[1] > 0 else self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos2[1])
            traveled = abs(current_pos - init_pos2[1])
            local_state = self.airsim_manager.get_car2_state()
        else:
            current_pos = car1_state["x"]
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos1[0] > 0 else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos1[0])
            traveled = abs(current_pos - init_pos1[0])
            local_state = self.airsim_manager.get_car1_state()

        proto_state = self.airsim_manager.get_proto_state(self.master_model)
        self.state = np.concatenate((local_state, proto_state))
        time.sleep(self.experiment.TIME_BETWEEN_STEPS)

        collision = self.airsim_manager.collision_occurred()
        reached_target = traveled >= required_distance

        reward = self.experiment.STARVATION_REWARD
        if collision:
            reward = self.experiment.COLLISION_REWARD
        elif reached_target:
            reward = self.experiment.REACHED_TARGET_REWARD

        done = collision or reached_target
        return self.state, reward, done, {}

    @staticmethod
    def get_action(model, current_state, total_steps, exploration_threshold):
        """
        Returns an action from the model given the current state. Uses deterministic
        prediction if total_steps exceeds the exploration threshold.
        """
        deterministic = total_steps >= exploration_threshold
        action = model.predict(current_state, deterministic=deterministic)
        return action