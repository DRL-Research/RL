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
        if self.experiment.ROLE == "BOTH":
            # Define the discrete action space for both agent cars (Car1 and Car3)
            self.action_space = spaces.MultiDiscrete([experiment.ACTION_SPACE_SIZE, experiment.ACTION_SPACE_SIZE])
            extended_size= experiment.STATE_INPUT_SIZE * 2
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(extended_size,), dtype=np.float32)
        else:

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

        if self.experiment.ROLE == "BOTH":
            # Get states for both agent cars
            car1_state = self.airsim_manager.get_car1_state()
            car3_state = self.airsim_manager.get_car3_state()
            proto_state = self.airsim_manager.get_proto_state(self.master_model)
            # Combine states from both agents with proto state
            self.state = np.concatenate((car1_state, car3_state, proto_state))
        else:
            if self.experiment.ROLE == self.experiment.CAR1_NAME:
                car_state = self.airsim_manager.get_car1_state()
            elif self.experiment.ROLE == self.experiment.CAR2_NAME:
                car_state = self.airsim_manager.get_car2_state()
            elif self.experiment.ROLE == self.experiment.CAR3_NAME:
                car_state = self.airsim_manager.get_car3_state()
            else:
                car_state = self.airsim_manager.get_car1_state()
            proto_state = self.airsim_manager.get_proto_state(self.master_model)
            self.state = np.concatenate((car_state, proto_state))
            return self.state

    def step(self, action):
        """
        Executes one step in the environment and returns (state, reward, done, info).
        """
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, True, {}
        # Handle BOTH role where both Car1 and Car3 are agents
        if self.experiment.ROLE == "BOTH":
            # Assuming action is now a tuple or array with two values [car1_action, car3_action]
            car1_action = action[0]
            car3_action = action[1]

            # Set throttle for Car1 based on its action
            throttle1 = self.experiment.THROTTLE_FAST if car1_action == 0 else self.experiment.THROTTLE_SLOW
            # Set throttle for Car3 based on its action
            throttle3 = self.experiment.THROTTLE_FAST if car3_action == 0 else self.experiment.THROTTLE_SLOW

            # Set fixed throttle for the non-agent cars
            throttle_non_agent = self.experiment.FIXED_THROTTLE

            # Apply controls to all cars
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle1), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent),
                                                 self.experiment.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle3), self.experiment.CAR3_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent),
                                                 self.experiment.CAR4_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.experiment.THROTTLE_FAST),
                                                 self.experiment.CAR5_NAME)

            # Get states and check for collision and target achievement for both agent cars
            car1_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR1_NAME)
            car3_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR3_NAME)
            init_pos1 = self.airsim_manager.get_car1_initial_position()
            init_pos3 = self.airsim_manager.get_car3_initial_position()

            # Calculate progress for Car1
            current_pos1 = car1_state["x"]
            desired_global1 = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                               if init_pos1[0] > 0 else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance1 = abs(desired_global1 - init_pos1[0])
            traveled1 = abs(current_pos1 - init_pos1[0])

            # Calculate progress for Car3
            current_pos3 = car3_state["y"]
            desired_global3 = (self.experiment.CAR3_DESIRED_POSITION_OPTION_1[1]
                               if init_pos3[1] > 0 else self.experiment.CAR3_DESIRED_POSITION_OPTION_2[1])
            required_distance3 = abs(desired_global3 - init_pos3[1])
            traveled3 = abs(current_pos3 - init_pos3[1])

            # Get local states for both agents
            local_state1 = self.airsim_manager.get_car1_state()
            local_state3 = self.airsim_manager.get_car3_state()

            # Update state to include both agent states concatenated with proto embedding
            proto_state = self.airsim_manager.get_proto_state(self.master_model)
            self.state = np.concatenate((local_state1, local_state3, proto_state))

            time.sleep(self.experiment.TIME_BETWEEN_STEPS)

            # Check for collisions and target achievement
            collision1 = self.airsim_manager.collision_occurred(self.experiment.CAR1_NAME)
            collision3 = self.airsim_manager.collision_occurred(self.experiment.CAR3_NAME)
            reached_target1 = traveled1 >= required_distance1
            reached_target3 = traveled3 >= required_distance3

            # Determine combined reward
            reward = self.experiment.STARVATION_REWARD
            if collision1 or collision3:
                reward = self.experiment.COLLISION_REWARD
            elif reached_target1 and reached_target3:
                reward = self.experiment.REACHED_TARGET_REWARD
            elif reached_target1 or reached_target3:
                reward = self.experiment.REACHED_TARGET_REWARD / 2  # Partial reward

            # Episode is done if either agent collides or both reach target
            done = (collision1 or collision3) or (reached_target1 and reached_target3)

            return self.state, reward, done, {}
        else:

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

        # Check observation shape to determine if we're dealing with BOTH agents
        if current_state.shape[0] > 8:  # Expanded state for both agents
            # For BOTH mode, we need to predict two actions
            actions = model.predict(current_state, deterministic=deterministic)
            return actions
        else:
            # For single agent mode
            action = model.predict(current_state, deterministic=deterministic)
            return action