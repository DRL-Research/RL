import time
import airsim
import gym
import numpy as np
from gym import spaces

'''
This class handles the agent’s interaction with AirSim.
It builds observations by concatenating a single vehicle’s local state (4-dim) with the master network’s proto embedding (4-dim),
yielding an 8-dim observation. In this single-agent configuration, the same action is applied to both vehicles.
'''

class Agent(gym.Env):
    def __init__(self, experiment, airsim_manager, master_model):
        super(Agent, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        # Define a discrete action space (e.g., 0 => FAST, 1 => SLOW)
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        # Observation is 8-dimensional: 4-dim local state + 4-dim proto embedding
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(experiment.STATE_INPUT_SIZE,),
                                            dtype=np.float32)
        self.state = None
        self.master_model = master_model
        self.reset()

    def reset(self) -> np.ndarray:
        # Reset both vehicles to their initial positions and clear any episode-specific state
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        # Choose the local state based on the experiment ROLE
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            local_state = self.airsim_manager.get_car1_state()
        else:
            local_state = self.airsim_manager.get_car2_state()
        # Get the master network’s proto embedding (4-dim)
        proto_state = self.airsim_manager.get_proto_state(self.master_model)
        # Concatenate to form an 8-dim observation
        self.state = np.concatenate((local_state, proto_state))
        print(f"Initial state: {self.state}")
        return self.state

    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, False, {}

        # Use the action for Car1 (the agent) and use a fixed throttle for Car2
        action_value = action  # action is a single integer

        # Map the action to throttle for Car1 (0 => FAST, 1 => SLOW)
        throttle1 = self.experiment.THROTTLE_FAST if action_value == 0 else self.experiment.THROTTLE_SLOW

        # For Car2, use a fixed throttle (e.g., THROTTLE_FAST or a preset value)
        throttle2 = self.experiment.FIXED_THROTTLE

        # Apply controls: agent's action only to Car1; fixed control to Car2
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle1), self.experiment.CAR1_NAME)
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle2), self.experiment.CAR2_NAME)

        # Get the global position and speed for both vehicles
        car1_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR1_NAME)
        car2_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR2_NAME)
        init_pos1 = self.airsim_manager.get_car1_initial_position()
        init_pos2 = self.airsim_manager.get_car2_initial_position()

        # Determine progress based on the ROLE of the agent
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            current_pos = car1_state["x"]
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos1[0] > 0
                              else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos1[0])
            traveled = abs(current_pos - init_pos1[0])
            local_state = self.airsim_manager.get_car1_state()
        elif self.experiment.ROLE == self.experiment.CAR2_NAME:
            current_pos = car2_state["y"]
            desired_global = (self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
                              if init_pos2[1] > 0
                              else self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos2[1])
            traveled = abs(current_pos - init_pos2[1])
            local_state = self.airsim_manager.get_car2_state()
        else:
            # Default to Car1 if ROLE is not specifically defined
            current_pos = car1_state["x"]
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos1[0] > 0
                              else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos1[0])
            traveled = abs(current_pos - init_pos1[0])
            local_state = self.airsim_manager.get_car1_state()

        # Get the master network’s proto embedding (4-dim)
        proto_state = self.airsim_manager.get_proto_state(self.master_model)
        # Build the new observation (8-dim)
        self.state = np.concatenate((local_state, proto_state))

        time.sleep(self.experiment.TIME_BETWEEN_STEPS)

        # Check for collision and whether the vehicle has reached the target
        collision = self.airsim_manager.collision_occurred()
        reached_target = traveled >= required_distance

        # Compute reward based on collision and target achievement
        reward = self.experiment.STARVATION_REWARD
        if collision:
            reward = self.experiment.COLLISION_REWARD
        elif reached_target:
            reward = self.experiment.REACHED_TARGET_REWARD

        done = collision or reached_target
        return self.state, reward, done, {}

    @staticmethod
    def get_action(model, current_state, total_steps, exploration_threshold, training_master):
        # Predict an action using the model's policy
        if total_steps < exploration_threshold:
            action = model.predict(current_state, deterministic=False)
            print(f"Exploring action: {action[0]}")
        else:
            action = model.predict(current_state, deterministic=True)
            print(f"Exploiting action: {action[0]}")
        return action

    def close_env(self):
        self.airsim_manager.close()