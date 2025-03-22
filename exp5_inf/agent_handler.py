import time
import airsim
import gym
import numpy as np
from gym import spaces

'''
This class is responsible for handling the agent's actions and observations.
The agent interacts with the environment (AirSim) by taking actions and observing the state.
The state is a combination of the agent's local state and the master network's state.
Because this is a multi-agent environment and agents are both using different roles (due to master-slave architecture),
the agent's role is defined in the experiment configuration.
'''

class Agent(gym.Env):
    def __init__(self, experiment, airsim_manager,master_model):
        super(Agent, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(experiment.STATE_INPUT_SIZE,), dtype=np.float32)
        self.state = None
        self.master_model = master_model
        self.reset()

    def reset(self) -> np.ndarray:
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            agent_state = self.airsim_manager.get_car1_state()
        else:
            agent_state = self.airsim_manager.get_car2_state()
        self.state = np.concatenate((agent_state, self.airsim_manager.get_proto_state(self.master_model)))
        #print(f"Initial state: {self.state}")
        return self.state

    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, False, {}
        # Split the action if a tuple is provided
        if isinstance(action, (tuple, list)):
            action1, action2 = action
        else:
            action1 = action2 = action
        # Map action to throttle (0 => FAST, 1 => SLOW)
        throttle1 = self.experiment.THROTTLE_FAST if action1 == 0 else self.experiment.THROTTLE_SLOW
        throttle2 = self.experiment.THROTTLE_FAST if action2 == 0 else self.experiment.THROTTLE_SLOW
        # Set controls for both vehicles
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle1), self.experiment.CAR1_NAME)
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle2), self.experiment.CAR2_NAME)
        # Get the global position (and speed) for each car
        car1_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR1_NAME)  # dict with x,y,Vx,Vy
        car2_state = self.airsim_manager.get_car_position_and_speed(self.experiment.CAR2_NAME)
        # Get the initial positions (stored when reset)
        init_pos1 = self.airsim_manager.get_car1_initial_position()  # [x, y]
        init_pos2 = self.airsim_manager.get_car2_initial_position()  # [x, y]

        # Depending on the role, decide which vehicle's progress to use:
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            # For Car1, we use the global x-axis:
            current_pos = car1_state["x"]
            # Determine desired global x based on initial position:
            if init_pos1[0] > 0:
                desired_global = self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
            else:
                desired_global = self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0]
            required_distance = abs(desired_global - init_pos1[0])
            traveled = abs(current_pos - init_pos1[0])
            # Build the observation: use Car1's local state plus the proto embedding
            local_state = self.airsim_manager.get_car1_state()
        elif self.experiment.ROLE == self.experiment.CAR2_NAME:
            # For Car2, assume the movement is along the y-axis:
            current_pos = car2_state["y"]
            if init_pos2[1] > 0:
                desired_global = self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
            else:
                desired_global = self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1]
            required_distance = abs(desired_global - init_pos2[1])
            traveled = abs(current_pos - init_pos2[1])
            local_state = self.airsim_manager.get_car2_state()
        else:
            # In case ROLE is BOTH – choose one vehicle's data (e.g., Car1)
            current_pos = car1_state["x"]
            if init_pos1[0] > 0:
                desired_global = self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
            else:
                desired_global = self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0]
            required_distance = abs(desired_global - init_pos1[0])
            traveled = abs(current_pos - init_pos1[0])
            local_state = self.airsim_manager.get_car1_state()

        # Get the master proto embedding (assumed to be 4-dim)
        proto_state = self.airsim_manager.get_proto_state(self.master_model)
        # Build the agent's observation: local state (4-dim) + proto state (4-dim) = 8-dim vector
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
    def get_action(model, current_state, total_steps, exploration_threshold,training_master):
        if (total_steps < exploration_threshold):
            action = model.predict(current_state, deterministic=False)
            #print(f"Exploiting action: {action[0]}")
        else:
            action = model.predict(current_state, deterministic=True)
            #print(f"Exploring action: {action}")
        return action

    # This function override base function in "GYM" environment. do not touch!
    def close_env(self):
        self.airsim_manager.close()

