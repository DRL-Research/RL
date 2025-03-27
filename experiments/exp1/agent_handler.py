import time
import airsim
import gym
import numpy as np
from gym import spaces
import random


class Agent(gym.Env):
    """
    Agent is responsible for handling RL agent (reset, step, action, ...)
    """

    def __init__(self, experiment, airsim_manager):
        super(Agent, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(experiment.STATE_INPUT_SIZE,), dtype=np.float32)
        self.state = None
        self.np_random = None
        self.reset()


    def seed(self, seed=None):
        """
        Set the seed for reproducibility.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Seed set to: {seed}")
        return [seed]


    def reset(self) -> np.ndarray:
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            self.state = self.airsim_manager.get_car1_state()
        else:
            self.state = self.airsim_manager.get_car2_state()
        return self.state

    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, False, {}

        throttle = self.experiment.THROTTLE_FAST if action == 0 else self.experiment.THROTTLE_SLOW
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.experiment.FIXED_THROTTLE),
                                                 self.experiment.CAR2_NAME)

        elif self.experiment.ROLE == self.experiment.CAR2_NAME:
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.experiment.FIXED_THROTTLE),
                                                 self.experiment.CAR1_NAME)

        # elif self.experiment.ROLE == 'Both':
        #     self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
        #     self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)

        time.sleep(self.experiment.TIME_BETWEEN_STEPS)

        match self.experiment.ROLE:
            case self.experiment.CAR1_NAME:
                self.state = self.airsim_manager.get_car1_state()
            case self.experiment.CAR2_NAME:
                self.state = self.airsim_manager.get_car2_state()

        collision = self.airsim_manager.collision_occurred()
        reached_target = self.airsim_manager.has_reached_target(self.state[:2])

        reward = self.experiment.STARVATION_REWARD
        if collision:
            reward = self.experiment.COLLISION_REWARD
        elif reached_target:
            reward = self.experiment.REACHED_TARGET_REWARD
        done = collision or reached_target
        return self.state, reward, done, {}


    @staticmethod
    def get_action(model, current_state, total_steps, exploration_threshold):
        # Predict an action using the model's policy
        if total_steps < exploration_threshold:
            action = model.predict(current_state, deterministic=False)
            print(f"Exploring action: {action[0]}")
        else:
            action = model.predict(current_state, deterministic=True)
            print(f"Exploiting action: {action[0]}")
        return action


    # TODO - why was this function removed in exp5?
    # This function override base function in "GYM" environment. do not touch!
    def render(self, mode='human'):
        pass

    # This function override base function in "GYM" environment. do not touch!
    def close_env(self):
        self.airsim_manager.close()

