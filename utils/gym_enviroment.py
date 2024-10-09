import time

import airsim
import gym
import numpy as np
from gym import spaces


class AirSimGymEnv(gym.Env):

    def __init__(self, experiment, airsim_manager):
        super(AirSimGymEnv, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)  # TODO: why observation is 2?
        self.state = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        car1_state = self.airsim_manager.get_car1_state()
        self.state = np.array([car1_state[0], car1_state[1]], dtype=np.float32)
        return self.state

    # TODO: step vs. step_rnd: what is the actual version?
    # TODO: each time you define throttle - use experiment.THROTTLE_FAST or experiment.THROTTLE_SlOW
    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, False, {}

        throttle = 1 if action == 0 else 0.5
        if self.experiment.ROLE == 'Car1':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=0.75), self.experiment.CAR2_NAME)
        elif self.experiment.ROLE == 'Car2':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=0.75), self.experiment.CAR1_NAME)
        elif self.experiment.ROLE == 'Both':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)

        time.sleep(self.experiment.TIME_BETWEEN_STEPS)
        car1_state = self.airsim_manager.get_car1_state()
        next_state = np.array([car1_state[0], car1_state[1]], dtype=np.float32)
        collision = self.airsim_manager.collision_occurred()
        reached_target = self.airsim_manager.has_reached_target(next_state)
        reward = -0.1
        if collision:
            reward = -20
        elif reached_target:
            reward = 10
        done = collision or reached_target
        if done:
            self.airsim_manager.pause_simulation()
        return next_state, reward, done, {}


    def step_rnd(self, action):
        if self.airsim_manager.is_simulation_paused():
            print("Simulation is paused, no step.")
            return self.state, 0, False, {}
        throttle = 0.75 if action == 0 else 0.5
        if self.experiment.ROLE == 'Car1':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.experiment.FIXED_THROTTLE), self.experiment.CAR2_NAME)
        elif self.experiment.ROLE == 'Car2':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.experiment.FIXED_THROTTLE), self.experiment.CAR1_NAME)
        elif self.experiment.ROLE == 'Both':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
        time.sleep(self.experiment.TIME_BETWEEN_STEPS)
        car1_state = self.airsim_manager.get_car1_state()
        next_state = np.array([car1_state[0], car1_state[1]], dtype=np.float32)
        collision = self.airsim_manager.collision_occurred()
        reached_target = self.airsim_manager.has_reached_target(next_state)
        reward = -0.1
        if collision:
            reward = -20
        elif reached_target:
            reward = 10
        done = collision or reached_target
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close_env(self):
        self.airsim_manager.close()

    def pause_simulation(self):
        self.airsim_manager.pause_simulation()

    def resume_simulation(self):
        self.airsim_manager.resume_simulation()