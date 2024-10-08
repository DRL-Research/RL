import gym
from gym import spaces
import numpy as np
import airsim
import time


class AirSimGymEnv(gym.Env):
    def __init__(self, config, airsim_manager):
        super(AirSimGymEnv, self).__init__()
        self.config = config
        self.airsim_manager = airsim_manager

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 2 actions: Throttle 0.75 or 0.5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.state = None
        self.reset()

    def reset(self):
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        car1_state = self.airsim_manager.get_car1_state()
        self.state = np.array([car1_state[0], car1_state[1]], dtype=np.float32)
        return self.state

    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            #print("Simulation is paused, no step.")
            return self.state, 0, False, {}
        throttle = 1 if action == 0 else 0.5
        if self.config.ROLE == 'Car1':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=0.75), self.config.CAR2_NAME)
        elif self.config.ROLE == 'Car2':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=0.75), self.config.CAR1_NAME)
        elif self.config.ROLE == 'Both':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR2_NAME)
        time.sleep(self.config.TIME_BETWEEN_STEPS)
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
        if self.config.ROLE == 'Car1':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.config.FIXED_THROTTLE), self.config.CAR2_NAME)
        elif self.config.ROLE == 'Car2':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=self.config.FIXED_THROTTLE), self.config.CAR1_NAME)
        elif self.config.ROLE == 'Both':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.config.CAR2_NAME)
        time.sleep(self.config.TIME_BETWEEN_STEPS)
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