import time
import airsim
import gym
import numpy as np
from gym import spaces

from airsim_manager import AirsimManager
from experiment import Experiment
class Agent(gym.Env):
    """
    Agent is responsible for handling RL agent (reset, step, action, ...)
    """
    def __init__(self, experiment, airsim_manager):
        super(Agent, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)  # TODO: why observation is 2?
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,),
                                            dtype=np.float32)  # TODO: why observation is 2?
        self.state = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()
        if self.experiment.ROLE == 'Car1':
            self.state = self.airsim_manager.get_car1_state()
        else:
            self.state = self.airsim_manager.get_car2_state()
        return self.state

    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, False, {}

        throttle = Experiment.THROTTLE_FAST if action == 0 else Experiment.THROTTLE_SLOW
        if self.experiment.ROLE == 'Car1':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=Experiment.FIXED_THROTTLE),
                                                 self.experiment.CAR2_NAME)
        elif self.experiment.ROLE == 'Car2':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=Experiment.FIXED_THROTTLE),
                                                 self.experiment.CAR1_NAME)
        elif self.experiment.ROLE == 'Both':
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
            self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)

        time.sleep(self.experiment.TIME_BETWEEN_STEPS)

        if self.experiment.ROLE == 'Car1':
            self.state = self.airsim_manager.get_car1_state()
        else:
            self.state = self.airsim_manager.get_car2_state()

        collision = self.airsim_manager.collision_occurred()
        reached_target = self.airsim_manager.has_reached_target(self.state[:2])  # בודק אם רכב 1 הגיע ליעד

        reward = -0.1
        if collision:
            reward = -20
        elif reached_target:
            reward = 10
        done = collision or reached_target
        return self.state, reward, done, {}

    # def reset(self) -> np.ndarray:
    #     self.airsim_manager.reset_cars_to_initial_positions()
    #     self.airsim_manager.reset_for_new_episode()
    #     car1_state = self.airsim_manager.get_car1_state()
    #     self.state = np.array([car1_state[0], car1_state[1]], dtype=np.float32)
    #     return self.state
    #
    # def step(self, action):
    #     if self.airsim_manager.is_simulation_paused():
    #         return self.state, 0, False, {}
    #     throttle = Experiment.THROTTLE_FAST if action == 0 else Experiment.THROTTLE_SLOW
    #     if self.experiment.ROLE == 'Car1':
    #         self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
    #         self.airsim_manager.set_car_controls(airsim.CarControls(throttle=Experiment.FIXED_THROTTLE), self.experiment.CAR2_NAME)
    #     elif self.experiment.ROLE == 'Car2':
    #         self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
    #         self.airsim_manager.set_car_controls(airsim.CarControls(throttle=Experiment.FIXED_THROTTLE), self.experiment.CAR1_NAME)
    #     elif self.experiment.ROLE == 'Both':
    #         self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)
    #         self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR2_NAME)
    #
    #     time.sleep(self.experiment.TIME_BETWEEN_STEPS)
    #     car1_state = self.airsim_manager.get_car1_state()
    #     next_state = np.array([car1_state[0], car1_state[1]], dtype=np.float32)
    #     collision = self.airsim_manager.collision_occurred()
    #     reached_target = self.airsim_manager.has_reached_target(next_state)
    #     reward = -0.1
    #     if collision:
    #         reward = -20
    #     elif reached_target:
    #         reward = 10
    #     done = collision or reached_target
    #     return next_state, reward, done, {}

    #These 2 functions override base function in "GYM" enviroment. do not touch!
    def render(self, mode='human'):
        pass
    def close_env(self):
        self.airsim_manager.close()

    def get_action(self,model,current_state,total_steps,exploration_threshold):
        if total_steps > exploration_threshold:
            action = model.predict(current_state, deterministic=True)
        else:
            action = model.predict(current_state, deterministic=False)
        return action
