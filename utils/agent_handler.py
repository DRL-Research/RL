import time
import torch
import airsim
import gym
import numpy as np
from gym import spaces

from utils.experiment.experiment_config import Experiment


class Agent(gym.Env):
    def __init__(self, experiment, airsim_manager, master_model):
        super(Agent, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.master_model = master_model  # Store the MasterModel
        self.action_space = spaces.Discrete(experiment.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(experiment.INPUT_SIZE,), dtype=np.float32)
        self.state = None
        self.reset()

    def reset(self):
        self.airsim_manager.reset_cars_to_initial_positions()
        combined_state = self.airsim_manager.get_combined_states()
        master_embedding = self.master_model.inference(combined_state.unsqueeze(0)).squeeze(0).numpy()
        self.state = np.concatenate((self.airsim_manager.get_car1_state(), master_embedding))
        return self.state


    def step(self, action):
        if self.airsim_manager.is_simulation_paused():
            return self.state, 0, False, {}

        # Convert action tensor to scalar
        if isinstance(action, torch.Tensor):
            action = action.item()


        throttle = Experiment.THROTTLE_FAST if action == 0 else Experiment.THROTTLE_SLOW
        self.airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle), self.experiment.CAR1_NAME)

        time.sleep(self.experiment.TIME_BETWEEN_STEPS)

        # Update state
        state_car1 = self.airsim_manager.get_car1_state()
        state_car2 = self.airsim_manager.get_car2_state()
        combined_state = np.concatenate((state_car1, state_car2))
        combined_state_tensor = torch.tensor(combined_state, dtype=torch.float32).unsqueeze(0)
        master_embedding = self.master_model.inference(combined_state_tensor).squeeze(0).numpy()
        self.state = np.concatenate((state_car1, master_embedding))

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
        if total_steps > exploration_threshold:
            action = model.predict(current_state, deterministic=True)
            #print(f"Exploiting action: {action}")
        else:
            action = model.predict(current_state, deterministic=False)
            #print(f"Exploring action: {action}")
        return action

    # This function override base function in "GYM" environment. do not touch!
    def render(self, mode='human'):
        pass

    # This function override base function in "GYM" environment. do not touch!
    def close_env(self):
        self.airsim_manager.close()


