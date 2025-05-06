import os
import time
import torch
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

import utils.experiment.experiment_config


###############################################
# Master Environment Definition
###############################################
class MasterEnv(gym.Env):
    """
    Environment for the Master Network:
      - Observation: Concatenation of Car1, Car2, Car3, Car4, and Car5 states (each 4-dimensional),
        resulting in a 20-dimensional vector.
      - Action: A continuous proto-action (embedding) of 4 dimensions.
      - Reward: Determined based on collision, target achievement, or a default step reward.
      - Done: True if a collision occurs, the target is reached, or max episode steps are exceeded.
    """
    def __init__(self, experiment, airsim_manager):
        super(MasterEnv, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager

        # Update observation space to 20 dimensions (4 dims from each of the 5 cars)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(experiment.EMBEDDING_SIZE,), dtype=np.float32)
        self.state = None
        self.current_step = 0
        self.done = False

    def reset(self):
        """
        Resets the environment: resets car positions and states.
        Returns the initial 20-dimensional observation.
        """
        self.done = False
        self.current_step = 0
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()

        # Get initial states from all 5 cars (each 4-dimensional)
        car1_state = self.airsim_manager.get_car1_state()
        car2_state = self.airsim_manager.get_car2_state()
        car3_state = self.airsim_manager.get_car3_state()
        car4_state = self.airsim_manager.get_car4_state()
        car5_state = self.airsim_manager.get_car5_state()
        self.state = np.concatenate([car1_state, car2_state, car3_state, car4_state, car5_state]).astype(np.float32)
        return self.state

    def step(self, action):
        """
        Executes one environment step using the given action.
        Updates the state, computes the reward, and determines whether the episode is done.
        """
        self.current_step += 1

        # Retrieve updated states from all five cars.
        car1_state = self.airsim_manager.get_car1_state()
        car2_state = self.airsim_manager.get_car2_state()
        car3_state = self.airsim_manager.get_car3_state()
        car4_state = self.airsim_manager.get_car4_state()
        car5_state = self.airsim_manager.get_car5_state()
        self.state = np.concatenate([car1_state, car2_state, car3_state, car4_state, car5_state]).astype(np.float32)

        # Check terminal conditions.
        collision = self.airsim_manager.collision_occurred()
        # Here we check target achievement using Car1's state (adjust as needed)
        reached_target = self.airsim_manager.has_reached_target(car1_state)

        # Default reward (e.g., starvation reward)
        reward = self.experiment.STARVATION_REWARD
        if collision:
            reward = self.experiment.COLLISION_REWARD
            self.done = True
        elif reached_target:
            reward = self.experiment.REACHED_TARGET_REWARD
            self.done = True

        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


###############################################
# Master Model Wrapper
###############################################

class MasterModel:
    """
    This wrapper creates and manages the master network using PPO.
    It uses the MasterEnv (which now provides a 20-dimensional observation from 5 cars)
    to generate a 4-dimensional proto-action (embedding).
    """
    def __init__(self,
                 experiment,
                 embedding_size=utils.experiment.experiment_config.Experiment.EMBEDDING_SIZE,
                 policy_kwargs=None,
                 learning_rate=0.0001,
                 n_steps=utils.experiment.experiment_config.Experiment.N_STEPS,
                 batch_size=32,
                 total_timesteps=22):
        self.experiment = experiment
        self.embedding_size = embedding_size
        self.total_timesteps = total_timesteps
        self.is_frozen = False

        # Create the updated MasterEnv and wrap it in a DummyVecEnv.
        self.master_env = MasterEnv(experiment=self.experiment)
        self.master_vec_env = DummyVecEnv([lambda: self.master_env])

        # Update network architecture to handle 20-dimensional input
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[128, 64, 32, 16])

        self.model = PPO(
            policy="MlpPolicy",
            env=self.master_vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_epochs=10,
            vf_coef=0.5,
            ent_coef=0.01,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            clip_range=0.2,
            clip_range_vf=1
        )

    def get_proto_action(self, observation, deterministic=True):
        """
        Given an observation (from MasterEnv), returns a 4-dimensional proto-action (embedding).
        """
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action[0]

    def freeze(self):
        """
        Freezes the master network (prevents further updates).
        """
        self.is_frozen = True
        self.model.policy.set_training_mode(False)

    def unfreeze(self):
        """
        Unfreezes the master network (allows updates).
        """
        self.is_frozen = False
        self.model.policy.set_training_mode(True)

    def save(self, filepath):
        """
        Saves the master model to the given filepath.
        """
        self.model.save(filepath)
        print(f"[MasterModel] Saved model to {filepath}")

    def load(self, filepath):
        """
        Loads a master model from the given filepath.
        """
        self.model = PPO.load(filepath, env=self.master_vec_env)
        print(f"[MasterModel] Loaded model from {filepath}")

    def set_logger(self, logger):
        """
        Sets the logger for the PPO model.
        """
        self.model.set_logger(logger)