import os
import time
import torch
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv


###############################################
# Master Environment Definition
###############################################

class MasterEnv(gym.Env):
    """
    Environment for the 'Master Network':
      - Observation: Concatenation of car1 (4 dims) and car2 (4 dims) states → 8-dimensional.
      - Action: A continuous proto-action (embedding) of 4 dimensions.
      - Reward: Determined based on whether a collision occurs, the target is reached, or a default step reward.
      - Done: True if a collision, target reached, or max episode steps are exceeded.
    """
    def __init__(self, experiment, airsim_manager):
        super(MasterEnv, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager

        # Define observation space: 8-dimensional (car1 state + car2 state).
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # Define action space: continuous, 4-dimensional proto-action.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.state = None
        self.current_step = 0
        self.max_episode_steps = 20  # Limit episode length.
        self.done = False

    def reset(self):
        """
        Resets the environment: resets car positions and states.
        Returns the initial observation.
        """
        self.done = False
        self.current_step = 0

        # Reset the simulation's car positions.
        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()

        # Get initial states for both cars (each 4-dimensional).
        car1_state = self.airsim_manager.get_car1_state()
        car2_state = self.airsim_manager.get_car2_state()
        # Combine states into an 8-dimensional observation.
        self.state = np.concatenate([car1_state, car2_state]).astype(np.float32)
        return self.state

    def step(self, action):
        """
        Executes one environment step using the given action.
        Updates the state, computes the reward, and determines whether the episode is done.
        """
        self.current_step += 1
        # Retrieve the updated states from both cars.
        car1_state = self.airsim_manager.get_car1_state()
        car2_state = self.airsim_manager.get_car2_state()
        self.state = np.concatenate([car1_state, car2_state]).astype(np.float32)

        # Check terminal conditions.
        collision = self.airsim_manager.collision_occurred()
        reached_target = self.airsim_manager.has_reached_target(self.state[:2])  # Using car1's position.

        # Default reward (e.g., starvation reward).
        reward = self.experiment.STARVATION_REWARD
        if collision:
            reward = self.experiment.COLLISION_REWARD  # For collision (e.g., -20).
            self.done = True
        elif reached_target:
            reward = self.experiment.REACHED_TARGET_REWARD  # For reaching target (e.g., +10).
            self.done = True

        # Terminate if maximum steps reached.
        if self.current_step >= self.max_episode_steps:
            self.done = True

        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        # Rendering is not implemented.
        pass

    def close(self):
        # Close environment resources if necessary.
        pass

###############################################
# Master Model Wrapper
###############################################

class MasterModel:
    """
    This wrapper creates and manages the master network using PPO.
    It uses the MasterEnv to generate the 4-dimensional proto-action (embedding).
    """
    def __init__(self,
                 experiment,
                 airsim_manager,
                 embedding_size=4,
                 policy_kwargs=None,
                 learning_rate=0.0001,
                 n_steps=90,
                 batch_size=32,
                 total_timesteps=1):
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.embedding_size = embedding_size
        self.total_timesteps = total_timesteps
        self.is_frozen = False

        # Construct the MasterEnv and wrap it in a DummyVecEnv.
        self.master_env = MasterEnv(experiment=self.experiment, airsim_manager=self.airsim_manager)
        self.master_vec_env = DummyVecEnv([lambda: self.master_env])

        # Set a default network architecture if not provided.
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[64,32,16,8 ])

        # PPO Model Configuration:
        # This configuration solved the issue where the MasterEnv agent was not learning effectively.
        # Key parameter explanations:
        # - learning_rate=0.0001: Low learning rate ensures stable updates, essential for complex environments.
        # - n_steps=90: Controls the rollout buffer size, balancing learning signal and noise.
        # - batch_size=32: Mini-batch size for updating the network, tuned to match n_steps for 4 episodes.
        # - policy_kwargs=dict(net_arch=[64, 32, 16, 8]): Custom MLP architecture, deep enough to model the environment dynamics. 4 layers with 64, 32, 16, and 8 units.
        # - n_epochs=10: Number of PPO epochs per update; allows better learning from collected experience.
        # - vf_coef=0.5: Balances the value function loss with policy loss.
        # - ent_coef=0.01: Encourages exploration by regularizing entropy.
        # - gae_lambda=0.95: Controls the bias-variance tradeoff in advantage estimation.
        # - max_grad_norm=0.5: Gradient clipping to prevent exploding gradients and stabilize training.
        # - clip_range=0.2: Limits policy update steps to prevent drastic changes.
        # - clip_range_vf=1: Clips value function updates to avoid instability.

        self.model = PPO(
            policy="MlpPolicy",
            env=self.master_vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            verbose=1,n_epochs=10,vf_coef= 0.5,ent_coef=0.01,gae_lambda=0.95,max_grad_norm=0.5,clip_range=0.2,clip_range_vf=1
        )

    def train_master(self, steps_for_train_master=None):
        """
        Updates the master network using the data currently in its rollout buffer.
        Here, we call the internal train() method, which processes the full buffer.
        """
        if self.is_frozen:
            print("[MasterModel] WARNING: Model is frozen. Unfreeze before training.")
            return

        if steps_for_train_master is None:
            steps_for_train_master = self.total_timesteps
        print(f"[MasterModel] Training for {steps_for_train_master} timesteps...")
        # Call the PPO model's internal training update on the filled rollout buffer.
        self.model.train()
        print("[MasterModel] Training completed.")

    def get_proto_action(self, observation, deterministic=True):
        """
        Given an observation (state of car1 and car2 concatenated),
        returns a 4-dimensional proto-action (embedding).
        """
        # Ensure observation shape is (1, features)
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
