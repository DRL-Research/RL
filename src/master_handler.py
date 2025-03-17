import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class MasterEnv(gym.Env):
    """
    Environment for the 'Master Network':
      - observation: state of car1 (4 dimensions) + state of car2 (4 dimensions) = vector of length 8
      - action: continuous vector of length 4 (proto-action) - Enviroment Embedding
      - reward: computed based on whether a collision occurred, the target was reached, or simply a step.
      - done: episode ends on collision or success (target reached).
    """

    def __init__(self, experiment, airsim_manager):
        super(MasterEnv, self).__init__()
        self.experiment = experiment
        self.airsim_manager = airsim_manager

        # observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(8,), dtype=np.float32
        )

        # proto-action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32
        )

        self.state = None
        self.current_step = 0
        self.max_episode_steps = 20  # Episode length limit
        self.done = False

    def reset(self):
        """
        Reset to initial state
        """
        self.done = False
        self.current_step = 0

        self.airsim_manager.reset_cars_to_initial_positions()
        self.airsim_manager.reset_for_new_episode()

        car1_state = self.airsim_manager.get_car1_state()  # shape (4,)
        car2_state = self.airsim_manager.get_car2_state()  # shape (4,)

        # Merge states
        self.state = np.concatenate([car1_state, car2_state]).astype(np.float32)
        return self.state

    def step(self, action):
        """
        Return the self state of the environment after taking a step with the given action.
        """
        self.current_step += 1
        # Retrieve updated state for car1 and car2
        car1_state = self.airsim_manager.get_car1_state()
        car2_state = self.airsim_manager.get_car2_state()
        self.state = np.concatenate([car1_state, car2_state]).astype(np.float32)

        # Check terminal conditions
        collision = self.airsim_manager.collision_occurred()
        reached_target = self.airsim_manager.has_reached_target(self.state[:2])  # XY of car1

        # Compute reward
        reward = self.experiment.STARVATION_REWARD  # e.g., 0.1 by default
        if collision:
            reward = self.experiment.COLLISION_REWARD  # -20
            self.done = True
        elif reached_target:
            reward = self.experiment.REACHED_TARGET_REWARD  # +10
            self.done = True

        # If we have exceeded the maximum steps per episode, finish the episode
        if self.current_step >= self.max_episode_steps:
            self.done = True

        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        '''
        not implemented, only for AirSim and GYM environments to run
        '''
        pass

    def close(self):
        # Close the environment if needed.
        pass


class MasterModel:
    """
    Wrapper class for the 'Master Network' based on stable-baselines3.
    """

    def __init__(self,
                 experiment,
                 airsim_manager,
                 embedding_size=4,
                 policy_kwargs=None,
                 learning_rate=0.005,
                 n_steps=30,
                 batch_size=32,
                 total_timesteps=10000):
        """
        :param experiment: Experiment configuration object with rewards, etc.
        :param airsim_manager: Class handling Airsim.
        :param embedding_size: Size of the embedding (proto-action) produced by the model.
        :param policy_kwargs: Neural network architecture (net_arch).
        :param learning_rate: Learning rate for PPO model.
        :param n_steps: Number of steps to collect before updating the PPO rollout.
        :param batch_size: Mini batch size for PPO updates.
        :param total_timesteps: Number of timesteps to train the master.
        """
        self.experiment = experiment
        self.airsim_manager = airsim_manager
        self.embedding_size = embedding_size
        self.total_timesteps = total_timesteps
        self.is_frozen = False

        # Construct the 'MasterEnv' environment for generating a 4D embedding
        self.master_env = MasterEnv(experiment=self.experiment, airsim_manager=self.airsim_manager)
        self.master_vec_env = DummyVecEnv([lambda: self.master_env])

        # If no parameters are provided, set a simple hidden network architecture
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[32, 16])

        # Create the PPO model - continuous action of size 4
        self.model = PPO(
            policy="MlpPolicy",
            env=self.master_vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            verbose=1
        )

    def train_master(self, total_timestamps=None):
        """
        Runs PPO training on the MasterEnv environment.
        """
        if self.is_frozen:
            print("[MasterModel] WARNING: Model is frozen. Unfreeze before training.")
            return

        if total_timestamps is None:
            total_timestamps = self.total_timesteps

        print(f"[MasterModel] Training for {total_timestamps} timesteps...")
        self.airsim_manager.pause_simulation()
        self.model.learn(total_timesteps=total_timestamps)
        self.airsim_manager.resume_simulation()
        print(f"[MasterModel] Training completed.")

    def get_proto_action(self, observation, deterministic=True):
        """
        Uses the trained model to produce a 4-dimensional embedding (proto-action).
        :param observation: np.array of size 8 (state of car1 + car2).
        :return: np.array of size 4.
        """
        # Ensure the input shape is compatible (1, batch)
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]

        action, _ = self.model.predict(observation, deterministic=deterministic)
        # print(f"[MasterModel] Predicted action: {action}")
        # Returns a vector of size (4,)
        return action[0]

    def freeze(self):
        """
        Freeze the parameters so that they are not updated in further training.
        Uses an internal flag and avoids calling learn if the model is 'Frozen'.
        """
        self.is_frozen = True
        self.model.policy.set_training_mode(False)

    def unfreeze(self):
        """
        Unfreeze the parameters – allows updates.
        """
        self.is_frozen = False
        self.model.policy.set_training_mode(True)

    def save(self, filepath):
        self.model.save(filepath)
        print(f"[MasterModel] Saved model to {filepath}")

    def load(self, filepath):
        self.model = PPO.load(filepath, env=self.master_vec_env)
        print(f"[MasterModel] Loaded model from {filepath}")

    def set_logger(self, logger):
        """
        Set the logger for the internal PPO model.
        """
        self.model.set_logger(logger)
