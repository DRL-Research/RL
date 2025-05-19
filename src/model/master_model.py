import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn

###############################################
# Master Network Components
###############################################
class CustomMasterNetwork(nn.Module):
    """
    Custom neural network for master model that outputs an embedding
    rather than direct actions.
    """

    def __init__(self, observation_dim, embedding_dim=4):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Network architecture
        self.shared_net = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        #outputs embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.Tanh()  # Normalized embedding in [-1, 1]
        )

        # outputs value
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        """
        Forward pass through the network
        """
        if len(obs.shape) > 1 and obs.shape[0] == 1 and obs.shape[1] % 4 == 0:
            pass
        elif len(obs.shape) == 2 and obs.shape[0] > 1 and obs.shape[1] == 4:
            obs = obs.reshape(1, -1)

        shared_features = self.shared_net(obs)
        embedding = self.embedding_head(shared_features)
        value = self.value_head(shared_features)

        return embedding, value


class CustomMasterPolicy(ActorCriticPolicy):
    """
    Custom policy for master model that outputs an embedding
    instead of a categorical distribution.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            embedding_dim=4,
            *args,
            **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        obs_dim = observation_space.shape[0]
        self.custom_network = CustomMasterNetwork(obs_dim, embedding_dim)

        # Set up optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass in the neural network
        Returns embedding, value, and log probability
        """
        #  tensor format
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        # Handle 2D input for multiple cars by reshaping
        if len(obs.shape) == 2 and obs.shape[0] > 1 and obs.shape[1] == 4:
            # Reshape from (num_cars, features_per_car) to (batch=1, num_cars * features_per_car)
            obs = obs.reshape(1, -1)
        embedding, value = self.custom_network(obs)
        # For PPO compatibility: we need to return a "log_prob"
        log_prob = -0.5 * torch.sum(torch.ones_like(embedding), dim=-1)

        return embedding, value, log_prob

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor
    ):
        """
        Evaluate actions according to the current policy,
        given the observations.
        """
        # Handle 2D input for multiple cars by reshaping
        if len(obs.shape) == 2 and obs.shape[1] == 4 * (obs.shape[0] // 4):
            # Reshape from (batch, cars * features) to (batch, cars * features)
            obs = obs.reshape(obs.shape[0], -1)
        embedding, values = self.custom_network(obs)
        log_prob = -0.5 * torch.sum(torch.ones_like(embedding), dim=-1)
        entropy = torch.sum(torch.ones_like(embedding) * 1.0, dim=-1)

        return values, log_prob, entropy

    def get_distribution(self, obs):
        """
        For compatibility with PPO - returns a dummy distribution
        """

        # Return a dummy distribution with appropriate methods
        class DummyDistribution:
            def __init__(self, embedding):
                self.embedding = embedding

            def log_prob(self, actions):
                return -0.5 * torch.sum(torch.ones_like(self.embedding), dim=-1)

            def entropy(self):
                return torch.sum(torch.ones_like(self.embedding) * 1.0, dim=-1)

        # Handle 2D input for multiple cars by reshaping
        if len(obs.shape) == 2 and obs.shape[0] > 1 and obs.shape[1] == 4:
            # Reshape from (num_cars, features_per_car) to (batch=1, num_cars * features_per_car)
            obs = obs.reshape(1, -1)

        embedding, _ = self.custom_network(obs)
        return DummyDistribution(embedding)

    def predict_values(self, obs):
        """
        Predict the values for the given observations
        """
        # Handle 2D input for multiple cars by reshaping
        if len(obs.shape) == 2 and obs.shape[0] > 1 and obs.shape[1] == 4:
            # Reshape from (num_cars, features_per_car) to (batch=1, num_cars * features_per_car)
            obs = obs.reshape(1, -1)

        _, values = self.custom_network(obs)
        return values


###############################################
# Master Model Class
###############################################
class MasterModel:
    """
    Master model for the Highway environment.
    Uses a custom policy to output embeddings instead of actions.

    This model can handle variable numbers of cars (up to 5) in the environment.
    """

    def __init__(self, embedding_size=4, experiment=None):
        self.embedding_size = embedding_size
        self.experiment = experiment
        self.is_frozen = False

        # Each car has 4 dimensions (x, y, vx, vy)
        max_cars = self.experiment.CARS_AMOUNT if experiment else 5
        self.observation_dim = max_cars * 4

        dummy_env = gym.Env()
        dummy_env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,)
        )
        # The action space is the embedding size
        dummy_env.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(embedding_size,), dtype=np.float32
        )

        # Initialize PPO with our custom policy
        if experiment and hasattr(experiment, 'PPO_NETWORK_ARCHITECTURE'):
            policy_kwargs = dict(net_arch=experiment.PPO_NETWORK_ARCHITECTURE)
        else:
            policy_kwargs = dict(net_arch=[64, 32, 16, 8])

        learning_rate = experiment.LEARNING_RATE if experiment else 3e-4
        n_steps = experiment.N_STEPS if experiment else 2048
        batch_size = experiment.BATCH_SIZE if experiment else 64

        self.model = PPO(
            policy=CustomMasterPolicy,
            env=dummy_env,
            verbose=1,
            policy_kwargs={"embedding_dim": embedding_size, **policy_kwargs},
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu"
        )

        # Create a rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=1
        )

    def unfreeze(self):
        """Enable training for master network"""
        self.is_frozen = False
        for param in self.model.policy.parameters():
            param.requires_grad = True
        self.model.policy.set_training_mode(True)

    def freeze(self):
        """Disable training for master network"""
        self.is_frozen = True
        for param in self.model.policy.parameters():
            param.requires_grad = False
        self.model.policy.set_training_mode(False)

    def get_proto_action(self, state_tensor):
        """
        Get embedding from master network.
        Used during inference or when master is frozen.

        Ensures the input is properly reshaped before passing to the model.
        """
        with torch.no_grad():
            # Handle input in both tensor and numpy array formats
            if isinstance(state_tensor, np.ndarray):
                # If it's a numpy array with shape (5, 4)
                if len(state_tensor.shape) == 2 and state_tensor.shape[0] > 1:
                    # Reshape to (1, 20)
                    reshaped_tensor = torch.tensor(state_tensor.reshape(1, -1), dtype=torch.float32)
                    #print(f"Reshaped numpy array from {state_tensor.shape} to {reshaped_tensor.shape}")
                elif len(state_tensor.shape) == 3:
                    # If it's a 3D array with shape like (1, 5, 4)
                    # Reshape to (batch, num_cars*features_per_car)
                    batch_size = state_tensor.shape[0]
                    flattened = state_tensor.reshape(batch_size, -1)
                    reshaped_tensor = torch.tensor(flattened, dtype=torch.float32)
                    #print(f"Reshaped 3D numpy array from {state_tensor.shape} to {reshaped_tensor.shape}")
                else:
                    # Already flat or single dimension
                    reshaped_tensor = torch.tensor(state_tensor, dtype=torch.float32)
                    if len(reshaped_tensor.shape) == 1:
                        reshaped_tensor = reshaped_tensor.unsqueeze(0)
            elif isinstance(state_tensor, torch.Tensor):
                # If it's a 3D tensor with shape (batch, num_cars, features_per_car)
                if len(state_tensor.shape) == 3:
                    # Reshape to (batch, num_cars*features_per_car)
                    batch_size = state_tensor.shape[0]
                    reshaped_tensor = state_tensor.reshape(batch_size, -1)
                # If it's a tensor with shape (5, 4)
                elif len(state_tensor.shape) == 2 and state_tensor.shape[0] > 1 and state_tensor.shape[1] == 4:
                    # Reshape to (1, 20)
                    reshaped_tensor = state_tensor.reshape(1, -1)
                    #print(f"Reshaped tensor from {state_tensor.shape} to {reshaped_tensor.shape}")
                elif len(state_tensor.shape) == 2 and state_tensor.shape[0] == 1:
                    # Already in correct shape (1, X)
                    reshaped_tensor = state_tensor
                elif len(state_tensor.shape) == 1:
                    # Add batch dimension if needed
                    reshaped_tensor = state_tensor.unsqueeze(0)
                else:
                    reshaped_tensor = state_tensor
            else:
                try:
                    reshaped_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)
                except:
                    raise TypeError(f"Cannot handle input of type {type(state_tensor)}")

            # Make sure we have the right dimension for the model
            if len(reshaped_tensor.shape) == 2 and reshaped_tensor.shape[1] != self.observation_dim:
                if reshaped_tensor.shape[1] < self.observation_dim:
                    # Pad with zeros if too small
                    padding = torch.zeros(reshaped_tensor.shape[0], self.observation_dim - reshaped_tensor.shape[1])
                    reshaped_tensor = torch.cat([reshaped_tensor, padding], dim=1)
                elif reshaped_tensor.shape[1] > self.observation_dim:
                    # Truncate if too large
                    reshaped_tensor = reshaped_tensor[:, :self.observation_dim]
                    print(f"Truncated tensor to shape {reshaped_tensor.shape}")
            embedding, _, _ = self.model.policy.forward(reshaped_tensor)
            return embedding.cpu().numpy()[0]
    def set_logger(self, logger):
        """Set logger for master network"""
        self.model.set_logger(logger)

    def save(self, path):
        """Save master network"""
        # Save the PPO model
        self.model.save(path)
        custom_params = {
            "embedding_size": self.embedding_size,
        }
        torch.save(custom_params, f"{path}_custom_params.pt")

    def load(self, path):
        """Load master network"""
        self.model = PPO.load(path)
        if os.path.exists(f"{path}_custom_params.pt"):
            custom_params = torch.load(f"{path}_custom_params.pt")
            self.embedding_size = custom_params["embedding_size"]