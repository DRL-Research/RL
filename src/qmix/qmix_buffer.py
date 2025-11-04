"""
Replay Buffer for QMIX
Stores episode data for training
"""

import numpy as np
import torch
from collections import deque
import random


class EpisodeData:
    """Storage for a single episode"""
    
    def __init__(self):
        self.observations = []  # List of [n_agents, obs_dim]
        self.states = []  # List of [state_dim]
        self.actions = []  # List of [n_agents]
        self.rewards = []  # List of scalars
        self.dones = []  # List of booleans
        
    def add(self, obs, state, actions, reward, done):
        """Add a timestep to the episode"""
        self.observations.append(obs)
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def __len__(self):
        return len(self.observations)


class ReplayBuffer:
    """
    Replay buffer for QMIX that stores complete episodes.
    """
    
    def __init__(self, buffer_size=5000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add_episode(self, episode):
        """Add a complete episode to the buffer"""
        if len(episode) > 0:
            self.buffer.append(episode)
            
    def can_sample(self, batch_size):
        """Check if we have enough episodes to sample"""
        return len(self.buffer) >= batch_size
        
    def sample(self, batch_size):
        """
        Sample a batch of episodes from the buffer.
        
        Returns:
            batch: Dictionary containing batched episode data
        """
        episodes = random.sample(self.buffer, batch_size)
        
        # Get maximum episode length in this batch
        max_len = max(len(ep) for ep in episodes)
        
        # Prepare batch arrays
        n_agents = len(episodes[0].observations[0])
        obs_dim = episodes[0].observations[0].shape[1] if len(episodes[0].observations[0].shape) > 1 else episodes[0].observations[0].shape[0] // n_agents
        state_dim = episodes[0].states[0].shape[0] if hasattr(episodes[0].states[0], 'shape') else len(episodes[0].states[0])
        
        batch = {
            'observations': np.zeros((batch_size, max_len, n_agents, obs_dim), dtype=np.float32),
            'states': np.zeros((batch_size, max_len, state_dim), dtype=np.float32),
            'actions': np.zeros((batch_size, max_len, n_agents), dtype=np.int64),
            'rewards': np.zeros((batch_size, max_len), dtype=np.float32),
            'dones': np.zeros((batch_size, max_len), dtype=np.bool_),
            'filled': np.zeros((batch_size, max_len), dtype=np.bool_)  # Mask for valid timesteps
        }
        
        # Fill batch
        for i, ep in enumerate(episodes):
            ep_len = len(ep)
            
            for t in range(ep_len):
                obs = np.array(ep.observations[t])
                if len(obs.shape) == 1:
                    # Reshape flat observation to [n_agents, obs_dim]
                    obs = obs.reshape(n_agents, -1)
                    
                batch['observations'][i, t] = obs
                batch['states'][i, t] = ep.states[t]
                batch['actions'][i, t] = ep.actions[t]
                batch['rewards'][i, t] = ep.rewards[t]
                batch['dones'][i, t] = ep.dones[t]
                batch['filled'][i, t] = True
                
        return batch
        
    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class DemonstrationBuffer(ReplayBuffer):
    """
    Specialized buffer for demonstration data.
    Stores both expert data and self-generated interaction data.
    """
    
    def __init__(self, buffer_size, expert_ratio=0.1):
        super().__init__(buffer_size)
        self.expert_ratio = expert_ratio
        self.expert_buffer = deque(maxlen=int(buffer_size * expert_ratio))
        self.interaction_buffer = deque(maxlen=int(buffer_size * (1 - expert_ratio)))
        
    def add_expert_episode(self, episode):
        """Add expert demonstration episode"""
        if len(episode) > 0:
            self.expert_buffer.append(episode)
            
    def add_interaction_episode(self, episode):
        """Add self-generated interaction episode"""
        if len(episode) > 0:
            self.interaction_buffer.append(episode)
            
    def sample(self, batch_size):
        """
        Sample from both expert and interaction data.
        Maintains the expert_ratio in the sampled batch.
        """
        n_expert = int(batch_size * self.expert_ratio)
        n_interaction = batch_size - n_expert
        
        # Sample expert episodes
        expert_episodes = []
        if len(self.expert_buffer) > 0:
            n_expert_sample = min(n_expert, len(self.expert_buffer))
            expert_episodes = random.sample(list(self.expert_buffer), n_expert_sample)
            
        # Sample interaction episodes
        interaction_episodes = []
        if len(self.interaction_buffer) > 0:
            n_interaction_needed = batch_size - len(expert_episodes)
            n_interaction_sample = min(n_interaction_needed, len(self.interaction_buffer))
            interaction_episodes = random.sample(list(self.interaction_buffer), n_interaction_sample)
            
        # Combine and create batch
        all_episodes = expert_episodes + interaction_episodes
        
        if len(all_episodes) == 0:
            return None
            
        # Use parent class method to create batch
        self.buffer = deque(all_episodes, maxlen=self.buffer_size)
        return super().sample(len(all_episodes))
        
    def __len__(self):
        return len(self.expert_buffer) + len(self.interaction_buffer)
