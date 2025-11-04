"""
QMIX Environment Wrapper for Highway-env
Adapts the highway intersection environment for QMIX training
"""

import gymnasium as gym
import numpy as np
import torch


class QMIXHighwayWrapper:
    """
    Wrapper for Highway-env intersection scenario compatible with QMIX.
    Provides partial observations for each agent and global state.
    """
    
    def __init__(self, env_config, render_mode=None):
        """
        Args:
            env_config: Environment configuration dict
            render_mode: 'human' or None
        """
        self.env = gym.make('RELintersection-v0', render_mode=render_mode, config=env_config)
        self.env_config = env_config
        
        # Get unwrapped environment
        self._unwrapped_env = self._get_unwrapped_env()
        
        # Determine number of agents
        self.n_agents = len(env_config.get('controlled_cars', {}))
        
        # Observation and state dimensions
        self.obs_dim = 4  # x, y, vx, vy for each agent's own state
        self.state_dim = self.n_agents * 4  # Global state includes all vehicles
        
        # Action space
        self.n_actions = len(env_config.get('action', {}).get('target_speeds', [5, 10]))
        
        self.current_step = 0
        self.max_steps = env_config.get('duration', 50) * 5  # duration in seconds * 5Hz
        
    def _get_unwrapped_env(self):
        """Get the unwrapped environment with controlled_vehicles attribute"""
        env = self.env
        while hasattr(env, 'env') and not hasattr(env, 'controlled_vehicles'):
            env = env.env
        return env
        
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            observations: [n_agents, obs_dim] numpy array
            state: [state_dim] numpy array
            info: dict
        """
        obs, info = self.env.reset()
        self.current_step = 0
        
        # Clean arrival flags
        if hasattr(self._unwrapped_env, 'controlled_vehicles'):
            for vehicle in self._unwrapped_env.controlled_vehicles:
                if hasattr(vehicle, 'has_been_parked'):
                    delattr(vehicle, 'has_been_parked')
                if hasattr(vehicle, 'is_arrived'):
                    delattr(vehicle, 'is_arrived')
        
        # Prepare observations and state
        observations, state = self._process_observation(obs)
        
        return observations, state, info
        
    def step(self, actions):
        """
        Execute actions for all agents.
        
        Args:
            actions: [n_agents] numpy array of discrete actions
        Returns:
            observations: [n_agents, obs_dim]
            state: [state_dim]
            reward: float (shared reward)
            done: bool
            truncated: bool
            info: dict
        """
        self.current_step += 1
        
        # Convert actions to tuple
        action_tuple = tuple(actions)
        
        # Step environment
        obs, reward, done, truncated, info = self.env.step(action_tuple)
        
        # Check for max steps
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Process observations
        observations, state = self._process_observation(obs)
        
        return observations, state, reward, done, truncated, info
        
    def _process_observation(self, obs):
        """
        Convert environment observation to QMIX format.
        
        Args:
            obs: Raw observation from environment
        Returns:
            observations: [n_agents, obs_dim] - partial observations
            state: [state_dim] - global state
        """
        # Handle different observation formats
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 1:
                # Flat observation - reshape to [n_agents, features_per_agent]
                obs = obs.reshape(self.n_agents, -1)
            state = obs.flatten()
        else:
            state = np.array(obs).flatten()
            obs = state.reshape(self.n_agents, -1)
            
        # Ensure correct dimensions
        if state.shape[0] < self.state_dim:
            padded_state = np.zeros(self.state_dim, dtype=np.float32)
            padded_state[:state.shape[0]] = state
            state = padded_state
        elif state.shape[0] > self.state_dim:
            state = state[:self.state_dim]
            
        # Create partial observations (each agent sees only its own state)
        observations = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            if i < obs.shape[0]:
                # Check if vehicle has arrived
                if (hasattr(self._unwrapped_env, 'controlled_vehicles') and
                    i < len(self._unwrapped_env.controlled_vehicles) and
                    hasattr(self._unwrapped_env.controlled_vehicles[i], 'is_arrived') and
                    self._unwrapped_env.controlled_vehicles[i].is_arrived):
                    observations[i] = np.zeros(self.obs_dim, dtype=np.float32)
                else:
                    # Take first 4 features (x, y, vx, vy)
                    agent_obs = obs[i].flatten()[:self.obs_dim]
                    if len(agent_obs) < self.obs_dim:
                        padded = np.zeros(self.obs_dim, dtype=np.float32)
                        padded[:len(agent_obs)] = agent_obs
                        observations[i] = padded
                    else:
                        observations[i] = agent_obs
                        
        return observations.astype(np.float32), state.astype(np.float32)
        
    def render(self):
        """Render the environment"""
        return self.env.render()
        
    def close(self):
        """Close the environment"""
        return self.env.close()
        

def create_qmix_env(env_config, render_mode=None):
    """
    Factory function to create QMIX-compatible environment.
    
    Args:
        env_config: Environment configuration dict
        render_mode: 'human' or None
    Returns:
        QMIXHighwayWrapper instance
    """
    return QMIXHighwayWrapper(env_config, render_mode)
