# """
# MAPPO for Unsignalized Intersection Navigation (With Logging)
# =============================================================
# Implementation of Multi-Agent PPO (IPPO with global observation).
#
# This file contains:
# 1. Scenario Management
# 2. PPO Networks (Actor + Critic)
# 3. Rollout Buffer (GAE)
# 4. MAPPO Agent
# 5. Environment Wrapper (Configured for global state visibility)
# 6. Training Loop (On-policy)
# 7. Main Execution
#
# Adapted by: Gemini
# Date: 2025
# """
# import csv
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical
# import random
# import gymnasium as gym
# import highway_env
# from datetime import datetime
# import os
# import matplotlib.pyplot as plt
# import json
# from copy import deepcopy
# import warnings
# import logging
#
# # Try to import Experiment config, otherwise define defaults
# try:
#     from experiment.experiment_config import Experiment
# except ImportError:
#     class Experiment:
#         COLLISION_REWARD = -5
#         REACHED_TARGET_REWARD = 1
#         STARVATION_REWARD = 0
#         EPISODE_MAX_TIME = 100
#         CARS_AMOUNT = 4
#         SPAWN_PROBABILITY = 0.6
#         SCREEN_WIDTH = 600
#         SCREEN_HEIGHT = 600
#
# warnings.filterwarnings('ignore')
#
# # ============================================================================
# # LOGGING SETUP
# # ============================================================================
# logger = logging.getLogger("MAPPO")
# logger.setLevel(logging.INFO)
#
# c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler('mappo_training.log', mode='w')
#
# log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(log_format)
# f_handler.setFormatter(log_format)
#
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)
#
# # Set random seeds
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device: {device}")
#
#
# # ============================================================================
# # SECTION 1: SCENARIO MANAGEMENT
# # ============================================================================
#
# def rotate_lane_id(lane_id, rotation):
#     if len(lane_id) < 2 or not lane_id[-1].isdigit():
#         return lane_id
#     prefix = lane_id[:-1]
#     direction = int(lane_id[-1])
#     new_direction = (direction + rotation) % 4
#     return prefix + str(new_direction)
#
# def rotate_scenario_clockwise(scenario, rotation):
#     rotated_scenario = []
#     for lane_key, destination, offset in scenario:
#         if isinstance(lane_key, tuple):
#             rotated_lane = tuple(
#                 rotate_lane_id(part, rotation) if isinstance(part, str) else part
#                 for part in lane_key
#             )
#         else:
#             rotated_lane = rotate_lane_id(lane_key, rotation)
#         rotated_dest = rotate_lane_id(destination, rotation)
#         rotated_scenario.append((rotated_lane, rotated_dest, offset))
#     return rotated_scenario
#
# class ScenarioManager:
#     def __init__(self):
#         self.all_scenarios = self._generate_all_scenarios()
#         logger.info(f"[ScenarioManager] Generated {len(self.all_scenarios)} scenarios")
#
#     def _get_base_scenarios(self):
#         # Using the same base scenarios as provided in the original code
#         base_scenarios = [
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", 0)], "static": [(('o0', 'ir0', 0), "o2", -50), (('o2', 'ir2', 0), "o0", -35), (('o0', 'ir0', 0), "o1", -70)]},
#              {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)], "static": [(('o2', 'ir2', 0), "o3", -25), (('o3', 'ir3', 0), "o0", -35), (('o0', 'ir0', 0), "o1", -15)]},
#              {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o2', 'ir2', 0), "o1", 0)], "static": [(('o1', 'ir1', 0), "o0", -20), (('o3', 'ir3', 0), "o2", -30), (('o0', 'ir0', 0), "o2", -40)]},
#              {"agents": [(('o1', 'ir1', 0), "o3", 0), (('o3', 'ir3', 0), "o1", 0)], "static": [(('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -25), (('o2', 'ir2', 0), "o1", -35)]},
#              {"agents": [(('o2', 'ir2', 0), "o0", 0), (('o1', 'ir1', 0), "o2", 0)], "static": [(('o0', 'ir0', 0), "o3", -45), (('o1', 'ir1', 0), "o2", -30), (('o2', 'ir2', 0), "o0", -20)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o0', 'ir0', 0), "o1", 20)], "static": [(('o0', 'ir0', 0), "o2", -50), (('o1', 'ir1', 0), "o3", -30), (('o2', 'ir2', 0), "o0", -25)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o2", 0)], "static": [(('o3', 'ir3', 0), "o2", -40), (('o2', 'ir2', 0), "o1", -30), (('o0', 'ir0', 0), "o3", -50)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o2', 'ir2', 0), "o0", 0)], "static": [(('o1', 'ir1', 0), "o3", -40), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -60)]},
#              {"agents": [(('o3', 'ir3', 0), "o1", 0), (('o0', 'ir0', 0), "o2", 0)], "static": [(('o1', 'ir1', 0), "o3", -20), (('o2', 'ir2', 0), "o0", -15), (('o3', 'ir3', 0), "o2", -45)]},
#              {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o3', 'ir3', 0), "o0", 0)], "static": [(('o1', 'ir1', 0), "o2", -25), (('o2', 'ir2', 0), "o3", -30), (('o0', 'ir0', 0), "o3", -60)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o0', 'ir0', 0), "o2", -30)], "static": [(('o2', 'ir2', 0), "o0", -60), (('o1', 'ir1', 0), "o3", -55), (('o3', 'ir3', 0), "o1", -65)]},
#              {"agents": [(('o1', 'ir1', 0), "o3", 0), (('o2', 'ir2', 0), "o0", 0)], "static": [(('o0', 'ir0', 0), "o2", -30), (('o3', 'ir3', 0), "o1", -25), (('o1', 'ir1', 0), "o0", -50)]},
#              {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o1', 'ir1', 0), "o0", 0)], "static": [(('o2', 'ir2', 0), "o1", -35), (('o3', 'ir3', 0), "o2", -40), (('o0', 'ir0', 0), "o1", -55)]},
#              {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o2', 'ir2', 0), "o3", 0)], "static": [(('o1', 'ir1', 0), "o2", -50), (('o3', 'ir3', 0), "o0", -55), (('o0', 'ir0', 0), "o2", -55)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", -40)], "static": [(('o2', 'ir2', 0), "o0", -50), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -20)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o2', 'ir2', 0), "o0", 0)], "static": [(('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -10)]},
#              {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o1', 'ir1', 0), "o2", 0)], "static": [(('o2', 'ir2', 0), "o1", -25), (('o3', 'ir3', 0), "o0", -30), (('o0', 'ir0', 0), "o2", -35)]},
#              {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)], "static": [(('o2', 'ir2', 0), "o3", -50), (('o3', 'ir3', 0), "o0", -55), (('o0', 'ir0', 0), "o3", -65)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", 50)], "static": [(('o2', 'ir2', 0), "o0", -60), (('o3', 'ir3', 0), "o1", -45), (('o0', 'ir0', 0), "o1", -75)]},
#              {"agents": [(('o2', 'ir2', 0), "o3", 0), (('o3', 'ir3', 0), "o2", 15)], "static": [(('o0', 'ir0', 0), "o1", -30), (('o1', 'ir1', 0), "o0", -35), (('o2', 'ir2', 0), "o0", -50)]},
#              {"agents": [(('o1', 'ir1', 0), "o2", 0), (('o0', 'ir0', 0), "o1", 0)], "static": [(('o2', 'ir2', 0), "o3", -25), (('o3', 'ir3', 0), "o0", -30), (('o1', 'ir1', 0), "o3", -40)]},
#              {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o3', 'ir3', 0), "o1", 0)], "static": [(('o1', 'ir1', 0), "o0", -35), (('o2', 'ir2', 0), "o3", -40), (('o0', 'ir0', 0), "o3", -50)]},
#              {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)], "static": [(('o2', 'ir2', 0), "o3", -30), (('o3', 'ir3', 0), "o0", -35), (('o0', 'ir0', 0), "o2", -60)]},
#              {"agents": [(('o2', 'ir2', 0), "o0", 0), (('o1', 'ir1', 0), "o3", 0)], "static": [(('o0', 'ir0', 0), "o2", -20), (('o3', 'ir3', 0), "o1", -25), (('o2', 'ir2', 0), "o1", -45)]},
#              {"agents": [(('o3', 'ir3', 0), "o2", 0), (('o0', 'ir0', 0), "o3", 0)], "static": [(('o1', 'ir1', 0), "o0", -30), (('o2', 'ir2', 0), "o1", -35), (('o3', 'ir3', 0), "o1", -55)]}
#         ]
#         return base_scenarios
#
#     def _generate_all_scenarios(self):
#         base_scenarios = self._get_base_scenarios()
#         all_scenarios = []
#         for base_scenario in base_scenarios:
#             all_scenarios.append(base_scenario)
#             for rotation in [1, 2, 3]:
#                 rotated_agents = rotate_scenario_clockwise(base_scenario["agents"], rotation)
#                 rotated_static = rotate_scenario_clockwise(base_scenario["static"], rotation)
#                 all_scenarios.append({"agents": rotated_agents, "static": rotated_static})
#         return all_scenarios
#
#     def get_scenario(self, idx):
#         return self.all_scenarios[idx]
#
#     def get_random_scenario(self):
#         return random.choice(self.all_scenarios)
#
#
# # ============================================================================
# # SECTION 2: PPO NETWORKS
# # ============================================================================
#
# class Actor(nn.Module):
#     def __init__(self, obs_dim, n_actions, hidden_dim=128):
#         super(Actor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, n_actions),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class Critic(nn.Module):
#     def __init__(self, obs_dim, hidden_dim=128):
#         super(Critic, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# # ============================================================================
# # SECTION 3: ROLLOUT BUFFER
# # ============================================================================
#
# class RolloutBuffer:
#     """Buffer for storing on-policy trajectories for PPO"""
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.state_values = []
#         self.is_terminals = []
#
#     def clear(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.state_values[:]
#         del self.is_terminals[:]
#
#     def add(self, state, action, logprob, reward, state_value, is_terminal):
#         self.states.append(state)
#         self.actions.append(action)
#         self.logprobs.append(logprob)
#         self.rewards.append(reward)
#         self.state_values.append(state_value)
#         self.is_terminals.append(is_terminal)
#
#
# # ============================================================================
# # SECTION 4: MAPPO AGENT
# # ============================================================================
#
# class PPOAgent:
#     """Single Agent PPO Implementation"""
#     def __init__(self, obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
#         self.device = device
#         self.gamma = gamma
#         self.eps_clip = eps_clip
#         self.K_epochs = K_epochs
#         self.buffer = RolloutBuffer()
#
#         self.policy = Actor(obs_dim, action_dim).to(device)
#         self.critic = Critic(obs_dim).to(device)
#
#         self.optimizer = torch.optim.Adam([
#             {'params': self.policy.parameters(), 'lr': lr_actor},
#             {'params': self.critic.parameters(), 'lr': lr_critic}
#         ])
#
#         self.policy_old = Actor(obs_dim, action_dim).to(device)
#         self.policy_old.load_state_dict(self.policy.state_dict())
#
#         self.MseLoss = nn.MSELoss()
#
#     def select_action(self, state):
#         with torch.no_grad():
#             state = torch.FloatTensor(state).to(self.device)
#             action_probs = self.policy_old(state)
#             dist = Categorical(action_probs)
#             action = dist.sample()
#             action_logprob = dist.log_prob(action)
#             state_val = self.critic(state)
#
#         # Store data in buffer
#         self.buffer.states.append(state)
#         self.buffer.actions.append(action)
#         self.buffer.logprobs.append(action_logprob)
#         self.buffer.state_values.append(state_val)
#
#         return action.item()
#
#     def update(self):
#         # Monte Carlo estimate of returns
#         rewards = []
#         discounted_reward = 0
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
#
#         # Normalizing the rewards
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         if len(rewards) > 1:
#             rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#
#         # Convert list to tensor
#         old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
#         old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
#         old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
#
#         # Optimize policy for K epochs
#         for _ in range(self.K_epochs):
#             # Evaluating old actions and values
#             action_probs = self.policy(old_states)
#             dist = Categorical(action_probs)
#             logprobs = dist.log_prob(old_actions)
#             dist_entropy = dist.entropy()
#             state_values = torch.squeeze(self.critic(old_states))
#
#             # Finding the ratio (pi_theta / pi_theta__old)
#             ratios = torch.exp(logprobs - old_logprobs)
#
#             # Finding Surrogate Loss
#             advantages = rewards - old_state_values.detach()
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#
#             # Final loss of clipped objective PPO
#             loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
#
#             # Take gradient step
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             self.optimizer.step()
#
#         # Copy new weights into old policy
#         self.policy_old.load_state_dict(self.policy.state_dict())
#         self.buffer.clear()
#
#         return loss.mean().item()
#
#     def save(self, path):
#         torch.save({
#             'policy': self.policy.state_dict(),
#             'critic': self.critic.state_dict(),
#             'policy_old': self.policy_old.state_dict(),
#             'optimizer': self.optimizer.state_dict()
#         }, path)
#
#     def load(self, path):
#         checkpoint = torch.load(path, map_location=self.device)
#         self.policy.load_state_dict(checkpoint['policy'])
#         self.critic.load_state_dict(checkpoint['critic'])
#         self.policy_old.load_state_dict(checkpoint['policy_old'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#
#
# class MAPPO:
#     """Wrapper class managing multiple PPO agents"""
#     def __init__(self, n_agents, obs_dim, n_actions, lr_actor=3e-4, lr_critic=1e-3,
#                  gamma=0.99, K_epochs=10, eps_clip=0.2, device='cpu'):
#         self.n_agents = n_agents
#         self.agents = [
#             PPOAgent(obs_dim, n_actions, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
#             for _ in range(n_agents)
#         ]
#
#     def select_action(self, obs_list):
#         actions = []
#         for i, agent in enumerate(self.agents):
#             # obs_list[i] is the observation for agent i
#             # Since obs is "global", it includes other vehicles
#             action = agent.select_action(obs_list[i])
#             actions.append(action)
#         return actions
#
#     def store_rewards(self, rewards, dones):
#         for i, agent in enumerate(self.agents):
#             agent.buffer.rewards.append(rewards[i])
#             agent.buffer.is_terminals.append(dones[i])
#
#     def update(self):
#         losses = []
#         for agent in self.agents:
#             loss = agent.update()
#             losses.append(loss)
#         return sum(losses) / len(losses)
#
#     def save(self, path_prefix):
#         for i, agent in enumerate(self.agents):
#             agent.save(f"{path_prefix}_agent_{i}.pt")
#
#     def load(self, path_prefix):
#         for i, agent in enumerate(self.agents):
#             agent.load(f"{path_prefix}_agent_{i}.pt")
#
#
# # ============================================================================
# # SECTION 5: ENVIRONMENT WRAPPER
# # ============================================================================
#
# class IntersectionWrapper:
#     """Wrapper for highway-env intersection with GLOBAL observation"""
#     def __init__(self, scenario_manager, n_agents=2, render_mode=None):
#         self.scenario_manager = scenario_manager
#         self.n_agents = n_agents
#
#         # KEY CHANGE: vehicles_count=10 and flatten=True
#         # This ensures that "For each state, each vehicle sees the other vehicles state"
#         # 10 is enough to cover all vehicles in these scenarios (usually 4-6 max)
#         self.vehicles_count = 10
#         self.features_count = 4
#
#         config = {
#             "observation": {
#                 "type": "Kinematics",
#                 "features": ["x", "y", "vx", "vy"],
#                 "features_range": {
#                     "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20],
#                 },
#                 "absolute": True,
#                 "flatten": True, # FLATTEN for MLP input
#                 "observe_intentions": False,
#                 "vehicles_count": self.vehicles_count, # See everyone
#             },
#             "action": {
#                 "type": "MultiAgentAction",
#                 "action_config": {
#                     "type": "DiscreteMetaAction",
#                     "lateral": False, "longitudinal": True, "target_speeds": [5, 10],
#                 },
#             },
#             "collision_reward": Experiment.COLLISION_REWARD,
#             "arrived_reward": Experiment.REACHED_TARGET_REWARD,
#             "normalize_reward": False, # Normalization helps PPO
#             "starvation_reward": Experiment.STARVATION_REWARD,
#             "offroad_terminal": False,
#             "duration": Experiment.EPISODE_MAX_TIME,
#             "initial_vehicle_count": Experiment.CARS_AMOUNT,
#             "spawn_probability": Experiment.SPAWN_PROBABILITY,
#             "screen_width": Experiment.SCREEN_WIDTH,
#             "screen_height": Experiment.SCREEN_HEIGHT,
#             "centering_position": [0.5, 0.6],
#             "scaling": 5.5,
#             "show_trajectories": True,
#             "render_agent": True
#         }
#
#         self.env = gym.make('intersection-v0', render_mode=render_mode, config=config)
#         logger.info(f"[Env] IntersectionWrapper Initialized (Mode: {render_mode}, Flattened)")
#
#     def reset(self, scenario_idx=None):
#         obs_raw, info = self.env.reset()
#         if scenario_idx is None:
#             scenario = self.scenario_manager.get_random_scenario()
#         else:
#             scenario = self.scenario_manager.get_scenario(scenario_idx)
#
#         # obs_raw should already be flattened and correct length due to config
#         # But we must ensure it matches n_agents and handles padding if needed
#         obs = []
#         expected_dim = self.vehicles_count * self.features_count
#
#         # In multi-agent highway-env, obs_raw is a tuple of arrays
#         for i in range(self.n_agents):
#             if i < len(obs_raw):
#                 agent_obs = obs_raw[i]
#                 # Pad if necessary (unlikely if vehicles_count is respected)
#                 if len(agent_obs) < expected_dim:
#                     agent_obs = np.pad(agent_obs, (0, expected_dim - len(agent_obs)))
#                 obs.append(agent_obs)
#             else:
#                 obs.append(np.zeros(expected_dim))
#
#         return obs
#
#     def step(self, actions):
#         if isinstance(actions, list):
#             actions = tuple(actions)
#         obs_raw, reward, done, truncated, info = self.env.step(actions)
#
#         obs = []
#         expected_dim = self.vehicles_count * self.features_count
#
#         for i in range(self.n_agents):
#             if i < len(obs_raw):
#                 agent_obs = obs_raw[i]
#                 if len(agent_obs) < expected_dim:
#                     agent_obs = np.pad(agent_obs, (0, expected_dim - len(agent_obs)))
#                 obs.append(agent_obs)
#             else:
#                 obs.append(np.zeros(expected_dim))
#
#         return obs, reward, done, truncated, info
#
#     def render(self):
#         return self.env.render()
#
#     def close(self):
#         self.env.close()
#
#
# # ============================================================================
# # SECTION 6: TRAINING LOOP
# # ============================================================================
#
# def train_mappo(
#         mappo_agent,
#         env,
#         n_episodes=5000,
#         update_timestep=2000, # Update PPO every 2000 timesteps
#         save_interval=500,
#         save_dir='results'
# ):
#     results = {'rewards': [], 'losses': [], 'success_rates': []}
#
#     logger.info("=" * 40)
#     logger.info("STARTING MAPPO TRAINING")
#     logger.info("=" * 40)
#
#     csv_file_path = os.path.join(save_dir, 'training_log.csv')
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['episode', 'result', 'reward', 'steps'])
#
#     time_step = 0
#     start_time = datetime.now()
#
#     for episode in range(n_episodes):
#         obs = env.reset()
#         episode_reward = 0
#         crashed = False
#         step_count = 0
#         done = False
#         truncated = False
#
#         while not (done or truncated):
#             time_step += 1
#             step_count += 1
#
#             # Select Action (Stochastic)
#             actions = mappo_agent.select_action(obs)
#
#             # Step Env
#             next_obs, reward, done, truncated, info = env.step(actions)
#
#             # Store Rewards
#             # Note: The way highway-env returns rewards is a tuple of rewards
#             # But sometimes if one agent is done, it might be tricky.
#             # We assume reward is a tuple of length n_agents.
#
#             # Create a list of done flags for each agent
#             # In intersection-v0, 'done' is usually a global bool,
#             # but we need per-agent for the buffer.
#             # We will use the global done for all agents here for simplicity
#             # or check info['agents_done'] if available.
#
#             dones_list = [done] * mappo_agent.n_agents
#
#             # Expand single reward to list if necessary (though env usually returns tuple)
#             if isinstance(reward, float):
#                  rewards_list = [reward] * mappo_agent.n_agents
#             else:
#                  rewards_list = list(reward)
#                  # Pad if fewer rewards than agents
#                  while len(rewards_list) < mappo_agent.n_agents:
#                      rewards_list.append(0.0)
#
#             mappo_agent.store_rewards(rewards_list, dones_list)
#
#             episode_reward += sum(rewards_list) / mappo_agent.n_agents
#             obs = next_obs
#
#             if info.get('crashed', False):
#                 crashed = True
#
#             # PPO Update
#             if time_step % update_timestep == 0:
#                 loss = mappo_agent.update()
#                 results['losses'].append(loss)
#                 logger.debug(f"[Update] PPO Update at step {time_step}. Loss: {loss:.4f}")
#
#         # ---------------------------------------------------------
#         # LOGGING
#         # ---------------------------------------------------------
#         if crashed: outcome = "Collision"
#         elif done and not crashed: outcome = "Success"
#         else: outcome = "Timeout"
#
#         with open(csv_file_path, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([episode + 1, outcome, round(episode_reward, 2), step_count])
#
#         results['rewards'].append(episode_reward)
#         results['success_rates'].append(0 if crashed else 1)
#
#         if (episode + 1) % 100 == 0:
#             avg_rew = np.mean(results['rewards'][-100:])
#             avg_suc = np.mean(results['success_rates'][-100:])
#             elapsed = datetime.now() - start_time
#             logger.info(f"Episode {episode + 1}/{n_episodes} | Rew: {avg_rew:.2f} | "
#                         f"Success: {avg_suc:.1%} | Time: {elapsed}")
#
#         if (episode + 1) % save_interval == 0:
#             mappo_agent.save(os.path.join(save_dir, f'mappo_ep{episode + 1}'))
#
#     return mappo_agent, results
#
#
# # ============================================================================
# # SECTION 7: EVALUATION & VISUALIZATION
# # ============================================================================
#
# def evaluate(mappo_agent, env, n_episodes=100, render=False):
#     logger.info("=" * 40)
#     logger.info(f"EVALUATION START ({n_episodes} episodes)")
#     logger.info("=" * 40)
#
#     success_count = 0
#     total_rewards = []
#
#     for i in range(n_episodes):
#         obs = env.reset(scenario_idx=i)
#         ep_reward = 0
#         done = False
#         truncated = False
#         crashed = False
#
#         while not (done or truncated):
#             # For evaluation, we ideally want deterministic actions (argmax),
#             # but standard PPO often just samples or takes mean.
#             # We will sample here as per standard PPO inference,
#             # or one could modify select_action to be deterministic.
#             actions = mappo_agent.select_action(obs)
#             obs, reward, done, truncated, info = env.step(actions)
#
#             if isinstance(reward, (list, tuple)):
#                 ep_reward += sum(reward)/len(reward)
#             else:
#                 ep_reward += reward
#
#             if info.get('crashed', False):
#                 crashed = True
#             if render:
#                 env.render()
#
#         total_rewards.append(ep_reward)
#         if not crashed:
#             success_count += 1
#
#         if (i+1) % 10 == 0:
#             logger.info(f"[Eval] {i+1}/{n_episodes} | Success: {success_count/(i+1):.2%}")
#
#     success_rate = success_count / n_episodes
#     avg_reward = np.mean(total_rewards)
#     logger.info(f"Evaluation Results: Success Rate: {success_rate:.2%} | Avg Reward: {avg_reward:.2f}")
#
#     return {'success_rate': success_rate, 'avg_reward': avg_reward}
#
# def save_results(train_results, eval_results, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, 'results.npz'),
#              rewards=train_results['rewards'],
#              losses=train_results['losses'])
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(np.convolve(train_results['rewards'], np.ones(100)/100, mode='valid'))
#     plt.title("Training Reward (Smoothed)")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.savefig(os.path.join(save_dir, 'reward_curve.png'))
#     plt.close()
#
#     with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
#         json.dump(eval_results, f, indent=4)
#     logger.info(f"Results saved to {save_dir}")
#
#
# # ============================================================================
# # SECTION 8: MAIN EXECUTION
# # ============================================================================
# def main():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_dir = f"results/mappo_{timestamp}"
#     os.makedirs(save_dir, exist_ok=True)
#
#     file_handler = logging.FileHandler(os.path.join(save_dir, 'experiment.log'))
#     file_handler.setFormatter(log_format)
#     logger.addHandler(file_handler)
#
#     scenario_manager = ScenarioManager()
#     n_agents = 2
#
#     # Obs Dim: vehicles_count (10) * features (4) = 40
#     # This matches the request that they see everyone.
#     obs_dim = 10 * 4
#     n_actions = 2
#
#     # ==========================================
#     # PHASE 1: TRAINING
#     # ==========================================
#     logger.info("Initializing Training Environment...")
#     env = IntersectionWrapper(scenario_manager, n_agents=n_agents, render_mode=None)
#
#     mappo_agent = MAPPO(
#         n_agents=n_agents,
#         obs_dim=obs_dim,
#         n_actions=n_actions,
#         lr_actor=3e-4,
#         lr_critic=1e-3,
#         gamma=0.99,
#         K_epochs=10,
#         eps_clip=0.2,
#         device=device
#     )
#
#     mappo_agent, train_results = train_mappo(
#         mappo_agent, env,
#         n_episodes=900, # Adjustable
#         update_timestep=2048,
#         save_dir=save_dir
#     )
#     env.close()
#
#     # ==========================================
#     # PHASE 2: VISUALIZATION
#     # ==========================================
#     logger.info("Initializing Evaluation Environment...")
#     visual_env = IntersectionWrapper(scenario_manager, n_agents=n_agents, render_mode='rgb_array')
#     eval_results = evaluate(mappo_agent, visual_env, n_episodes=100, render=True)
#
#     save_results(train_results, eval_results, save_dir)
#     visual_env.close()
#
# if __name__ == "__main__":
#     main()

"""
MAPPO for Unsignalized Intersection Navigation (Optimized)
==========================================================
Implementation of Multi-Agent PPO (IPPO with global observation).

Improvements:
- Generalized Advantage Estimation (GAE)
- Mini-batch updates
- Orthogonal Initialization
- Gradient Clipping
- Advantage Normalization
- Relative Observation Space (Better generalization)
- Entropy Annealing

Adapted by: Gemini
Date: 2025
"""
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
import random
import gymnasium as gym
import highway_env
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json
from copy import deepcopy
import warnings
import logging

# Try to import Experiment config, otherwise define defaults
try:
    from experiment.experiment_config import Experiment
except ImportError:
    class Experiment:
        COLLISION_REWARD = -5
        REACHED_TARGET_REWARD = 1
        STARVATION_REWARD = 0
        EPISODE_MAX_TIME = 100
        CARS_AMOUNT = 4
        SPAWN_PROBABILITY = 0.6
        SCREEN_WIDTH = 600
        SCREEN_HEIGHT = 600

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================
logger = logging.getLogger("MAPPO")
logger.setLevel(logging.INFO)

if not logger.handlers:
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('mappo_training.log', mode='w')

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ============================================================================
# SECTION 1: SCENARIO MANAGEMENT
# ============================================================================

def rotate_lane_id(lane_id, rotation):
    if len(lane_id) < 2 or not lane_id[-1].isdigit():
        return lane_id
    prefix = lane_id[:-1]
    direction = int(lane_id[-1])
    new_direction = (direction + rotation) % 4
    return prefix + str(new_direction)


def rotate_scenario_clockwise(scenario, rotation):
    rotated_scenario = []
    for lane_key, destination, offset in scenario:
        if isinstance(lane_key, tuple):
            rotated_lane = tuple(
                rotate_lane_id(part, rotation) if isinstance(part, str) else part
                for part in lane_key
            )
        else:
            rotated_lane = rotate_lane_id(lane_key, rotation)
        rotated_dest = rotate_lane_id(destination, rotation)
        rotated_scenario.append((rotated_lane, rotated_dest, offset))
    return rotated_scenario


class ScenarioManager:
    def __init__(self):
        self.all_scenarios = self._generate_all_scenarios()
        logger.info(f"[ScenarioManager] Generated {len(self.all_scenarios)} scenarios")

    def _get_base_scenarios(self):
        # Using the same base scenarios as provided in the original code
        base_scenarios = [
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", 0)],
             "static": [(('o0', 'ir0', 0), "o2", -50), (('o2', 'ir2', 0), "o0", -35), (('o0', 'ir0', 0), "o1", -70)]},
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -25), (('o3', 'ir3', 0), "o0", -35), (('o0', 'ir0', 0), "o1", -15)]},
            {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o2', 'ir2', 0), "o1", 0)],
             "static": [(('o1', 'ir1', 0), "o0", -20), (('o3', 'ir3', 0), "o2", -30), (('o0', 'ir0', 0), "o2", -40)]},
            {"agents": [(('o1', 'ir1', 0), "o3", 0), (('o3', 'ir3', 0), "o1", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -25), (('o2', 'ir2', 0), "o1", -35)]},
            {"agents": [(('o2', 'ir2', 0), "o0", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o0', 'ir0', 0), "o3", -45), (('o1', 'ir1', 0), "o2", -30), (('o2', 'ir2', 0), "o0", -20)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o0', 'ir0', 0), "o1", 20)],
             "static": [(('o0', 'ir0', 0), "o2", -50), (('o1', 'ir1', 0), "o3", -30), (('o2', 'ir2', 0), "o0", -25)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o3', 'ir3', 0), "o2", -40), (('o2', 'ir2', 0), "o1", -30), (('o0', 'ir0', 0), "o3", -50)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o2', 'ir2', 0), "o0", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -40), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -60)]},
            {"agents": [(('o3', 'ir3', 0), "o1", 0), (('o0', 'ir0', 0), "o2", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -20), (('o2', 'ir2', 0), "o0", -15), (('o3', 'ir3', 0), "o2", -45)]},
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o3', 'ir3', 0), "o0", 0)],
             "static": [(('o1', 'ir1', 0), "o2", -25), (('o2', 'ir2', 0), "o3", -30), (('o0', 'ir0', 0), "o3", -60)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o0', 'ir0', 0), "o2", -30)],
             "static": [(('o2', 'ir2', 0), "o0", -60), (('o1', 'ir1', 0), "o3", -55), (('o3', 'ir3', 0), "o1", -65)]},
            {"agents": [(('o1', 'ir1', 0), "o3", 0), (('o2', 'ir2', 0), "o0", 0)],
             "static": [(('o0', 'ir0', 0), "o2", -30), (('o3', 'ir3', 0), "o1", -25), (('o1', 'ir1', 0), "o0", -50)]},
            {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o1', 'ir1', 0), "o0", 0)],
             "static": [(('o2', 'ir2', 0), "o1", -35), (('o3', 'ir3', 0), "o2", -40), (('o0', 'ir0', 0), "o1", -55)]},
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o2', 'ir2', 0), "o3", 0)],
             "static": [(('o1', 'ir1', 0), "o2", -50), (('o3', 'ir3', 0), "o0", -55), (('o0', 'ir0', 0), "o2", -55)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", -40)],
             "static": [(('o2', 'ir2', 0), "o0", -50), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -20)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o2', 'ir2', 0), "o0", 0)],
             "static": [(('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -35), (('o0', 'ir0', 0), "o1", -10)]},
            {"agents": [(('o0', 'ir0', 0), "o3", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o1", -25), (('o3', 'ir3', 0), "o0", -30), (('o0', 'ir0', 0), "o2", -35)]},
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -50), (('o3', 'ir3', 0), "o0", -55), (('o0', 'ir0', 0), "o3", -65)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o1', 'ir1', 0), "o3", 50)],
             "static": [(('o2', 'ir2', 0), "o0", -60), (('o3', 'ir3', 0), "o1", -45), (('o0', 'ir0', 0), "o1", -75)]},
            {"agents": [(('o2', 'ir2', 0), "o3", 0), (('o3', 'ir3', 0), "o2", 15)],
             "static": [(('o0', 'ir0', 0), "o1", -30), (('o1', 'ir1', 0), "o0", -35), (('o2', 'ir2', 0), "o0", -50)]},
            {"agents": [(('o1', 'ir1', 0), "o2", 0), (('o0', 'ir0', 0), "o1", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -25), (('o3', 'ir3', 0), "o0", -30), (('o1', 'ir1', 0), "o3", -40)]},
            {"agents": [(('o0', 'ir0', 0), "o2", 0), (('o3', 'ir3', 0), "o1", 0)],
             "static": [(('o1', 'ir1', 0), "o0", -35), (('o2', 'ir2', 0), "o3", -40), (('o0', 'ir0', 0), "o3", -50)]},
            {"agents": [(('o0', 'ir0', 0), "o1", 0), (('o1', 'ir1', 0), "o2", 0)],
             "static": [(('o2', 'ir2', 0), "o3", -30), (('o3', 'ir3', 0), "o0", -35), (('o0', 'ir0', 0), "o2", -60)]},
            {"agents": [(('o2', 'ir2', 0), "o0", 0), (('o1', 'ir1', 0), "o3", 0)],
             "static": [(('o0', 'ir0', 0), "o2", -20), (('o3', 'ir3', 0), "o1", -25), (('o2', 'ir2', 0), "o1", -45)]},
            {"agents": [(('o3', 'ir3', 0), "o2", 0), (('o0', 'ir0', 0), "o3", 0)],
             "static": [(('o1', 'ir1', 0), "o0", -30), (('o2', 'ir2', 0), "o1", -35), (('o3', 'ir3', 0), "o1", -55)]}
        ]
        return base_scenarios

    def _generate_all_scenarios(self):
        base_scenarios = self._get_base_scenarios()
        all_scenarios = []
        for base_scenario in base_scenarios:
            all_scenarios.append(base_scenario)
            for rotation in [1, 2, 3]:
                rotated_agents = rotate_scenario_clockwise(base_scenario["agents"], rotation)
                rotated_static = rotate_scenario_clockwise(base_scenario["static"], rotation)
                all_scenarios.append({"agents": rotated_agents, "static": rotated_static})
        return all_scenarios

    def get_scenario(self, idx):
        return self.all_scenarios[idx]

    def get_random_scenario(self):
        return random.choice(self.all_scenarios)


# ============================================================================
# SECTION 2: PPO NETWORKS (OPTIMIZED)
# ============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for PPO stability"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# SECTION 3: ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


# ============================================================================
# SECTION 4: MAPPO AGENT (OPTIMIZED)
# ============================================================================

class PPOAgent:
    """Optimized PPO Agent with GAE and Mini-batch Updates"""

    def __init__(self, obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = 0.95  # GAE parameter
        self.buffer = RolloutBuffer()
        self.batch_size = 64  # Batch size for updates

        self.policy = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = Actor(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs = self.policy_old(state)

            if deterministic:
                action = torch.argmax(action_probs)
                action_logprob = torch.log(action_probs.squeeze(0)[action])
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                action_logprob = dist.log_prob(action)

            state_val = self.critic(state)

        if not deterministic:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.item()

    def update(self, entropy_coef=0.01):
        # Convert buffer to tensors
        old_states = torch.stack(self.buffer.states).detach().to(self.device).squeeze()
        old_actions = torch.stack(self.buffer.actions).detach().to(self.device).squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(self.device).squeeze()
        old_state_values = torch.stack(self.buffer.state_values).detach().to(self.device).squeeze()
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals

        # ----------------------------------------------------
        # GAE (Generalized Advantage Estimation) Calculation
        # ----------------------------------------------------
        advantages = []
        last_gae_lam = 0

        # Calculate advantages backwards
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - float(is_terminals[-1])
                # We assume next value is 0 for terminal state of batch or use bootstrapping if not done
                # Simplification: assume 0 if done, or we'd need 'next_state' in buffer
                next_value = 0
            else:
                next_non_terminal = 1.0 - float(is_terminals[i])
                next_value = old_state_values[i + 1].item()

            delta = rewards[i] + self.gamma * next_value * next_non_terminal - old_state_values[i].item()
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)

        # Normalizing the Advantages (Critical for convergence)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Returns = Advantages + Values
        returns = advantages + old_state_values

        # ----------------------------------------------------
        # Mini-batch Updates
        # ----------------------------------------------------
        dataset_size = len(old_states)
        batch_indices = BatchSampler(SubsetRandomSampler(range(dataset_size)), self.batch_size, drop_last=False)

        total_loss = 0

        for _ in range(self.K_epochs):
            for indices in batch_indices:
                indices = torch.tensor(indices).to(self.device)

                # Sample batch
                sample_states = old_states[indices]
                sample_actions = old_actions[indices]
                sample_logprobs = old_logprobs[indices]
                sample_advantages = advantages[indices]
                sample_returns = returns[indices]

                # Evaluate old actions and values
                action_probs = self.policy(sample_states)
                dist = Categorical(action_probs)
                logprobs = dist.log_prob(sample_actions)
                dist_entropy = dist.entropy()
                state_values = self.critic(sample_states).squeeze()

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - sample_logprobs)

                # Finding Surrogate Loss
                surr1 = ratios * sample_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * sample_advantages

                # Loss components
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * self.MseLoss(state_values, sample_returns)
                entropy_loss = -entropy_coef * dist_entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                # Take gradient step with clipping
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return total_loss / (self.K_epochs * len(batch_indices))

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.policy_old.load_state_dict(checkpoint['policy_old'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class MAPPO:
    """Wrapper class managing multiple PPO agents"""

    def __init__(self, n_agents, obs_dim, n_actions, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, K_epochs=10, eps_clip=0.2, device='cpu'):
        self.n_agents = n_agents
        self.agents = [
            PPOAgent(obs_dim, n_actions, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
            for _ in range(n_agents)
        ]

    def select_action(self, obs_list, deterministic=False):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(obs_list[i], deterministic)
            actions.append(action)
        return actions

    def store_rewards(self, rewards, dones):
        for i, agent in enumerate(self.agents):
            agent.buffer.rewards.append(rewards[i])
            agent.buffer.is_terminals.append(dones[i])

    def update(self, entropy_coef):
        losses = []
        for agent in self.agents:
            loss = agent.update(entropy_coef)
            losses.append(loss)
        return sum(losses) / len(losses)

    def save(self, path_prefix):
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent_{i}.pt")

    def load(self, path_prefix):
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent_{i}.pt")


# ============================================================================
# SECTION 5: ENVIRONMENT WRAPPER
# ============================================================================

class IntersectionWrapper:
    """Wrapper for highway-env intersection with GLOBAL observation"""

    def __init__(self, scenario_manager, n_agents=2, render_mode=None):
        self.scenario_manager = scenario_manager
        self.n_agents = n_agents

        self.vehicles_count = 10
        self.features_count = 4

        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20],
                },
                # IMPROVEMENT: Use Relative Coordinates (absolute=False)
                # This helps the model generalize faster (e.g., "object is 10m away")
                # rather than memorizing grid coordinates.
                "absolute": False,
                "flatten": True,
                "observe_intentions": False,
                "vehicles_count": self.vehicles_count,
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": False, "longitudinal": True,
                    "target_speeds": [5, 12],  # Increased high speed slightly to encourage faster completion
                },
            },
            "collision_reward": Experiment.COLLISION_REWARD,
            "arrived_reward": Experiment.REACHED_TARGET_REWARD,
            "normalize_reward": True,  # internal normalization helps convergence
            "starvation_reward": Experiment.STARVATION_REWARD,
            "offroad_terminal": False,
            "duration": Experiment.EPISODE_MAX_TIME,
            "initial_vehicle_count": Experiment.CARS_AMOUNT,
            "spawn_probability": Experiment.SPAWN_PROBABILITY,
            "screen_width": Experiment.SCREEN_WIDTH,
            "screen_height": Experiment.SCREEN_HEIGHT,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": True
        }

        self.env = gym.make('intersection-v0', render_mode=render_mode, config=config)
        logger.info(f"[Env] IntersectionWrapper Initialized (Mode: {render_mode}, Relative Obs)")

    def reset(self, scenario_idx=None):
        obs_raw, info = self.env.reset()
        if scenario_idx is None:
            scenario = self.scenario_manager.get_random_scenario()
        else:
            scenario = self.scenario_manager.get_scenario(scenario_idx)

        obs = []
        expected_dim = self.vehicles_count * self.features_count

        for i in range(self.n_agents):
            if i < len(obs_raw):
                agent_obs = obs_raw[i]
                if len(agent_obs) < expected_dim:
                    agent_obs = np.pad(agent_obs, (0, expected_dim - len(agent_obs)))
                obs.append(agent_obs)
            else:
                obs.append(np.zeros(expected_dim))

        return obs

    def step(self, actions):
        if isinstance(actions, list):
            actions = tuple(actions)
        obs_raw, reward, done, truncated, info = self.env.step(actions)

        obs = []
        expected_dim = self.vehicles_count * self.features_count

        for i in range(self.n_agents):
            if i < len(obs_raw):
                agent_obs = obs_raw[i]
                if len(agent_obs) < expected_dim:
                    agent_obs = np.pad(agent_obs, (0, expected_dim - len(agent_obs)))
                obs.append(agent_obs)
            else:
                obs.append(np.zeros(expected_dim))

        return obs, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================================
# SECTION 6: TRAINING LOOP (OPTIMIZED)
# ============================================================================

def train_mappo(
        mappo_agent,
        env,
        n_episodes=5000,
        update_timestep=2048,  # Increased batch size for more stable GAE
        save_interval=500,
        save_dir='results'
):
    results = {'rewards': [], 'losses': [], 'success_rates': []}

    logger.info("=" * 40)
    logger.info("STARTING OPTIMIZED MAPPO TRAINING")
    logger.info("=" * 40)

    csv_file_path = os.path.join(save_dir, 'training_log.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['episode', 'result', 'reward', 'steps'])

    time_step = 0
    start_time = datetime.now()

    # ENTROPY ANNEALING
    # Start high to explore, end low to exploit (drive fast and safe)
    entropy_start = 0.05
    entropy_end = 0.005
    entropy_decay_episodes = n_episodes * 0.8

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        crashed = False
        step_count = 0
        done = False
        truncated = False

        # Calculate current entropy coefficient
        progress = min(1.0, episode / entropy_decay_episodes)
        current_entropy = entropy_start - (entropy_start - entropy_end) * progress

        while not (done or truncated):
            time_step += 1
            step_count += 1

            # Select Action
            actions = mappo_agent.select_action(obs, deterministic=False)

            # Step Env
            next_obs, reward, done, truncated, info = env.step(actions)

            dones_list = [done] * mappo_agent.n_agents

            if isinstance(reward, float):
                rewards_list = [reward] * mappo_agent.n_agents
            else:
                rewards_list = list(reward)
                while len(rewards_list) < mappo_agent.n_agents:
                    rewards_list.append(0.0)

            mappo_agent.store_rewards(rewards_list, dones_list)

            episode_reward += sum(rewards_list) / mappo_agent.n_agents
            obs = next_obs

            if info.get('crashed', False):
                crashed = True

            # PPO Update
            if time_step % update_timestep == 0:
                loss = mappo_agent.update(entropy_coef=current_entropy)
                results['losses'].append(loss)
                logger.debug(f"[Update] Step {time_step} | Loss: {loss:.4f} | Ent: {current_entropy:.4f}")

        # Logging
        if crashed:
            outcome = "Collision"
        elif done and not crashed:
            outcome = "Success"
        else:
            outcome = "Timeout"

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode + 1, outcome, round(episode_reward, 2), step_count])

        results['rewards'].append(episode_reward)
        results['success_rates'].append(0 if crashed else 1)

        if (episode + 1) % 100 == 0:
            avg_rew = np.mean(results['rewards'][-100:])
            avg_suc = np.mean(results['success_rates'][-100:])
            elapsed = datetime.now() - start_time
            logger.info(f"Episode {episode + 1}/{n_episodes} | Rew: {avg_rew:.2f} | "
                        f"Success: {avg_suc:.1%} | Time: {elapsed}")

        if (episode + 1) % save_interval == 0:
            mappo_agent.save(os.path.join(save_dir, f'mappo_ep{episode + 1}'))

    return mappo_agent, results


# ============================================================================
# SECTION 7: EVALUATION & VISUALIZATION
# ============================================================================

def evaluate(mappo_agent, env, n_episodes=100, render=False):
    logger.info("=" * 40)
    logger.info(f"EVALUATION START ({n_episodes} episodes)")
    logger.info("=" * 40)

    success_count = 0
    total_rewards = []
    total_steps = []

    for i in range(n_episodes):
        obs = env.reset(scenario_idx=i)
        ep_reward = 0
        steps = 0
        done = False
        truncated = False
        crashed = False

        while not (done or truncated):
            # Deterministic action selection for best performance
            actions = mappo_agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(actions)

            steps += 1
            if isinstance(reward, (list, tuple)):
                ep_reward += sum(reward) / len(reward)
            else:
                ep_reward += reward

            if info.get('crashed', False):
                crashed = True
            if render:
                env.render()

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        if not crashed:
            success_count += 1

        if (i + 1) % 10 == 0:
            logger.info(f"[Eval] {i + 1}/{n_episodes} | Success: {success_count / (i + 1):.2%}")

    success_rate = success_count / n_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    logger.info(f"Results: Success: {success_rate:.2%} | Avg Rew: {avg_reward:.2f} | Avg Steps: {avg_steps:.1f}")

    return {'success_rate': success_rate, 'avg_reward': avg_reward, 'avg_steps': avg_steps}


def save_results(train_results, eval_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'results.npz'),
             rewards=train_results['rewards'],
             losses=train_results['losses'])

    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(train_results['rewards'], np.ones(100) / 100, mode='valid'))
    plt.title("Training Reward (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, 'reward_curve.png'))
    plt.close()

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Results saved to {save_dir}")


# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/mappo_opt_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(save_dir, 'experiment.log'))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    scenario_manager = ScenarioManager()
    n_agents = 2

    # Obs Dim: vehicles_count (10) * features (4)
    obs_dim = 10 * 4
    n_actions = 2

    # ==========================================
    # PHASE 1: TRAINING
    # ==========================================
    logger.info("Initializing Training Environment...")
    env = IntersectionWrapper(scenario_manager, n_agents=n_agents, render_mode=None)

    mappo_agent = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr_actor=3e-4,  # Standard efficient rate
        lr_critic=5e-4,  # Slightly higher for critic
        gamma=0.99,
        K_epochs=15,  # More epochs per batch for sample efficiency
        eps_clip=0.2,
        device=device
    )

    mappo_agent, train_results = train_mappo(
        mappo_agent, env,
        n_episodes=900,
        update_timestep=2048,
        save_dir=save_dir
    )
    env.close()

    # ==========================================
    # PHASE 2: VISUALIZATION
    # ==========================================
    logger.info("Initializing Evaluation Environment...")
    visual_env = IntersectionWrapper(scenario_manager, n_agents=n_agents, render_mode='rgb_array')
    eval_results = evaluate(mappo_agent, visual_env, n_episodes=100, render=True)

    save_results(train_results, eval_results, save_dir)
    visual_env.close()


if __name__ == "__main__":
    main()
