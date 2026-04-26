# ntrain.py
"""
Training with coordination rewards that require rich embeddings.
"""

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from nmaster import Master
from nagent import Agent
import os
import csv


def make_env(config):
    env = gym.make("intersection-v1", render_mode=None)
    env.unwrapped.configure({
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["n_agents"],
                "see_behind": True,
            }
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "target_speeds": config["target_speeds"],
                "longitudinal": True,
                "lateral": False,
            }
        },
        "duration": config["duration"],
        "controlled_vehicles": config["n_agents"],
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "collision_reward": config["collision_reward"],
        "high_speed_reward": config["high_speed_reward"],
        "arrived_reward": config["arrived_reward"],
        "reward_speed_range": config["reward_speed_range"],
        "policy_frequency": 1,
        "simulation_frequency": 15,
    })
    env.reset()
    return env


def get_global_state(obs, n_agents):
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs.flatten().astype(np.float32)


def get_local_obs_list(obs, n_agents):
    if isinstance(obs, tuple):
        return [o[0].astype(np.float32) for o in obs]
    return [obs[i][0].astype(np.float32) for i in range(n_agents)]


def compute_min_distance(local_obs_list):
    positions = np.array([[obs[0], obs[1]] for obs in local_obs_list])
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, dist)
    return min_dist if min_dist != float('inf') else 100.0


def compute_coordination_reward(local_obs_list, actions, config):
    """
    Compute coordination reward that requires RICH encoding:
    - Bonus for smooth speed distribution (not all same, not all different)
    - Bonus for maintaining safe distances
    - Penalty for dangerous proximity
    
    This reward structure REQUIRES understanding of ALL agent states,
    which dim=2 cannot encode but dim=4 can.
    """
    n_agents = len(local_obs_list)
    
    # Extract velocities
    velocities = np.array([obs[2] for obs in local_obs_list])
    positions = np.array([[obs[0], obs[1]] for obs in local_obs_list])
    
    reward = 0.0
    
    # 1. Velocity coordination: reward for diverse but coordinated speeds
    # Ideal: some agents slow, some fast - requires understanding global state
    vel_std = np.std(velocities)
    vel_mean = np.mean(np.abs(velocities))
    
    # Sweet spot: moderate variance, good mean speed
    if 2.0 < vel_std < 8.0 and vel_mean > 5.0:
        reward += config.get("coord_velocity_bonus", 1.0)
    
    # 2. Safety coordination: reward for maintaining safe distances
    min_dist = compute_min_distance(local_obs_list)
    
    if min_dist > 15.0:  # Very safe
        reward += config.get("coord_safe_bonus", 2.0)
    elif min_dist > 10.0:  # Safe
        reward += config.get("coord_safe_bonus", 1.0)
    elif min_dist < 5.0:  # Dangerous
        reward -= config.get("coord_danger_penalty", 3.0)
    
    # 3. Action diversity: reward for coordinated but different actions
    # This requires understanding which agent should do what
    unique_actions = len(set(actions))
    if 2 <= unique_actions <= n_agents - 1:  # Some diversity
        reward += config.get("coord_diversity_bonus", 0.5)
    
    return reward


def compute_action_diversity(actions):
    unique, counts = np.unique(actions, return_counts=True)
    probs = counts / len(actions)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    return float(entropy)


class ExperimentLogger:
    def __init__(self, save_dir: str, embedding_dim: int):
        self.save_dir = save_dir
        self.embedding_dim = embedding_dim
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_data = []
        self.step_data = []
        self.master_loss_data = []
        self.agent_loss_data = []
        self.embedding_data = []
        
    def log_episode(self, episode, reward, crashed, steps, mean_speed, arrived_count, 
                    min_distance, action_entropy, embedding_variance, coord_reward):
        self.episode_data.append({
            "embedding_dim": self.embedding_dim,
            "episode": episode,
            "reward": reward,
            "crashed": int(crashed),
            "steps": steps,
            "mean_speed": mean_speed,
            "arrived_count": arrived_count,
            "min_distance": min_distance,
            "action_entropy": action_entropy,
            "embedding_variance": embedding_variance,
            "coord_reward": coord_reward,
        })
    
    def log_step(self, episode, step, embedding, actions, reward, crashed, min_dist):
        if step % 5 == 0:
            self.step_data.append({
                "embedding_dim": self.embedding_dim,
                "episode": episode,
                "step": step,
                "embedding": embedding.tolist(),
                "embedding_norm": float(np.linalg.norm(embedding)),
                "embedding_std": float(np.std(embedding)),
                "actions": actions,
                "reward": reward,
                "crashed": int(crashed),
                "min_distance": min_dist,
            })
    
    def log_master_loss(self, episode, stats):
        self.master_loss_data.append({
            "embedding_dim": self.embedding_dim,
            "episode": episode,
            **stats
        })
    
    def log_agent_loss(self, episode, stats):
        self.agent_loss_data.append({
            "embedding_dim": self.embedding_dim,
            "episode": episode,
            **stats
        })
    
    def log_embedding(self, episode, step, embedding, crashed, reward):
        self.embedding_data.append({
            "embedding_dim": self.embedding_dim,
            "episode": episode,
            "step": step,
            "embedding": embedding.tolist(),
            "crashed": int(crashed),
            "reward": reward,
        })
        
    def save_all(self):
        for name, data in [
            ("episodes", self.episode_data),
            ("steps", self.step_data),
            ("master_loss", self.master_loss_data),
            ("agent_loss", self.agent_loss_data),
            ("embeddings", self.embedding_data),
        ]:
            if data:
                filepath = os.path.join(self.save_dir, f"{name}_dim{self.embedding_dim}.csv")
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)


def train(config, embedding_dim: int, logger: ExperimentLogger):
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    config["embedding_dim"] = embedding_dim
    n_agents = config["n_agents"]
    env = make_env(config)
    
    master = Master(config)
    agent = Agent(config)
    
    total_episodes = config["total_episodes"]
    
    print(f"\n{'='*60}")
    print(f"Training with embedding_dim = {embedding_dim}")
    print(f"{'='*60}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        ep_coord_reward = 0.0
        crashed = False
        step = 0
        speeds = []
        arrived_count = 0
        min_distances = []
        action_entropies = []
        embedding_variances = []
        
        while not done and not truncated:
            global_state = get_global_state(obs, n_agents)
            embedding, action_bias = master.get_action(global_state, deterministic=False)
            
            local_obs_list = get_local_obs_list(obs, n_agents)
            
            # Agent receives embedding directly (not action_bias)
            actions = agent.get_actions(local_obs_list, embedding, deterministic=False)
            
            for lo in local_obs_list:
                speeds.append(float(np.abs(lo[2])))
            
            min_dist = compute_min_distance(local_obs_list)
            min_distances.append(min_dist)
            
            action_div = compute_action_diversity(actions)
            action_entropies.append(action_div)
            embedding_variances.append(float(np.var(embedding)))
            
            next_obs, env_reward, done, truncated, info = env.step(tuple(actions))
            
            # Add coordination reward
            coord_reward = compute_coordination_reward(local_obs_list, actions, config)
            total_reward = float(env_reward) + coord_reward
            
            ep_reward += total_reward
            ep_coord_reward += coord_reward
            
            step_crashed = float(env_reward) < -100
            if step_crashed:
                crashed = True
            
            if info.get("arrived", False):
                arrived_count += 1
            
            # Logging
            logger.log_step(ep, step, embedding, actions, total_reward, step_crashed, min_dist)
            if step % 3 == 0:
                logger.log_embedding(ep, step, embedding, step_crashed, total_reward)
            
            next_global_state = get_global_state(next_obs, n_agents)
            next_local_obs_list = get_local_obs_list(next_obs, n_agents)
            
            # Store transitions
            master.store(global_state, embedding, action_bias, total_reward, 
                        next_global_state, done or truncated)
            
            agent_reward = total_reward / n_agents
            agent.store(local_obs_list, embedding, actions,
                       [agent_reward] * n_agents, next_local_obs_list,
                       [done or truncated] * n_agents)
            
            obs = next_obs
            step += 1
        
        # Episode stats
        mean_speed = np.mean(speeds) if speeds else 0.0
        mean_min_dist = np.mean(min_distances) if min_distances else 0.0
        mean_action_entropy = np.mean(action_entropies) if action_entropies else 0.0
        mean_emb_var = np.mean(embedding_variances) if embedding_variances else 0.0
        
        logger.log_episode(ep, ep_reward, crashed, step, mean_speed, arrived_count,
                          mean_min_dist, mean_action_entropy, mean_emb_var, ep_coord_reward)
        
        # Train
        if (ep + 1) % config["train_every_n_episodes"] == 0:
            m_stats = master.train()
            if m_stats:
                logger.log_master_loss(ep, m_stats)
            
            a_stats = agent.train()
            if a_stats:
                logger.log_agent_loss(ep, a_stats)
        
        # Progress
        if (ep + 1) % config["print_every_n_episodes"] == 0:
            recent = logger.episode_data[-50:]
            r = np.mean([d["reward"] for d in recent]) if recent else 0.0
            nc = 1.0 - np.mean([d["crashed"] for d in recent]) if recent else 0.0
            print(f"  Ep {ep+1}/{total_episodes} | R: {r:.1f} | NoCrash: {nc:.0%}")
    
    master.save(f"{save_dir}/master_dim{embedding_dim}")
    agent.save(f"{save_dir}/agent_dim{embedding_dim}")
    
    env.close()
    return logger
