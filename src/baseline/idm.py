import os
import csv
import logging
import json
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IDMModel:
    def __init__(self, env, num_agents: int, target_speeds: list, max_speed: float = 10.0,
                 comfort_acc: float = 1.0, comfort_dec: float = 1.5,
                 safe_time_headway: float = 1.5, min_gap: float = 2.0,
                 delta: float = 4.0, intersection_yield_dist: float = 15.0):
        self.env = env
        self.num_agents = num_agents
        self.target_speeds = target_speeds
        
        # IDM Parameters
        self.v0 = max_speed
        self.a = comfort_acc
        self.b = comfort_dec
        self.T = safe_time_headway
        self.s0 = min_gap
        self.delta = delta
        
        # Intersection heuristic parameters
        self.intersection_yield_dist = intersection_yield_dist

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        unwrapped_env = getattr(self.env, 'unwrapped', self.env)
        
        if hasattr(unwrapped_env, "envs"):
            unwrapped_env = unwrapped_env.envs[0].unwrapped
            
        controlled_vehicles = getattr(unwrapped_env, "controlled_vehicles", [])
        
        actions = []
        for i in range(self.num_agents):
            if i >= len(controlled_vehicles):
                actions.append(0)
                continue
                
            ego = controlled_vehicles[i]
            if getattr(ego, "is_arrived", False) or getattr(ego, "crashed", False):
                actions.append(0)
                continue
                
            # Get the front vehicle on the same lane or a crossing vehicle at the intersection
            front_vehicle, distance = self._get_front_vehicle(unwrapped_env, ego)
            
            # IDM formula
            v = ego.speed
            
            if front_vehicle is not None:
                if ego.lane_index == front_vehicle.lane_index:
                    delta_v = v - front_vehicle.speed
                else:
                    delta_v = v # treat as stationary obstacle at the intersection
                s = max(distance, 0.1) # avoid division by zero
                s_star = self.s0 + max(0, v * self.T + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))
                interaction_term = (s_star / s) ** 2
            else:
                interaction_term = 0.0
                
            # Compute IDM acceleration
            acceleration = self.a * (1 - (v / self.v0) ** self.delta - interaction_term)
            
            # Map acceleration to one of the target speeds [5, 10]
            # Since action 0 maps to 5 and action 1 maps to 10
            if acceleration > 0:
                desired_speed = v + acceleration
                if desired_speed > 7.5:
                    action = 1 # correspond to 10 m/s
                else:
                    action = 0 # 5 m/s
            else:
                action = 0 # slow down to 5 m/s

            actions.append(action)
            
        return np.array(actions), None
        
    def _get_front_vehicle(self, env, ego):
        if not hasattr(env, "road") or env.road is None:
            return None, float('inf')
            
        best_vehicle = None
        min_dist = float('inf')
        
        # Ego distance to intersection center (0,0)
        ego_dist_to_center = np.linalg.norm(ego.position)
        
        for v in env.road.vehicles:
            if v is ego or getattr(v, "crashed", False):
                continue
                
            # Same lane check
            if ego.lane_index == v.lane_index:
                dist = ego.lane_distance_to(v)
                if 0 < dist < min_dist:
                    min_dist = dist
                    best_vehicle = v
            else:
                # Intersection yielding heuristic
                # If we are approaching the intersection and the other vehicle is also approaching
                v_dist_to_center = np.linalg.norm(v.position)
                
                # Check if ego is before the intersection and other vehicle is also around
                if ego_dist_to_center < self.intersection_yield_dist and v_dist_to_center < self.intersection_yield_dist:
                    # Yield if the other vehicle is closer to the center or very close to it
                    if v_dist_to_center < ego_dist_to_center + 1.0:
                        # Treat it as an obstacle at the intersection boundary (stop before entering)
                        virtual_dist = max(0.1, ego_dist_to_center - 7.0)
                        if virtual_dist < min_dist:
                            min_dist = virtual_dist
                            best_vehicle = v

        return best_vehicle, min_dist

class IDMTrainer:
    def __init__(self, experiment_config, env_config):
        self.experiment_config = experiment_config
        self.algorithm = "idm"
        self.env = gym.make(experiment_config.ENV_ID, render_mode=experiment_config.RENDER_MODE, config=env_config)
        self.num_agents = len(env_config["controlled_cars"])
        self.total_episodes = int(experiment_config.EPISODES_PER_CYCLE * experiment_config.CYCLES)
        
        target_speeds = env_config.get("action", {}).get("target_speeds", [5, 10])
        
        self.agent_model = IDMModel(
            env=self.env,
            num_agents=self.num_agents,
            target_speeds=target_speeds,
            max_speed=10.0,
            comfort_acc=1.5,
            comfort_dec=2.0,
            safe_time_headway=1.5,
            min_gap=2.0,
            intersection_yield_dist=25.0
        )
        
        self.history = {
            "episode_rewards": [],
            "success_flags": [],
            "collision_flags": [],
            "episode_lengths": []
        }

    def close(self):
        self.env.close()

    def _has_any_controlled_collision(self, info: dict) -> bool:
        if bool(info.get("crashed", False)):
            return True
        controlled_vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        return any(getattr(vehicle, "crashed", False) for vehicle in controlled_vehicles[: self.num_agents])

    def train(self):
        collision_count = 0
        
        for episode_index in range(1, self.total_episodes + 1):
            obs, _ = self.env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            collision_occurred = False
            
            while not terminated and not truncated:
                action_tuple, _ = self.agent_model.predict(obs, deterministic=True)
                
                if self.experiment_config.RENDER_MODE is not None:
                    self.env.render()
                    
                obs, reward, terminated, truncated, info = self.env.step(tuple(int(a) for a in action_tuple))
                
                episode_reward += float(reward)
                episode_length += 1
                collision_occurred = collision_occurred or self._has_any_controlled_collision(info)
            
            episode_success = bool(terminated and not truncated and not collision_occurred)
            collision_count += int(collision_occurred)
            
            self.history["episode_rewards"].append(episode_reward)
            self.history["success_flags"].append(int(episode_success))
            self.history["collision_flags"].append(int(collision_occurred))
            self.history["episode_lengths"].append(episode_length)
            
            logger.info(
                "[%s] Episode %d/%d | reward=%.2f | success=%s | collision=%s",
                self.algorithm,
                episode_index,
                self.total_episodes,
                episode_reward,
                episode_success,
                collision_occurred
            )
            
        return collision_count, self.history

def run_idm_experiment(experiment_config, env_config):
    trainer = IDMTrainer(experiment_config, env_config)
    try:
        collision_count, history = trainer.train()
    finally:
        trainer.close()
        
    return None, history, collision_count
