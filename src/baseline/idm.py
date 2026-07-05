import os
import time
import json
import logging
import numpy as np

from src.experiment.experiment_config import Experiment
from src.training.general_utils import setup_experiment_dirs
from src.model.agent_handler import Driver, DummyVecEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IDMAgent:
    def __init__(self, wrapped_env):
        self.env = wrapped_env
        self.action_space = wrapped_env.action_space

    def predict(self, observation, deterministic=True):
        env_unwrapped = self.env.envs[0]._get_unwrapped_env()
        actions = []
        from highway_env.vehicle.behavior import IDMVehicle
        
        for v in env_unwrapped.controlled_vehicles:
            if hasattr(v, 'is_arrived') and v.is_arrived:
                actions.append(0)
                continue
                
            # Create a temporary IDM vehicle to calculate acceleration
            idm_v = IDMVehicle.create_from(v)
            idm_v.target_lane_index = v.target_lane_index
            idm_v.route = v.route
            
            front_vehicle, rear_vehicle = idm_v.road.neighbour_vehicles(idm_v, idm_v.lane_index)
            acc = idm_v.acceleration(ego_vehicle=idm_v, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
            
            # --- Intersection Yielding Logic ---
            d_ego = np.linalg.norm(v.position)
            vel_ego = np.array([v.speed * np.cos(v.heading), v.speed * np.sin(v.heading)])
            if hasattr(v, 'velocity'): vel_ego = v.velocity
                
            approaching_ego = np.dot(v.position, vel_ego) < 0
            
            if approaching_ego and d_ego < 50.0:
                tti_ego = d_ego / max(v.speed, 0.1)
                
                for other in env_unwrapped.road.vehicles:
                    if other is v:
                        continue
                        
                    # Ignore vehicles in our own lane since IDM handles them
                    if hasattr(other, 'lane_index') and other.lane_index == v.lane_index:
                        continue
                        
                    d_other = np.linalg.norm(other.position)
                    vel_other = np.array([other.speed * np.cos(other.heading), other.speed * np.sin(other.heading)])
                    if hasattr(other, 'velocity'): vel_other = other.velocity
                        
                    approaching_other = np.dot(other.position, vel_other) < 0
                    
                    # 1. Other vehicle is already inside the intersection
                    if d_other < 10.0:
                        if d_ego < 30.0:
                            acc = -5.0
                            break
                            
                    # 2. Other vehicle is also approaching
                    if approaching_other and d_other < 50.0:
                        tti_other = d_other / max(other.speed, 0.1)
                        
                        # Collision risk: arriving around the same time
                        if abs(tti_ego - tti_other) < 2.5:
                            # Rule: vehicle further away yields
                            if d_ego > d_other + 0.5:
                                acc = -5.0
                                break
                            # Tie-break using memory address
                            elif abs(d_ego - d_other) <= 0.5 and id(v) > id(other):
                                acc = -5.0
                                break
            # -----------------------------------
            
            # Map IDM acceleration to discrete actions: 1 = FASTER, 0 = SLOWER
            if acc > 0:
                actions.append(1)
            else:
                actions.append(0)
                
        return np.array([actions]), None


def run_idm_experiment(experiment_config, env_config):
    logger.info("Running IDM Baseline (Inference Only)")
    
    setup_experiment_dirs(experiment_config.EXPERIMENT_PATH)
    experiment_config.CONFIG = env_config
    
    env_fn = lambda: Driver(experiment_config)
    wrapped_env = DummyVecEnv([env_fn])
    
    agent_model = IDMAgent(wrapped_env)
    
    results = {
        "episode_rewards": [],
        "success_flags": [],
        "collision_flags": [],
        "episode_lengths": []
    }
    
    collision_counter = 0
    total_episodes = experiment_config.CYCLES * experiment_config.EPISODES_PER_CYCLE
    
    for episode in range(1, total_episodes + 1):
        obs, _ = wrapped_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        crashed = False
        
        while not done and not truncated:
            action, _ = agent_model.predict(obs, deterministic=True)
            action = np.asarray(action).astype(np.int64).flatten()
            
            obs, reward, done, truncated, info = wrapped_env.step(tuple(int(a) for a in action))
            
            episode_reward += reward
            steps += 1
            wrapped_env.envs[0].render()
            
            if done and info.get("crashed", False):
                crashed = True
                
        if crashed:
            collision_counter += 1
            logger.warning(f"Episode {episode} ended with Collision")
        else:
            logger.info(f"Episode {episode} ended with Success")
            
        logger.info(f"Result: {'Collision' if crashed else 'Success'} | Reward: {episode_reward:.2f} | Steps: {steps}")
        
        results["episode_rewards"].append(float(episode_reward))
        results["success_flags"].append(float(not crashed))
        results["collision_flags"].append(float(crashed))
        results["episode_lengths"].append(float(steps))
        
    wrapped_env.close()
    
    logger.info(f"IDM evaluation completed. Total collisions: {collision_counter}")
    
    # We must return a structure that the caller expects, which is typically:
    # agent_model, master_model, collision_counter, history, all_actions, results
    # or similar.
    # From run_parallel_experiment.py:
    # _, history, _ = run_experiment(config, env_config)
    # So we return None, results, None
    return None, results, None
