# ntest.py
"""
OOD Evaluation - Tests generalization to unseen scenarios.
"""

import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd
from typing import List

from nmaster import Master
from nagent import Agent
# Replace HARD_SCENARIOS with the original OOD scenarios
HARD_SCENARIOS = {
    "normal": {
        "target_speeds": [0, 10, 20],
        "duration": 25,
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "collision_reward": -300,
        "high_speed_reward": 0.3,
        "arrived_reward": 50,
        "reward_speed_range": [0, 9],
    },
    "high_speed": {
        "target_speeds": [0, 15, 30],
        "duration": 25,
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "collision_reward": -300,
        "high_speed_reward": 0.5,
        "arrived_reward": 60,
        "reward_speed_range": [0, 15],
    },
    "long_duration": {
        "target_speeds": [0, 10, 20],
        "duration": 50,
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "collision_reward": -300,
        "high_speed_reward": 0.3,
        "arrived_reward": 50,
        "reward_speed_range": [0, 9],
    },
    "dense_traffic": {
        "target_speeds": [0, 10, 20],
        "duration": 25,
        "initial_vehicle_count": 3,
        "spawn_probability": 0.3,
        "collision_reward": -300,
        "high_speed_reward": 0.3,
        "arrived_reward": 50,
        "reward_speed_range": [0, 9],
    },
    "strict_penalty": {
        "target_speeds": [0, 10, 20],
        "duration": 25,
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "collision_reward": -500,
        "high_speed_reward": 0.3,
        "arrived_reward": 50,
        "reward_speed_range": [0, 9],
    },
}

def make_env_ood(config, scenario: str):
    env = gym.make("intersection-v1", render_mode=None)
    
    base_config = {
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["n_agents"],
                "see_behind": True,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "target_speeds": config["target_speeds"],
                "longitudinal": True,
                "lateral": False,
            },
        },
        "controlled_vehicles": config["n_agents"],
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "policy_frequency": 1,
        "simulation_frequency": 15,
    }
    
    if scenario == "normal":
        base_config.update({
            "duration": config["duration"],
            "collision_reward": config["collision_reward"],
            "high_speed_reward": config["high_speed_reward"],
            "arrived_reward": config["arrived_reward"],
            "reward_speed_range": config["reward_speed_range"],
        })
    elif scenario == "high_speed":
        base_config["action"]["action_config"]["target_speeds"] = [0, 15, 30]
        base_config.update({
            "duration": 25,
            "collision_reward": -300,
            "high_speed_reward": 0.5,
            "arrived_reward": 60,
            "reward_speed_range": [0, 15],
        })
    elif scenario == "long_duration":
        base_config.update({
            "duration": 50,
            "collision_reward": config["collision_reward"],
            "high_speed_reward": config["high_speed_reward"],
            "arrived_reward": config["arrived_reward"],
            "reward_speed_range": config["reward_speed_range"],
        })
    elif scenario == "dense_traffic":
        base_config.update({
            "duration": config["duration"],
            "initial_vehicle_count": 3,
            "spawn_probability": 0.3,
            "collision_reward": config["collision_reward"],
            "high_speed_reward": config["high_speed_reward"],
            "arrived_reward": config["arrived_reward"],
            "reward_speed_range": config["reward_speed_range"],
        })
    elif scenario == "strict_penalty":
        base_config.update({
            "duration": config["duration"],
            "collision_reward": -500,
            "high_speed_reward": config["high_speed_reward"],
            "arrived_reward": config["arrived_reward"],
            "reward_speed_range": config["reward_speed_range"],
        })
    
    env.unwrapped.configure(base_config)
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


def evaluate_scenario(config, master, agent, scenario: str, n_episodes: int = 30):
    env = make_env_ood(config, scenario)
    n_agents = config["n_agents"]
    
    results = []
    
    for mode in ["trained", "zero", "random"]:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = truncated = False
            ep_reward = 0.0
            crashed = False
            steps = 0
            
            while not done and not truncated:
                global_state = get_global_state(obs, n_agents)
                local_obs_list = get_local_obs_list(obs, n_agents)
                
                if mode == "trained":
                    embedding, _ = master.get_action(global_state, deterministic=True)
                elif mode == "zero":
                    embedding = np.zeros(config["embedding_dim"], dtype=np.float32)
                else:  # random
                    embedding = np.random.uniform(-1, 1, config["embedding_dim"]).astype(np.float32)
                
                actions = agent.get_actions(local_obs_list, embedding, deterministic=True)
                
                obs, reward, done, truncated, info = env.step(tuple(actions))
                ep_reward += float(reward)
                steps += 1
                
                if info.get("crashed", False):
                    crashed = True
            
            results.append({
                "scenario": scenario,
                "mode": mode,
                "episode": ep,
                "reward": ep_reward,
                "crashed": int(crashed),
                "steps": steps,
            })
    
    env.close()
    return results


def run_ood_evaluation(config, models_dir: str, embedding_dims: List[int], save_dir: str):
    scenarios = ["normal", "high_speed", "long_duration", "dense_traffic", "strict_penalty"]
    all_results = []
    
    for dim in embedding_dims:
        print(f"\nEvaluating dim={dim}")
        
        test_config = config.copy()
        test_config["embedding_dim"] = dim
        
        master = Master(test_config)
        agent = Agent(test_config)
        
        try:
            master.load(f"{models_dir}/master_dim{dim}")
            agent.load(f"{models_dir}/agent_dim{dim}")
        except FileNotFoundError:
            print(f"  Models not found, skipping...")
            continue
        
        for scenario in scenarios:
            print(f"  Scenario: {scenario}")
            results = evaluate_scenario(test_config, master, agent, scenario)
            for r in results:
                r["embedding_dim"] = dim
            all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "ood_evaluation.csv"), index=False)
    
    print("\n" + "="*60)
    print("OOD EVALUATION SUMMARY")
    print("="*60)
    summary = df.groupby(['embedding_dim', 'scenario', 'mode']).agg({
        'reward': ['mean', 'std'],
        'crashed': 'mean',
    }).round(2)
    print(summary)
    
    return df
