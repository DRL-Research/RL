# prepare.py (Immutable / AutoResearch Test Harness)
# WARNING: DO NOT MODIFY THIS FILE. The AI agent must only modify train.py.

import os
import sys
import csv
import time
from datetime import datetime
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src.experiment import scenarios_config as sc
from src.training.general_utils import initialize_models, ensure_tensor, get_scaler_action_and_action_array

# Standardize environment registration
patch_intersection_env()
register_intersection_env()

# Fixed validation seeds to ensure perfect reproducibility across experiments
VALIDATION_SEEDS = [42, 100, 2026]
RESULTS_FILE = "results.tsv"

def get_validation_seeds():
    """Return the list of fixed seeds used for model validation."""
    return VALIDATION_SEEDS

def evaluate_model(experiment_config, env_config, master_model, agent_model, seeds=None, episodes_per_seed=3, render=False):
    """
    Standardized evaluation function.
    Runs the cooperative MARL model deterministically across fixed seeds.
    Returns: (val_success_rate, val_mean_reward, val_mean_collisions)
    """
    if seeds is None:
        seeds = VALIDATION_SEEDS
    
    # Temporarily force render mode if requested
    original_render_mode = experiment_config.RENDER_MODE
    if render:
        experiment_config.RENDER_MODE = 'human'
    else:
        experiment_config.RENDER_MODE = None

    # Initialize environment and model wrappers
    # We re-initialize the models/env to ensure a clean evaluation state
    _, _, env = initialize_models(experiment_config, env_config)
    
    total_reward = 0.0
    total_collisions = 0
    total_episodes = 0
    
    # Save original model training state and put in evaluation mode
    master_model.freeze()
    if hasattr(agent_model, 'policy') and hasattr(agent_model.policy, 'eval'):
        agent_model.policy.eval()

    print(f"\n--- Starting Standardized Evaluation ({len(seeds)} seeds x {episodes_per_seed} episodes) ---")

    for seed in seeds:
        for ep in range(episodes_per_seed):
            current_seed = seed + ep
            obs, info = env.reset(seed=current_seed)
            done, truncated = False, False
            episode_reward = 0.0
            crashed = False
            
            while not done and not truncated:
                # 1. Retrieve current global intersection state (without embedding)
                global_state = env.env.current_state
                
                # 2. Get master's latent embedding
                embedding, _, _ = master_model.get_proto_action(ensure_tensor(global_state))
                
                # 3. Predict agent actions deterministically using policy directly
                actions = []
                for car_obs in obs:
                    # obs elements are [car_state || embedding]
                    car_action, _ = agent_model.predict(car_obs, deterministic=True)
                    actions.append(car_action)
                
                # 4. Format actions into environment-compatible tuple
                cars_scalar_action = []
                for action in actions:
                    car_scalar, _ = get_scaler_action_and_action_array(action)
                    cars_scalar_action.append(car_scalar)
                
                # 5. Step the environment
                obs, reward, done, truncated, info = env.step(tuple(cars_scalar_action))
                episode_reward += reward
                
                if done and info.get("crashed", False):
                    crashed = True
            
            total_reward += episode_reward
            if crashed:
                total_collisions += 1
            total_episodes += 1
            
            status = "COLLISION" if crashed else "SUCCESS"
            print(f"Seed {seed} | Episode {ep+1} | Reward: {episode_reward:.2f} | Status: {status}")

    # Restore original render mode
    experiment_config.RENDER_MODE = original_render_mode
    
    mean_reward = total_reward / total_episodes
    success_rate = (total_episodes - total_collisions) / total_episodes
    mean_collisions = total_collisions / total_episodes
    
    print(f"\nEvaluation Results:")
    print(f"  Success Rate:     {success_rate * 100.0:.1f}%")
    print(f"  Mean Reward:      {mean_reward:.2f}")
    print(f"  Mean Collisions:  {mean_collisions:.2f}")
    print("--------------------------------------------------------------------\n")
    
    # Close environment cleanly
    env.close()
    
    return success_rate, mean_reward, mean_collisions

def load_results_history(file_path=RESULTS_FILE):
    """Read the results.tsv file and return history as a list of dicts."""
    if not os.path.exists(file_path):
        return []
    
    history = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            history.append({
                "iteration": int(row["iteration"]),
                "success_rate": float(row["success_rate"]),
                "mean_reward": float(row["mean_reward"]),
                "mean_collisions": float(row["mean_collisions"]),
                "timestamp": row["timestamp"],
                "description": row["description"]
            })
    return history

def load_best_score(file_path=RESULTS_FILE):
    """Scan results.tsv and return the best metrics found so far (success_rate, mean_reward)."""
    history = load_results_history(file_path)
    if not history:
        return 0.0, -999999.0
    
    # Find best iteration based primarily on Success Rate, breaking ties with Reward
    best_run = max(history, key=lambda x: (x["success_rate"], x["mean_reward"]))
    return best_run["success_rate"], best_run["mean_reward"]

def save_results_tsv(iteration, success_rate, mean_reward, mean_collisions, description, file_path=RESULTS_FILE):
    """Append a new experiment entry to results.tsv."""
    file_exists = os.path.exists(file_path)
    
    fieldnames = ["iteration", "success_rate", "mean_reward", "mean_collisions", "timestamp", "description"]
    
    with open(file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "iteration": iteration,
            "success_rate": round(success_rate, 4),
            "mean_reward": round(mean_reward, 2),
            "mean_collisions": round(mean_collisions, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description
        })
    print(f"Saved results to {file_path} for iteration {iteration}.")
