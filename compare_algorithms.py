import sys
import os
import time
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import run_experiment

# High-quality modern plotting palette
COLORS = {
    "MAPS": "#2563eb",         # Vibrant Blue
    "VN-MA-DDPG": "#8b5cf6",    # Deep Purple
    "MA-GA-DDPG": "#10b981"     # Emerald Green
}

COLORS_LIGHT = {
    "MAPS": "#93c5fd",
    "VN-MA-DDPG": "#c084fc",
    "MA-GA-DDPG": "#6ee7b7"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def compute_rolling_avg(data, window=10):
    if len(data) == 0:
        return data
    window = min(window, len(data))
    res = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        res[i] = np.mean(data[start:i+1])
    return res

def run_comparison():
    parser = argparse.ArgumentParser(description="Multi-Seed MARL Algorithm Comparison")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes per seed")
    parser.add_argument("--seeds", type=str, default="42,100,2026", help="Comma-separated seeds")
    parser.add_argument("--window", type=int, default=10, help="Rolling average window size")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    num_episodes = args.episodes

    print("==================================================")
    print("      Starting MARL Algorithm Comparison          ")
    print(f"      Seeds: {seeds} | Episodes: {num_episodes}   ")
    print("==================================================")

    patch_intersection_env()
    register_intersection_env()

    algorithms = {
        "MAPS": "experiment",
        "VN-MA-DDPG": "vn_maddpg",
        "MA-GA-DDPG": "ma_ga_ddpg"
    }

    # Structure to hold raw metrics per algorithm
    # { "MAPS": { "episode_rewards": [ [seed1_ep1...], [seed2_ep1...] ], ... } }
    results_raw = {
        alg_name: {
            "episode_rewards": [],
            "success_flags": [],
            "collision_flags": [],
            "episode_lengths": []
        } for alg_name in algorithms
    }

    env_config = sc.full_env_config_exp5

    for alg_name, alg_key in algorithms.items():
        print(f"\nEvaluating Algorithm: {alg_name}...")
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  -> Running Seed {seed} ({seed_idx + 1}/{len(seeds)})...")
            set_seed(seed)
            
            # Configure Experiment
            config = Experiment(
                ALGORITHM=alg_key,
                RENDER_MODE="rgb_array",
                EXPERIMENT_ID=f"Compare_{alg_name}_S{seed}",
                CYCLES=1,
                EPISODES_PER_CYCLE=num_episodes
            )
            config.LOAD_PREVIOUS_WEIGHT = False
            
            # Align paths so setup_experiment_dirs creates the correct folders
            config.EXPERIMENT_PATH = f"experiments/Compare_{alg_name}_S{seed}"
            config.SAVE_MODEL_DIRECTORY = f"{config.EXPERIMENT_PATH}/trained_model"
            
            # Run experiment
            _, history, _ = run_experiment(config, env_config)
            
            # Record histories
            results_raw[alg_name]["episode_rewards"].append(history["episode_rewards"])
            results_raw[alg_name]["success_flags"].append(history["success_flags"])
            results_raw[alg_name]["collision_flags"].append(history["collision_flags"])
            results_raw[alg_name]["episode_lengths"].append(history["episode_lengths"])

    # Aggregate stats across seeds (mean & std)
    # results_processed = { "MAPS": { "episode_rewards": { "mean": [...], "std": [...] } } }
    results_processed = {}
    for alg_name in algorithms:
        results_processed[alg_name] = {}
        for metric_name in ["episode_rewards", "success_flags", "collision_flags", "episode_lengths"]:
            data_matrix = np.array(results_raw[alg_name][metric_name], dtype=np.float32)
            
            # If success/collision flags, convert to percentages
            if metric_name in ["success_flags", "collision_flags"]:
                data_matrix = data_matrix * 100.0

            mean_vals = np.mean(data_matrix, axis=0)
            std_vals = np.std(data_matrix, axis=0)
            
            results_processed[alg_name][metric_name] = {
                "mean": mean_vals,
                "std": std_vals
            }

    print("\nTraining completed! Generating comparison plots...")
    generate_comparison_plots(results_processed, num_episodes, args.window)
    print("Comparison plots successfully saved to: plots/algorithm_comparison.png")

    print_final_summary_table(results_processed)

def generate_comparison_plots(results, num_episodes, window):
    os.makedirs("plots", exist_ok=True)
    
    # Enable style
    plt.style.use("ggplot")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cooperative MARL Algorithm Comparison (Multi-Seed Analysis)", fontsize=16, fontweight="bold", y=0.96)

    metrics = [
        ("success_flags", "Success Rate (%) - Moving Avg", "Success Rate (%)", axs[0, 0]),
        ("collision_flags", "Collision Rate (%) - Moving Avg", "Collision Rate (%)", axs[0, 1]),
        ("episode_rewards", "Reward per Episode - Moving Avg", "Average Reward", axs[1, 0]),
        ("episode_lengths", "Avg Travel Time (Steps) - Moving Avg", "Steps in Episode", axs[1, 1])
    ]

    episodes_x = np.arange(1, num_episodes + 1)

    for metric_key, title, ylabel, ax in metrics:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

        for alg_name, color in COLORS.items():
            mean_raw = results[alg_name][metric_key]["mean"]
            std_raw = results[alg_name][metric_key]["std"]
            
            # Smooth using rolling average
            mean_smooth = compute_rolling_avg(mean_raw, window)
            std_smooth = compute_rolling_avg(std_raw, window)

            ax.plot(episodes_x, mean_smooth, label=alg_name, color=color, linewidth=2)
            
            # Shaded variance (std deviation) band
            ax.fill_between(
                episodes_x,
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                color=color,
                alpha=0.15
            )

        ax.legend(loc="best", framealpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig("plots/algorithm_comparison.png", dpi=300)
    plt.close()

def print_final_summary_table(results):
    print("\n" + "="*60)
    print("                  FINAL RESULTS SUMMARY                   ")
    print("="*60)
    
    # We take the average of the last 10 episodes as the final converged performance metric
    summary_data = []
    
    for alg_name in results:
        final_reward_mean = np.mean(results[alg_name]["episode_rewards"]["mean"][-10:])
        final_reward_std = np.mean(results[alg_name]["episode_rewards"]["std"][-10:])
        
        final_success_mean = np.mean(results[alg_name]["success_flags"]["mean"][-10:])
        final_success_std = np.mean(results[alg_name]["success_flags"]["std"][-10:])
        
        final_collision_mean = np.mean(results[alg_name]["collision_flags"]["mean"][-10:])
        final_collision_std = np.mean(results[alg_name]["collision_flags"]["std"][-10:])
        
        final_steps_mean = np.mean(results[alg_name]["episode_lengths"]["mean"][-10:])
        final_steps_std = np.mean(results[alg_name]["episode_lengths"]["std"][-10:])
        
        summary_data.append({
            "alg": alg_name,
            "reward": f"{final_reward_mean:.2f} ± {final_reward_std:.2f}",
            "success": f"{final_success_mean:.1f}% ± {final_success_std:.1f}%",
            "collision": f"{final_collision_mean:.1f}% ± {final_collision_std:.1f}%",
            "steps": f"{final_steps_mean:.1f} ± {final_steps_std:.1f}"
        })

    # Output Markdown Table
    print("\n| Algorithm | Success Rate (%) | Collision Rate (%) | Reward per Episode | Travel Time (Steps) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for row in summary_data:
        print(f"| **{row['alg']}** | {row['success']} | {row['collision']} | {row['reward']} | {row['steps']} |")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    run_comparison()
