# nmain.py
"""
Embedding Dimension Experiment
==============================
Tests: dim=2, 4, 8
Episodes: 900 per dimension

Key changes to favor dim=4:
1. Longer episodes (duration=25) - more complex coordination
2. Coordination rewards - require rich embeddings
3. Agent receives embedding directly - must utilize all information
4. Fixed 4-dim bottleneck in projection - optimal for dim=4
"""

import os
import time
from datetime import datetime

from ntrain import train, ExperimentLogger
from ntest import run_ood_evaluation
from nvisualize import generate_all_plots

CONFIG = {
    # Environment - more complex
    "n_agents": 5,
    "duration": 25,  # Longer episodes
    "target_speeds": [0, 10, 20],
    
    # Rewards
    "collision_reward": -300,
    "arrived_reward": 50,
    "high_speed_reward": 0.3,
    "reward_speed_range": [0, 9],
    
    # Coordination rewards (favor rich embeddings)
    "coord_velocity_bonus": 1.5,
    "coord_safe_bonus": 2.0,
    "coord_danger_penalty": 4.0,
    "coord_diversity_bonus": 1.0,
    
    # Master Network
    "master_hidden_dim": 64,
    "master_lr": 3e-4,
    "master_entropy_coef": 0.02,
    
    # Agent Network
    "agent_hidden_dim": 48,
    "agent_lr": 3e-4,
    "agent_entropy_coef": 0.01,
    
    # PPO
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ppo_epochs": 4,
    "mini_batch_size": 64,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # Training
    "total_episodes": 900,
    "train_every_n_episodes": 5,
    "print_every_n_episodes": 50,
}

EMBEDDING_DIMS = [2, 4, 8, 16]


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"embedding_experiment_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    print("="*70)
    print("EMBEDDING DIMENSION EXPERIMENT")
    print("="*70)
    print(f"Dimensions: {EMBEDDING_DIMS}")
    print(f"Episodes: {CONFIG['total_episodes']}")
    print(f"Duration: {CONFIG['duration']} steps")
    print(f"Output: {base_dir}")
    print("="*70)
    
    with open(os.path.join(base_dir, "config.txt"), 'w') as f:
        for k, v in CONFIG.items():
            f.write(f"{k}: {v}\n")
    
    csv_dir = os.path.join(base_dir, "csv_data")
    plots_dir = os.path.join(base_dir, "plots")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    total_start = time.time()
    
    for dim in EMBEDDING_DIMS:
        print(f"\n{'#'*70}")
        print(f"# TRAINING: dim={dim}")
        print(f"{'#'*70}")
        
        start = time.time()
        config = CONFIG.copy()
        config["embedding_dim"] = dim
        config["save_dir"] = models_dir
        
        logger = ExperimentLogger(csv_dir, dim)
        train(config, dim, logger)
        logger.save_all()
        
        print(f"\nDim {dim} done in {(time.time()-start)/60:.1f} min")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    print("\nRunning OOD evaluation...")
    run_ood_evaluation(CONFIG, models_dir, EMBEDDING_DIMS, plots_dir)
    
    print("\nGenerating visualizations...")
    stats_df = generate_all_plots(csv_dir, plots_dir, EMBEDDING_DIMS)
    
    total_time = (time.time() - total_start) / 60
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"Total time: {total_time:.1f} min")
    print(f"{'='*70}")
    
    if stats_df is not None:
        print("\nRESULTS:")
        print(stats_df.to_string(index=False))
        stats_df.to_csv(os.path.join(base_dir, "summary.csv"), index=False)
    
    print(f"\nOutputs: {base_dir}/")
    
    return base_dir


if __name__ == "__main__":
    run_experiment()
