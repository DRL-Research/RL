# train.py (Mutable / AI Agent Workspace)
# This is the ONLY file you are permitted to edit to optimize the model.
# Point your agent here and let it run experiments autonomously!

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import AutoResearch test harness
import prepare

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import run_experiment

# =====================================================================
# 1. ARCHITECTURE OPTIMIZATION (AI Agent Sandbox)
# Redefine the Master Model's Feature Extractor here.
# You can modify layer dimensions, add normalization/dropout, change
# activation functions, or adjust the skip connections.
# =====================================================================

class SimpleResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Dimensions are determined by observation_space (usually 5 cars * 4 features = 20)
        input_dim = observation_space.shape[0]
        
        # Initial transformation layer
        self.input_layer = nn.Linear(input_dim, 128)
        
        # ResNet Blocks with skip connections
        self.res_block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Final output feature layer
        self.output_layer = nn.Linear(128, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        # Initial activation
        x = self.relu(self.input_layer(observations))
        
        # Block 1 with skip connection
        residual1 = x
        x = self.res_block1(x) + residual1
        x = self.relu(x)
        
        # Block 2 with skip connection
        residual2 = x
        x = self.res_block2(x) + residual2
        x = self.relu(x)
        
        return self.output_layer(x)

# Inject the custom extractor into the MasterModel module
# This monkey-patch ensures that the main training pipeline will utilize
# your optimized neural architecture when creating the Master PPO policy.
import src.model.master_model
src.model.master_model.SimpleResNetExtractor = SimpleResNetExtractor


# =====================================================================
# 2. HYPERPARAMETER & REWARD OPTIMIZATION
# Customize learning rate, episode settings, and environmental rewards here.
# =====================================================================

# Training Length Configuration
CYCLES = 1                 # Number of cycles
EPISODES_PER_CYCLE = 150   # Keep it short (e.g. 150) for fast 3-minute iteration loops

# Neural Model Training Parameters
LEARNING_RATE = 0.005      # Optimizer step size
N_STEPS = 64               # Steps collected per rollout buffer
BATCH_SIZE = 32            # Batch size for policy updates
EMBEDDING_SIZE = 4         # Latent embedding representation size

# Environment Reward Profiles
REACHED_TARGET_REWARD = 50   # Reward when a car safely exits the intersection
COLLISION_REWARD = -300      # Penalty when a collision occurs
STARVATION_REWARD = -5       # Penalty when cars move too slow
HIGH_SPEED_REWARD = 5        # Reward for navigating quickly and safely


# =====================================================================
# 3. EXPERIMENT LOOP
# =====================================================================

def main():
    print("==================================================")
    print("      Starting AutoResearch Experiment            ")
    print("==================================================")
    
    # 1. Define experiment configuration with our tunable parameters
    experiment_config = Experiment(
        ALGORITHM="experiment",
        RENDER_MODE=None,  # No rendering during automated training
        EXPERIMENT_ID="autoresearch_iter",
        CYCLES=CYCLES,
        EPISODES_PER_CYCLE=EPISODES_PER_CYCLE,
        LEARNING_RATE=LEARNING_RATE,
        N_STEPS=N_STEPS,
        BATCH_SIZE=BATCH_SIZE,
        EMBEDDING_SIZE=EMBEDDING_SIZE,
        REACHED_TARGET_REWARD=REACHED_TARGET_REWARD,
        COLLISION_REWARD=COLLISION_REWARD,
        STARVATION_REWARD=STARVATION_REWARD
    )
    experiment_config.HIGH_SPEED_REWARD = HIGH_SPEED_REWARD
    
    # Setup paths and configs
    experiment_config.LOAD_PREVIOUS_WEIGHT = False
    experiment_config.EXPERIMENT_PATH = "experiments/autoresearch_iter"
    experiment_config.SAVE_MODEL_DIRECTORY = f"{experiment_config.EXPERIMENT_PATH}/trained_model"
    env_config = sc.full_env_config_exp5

    # 2. Run the training cycle
    print(f"Training parameters: Cycles={CYCLES}, Episodes/Cycle={EPISODES_PER_CYCLE}, LR={LEARNING_RATE}")
    print("Starting model training...")
    
    models_tuple, training_results, collision_counter = run_experiment(experiment_config, env_config)
    agent_model, master_model = models_tuple
    
    print("Training finished! Initializing evaluation phase...")

    # 3. Run evaluation across fixed validation seeds (fully reproducible)
    success_rate, mean_reward, mean_collisions = prepare.evaluate_model(
        experiment_config=experiment_config,
        env_config=env_config,
        master_model=master_model,
        agent_model=agent_model,
        episodes_per_seed=3,
        render=False
    )
    
    # 4. Read best score from results history to check for improvement
    best_success, best_reward = prepare.load_best_score()
    print(f"Current Best Success Rate: {best_success * 100.0:.1f}% | Best Mean Reward: {best_reward:.2f}")
    print(f"This Run Success Rate:     {success_rate * 100.0:.1f}% | This Run Mean Reward: {mean_reward:.2f}")
    
    is_improvement = (success_rate > best_success) or (success_rate == best_success and mean_reward > best_reward)
    
    # Find next iteration index
    history = prepare.load_results_history()
    next_iteration = len(history) + 1
    
    description = f"LR={LEARNING_RATE}, Batch={BATCH_SIZE}, RewardCoefs(Target={REACHED_TARGET_REWARD}, Crash={COLLISION_REWARD})"
    
    if is_improvement:
        print("\n[SUCCESS] This iteration improved the metrics!")
        print("Saving new best parameters...")
    else:
        print("\n[NO IMPROVEMENT] No improvement compared to best. Discard this configuration.")
        
    # Append results to the results.tsv file
    prepare.save_results_tsv(
        iteration=next_iteration,
        success_rate=success_rate,
        mean_reward=mean_reward,
        mean_collisions=mean_collisions,
        description=description
    )
    
    # Return exit code based on whether it improved the model
    # Exit code 0 indicates improvement (keep changes), 1 indicates no improvement (revert)
    sys.exit(0 if is_improvement else 1)

if __name__ == "__main__":
    main()
