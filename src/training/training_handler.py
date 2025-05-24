import logging

import numpy as np
import torch

from logger.utils import log_training_results_to_neptune
from src.model.model_handler import load_models, save_models
from src.plotting_utils.plotting_utils import plot_training_results
from src.training.episode_utils import process_episode
from src.training.general_utils import (
    setup_experiment_dirs,
    initialize_models,
    setup_loggers,
    close_everything, ensure_tensor, )
from src.training.training_loop_utils import init_training_results, prepare_models_for_cycle, perform_training_phase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##########################################
# Training Loop
##########################################

def training_loop(experiment, env, agent_model, master_model):
    """
    Main training loop that orchestrates cycles and episodes.
    """
    collision_counter, episode_counter, total_steps = 0, 0, 0

    results = init_training_results()

    for cycle_num in range(1, experiment.CYCLES + 1):
        train_both, training_master, training_agent = prepare_models_for_cycle(cycle_num, experiment.CYCLES,
                                                                               master_model, agent_model)
        for _ in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            reward, actions, steps, crashed = process_episode(episode_counter, total_steps, env, master_model,
                                                              agent_model, experiment, train_both, training_master)
            if crashed:
                collision_counter += 1

            total_steps += steps
            results["episode_rewards"].append(reward)
            results["all_actions"].append(actions)

            # Prepare state for training
            with torch.no_grad():
                _, _ = env.reset()
                full_state = env.env.current_state
                state_tensor = ensure_tensor(full_state)

            perform_training_phase(train_both, training_master, training_agent, master_model, agent_model, full_state,
                                   state_tensor, results)

    print("Training completed.")
    return agent_model, master_model, collision_counter, results["episode_rewards"], results["all_actions"], results


##########################################
# Evaluation
##########################################

def run_evaluation(experiment_config, env, agent_model):
    """Run evaluation episodes in inference mode"""
    eval_rewards = []
    eval_actions = []
    eval_episodes = getattr(experiment_config, 'EVAL_EPISODES', 5)  # Default to 5

    for episode in range(eval_episodes):
        print(f"Evaluation episode {episode + 1}/{eval_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        actions = []

        while not done and not truncated:
            action, _ = agent_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            actions.append(action[0])
            env.render()

        eval_rewards.append(episode_reward)
        eval_actions.append(actions)
        print(f"Episode {episode + 1} reward: {episode_reward}")
        print(f"Success: {not info.get('crashed', False)}")

    print(f"Average evaluation reward: {np.mean(eval_rewards)}")
    return eval_rewards, eval_actions


##########################################
# Inference & Training Mode Handlers
##########################################

def run_inference_mode(experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger):
    """Run inference episodes only."""
    if experiment_config.LOAD_PREVIOUS_WEIGHT and experiment_config.LOAD_MODEL_DIRECTORY:
        loaded = load_models(agent_model, master_model, experiment_config.LOAD_MODEL_DIRECTORY)
        if loaded:
            print("Models will be trained from loaded weights!")
        else:
            print("Starting inference with untrained models.")
    else:
        print("Starting inference with untrained models (No previous weights loaded)")

    eval_rewards, eval_actions = run_evaluation(experiment_config, wrapped_env, agent_model)
    close_everything(wrapped_env, agent_logger, master_logger)
    return agent_model, master_model, 0


def run_training_mode(experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger):
    """Run the training loop and handle saving/logging."""
    agent_model, master_model, collision_counter, all_rewards, all_actions, training_results = (
        training_loop(experiment=experiment_config, env=wrapped_env, agent_model=agent_model,
                      master_model=master_model))
    save_models(agent_model, master_model, experiment_config.SAVE_MODEL_DIRECTORY)
    plot_training_results(experiment_config, training_results, show_plots=True)
    log_training_results_to_neptune(experiment_config.logger, training_results)
    print("Training completed.")
    print("Total collisions:", collision_counter)
    close_everything(wrapped_env, agent_logger, master_logger)
    return agent_model, master_model, collision_counter


##########################################
# Main experiment entrypoint
##########################################

def run_experiment(experiment_config, env_config):
    print(
        f"Environment configuration: {len(env_config['controlled_cars'])} controlled cars, {len(env_config['static_cars'])} static cars"
    )
    setup_experiment_dirs(experiment_config.EXPERIMENT_PATH)
    master_model, agent_model, wrapped_env = initialize_models(experiment_config, env_config)
    agent_logger, master_logger = setup_loggers(experiment_config.EXPERIMENT_PATH)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    if experiment_config.ONLY_INFERENCE:
        print("Running in inference-only mode")
        return run_inference_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )
    else:
        print("Running in training mode")
        return run_training_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )
