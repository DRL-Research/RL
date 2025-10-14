import copy
import logging
import os
from itertools import product

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from src.model.model_handler import load_models, save_models
from src.plotting_utils.plotting_utils import plot_training_results
from src.project_globals import rollout_buffers
from src.training.episode_utils import process_episode
from src.training.general_utils import (
    setup_experiment_dirs,
    initialize_models,
    setup_loggers,
    close_everything, ensure_tensor, )
from src.training.training_loop_utils import init_training_results, prepare_models_for_cycle, perform_training_phase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_param_value_for_label(value):
    if isinstance(value, (list, tuple)):
        formatted = "-".join(_format_param_value_for_label(v) for v in value)
    elif isinstance(value, float):
        formatted = f"{value:.6g}"
    else:
        formatted = str(value)

    formatted = formatted.replace(".", "p").replace("-", "m")
    formatted = formatted.replace(" ", "")
    formatted = formatted.replace("[", "").replace("]", "")
    return formatted


def _build_param_label(params):
    if not params:
        return "default"

    parts = []
    for key in sorted(params):
        value_label = _format_param_value_for_label(params[key])
        parts.append(f"{key}-{value_label}")
    return "_".join(parts)


##########################################
# Training Loop
##########################################

def training_loop(experiment, env, agent_model, master_model):
    """
    Main training loop that orchestrates cycles and episodes.
    """

    # Add rollout buffer per controlled car
    for _ in env.env.config["controlled_cars"]:
        new_rollout_buffer_instance = RolloutBuffer(
            buffer_size=experiment.N_STEPS,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(experiment.STATE_INPUT_SIZE,)),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=1
        )
        rollout_buffers.append(new_rollout_buffer_instance)

    collision_counter, episode_counter, total_steps = 0, 0, 0

    results = init_training_results()

    for cycle_num in range(1, experiment.CYCLES + 1):
        print('Cycle', cycle_num, 'out of ', experiment.CYCLES)
        train_both, training_master, training_agent = prepare_models_for_cycle(cycle_num, experiment.CYCLES,
                                                                               master_model, agent_model)
        for _ in range(experiment.EPISODES_PER_CYCLE):

            episode_counter += 1
            print()
            print("*" * 40)
            print('Episode ', episode_counter, '/', experiment.EPISODES_PER_CYCLE * experiment.CYCLES, 'episodes')
            print("*" * 40)
            print()
            episode_rewards, actions, steps, crashed = process_episode(episode_counter, total_steps, env, master_model,
                                                                       agent_model, experiment, train_both,
                                                                       training_master)
            if crashed:
                collision_counter += 1

            total_steps += steps
            results["episode_rewards"].append(episode_rewards)
            results["all_actions"].append(actions)

            # Prepare state for training
            with torch.no_grad():
                _, _ = env.reset()
                full_state = env.env.current_state
                state_tensor = ensure_tensor(full_state)

            if episode_counter % experiment.EPISODE_AMOUNT_FOR_TRAIN == 0:
                perform_training_phase(train_both, training_master, training_agent, master_model, agent_model,
                                       full_state,
                                       state_tensor, results)

    print("Training completed.")
    return agent_model, master_model, collision_counter, results["episode_rewards"], results["all_actions"], results


#
# def training_loop(experiment, env, agent_model, master_model):
#     """
#     Main training loop that orchestrates cycles and episodes.
#     """
#
#     collision_counter, episode_counter, total_steps = 0, 0, 0
#
#     results = init_training_results()
#
#     for cycle_num in range(1, experiment.CYCLES + 1):
#         print('Cycle', cycle_num,'out of ', experiment.CYCLES)
#         train_both, training_master, training_agent = prepare_models_for_cycle(cycle_num, experiment.CYCLES,
#                                                                                master_model, agent_model)
#         for _ in range(experiment.EPISODES_PER_CYCLE):
#
#             episode_counter += 1
#             print('This is the ',episode_counter,'Out of',experiment.EPISODES_PER_CYCLE* experiment.CYCLES, 'episodes')
#             episode_rewards, actions, steps, crashed = process_episode(episode_counter, total_steps, env, master_model,
#                                                               agent_model, experiment, train_both, training_master)
#             if crashed:
#                 collision_counter += 1
#
#             total_steps += steps
#             results["episode_rewards"].append(episode_rewards)
#             results["all_actions"].append(actions)
#
#             # Prepare state for training
#             with torch.no_grad():
#                 _, _, _ = env.reset()
#                 full_state = env.env.current_state
#                 state_tensor = ensure_tensor(full_state)
#
#             if episode_counter % Experiment.EPISODE_AMOUNT_FOR_TRAIN == 0 :
#                 perform_training_phase(train_both, training_master, training_agent, master_model, agent_model, full_state,
#                                        state_tensor, results)
#
#     print("Training completed.")
#     return agent_model, master_model, collision_counter, results["episode_rewards"], results["all_actions"], results


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
    # log_training_results_to_neptune(experiment_config.logger, training_results)
    print("Training completed.")
    print("Total collisions:", collision_counter)
    # close_everything(wrapped_env, agent_logger, master_logger)
    return agent_model, master_model, collision_counter


##########################################
# Main experiment entrypoint
##########################################


def _run_single_experiment(experiment_config, env_config, master_params=None):
    experiment_config.SELECTED_MASTER_PARAMS = master_params

    print(
        f"Environment configuration: {len(env_config['controlled_cars'])} controlled cars, {len(env_config['static_cars'])} static cars"
    )

    setup_experiment_dirs(experiment_config.EXPERIMENT_PATH)
    master_model, agent_model, wrapped_env = initialize_models(
        experiment_config, env_config, master_hyperparams=master_params
    )
    agent_logger, master_logger = setup_loggers(experiment_config.EXPERIMENT_PATH)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    if experiment_config.ONLY_INFERENCE:
        print("Running in inference-only mode")
        result = run_inference_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )
    else:
        print("Running in training mode")
        result = run_training_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )

    collision_counter = result[2] if len(result) >= 3 else None
    if collision_counter is not None:
        params_display = (
            experiment_config.SELECTED_MASTER_PARAMS
            if experiment_config.SELECTED_MASTER_PARAMS is not None
            else "default"
        )
        print(
            f"[RunSummary] Selected master params: {params_display} -> crashes: {collision_counter}"
        )

    return result


def run_experiment(experiment_config, env_config):
    param_grid = getattr(experiment_config, "MASTER_PARAM_GRID", None)

    if param_grid:
        grid_keys = [key for key, values in param_grid.items() if values]
        if not grid_keys:
            experiment_config.GRID_SEARCH_LABEL = None
            return _run_single_experiment(experiment_config, env_config)

        combinations = list(product(*(param_grid[key] for key in grid_keys)))
        print(f"[GridSearch] Running {len(combinations)} master hyperparameter combinations")

        base_path = experiment_config.EXPERIMENT_PATH
        setup_experiment_dirs(base_path)

        results_summary = []
        for combo in combinations:
            params = dict(zip(grid_keys, combo))
            label = _build_param_label(params)
            run_config = copy.deepcopy(experiment_config)
            run_config.SAVE_MODEL_DIRECTORY = os.path.join(run_config.EXPERIMENT_PATH, "trained_model")
            run_config.GRID_SEARCH_LABEL = label
            run_config.MASTER_PARAM_GRID = None

            print(f"[GridSearch] Starting combination {label}: {params}")
            result = _run_single_experiment(run_config, env_config, params)
            collision_counter = result[2] if len(result) >= 3 else None
            results_summary.append({
                "label": label,
                "params": params,
                "crashes": collision_counter,
            })

        return results_summary

    experiment_config.GRID_SEARCH_LABEL = None
    return _run_single_experiment(experiment_config, env_config, None)
