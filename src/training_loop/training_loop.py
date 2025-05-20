import traceback

import numpy as np
import torch

from logger.utils import log_training_results_to_neptune
from src.model.agent_handler import Agent
from src.model.model_handler import load_models, save_models
from src.plotting_utils.plotting_utils import plot_training_results
from src.training_loop.utils import (
    setup_experiment_dirs,
    initialize_models,
    setup_loggers,
    close_everything, ensure_tensor, flatten_obs, combine_agent_obs, get_action_array,
)


##########################################
# Training routines for models
##########################################

def train_master_and_reset_buffer(master_model, full_obs):
    """Trains the master model and resets its buffer - Returns loss values"""
    policy_loss_val = value_loss_val = total_loss_val = None
    try:
        with torch.no_grad():
            last_master_tensor = ensure_tensor(full_obs)
            last_value = master_model.model.policy.predict_values(last_master_tensor)
        if master_model.rollout_buffer.pos == 0:
            print("Master model: No data to train on")
            return None
        master_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
        orig_get = master_model.rollout_buffer.get

        def modified_get(batch_size):
            if not master_model.rollout_buffer.full:
                orig_full = master_model.rollout_buffer.full
                master_model.rollout_buffer.full = True
                try:
                    indices = np.arange(master_model.rollout_buffer.pos)
                    if len(indices) > 0:
                        return master_model.rollout_buffer._get_samples(indices)
                    else:
                        return None
                finally:
                    master_model.rollout_buffer.full = orig_full
            else:
                return next(orig_get(batch_size))

        try:
            master_model.rollout_buffer.get = modified_get
            rollout_data = master_model.rollout_buffer.get(batch_size=None)
            if rollout_data is not None:
                steps_trained = len(rollout_data.observations)
                print(f"Master model trained on {steps_trained} steps")
                observations_tensor = torch.FloatTensor(rollout_data.observations)
                actions_tensor = torch.FloatTensor(rollout_data.actions)
                policy = master_model.model.policy
                optimizer = policy.optimizer
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()
                policy_loss = -log_probs.mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()
                print(
                    f"Master loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
            else:
                print("Master model: No valid data for training")
        finally:
            master_model.rollout_buffer.get = orig_get
    except Exception as e:
        print(f"Error during master training: {str(e)}")
        traceback.print_exc()
    master_model.rollout_buffer.reset()
    print("Master buffer reset")
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


def train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor):
    """Trains the agent model and resets its buffer - Returns loss values"""
    policy_loss_val = value_loss_val = total_loss_val = None
    try:
        with torch.no_grad():
            shaped_tensor = ensure_tensor(last_master_tensor)
            embedding = master_model.get_proto_action(shaped_tensor)
            car_state = flatten_obs(last_master_tensor)
            expected_dim = agent_model.policy.observation_space.shape[0]
            agent_obs = combine_agent_obs(car_state, embedding, expected_dim)
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
            last_value = agent_model.policy.predict_values(agent_obs_tensor)
        if agent_model.rollout_buffer.pos == 0:
            print("Agent model: No data to train on")
            return None
        agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
        orig_get = agent_model.rollout_buffer.get

        def modified_get(batch_size):
            if not agent_model.rollout_buffer.full:
                orig_full = agent_model.rollout_buffer.full
                agent_model.rollout_buffer.full = True
                try:
                    indices = np.arange(agent_model.rollout_buffer.pos)
                    if len(indices) > 0:
                        return agent_model.rollout_buffer._get_samples(indices)
                    else:
                        return None
                finally:
                    agent_model.rollout_buffer.full = orig_full
            else:
                return next(orig_get(batch_size))

        try:
            agent_model.rollout_buffer.get = modified_get
            rollout_data = agent_model.rollout_buffer.get(batch_size=None)
            if rollout_data is not None:
                steps_trained = len(rollout_data.observations)
                print(f"Agent model trained on {steps_trained} steps")
                observations_tensor = torch.FloatTensor(rollout_data.observations)
                actions_tensor = torch.FloatTensor(rollout_data.actions)
                policy = agent_model.policy
                optimizer = policy.optimizer
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()
                advantages_tensor = torch.FloatTensor(rollout_data.advantages)
                policy_loss = -(log_probs * advantages_tensor).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()
                print(
                    f"Agent loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
            else:
                print("Agent model: No valid data for training")
        finally:
            agent_model.rollout_buffer.get = orig_get
    except Exception as e:
        print(f"Error during agent training: {str(e)}")
        traceback.print_exc()
    agent_model.rollout_buffer.reset()
    print("Agent buffer reset")
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


##########################################
# Episode Runner
##########################################

def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    all_rewards, actions_per_episode = [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    crashed = False
    current_obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        steps_counter += 1
        full_obs = env.env.current_state
        master_input = ensure_tensor(full_obs)
        if train_both or training_master:
            with torch.no_grad():
                proto_action, value, log_prob = master_model.model.policy.forward(master_input)
                embedding = proto_action.cpu().numpy()[0]
        else:
            embedding = master_model.get_proto_action(master_input)
        car1_state = flatten_obs(full_obs)
        agent_obs = combine_agent_obs(car1_state, embedding, agent_model.policy.observation_space.shape[0])
        agent_action = Agent.get_action(agent_model, current_obs, total_steps,
                                        experiment.EXPLORATION_EXPLOITATION_THRESHOLD)
        scalar_action = agent_action[0] if (hasattr(agent_action, 'shape') and len(agent_action.shape) > 0) else \
            (agent_action[0] if isinstance(agent_action, list) and len(agent_action) > 0 else agent_action)
        action_array = get_action_array(agent_action)

        if train_both or not training_master:
            with torch.no_grad():
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                agent_value = agent_model.policy.predict_values(obs_tensor)
                dist = agent_model.policy.get_distribution(obs_tensor)
                log_prob_agent = dist.log_prob(torch.tensor(action_array, dtype=torch.long))

        env.render()
        next_obs, reward, done, truncated, info = env.step(scalar_action)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        actions_per_episode.append(scalar_action)

        if done and info.get("crashed", False):
            crashed = True

        episode_start = (steps_counter == 1)
        if train_both:
            full_obs_flat = full_obs.reshape(-1) if isinstance(full_obs, np.ndarray) and len(
                full_obs.shape) == 2 else full_obs
            agent_model.rollout_buffer.add(agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent)
            master_model.rollout_buffer.add(full_obs_flat, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            full_obs_flat = full_obs.reshape(-1) if isinstance(full_obs, np.ndarray) and len(
                full_obs.shape) == 2 else full_obs
            master_model.rollout_buffer.add(full_obs_flat, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent)

        current_obs = next_obs

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


##########################################
# Training Loop
##########################################

def training_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    master_value_losses, master_policy_losses, master_total_losses = [], [], []
    agent_value_losses, agent_policy_losses, agent_total_losses = [], [], []
    episode_rewards = []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")
        train_both = (cycle == 0)
        training_master = (not train_both) and (cycle % 2 == 1)
        training_agent = (not train_both) and (cycle % 2 == 0)

        if train_both or training_master:
            master_model.unfreeze()
        else:
            master_model.freeze()
        agent_model.policy.set_training_mode(train_both or training_agent)

        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} @")
            master_model.rollout_buffer.reset()
            agent_model.rollout_buffer.reset()

            episode_sum_of_rewards, actions_per_episode, steps_counter, crashed = run_episode(
                experiment, total_steps, env, master_model, agent_model,
                train_both=train_both, training_master=training_master
            )
            if crashed:
                collision_counter += 1
                print('Episode result: Collision')
            else:
                print('Episode result: Success')

            total_steps += steps_counter
            all_rewards.append(episode_sum_of_rewards)
            all_actions.append(actions_per_episode)
            episode_rewards.append(episode_sum_of_rewards)
            print(f"Episode summary - Reward: {episode_sum_of_rewards}, Steps: {steps_counter}")
            print(
                f"Buffer content - Master: {master_model.rollout_buffer.pos} steps, Agent: {agent_model.rollout_buffer.pos} steps")

            # Prepare tensors for last value predictions
            with torch.no_grad():
                current_obs, _ = env.reset()
                full_state = env.env.current_state
                last_master_tensor = ensure_tensor(full_state)

            # Train based on cycle
            if train_both:
                print("Training both networks...")
                master_losses = train_master_and_reset_buffer(master_model, full_state)
                if master_losses:
                    master_policy_losses.append(master_losses[0])
                    master_value_losses.append(master_losses[1])
                    master_total_losses.append(master_losses[2])
                agent_losses = train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor)
                if agent_losses:
                    agent_policy_losses.append(agent_losses[0])
                    agent_value_losses.append(agent_losses[1])
                    agent_total_losses.append(agent_losses[2])
            elif training_master:
                print("Training master only...")
                master_losses = train_master_and_reset_buffer(master_model, full_state)
                if master_losses:
                    master_policy_losses.append(master_losses[0])
                    master_value_losses.append(master_losses[1])
                    master_total_losses.append(master_losses[2])
                agent_policy_losses.append(None)
                agent_value_losses.append(None)
                agent_total_losses.append(None)
            elif training_agent:
                print("Training agent only...")
                agent_losses = train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor)
                if agent_losses:
                    agent_policy_losses.append(agent_losses[0])
                    agent_value_losses.append(agent_losses[1])
                    agent_total_losses.append(agent_losses[2])
                master_policy_losses.append(None)
                master_value_losses.append(None)
                master_total_losses.append(None)

    training_results = {
        "episode_rewards": episode_rewards,
        "master_policy_losses": master_policy_losses,
        "master_value_losses": master_value_losses,
        "master_total_losses": master_total_losses,
        "agent_policy_losses": agent_policy_losses,
        "agent_value_losses": agent_value_losses,
        "agent_total_losses": agent_total_losses,
        "all_actions": all_actions
    }
    print("Training completed.")

    return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions, training_results


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
    print("Running full training")
    p_agent_loss, p_master_loss, p_episode_counter = [], [], []
    # Training
    p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions, training_results = \
        training_loop(p_agent_loss, p_master_loss, p_episode_counter,
                      experiment_config, wrapped_env, agent_model, master_model)
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
        return run_training_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )
