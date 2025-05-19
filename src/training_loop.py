import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.model.agent_handler import Agent, DummyVecEnv
from src.model.master_model import MasterModel
from src.model.model_handler import Model


def run_experiment(experiment_config, env_config):
    """
    Main function to run a complete experiment with master-agent architecture
    """
    print(
        f"Environment configuration: {len(env_config['controlled_cars'])} controlled cars, {len(env_config['static_cars'])} static cars")

    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_config.EXPERIMENT_PATH, exist_ok=True)

    # Initialize tracking variables
    p_agent_loss, p_master_loss, p_episode_counter = [], [], []

    # Attach environment configuration to experiment_config for easy access
    experiment_config.CONFIG = env_config

    # Create master model
    master_model = MasterModel(
        embedding_size=experiment_config.EMBEDDING_SIZE,
        experiment=experiment_config
    )

    # Create environment wrapper with Agent handler
    env_fn = lambda: Agent(experiment_config, master_model=master_model)
    wrapped_env = DummyVecEnv([env_fn])

    # Create agent model
    agent_model = Model(wrapped_env, experiment_config).model

    # Configure loggers
    base_path = experiment_config.EXPERIMENT_PATH
    agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
    master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    # Inference or training mode
    if experiment_config.ONLY_INFERENCE:
        print("Running in inference-only mode")
        if experiment_config.LOAD_PREVIOUS_WEIGHT and experiment_config.LOAD_MODEL_DIRECTORY:
            try:
                agent_model.load(f"{experiment_config.LOAD_MODEL_DIRECTORY}_agent.pth")
                master_model.load(f"{experiment_config.LOAD_MODEL_DIRECTORY}_master.pth")
                print(
                    f"Loaded weights from {experiment_config.LOAD_MODEL_DIRECTORY}, models will be trained from this point!")
            except Exception as e:
                print(f"Failed to load weights: {e}")
                print("Starting training from scratch")
        else:
            print("Starting training from scratch (No previous weights loaded)")
        # Load pre-trained models only in inference mode
        try:
            agent_model.load(f"{experiment_config.LOAD_MODEL_DIRECTORY}_agent.pth")
            master_model.load(f"{experiment_config.LOAD_MODEL_DIRECTORY}_master.pth")
            print(f"Loaded weights from {experiment_config.LOAD_MODEL_DIRECTORY} for inference.")
        except Exception as e:
            print(f"Failed to load inference models: {e}")
            print("Using untrained models for inference.")

        # Run evaluation episodes
        eval_rewards, eval_actions = run_evaluation(experiment_config, wrapped_env, agent_model)
        wrapped_env.close()
        agent_logger.close()
        master_logger.close()

        return agent_model, master_model, 0

    # Regular training run
    print("Running full training")

    # Run the training loop
    p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions, training_results = \
        training_loop(p_agent_loss, p_master_loss, p_episode_counter,
                      experiment_config, wrapped_env, agent_model, master_model)

    # Save models
    agent_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_agent.pth")
    master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")
    print("Models saved")

    # Close loggers
    agent_logger.close()
    master_logger.close()

    plot_training_results(experiment_config, training_results, show_plots=True)
    log_training_results_to_neptune(experiment_config.logger, training_results)

    print("Training completed.")
    print("Total collisions:", collision_counter)

    # Close environment
    wrapped_env.close()

    return agent_model, master_model, collision_counter


def run_evaluation(experiment_config, env, agent_model):
    """Run evaluation episodes in inference mode"""
    eval_rewards = []
    eval_actions = []

    eval_episodes = getattr(experiment_config, 'EVAL_EPISODES', 5)  # Default to 5 episodes if not specified

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


def training_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model):
    """
    Main training loop
    """
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    master_value_losses = []
    master_policy_losses = []
    master_total_losses = []

    agent_value_losses = []
    agent_policy_losses = []
    agent_total_losses = []

    episode_rewards = []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")

        # Set training mode based on cycle
        train_both = (cycle == 0)
        training_master = (not train_both) and (cycle % 2 == 1)
        training_agent = (not train_both) and (cycle % 2 == 0)

        # Set models' training states
        if train_both or training_master:
            master_model.unfreeze()
        else:
            master_model.freeze()

        agent_model.policy.set_training_mode(train_both or training_agent)

        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} @")

            # Reset buffers at start of episode
            master_model.rollout_buffer.reset()
            agent_model.rollout_buffer.reset()

            # Run episode and collect data
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

            # Train on episode data
            with torch.no_grad():
                current_obs, _ = env.reset()
                full_state = env.env.current_state
                last_master_tensor = torch.tensor(full_state, dtype=torch.float32).unsqueeze(0)

            # Train based on cycle
            if train_both:
                print("Training both networks...")
                master_losses = train_master_and_reset_buffer(master_model, full_state)
                # 0,1,2 are value,policy adn total loss
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

                # Add None for agent losses to maintain alignment
                agent_policy_losses.append(None)
                agent_value_losses.append(None)
                agent_total_losses.append(None)

            elif training_agent:
                print("Training agent only...")
                agent_losses = train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor)
                if agent_losses:
                    agent_policy_losses.append(agent_losses[0])
                    agent_value_losses.append(agent_losses[1])
                    agent_total_losses.append(agent_losses[2])

                # Add None for master losses to maintain alignment
                master_policy_losses.append(None)
                master_value_losses.append(None)
                master_total_losses.append(None)

    # save results as dict
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


def log_training_results_to_neptune(logger, training_results):
    """
    Logs training results (reward/losses per episode) to Neptune.
    """
    episode_rewards = training_results["episode_rewards"]
    master_policy_losses = training_results["master_policy_losses"]
    master_value_losses = training_results["master_value_losses"]
    master_total_losses = training_results["master_total_losses"]
    agent_policy_losses = training_results["agent_policy_losses"]
    agent_value_losses = training_results["agent_value_losses"]
    agent_total_losses = training_results["agent_total_losses"]

    for i in range(len(episode_rewards)):
        logger.log_metric("episode/reward", episode_rewards[i])
        if master_policy_losses[i] is not None:
            logger.log_metric("master/policy_loss", master_policy_losses[i])
            logger.log_metric("master/value_loss", master_value_losses[i])
            logger.log_metric("master/total_loss", master_total_losses[i])
        if agent_policy_losses[i] is not None:
            logger.log_metric("agent/policy_loss", agent_policy_losses[i])
            logger.log_metric("agent/value_loss", agent_value_losses[i])
            logger.log_metric("agent/total_loss", agent_total_losses[i])


def monitor_episode_results(master_model, agent_model):
    """Prints minimal statistics about the rollout buffers"""
    master_buffer_size = master_model.rollout_buffer.pos
    agent_buffer_size = agent_model.rollout_buffer.pos

    print(f"Current buffer sizes - Master: {master_buffer_size}, Agent: {agent_buffer_size}")

    if master_buffer_size > 0 and hasattr(master_model.rollout_buffer, 'rewards'):
        rewards = master_model.rollout_buffer.rewards[:master_buffer_size]
        print(f"Episode rewards - Min: {rewards.min():.2f}, Max: {rewards.max():.2f}, Avg: {rewards.mean():.2f}")


def plot_training_results(experiment, results, show_plots=True):
    """
    Generate improved training visualization with loss curves
    Args:
        experiment: the experiment configuration
        results: dictionary with training results
        show_plots: if True, displays the plots interactively
    """
    print("Generating detailed training plots...")

    # Create directory for saving plots
    plots_dir = os.path.join(experiment.EXPERIMENT_PATH, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Unpack the results
    episode_rewards = results["episode_rewards"]
    master_policy_losses = results["master_policy_losses"]
    master_value_losses = results["master_value_losses"]
    master_total_losses = results["master_total_losses"]
    agent_policy_losses = results["agent_policy_losses"]
    agent_value_losses = results["agent_value_losses"]
    agent_total_losses = results["agent_total_losses"]

    # Create x-axis for episodes
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Set up the figure style
    plt.style.use('ggplot')

    # 1. Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.title('Episode Rewards', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2. Plot Master value loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(master_value_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [master_value_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#D95F02', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Master Value Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'master_value_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 3. Plot Master total loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(master_total_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [master_total_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#7570B3', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Master Total Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'master_total_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 4. Plot Agent value loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(agent_value_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [agent_value_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#1B9E77', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Agent Value Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'agent_value_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 5. Plot Agent total loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(agent_total_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [agent_total_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#E7298A', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Agent Total Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'agent_total_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    plt.figure(figsize=(12, 8))

    # Master losses
    master_valid_indices = [i for i, val in enumerate(master_total_losses) if val is not None]
    master_valid_episodes = [episodes[i] for i in master_valid_indices]
    master_valid_total_losses = [master_total_losses[i] for i in master_valid_indices]
    master_valid_value_losses = [master_value_losses[i] for i in master_valid_indices]

    # Agent losses
    agent_valid_indices = [i for i, val in enumerate(agent_total_losses) if val is not None]
    agent_valid_episodes = [episodes[i] for i in agent_valid_indices]
    agent_valid_total_losses = [agent_total_losses[i] for i in agent_valid_indices]
    agent_valid_value_losses = [agent_value_losses[i] for i in agent_valid_indices]

    if master_valid_total_losses:
        plt.plot(master_valid_episodes, master_valid_total_losses, 'o-', label='Master Total Loss',
                 color='#7570B3', linewidth=2, markersize=6)
        plt.plot(master_valid_episodes, master_valid_value_losses, 's-', label='Master Value Loss',
                 color='#D95F02', linewidth=2, markersize=6)

    if agent_valid_total_losses:
        plt.plot(agent_valid_episodes, agent_valid_total_losses, 'o-', label='Agent Total Loss',
                 color='#E7298A', linewidth=2, markersize=6)
        plt.plot(agent_valid_episodes, agent_valid_value_losses, 's-', label='Agent Value Loss',
                 color='#1B9E77', linewidth=2, markersize=6)

    plt.grid(True, alpha=0.3)
    plt.title('Training Losses', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'combined_losses.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 7. Plot combined training metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot rewards
    ax1.plot(episodes, episode_rewards, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('Episode Rewards', fontsize=16)
    ax1.set_ylabel('Reward', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot losses on second axis
    if master_valid_total_losses:
        ax2.plot(master_valid_episodes, master_valid_total_losses, 'o-', label='Master Loss',
                 color='#7570B3', linewidth=2, markersize=6)

    if agent_valid_total_losses:
        ax2.plot(agent_valid_episodes, agent_valid_total_losses, 'o-', label='Agent Loss',
                 color='#E7298A', linewidth=2, markersize=6)

    ax2.set_title('Training Losses', fontsize=16)
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Loss (log scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_summary.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()
    if show_plots:
        print("Plots displayed. Close all plot windows to continue.")

    print(f"All plots saved to: {plots_dir}")


def train_master_and_reset_buffer(master_model, full_obs):
    """Trains the master model and resets its buffer - Returns loss values"""

    policy_loss_val = None
    value_loss_val = None
    total_loss_val = None

    try:
        # Prepare tensor for last value prediction
        with torch.no_grad():
            if isinstance(full_obs, np.ndarray):
                if len(full_obs.shape) == 3:
                    last_master_tensor = torch.tensor(full_obs.reshape(1, -1), dtype=torch.float32)
                elif len(full_obs.shape) == 2 and full_obs.shape[0] > 1:
                    last_master_tensor = torch.tensor(full_obs.reshape(1, -1), dtype=torch.float32)
                else:
                    last_master_tensor = torch.tensor(full_obs, dtype=torch.float32).unsqueeze(0)
            elif isinstance(full_obs, torch.Tensor):
                if len(full_obs.shape) == 3:
                    last_master_tensor = full_obs.reshape(1, -1)
                elif len(full_obs.shape) == 2 and full_obs.shape[0] > 1:  # שגיאה הייתה כאן
                    last_master_tensor = full_obs.reshape(1, -1)
                else:
                    last_master_tensor = full_obs.unsqueeze(0) if len(full_obs.shape) == 1 else full_obs
            else:
                try:
                    last_master_tensor = torch.tensor(full_obs, dtype=torch.float32).unsqueeze(0)
                except:
                    last_master_tensor = torch.zeros((1, master_model.observation_dim), dtype=torch.float32)

            # Get last value for returns computation
            last_value = master_model.model.policy.predict_values(last_master_tensor)

        # Skip training if buffer is empty
        if master_model.rollout_buffer.pos == 0:
            print("Master model: No data to train on")
            return None

        # Compute returns and advantage
        master_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))

        # Override the get method to allow partial buffer
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

                # Training process
                observations = rollout_data.observations
                actions = rollout_data.actions

                # Convert to tensors
                observations_tensor = torch.FloatTensor(observations)
                actions_tensor = torch.FloatTensor(actions)

                # Get policy and optimizer
                policy = master_model.model.policy
                optimizer = policy.optimizer

                # Forward pass
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)

                # Calculate value loss
                value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()

                # Calculate policy loss - FIXED to use a simple policy gradient approach
                policy_loss = -log_probs.mean()  # Simplified policy loss

                # Calculate entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                # Save loss values for plotting
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()

                print(
                    f"Master loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
            else:
                print("Master model: No valid data for training")
        finally:
            # Restore original method
            master_model.rollout_buffer.get = orig_get

    except Exception as e:
        print(f"Error during master training: {str(e)}")
        traceback.print_exc()

    # Reset buffer
    master_model.rollout_buffer.reset()
    print("Master buffer reset")

    # Return loss values if training occurred
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


def train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor):
    """Trains the agent model and resets its buffer - Returns loss values"""

    policy_loss_val = None
    value_loss_val = None
    total_loss_val = None

    try:
        with torch.no_grad():
            # Prepare tensor for embedding calculation
            if isinstance(last_master_tensor, np.ndarray):
                if len(last_master_tensor.shape) == 3:
                    shaped_tensor = torch.tensor(last_master_tensor.reshape(1, -1), dtype=torch.float32)
                elif len(last_master_tensor.shape) == 2 and last_master_tensor.shape[0] > 1:
                    shaped_tensor = torch.tensor(last_master_tensor.reshape(1, -1), dtype=torch.float32)
                else:
                    shaped_tensor = torch.tensor(last_master_tensor, dtype=torch.float32).unsqueeze(0)
            elif isinstance(last_master_tensor, torch.Tensor):
                if len(last_master_tensor.shape) == 3:
                    shaped_tensor = last_master_tensor.reshape(1, -1)
                elif len(last_master_tensor.shape) == 2 and last_master_tensor.shape[0] > 1:
                    shaped_tensor = last_master_tensor.reshape(1, -1)
                else:
                    shaped_tensor = last_master_tensor.unsqueeze(0) if len(
                        last_master_tensor.shape) == 1 else last_master_tensor

            # Get embedding
            embedding = master_model.get_proto_action(shaped_tensor)

            # Extract car1 state
            if isinstance(last_master_tensor, np.ndarray):
                if len(last_master_tensor.shape) == 3:
                    car_state = last_master_tensor[0, 0].flatten()
                elif len(last_master_tensor.shape) == 2 and last_master_tensor.shape[0] > 1:
                    car_state = last_master_tensor[0].flatten()
                else:
                    car_state = last_master_tensor[:4].flatten()
            elif isinstance(last_master_tensor, torch.Tensor):
                if len(last_master_tensor.shape) == 3:
                    car_state = last_master_tensor[0, 0].cpu().numpy().flatten()
                elif len(last_master_tensor.shape) == 2 and last_master_tensor.shape[0] > 1:
                    car_state = last_master_tensor[0].cpu().numpy().flatten()
                else:
                    car_state = last_master_tensor[:4].cpu().numpy().flatten()

            # Ensure 1D arrays
            car_state = np.array(car_state).flatten()
            embedding = np.array(embedding).flatten()

            # Create agent observation
            agent_obs = np.concatenate((car_state, embedding))
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)

            # Ensure dimensions match the network
            expected_dim = agent_model.policy.observation_space.shape[0]
            if agent_obs_tensor.shape[1] != expected_dim:
                if agent_obs_tensor.shape[1] > expected_dim:
                    agent_obs_tensor = agent_obs_tensor[:, :expected_dim]
                else:
                    padding = torch.zeros((1, expected_dim - agent_obs_tensor.shape[1]), dtype=torch.float32)
                    agent_obs_tensor = torch.cat([agent_obs_tensor, padding], dim=1)

            last_value = agent_model.policy.predict_values(agent_obs_tensor)

        # Skip training if buffer is empty
        if agent_model.rollout_buffer.pos == 0:
            print("Agent model: No data to train on")
            return None

        # Compute returns and advantage
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

                # Training process
                observations = rollout_data.observations
                actions = rollout_data.actions

                # Convert to tensors
                observations_tensor = torch.FloatTensor(observations)
                actions_tensor = torch.FloatTensor(actions)

                # Forward pass
                policy = agent_model.policy
                optimizer = policy.optimizer

                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)

                # Calculate value loss
                value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()

                # Calculate policy loss - Using advantages
                advantages = rollout_data.advantages
                advantages_tensor = torch.FloatTensor(advantages)
                policy_loss = -(log_probs * advantages_tensor).mean()

                # Calculate entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                # Save loss values for plotting
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()

                print(
                    f"Agent loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
            else:
                print("Agent model: No valid data for training")
        finally:
            # Restore original method
            agent_model.rollout_buffer.get = orig_get

    except Exception as e:
        print(f"Error during agent training: {str(e)}")
        traceback.print_exc()

    # Reset buffer
    agent_model.rollout_buffer.reset()
    print("Agent buffer reset")

    # Return loss values if training occurred
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    """
    Runs a single episode in the intersection environment, collecting states, rewards, and actions.
    """
    all_rewards, actions_per_episode = [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    crashed = False

    # Reset environment to start a new episode
    current_obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        steps_counter += 1

        # Get the full state from environment
        full_obs = env.env.current_state

        if isinstance(full_obs, np.ndarray):
            if len(full_obs.shape) == 2 and full_obs.shape[0] > 1:
                master_input = torch.tensor(full_obs.reshape(1, -1), dtype=torch.float32)
            else:
                master_input = torch.tensor(full_obs, dtype=torch.float32).unsqueeze(0)
        elif isinstance(full_obs, torch.Tensor):
            if len(full_obs.shape) == 2 and full_obs.shape[0] > 1:
                master_input = full_obs.reshape(1, -1)
            else:
                master_input = full_obs.unsqueeze(0) if len(full_obs.shape) == 1 else full_obs
        else:
            raise ValueError(f"Unexpected type for observation: {type(full_obs)}")

        # Get embedding from master network
        if train_both or training_master:
            with torch.no_grad():
                proto_action, value, log_prob = master_model.model.policy.forward(master_input)
                embedding = proto_action.cpu().numpy()[0]
        else:
            embedding = master_model.get_proto_action(master_input)

        # Combine car1 state with master's embedding for agent
        if isinstance(full_obs, np.ndarray):
            if len(full_obs.shape) == 2 and full_obs.shape[0] > 1:
                car1_state = full_obs[0].flatten()  # First car's state
            else:
                car1_state = full_obs[:4].flatten()  # First 4 elements
        elif isinstance(full_obs, torch.Tensor):
            if len(full_obs.shape) == 2 and full_obs.shape[0] > 1:
                car1_state = full_obs[0].cpu().numpy().flatten()
            else:
                car1_state = full_obs[:4].cpu().numpy().flatten()
        else:
            car1_state = np.zeros(4)
        if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
            embedding = embedding.flatten()

        # Create agent observation
        try:
            agent_obs = np.concatenate((car1_state, embedding))
        except ValueError as e:
            # Fallback to zeros
            agent_obs = np.zeros(8)  # 4 for car + 4 for embedding
            agent_obs[:len(car1_state)] = car1_state.flatten()[:4]
            if hasattr(embedding, 'shape'):
                agent_obs[4:4 + len(embedding.flatten())] = embedding.flatten()[:4]

        # Get agent action
        agent_action = Agent.get_action(agent_model, current_obs, total_steps,
                                        experiment.EXPLORATION_EXPLOITATION_THRESHOLD)
        if hasattr(agent_action, 'shape') and len(agent_action.shape) > 0:
            scalar_action = agent_action[0]
        elif isinstance(agent_action, list) and len(agent_action) > 0:
            scalar_action = agent_action[0]
        else:
            scalar_action = agent_action

        # For PPO buffer update
        if train_both or not training_master:
            with torch.no_grad():
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                agent_value = agent_model.policy.predict_values(obs_tensor)
                dist = agent_model.policy.get_distribution(obs_tensor)
                action_array = np.array([[scalar_action]])
                action_tensor = torch.tensor(action_array, dtype=torch.long)
                log_prob_agent = dist.log_prob(action_tensor)

        # Take step in environment
        env.render()
        next_obs, reward, done, truncated, info = env.step(scalar_action)

        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        actions_per_episode.append(scalar_action)

        # Check for collision
        if done and info.get("crashed", False):
            crashed = True

        # Add to rollout buffers
        episode_start = (steps_counter == 1)

        if train_both:
            # For master buffer, ensure full_obs is flattened
            full_obs_flat = full_obs.reshape(-1) if isinstance(full_obs, np.ndarray) and len(
                full_obs.shape) == 2 else full_obs

            # Add to agent buffer
            agent_model.rollout_buffer.add(
                agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent
            )

            # Add to master buffer
            master_model.rollout_buffer.add(
                full_obs_flat, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            # For master buffer, ensure full_obs is flattened
            full_obs_flat = full_obs.reshape(-1) if isinstance(full_obs, np.ndarray) and len(
                full_obs.shape) == 2 else full_obs

            master_model.rollout_buffer.add(
                full_obs_flat, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(
                agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent
            )

        # Update current observation
        current_obs = next_obs

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


def monitor_rollout_buffers(master_model, agent_model):
    """Prints statistics about the rollout buffers for debugging"""
    print("\n--- Rollout Buffer Statistics ---")

    # Master buffer stats
    print("Master rollout buffer:")
    print(f"  Buffer size: {master_model.rollout_buffer.buffer_size}")
    print(f"  Current position: {master_model.rollout_buffer.pos}")
    print(f"  Is full: {master_model.rollout_buffer.full}")

    if hasattr(master_model.rollout_buffer, 'observations') and master_model.rollout_buffer.observations is not None:
        print(f"  Observations shape: {master_model.rollout_buffer.observations.shape}")

    if hasattr(master_model.rollout_buffer, 'actions') and master_model.rollout_buffer.actions is not None:
        print(f"  Actions shape: {master_model.rollout_buffer.actions.shape}")

    if hasattr(master_model.rollout_buffer, 'rewards') and master_model.rollout_buffer.rewards is not None:
        rewards = master_model.rollout_buffer.rewards[:master_model.rollout_buffer.pos]
        if len(rewards) > 0:
            print(f"  Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
            print(f"  Rewards: {rewards}")

    # Agent buffer stats
    print("\nAgent rollout buffer:")
    print(f"  Buffer size: {agent_model.rollout_buffer.buffer_size}")
    print(f"  Current position: {agent_model.rollout_buffer.pos}")
    print(f"  Is full: {agent_model.rollout_buffer.full}")

    if hasattr(agent_model.rollout_buffer, 'observations') and agent_model.rollout_buffer.observations is not None:
        print(f"  Observations shape: {agent_model.rollout_buffer.observations.shape}")

    if hasattr(agent_model.rollout_buffer, 'actions') and agent_model.rollout_buffer.actions is not None:
        print(f"  Actions shape: {agent_model.rollout_buffer.actions.shape}")

    if hasattr(agent_model.rollout_buffer, 'rewards') and agent_model.rollout_buffer.rewards is not None:
        rewards = agent_model.rollout_buffer.rewards[:agent_model.rollout_buffer.pos]
        if len(rewards) > 0:
            print(f"  Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
            print(f"  Rewards: {rewards}")

    print("-------------------------------\n")
