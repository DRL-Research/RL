import os
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.plotting_utils import PlottingUtils
from utils.model.model_handler import Model
from agent_handler import Agent, DummyVecEnv
from master_model import MasterModel

# Register the environment if needed
# You may need to adapt this based on your specific Highway environment
try:
    gym.register(
        id='RELintersection-v0',
        entry_point='envs.intersection:RELIntersectionEnv',
    )
except gym.error.RegistrationError:
    # Environment already registered
    pass


def run_experiment(experiment_config, config):
    """
    Main function to run a complete experiment with master-agent architecture
    in the Highway environment.

    Args:
        experiment_config: Configuration for experiment parameters
        config: Configuration for the Highway environment

    Returns:
        agent_model: Trained agent model
        master_model: Trained master model
        collision_counter: Number of collisions during training
    """
    print("Starting experiment:", experiment_config.EXPERIMENT_ID)

    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_config.EXPERIMENT_PATH, exist_ok=True)

    # Initialize tracking variables
    p_agent_loss, p_master_loss, p_episode_counter = [], [], []

    # Create master model
    master_model = MasterModel(embedding_size=4, experiment=experiment_config)

    # Create agent environment wrapper
    env = DummyVecEnv([lambda: Agent(experiment_config, master_model=master_model)])

    # Create agent model (using your Model class)
    agent_model = Model(env, experiment_config).model

    # Configure loggers
    base_path = experiment_config.EXPERIMENT_PATH
    agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
    master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    # Training process
    if experiment_config.ONLY_INFERENCE:
        print("Running in inference-only mode")
        agent_model.load(f"{experiment_config.LOAD_WEIGHT_DIRECTORY}_agent.pth")
        master_model.load(f"{experiment_config.LOAD_WEIGHT_DIRECTORY}_master.pth")
        print(f"Loaded weights from {experiment_config.LOAD_WEIGHT_DIRECTORY} for inference.")

        # Run some evaluation episodes if needed
        # This section can be expanded for evaluation metrics
        eval_rewards = []
        eval_actions = []

        for episode in range(experiment_config.EVAL_EPISODES):
            print(f"Evaluation episode {episode + 1}/{experiment_config.EVAL_EPISODES}")
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            actions = []

            while not done and not truncated:
                # Get embedding from master
                master_input = torch.tensor(obs[:20], dtype=torch.float32).unsqueeze(0)  # Full state to master
                embedding = master_model.get_proto_action(master_input)

                # Combine car state with embedding for agent
                agent_obs = np.concatenate((obs[:4], embedding))  # Assuming first 4 values are ego car state

                # Get action from agent
                action, _ = agent_model.predict(agent_obs, deterministic=True)

                # Take step
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                actions.append(action[0])

            eval_rewards.append(episode_reward)
            eval_actions.append(actions)
            print(f"Episode {episode + 1} reward: {episode_reward}")

        print(f"Average evaluation reward: {np.mean(eval_rewards)}")

        # Close environments and loggers
        env.close()
        agent_logger.close()
        master_logger.close()

        return agent_model, master_model, 0

    # Regular training run
    print("Running full training")

    # Load previous weights if specified
    if experiment_config.LOAD_PREVIOUS_WEIGHT:
        try:
            agent_model.load(f"{experiment_config.LOAD_WEIGHT_DIRECTORY}_agent.pth")
            master_model.load(f"{experiment_config.LOAD_WEIGHT_DIRECTORY}_master.pth")
            print(
                f"Loaded weights from {experiment_config.LOAD_WEIGHT_DIRECTORY}, models will be trained from this point!")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            print("Starting training from scratch")

    # Run the training loop
    p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions = \
        training_loop(p_agent_loss, p_master_loss, p_episode_counter,
                      experiment_config, env, agent_model, master_model)

    # Save models
    agent_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_agent.pth")
    master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")
    print("Models saved")

    # Close loggers
    agent_logger.close()
    master_logger.close()

    # Plot results
    plot_results(experiment_config, all_rewards, all_actions)

    print("Training completed.")
    print("Total collisions:", collision_counter)

    # Close environment
    env.close()

    return agent_model, master_model, collision_counter


def training_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model):
    """
    Runs the overall training loop over cycles and episodes for Highway environment.
    In cycle 0, both the master and agent are trained.
    In cycle 1, only the master is trained.
    In cycle 2, only the agent is trained.
    """
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")
        # Determine training mode based on the cycle:
        train_both = (cycle == 0)
        training_master = (not train_both) and (cycle % 2 == 1)
        # For cycle 2: (not train_both) and (cycle % 2 == 0) => agent training only

        # Set training modes for both networks
        master_model.unfreeze() if train_both or training_master else master_model.freeze()
        agent_model.policy.set_training_mode(train_both or (not training_master))

        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} @")

            episode_sum_of_rewards, actions_per_episode, steps_counter, crashed = run_episode(
                experiment, total_steps, env, master_model, agent_model,
                train_both=train_both, training_master=training_master
            )

            if crashed:
                collision_counter += 1
                print('Collision!')
            else:
                print('Success!')

            total_steps += steps_counter
            all_rewards.append(episode_sum_of_rewards)
            all_actions.append(actions_per_episode)

            print(f"  Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
            print(f"  Total Steps: {total_steps}, Episode steps: {steps_counter}")

            # Train models when rollout buffers are full or at designated interval
            if (master_model.rollout_buffer.full or agent_model.rollout_buffer.full or
                    episode_counter % experiment.EPISODE_AMOUNT_FOR_TRAIN == 0):

                # Get last state for training updates
                with torch.no_grad():
                    # Get current observation
                    current_obs, _ = env.reset()
                    # Full observation for master
                    full_obs = current_obs[:20]  # Adjust based on your observation structure
                    last_master_tensor = torch.tensor(full_obs, dtype=torch.float32).unsqueeze(0)

                if train_both:
                    # In cycle 0: train both master and agent
                    last_master_tensor = train_master_and_reset_buffer(env, master_model, full_obs)
                    train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor)
                elif training_master:
                    # In cycle 1: train only the master
                    train_master_and_reset_buffer(env, master_model, full_obs)
                else:
                    # In cycle 2: train only the agent
                    train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor)

    print("Training completed.")
    return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    """
    Runs a single episode in the Highway environment, collecting states, rewards, and actions.
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

        # Extract full state (all cars) for master
        full_obs = current_obs[:20]  # Adjust based on your observation structure

        # Get master input
        master_input = torch.tensor(full_obs, dtype=torch.float32).unsqueeze(0)

        # Get embedding from master network
        if train_both or training_master:
            with torch.no_grad():
                proto_action, value, log_prob = master_model.model.policy.forward(master_input)
                embedding = proto_action.cpu().numpy()[0]
        else:
            embedding = master_model.get_proto_action(master_input)

        # Combine ego car state with master's embedding for agent
        ego_car_state = current_obs[:4]  # Assuming first 4 values are ego car state
        agent_obs = np.concatenate((ego_car_state, embedding))

        # Get agent action
        agent_action = Agent.get_action(agent_model, agent_obs, total_steps,
                                        experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
        scalar_action = agent_action[0]

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
        print(f"Action: {agent_action}")
        next_obs, reward, done, truncated, info = env.step(agent_action)

        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        actions_per_episode.append(scalar_action)

        # Check for collision
        if done and info.get("crashed", False):
            crashed = True

        # Add to rollout buffers
        episode_start = (steps_counter == 1)
        if train_both:
            agent_model.rollout_buffer.add(
                agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent
            )
            master_model.rollout_buffer.add(full_obs, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            master_model.rollout_buffer.add(full_obs, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(
                agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent
            )

        # Update current observation
        current_obs = next_obs

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


def train_master_and_reset_buffer(env, master_model, full_obs):
    """Trains the master model and resets its buffer"""
    with torch.no_grad():
        last_master_tensor = torch.tensor(full_obs, dtype=torch.float32).unsqueeze(0)
        last_value = master_model.model.policy.predict_values(last_master_tensor)

    master_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Master...")
    master_model.model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    master_model.rollout_buffer.reset()
    return last_master_tensor


def train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor):
    """Trains the agent model and resets its buffer"""
    with torch.no_grad():
        # Get embedding from master
        embedding = master_model.get_proto_action(last_master_tensor)

        # Combine with car state
        car_state = last_master_tensor.cpu().numpy()[0][:4]  # Assuming first 4 values are ego car state
        agent_obs = np.concatenate((car_state, embedding))

        # Get value for advantage computation
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
        last_value = agent_model.policy.predict_values(agent_obs_tensor)

    agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Agent...")
    agent_model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    agent_model.rollout_buffer.reset()


def plot_results(experiment, all_rewards, all_actions):
    """Plot training results"""
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH, experiment.EXPERIMENT_ID)
    PlottingUtils.plot_rewards(all_rewards, experiment.EXPERIMENT_ID)
    PlottingUtils.plot_actions(all_actions, experiment.EXPERIMENT_ID)
    PlottingUtils.show_plots()