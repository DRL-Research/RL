
from master_handler import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.model.model_handler import Model
from utils.agent_handler import Agent
from utils.airsim_manager import AirsimManager
from utils.plotting_utils import PlottingUtils


def training_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model, agent):
    """
    Training/inference loop for a two-vehicle setup.

    In ONLY_INFERENCE mode:
      - agent_model is expected to be a dictionary with keys 'agent1' and 'agent2',
        each holding a loaded agent model.
      - Each vehicle builds its own observation (its local state concatenated with the proto embedding)
        and uses its corresponding model.
    In training mode, a single agent model is used for both vehicles.
    """
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    if experiment.ONLY_INFERENCE:
        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} INFERENCE @")
            # Run an inference episode that uses separate agent models for each vehicle.
            episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, episode_states = run_episode_inference(
                env, agent_model['agent1'], agent_model['agent2'], master_model, experiment
            )
            total_steps += steps
            all_rewards.append(episode_rewards)
            all_actions.append(episode_actions)
        print("Inference completed.")
        return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions

    else:
        # Training mode: a single agent model is used for both vehicles.
        for cycle in range(experiment.CYCLES):
            print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")
            if cycle % 2 == 1:
                print("Training Master Network (Agent Network is frozen)")
                master_model.unfreeze()
                agent_model.policy.set_training_mode(False)
                master_training_steps = 0

                for episode in range(experiment.EPISODES_PER_CYCLE):
                    episode_counter += 1
                    p_episode_counter.append(episode_counter)
                    print(f"  @ Episode {episode_counter} (Master Training) @")
                    episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, _ = run_episode(
                        p_master_loss, p_agent_loss, experiment, total_steps, env, master_model, agent_model, agent,
                        training_master=True
                    )
                    total_steps += steps
                    master_training_steps += steps
                    all_rewards.append(episode_rewards)
                    all_actions.append(episode_actions)
                    if episode_counter % 4 == 0:
                        master_model.train_master(master_training_steps)
                        master_training_steps = 0
            else:
                print("Training Agent Network (Master Network is frozen)")
                master_model.freeze()
                agent_model.policy.set_training_mode(True)
                agent_training_steps = 0

                for episode in range(experiment.EPISODES_PER_CYCLE):
                    episode_counter += 1
                    p_episode_counter.append(episode_counter)
                    print(f"  @ Episode {episode_counter} (Agent Training) @")
                    episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, _ = run_episode(
                        p_master_loss, p_agent_loss, experiment, total_steps, env, master_model, agent_model, agent,
                        training_master=False
                    )
                    agent_training_steps += steps
                    if episode % 4 == 0:
                        agent_model.learn(total_timesteps=agent_training_steps, log_interval=1)
                        agent_training_steps = 0
                    total_steps += steps
                    all_rewards.append(episode_rewards)
                    all_actions.append(episode_actions)
        print("Training completed.")
        return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions


def run_episode_inference(env, agent1_model, agent2_model, master_model, experiment):
    """
    Runs one inference episode for both vehicles.

    Each vehicle:
      - Retrieves its local 4-dim state.
      - The master network receives the concatenated local states (8-dim) and produces a 4-dim proto embedding.
      - Each vehicle’s observation is built by concatenating its own local state (4-dim) with the proto embedding (4-dim).
      - Each loaded agent model is used (deterministically) to choose its action.

    Returns:
      episode_sum: Total reward over the episode.
      actions_list: List of tuples (action1, action2) for each step.
      steps_counter: Number of steps.
      p_agent_loss: (Empty list in inference mode)
      p_master_loss: (Empty list in inference mode)
      all_states: List of master inputs from each step.
    """
    episode_sum = 0
    steps_counter = 0
    actions_list = []
    all_states = []
    p_agent_loss = []
    p_master_loss = []
    resume_experiment_simulation(env)

    while True:
        steps_counter += 1
        # Retrieve local states for Car1 and Car2 (each 4-dim)
        state_car1 = env.envs[0].airsim_manager.get_car1_state()
        state_car2 = env.envs[0].airsim_manager.get_car2_state()
        # Build master input: concatenated local states (8-dim vector)
        master_input = torch.tensor(np.concatenate((state_car1, state_car2))[np.newaxis, :], dtype=torch.float32)
        # Get proto embedding (4-dim) from master network
        proto_embedding = master_model.get_proto_action(master_input)
        # Build separate observations for each vehicle:
        obs_agent1 = np.concatenate((state_car1, proto_embedding))
        obs_agent2 = np.concatenate((state_car2, proto_embedding))
        # Get actions using Agent's static get_action method (set total_steps high for deterministic mode)
        action1 = Agent.get_action(agent1_model, obs_agent1, 100000, experiment.EXPLORATION_EXPLOTATION_THRESHOLD,
                                   False)
        action2 = Agent.get_action(agent2_model, obs_agent2, 100000, experiment.EXPLORATION_EXPLOTATION_THRESHOLD,
                                   False)
        print("agent 1 action :", action1)
        print("agent 2 action :", action2)
        print("proto action :", proto_embedding)
        combined_action = (action1, action2)
        # Step the environment using the tuple of actions
        next_state, reward, done, info = env.step(combined_action)
        all_states.append(master_input)
        episode_sum += reward
        actions_list.append(combined_action)
        if done:
            pause_experiment_simulation(env)
            break

    return episode_sum, actions_list, steps_counter, p_agent_loss, p_master_loss, all_states


def run_episode(p_master_loss, p_agent_loss, experiment, total_steps, env, master_model, agent_model, agent,
                training_master):
    """
    Runs one episode in training mode.

    In training mode, a single agent model is used for both vehicles.
    Here we build one observation (using Car1’s local state plus the master proto embedding)
    and use that to get an action which is applied to both vehicles.
    """
    all_states = []
    all_rewards = []
    done = False
    episode_sum = 0
    steps_counter = 0
    actions_list = []
    resume_experiment_simulation(env)

    while not done:
        print('total_steps', total_steps)
        steps_counter += 1
        state_car1 = env.envs[0].airsim_manager.get_car1_state()
        state_car2 = env.envs[0].airsim_manager.get_car2_state()
        master_input = torch.tensor(np.concatenate((state_car1, state_car2))[np.newaxis, :], dtype=torch.float32)
        proto_embedding = master_model.get_proto_action(master_input)
        # For training mode, use Car1's state plus proto embedding as observation.
        obs = np.concatenate((state_car1, proto_embedding))
        agent_action = Agent.get_action(agent_model, obs, total_steps, experiment.EXPLORATION_EXPLOTATION_THRESHOLD,
                                        training_master)
        next_state, reward, done, info = env.step(agent_action)
        all_states.append(master_input)
        all_rewards.append(reward)
        episode_sum += reward
        actions_list.append(agent_action)
        if done:
            all_rewards.append(reward)
            pause_experiment_simulation(env)
            if not training_master:
                if p_master_loss:
                    p_master_loss.append(p_master_loss[-1])
                else:
                    p_master_loss.append(0)
            break

    return episode_sum, actions_list, steps_counter, p_agent_loss, p_master_loss, all_states


def plot_results(path, all_rewards, all_actions):
    PlottingUtils.plot_losses(path)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()


def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()


def plot_metrics(metrics_path):
    metrics = pd.read_csv(metrics_path)
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["loss"], label="Loss")
    plt.title("Loss over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["entropy"], label="Entropy", color="orange")
    plt.title("Entropy over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["return"], label="Return", color="green")
    plt.title("Return over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.show()


# -------------------------------
# Main Experiment Runner
# -------------------------------
def run_experiment(experiment_config):
    if experiment_config.ONLY_INFERENCE:
        p_total_rewards = []
        p_agent_loss = []
        p_master_loss = []
        p_master_entropy = []
        p_episode_counter = []
        # Initialize AirSim Manager
        airsim_manager = AirsimManager(experiment_config)
        # Initialize Master Model and load pre-trained weights
        master_model = MasterModel(embedding_size=4, airsim_manager=airsim_manager, experiment=experiment_config)
        master_model.load("experiments/16_02_2025-11_15_40_Experiment5/trained_model_master.zip")
        # Create two separate Agent instances (one for each car) with the same master model
        agent1 = Agent(experiment_config, airsim_manager, master_model)
        agent2 = Agent(experiment_config, airsim_manager, master_model)
        # Create the shared environment
        env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
        # Load separate pre-trained agent models for each vehicle
        agent1_model = Model.load("experiments/16_02_2025-11_56_55_Experiment5/trained_model.zip", env,
                                  experiment_config).model
        agent2_model = Model.load("experiments/16_02_2025-11_56_55_Experiment5/trained_model.zip", env,
                                  experiment_config).model
        # Configure loggers
        base_path = experiment_config.EXPERIMENT_PATH
        agent_logger_path = os.path.join(base_path, "agent_logs")
        agent_logger = configure(agent_logger_path, ["stdout", "csv", "tensorboard"])
        master_logger_path = os.path.join(base_path, "master_logs")
        master_model_logger = configure(master_logger_path, ["stdout", "csv", "tensorboard"])
        agent1_model.set_logger(agent_logger)
        master_model.set_logger(master_model_logger)
        # Create a dictionary for the agent models
        agent_models = {'agent1': agent1_model, 'agent2': agent2_model}
        # Run the inference loop; note that the inference branch uses the two-agent-model version.
        p_episode_counter, p_agent_loss, p_master_loss, _, collision_counter, all_rewards, all_actions = training_loop(
            p_agent_loss, p_master_loss, p_episode_counter,
            experiment=experiment_config,
            env=env,
            agent_model=agent_models,  # dictionary with keys 'agent1' and 'agent2'
            master_model=master_model,
            agent=agent1  # Pass one of the agent instances; the env is shared.
        )
        print(p_master_loss)
        # Optionally re-save the loaded models
        agent1_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_agent1_inference")
        agent2_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_agent2_inference")
        master_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_master_inference")
        agent_logger.close()
        master_model_logger.close()
        print("Models loaded and inference completed")
        print("Total collisions:", collision_counter)
    else:
        p_total_rewards = []
        p_agent_loss = []
        p_master_loss = []
        p_master_entropy = []
        p_episode_counter = []
        airsim_manager = AirsimManager(experiment_config)
        master_model = MasterModel(embedding_size=4, airsim_manager=airsim_manager, experiment=experiment_config)
        agent = Agent(experiment_config, airsim_manager, master_model)
        env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
        agent_model = Model(env, experiment_config).model
        base_path = experiment_config.EXPERIMENT_PATH
        agent_logger_path = os.path.join(base_path, "agent_logs")
        agent_logger = configure(agent_logger_path, ["stdout", "csv", "tensorboard"])
        master_logger_path = os.path.join(base_path, "master_logs")
        master_model_logger = configure(master_logger_path, ["stdout", "csv", "tensorboard"])
        agent_model.set_logger(agent_logger)
        master_model.set_logger(master_model_logger)
        p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions = training_loop(
            p_agent_loss, p_master_loss, p_episode_counter,
            experiment=experiment_config,
            env=env,
            agent_model=agent_model,
            master_model=master_model,
            agent=agent,
        )
        print(p_master_loss)
        plot_results(path=agent_logger_path, all_rewards=all_rewards, all_actions=all_actions)
        plot_results(path=master_logger_path, all_rewards=all_rewards, all_actions=all_actions)
        agent_model.save(experiment_config.SAVE_MODEL_DIRECTORY)
        master_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_master")
        agent_logger.close()
        master_model_logger.close()
        print("Models saved")
        print("Total collisions:", collision_counter)


# -------------------------------
# Override GYM Environment Functions
# -------------------------------
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()


def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
