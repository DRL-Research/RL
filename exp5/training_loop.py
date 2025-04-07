import os
import time
import torch
import numpy as np
import gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from agent_handler import Agent
from src.airsim_manager import AirsimManager
from src.experiment_config import Experiment
from src.master_handler import MasterModel
from src.model_handler import Model
from utils.plotting_utils import plot_results


def training_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model):
    """
    Runs the overall training loop over cycles and episodes.
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

        master_model.unfreeze() if train_both or training_master else master_model.freeze()
        agent_model.policy.set_training_mode(train_both or (not training_master))

        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} @")

            episode_rewards, episode_actions, steps, *_ = run_episode(
                experiment, total_steps, env, master_model, agent_model,
                train_both=train_both, training_master=training_master
            )
            pause_experiment_simulation(env)
            total_steps += steps
            all_rewards.append(episode_rewards)
            all_actions.append(episode_actions)

            # Only update the networks when the rollout buffers are full or at the designated interval.
            if (master_model.model.rollout_buffer.full or agent_model.rollout_buffer.full or
                    episode_counter % Experiment.EPISODE_AMOUNT_FOR_TRAIN == 0):
                # Compute a last master tensor for use in training updates.
                with torch.no_grad():
                    last_master_obs = np.concatenate((
                        env.envs[0].airsim_manager.get_car1_state(),
                        env.envs[0].airsim_manager.get_car2_state(),
                        env.envs[0].airsim_manager.get_car3_state()
                    ))
                    last_master_tensor = torch.tensor(last_master_obs, dtype=torch.float32).unsqueeze(0)
                if train_both:
                    # In cycle 0: train both master and agent.
                    last_master_tensor = train_master_and_reset_buffer(env, master_model)
                    train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor)
                elif training_master:
                    # In cycle 1: train only the master.
                    train_master_and_reset_buffer(env, master_model)
                else:
                    # In cycle 2: train only the agent.
                    train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor)
    print("Training completed.")
    return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    """
    Runs a single episode in the environment, collecting states, rewards, and actions.
    The master network input is constructed from Car1, Car2, and Car3 states (12 dimensions).
    The agent's observation consists of Car1 state (4 dimensions) concatenated with the 4-dim proto embedding.
    """
    all_states, all_rewards, actions_per_episode = [], [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    done = False
    # Resume simulation (assumed not to call reset during learning update)
    resume_experiment_simulation(env)

    while not done:
        steps_counter += 1
        # Get states for Car1, Car2, and Car3.
        car1_state = env.envs[0].airsim_manager.get_car1_state()
        car2_state = env.envs[0].airsim_manager.get_car2_state()
        car3_state = env.envs[0].airsim_manager.get_car3_state()  # Include Car3 state.
        # Concatenate states to form a 12-dimensional observation for the master network.
        master_obs_as_array = np.concatenate((car1_state, car2_state, car3_state))
        master_input = torch.tensor(master_obs_as_array, dtype=torch.float32).unsqueeze(0)

        if train_both or training_master:
            with torch.no_grad():
                proto_action, value, log_prob = master_model.model.policy.forward(master_input)
                embedding = proto_action.cpu().numpy()[0]
        else:
            embedding = master_model.get_proto_action(master_input)

        # The agent's observation is Car1's state concatenated with the 4-dim proto embedding.
        agent_obs = np.concatenate((car1_state, embedding))
        agent_action = Agent.get_action(agent_model, agent_obs, total_steps,
                                        experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
        scalar_action = agent_action[0]

        if train_both or not training_master:
            with torch.no_grad():
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                agent_value = agent_model.policy.predict_values(obs_tensor)
                dist = agent_model.policy.get_distribution(obs_tensor)
                action_array = np.array([[scalar_action]])
                action_tensor = torch.tensor(action_array, dtype=torch.long)
                log_prob_agent = dist.log_prob(action_tensor)
        next_obs, reward, done, info = env.step(agent_action)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        all_states.append(master_obs_as_array)
        actions_per_episode.append(scalar_action)

        episode_start = (steps_counter == 1)
        if train_both:
            agent_model.rollout_buffer.add(
                agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent
            )
            master_model.model.rollout_buffer.add(master_obs_as_array, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            master_model.model.rollout_buffer.add(master_obs_as_array, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(
                agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent
            )
    pause_experiment_simulation(env)
    return episode_sum_of_rewards, actions_per_episode, steps_counter, None, None, all_states


def run_experiment(experiment_config):
    """
    Sets up the environment, models, and logging, then runs the training loop.
    Finally, saves the trained models.
    """
    p_agent_loss, p_master_loss, p_episode_counter = [], [], []
    airsim_manager = AirsimManager(experiment_config)
    master_model = MasterModel(embedding_size=4, airsim_manager=airsim_manager, experiment=experiment_config)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
    agent_model = Model(env, experiment_config).model

    base_path = experiment_config.EXPERIMENT_PATH
    agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
    master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    results = training_loop(p_agent_loss, p_master_loss, p_episode_counter,
                              experiment_config, env, agent_model, master_model)
    if not experiment_config.ONLY_INFERENCE:
        plot_results(path=os.path.join(base_path, "agent_logs"), all_rewards=results[5], all_actions=results[6])
        plot_results(path=os.path.join(base_path, "master_logs"), all_rewards=results[5], all_actions=results[6])
    agent_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_agent.pth")
    master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")
    print("Models saved.")
    env.close()


def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()


def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()


def train_master_and_reset_buffer(env, master_model):
    with torch.no_grad():
        # Concatenate states from Car1, Car2, and Car3 (12-dimensional master state)
        last_master_obs = np.concatenate((
            env.envs[0].airsim_manager.get_car1_state(),
            env.envs[0].airsim_manager.get_car2_state(),
            env.envs[0].airsim_manager.get_car3_state()
        ))
        last_master_tensor = torch.tensor(last_master_obs, dtype=torch.float32).unsqueeze(0)
        last_value = master_model.model.policy.predict_values(last_master_tensor)
    master_model.model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Master...")
    master_model.model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    master_model.model.rollout_buffer.reset()
    return last_master_tensor


def train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor):
    with torch.no_grad():
        car1_state = env.envs[0].airsim_manager.get_car1_state()
        embedding = master_model.get_proto_action(last_master_tensor)
        agent_obs = np.concatenate((car1_state, embedding))
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
        last_value = agent_model.policy.predict_values(agent_obs_tensor)
    agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Agent...")
    agent_model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    agent_model.rollout_buffer.reset()
