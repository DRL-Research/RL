import os
import torch
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from agent_handler import Agent
from src.airsim_manager import AirsimManager
from src.experiment_config import Experiment
from src.master_handler import MasterModel
from src.model_handler import Model
from utils.plotting_utils import plot_results
import json


def training_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")

        # Define the structure of the training process
        train_both = cycle == 0
        training_master = not train_both and cycle % 2 == 1

        # Update weights logic
        master_model.unfreeze() if train_both or training_master else master_model.freeze()
        agent_model.policy.set_training_mode(train_both or not training_master)

        # Enter episode loop
        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} @")

            episode_rewards, episode_actions, steps, *_ = run_episode(
                experiment, episode, total_steps, env, master_model, agent_model,
                train_both=train_both, training_master=training_master
            )

            # After Done, stop simulation to update network
            pause_experiment_simulation(env)
            total_steps += steps
            all_rewards.append(episode_rewards)

            all_actions.append(episode_actions)

            # Check if the buffer is full or if we reached the episode count for updating the network
            if master_model.model.rollout_buffer.full or agent_model.rollout_buffer.full or episode_counter % Experiment.EPISODE_AMOUNT_FOR_TRAIN == 0:
                last_master_tensor = train_master_and_reset_buffer(env, master_model, agent_model)
                train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor)

    print("Training completed.")
    return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions


def run_episode(experiment, episode, total_steps, env, master_model, agent_model, train_both, training_master):
    all_states,all_rewards,actions_per_episode = [],[],[]
    steps_counter,episode_sum_of_rewards = 0,0
    done = False

    env.reset()
    resume_experiment_simulation(env)

    # Run the episode
    while not done:
        steps_counter += 1

        # Get the state of the cars
        car1_state = env.envs[0].airsim_manager.get_car1_state()
        car2_state = env.envs[0].airsim_manager.get_car2_state()
        # Get the master network’s proto embedding (4-dim)
        master_obs_as_tensor = np.concatenate((car1_state, car2_state))
        master_tensor = torch.tensor(master_obs_as_tensor).unsqueeze(0).float()

        # Get the proto action from the master model
        if train_both or training_master:
            # Forward pass in Master
            with torch.no_grad():
                proto_action, value, log_prob = master_model.model.policy.forward(master_tensor)
                embedding = proto_action.cpu().numpy()[0]
        else:
            # Inference only
            embedding = master_model.get_proto_action(master_tensor)

        # Build the new observation (8-dim)
        agent_obs = np.concatenate((car1_state, embedding))
        agent_action = Agent.get_action(agent_model, agent_obs, total_steps,
                                        experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
        scalar_action = agent_action[0]

        experiment.logger.run["Actions_Car1"].append(scalar_action)
        experiment.logger.run["Embedding"].log(json.dumps(embedding.astype(float).tolist()))

        # Get the value and log_prob of the agent
        if train_both or not training_master:
            with torch.no_grad():
                obs_tensor = torch.tensor(agent_obs).unsqueeze(0).float()
                agent_value = agent_model.policy.predict_values(obs_tensor)
                dist = agent_model.policy.get_distribution(obs_tensor)
                # Convert the action to a tensor
                action_array = np.array([[scalar_action]])
                action_tensor = torch.tensor(action_array, dtype=torch.long)
                log_prob_agent = dist.log_prob(action_tensor)

        # Take a step in the environment
        next_obs, reward, done, info = env.step(agent_action)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        experiment.logger.run["episode_reward"].append(reward)

        all_states.append(master_obs_as_tensor)
        actions_per_episode.append(scalar_action)

        # experiment.logger.run["rewards"].append(reward, step=total_steps)

        # Get speed from AirSim
        speed_car1 = env.envs[0].airsim_manager.get_vehicle_speed(experiment.CAR1_NAME)
        speed_car2 = env.envs[0].airsim_manager.get_vehicle_speed(experiment.CAR2_NAME)

        # Log collision flag (1 if collision occurred at that step, else 0)
        is_collision = int(reward <= experiment.COLLISION_REWARD)
        experiment.logger.run["Speed"].append(
            {"speed_car1": speed_car1, "speed_car2": speed_car2, "episode": episode,
             "collision_flag": is_collision * 12}
        )

        # Check if the episode is at the start (for adding the first observation to the buffer)
        episode_start = (steps_counter == 1)
        if train_both:
            agent_model.rollout_buffer.add(
                agent_obs,  # current state
                action_array,  # action
                reward,  # reward
                episode_start,  # is it the start of the episode
                agent_value,  # value (the critic's estimate for this state)
                log_prob_agent  # log_prob of the action by the agent
            )
            master_model.model.rollout_buffer.add(master_obs_as_tensor, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            master_model.model.rollout_buffer.add(master_obs_as_tensor, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(
                agent_obs,
                action_array,
                reward,
                episode_start,
                agent_value,
                log_prob_agent
            )
    pause_experiment_simulation(env)
    return episode_sum_of_rewards, actions_per_episode, steps_counter, None, None, all_states

def run_experiment(experiment_config):
    # Log hyperparameters to Neptune
    experiment_config.logger.log_hyperparameters(experiment_config)

    p_agent_loss, p_master_loss, p_episode_counter = [], [], []

    # Create the environment
    airsim_manager = AirsimManager(experiment_config)

    # Create the agent and master models
    master_model = MasterModel(embedding_size=4, airsim_manager=airsim_manager, experiment=experiment_config)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
    agent_model = Model(env, experiment_config).model

    base_path = experiment_config.EXPERIMENT_PATH

    # Stable-Baselines loggers
    agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
    master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    # Run the training loop
    results = training_loop(p_agent_loss, p_master_loss, p_episode_counter,
                            experiment_config, env, agent_model, master_model)

    # if not experiment_config.ONLY_INFERENCE:
    #     plot_results(path=os.path.join(base_path, "agent_logs"),
    #                  all_rewards=results[5], all_actions=results[6])
    #     plot_results(path=os.path.join(base_path, "master_logs"),
    #                  all_rewards=results[5], all_actions=results[6])


    agent_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}/trained_model_agent.pth")
    master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}/trained_model_master.pth")

    # Close Stable-Baselines loggers
    agent_logger.close()
    master_logger.close()

    experiment_config.logger.log_from_csv(
        path=f"{experiment_config.SAVE_MODEL_DIRECTORY}/agent_logs/progress.csv",
        column_name="train/value_loss",
        metric_name="agent_value_loss"
    )
    experiment_config.logger.log_from_csv(
        path=f"{experiment_config.SAVE_MODEL_DIRECTORY}/master_logs/progress.csv",
        column_name="train/value_loss",
        metric_name="master_value_loss"
    )

    # Upload final models to Neptune
    experiment_config.logger.upload_model(
        f"{experiment_config.SAVE_MODEL_DIRECTORY}/trained_model_agent.pth", model_name="trained_agent"
    )
    experiment_config.logger.upload_model(
        f"{experiment_config.SAVE_MODEL_DIRECTORY}/trained_model_master.pth", model_name="trained_master"
    )
    print("Models saved.")
    env.close()

def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()

def train_master_and_reset_buffer(env, master_model, agent_model):
    with torch.no_grad():
        last_master_obs = np.concatenate((
            env.envs[0].airsim_manager.get_car1_state(),
            env.envs[0].airsim_manager.get_car2_state()
        ))
        last_master_tensor = torch.tensor(last_master_obs).unsqueeze(0).float()
        last_value = master_model.model.policy.predict_values(last_master_tensor)
    master_model.model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Master...")
    master_model.model.learn(
        total_timesteps=1,
        reset_num_timesteps=False,
        log_interval=1
    )
    master_model.model.rollout_buffer.reset()
    return last_master_tensor

def train_agent_and_reset_buffer(env, master_model, agent_model, last_master_tensor):
    with torch.no_grad():
        car1_state = env.envs[0].airsim_manager.get_car1_state()
        embedding = master_model.get_proto_action(last_master_tensor)
        agent_obs = np.concatenate((car1_state, embedding))
        agent_obs_tensor = torch.tensor(agent_obs).unsqueeze(0).float()
        last_value = agent_model.policy.predict_values(agent_obs_tensor)
    agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Agent...")
    agent_model.learn(
        total_timesteps=1,
        reset_num_timesteps=False,
        log_interval=1
    )
    agent_model.rollout_buffer.reset()
