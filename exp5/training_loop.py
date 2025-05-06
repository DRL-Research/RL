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


def training_loop_both(p_agent_loss, p_master_loss, p_episode_counter, experiment, airsim_manager, master_model,
                       agent_model_car1, agent_model_car3):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards_car1, all_rewards_car3 = [], []
    all_actions_car1, all_actions_car3 = [], []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")
        # Determine training mode based on the cycle
        train_both = (cycle == 0)
        training_master = (not train_both) and (cycle % 2 == 1)

        master_model.unfreeze() if train_both or training_master else master_model.freeze()
        agent_model_car1.policy.set_training_mode(train_both or (not training_master))
        agent_model_car3.policy.set_training_mode(train_both or (not training_master))

        for episode in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            p_episode_counter.append(episode_counter)
            print(f"  @ Episode {episode_counter} @")

            # Run episode for both agents
            results = run_episode_both(
                experiment, total_steps, airsim_manager, master_model,
                agent_model_car1, agent_model_car3,
                train_both=train_both, training_master=training_master
            )

            episode_rewards_car1, episode_rewards_car3, episode_actions_car1, episode_actions_car3, steps = results

            total_steps += steps
            all_rewards_car1.append(episode_rewards_car1)
            all_rewards_car3.append(episode_rewards_car3)
            all_actions_car1.append(episode_actions_car1)
            all_actions_car3.append(episode_actions_car3)

            print(
                f"  Episode {episode_counter} finished - Car1 reward: {episode_rewards_car1}, Car3 reward: {episode_rewards_car3}")

            # Training updates
            if episode_counter % experiment.EPISODE_AMOUNT_FOR_TRAIN == 0:
                # Get master observation
                with torch.no_grad():
                    last_master_obs = np.concatenate((
                        airsim_manager.get_car1_state(),
                        airsim_manager.get_car2_state(),
                        airsim_manager.get_car3_state(),
                        airsim_manager.get_car4_state(),
                        airsim_manager.get_car5_state()
                    ))
                    last_master_tensor = torch.tensor(last_master_obs, dtype=torch.float32).unsqueeze(0)

                if train_both:
                    # Train master first to get updated embedding
                    last_master_tensor = update_master_model(airsim_manager, master_model)
                    # Then train both agents
                    update_agent_model_car1(airsim_manager, master_model, agent_model_car1, last_master_tensor)
                    update_agent_model_car3(airsim_manager, master_model, agent_model_car3, last_master_tensor)
                elif training_master:
                    # Train only master
                    update_master_model(airsim_manager, master_model)
                else:
                    # Train only agents
                    update_agent_model_car1(airsim_manager, master_model, agent_model_car1, last_master_tensor)
                    update_agent_model_car3(airsim_manager, master_model, agent_model_car3, last_master_tensor)

    return p_episode_counter, p_agent_loss, p_master_loss, [agent_model_car1, agent_model_car3], collision_counter, [
        all_rewards_car1, all_rewards_car3], [all_actions_car1, all_actions_car3]


def run_episode_both(experiment, total_steps, airsim_manager, master_model, agent_model_car1, agent_model_car3,
                     train_both, training_master):
    import airsim  # Making sure airsim is imported

    rewards_car1, rewards_car3 = [], []
    actions_car1, actions_car3 = [], []
    steps_counter, car1_total_reward, car3_total_reward = 0, 0, 0
    done_car1, done_car3 = False, False

    # Resume simulation
    airsim_manager.resume_simulation()

    # Reset car positions
    airsim_manager.reset_cars_to_initial_positions()
    airsim_manager.reset_for_new_episode()

    while not (done_car1 and done_car3):
        steps_counter += 1

        # Get states for all cars
        car1_state = airsim_manager.get_car1_state()
        car2_state = airsim_manager.get_car2_state()
        car3_state = airsim_manager.get_car3_state()
        car4_state = airsim_manager.get_car4_state()
        car5_state = airsim_manager.get_car5_state()

        # Check for out of bounds
        if any(abs(state[2]) > 100 or abs(state[3]) > 100 for state in
               [car1_state, car2_state, car3_state, car4_state, car5_state]):
            print("Car out of bounds detected. Resetting simulation.")
            airsim_manager.pause_simulation()
            airsim_manager.hard_reset_simulation()
            return 0, 0, [], [], steps_counter

        # Get master embedding
        master_obs = np.concatenate((car1_state, car2_state, car3_state, car4_state, car5_state))
        master_input = torch.tensor(master_obs, dtype=torch.float32).unsqueeze(0)

        if train_both or training_master:
            with torch.no_grad():
                proto_action, value, log_prob = master_model.model.policy.forward(master_input)
                embedding = proto_action.cpu().numpy()[0]
        else:
            embedding = master_model.get_proto_action(master_input)

        # Create observations for each agent (car state + embedding)
        # Note: Each agent just needs its own state + embedding
        car1_obs = np.concatenate((car1_state, embedding))  # This should be 8-dimensional
        car3_obs = np.concatenate((car3_state, embedding))  # This should be 8-dimensional

        # Get actions for each agent if not done
        if not done_car1:
            car1_action = Agent.get_action(agent_model_car1, car1_obs, total_steps,
                                           experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
            car1_action_value = car1_action[0]
            actions_car1.append(car1_action_value)
        else:
            car1_action_value = 1  # SLOW if done

        if not done_car3:
            car3_action = Agent.get_action(agent_model_car3, car3_obs, total_steps,
                                           experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
            car3_action_value = car3_action[0]
            actions_car3.append(car3_action_value)
        else:
            car3_action_value = 1  # SLOW if done

        # Apply controls based on actions
        throttle1 = experiment.THROTTLE_FAST if car1_action_value == 0 else experiment.THROTTLE_SLOW
        throttle3 = experiment.THROTTLE_FAST if car3_action_value == 0 else experiment.THROTTLE_SLOW
        throttle_non_agent = experiment.FIXED_THROTTLE

        airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle1), experiment.CAR1_NAME)
        airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent), experiment.CAR2_NAME)
        airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle3), experiment.CAR3_NAME)
        airsim_manager.set_car_controls(airsim.CarControls(throttle=throttle_non_agent), experiment.CAR4_NAME)
        airsim_manager.set_car_controls(airsim.CarControls(throttle=experiment.THROTTLE_FAST), experiment.CAR5_NAME)

        # Calculate rewards and check completion for each agent
        if not done_car1:
            car1_pos = airsim_manager.get_car_position_and_speed(experiment.CAR1_NAME)
            init_pos1 = airsim_manager.get_car1_initial_position()
            current_pos1 = car1_pos["x"]
            desired_global1 = (experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                               if init_pos1[0] > 0 else experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance1 = abs(desired_global1 - init_pos1[0])
            traveled1 = abs(current_pos1 - init_pos1[0])

            collision1 = airsim_manager.collision_occurred(experiment.CAR1_NAME)
            reached_target1 = traveled1 >= required_distance1

            reward1 = experiment.STARVATION_REWARD
            if collision1:
                reward1 = experiment.COLLISION_REWARD
                done_car1 = True
            elif reached_target1:
                reward1 = experiment.REACHED_TARGET_REWARD
                done_car1 = True

            car1_total_reward += reward1
            rewards_car1.append(reward1)

            # Update car1 agent buffer
            if train_both or not training_master:
                # Check if buffer is full before adding
                if not agent_model_car1.rollout_buffer.full:
                    with torch.no_grad():
                        obs_tensor1 = torch.tensor(car1_obs, dtype=torch.float32).unsqueeze(0)
                        value1 = agent_model_car1.policy.predict_values(obs_tensor1)
                        dist1 = agent_model_car1.policy.get_distribution(obs_tensor1)
                        action_array1 = np.array([[car1_action_value]])
                        action_tensor1 = torch.tensor(action_array1, dtype=torch.long)
                        log_prob1 = dist1.log_prob(action_tensor1)

                    episode_start = (steps_counter == 1)
                    agent_model_car1.rollout_buffer.add(
                        car1_obs, action_array1, reward1, episode_start, value1, log_prob1
                    )

        if not done_car3:
            car3_pos = airsim_manager.get_car_position_and_speed(experiment.CAR3_NAME)
            init_pos3 = airsim_manager.get_car3_initial_position()
            current_pos3 = car3_pos["y"]
            desired_global3 = (experiment.CAR3_DESIRED_POSITION_OPTION_1[1]
                               if init_pos3[1] > 0 else experiment.CAR3_DESIRED_POSITION_OPTION_2[1])
            required_distance3 = abs(desired_global3 - init_pos3[1])
            traveled3 = abs(current_pos3 - init_pos3[1])

            collision3 = airsim_manager.collision_occurred(experiment.CAR3_NAME)
            reached_target3 = traveled3 >= required_distance3

            reward3 = experiment.STARVATION_REWARD
            if collision3:
                reward3 = experiment.COLLISION_REWARD
                done_car3 = True
            elif reached_target3:
                reward3 = experiment.REACHED_TARGET_REWARD
                done_car3 = True

            car3_total_reward += reward3
            rewards_car3.append(reward3)

            # Update car3 agent buffer
            if train_both or not training_master:
                if not agent_model_car3.rollout_buffer.full:
                    with torch.no_grad():
                        obs_tensor3 = torch.tensor(car3_obs, dtype=torch.float32).unsqueeze(0)
                        value3 = agent_model_car3.policy.predict_values(obs_tensor3)
                        dist3 = agent_model_car3.policy.get_distribution(obs_tensor3)
                        action_array3 = np.array([[car3_action_value]])
                        action_tensor3 = torch.tensor(action_array3, dtype=torch.long)
                        log_prob3 = dist3.log_prob(action_tensor3)

                    episode_start = (steps_counter == 1)
                    agent_model_car3.rollout_buffer.add(
                        car3_obs, action_array3, reward3, episode_start, value3, log_prob3
                    )



        # Update master buffer if needed
        if train_both or training_master:
            # Use combined reward for master model
            master_reward = reward1 + reward3
            episode_start = (steps_counter == 1)
            master_model.model.rollout_buffer.add(
                master_obs, embedding, master_reward, episode_start, value, log_prob
            )

        time.sleep(experiment.TIME_BETWEEN_STEPS)

    airsim_manager.pause_simulation()
    return car1_total_reward, car3_total_reward, actions_car1, actions_car3, steps_counter


def update_master_model(airsim_manager, master_model):
    """Replacement for train_master_and_reset_buffer"""
    with torch.no_grad():
        # Concatenate states from all cars (20-dimensional master state)
        last_master_obs = np.concatenate((
            airsim_manager.get_car1_state(),
            airsim_manager.get_car2_state(),
            airsim_manager.get_car3_state(),
            airsim_manager.get_car4_state(),
            airsim_manager.get_car5_state()
        ))
        last_master_tensor = torch.tensor(last_master_obs, dtype=torch.float32).unsqueeze(0)
        last_value = master_model.model.policy.predict_values(last_master_tensor)

    master_model.model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Master...")
    master_model.model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    master_model.model.rollout_buffer.reset()
    return last_master_tensor


def update_agent_model_car1(airsim_manager, master_model, agent_model, last_master_tensor):
    """Replacement for train_agent_and_reset_buffer_car1"""
    with torch.no_grad():
        car1_state = airsim_manager.get_car1_state()
        embedding = master_model.get_proto_action(last_master_tensor)
        agent_obs = np.concatenate((car1_state, embedding))
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
        last_value = agent_model.policy.predict_values(agent_obs_tensor)

    agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Car1 Agent...")
    agent_model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    agent_model.rollout_buffer.reset()


def update_agent_model_car3(airsim_manager, master_model, agent_model, last_master_tensor):
    """Replacement for train_agent_and_reset_buffer_car3"""
    with torch.no_grad():
        car3_state = airsim_manager.get_car3_state()
        embedding = master_model.get_proto_action(last_master_tensor)
        agent_obs = np.concatenate((car3_state, embedding))
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
        last_value = agent_model.policy.predict_values(agent_obs_tensor)

    agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
    print("Training Car3 Agent...")
    agent_model.learn(total_timesteps=25, reset_num_timesteps=False, log_interval=1)
    agent_model.rollout_buffer.reset()


def run_experiment(experiment_config):
    """
    Sets up the environment, models, and logging, then runs the training loop.
    """
    import airsim  # Making sure airsim is imported
    import time
    import copy

    p_agent_loss, p_master_loss, p_episode_counter = [], [], []
    airsim_manager = AirsimManager(experiment_config)
    master_model = MasterModel(embedding_size=4, airsim_manager=airsim_manager, experiment=experiment_config)

    if experiment_config.ROLE == "BOTH":
        # Create separate experiment configs for Car1 and Car3
        car1_experiment = copy.deepcopy(experiment_config)
        car1_experiment.ROLE = car1_experiment.CAR1_NAME  # Set to Car1

        car3_experiment = copy.deepcopy(experiment_config)
        car3_experiment.ROLE = car3_experiment.CAR3_NAME  # Set to Car3

        # Create separate environments and models for each agent with their specific role
        env_car1 = DummyVecEnv([lambda: Agent(car1_experiment, airsim_manager, master_model)])
        agent_model_car1 = Model(env_car1, car1_experiment).model

        env_car3 = DummyVecEnv([lambda: Agent(car3_experiment, airsim_manager, master_model)])
        agent_model_car3 = Model(env_car3, car3_experiment).model

        # Set up loggers
        base_path = experiment_config.EXPERIMENT_PATH
        agent1_logger = configure(os.path.join(base_path, "agent1_logs"), ["stdout", "csv", "tensorboard"])
        agent3_logger = configure(os.path.join(base_path, "agent3_logs"), ["stdout", "csv", "tensorboard"])
        master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])

        agent_model_car1.set_logger(agent1_logger)
        agent_model_car3.set_logger(agent3_logger)
        master_model.set_logger(master_logger)

        # Run training for both agents
        results = training_loop_both(p_agent_loss, p_master_loss, p_episode_counter,
                                     experiment_config, airsim_manager, master_model,
                                     agent_model_car1, agent_model_car3)

        # Save models
        if not experiment_config.ONLY_INFERENCE:
            plot_results(path=os.path.join(base_path, "agent1_logs"), all_rewards=results[5][0],
                         all_actions=results[6][0])
            plot_results(path=os.path.join(base_path, "agent3_logs"), all_rewards=results[5][1],
                         all_actions=results[6][1])
            plot_results(path=os.path.join(base_path, "master_logs"), all_rewards=results[5][0],
                         all_actions=results[6][0])

        agent_model_car1.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_agent_car1.pth")
        agent_model_car3.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_agent_car3.pth")
        master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")
    else:
        # Original single-agent code
        env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
        agent_model = Model(env, experiment_config).model

        base_path = experiment_config.EXPERIMENT_PATH
        agent_logger = configure(os.path.join(base_path, "agent_logs"), ["stdout", "csv", "tensorboard"])
        master_logger = configure(os.path.join(base_path, "master_logs"), ["stdout", "csv", "tensorboard"])
        agent_model.set_logger(agent_logger)
        master_model.set_logger(master_logger)

        # Here, use your existing training_loop function
        from training_loop import training_loop  # Import the existing function
        results = training_loop(p_agent_loss, p_master_loss, p_episode_counter,
                                experiment_config, env, agent_model, master_model)

        if not experiment_config.ONLY_INFERENCE:
            plot_results(path=os.path.join(base_path, "agent_logs"), all_rewards=results[5], all_actions=results[6])
            plot_results(path=os.path.join(base_path, "master_logs"), all_rewards=results[5], all_actions=results[6])

        agent_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_agent.pth")
        master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")

    print("Models saved.")
    if experiment_config.ROLE != "BOTH":
        env.close()