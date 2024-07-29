from utils.NN_utils import *
from utils.airsim_manager import AirsimManager
from utils.logger import Logger
from utils.rl import RL


def training_loop_ido(config):
    # classes init
    logger = Logger(config)
    airsim = AirsimManager(config)
    nn_handler = NN_handler(config)
    rl = RL(config, logger, airsim, nn_handler)

    if config.ONLY_INFERENCE:
        rl.network = nn_handler.load_weights_to_network(rl.network)

    # Initialize counters for the experiment
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0

    # Main loop for each episode
    for episode in range(config.MAX_EPISODES):

        # choose constant action for car2 for whole trajectory
        if np.random.randint(2) == 1:
            config.CAR2_CONSTANT_ACTION = 0
        else:
            config.CAR2_CONSTANT_ACTION = 1

        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@ Episode {episode} @")

        rl.current_trajectory = []

        # Inner loop for each step in an episode
        for step in range(config.MAX_STEPS):
            steps_counter += 1

            # Perform a step
            state, action, next_state, collision_occurred, reached_target, reward = rl.step_agent_only()

            # Add current step to trajectory (once for car1 action, and once for car2 (both have same reward))
            if collision_occurred or reached_target:
                done = True
            else:
                done = False
            rl.current_trajectory.append((state, action, reward, next_state, done))

            # if config.TRAIN_OPTION == 'step':
            #     step_loss = rl.train_trajectory(train_only_last_step=True, episode_counter=episode_counter)
            #     logger.log_scaler("loss_per_step", steps_counter, step_loss)

            # Update the sum of rewards
            episode_sum_of_rewards += reward

            # Handle collision or target reached scenarios
            if collision_occurred or reached_target:
                airsim.reset_cars_to_initial_positions()

                # Log the episode's sum of rewards
                logger.log_scaler("episode_sum_of_rewards", episode_counter, episode_sum_of_rewards)

                # Update collision count and break the loop
                if collision_occurred:
                    collision_counter += 1
                break

        # Update epsilon greedy after each trajectory
        rl.updateEpsilon()
        print(f"epsilon value: {rl.epsilon}")

        rl.memory.append(rl.current_trajectory)

        if config.TRAIN_OPTION == 'trajectory' and not config.ONLY_INFERENCE:

            # trajectory_loss = rl.train_trajectory(train_only_last_step=False, episode_counter=episode_counter)
            trajectory_loss = rl.replay()
            rl.update_target_model()

            if trajectory_loss is not None:
                logger.log_scaler("loss_per_trajectory", episode_counter, trajectory_loss)

            if config.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY:
                logger.log_same_action_selected_in_trajectory()

    # Save the network weights after the experiment ended (name of weights file is set in config)
    nn_handler.save_network_weights(rl.network)

    # reset cars to initial settings file position, before the next experiment
    airsim.reset_cars_to_initial_settings_file_positions()

    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)



# TODO: see diff from training_loop_agent_only()
# def training_loop(config):
#
#     # classes init
#     logger = Logger(config)
#     airsim = AirsimManager(config)
#     nn_handler = NN_handler(config)
#     rl = RL(config, logger, airsim, nn_handler)
#
#     # Initialize counters for the experiment
#     collision_counter = 0
#     episode_counter = 0
#     steps_counter = 0
#
#     # Main loop for each episode
#     for episode in range(config.MAX_EPISODES):
#         episode_counter += 1
#         episode_sum_of_rewards = 0
#         print(f"@ Episode {episode} @")
#
#         rl.current_trajectory = []
#
#         if config.COPY_CAR1_NETWORK_TO_CAR2 and episode_counter % config.COPY_CAR1_NETWORK_TO_CAR2_EPISODE_AMOUNT == 0:
#             rl.copy_network()
#
#         # config.ALTERNATE_MASTER_AND_AGENT_TRAINING
#         if config.ALTERNATE_MASTER_AND_AGENT_TRAINING and episode_counter % config.ALTERNATE_TRAINING_EPISODE_AMOUNT == 0:
#             rl.freeze_master = not rl.freeze_master
#             nn_handler.alternate_master_and_agent_training(rl.network, rl.freeze_master)
#
#         # Inner loop for each step in an episode
#         for step in range(config.MAX_STEPS):
#             steps_counter += 1
#
#             # Perform a step
#             states, actions, next_states, collision_occurred, reached_target, reward = rl.step()
#
#             # Add current step to trajectory (once for car1 action, and once for car2 (both have same reward))
#             rl.current_trajectory.append((states[0], actions[0], next_states[0], reward))
#             rl.current_trajectory.append((states[1], actions[1], next_states[1], reward))
#
#             if config.TRAIN_OPTION == 'step':
#                 step_loss = rl.train_trajectory(train_only_last_step=True, episode_counter=episode_counter)
#                 logger.log_scaler("loss_per_step", steps_counter, step_loss)
#
#             # Update the sum of rewards
#             episode_sum_of_rewards += reward
#
#             # Handle collision or target reached scenarios
#             if collision_occurred or reached_target:
#                 airsim.reset_cars_to_initial_positions()
#
#                 # Log the episode's sum of rewards
#                 logger.log_scaler("episode_sum_of_rewards", episode_counter, episode_sum_of_rewards)
#
#                 # Update collision count and break the loop
#                 if collision_occurred:
#                     collision_counter += 1
#                 break
#
#         # Update epsilon greedy after each trajectory
#         rl.epsilon *= rl.epsilon_decay
#
#         if config.TRAIN_OPTION == 'trajectory':
#             trajectory_loss = rl.train_trajectory(train_only_last_step=False, episode_counter=episode_counter)
#             logger.log_scaler("loss_per_trajectory", episode_counter, trajectory_loss)
#             if config.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY:
#                 logger.log_same_action_selected_in_trajectory()
#
#         rl.trajectories.append(rl.current_trajectory)
#
#         # TODO: complete train_batch_of_trajectories
#         # if config.TRAIN_OPTION == 'batch_of_trajectories':
#         #     if episode_counter % batch_size:
#         #         agent.train_batch_of_trajectories(rl.trajectories)
#
#     # Save the network weights after the experiment ended (name of weights file is set in config)
#     nn_handler.save_network_weights(rl.network)
#
#     print("---- Experiment Ended ----")
#     print("The amount of collisions:")
#     print(collision_counter)


# def training_loop_agent_only(config):
#
#     # classes init
#     logger = Logger(config)
#     airsim = AirsimManager(config)
#     nn_handler = NN_handler(config)
#     rl = RL(config, logger, airsim, nn_handler)
#
#     if config.LOAD_WEIGHT_DIRECTORY is not None:
#         rl.network = nn_handler.load_weights_to_network(rl.network)
#
#     # Initialize counters for the experiment
#     collision_counter = 0
#     episode_counter = 0
#     steps_counter = 0
#
#     # Main loop for each episode
#     for episode in range(config.MAX_EPISODES):
#
#         # choose constant action for car2 for whole trajectory
#         if np.random.randint(2) == 1:
#             config.CAR2_CONSTANT_ACTION = 0
#         else:
#             config.CAR2_CONSTANT_ACTION = 1
#
#         episode_counter += 1
#         episode_sum_of_rewards = 0
#         print(f"@ Episode {episode} @")
#
#         rl.current_trajectory = []
#
#         # Inner loop for each step in an episode
#         for step in range(config.MAX_STEPS):
#             steps_counter += 1
#
#             # Perform a step
#             states, actions, next_states, collision_occurred, reached_target, reward = rl.step_agent_only()
#
#             # Add current step to trajectory (once for car1 action, and once for car2 (both have same reward))
#             rl.current_trajectory.append((states, actions, next_states, reward))
#
#             # if config.TRAIN_OPTION == 'step':
#             #     step_loss = rl.train_trajectory(train_only_last_step=True, episode_counter=episode_counter)
#             #     logger.log_scaler("loss_per_step", steps_counter, step_loss)
#
#             # Update the sum of rewards
#             episode_sum_of_rewards += reward
#
#             # Handle collision or target reached scenarios
#             if collision_occurred or reached_target:
#                 airsim.reset_cars_to_initial_positions()
#
#                 # Log the episode's sum of rewards
#                 logger.log_scaler("episode_sum_of_rewards", episode_counter, episode_sum_of_rewards)
#
#                 # Update collision count and break the loop
#                 if collision_occurred:
#                     collision_counter += 1
#                 break
#
#         # Update epsilon greedy after each trajectory
#         rl.epsilon *= rl.epsilon_decay
#         print(f"epsilon value: {rl.epsilon}")
#
#         if config.TRAIN_OPTION == 'trajectory' and not config.ONLY_INFERENCE:
#             trajectory_loss = rl.train_trajectory(train_only_last_step=False, episode_counter=episode_counter)
#             logger.log_scaler("loss_per_trajectory", episode_counter, trajectory_loss)
#             if config.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY:
#                 logger.log_same_action_selected_in_trajectory()
#
#         rl.trajectories.append(rl.current_trajectory)
#
#         # TODO: complete train_batch_of_trajectories
#         # if config.TRAIN_OPTION == 'batch_of_trajectories':
#         #     if episode_counter % batch_size:
#         #         agent.train_batch_of_trajectories(rl.trajectories)
#
#     # Save the network weights after the experiment ended (name of weights file is set in config)
#     nn_handler.save_network_weights(rl.network)
#
#     # reset cars to initial settings file position, before the next experiment
#     airsim.reset_cars_to_initial_settings_file_positions()
#
#     print("---- Experiment Ended ----")
#     print("The amount of collisions:")
#     print(collision_counter)

