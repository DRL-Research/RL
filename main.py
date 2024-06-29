from config import MAX_EPISODES, MAX_STEPS, EXPERIMENT_DATE_TIME, TRAIN_OPTION, ALTERNATE_TRAINING_EPISODE_AMOUNT
from utils.NN_utils import *
from utils.airsim_manager import AirsimManager
from utils.logger import Logger
from utils.rl import RL

if __name__ == "__main__":

    # TODO: mission from chat with Gilad
    # TODO: check that weights really do not change when freezing master for example.
    # TODO: print the important settings of this experiment
    # TODO: add an ability to change the experiment name to a more indicative name

    logger = Logger(EXPERIMENT_ID, EXPERIMENT_DATE_TIME)
    airsim = AirsimManager()
    rl = RL(logger=logger, airsim=airsim)

    # Initialize counters for the experiment
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0

    # Main loop for each episode
    for episode in range(MAX_EPISODES):
        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@ Episode {episode} @")

        rl.current_trajectory = []

        # Alternate training between master and agent layers
        if episode_counter % ALTERNATE_TRAINING_EPISODE_AMOUNT == 0:
            rl.freeze_master = not rl.freeze_master
            alternate_master_and_agent_training(rl.network, rl.freeze_master)

        # Inner loop for each step in an episode
        for step in range(MAX_STEPS):
            steps_counter += 1

            # Perform a step
            states, actions, next_states, collision_occurred, reached_target, reward = rl.step()
            # print(f"actions selected: {actions}")

            # Add current step to trajectory (once for car1 action, and once for car2 (both have same reward))
            rl.current_trajectory.append((states[0], actions[0], next_states[0], reward))
            rl.current_trajectory.append((states[1], actions[1], next_states[1], reward))

            if TRAIN_OPTION == 'step':
                step_loss = rl.train_trajectory(train_only_last_step=True)
                logger.log("loss_per_step", steps_counter, step_loss)

            # Update the sum of rewards
            episode_sum_of_rewards += reward

            # Handle collision or target reached scenarios
            if collision_occurred or reached_target:
                airsim.reset_cars_to_initial_positions()

                # Log the episode's sum of rewards
                logger.log("episode_sum_of_rewards", episode_counter, episode_sum_of_rewards)

                # Update collision count and break the loop
                if collision_occurred:
                    collision_counter += 1
                break

        # Update epsilon greedy after each trajectory
        rl.epsilon *= rl.epsilon_decay
        print(rl.epsilon)

        if TRAIN_OPTION == 'trajectory':
            trajectory_loss = rl.train_trajectory(False, episode_counter)
            logger.log("loss_per_trajectory", episode_counter, trajectory_loss)

        rl.trajectories.append(rl.current_trajectory)

        # TODO: complete train_batch_of_trajectories
        # if TRAIN_OPTION == 'batch_of_trajectories':
        #     if episode_counter % batch_size:
        #         agent.train_batch_of_trajectories(rl.trajectories)

    # Save the network weights after the experiment ended (name of weights file is set in config)
    save_network_weights(rl.network)

    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)
