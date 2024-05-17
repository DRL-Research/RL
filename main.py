from config import MAX_EPISODES, MAX_STEPS, EXPERIMENT_DATE_TIME, CAR1_NAME, CAR2_NAME
from utils.NN_utils import *
from utils.airsim_manager import AirsimManager
from utils.logger import Logger
from utils.rl import RL

if __name__ == "__main__":

    logger = Logger(EXPERIMENT_ID, EXPERIMENT_DATE_TIME)
    airsim = AirsimManager()
    rl = RL(logger=logger, airsim=airsim)

    # Initialize counters for the experiment
    collision_counter = 0
    episode_counter = -1
    steps_counter = 0

    # Main loop for each episode
    for episode in range(MAX_EPISODES):
        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@ Episode {episode} @")

        # alternating training
        # if episode_counter % ALTERNATE_TRAINING_EPISODE_AMOUNT == 0:
        #     rl_agent.local_network_car2 = copy_network(rl_agent.local_network_car1)

        # Inner loop for each step in an episode
        for step in range(MAX_STEPS):
            steps_counter += 1

            # Perform a step
            collision_occurred, reached_target, reward = rl.step(steps_counter)

            # Update the sum of rewards
            episode_sum_of_rewards += reward

            # Handle collision or target reached scenarios
            if collision_occurred or reached_target:
                airsim.reset_cars_to_initial_positions()

                # Log the episode's sum of rewards
                logger.log("episode_sum_of_rewards", episode_sum_of_rewards, episode_counter)

                # Update collision count and break the loop
                if collision_occurred:
                    collision_counter += 1
                break

    # Save the network weights after the experiment ended (name of weights file is set in config)
    save_network_weights(rl.network)

    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)
