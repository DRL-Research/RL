import tensorflow as tf
from utils.NN_utils import *
from utils.airsim_utils import *
from utils.rl_agent_utils import RLAgent
from utils.tensorboard_utils import TensorBoard
from config import EXPERIMENT_ID, GLOBAL_EXPERIMENT, ALTERNATE_TRAINING_EPISODE_AMOUNT, MAX_EPISODES, \
    MAX_STEPS, EXPERIMENT_DATE_TIME

# TODO: go through all code and replace "Car1" and "Car2" to CAR1_NAME and CAR2_NAME

if __name__ == "__main__":

    tensorboard = TensorBoard(EXPERIMENT_ID, EXPERIMENT_DATE_TIME).tensorboard
    rl_agent = RLAgent(tensorboard=tensorboard)
    airsim_client_handler = AirsimClientHandler()
    airsim_client = airsim_client_handler.airsim_client

    # Initialize counters for the experiment
    collision_counter = 0
    episode_counter = -1
    steps_counter = 0

    # Main loop for each episode
    for episode in range(MAX_EPISODES):
        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@ Episode {episode} @")

        # Check for alternating training episodes
        if episode_counter % ALTERNATE_TRAINING_EPISODE_AMOUNT == 0:
            rl_agent.local_network_car2 = copy_network(rl_agent.local_network_car1)

        # Inner loop for each step in an episode
        for step in range(MAX_STEPS):
            steps_counter += 1

            # Perform a step with the RL agent
            collision_happened, reached_target, updated_controls_car1, \
                updated_controls_car2, reward = rl_agent.step_local_2_cars(airsim_client_handler, steps_counter)

            # Update the sum of rewards
            episode_sum_of_rewards += reward

            # Handle collision or target reached scenarios
            if collision_happened or reached_target:
                airsim_client_handler.reset_cars_to_initial_positions()

                # Log the episode's sum of rewards
                with tensorboard.as_default():
                    tf.summary.scalar('episode_sum_of_rewards', episode_sum_of_rewards, step=episode_counter)

                # Update collision count and break the loop
                if collision_happened:
                    collision_counter += 1
                break

            # Set controls for the cars
            airsim_client.setCarControls(updated_controls_car1, "Car1")
            airsim_client.setCarControls(updated_controls_car2, "Car2")

    # Save the network weights after the experiment
    save_network_weights(rl_agent)

    # Log experiment completion and collision count
    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)
