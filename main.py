import tensorflow as tf
from ExperimentParams import ExperimentParams
from utils.NN_utils import *
from utils.airsim_utils import *


def main():

    # Initialize experiment parameters and RL agent
    experiment_params = ExperimentParams()
    RL_Agent = experiment_params.RL_Agent
    airsim_client = experiment_params.airsim_client

    # Initialize counters for the experiment
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0

    # Main loop for each episode
    for episode in range(experiment_params.max_episodes):
        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@ Episode {episode} @")

        # Check for alternating training episodes
        if episode_counter % experiment_params.alternate_training_episode_amount == 0:
            RL_Agent.local_network_car2 = copy_network(RL_Agent.local_network_car1)

        # Inner loop for each step in an episode
        for step in range(experiment_params.max_steps):
            steps_counter += 1

            # Perform a step with the RL agent
            collision_happened, reached_target, updated_controls_car1, \
                updated_controls_car2, reward = RL_Agent.step_local_2_cars(airsim_client, steps_counter)

            # Update the sum of rewards
            episode_sum_of_rewards += reward

            # Handle collision or target reached scenarios
            if collision_happened or reached_target:
                reset_cars_to_initial_positions(experiment_params, airsim_client)

                # Log the episode's sum of rewards
                with experiment_params.tensorboard.as_default():
                    tf.summary.scalar('episode_sum_of_rewards', episode_sum_of_rewards, step=episode_counter)

                # Update collision count and break the loop
                if collision_happened:
                    collision_counter += 1
                break

            # Set controls for the cars
            airsim_client.setCarControls(updated_controls_car1, "Car1")
            airsim_client.setCarControls(updated_controls_car2, "Car2")

    # Save the network weights after the experiment
    save_network_weights(experiment_params, RL_Agent)

    # Log experiment completion and collision count
    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)


if __name__ == "__main__":
    main()
