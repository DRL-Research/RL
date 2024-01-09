import tensorflow as tf
from config.ExperimentParamsAccordingToConfig import ExperimentParamsAccordingToConfig
from utils.NN_utils import *


def main():

    experiment_params = ExperimentParamsAccordingToConfig()
    RL_Agent = experiment_params.RL_Agent
    airsim_client = experiment_params.airsim_client

    # if global_experiment:
    #     RL_Agent.local_and_global_network.load_weights(load_weight)
    # else:

    # comment the next line if this is the first run of a new experiment
    # RL_Agent.local_network.load_weights(experiment_params.load_weight_directory)

    # Start the experiment:
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    for episode in range(experiment_params.max_episodes):

        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@ Episode {episode} @")

        if episode_counter % 20 == 0:
            if experiment_params.alternate_car == 1:
                RL_Agent.alternate_car = 2
                RL_Agent.alternate_training_network = copy_network(RL_Agent.local_network)
            else:
                RL_Agent.alternate_car = 1
                RL_Agent.alternate_training_network = copy_network(RL_Agent.local_network)

        for step in range(experiment_params.max_steps):

            steps_counter += 1
            if experiment_params.global_experiment:
                collision_happened, reached_target, updated_controls_car1, updated_controls_car2, reward = RL_Agent.step_with_global(airsim_client, steps_counter)
            else:
                collision_happened, reached_target, updated_controls_car1, updated_controls_car2, reward = RL_Agent.step_local_2_cars(airsim_client, steps_counter)

            # log
            episode_sum_of_rewards += reward

            if collision_happened or reached_target:
                # reset the cars to starting position
                airsim_client.reset()
                # log
                with experiment_params.tensorboard.as_default():
                    tf.summary.scalar('episode_sum_of_rewards', episode_sum_of_rewards, step=episode_counter)
                if collision_happened:
                    collision_counter += 1

                break

            airsim_client.setCarControls(updated_controls_car1, "Car1")
            airsim_client.setCarControls(updated_controls_car2, "Car2")

    # save network weights
    save_network_weights(experiment_params, RL_Agent)

    # Experiment Ended
    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)


if __name__ == "__main__":
    main()
