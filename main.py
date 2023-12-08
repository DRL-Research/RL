from datetime import datetime
import airsim
import tensorflow as tf
from RL import RL
import os
import json
from config.ExperimentParamsAccordingToConfig import ExperimentParamsAccordingToConfig



def main():

    experiment_params = ExperimentParamsAccordingToConfig()

    """
    Change to the desired .h5 weights file, comment out the next line on first run & runs that did not converge.
    Do not override a converged run's weights file! Load it but save under another path so you'll be able to
    revert back to it in case the following run did not converge. E.g.: <...weights_1.h5>, <...weights_2.h5>
    """
    # if global_experiment:
    #     rl.local_and_global_network.load_weights(load_weight)
    # else:
    #     rl.local_network.load_weights(load_weight)

    rl = experiment_params.rl
    airsim_client = experiment_params.airsim_client


    # Start the experiment:
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    for episode in range(experiment_params.max_episodes):

        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@@@@ Episode #{episode} @@@@")

        if episode_counter % 20 == 0:
            if experiment_params.alternate_car == 1:
                rl.alternate_car = 2
                rl.alternate_training_network = rl.copy_network(rl.local_network)
            else:
                rl.alternate_car = 1
                rl.alternate_training_network = rl.copy_network(rl.local_network)

        for step in range(experiment_params.max_steps):

            steps_counter += 1
            # perform a step in the environment, and get feedback about collision and updated controls:
            if experiment_params.global_experiment:
                done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_with_global(airsim_client, steps_counter)
            else:
                done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_only_local_2_cars(airsim_client, steps_counter)

            # log
            episode_sum_of_rewards += reward

            if done or reached_target:
                # reset the environment in case of a collision:
                airsim_client.reset()
                # log
                with experiment_params.tensorboard.as_default():
                    tf.summary.scalar('episode_sum_of_rewards', episode_sum_of_rewards, step=episode_counter)
                if done:
                    collision_counter += 1

                break

            airsim_client.setCarControls(updated_controls_car1, "Car1")
            airsim_client.setCarControls(updated_controls_car2, "Car2")


    """
    For runs which load prior converged runs' weights, update the save path in order not to override the saved weights.
    E.g.: <...weights_1.h5>, <...weights_2.h5>
    """
    if not os.path.exists(experiment_params.save_weights_directory):
        os.makedirs(experiment_params.save_weights_directory)
    rl.local_network.save_weights(experiment_params.save_weights_directory + experiment_params.save_weight_current_directory)

    print("@@@@ Run Ended @@@@")
    print(collision_counter)


if __name__ == "__main__":
    main()
