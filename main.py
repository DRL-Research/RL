import tensorflow as tf
import os
from config.ExperimentParamsAccordingToConfig import ExperimentParamsAccordingToConfig


def main():

    experiment_params = ExperimentParamsAccordingToConfig()
    rl = experiment_params.rl
    airsim_client = experiment_params.airsim_client

    # if global_experiment:
    #     rl.local_and_global_network.load_weights(load_weight)
    # else:

    # comment the next line if this is the first run of a new experiment
    # rl.local_network.load_weights(experiment_params.load_weight_directory)

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
                rl.alternate_car = 2
                rl.alternate_training_network = rl.copy_network(rl.local_network)
            else:
                rl.alternate_car = 1
                rl.alternate_training_network = rl.copy_network(rl.local_network)

        for step in range(experiment_params.max_steps):

            steps_counter += 1
            if experiment_params.global_experiment:
                done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_with_global(airsim_client, steps_counter)
            else:
                done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_local_2_cars(airsim_client, steps_counter)

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

    # save network weights
    if not os.path.exists("experiments/" + experiment_params.experiment_id + "/weights"):
        os.makedirs("experiments/" + experiment_params.experiment_id + "/weights")
    rl.local_network.save_weights("experiments/" + experiment_params.experiment_id + "/weights" + experiment_params.weights_to_save_id)

    print("---- Run Ended ----")
    print("The amount of collisions:")
    print(collision_counter)


if __name__ == "__main__":
    main()
