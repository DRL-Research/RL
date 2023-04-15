
from datetime import datetime
import airsim
import tensorflow as tf
from RL import RL
import numpy as np

if __name__ == '__main__':

    log_directory = "exp2"

    # Create an airsim client instance:
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True, "Car1")
    airsim_client.enableApiControl(True, "Car2")

    # initialize params:
    # set car 1:
    print("start testing:")
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    airsim_client.setCarControls(car_controls, "Car1")
    # set car 2:
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    airsim_client.setCarControls(car_controls, "Car2")

    log_dir = log_directory+"/rewards/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.summary.create_file_writer(log_dir)

    # define object of RL
    # Define here the parameters of the experiment:
    max_episodes = 100
    max_steps = 500
    only_local = True
    two_cars_local = True  # two cars using the local network / only one
    RL = RL(learning_rate=0.003,
            verbose=0,
            with_per=True,
            two_cars_local=two_cars_local,
            log_directory=log_directory)


    """
    Change to the desired .h5 weights file, comment out the next line on first run & runs that did not converge.
    Do not override a converged run's weights file! Load it but save under another path so you'll be able to
    revert back to it in case the following run did not converge. E.g.: <...weights_1.h5>, <...weights_2.h5>
    """
    RL.local_network.load_weights('exp1/weights/12_sixth_right.h5')


    # Start the experiment:
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    for episode in range(max_episodes):

        value = np.random.randint(3, size=(1, 1))

        if not two_cars_local:
            if value == 0:
                car2speed = 0.65
            if value == 1:
                car2speed = 0.73
            if value == 2:
                car2speed = 0.8
            print(car2speed)

        episode_counter += 1
        episode_sum_of_rewards = 0
        print(f"@@@@ Episode #{episode} @@@@")

        for step in range(max_steps):

            steps_counter += 1
            # perform a step in the environment, and get feedback about collision and updated controls:
            if not two_cars_local:
                done, reached_target, updated_controls, reward = RL.step_only_local(airsim_client, steps_counter)
            else:
                done, reached_target, updated_controls_car1, updated_controls_car2, reward = RL.step_only_local_2_cars(airsim_client, steps_counter)

            # log
            episode_sum_of_rewards += reward

            if done or reached_target:
                # reset the environment in case of a collision:
                airsim_client.reset()
                # log
                with tensorboard.as_default():
                    tf.summary.scalar('episode_sum_of_rewards', episode_sum_of_rewards, step=episode_counter)
                if done:
                    collision_counter += 1

                break


            if two_cars_local:
                airsim_client.setCarControls(updated_controls_car1, "Car1")
                airsim_client.setCarControls(updated_controls_car2, "Car2")
            else:
                # update controls of Car 1 based on the RL algorithm:
                airsim_client.setCarControls(updated_controls, "Car1")
                # airsim_client = RL.updateControls(airsim_client, ["Car1"], [updated_controls])
                # update controls of Car 2:
                car_controls = airsim.CarControls()
                car_controls.throttle = car2speed
                airsim_client.setCarControls(car_controls, "Car2")

    """
    For runs which load prior converged runs' weights, update the save path in order not to override the saved weights.
    E.g.: <...weights_1.h5>, <...weights_2.h5>
    """
    RL.local_network.save_weights('exp2/weights/first_run.h5')

    print("@@@@ Run Ended @@@@")
    print(collision_counter)
