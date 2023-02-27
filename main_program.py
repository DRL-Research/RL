
from datetime import datetime
import airsim
import tensorflow as tf
from RL import RL
import numpy as np

if __name__ == '__main__':

    # activate tensor board (tensorboard)
    # in Terminal, in pycharm, run: python -m tensorboard.main --logdir=old_logs/

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

    log_dir = "logs/rewards/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.summary.create_file_writer(log_dir)



    # define object of RL
    # Define here the parameters of the experiment:
    RL = RL.RL(learning_rate=0.003,
               verbose=0,
               with_per=True)
    max_episodes = 400
    max_steps = 500
    only_local = True
    # frequency = 40  # the higher the "frequency" -> the slower the samples are taken.






    # Start the experiment:
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    for episode in range(max_episodes):

        value = np.random.randint(2, size=(1,1))

        if value == 0:
            car2speed = 0.65
        if value == 1:
            car2speed = 0.75


        print(car2speed)

        episode_counter += 1
        episode_sum_of_rewards = 0
        print("new episode")

        for step in range(max_steps):

            steps_counter += 1
            # perform a step in the environment, and get feedback about collision and updated controls:
            if only_local:
                done, reached_target, updated_controls, reward = RL.step_only_local(airsim_client, steps_counter)
            else:
                done, reached_target, updated_controls, reward = RL.step_with_global(airsim_client, steps_counter)

            # log
            episode_sum_of_rewards += reward

            if done or reached_target:
                # reset the environment in case of a collision:
                airsim_client.reset()
                # log
                # if I want using avg reward:
                # if episode > 98:
                    # Check if solved
                    # average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                #
                with tensorboard.as_default():
                    tf.summary.scalar('episode_sum_of_rewards', episode_sum_of_rewards, step=episode_counter)
                if done:
                    collision_counter += 1

                break

            # update controls of Car 1 based on the RL algorithm:
            airsim_client.setCarControls(updated_controls, "Car1")


            # update controls of Car 2:
            car_controls = airsim.CarControls()
            car_controls.throttle = car2speed
            airsim_client.setCarControls(car_controls, "Car2")


    print(collision_counter)



