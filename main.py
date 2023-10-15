from datetime import datetime
import airsim
import tensorflow as tf
from RL import RL
import os
import json


def init_log_directories(log_directory):
    log_directory = "experiments/" + log_directory
    save_weights_directory = log_directory + "/weights"
    log_dir = log_directory + "/rewards/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.summary.create_file_writer(log_dir)
    return save_weights_directory, tensorboard
def init_airsim_client():
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True, "Car1")
    airsim_client.enableApiControl(True, "Car2")
    #
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    airsim_client.setCarControls(car_controls, "Car1")
    # set car 2:
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    airsim_client.setCarControls(car_controls, "Car2")
    return airsim_client
def load_airsim_settings(car1_location, car2_location):
    # get settings tamplate from project directory airsim_settings/settings.json
    directory_path = 'airsim_settings/'
    file_name = 'settings.json'
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
        print(json_data)

    # modify the data
    modified_settings = json_data
    modified_settings["Vehicles"]["Car1"]['X'] = car1_location[0]
    modified_settings["Vehicles"]["Car1"]['Y'] = car1_location[1]
    modified_settings["Vehicles"]["Car2"]['X'] = car2_location[0]
    modified_settings["Vehicles"]["Car2"]['Y'] = car2_location[1]

    # write the modified json file to documents/airsim folder
    output_directory = os.path.expanduser('~/Documents/AirSim')
    output_file_name = 'settings.json'
    output_file_path = os.path.join(output_directory, output_file_name)
    with open(output_file_path, 'w') as json_file:
        json.dump(modified_settings, json_file, indent=4)


# define parameters of the experiment:
log_directory = "exp3"
load_weight = 'experiments/exp2/weights/12_sixth_right.h5'
save_weight = '/1_first_left.h5'

car1_location = [-20, 0]
car2_location = [0, -20]

max_episodes = 100
max_steps = 500
alternate_training = True
alternate_car = 1
rl = RL(learning_rate=0.003,
        verbose=0,
        with_per=True,
        two_cars_local=True,
        log_directory=log_directory,
        alternate_training=alternate_training,
        alternate_car=alternate_car)


# load airsim settings for the experiment
load_airsim_settings(car1_location, car2_location)
print("settings ready, press play in unreal simulation.")

# init loss, reward, weight, tensorboard directories.
save_weights_directory, tensorboard = init_log_directories(log_directory)

# init airsim client instance + enable control (currently 2 cars):
airsim_client = init_airsim_client()



"""
Change to the desired .h5 weights file, comment out the next line on first run & runs that did not converge.
Do not override a converged run's weights file! Load it but save under another path so you'll be able to
revert back to it in case the following run did not converge. E.g.: <...weights_1.h5>, <...weights_2.h5>
"""
rl.local_network.load_weights(load_weight)

# Start the experiment:
collision_counter = 0
episode_counter = 0
steps_counter = 0
for episode in range(max_episodes):

    episode_counter += 1
    episode_sum_of_rewards = 0
    print(f"@@@@ Episode #{episode} @@@@")

    if episode_counter % 20 == 0:
        if alternate_car == 1:
            rl.alternate_car = 2
            rl.alternate_training_network = rl.copy_network(rl.local_network)
        else:
            rl.alternate_car = 1
            rl.alternate_training_network = rl.copy_network(rl.local_network)

    for step in range(max_steps):

        steps_counter += 1
        # perform a step in the environment, and get feedback about collision and updated controls:
        done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_only_local_2_cars(
            airsim_client, steps_counter)

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

        airsim_client.setCarControls(updated_controls_car1, "Car1")
        airsim_client.setCarControls(updated_controls_car2, "Car2")


"""
For runs which load prior converged runs' weights, update the save path in order not to override the saved weights.
E.g.: <...weights_1.h5>, <...weights_2.h5>
"""
if not os.path.exists(save_weights_directory):
    os.makedirs(save_weights_directory)
rl.local_network.save_weights(save_weights_directory + save_weight)

print("@@@@ Run Ended @@@@")
print(collision_counter)





