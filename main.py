import copy
from datetime import datetime
import tensorflow as tf
import numpy as np
from RL import RL
import airsim
import os
import h5py

# functions to print the h5 file
"""functions to print the h5 file (function opens the HDF5 file using h5py.
File in read mode and then prints the keys in the file (top-level groups))"""


def print_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        print("Keys in the HDF5 file:")
        print(list(f.keys()))

        print("\n--- Contents of the HDF5 file ---")
        print_h5_group(f)


"""function to recursively print the contents of each group, including nested groups and datasets"""


def print_h5_group(group, indent=0):
    for key in group.keys():
        print(' ' * indent + key)

        if isinstance(group[key], h5py.Group):
            print_h5_group(group[key], indent + 4)
        elif isinstance(group[key], h5py.Dataset):
            print_dataset(group[key], indent + 4)


"""function is used to print information about individual datasets, including their shape, data type, and the actual 
data values. """


def print_dataset(dataset, indent=0):
    print(' ' * indent + "Shape:", dataset.shape)
    print(' ' * indent + "Data type:", dataset.dtype)
    print(' ' * indent + "Data:")
    print(dataset[()])


def load_global_network_weights_from_local_network(model, load_weight):
    # Load local weights from the saved model:
    rl.local_network.load_weights(load_weight)
    pretrained_model = rl.local_network  # includes local trained weights
    print(pretrained_model.summary())

    # Update the first local layer (local_16_layer) - (input: 9 local neurons + 5 embedding neurons)
    local_16_layer_weights_in_global_network = model.get_layer("local_16_layer").get_weights()[0]
    local_16_layer_weights_in_global_network[-9:, :] = pretrained_model.get_layer("16_layer").get_weights()[0]
    new_weights = [local_16_layer_weights_in_global_network, model.get_layer("local_16_layer").get_weights()[1]]
    model.get_layer("local_16_layer").set_weights(new_weights)

    # Define layers to get weights from (pretrained local model):
    layers_to_get_weights_from = ["8_layer", "2_layer"]
    pretrained_layers = [pretrained_model.get_layer(layer_name) for layer_name in layers_to_get_weights_from]

    # Define layers to set weights to (global model):
    layers_to_set_weights_to = ["local_8_layer", "local_2_layer"]
    layers_to_update_in_global_network = [model.get_layer(layer_name) for layer_name in layers_to_set_weights_to]

    # Set the weights to the desired layers
    for layer, pretrained_layer in zip(layers_to_update_in_global_network, pretrained_layers):
        print(pretrained_layer.get_weights())
        layer.set_weights(pretrained_layer.get_weights())




if __name__ == '__main__':

    log_directory = "exp3"
    load_weight = 'exp3/weights/11_fifth_right.h5'
    save_weights_directory = log_directory + "/weights"
    save_weight = '/12_sixth_right.h5'
    weights_from_local_network = False

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

    log_dir = log_directory + "/rewards/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.summary.create_file_writer(log_dir)

    # define object of RL
    # Define here the parameters of the experiment:
    max_episodes = 100
    max_steps = 500
    only_local = False
    two_cars_local = True  # two cars using the local network / only one
    alternate_training = True
    alternate_car = 1
    rl = RL(learning_rate=0.003,
            verbose=0,
            with_per=True,
            two_cars_local=two_cars_local,
            log_directory=log_directory,
            alternate_training=alternate_training,
            alternate_car=alternate_car)

    """
    Change to the desired .h5 weights file, comment out the next line on first run & runs that did not converge.
    Do not override a converged run's weights file! Load it but save under another path so you'll be able to
    revert back to it in case the following run did not converge. E.g.: <...weights_1.h5>, <...weights_2.h5>
    """
    if only_local:
        rl.local_network.load_weights(load_weight)
    else:
        if weights_from_local_network:
            G_model = rl.local_and_global_network
            load_global_network_weights_from_local_network(G_model, load_weight)
        else:
            rl.local_and_global_network.load_weights(load_weight)

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

        # alternate learning
        if episode_counter % 20 == 0:
            if alternate_car == 1:
                rl.alternate_car = 2
                if only_local:
                    rl.alternate_training_network = rl.copy_network(rl.local_network)
            else:
                rl.alternate_car = 1
                if only_local:
                    rl.alternate_training_network = rl.copy_network(rl.local_network)

        for step in range(max_steps):

            steps_counter += 1
            # perform a step in the environment, and get feedback about collision and updated controls:
            if (only_local):
                if not two_cars_local:
                    done, reached_target, updated_controls, reward = rl.step_only_local(airsim_client, steps_counter)
                else:
                    done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_only_local_2_cars(
                        airsim_client, steps_counter)
            else:
                done, reached_target, updated_controls_car1, updated_controls_car2, reward = rl.step_with_global(
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

            if two_cars_local or not only_local:
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
    if not os.path.exists(save_weights_directory):
        os.makedirs(save_weights_directory)
    if only_local:
        rl.local_network.save_weights(save_weights_directory + save_weight)
    else:
        rl.local_and_global_network.save_weights(save_weights_directory + save_weight)

    print("@@@@ Run Ended @@@@")
    print(collision_counter)
