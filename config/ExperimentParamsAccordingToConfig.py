import tensorflow as tf
from configobj import ConfigObj
from datetime import datetime
import airsim
import os
import json
from RL import RL


class ExperimentParamsAccordingToConfig:

    def __init__(self):

        config = ConfigObj('config/config.ini')

        self.log_directory = config['ExperimentSettings']['experiment_log_directory']
        self.load_weight_directory = config['ExperimentSettings']['load_weight_directory']
        # TODO: save_weight_current_directory is not an indactive name...
        self.save_weight_current_directory = config['ExperimentSettings']['save_weight_current_directory']
        # TODO: maybe there is a cleaner way to do init_logging function (maybe without a function..)
        self.save_weights_directory, self.tensorboard = init_logging(self.log_directory)

        self.car1_location = [int(item) for item in config['ExperimentSettings'].as_list('car1_location')]
        self.car2_location = [int(item) for item in config['ExperimentSettings'].as_list('car2_location')]

        self.global_experiment = config['ExperimentSettings'].as_bool('global_experiment')
        self.max_episodes = config['ExperimentSettings'].as_int('max_episodes')
        self.max_steps = config['ExperimentSettings'].as_int('max_steps')

        self.alternate_training = config['ExperimentSettings'].as_bool('alternate_training')
        self.alternate_car = config['ExperimentSettings'].as_int('alternate_car')

        self.rl = RL(learning_rate=0.003,
                     verbose=0,
                     with_per=True,
                     log_directory=self.log_directory,
                     alternate_training=self.alternate_training,
                     alternate_car=self.alternate_car)

        # load airsim settings for the experiment
        load_airsim_settings(self.car1_location, self.car2_location)
        print("settings ready, press play in unreal simulation.")

        # init airsim client instance + enable control (currently 2 cars):
        self.airsim_client = init_airsim_client()



def init_logging(log_directory):
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
