from configobj import ConfigObj
from datetime import datetime
from RL.utils.rl_agent_utils import RLAgent
from RL.utils.airsim_utils import *
from RL.utils.tensorboard_utils import *


class ExperimentParamsAccordingToConfig:

    def __init__(self):
        """
           This function reads configuration settings from 'config/config.ini'
           and sets up various parameters for your experiment.
        """
        # Load configuration from 'config/config.ini'
        config = ConfigObj('config/config.ini')

        # Initialize attributes
        self.current_date_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.experiment_id = config['ExperimentSettings']['experiment_id']
        self.weights_to_save_id = config['ExperimentSettings']['weights_to_save_id']
        self.load_weight_directory = config['ExperimentSettings']['load_weight_directory']
        self.tensorboard = init_tensorboard(self.experiment_id, self.current_date_time)

        self.car1_location = [int(item) for item in config['ExperimentSettings'].as_list('car1_location')]
        self.car2_location = [int(item) for item in config['ExperimentSettings'].as_list('car2_location')]

        self.global_experiment = config['ExperimentSettings'].as_bool('global_experiment')
        self.max_episodes = config['ExperimentSettings'].as_int('max_episodes')
        self.max_steps = config['ExperimentSettings'].as_int('max_steps')

        self.alternate_training = config['ExperimentSettings'].as_bool('alternate_training')
        self.alternate_car = config['ExperimentSettings'].as_int('alternate_car')

        # Initialize RLAgent
        self.RL_Agent = RLAgent(learning_rate=0.003,
                                verbose=0,
                                experiment_id=self.experiment_id,
                                alternate_training=self.alternate_training,
                                alternate_car=self.alternate_car,
                                current_date_time=self.current_date_time,
                                tensorboard=self.tensorboard)

        # Load airsim settings for the experiment
        update_airsim_settings_file(self.car1_location, self.car2_location)
        print("Settings are ready. Please press play in the Unreal simulation.")

        # Initialize airsim client instance + enable control (currently 2 cars)
        self.airsim_client = init_airsim_client()
