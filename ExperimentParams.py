from datetime import datetime
from RL.utils.rl_agent_utils import RLAgent
from RL.utils.airsim_utils import *
from RL.utils.tensorboard_utils import *


class ExperimentParams:

    def __init__(self):
        # Experiment configuration
        self.experiment_id = "exp4"
        self.load_weight_directory = "experiments/exp4/weights/4_forth_left.h5"
        self.weights_to_save_id = "/epochs_0_100.h5"
        self.global_experiment = False

        # Car start positions and orientations
        # Note: If changing the start location during an experiment,
        # update the functions: set_coordinates_of_car_once() and reset_cars_to_initial_positions()
        self.car1_start_location = [-20, 0]
        self.car2_start_location = [0, -20]
        self.car1_start_yaw = 0
        self.car2_start_yaw = 90

        # Training configuration
        self.max_episodes = 1000
        self.max_steps = 500
        # Alternate training episodes amount - after how many episodes the local network of car1 is copied to car2
        self.alternate_training_episode_amount = 10

        # RL Agent configuration
        self.learning_rate = 0.003
        self.verbose_RL_Agent = False
        self.current_date_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.tensorboard = init_tensorboard(self.experiment_id, self.current_date_time)
        self.RL_Agent = RLAgent(learning_rate=self.learning_rate,
                                verbose=self.verbose_RL_Agent,
                                tensorboard=self.tensorboard)

        # AirSim client initialization
        self.airsim_client = init_airsim_client()

        # Initial position offsets for cars
        self.car1_x_offset, self.car1_y_offset = 0, 0
        self.car2_x_offset, self.car2_y_offset = 0, 0
        get_cars_initial_offset(self)
        reset_cars_to_initial_positions(self, self.airsim_client)

