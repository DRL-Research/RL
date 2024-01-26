from datetime import datetime
from RL.utils.rl_agent_utils import RLAgent
from RL.utils.airsim_utils import *
from RL.utils.tensorboard_utils import *


class ExperimentParams:

    def __init__(self):

        self.experiment_id = "exp4"
        self.weights_to_save_id = "/4_forth_left.h5"
        self.load_weight_directory = "experiments/exp4/weights/3_third_left.h5"
        self.global_experiment = False
        self.car1_start_location = [0, 0]
        self.car2_start_location = [0, -5]
        self.car1_start_yaw = 0
        self.car2_start_yaw = 90
        self.max_episodes = 3
        self.max_steps = 500
        self.alternate_training = True
        self.alternate_car = 1
        self.learning_rate = 0.003
        self.verbose_RL_Agent = False

        self.current_date_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.tensorboard = init_tensorboard(self.experiment_id, self.current_date_time)
        self.RL_Agent = RLAgent(learning_rate=self.learning_rate,
                                verbose=self.verbose_RL_Agent,
                                experiment_id=self.experiment_id,
                                alternate_training=self.alternate_training,
                                alternate_car=self.alternate_car,
                                current_date_time=self.current_date_time,
                                tensorboard=self.tensorboard)

        # Initialize airsim client instance + enable control (currently 2 cars)
        self.airsim_client = init_airsim_client()

        # move cars to initial positions (currently 2 cars)
        move_cars_to_initial_positions(self.airsim_client,
                                       self.car1_start_location,
                                       self.car2_start_location,
                                       self.car1_start_yaw,
                                       self.car2_start_yaw)
