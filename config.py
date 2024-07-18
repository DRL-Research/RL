from datetime import datetime
import numpy as np


class Config:    

    def __init__(self):

        # Experiment Configuration
        self.EXPERIMENT_ID = "EXP1"
        self.EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.WEIGHTS_TO_SAVE_NAME = "epochs_0_100"
        # self.LOAD_WEIGHT_DIRECTORY = "experiments/local_experiment/weights/4_forth_left.h5"

        # Training Configuration
        # self.AGENT_ONLY = True  # the network and training process is on agent only, TODO: change the import of rl/rl_agent_only
        self.TRAIN_OPTION = "trajectory"  # options: step/trajectory/batch_of_trajectories
        self.ALTERNATE_MASTER_AND_AGENT_TRAINING = True  # options: True/False
        self.ALTERNATE_TRAINING_EPISODE_AMOUNT = 20
        self.MAX_EPISODES = 100  # 100
        self.MAX_STEPS = 500
        self.EPSILON_DECAY = 0.9  # 0.9
        self.LEARNING_RATE = 0.003
        self.LOSS_FUNCTION = "mse"
        self.TIME_BETWEEN_STEPS = 2.0
        self.COPY_CAR1_NETWORK_TO_CAR2 = True  # options: True/False
        self.COPY_CAR1_NETWORK_TO_CAR2_EPISODE_AMOUNT = 1

        # Logging Configuration
        self.LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES = 5
        self.LOG_ACTIONS_SELECTED = True  # options: True/False
        self.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY = True  # options: True/False
        self.LOG_CAR_STATES = False  # options: True/False
        self.LOG_Q_VALUES = False  # options: True/False
        self.LOG_WEIGHTS_ARE_IDENTICAL = False  # options: True/False

        # Cars Configuration
        self.CAR1_NAME = "Car1"
        self.CAR2_NAME = "Car2"
        self.CAR1_INITIAL_POSITION = [-25, 0]
        self.CAR2_INITIAL_POSITION = [0, -30]
        self.CAR1_INITIAL_YAW = 0
        self.CAR2_INITIAL_YAW = 90
        self.CAR1_DESIRED_POSITION = np.array([10, 0])

        # Reward Configuration
        self.REACHED_TARGET_REWARD = 10
        self.COLLISION_REWARD = -20
        self.STARVATION_REWARD = -0.1
        self.SAFETY_DISTANCE_FOR_BONUS = 1200
        self.KEEPING_SAFETY_DISTANCE_REWARD = 2
        self.SAFETY_DISTANCE_FOR_PUNISH = 1200
        self.NOT_KEEPING_SAFETY_DISTANCE_REWARD = -2

    def set_config(self, config_dict):
        print(config_dict)
        print(type(config_dict))
        for attr_name, attr_value in config_dict.items():
            print(attr_name, attr_value)
            setattr(self, attr_name, attr_value)
