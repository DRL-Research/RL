from datetime import datetime
import numpy as np
class Config:

    def __init__(self):
        # Experiment Configuration
        self.EXPERIMENT_ID = "Car_2_Random_Side"
        self.EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.WEIGHTS_TO_SAVE_NAME = "PPO_model"
        self.LOAD_WEIGHT_DIRECTORY = None

        # Path Configuration
        self.SAVE_WEIGHT_DIRECTORY = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"

        # Training Configuration
        self.AGENT_ONLY = False
        self.TRAIN_OPTION = "trajectory"
        self.ALTERNATE_MASTER_AND_AGENT_TRAINING = True
        self.ALTERNATE_TRAINING_EPISODE_AMOUNT = 20
        self.MAX_EPISODES = 50
        self.MAX_STEPS = 10
        self.LEARNING_RATE = 0.0001
        self.N_STEPS = 160
        self.BATCH_SIZE = 160
        self.LOSS_FUNCTION = "mse"
        self.TIME_BETWEEN_STEPS = 0.005
        self.EPOCHS = 1
        self.ONLY_INFERENCE = False
        self.COPY_CAR1_NETWORK_TO_CAR2 = True
        self.COPY_CAR1_NETWORK_TO_CAR2_EPISODE_AMOUNT = 1
        self.CAR2_CONSTANT_ACTION = (0.75 + 0.5) / 2
        self.SET_CAR2_INITIAL_DIRECTION_MANUALLY = True
        self.CAR2_INITIAL_DIRECTION = 1
        self.FIXED_THROTTLE = (0.75 + 0.5) / 2

        # Logging Configuration
        self.LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES = 5
        self.LOG_ACTIONS_SELECTED = True
        self.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY = True
        self.LOG_CAR_STATES = False
        self.LOG_Q_VALUES = False
        self.LOG_WEIGHTS_ARE_IDENTICAL = False

        # Cars Configuration
        self.CAR1_NAME = "Car1"
        self.CAR2_NAME = "Car2"
        self.CAR1_INITIAL_POSITION = [-20, 0]
        self.CAR2_INITIAL_POSITION = [0, -20]
        self.CAR1_INITIAL_YAW = 0
        self.CAR2_INITIAL_YAW = 90
        self.CAR1_DESIRED_POSITION = np.array([10, 0])
        self.ROLE = None #which car using the DRL model?

        # State Configuration
        self.AGENT_INPUT_SIZE = 2

        # Reward Configuration
        self.REACHED_TARGET_REWARD = 10
        self.COLLISION_REWARD = -20
        self.STARVATION_REWARD = -0.1
        self.SAFETY_DISTANCE_FOR_BONUS = 1200
        self.KEEPING_SAFETY_DISTANCE_REWARD = 2
        self.SAFETY_DISTANCE_FOR_PUNISH = 1200
        self.NOT_KEEPING_SAFETY_DISTANCE_REWARD = -2
        self.EXPLORATION_EXPLOTATION_THRESHOLD = 2500


    def set_config(self, config_dict):
        for attr_name, attr_value in config_dict.items():
            print(attr_name, attr_value)
            setattr(self, attr_name, attr_value)
