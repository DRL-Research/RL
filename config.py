from datetime import datetime
import numpy as np


# Experiment Configuration
EXPERIMENT_ID = "EXP8"
EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
WEIGHTS_TO_SAVE_NAME = "epochs_0_100"
# LOAD_WEIGHT_DIRECTORY = "experiments/local_experiment/weights/4_forth_left.h5"

# Training Configuration
# AGENT_ONLY = True  # the network and training process is on agent only, TODO: change the import of rl/rl_agent_only
TRAIN_OPTION = "trajectory"  # options: step/trajectory/batch_of_trajectories
ALTERNATE_TRAINING_EPISODE_AMOUNT = 20
MAX_EPISODES = 100  # 100
MAX_STEPS = 500
EPSILON_DECAY = 0.9  # 0.9
LEARNING_RATE = 0.003
LOSS_FUNCTION = "mse"

# Logging Configuration
LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES = 5
LOG_ACTIONS_SELECTED = True  # options: True/False
LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY = True  # options: True/False
LOG_CAR_STATES = False  # options: True/False
LOG_Q_VALUES = False  # options: True/False


# Cars Configuration
CAR1_NAME = "Car1"
CAR2_NAME = "Car2"
CAR1_INITIAL_POSITION = [-25, 0]
CAR2_INITIAL_POSITION = [0, -30]
CAR1_INITIAL_YAW = 0
CAR2_INITIAL_YAW = 90
CAR1_DESIRED_POSITION = np.array([10, 0])

# Reward Configuration
REACHED_TARGET_REWARD = 10
COLLISION_REWARD = -20
STARVATION_REWARD = -0.1
SAFETY_DISTANCE_FOR_BONUS = 1200
KEEPING_SAFETY_DISTANCE_REWARD = 2
SAFETY_DISTANCE_FOR_PUNISH = 1200
NOT_KEEPING_SAFETY_DISTANCE_REWARD = -2
