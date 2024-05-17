from datetime import datetime
import numpy as np

# Path definition
EXPERIMENT_ID = "global_experiment"
WEIGHTS_TO_SAVE_NAME = "epochs_0_100"
# LOAD_WEIGHT_DIRECTORY = "experiments/local_experiment/weights/4_forth_left.h5"

# Mode settings
GLOBAL_EXPERIMENT = False

# Car start positions and orientations
CAR1_INITIAL_POSITION = [-20, 0]
CAR2_INITIAL_POSITION = [0, -20]
CAR1_DESIRED_POSITION = np.array([10, 0])
CAR1_INITIAL_YAW = 0
CAR2_INITIAL_YAW = 90

# Training configuration
ALTERNATE_TRAINING_EPISODE_AMOUNT = 10  # after how many episodes the local network of car1 is copied to car2
MAX_EPISODES = 1000
MAX_STEPS = 500
LEARNING_RATE = 0.003

EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

CAR1_NAME = "Car1"
CAR2_NAME = "Car2"

REACHED_TARGET_REWARD = 1000
COLLISION_REWARD = -1000
STARVATION_REWARD = -0.1
SAFETY_DISTANCE_FOR_BONUS = 100
KEEPING_SAFETY_DISTANCE_REWARD = 60
SAFETY_DISTANCE_FOR_PUNISH = 70
NOT_KEEPING_SAFETY_DISTANCE_REWARD = -150
