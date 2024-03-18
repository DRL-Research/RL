from datetime import datetime
from RL.utils.airsim_utils import *


# Path definition
EXPERIMENT_ID = "local_experiment"
LOAD_WEIGHT_DIRECTORY = "experiments/exp4/weights/4_forth_left.h5"
WEIGHTS_TO_SAVE_ID = "/epochs_0_100.h5"

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

