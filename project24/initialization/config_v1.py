from datetime import datetime
import numpy as np

CREATE_SUB_PLOTS = False
CREATE_MAIN_PLOT = True

# Path definition
EXPERIMENT_ID = "global_experiment"
WEIGHTS_TO_SAVE_NAME = "epochs_0_100"
# LOAD_WEIGHT_DIRECTORY = "experiments/local_experiment/weights/4_forth_left.h5"


# Car start positions and orientations
CAR1_INITIAL_POSITION = [6, 24, 0]
CAR2_INITIAL_POSITION = [-19, 1, 0]  # on the left side
# CAR3_INITIAL_POSITION = [4, -24, 0]
# CAR4_INITIAL_POSITION = [29, 1, 0]  # on the right side

# NOTE - for now we dont use it
CAR1_DESIRED_POSITION = np.array([10, 0, 0])
CAR2_DESIRED_POSITION = np.array([10, 0, 0])
# CAR3_DESIRED_POSITION = np.array([10, 0, 0])
# CAR4_DESIRED_POSITION = np.array([10, 0, 0])

CAR1_INITIAL_YAW = 270  # 0
CAR2_INITIAL_YAW = 0  # 90
# CAR3_INITIAL_YAW = 90  # 180
# CAR4_INITIAL_YAW = 180  # 270

# Training configuration
ALTERNATE_TRAINING_EPISODE_AMOUNT = 10  # after how many episodes the local network of car1 is copied to car2
MAX_EPISODES = 1000
MAX_STEPS = 500
LEARNING_RATE = 0.003

EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

NUMBER_OF_CAR_IN_SIMULATION = 2

CAR1_NAME = "Car1"
USE_CAR1 = True
CAR2_NAME = "Car2"
USE_CAR2 = True
# CAR3_NAME = "Car3"
# USE_CAR3 = True
# CAR4_NAME = "Car4"
# USE_CAR4 = True

REACHED_TARGET_REWARD = 1000
COLLISION_REWARD = -1000
STARVATION_REWARD = -0.1
SAFETY_DISTANCE_FOR_BONUS = 100
KEEPING_SAFETY_DISTANCE_REWARD = 60
SAFETY_DISTANCE_FOR_PUNISH = 70
NOT_KEEPING_SAFETY_DISTANCE_REWARD = -150


""" Turn Consts """

ZERO_YAW_LOW_BOUNDREY = -2
ZERO_YAW_HIGH_BOUNDERY = 2
NINETY_YAW_LOW_BOUNDREY = 88
NINETY_YAW_HIGH_BOUNDERY = 92
ONE_EIGHTY_YAW_LOW_BOUNDREY = 178
ONE_EIGHTY_YAW_HIGH_BOUNDERY = 182

TURN_DIRECTION_RIGHT = 'right'
TURN_DIRECTION_LEFT = 'left'
TURN_DIRECTION_STRAIGHT = 'straight'

BASE_ROTATION = [0, 0, 0]

# Global variables for distance
FORWARD_DISTANCE_LEFT_TURN = 24
SIDE_DISTANCE_LEFT_TURN = 18

FORWARD_DISTANCE_RIGHT_TURN = 12
SIDE_DISTANCE_RIGHT_TURN = 12

FORWARD_DISTANCE_STRAIGHT = 36

TIME_TO_KEEP_STRAIGHT_AFTER_TURN = 5

DISTANCE_BEFORE_START_TURNING = 6

""" Follow Handler Consts """
K_VEL = 2.0     # Arbitrary
MAX_VELOCITY = 8.0  # m/s
MIN_VELOCITY = 3.0   # m/s
LOOKAHEAD = 5.0     # meters
K_STEER = 10.0      # Stanley steering coefficient  # shahar changes 2 to be 10

""" Perception """
CAR_DETECTION = False
