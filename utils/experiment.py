from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np


@dataclass
class Experiment:

    # TODO: go over all params, delete non-relevant, and define better the relevant (with gil, next meeting)

    EXPERIMENT_ID: str = ""
    CAR1_INITIAL_POSITION: List[int] = [30, 0],
    CAR2_INITIAL_POSITION: List[int] = [0, 30],
    CAR1_INITIAL_YAW: int = 0
    CAR2_INITIAL_YAW: int = 270
    ONLY_INFERENCE: bool = False
    MAX_EPISODES: int =  100
    LOAD_WEIGHT_DIRECTORY: str = ""
    ROLE: str = "" # which car using the DRL model?
    EXPLORATION_EXPLOTATION_THRESHOLD: int = 2500
    TIME_BETWEEN_STEPS: float = 0.05
    LEARNING_RATE: int = 1
    N_STEPS: int = 160
    BATCH_SIZE: int = 160
    FIXED_THROTTLE: int = 1

    # Path Configuration
    WEIGHTS_TO_SAVE_NAME: str = "PPO_model"

    # Training Configuration
    AGENT_ONLY = False
    TRAIN_OPTION = "trajectory"
    ALTERNATE_MASTER_AND_AGENT_TRAINING = True
    ALTERNATE_TRAINING_EPISODE_AMOUNT = 20
    MAX_STEPS = 10
    LOSS_FUNCTION = "mse"
    EPOCHS = 1
    COPY_CAR1_NETWORK_TO_CAR2 = True
    COPY_CAR1_NETWORK_TO_CAR2_EPISODE_AMOUNT = 1
    CAR2_CONSTANT_ACTION = (0.75 + 0.5) / 2
    SET_CAR2_INITIAL_DIRECTION_MANUALLY = True
    CAR2_INITIAL_DIRECTION = 1

    # Logging Configuration
    LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES = 5
    LOG_ACTIONS_SELECTED = True
    LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY = True
    LOG_CAR_STATES = False
    LOG_Q_VALUES = False
    LOG_WEIGHTS_ARE_IDENTICAL = False

    # Cars Configuration
    CAR1_NAME = "Car1"
    CAR2_NAME = "Car2"
    CAR1_DESIRED_POSITION = np.array([10, 0])

    # State Configuration
    AGENT_INPUT_SIZE = 2

    # Reward Configuration
    REACHED_TARGET_REWARD = 10
    COLLISION_REWARD = -20
    STARVATION_REWARD = -0.1
    SAFETY_DISTANCE_FOR_BONUS = 1200
    KEEPING_SAFETY_DISTANCE_REWARD = 2
    SAFETY_DISTANCE_FOR_PUNISH = 1200
    NOT_KEEPING_SAFETY_DISTANCE_REWARD = -2

    # Constants
    EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    SAVE_MODEL_DIRECTORY = f"experiments/{EXPERIMENT_DATE_TIME}_{EXPERIMENT_ID}/model"


