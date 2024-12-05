from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np

from experiment.experiment_constants import Role, CarName
from model.model_constants import ModelType


@dataclass
class Experiment:

    # General Experiment Settings
    EXPERIMENT_ID: str = ""
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

    # Model and Training Configuration
    ROLE: Role = None  # Which car is using the DRL model. Car1, Car2, Both
    LEARNING_RATE: float = None
    N_STEPS: int = 160
    BATCH_SIZE: int = 160
    EXPLORATION_EXPLOTATION_THRESHOLD: int = 2500
    LOSS_FUNCTION: str = "mse"
    EPOCHS: int = 100
    TIME_BETWEEN_STEPS: float = 0.5
    MODEL_TYPE: ModelType = None

    # Car 1 Settings
    CAR1_NAME: CarName = CarName.CAR1
    CAR1_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [30, 0])
    CAR1_INITIAL_YAW_OPTION_1: int = 180
    CAR1_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-30, 0])
    CAR1_INITIAL_YAW_OPTION_2: int = 0
    CAR1_DESIRED_POSITION_OPTION_1 = np.array([-10, 0])
    CAR1_DESIRED_POSITION_OPTION_2 = np.array([10, 0])

    # Car 2 Settings
    CAR2_NAME: CarName = CarName.CAR2
    CAR2_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, 30])
    CAR2_INITIAL_YAW_OPTION_1: int = 270
    CAR2_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, -30])
    CAR2_INITIAL_YAW_OPTION_2: int = 90

    # State Configuration
    INPUT_SIZE = 8

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 0.4
    THROTTLE_SLOW: float = 0.7
    FIXED_THROTTLE: float = (THROTTLE_FAST + THROTTLE_SLOW) / 2

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 10
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.1

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights
    EXPERIMENT_PATH = f"experiments/{EXPERIMENT_DATE_TIME}_{EXPERIMENT_ID}"
    SAVE_MODEL_DIRECTORY = f"{EXPERIMENT_PATH}/trained_model"
