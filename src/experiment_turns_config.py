from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np

from utils.experiment.experiment_constants import Role, CarName
from utils.model.model_constants import ModelType


@dataclass
class ExperimentTurns:
    # General Experiment Settings
    EXPERIMENT_ID: str = ""
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

    # Model and Training Configuration
    MODEL_TYPE: ModelType = None
    ROLE: Role = None  # Which car is using the DRL model. Car1, Car2, Both
    EPOCHS: int = 3
    LEARNING_RATE: float = 0.01
    N_STEPS: int = 160
    BATCH_SIZE: int = 160
    TIME_BETWEEN_STEPS: float = 0.75
    LOSS_FUNCTION: str = "mse"
    EXPLORATION_EXPLOTATION_THRESHOLD: float = 0.2

    # Car 1 Settings
    CAR1_NAME: CarName = CarName.CAR1
    CAR1_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [30, 0])
    CAR1_INITIAL_YAW_OPTION_1: int = 180
    CAR1_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-30, 0])
    CAR1_INITIAL_YAW_OPTION_2: int = 0
    CAR1_DESIRED_POSITION_OPTION_1 = np.array([-30, 0])
    CAR1_DESIRED_POSITION_OPTION_2 = np.array([30, 0])

    # Car 2 Settings
    CAR2_NAME: CarName = CarName.CAR2
    CAR2_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, 30])
    CAR2_INITIAL_YAW_OPTION_1: int = 270
    CAR2_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, -30])
    CAR2_INITIAL_YAW_OPTION_2: int = 90

    # Cars Setup Configuration
    RANDOM_INIT = True

    # Network Configuration
    PPO_NETWORK_ARCHITECTURE = {'pi': [32, 16], 'vf': [32, 16]}

    # State Configuration
    INPUT_SIZE = 8

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 1
    THROTTLE_SLOW: float = 0.6
    FIXED_THROTTLE: float = (THROTTLE_FAST + THROTTLE_SLOW) / 2

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 10
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.1

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"
