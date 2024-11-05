from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import numpy as np


@dataclass
class Experiment:

    # General Experiment Settings
    EXPERIMENT_ID: str
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

    # Model and Training Configuration
    ROLE: str = ""  # Which car is using the DRL model. Car1, Car2,Both
    LOAD_WEIGHT_DIRECTORY: str = ""  # Directory for loading weights
    LEARNING_RATE: float = 1.0
    N_STEPS: int = 160
    BATCH_SIZE: int = 160
    EXPLORATION_EXPLOTATION_THRESHOLD: int = 2500
    LOSS_FUNCTION: str = "mse"
    EPOCHS: int = 100
    TIME_BETWEEN_STEPS: float = 0.05

    # Car 1 Settings
    CAR1_NAME: str = "Car1"
    CAR1_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [30, 0])
    CAR1_INITIAL_YAW_OPTION_1: int = 180
    CAR1_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-30, 0])
    CAR1_INITIAL_YAW_OPTION_2: int = 0
    CAR1_DESIRED_POSITION_OPTION_1 = np.array([-10, 0])
    CAR1_DESIRED_POSITION_OPTION_2 = np.array([10, 0])

    # Car 2 Settings
    CAR2_NAME: str = "Car2"
    CAR2_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, 30])
    CAR2_INITIAL_YAW_OPTION_1: int = 270
    CAR2_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, -30])
    CAR2_INITIAL_YAW_OPTION_2: int = 90
    CAR2_CONSTANT_ACTION: float = (0.75 + 0.5) / 2

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 0.75
    THROTTLE_SLOW: float = 0.5
    FIXED_THROTTLE: float = (THROTTLE_FAST + THROTTLE_SLOW) / 2

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 10
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.1

    # Path Configuration
    # WEIGHTS_TO_SAVE_NAME: str = ""  # uncomment If needed

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"
