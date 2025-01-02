from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np

from utils.experiment.experiment_constants import Role, CarName
from utils.logger.neptune_logger import NeptuneLogger
from utils.model.model_constants import ModelType

import json


@dataclass
class Experiment:
    # General Experiment Settings
    EXPERIMENT_ID: str = ""
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

    # Model and Training Configuration
    MODEL_TYPE: ModelType = None
    ROLE: Role = None  # Which car is using the DRL model. Car1, Car2, Both
    EPOCHS: int = 100
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 160
    BATCH_SIZE: int = 160
    TIME_BETWEEN_STEPS: float = 0.5
    LOSS_FUNCTION: str = "mse"
    EXPLORATION_EXPLOTATION_THRESHOLD: int = 50

    # Seed for reproducibility
    SEED: int = 42

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

    # Cars Setup Configuration
    RANDOM_INIT = True

    # Network Configuration
    PPO_NETWORK_ARCHITECTURE = {'pi': [32, 32], 'vf': [32, 32]}

    # State Configuration
    INPUT_SIZE = 8

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 0.7
    THROTTLE_SLOW: float = 0.4
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

        # Load API token from external JSON file
        try:
            with open("utils/logger/token.json", "r") as f:
                config = json.load(f)
                api_token = config["api_token"]
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError("Failed to load API token from config.json") from e

        self.logger = NeptuneLogger(
            project_name="AS-DRL/DRL-Research",
            api_token=api_token,
            run_name=self.EXPERIMENT_ID,
            tags=["experiment", "training"]
        )
