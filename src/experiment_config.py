from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import numpy as np
from src.constants import Role, CarName, ModelType

from src.logger.neptune_logger import NeptuneLogger
import json
from pathlib import Path


@dataclass
class Experiment:
    # General Experiment Settings
    EPISODES_PER_CYCLE = 50
    CYCLES = 3
    EXPERIMENT_ID: str = ""
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    SELF_PLAY_MODE: bool = False
    MASTER_TRAINED_MODEL: str = "EXP5_Inference_Models/master_trained_model.zip"
    AGENT_TRAINED_MODEL: str = "EXP5_Inference_Models/agent_trained_model.zip"
    CARS_AMOUNT= 1
    NUM_CARS = 3

    # Model and Training Configuration-
    EPISODE_AMOUNT_FOR_TRAIN : int = 4
    MODEL_TYPE: ModelType = None
    ROLE: Role = CarName.CAR1  # Which car is using the DRL model. Car1, Car2, Both
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.05
    N_STEPS: int = 90
    BATCH_SIZE: int = 32
    TIME_BETWEEN_STEPS: float = 0.05
    LOSS_FUNCTION: str = "mse"
    EXPLORATION_EXPLOTATION_THRESHOLD: int = None
    SEED: int = 42

    # Car 1 Settings
    CAR1_NAME: CarName = CarName.CAR1
    CAR1_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [30, 0])
    CAR1_INITIAL_YAW_OPTION_1: int = 180
    CAR1_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-30, 0])
    CAR1_INITIAL_YAW_OPTION_2: int = 0
    CAR1_DESIRED_POSITION_OPTION_1 = np.array([-10, 0])
    CAR1_DESIRED_POSITION_OPTION_2 = np.array([10, 0])

    # Car 2,3  Settings
    CAR2_NAME: CarName = CarName.CAR2
    CAR2_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, 30])
    CAR2_INITIAL_YAW_OPTION_1: int = 270
    CAR2_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, -30])
    CAR2_INITIAL_YAW_OPTION_2: int = 90
    CAR2_DESIRED_POSITION_OPTION_1 = np.array([0, -10])
    CAR2_DESIRED_POSITION_OPTION_2 = np.array([0, 10])

    # Master embedding size configuration
    EMBEDDING_SIZE: int = 4


    # Cars Setup Configuration
    RANDOM_INIT = False
    INIT_SERIAL: bool = True

    # Network Configuration
    PPO_NETWORK_ARCHITECTURE = {'pi': [32, 16 , 8 ], 'vf': [32, 16 , 8 , 4]}

    # State Configuration
    STATE_INPUT_SIZE = 4

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 0.8
    THROTTLE_SLOW: float = 0.4
    FIXED_THROTTLE: float = 0.6

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 20
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.5

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"

        try:
            token_path = Path(__file__).resolve().parents[1] / "src" / "logger" / "token.json"
            print(token_path)
            with open(token_path, "r") as f:
                config = json.load(f)
                print(config)
                api_token = config["api_token"]
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError("Failed to load API token from token.json") from e

        self.logger = NeptuneLogger(
            project_name="AS-DRL/DRL-Research",
            api_token=api_token,
            run_name=self.EXPERIMENT_ID,
            tags=["experiment", "training"]
        )
