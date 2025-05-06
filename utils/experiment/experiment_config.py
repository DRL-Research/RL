from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import numpy as np
from utils.experiment.experiment_constants import Role, CarName


@dataclass
class Experiment:
    LOAD_PREVIOUS_WEIGHT = True
    BYPASS_RANDOM_INITIALIZATION = False
    # General Experiment Settings
    EPISODES_PER_CYCLE: int = 100
    CYCLES: int = 3
    EXPERIMENT_ID: str = ""
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = field(default_factory=lambda: datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
    SELF_PLAY_MODE: bool = False
    MASTER_TRAINED_MODEL: str = "EXP5_Inference_Models/master_trained_model.zip"
    AGENT_TRAINED_MODEL: str = "EXP5_Inference_Models/agent_trained_model.zip"
    CARS_AMOUNT: int = 5  # Updated to 5 cars

    # Model and Training Configuration
    EPISODE_AMOUNT_FOR_TRAIN: int = 1
    ROLE: Role = CarName.CAR1  # Which car uses the DRL model: Car1, Car2, or Both
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.05
    N_STEPS: int = 22
    BATCH_SIZE: int = 32
    TIME_BETWEEN_STEPS: float = 0.05
    LOSS_FUNCTION: str = "mse"
    EXPLORATION_EXPLOTATION_THRESHOLD: int = 50

    # Car 1 Settings (Agent)
    CAR1_NAME: CarName = CarName.CAR1
    CAR1_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [30, 0])
    CAR1_INITIAL_YAW_OPTION_1: int = 180
    CAR1_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-30, 0])
    CAR1_INITIAL_YAW_OPTION_2: int = 0
    CAR1_DESIRED_POSITION_OPTION_1: np.ndarray = field(default_factory=lambda: np.array([-10, 0]))
    CAR1_DESIRED_POSITION_OPTION_2: np.ndarray = field(default_factory=lambda: np.array([10, 0]))

    # Car 2 and Car 3 Settings (Perpendicular cars)
    CAR2_NAME: CarName = CarName.CAR2
    CAR2_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, 30])
    CAR2_INITIAL_YAW_OPTION_1: int = 270
    CAR2_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, -30])
    CAR2_INITIAL_YAW_OPTION_2: int = 90
    CAR2_DESIRED_POSITION_OPTION_1: np.ndarray = field(default_factory=lambda: np.array([0, -10]))
    CAR2_DESIRED_POSITION_OPTION_2: np.ndarray = field(default_factory=lambda: np.array([0, 10]))
    # CAR3_NAME: CarName = CarName.CAR3
    #
    # # New Car 4 Settings (In front of Car1)
    # CAR4_NAME: CarName = CarName.CAR4
    # CAR4_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [20, 0])
    # CAR4_INITIAL_YAW_OPTION_1: int = 180
    # CAR4_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-20, 0])
    # CAR4_INITIAL_YAW_OPTION_2: int = 0
    #
    # # New Car 5 Settings (Opposite direction to Cars 2/3)
    # CAR5_NAME: CarName = CarName.CAR5
    # CAR5_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, -30])
    # CAR5_INITIAL_YAW_OPTION_1: int = 90
    # CAR5_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, 30])
    # CAR5_INITIAL_YAW_OPTION_2: int = 270

    # Master embedding size configuration
    EMBEDDING_SIZE: int = 4

    # Cars Setup Configuration
    RANDOM_INIT: bool = False
    INIT_SERIAL: bool = False

    # Network Configuration
    PPO_NETWORK_ARCHITECTURE: Dict[str, List[int]] = field(
        default_factory=lambda: {'pi': [64, 32, 16, 8], 'vf': [64, 32, 16, 8]})

    # State Configuration - still 8-dimensional (4 from car state + 4 from master embedding)
    STATE_INPUT_SIZE: int = 8

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 1
    THROTTLE_SLOW: float = 0.4
    FIXED_THROTTLE: float = 0.6

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 20
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.5

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights

    # Computed fields (not passed via __init__)
    EXPERIMENT_PATH: str = field(init=False)
    SAVE_MODEL_DIRECTORY: str = field(init=False)

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"