from dataclasses import dataclass
from datetime import datetime
from typing import List
from dataclasses import field
import numpy as np


@dataclass
class Experiment:
    #Cars positions and yaw
    EXPERIMENT_ID: str = ""
    CAR_INITAL_POS_NULL=None
    CAR1_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [30, 0])
    CAR1_INITIAL_YAW_OPTION_1: int = 180
    CAR1_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [-30, 0])
    CAR1_INITIAL_YAW_OPTION_2: int = 0
    CAR1_DESIRED_POSITION_OPTION_1 = np.array([-10, 0])
    CAR1_DESIRED_POSITION_OPTION_2 = np.array([10, 0])
    CAR2_INITIAL_POSITION_OPTION_1: List[int] = field(default_factory=lambda: [0, 30])
    CAR2_INITIAL_YAW_OPTION_1: int = 270
    CAR2_INITIAL_POSITION_OPTION_2: List[int] = field(default_factory=lambda: [0, -30])
    CAR2_INITIAL_YAW_OPTION_2: int = 90

    #Experimental and model
    ONLY_INFERENCE: bool = False
    MAX_EPISODES: int =  100
    LOAD_WEIGHT_DIRECTORY: str = ""
    ROLE: str = "" # which car using the DRL model
    EXPLORATION_EXPLOTATION_THRESHOLD: int = 2500
    TIME_BETWEEN_STEPS: float = 0.05
    LEARNING_RATE: int = 1
    N_STEPS: int = 160
    BATCH_SIZE: int = 160
    FIXED_THROTTLE: int = 1

    # Path Configuration
    WEIGHTS_TO_SAVE_NAME: str = ""
    EXPERIMENT_DATE_TIME = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    SAVE_MODEL_DIRECTORY = f"experiments/{EXPERIMENT_DATE_TIME}_{EXPERIMENT_ID}/model"

    # Training Configuration
    AGENT_ONLY = False
    LOSS_FUNCTION = "mse"
    EPOCHS = 1
    CAR2_CONSTANT_ACTION = (0.75 + 0.5) / 2

    # Cars Configuration
    CAR1_NAME = "Car1"
    CAR2_NAME = "Car2"

    # State Configuration
    AGENT_INPUT_SIZE = 2

    # Reward Configuration
    REACHED_TARGET_REWARD = 10
    COLLISION_REWARD = -20
    STARVATION_REWARD = -0.1



