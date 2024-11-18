from dataclasses import dataclass, field
from typing import List , Dict
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
    TIME_BETWEEN_STEPS: float = 0.5
    MODEL_TYPE: str = 'PPO'  # Can be changed to 'A2C' or 'DQN'

    # Specific configurations for DQN
    BUFFER_SIZE: int = 10000
    TARGET_UPDATE_INTERVAL: int = 1000
    EXPLORATION_FRACTION: float = 0.1
    EXPLORATION_FINAL_EPS: float = 0.02
    EXPLORATION_INITIAL_EPS: float = 1.0

    # Specific configurations for PPO and A2C
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    ENT_COEF: float = 0.0
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    USE_SDE: bool = False
    SDE_SAMPLE_FREQ: int = -1

    # Model specific configurations (e.g., custom configs per model)
    MODEL_CONFIGS: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        # Default configurations per model type
        self.MODEL_CONFIGS = {
            'PPO': {'N_STEPS': self.N_STEPS, 'BATCH_SIZE': self.BATCH_SIZE, 'GAMMA': self.GAMMA,
                    'GAE_LAMBDA': self.GAE_LAMBDA, 'ENT_COEF': self.ENT_COEF, 'VF_COEF': self.VF_COEF,
                    'MAX_GRAD_NORM': self.MAX_GRAD_NORM, 'LEARNING_RATE': self.LEARNING_RATE, 'USE_SDE': self.USE_SDE,
                    'SDE_SAMPLE_FREQ': self.SDE_SAMPLE_FREQ},
            'DQN': {'LEARNING_RATE': self.LEARNING_RATE, 'BUFFER_SIZE': self.BUFFER_SIZE, 'BATCH_SIZE': self.BATCH_SIZE,
                    'GAMMA': self.GAMMA, 'TARGET_UPDATE_INTERVAL': self.TARGET_UPDATE_INTERVAL, 'TRAIN_FREQ': 1,
                    'GRADIENT_STEPS': self.BATCH_SIZE, 'EXPLORATION_FRACTION': self.EXPLORATION_FRACTION,
                    'EXPLORATION_INITIAL_EPS': self.EXPLORATION_INITIAL_EPS,
                    'EXPLORATION_FINAL_EPS': self.EXPLORATION_FINAL_EPS},
            'A2C': {'N_STEPS': self.N_STEPS, 'GAMMA': self.GAMMA, 'GAE_LAMBDA': self.GAE_LAMBDA,
                    'ENT_COEF': self.ENT_COEF, 'VF_COEF': self.VF_COEF, 'MAX_GRAD_NORM': self.MAX_GRAD_NORM,
                    'LEARNING_RATE': self.LEARNING_RATE}
        }

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

    # State Configuration
    INPUT_SIZE = 8

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 0.1
    THROTTLE_SLOW: float = 0.9
    FIXED_THROTTLE: float = (THROTTLE_FAST + THROTTLE_SLOW) / 2

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 10
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.1

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"
