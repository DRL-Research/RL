import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

from logger.neptune_logger import NeptuneLogger


@dataclass
class Experiment:
    LOAD_PREVIOUS_WEIGHT = True
    BYPASS_RANDOM_INITIALIZATION = False

    # General Experiment Settings
    EPISODES_PER_CYCLE: int = 50
    CYCLES: int = 3
    EXPERIMENT_ID: str = "fixed_training"
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = field(default_factory=lambda: datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
    SELF_PLAY_MODE: bool = False
    MASTER_TRAINED_MODEL: str = "EXP5_Inference_Models/master_trained_model.zip"
    AGENT_TRAINED_MODEL: str = "EXP5_Inference_Models/agent_trained_model.zip"
    CARS_AMOUNT: int = 5  # Updated to 5 cars
    SPAWN_PROBABILITY: float = 0

    # Model and Training Configuration
    EPISODE_AMOUNT_FOR_TRAIN: int = 3  # Train after 3 episodes instead of 1
    EPOCHS: int = None
    LEARNING_RATE: float = 0.005  # Reduced from 0.05
    N_STEPS: int = 64  # Increased from 30 to accommodate more steps
    BATCH_SIZE: int = 32
    EPISODE_MAX_TIME = 50  # seconds
    LOSS_FUNCTION: str = "mse"
    EXPLORATION_EXPLOITATION_THRESHOLD: int = 50

    # Cars Configuration - distance from intersection
    LONGITUDINAL: int = 40
    LATERAL: int = 0

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
    THROTTLE_FAST: float = 10
    THROTTLE_SLOW: float = 5
    FIXED_THROTTLE: float = 7.5

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 20
    COLLISION_REWARD: int = -20
    STARVATION_REWARD: float = -0.5

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights
    MODEL_TYPE: str = "PPO"  # Model type (e.g., PPO, DQN)
    # Computed fields (not passed via __init__)
    EXPERIMENT_PATH: str = field(init=False)
    SAVE_MODEL_DIRECTORY: str = field(init=False)

    # Lanes Directions
    SOUTH_TO_NORTH = ("o0", "ir0", 0)
    EAST_TO_WEST = ("o1", "ir1", 0)
    WEST_TO_EAST = ("o3", "ir3", 0)
    NORTH_TO_SOUTH = ("o2", "ir2", 0)

    # Directions for the cars
    OUTER_SOUTH = "o0"
    OUTER_WEST = "o1"
    OUTER_NORTH = "o2"
    OUTER_EAST = "o3"
    INNER_SOUTH = "i0"
    INNER_WEST = "i1"
    INNER_NORTH = "i2"
    INNER_EAST = "i3"


    # Simulations Graphics
    SCREEN_WIDTH: int = 600
    SCREEN_HEIGHT: int = 600
    CENTERING_POSITION: List[float] = field(default_factory=lambda: [0.5, 0.6])  # Do not change, this centers the simulation
    SCALING: float = 5.5 * 1.3

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"


        # Load API token from external JSON file
        try:
            with open("logger/token.json", "r") as f:
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