from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class Experiment:
    LOAD_PREVIOUS_WEIGHT = True
    BYPASS_RANDOM_INITIALIZATION = False

    # General Experiment Settings
    EPISODES_PER_CYCLE: int = 300
    CYCLES: int = 3
    EXPERIMENT_ID: str = "fixed_training"
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = field(default_factory=lambda: datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
    SELF_PLAY_MODE: bool = False
    MASTER_TRAINED_MODEL: str = "EXP5_Inference_Models/master_trained_model.zip"
    AGENT_TRAINED_MODEL: str = "EXP5_Inference_Models/agent_trained_model.zip"
    CARS_AMOUNT: int = 5
    SPAWN_PROBABILITY: float = 0
    RENDER_MODE: str | None = "rgb_array"

    # Model and Training Configuration
    EPISODE_AMOUNT_FOR_TRAIN: int = 2
    EPOCHS: int = None
    LEARNING_RATE: float = 0.005
    N_STEPS: int = 64
    BATCH_SIZE: int = 32
    EPISODE_MAX_TIME = 50  # seconds
    LOSS_FUNCTION: str = "mse"
    EXPLORATION_EXPLOITATION_THRESHOLD: int = 800

    # Cars Configuration - distance from intersection
    LONGITUDINAL: int = 40
    LATERAL: int = 0

    # Master embedding size configuration
    EMBEDDING_SIZE: int = 4

    # Cars Setup Configuration
    RANDOM_INIT: bool = False
    INIT_SERIAL: bool = False

    # State Configuration
    AGENT_STATE_SIZE: int = 4
    STATE_INPUT_SIZE: int = EMBEDDING_SIZE + AGENT_STATE_SIZE  # 8

    # Action Configuration
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 50
    THROTTLE_SLOW: float = 10
    FIXED_THROTTLE: float = 12

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 50
    COLLISION_REWARD: int = -300
    STARVATION_REWARD: float = -5
    HIGH_SPEED_REWARD = 5

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""
    MODEL_TYPE: str = "PPO"
    EXPERIMENT_PATH: str = field(init=False)
    SAVE_MODEL_DIRECTORY: str = field(init=False)

    # Scenario folder — if set, the experiment loads scenarios from this folder
    # instead of the hard-coded lists in scenarios.py.
    # Set to a path like "saved_scenarios/double_intersection" or leave empty to
    # use the default scenario lists.
    SCENARIOS_FOLDER: str = ""

    # Lanes Directions
    SOUTH_TO_NORTH = ("o0", "ir0", 0)
    WEST_TO_EAST = ("o1", "ir1", 0)
    NORTH_TO_SOUTH = ("o2", "ir2", 0)
    EAST_TO_WEST = ("o3", "ir3", 0)

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
    SCREEN_WIDTH: int = 900
    SCREEN_HEIGHT: int = 800
    CENTERING_POSITION: List[float] = field(default_factory=lambda: [0.5, 0.6])
    SCALING: float = 3 * 1.3

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"
