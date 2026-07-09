import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict




@dataclass
class Experiment:
    ALGORITHM: str = "experiment"  # experiment | baseline | maddpg | vn_maddpg | attention_maddpg | ma_ga_ddpg | ippo | coma
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
    CARS_AMOUNT: int = 5  # Updated to 5 cars
    SPAWN_PROBABILITY: float = 0
    RENDER_MODE: str|None = "rgb_array"
    ENV_ID: str = "RELintersection-v0"

    # Model and Training Configuration
    EPISODE_AMOUNT_FOR_TRAIN: int = 2  # Train after x episodes instead of 1
    EPOCHS: int = None
    LEARNING_RATE: float = 0.005  # Reduced from 0.05
    N_STEPS: int = 64  # Increased from 30 to accommodate more steps
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

    # Network Configuration
    #PPO_NETWORK_ARCHITECTURE: Dict[str, List[int]] = field(
    #    default_factory=lambda: {'pi': [64, 32, 16, 8], 'vf': [64, 32, 16, 8]})

    # State Configuration - still 8-dimensional (4 from car state + 4 from master embedding)
    AGENT_STATE_SIZE: int = 4
    STATE_INPUT_SIZE: int = EMBEDDING_SIZE + AGENT_STATE_SIZE  # 8

    # Action Configurationss
    ACTION_SPACE_SIZE: int = 2
    THROTTLE_FAST: float = 50
    THROTTLE_SLOW: float = 10
    FIXED_THROTTLE: float = 12  # Fixed throttle for all cars

    # Reward Configuration
    REACHED_TARGET_REWARD: int = 50
    COLLISION_REWARD: int = -300
    STARVATION_REWARD: float = -5
    HIGH_SPEED_REWARD=5

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights
    MODEL_TYPE: str = "PPO"  # Model type (e.g., PPO, DQN)

    # Baseline (MADDPG / VN-MADDPG) configuration
    BASELINE_ACTOR_LR: float = 5e-4
    BASELINE_CRITIC_LR: float = 5e-4
    BASELINE_GAMMA: float = 0.99
    BASELINE_TAU: float = 0.01
    BASELINE_BATCH_SIZE: int = 256
    BASELINE_BUFFER_SIZE: int = 100000
    BASELINE_WARMUP_STEPS: int = 1000
    BASELINE_UPDATES_PER_STEP: int = 1
    BASELINE_TRAIN_EVERY: int = 1
    BASELINE_TARGET_UPDATE_INTERVAL: int = 90
    BASELINE_HIDDEN_DIM: int = 128
    BASELINE_INITIAL_NOISE: float = 0.25
    BASELINE_FINAL_NOISE: float = 0.0
    BASELINE_PRIORITY_ALPHA: float = 0.6
    BASELINE_PRIORITY_BETA_START: float = 0.4
    BASELINE_GUMBEL_TAU: float = 1.0
    BASELINE_MAX_GRAD_NORM: float = 1.0
    BASELINE_EVAL_EPISODES: int = 5

    # Attention-MADDPG / MA-GA-DDPG configuration
    MA_GA_ACTOR_LR: float = 1e-2
    MA_GA_CRITIC_LR: float = 1e-2
    MA_GA_GAMMA: float = 0.95
    MA_GA_TAU: float = 0.01
    MA_GA_BATCH_SIZE: int = 128
    MA_GA_BUFFER_SIZE: int = 10000
    MA_GA_WARMUP_STEPS: int = 128
    MA_GA_UPDATES_PER_STEP: int = 1
    MA_GA_TRAIN_EVERY: int = 100
    MA_GA_TARGET_UPDATE_INTERVAL: int = 1
    MA_GA_HIDDEN_DIM: int = 128
    MA_GA_ATTENTION_HEADS: int = 2
    MA_GA_INITIAL_NOISE: float = 0.15
    MA_GA_FINAL_NOISE: float = 0.15
    MA_GA_NOISE_SIGMA: float = 0.2
    MA_GA_GUMBEL_TAU: float = 1.0
    MA_GA_MAX_GRAD_NORM: float = 1.0
    MA_GA_EVAL_EPISODES: int = 5
    MA_GA_INTERACTION_DISTANCE: float = 1.25
    MA_GA_ATTENTION_THRESHOLD: float = 0.05
    MA_GA_MAX_INTERACTION_OBJECTS: int = 5
    MA_GA_PREDICTION_STEPS: int = 5
    MA_GA_PREDICTION_DELTA: float = 1.0
    MA_GA_CONFLICT_DISTANCE: float = 0.15
    MA_GA_SLOWER_SCALE: float = 0.5
    MA_GA_FASTER_SCALE: float = 1.25
    MA_GA_MAX_PREDICTED_SPEED: float = 1.0

    # IPPO configuration
    IPPO_ACTOR_LR: float = 3e-4
    IPPO_CRITIC_LR: float = 1e-3
    IPPO_GAMMA: float = 0.99
    IPPO_GAE_LAMBDA: float = 0.95
    IPPO_CLIP_EPSILON: float = 0.2
    IPPO_ENTROPY_COEF: float = 0.01
    IPPO_EPOCHS: int = 10
    IPPO_BATCH_SIZE: int = 64
    IPPO_HIDDEN_DIM: int = 64
    IPPO_ROLLOUT_STEPS: int = 2048

    # COMA configuration
    COMA_ACTOR_LR: float = 3e-4
    COMA_CRITIC_LR: float = 1e-3
    COMA_GAMMA: float = 0.99
    COMA_GAE_LAMBDA: float = 0.95
    COMA_CLIP_EPSILON: float = 0.2
    COMA_ENTROPY_COEF: float = 0.01
    COMA_EPOCHS: int = 10
    COMA_BATCH_SIZE: int = 64
    COMA_HIDDEN_DIM: int = 64
    COMA_ROLLOUT_STEPS: int = 2048
    COMA_TARGET_UPDATE_INTERVAL: int = 10

    # VDN configuration
    VDN_LR: float = 1e-3
    VDN_GAMMA: float = 0.99
    VDN_BATCH_SIZE: int = 64
    VDN_BUFFER_SIZE: int = 100000
    VDN_TARGET_UPDATE_INTERVAL: int = 100
    VDN_HIDDEN_DIM: int = 64
    VDN_EPSILON_START: float = 1.0
    VDN_EPSILON_MIN: float = 0.05
    VDN_EPSILON_DECAY: float = 0.995

    # Computed fields (not passed via __init__)
    EXPERIMENT_PATH: str = field(init=False)
    SAVE_MODEL_DIRECTORY: str = field(init=False)

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
    # SCREEN_WIDTH: int = 600
    SCREEN_WIDTH: int = 900
    # SCREEN_HEIGHT: int = 600
    SCREEN_HEIGHT: int = 800
    CENTERING_POSITION: List[float] = field(default_factory=lambda: [0.5, 0.6])  # Do not change, this centers the simulation
    # SCALING: float = 5.5 * 1.3
    SCALING: float = 3 * 1.3

    def __post_init__(self):
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_DATE_TIME}_{self.EXPERIMENT_ID}"
        self.SAVE_MODEL_DIRECTORY = f"{self.EXPERIMENT_PATH}/trained_model"


        # Load API token from external JSON file
        try:
            with open("logger/token.json", "r") as f:
                config = json.load(f)
                api_token = config["api_token"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            api_token = None

        # self.logger = NeptuneLogger(
        #     project_name="AS-DRL/DRL-Research",
        #     api_token=api_token,
        #     run_name=self.EXPERIMENT_ID,
        #     tags=["experiment", "training"]
        # )
