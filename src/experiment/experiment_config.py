import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class Experiment:
    LOAD_PREVIOUS_WEIGHT = True
    BYPASS_RANDOM_INITIALIZATION = False

    # General Experiment Settings
    EPISODES_PER_CYCLE: int = 300
    CYCLES: int = 4
    EXPERIMENT_ID: str = "hierarchical_6cars"
    ONLY_INFERENCE: bool = False
    EXPERIMENT_DATE_TIME: str = field(default_factory=lambda: datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
    SELF_PLAY_MODE: bool = False
    MASTER_TRAINED_MODEL: str = "EXP5_Inference_Models/master_trained_model.zip"
    AGENT_TRAINED_MODEL: str = "EXP5_Inference_Models/agent_trained_model.zip"
    CARS_AMOUNT: int = 6  # 6 controlled vehicles
    SPAWN_PROBABILITY: float = 0
    RENDER_MODE: str|None = "rgb_array"

    # Model and Training Configuration
    EPISODE_AMOUNT_FOR_TRAIN: int = 5  # accumulate 5 episodes then train (~60-100 steps/batch)
    EPOCHS: int = None
    LEARNING_RATE: float = 0.005  # Reduced from 0.05
    # Rollout length for SB3 buffers. 3 episodes × ~100 steps ≈ 300; keep modest margin
    # so compute_returns_gae_on_prefix rarely sees huge empty tails.
    N_STEPS: int = 384

    # ── Reward / training-schedule knobs (used by sweep) ─────────────────────────
    # 'global' → every network gets the global min reward (simplest cooperative)
    # 'group'  → agents + their LM share the group-min; GM gets global min
    AGENT_REWARD_MODE: str = 'group'
    # True  → cycles 1-3 all co-train (master+agents together); cycle 4 = GM only
    # False → original separated schedule (1=both, 2=LM only, 3=agents only, 4=GM)
    COTRAIN_CYCLES: bool = True
    # If True: *every* cycle runs train_both (LM+GM+agents together). No alternating
    # freeze — removes visible cycle boundaries and non-stationarity from frozen peers.
    FULL_JOINT_TRAINING: bool = False

    # ── Learning-rate / entropy / optimisation knobs (swept in grid search) ──────
    # Entropy coefficient for the PPO loss: loss += -ent_coef * H(π)
    # Higher → policy stays stochastic longer (prevents degenerate "always slow").
    # ENT_COEF is the initial (start-of-training) value.
    # ENT_COEF_FINAL is the value at the last episode (linear annealing).
    # Set ENT_COEF_FINAL == ENT_COEF to disable annealing (constant entropy).
    ENT_COEF: float = 0.01
    ENT_COEF_FINAL: float = 0.0
    # Per-agent PPO learning rate
    AGENT_LR: float = 7e-4
    # Shared Master PPO learning rate
    MASTER_LR: float = 1e-4

    # ── PPO core hyper-parameters (affect both agent and master) ─────────────────
    # Discount factor used in RolloutBuffer.compute_returns_and_advantage
    GAMMA: float = 0.99
    # GAE lambda — higher → lower variance, higher bias in advantage estimates
    GAE_LAMBDA: float = 0.95
    # PPO clipping range for the surrogate ratio objective (used in training_loop_utils)
    CLIP_RANGE: float = 0.2

    # ── Agent network architecture ────────────────────────────────────────────────
    # 'tiny'    → [16, 16]
    # 'small'   → [32, 32]   (current default)
    # 'medium'  → [64, 64]
    # 'large'   → [128, 128]
    # 'deep'    → [64, 64, 64]
    # 'wide'    → [256, 256]
    # 'deep_lg' → [128, 128, 128]
    AGENT_NET_ARCH: str = 'small'

    # ── PPO loss coefficients ──────────────────────────────────────────────────────
    # Weight of the value-function loss term (0.5 × MSE)
    VF_COEF: float = 0.5
    # Number of gradient update epochs over the same rollout buffer per training call.
    # Standard PPO uses 10; we currently do 1 (vanilla PG efficiency).
    N_PPO_EPOCHS: int = 1

    # Extra epochs of PURE VALUE-FUNCTION updates run BEFORE the joint PPO epochs.
    # During these epochs only the value loss is backpropagated — the policy head
    # is effectively frozen (no policy_loss term, no entropy term).
    # This helps V(s) converge to accurate return estimates BEFORE the policy
    # gradient step, reducing the "moving target" problem (rising value loss).
    # 0 = disabled (current behaviour).  Recommended: 10–20.
    N_VALUE_EPOCHS: int = 0

    # ── Per-step environment rewards ──────────────────────────────────────────────
    # These flow into make_env_config_exp7 so they are fully sweepable.
    # Currently: starvation=-5 (every step), high_speed=+5 (when going fast).
    # Setting both to 0 gives pure terminal-only rewards (simpler signal).
    STARVATION_REWARD: float = -5
    HIGH_SPEED_REWARD: float = 5
    BATCH_SIZE: int = 32
    EPISODE_MAX_TIME = 50  # seconds
    LOSS_FUNCTION: str = "mse"
    # Global env steps before switching from random {0,1} to policy actions.
    # Overridden automatically when WARMUP_EPISODES > 0 (see training_handler).
    # 0 = always use policy.
    EXPLORATION_EXPLOITATION_THRESHOLD: int = 0

    # ── Warm-up & peak-lock (exploration schedule) ────────────────────────────
    # Number of episodes at the start of training where agents take RANDOM actions
    # (uniform {0,1}).  Master already explores via stochastic sampling during this
    # phase.  0 = disabled (policy used from episode 1).
    WARMUP_EPISODES: int = 0

    # When rolling-20 arrival rate reaches or exceeds this value (%), ent_coef is
    # permanently set to 0 for all subsequent training updates — "peak lock".
    # 0.0 = disabled.  Example: 85.0 locks entropy once 85 % arrival is sustained.
    PEAK_ARRIVAL_THRESHOLD: float = 0.0

    # Cars Configuration - distance from intersection
    LONGITUDINAL: int = 40
    LATERAL: int = 0

    # Master embedding size configuration
    EMBEDDING_SIZE: int = 4

    # Hierarchical master configuration
    # Each Master slot = [4D state/embedding + 1D identifier bit]
    NUM_MASTER_SLOTS: int = 5        # max subordinates per master (pad with zeros if fewer)
    NUM_LOCAL_MASTERS: int = 2       # Local Master 1 → agents 0-2, Local Master 2 → agents 3-5
    AGENTS_PER_LOCAL_MASTER: int = 3
    MASTER_OBS_DIM: int = NUM_MASTER_SLOTS * 5  # 5 slots × (4D state + 1D identifier bit) = 25

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

    # Reward Configuration (instance fields — all sweepable)
    REACHED_TARGET_REWARD: int = 50
    COLLISION_REWARD: int = -300

    # Path Configuration
    LOAD_MODEL_DIRECTORY: str = ""  # Directory for loading weights
    MODEL_TYPE: str = "PPO"  # Model type (e.g., PPO, DQN)
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


        # Neptune logging is optional; gracefully skip if token file is absent.
        # self.logger = NeptuneLogger(
        #     project_name="AS-DRL/DRL-Research",
        #     api_token=api_token,
        #     run_name=self.EXPERIMENT_ID,
        #     tags=["experiment", "training"]
        # )