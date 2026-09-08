from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Experiment:
    CARS_AMOUNT: int = 6
    SPAWN_PROBABILITY: float = 0
    RENDER_MODE: str | None = None
    EPISODE_MAX_TIME: int = 200
    SCREEN_WIDTH: int = 900
    SCREEN_HEIGHT: int = 800
    SCALING: float = 3 * 1.3
    STARVATION_REWARD: float = 0.0
    HIGH_SPEED_REWARD: float = 5.0
    COLLISION_REWARD: float = -50.0
    REACHED_TARGET_REWARD: float = 50.0
    FIXED_THROTTLE: float = 12.0
    EMBEDDING_SIZE: int = 4
    NUM_MASTER_SLOTS: int = 5
    MASTER_OBS_DIM: int = 25
    STATE_INPUT_SIZE: int = 8
    ACTION_SPACE_SIZE: int = 2
    AGENT_LR: float = 3e-3
    MASTER_LR: float = 3e-4
    GAMMA: float = 0.90
    CLIP_RANGE: float = 0.2
    VF_COEF: float = 1.0
    ENT_COEF: float = 0.005
    N_STEPS: int = 4096
    WARMUP_EPISODES: int = 0
    PEAK_ARRIVAL_THRESHOLD: float = 0.0
    EXPERIMENT_ID: str = "hdmaps"
    EXPERIMENT_DATE_TIME: str = field(default_factory=lambda: datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
    CENTERING_POSITION: List[float] = field(default_factory=lambda: [0.5, 0.6])
    EXPERIMENT_PATH: str = ""
    SAVE_MODEL_DIRECTORY: str = ""
