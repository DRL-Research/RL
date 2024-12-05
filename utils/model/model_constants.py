from dataclasses import dataclass


@dataclass
class ModelType:
    PPO = "PPO"
    DQN = "DQN"
    A2C = "A2C"


# Specific configurations for PPO and A2C
# @dataclass
# class PPO_and_A2C_Constants:
#     GAMMA: float = 0.99
#     GAE_LAMBDA: float = 0.95
#     ENT_COEF: float = 0.0
#     VF_COEF: float = 0.5
#     MAX_GRAD_NORM: float = 0.5
#     USE_SDE: bool = False
#     SDE_SAMPLE_FREQ: int = -1
#
#
# @dataclass
# class DQN_Constants:
#     BUFFER_SIZE: int = 10000
#     TARGET_UPDATE_INTERVAL: int = 1000
#     EXPLORATION_FRACTION: float = 0.1
#     EXPLORATION_FINAL_EPS: float = 0.02
#     EXPLORATION_INITIAL_EPS: float = 1.0


Policy = "MlpPolicy"

