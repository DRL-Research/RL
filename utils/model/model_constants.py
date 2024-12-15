from dataclasses import dataclass


@dataclass
class ModelType:
    PPO = "PPO"
    DQN = "DQN"
    A2C = "A2C"


Policy = "MlpPolicy"

