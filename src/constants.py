class StartingLocation:
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class CarName:
    CAR1 = "Car1"
    CAR2 = "Car2"


class Role:
    CAR1 = CarName.CAR1
    CAR2 = CarName.CAR2
    BOTH = "Both"


class ModelType:
    PPO = "PPO"
    DQN = "DQN"
    A2C = "A2C"


class Policy:
    MlpPolicy = "MlpPolicy"
