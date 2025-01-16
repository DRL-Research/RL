from dataclasses import dataclass


@dataclass
class CarName:
    CAR1 = "Car1"
    CAR2 = "Car2"
    CAR3 = "Car3"

@dataclass
class Role:
    CAR1 = CarName.CAR1
    CAR2 = CarName.CAR2
    BOTH = "Both"

@dataclass
class StartingLocation:
    LEFT = "Left"
    RIGHT = "Right"