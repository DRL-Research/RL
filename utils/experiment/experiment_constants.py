from dataclasses import dataclass


@dataclass
class CarName:
    CAR1 = "Car1"
    CAR2 = "Car2"


@dataclass
class Role:
    CAR1 = CarName.CAR1
    CAR2 = CarName.CAR2
    BOTH = "Both"

