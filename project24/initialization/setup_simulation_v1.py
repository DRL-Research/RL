from typing import NamedTuple, Dict
from project24.initialization.config_v1 import *
import airsim

class Car(NamedTuple):
    name: str
    speed: float
    yaw: float
    initial_position: list
    destination: dict
    is_active: bool

    def get_start_location_x(self, offset):
        return self.initial_position[0] - offset

    def get_start_location_y(self, offset, side=None):
        if side is None:
            return self.initial_position[1] - offset
        else:
            return side * self.initial_position[1] - offset


class CarDict(Dict[str, Car]):
    pass


class SetupManager:
    def __init__(self):
        self.cars = CarDict()
        self.max_cars_we_can_handle = 4
        car1 = Car(name=CAR1_NAME, speed=0.0, yaw=CAR1_INITIAL_YAW, initial_position=CAR1_INITIAL_POSITION,
                   destination=CAR1_DESIRED_POSITION, is_active=USE_CAR1)
        car2 = Car(name=CAR2_NAME, speed=0.0, yaw=CAR2_INITIAL_YAW, initial_position=CAR2_INITIAL_POSITION,
                   destination=CAR2_DESIRED_POSITION, is_active=USE_CAR2)
        # car3 = Car(name=CAR3_NAME, speed=0.0, yaw=CAR3_INITIAL_YAW, initial_position=CAR3_INITIAL_POSITION,
        #            destination=CAR3_DESIRED_POSITION, is_active=USE_CAR3)
        # car4 = Car(name=CAR4_NAME, speed=0.0, yaw=CAR4_INITIAL_YAW, initial_position=CAR4_INITIAL_POSITION,
        #            destination=CAR4_DESIRED_POSITION, is_active=USE_CAR4)
        cars = [car1, car2]
        for car in cars:
            if car.is_active:
                self.cars[car.name] = car

        self.cars_names = list(self.cars.keys())
        self.n_active_cars = len(self.cars)
        if self.n_active_cars != NUMBER_OF_CAR_IN_SIMULATION:
            raise Exception('MisMatch between the number of cars in the simulation and the number of active cars')
