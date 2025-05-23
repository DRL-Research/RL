from src.experiment.experiment_config import Experiment
import random

exp3_speed = random.choice([Experiment.THROTTLE_SLOW, Experiment.THROTTLE_FAST])
CONFIG = {
    "controlled_cars": {
        "car1": {
        "start_lane": Experiment.SOUTH_FACING_NORTH,
        "destination": Experiment.OUTER_NORTH,
        "speed": exp3_speed,  # Same speed for both cars
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": Experiment.LATERAL
        }
            },
        "color": (0, 204, 0)  # Green car
    },
    "static_cars": {

        "car2": {
        "start_lane": Experiment.EAST_FACING_WEST,
        "destination": Experiment.OUTER_EAST,
        "speed": exp3_speed,  # Same speed for both cars
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": Experiment.LATERAL
            },
        }
    }
}