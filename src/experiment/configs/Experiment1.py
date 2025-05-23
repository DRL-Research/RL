from src.experiment.experiment_config import Experiment

CONFIG = {
    "controlled_cars": {
        "controlled_car1": {
            "start_lane": Experiment.NORTH_FACING_SOUTH,
            "destination": Experiment.OUTER_WEST,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {"longitudinal": Experiment.LONGITUDINAL, "lateral": Experiment.LATERAL},
            "color": (0, 204, 0),
        },
    },
    "static_cars": {
        "static_car1": {
            "start_lane": Experiment.WEST_FACING_EAST,
            "destination": Experiment.OUTER_EAST,
            "speed": Experiment.THROTTLE_FAST,
            "init_location": {"longitudinal": Experiment.LONGITUDINAL, "lateral": Experiment.LATERAL},
        },
    },
}
