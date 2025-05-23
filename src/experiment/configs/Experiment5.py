from src.experiment.experiment_config import Experiment

CONFIG = {
    "controlled_cars": {
        "car1": {
            "start_lane": Experiment.SOUTH_FACING_NORTH,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {"longitudinal": Experiment.LONGITUDINAL, "lateral": Experiment.LATERAL},
            "color": (0, 204, 0),
        },
    },
    "static_cars": {
        "car2": {
            "start_lane": Experiment.WEST_FACING_EAST,
            "destination": Experiment.OUTER_WEST,
            "speed": Experiment.THROTTLE_FAST,
            "init_location": {"longitudinal": Experiment.LONGITUDINAL, "lateral": Experiment.LATERAL},
        },
        "car3": {
            "start_lane": Experiment.EAST_FACING_WEST,
            "destination": Experiment.OUTER_SOUTH,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {"longitudinal": 20.0, "lateral": Experiment.LATERAL},
        },
        "car4": {
            "start_lane": Experiment.EAST_FACING_WEST,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_FAST,
            "init_location": {"longitudinal": 50.0, "lateral": Experiment.LATERAL},
        },
        "car5": {
            "start_lane": Experiment.NORTH_FACING_SOUTH,
            "destination": Experiment.OUTER_WEST,
            "speed": Experiment.THROTTLE_FAST,
            "init_location": {"longitudinal": 30.0, "lateral": Experiment.LATERAL},
        },
    },
}
