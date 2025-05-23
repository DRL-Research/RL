from src.experiment.experiment_config import Experiment

CONFIG = {
    "controlled_cars": {
        "car1": {
            "start_lane": Experiment.SOUTH_FACING_NORTH,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,  # Initial speed is FAST
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            },
            "color": (0, 204, 0)  # Green car
        }
    },
    "static_cars": {
        "car2": {
            "start_lane": Experiment.WEST_FACING_EAST,
            "destination": Experiment.OUTER_EAST,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            }
        },
        "car3": {
            "start_lane": Experiment.EAST_FACING_WEST,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            }
        },
        "car4": {
            "start_lane": Experiment.EAST_FACING_WEST,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL - 10,
                "lateral": Experiment.LATERAL
            }
        }
    },
}