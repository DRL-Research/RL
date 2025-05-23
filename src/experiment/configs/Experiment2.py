from src.experiment.experiment_config import Experiment

CONFIG = {
    "controlled_cars": {
        "car1": {
        "start_lane": Experiment.SOUTH_FACING_NORTH,
        "destination": Experiment.OUTER_NORTH,
        "speed": Experiment.THROTTLE_FAST,
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": Experiment.LATERAL
        },
        "color": (0, 204, 0)  # green
        }
    }
    ,
    "static_cars": {
        "car2": {
        "start_lane": Experiment.EAST_FACING_WEST,
        "destination": Experiment.OUTER_EAST,
        "speed": Experiment.THROTTLE_FAST,
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": Experiment.LATERAL
            }
        }
    }
}
