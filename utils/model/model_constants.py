from dataclasses import dataclass
import random
from experiment.experiment_config import Experiment
from utils.experiment.experiment_config import Experiment



def create_full_environment_config(env_config):
    """
    Creates a full experiment configuration by combining a default configuration
    with user-specified experiment configurations. If a key in the experiment_configs
    dictionary overlaps with a key in the defaults, the value from experiment_configs
    will overwrite the default value.

    The default configuration includes various settings for observations, actions,
    rewards, and environment parameters.

    Args:
        env_config (dict): A dictionary containing configuration updates.
                                   Keys in this dictionary will override the corresponding
                                   keys in the default configuration.

    Returns:
        dict: A dictionary representing the complete experiment configuration,
              enriched with the values from experiment_configs.

    Explanation of the feature in the configuration of experiment2:
    presence - Disambiguate agents at 0 offset from non-existent agents.
    x - World offset of ego vehicle or offset to ego vehicle on the x axis.
    y - World offset of ego vehicle or offset to ego vehicle on the y axis.
    vx - Velocity on the x axis of vehicle.
    vy - Velocity on the y axis of vehicle.
    cos_h, sin_h - Trigonometric heading of vehicle.
    """
    default_config = {
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "flatten": False,
            "observe_intentions": False,
        },
        "action": {
            "type": "CustomDiscreteAction",
            "target_speeds": [5, 10],  # speed is in [m/s]
        },
        # our current reward structure is: collision: -20, arriving: +10, every step (starvation): -0.1
        "collision_reward": Experiment.COLLISION_REWARD,
        "arrived_reward": Experiment.REACHED_TARGET_REWARD,  # reward for arriving at the destination
        "normalize_reward": False,
        "starvation_reward": Experiment.STARVATION_REWARD,  # reward for every step taken
        "offroad_terminal": False,
        "duration": Experiment.EPISODE_MAX_TIME,
        "initial_vehicle_count": Experiment.CARS_AMOUNT,
        "spawn_probability": Experiment.SPAWN_PROBABILITY,
        "screen_width": Experiment.SCREEN_WIDTH,
        "screen_height": Experiment.SCREEN_HEIGHT,
        "centering_position": [0.5, 0.6],  # Do not change, this centers the simulation
        "scaling": Experiment.SCALING,
        "reward_speed_range": [7.0, 9.0],  # range of speed for which you get rewarded the high_speed_reward
        "high_speed_reward": 2,  # reward for obtaining high speed
    }
    default_config.update(env_config)  # Update with values from updates
    return default_config


CONFIG_EXP2 = {
    "controlled_cars": {
        "car1": {
        "start_lane": Experiment.SOUTH_TO_NORTH,
        "destination": Experiment.OUTER_NORTH,
        "speed": Experiment.THROTTLE_FAST,
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": 0
        },
        "color": (0, 204, 0)  # green
        }
    }
    ,
    "static_cars": {
        "car2": {
        "start_lane": Experiment.EAST_TO_WEST,
        "destination": Experiment.OUTER_EAST,
        "speed": Experiment.THROTTLE_FAST,
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": Experiment.LATERAL
            }
        }
    }
}


CONFIG_EXP1 = {
    "controlled_cars": {
        "car1": {
            "start_lane": Experiment.SOUTH_TO_NORTH,
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
            "start_lane": Experiment.WEST_TO_EAST,
            "destination": Experiment.OUTER_EAST,
            "speed": Experiment.THROTTLE_SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            }
        }
    }
    }


exp3_speed = random.choice([Experiment.THROTTLE_SLOW, Experiment.THROTTLE_FAST])
CONFIG_EXP3 = {
    "controlled_cars": {
        "car1": {
        "start_lane": Experiment.SOUTH_TO_NORTH,
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
        "start_lane": Experiment.EAST_TO_WEST,
        "destination": Experiment.OUTER_EAST,
        "speed": exp3_speed,  # Same speed for both cars
        "init_location": {
            "longitudinal": Experiment.LONGITUDINAL,
            "lateral": Experiment.LATERAL
            },
        }
    }
}

CONFIG_EXP4 = {
    "controlled_cars": {
        "car1": {
            "start_lane": Experiment.SOUTH_TO_NORTH,
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
            "start_lane": Experiment.WEST_TO_EAST,
            "destination": Experiment.OUTER_EAST,
            "speed": Experiment.THROTTLE_SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            }
        },
        "car3": {
            "start_lane": Experiment.EAST_TO_WEST,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            }
        },
        "car4": {
            "start_lane": Experiment.EAST_TO_WEST,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL - 10,
                "lateral": Experiment.LATERAL
            }
        }
    },
}


CONFIG_EXP5 = {
    "controlled_cars": {
        "car1": {
            "start_lane": Experiment.SOUTH_TO_NORTH,
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
            "start_lane": Experiment.WEST_TO_EAST,
            "destination": Experiment.OUTER_EAST,
            "speed": Experiment.THROTTLE_SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            }
        },
        "car3": {
            "start_lane": Experiment.EAST_TO_WEST,
            "destination": Experiment.OUTER_SOUTH,
            "speed": Experiment.THROTTLE_SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL - 20,
                "lateral": Experiment.LATERAL
            }
        },
        "car4": {
            "start_lane": Experiment.EAST_TO_WEST,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_FAST,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL + 10,
                "lateral": Experiment.LATERAL
            }
        },
        "car5": {
            "start_lane": Experiment.NORTH_TO_SOUTH,
            "destination": Experiment.OUTER_WEST,
            "speed": Experiment.THROTTLE_FAST,  # **Low speed for first case**
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL - 10,
                "lateral": Experiment.LATERAL
            }
        }
    }
    }

full_env_config_exp1 = create_full_environment_config(CONFIG_EXP1)
full_env_config_exp2 = create_full_environment_config(CONFIG_EXP2)
full_env_config_exp3 = create_full_environment_config(CONFIG_EXP3)
full_env_config_exp4 = create_full_environment_config(CONFIG_EXP4)
full_env_config_exp5 = create_full_environment_config(CONFIG_EXP5)
