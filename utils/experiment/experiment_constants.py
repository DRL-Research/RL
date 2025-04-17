from dataclasses import dataclass
import random


@dataclass
class Speed:
    FAST = 10
    SLOW = 5

@dataclass
class Lanes:
    SOUTH_TO_NORTH = ("o0", "ir0", 0)
    EAST_TO_WEST = ("o1", "ir1", 0)
    WEST_TO_EAST = ("o3", "ir3", 0)
    NORTH_TO_SOUTH = ("o2", "ir2", 0)


@dataclass
class Direction:
    """
    (o:outer | i:inner +[r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)
    """
    OUTER_SOUTH = "o0"
    OUTER_WEST = "o1"
    OUTER_NORTH = "o2"
    OUTER_EAST = "o3"
    INNER_SOUTH = "i0"
    INNER_WEST = "i1"
    INNER_NORTH = "i2"
    INNER_EAST = "i3"


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
        "high_speed_reward": 2,  # reward for obtaining high speed
        "collision_reward": -20,
        "arrived_reward": 10,  # reward for arriving at the destination
        "reward_speed_range": [7.0, 9.0],  # range of speed for which you get rewarded the high_speed_reward
        "normalize_reward": False,
        "starvation_reward": -0.1,  # reward for every step taken
        "offroad_terminal": False,
        "duration": 13,  # [s]
        "initial_vehicle_count": 0,
        "spawn_probability": 0.6,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],  # Do not change, this centers the simulation
        "scaling": 5.5 * 1.3,
    }
    default_config.update(env_config)  # Update with values from updates
    return default_config


CONFIG_EXP2 = {
    "controlled_cars": {
        "car1": {
        "start_lane": Lanes.SOUTH_TO_NORTH,
        "destination": Direction.OUTER_NORTH,
        "speed": Speed.FAST,
        "init_location": {
            "longitudinal": 40,
            "lateral": 0
        },
        "color": (0, 204, 0)  # green
        }
    }
    ,
    "static_cars": {
        "car2": {
        "start_lane": Lanes.EAST_TO_WEST,
        "destination": Direction.OUTER_EAST,
        "speed": Speed.FAST,
        "init_location": {
            "longitudinal": 40,
            "lateral": 0
            }
        }
    }
}


CONFIG_EXP1 = {
    "controlled_cars": {
        "car1": {
            "start_lane": Lanes.SOUTH_TO_NORTH,
            "destination": Direction.OUTER_NORTH,
            "speed": Speed.SLOW,  # Initial speed is FAST
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            },
            "color": (0, 204, 0)  # Green car
        }
    },
    "static_cars": {
        "car2": {
            "start_lane": Lanes.WEST_TO_EAST,
            "destination": Direction.OUTER_EAST,
            "speed": Speed.SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            }
        }
    }
    }


exp3_speed = random.choice([Speed.SLOW, Speed.FAST])
CONFIG_EXP3 = {
    "controlled_cars": {
        "car1": {
        "start_lane": Lanes.SOUTH_TO_NORTH,
        "destination": Direction.OUTER_NORTH,
        "speed": exp3_speed,  # Same speed for both cars
        "init_location": {
            "longitudinal": 40,
            "lateral": 0
        }
            },
        "color": (0, 204, 0)  # Green car
    },
    "static_cars": {

        "car2": {
        "start_lane": Lanes.EAST_TO_WEST,
        "destination": Direction.OUTER_EAST,
        "speed": exp3_speed,  # Same speed for both cars
        "init_location": {
            "longitudinal": 40,
            "lateral": 0
            },
        }
    }
}

CONFIG_EXP4 = {
    "controlled_cars": {
        "car1": {
            "start_lane": Lanes.SOUTH_TO_NORTH,
            "destination": Direction.OUTER_NORTH,
            "speed": Speed.SLOW,  # Initial speed is FAST
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            },
            "color": (0, 204, 0)  # Green car
        }
    },
    "static_cars": {
        "car2": {
            "start_lane": Lanes.WEST_TO_EAST,
            "destination": Direction.OUTER_EAST,
            "speed": Speed.SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            }
        },
        "car3": {
            "start_lane": Lanes.EAST_TO_WEST,
            "destination": Direction.OUTER_NORTH,
            "speed": Speed.FAST,
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            }
        },
        "car4": {
            "start_lane": Lanes.EAST_TO_WEST,
            "destination": Direction.OUTER_NORTH,
            "speed": Speed.SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 30,
                "lateral": 0
            }
        }
    },
}


CONFIG_EXP5 = {
    "controlled_cars": {
        "car1": {
            "start_lane": Lanes.SOUTH_TO_NORTH,
            "destination": Direction.OUTER_NORTH,
            "speed": Speed.SLOW,  # Initial speed is FAST
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            },
            "color": (0, 204, 0)  # Green car
        }
    },
    "static_cars": {
        "car2": {
            "start_lane": Lanes.WEST_TO_EAST,
            "destination": Direction.OUTER_EAST,
            "speed": Speed.SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            }
        },
        "car3": {
            "start_lane": Lanes.EAST_TO_WEST,
            "destination": Direction.OUTER_SOUTH,
            "speed": Speed.SLOW,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 20,
                "lateral": 0
            }
        },
        "car4": {
            "start_lane": Lanes.EAST_TO_WEST,
            "destination": Direction.OUTER_NORTH,
            "speed": Speed.FAST,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 50,
                "lateral": 0
            }
        },
        "car5": {
            "start_lane": Lanes.NORTH_TO_SOUTH,
            "destination": Direction.OUTER_WEST,
            "speed": Speed.FAST,  # **Low speed for first case**
            "init_location": {
                "longitudinal": 30,
                "lateral": 0
            }
        }
    }
    }

full_env_config_exp1 = create_full_environment_config(CONFIG_EXP1)
full_env_config_exp2 = create_full_environment_config(CONFIG_EXP2)
full_env_config_exp3 = create_full_environment_config(CONFIG_EXP3)
full_env_config_exp4 = create_full_environment_config(CONFIG_EXP4)
full_env_config_exp5 = create_full_environment_config(CONFIG_EXP5)

