from dataclasses import dataclass


# @dataclass
# class CarName:
#     CAR1 = "Car1"
#     CAR2 = "Car2"
#
#
# @dataclass
# class Role:
#     CAR1 = CarName.CAR1
#     CAR2 = CarName.CAR2
#     BOTH = "Both"


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


"""
Explanation of the feature in the configuration of experiment2:
    presence - Disambiguate agents at 0 offset from non-existent agents.
    x - World offset of ego vehicle or offset to ego vehicle on the x axis.
    y - World offset of ego vehicle or offset to ego vehicle on the y axis.
    vx - Velocity on the x axis of vehicle.
    vy - Velocity on the y axis of vehicle.
    cos_h, sin_h - Trigonometric heading of vehicle.
"""

CONFIG_EXP2 = (
    {
        "car1": {
            "start_lane": ("o0", "ir0", 0),
            "destination": "o2",
            "speed": Speed.FAST,
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            },
            "color": (0, 204, 0)  # green
        },
        "car2": {
            "start_lane": ("o1", "ir1", 0),
            "destination": "o3",
            "speed": Speed.FAST,
            "init_location": {
                "longitudinal": 40,
                "lateral": 0
            },
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 2,
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
            "target_speeds": [5, 10],  # speed are in [m/s]
        },
        "high_speed_reward": 7,  # reward for obtaining high speed
        "collision_reward": -27,
        # our current reward structure is: collision: -20, arriving: +10, every step (starvation): -0.1
        "arrived_reward": 17,  # reward for arriving at the destination
        "reward_speed_range": [7.0, 9.0],  # range of speed for which you get rewarded the high_speed_reward
        "normalize_reward": False,
        "starvation_reward": -0.1,
        "offroad_terminal": False,
        "duration": 13,  # [s]
        "initial_vehicle_count": 0,
        "spawn_probability": 0.6,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],  # Do not change, this centers the simulation
        "scaling": 5.5 * 1.3,
        # "destination": "o2",
        # "controlled_vehicles": 1,
    })
