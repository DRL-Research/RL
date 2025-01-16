from enum import Enum


class Speed(Enum):
    FAST = 10
    SLOW = 5

class Direction(Enum):
    """
    (o:outer | i:inner +[r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)
    """
    SOUTH = "0"
    WEST = "1"
    NORTH = "2"
    EAST = "3"


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
                "destination": "o2", "speed": 20,
                "init_location": 40,
                "color": (0, 204, 0)  # green
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
        "high_speed_reward": 2,  # reward for obtaining high speed
        "collision_reward": -20,
        # our current reward structure is: collision: -20, arriving: +10, every step (starvation): -0.1
        "arrived_reward": 10,  # reward for arriving at the destination
        "reward_speed_range": [7.0, 9.0],  # range of speed for which you get rewarded the high_speed_reward
        "normalize_reward": False,
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
