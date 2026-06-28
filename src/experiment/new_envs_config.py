"""
Environment configurations for the requested new environments.

Supported env ids:
  - RELintersection-v0
  - RELroundabout-v0
  - RELdouble-intersection-v0
"""

from copy import deepcopy

from src.experiment.scenarios_config import create_full_environment_config


_INTERSECTION_CONTROLLED_CARS = {
    "car1": {
        "start_lane": ("o0", "ir0", 0),
        "destination": "o2",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 204, 0),
    },
    "car2": {
        "start_lane": ("o1", "ir1", 0),
        "destination": "o3",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 0, 204),
    },
    "car3": {
        "start_lane": ("o2", "ir2", 0),
        "destination": "o0",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 0, 0),
    },
    "car4": {
        "start_lane": ("o3", "ir3", 0),
        "destination": "o1",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 204, 0),
    },
    "car5": {
        "start_lane": ("o0", "ir0", 0),
        "destination": "o1",
        "speed": 5,
        "init_location": {"longitudinal": 15, "lateral": 0},
        "color": (0, 204, 204),
    },
    "car6": {
        "start_lane": ("o2", "ir2", 0),
        "destination": "o3",
        "speed": 5,
        "init_location": {"longitudinal": 15, "lateral": 0},
        "color": (204, 0, 204),
    },
}

_ROUNDABOUT_CONTROLLED_CARS = {
    "car1": {
        "start_lane": ("o0", "ir0", 0),
        "destination": "o2",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 204, 0),
    },
    "car2": {
        "start_lane": ("o1", "ir1", 0),
        "destination": "o3",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 0, 204),
    },
    "car3": {
        "start_lane": ("o2", "ir2", 0),
        "destination": "o0",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 0, 0),
    },
    "car4": {
        "start_lane": ("o3", "ir3", 0),
        "destination": "o1",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 204, 0),
    },
    "car5": {
        "start_lane": ("o0", "ir0", 0),
        "destination": "o3",
        "speed": 5,
        "init_location": {"longitudinal": 15, "lateral": 0},
        "color": (0, 204, 204),
    },
    "car6": {
        "start_lane": ("o2", "ir2", 0),
        "destination": "o1",
        "speed": 5,
        "init_location": {"longitudinal": 15, "lateral": 0},
        "color": (204, 0, 204),
    },
}

_DOUBLE_INTERSECTION_CONTROLLED_CARS = {
    "car1": {
        "start_lane": ("A_o0", "A_ir0", 0),
        "destination": "A_o2",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 204, 0),
    },
    "car2": {
        "start_lane": ("A_o1", "A_ir1", 0),
        "destination": "A_o0",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 0, 204),
    },
    "car3": {
        "start_lane": ("A_o2", "A_ir2", 0),
        "destination": "A_o0",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 0, 0),
    },
    "car4": {
        "start_lane": ("B_o0", "B_ir0", 0),
        "destination": "B_o2",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 204, 0),
    },
    "car5": {
        "start_lane": ("B_o2", "B_ir2", 0),
        "destination": "B_o0",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (0, 204, 204),
    },
    "car6": {
        "start_lane": ("B_o3", "B_ir3", 0),
        "destination": "B_o2",
        "speed": 5,
        "init_location": {"longitudinal": 40, "lateral": 0},
        "color": (204, 0, 204),
    },
}


def _build_env_config(
    *,
    controlled_cars: dict,
    collision_reward: int,
    arrived_reward: int,
    starvation_reward: float,
    high_speed_reward: float,
    target_speeds: list | None,
) -> dict:
    config = create_full_environment_config(
        {
            "controlled_cars": controlled_cars,
            "static_cars": {},
            "collision_reward": collision_reward,
            "arrived_reward": arrived_reward,
            "starvation_reward": starvation_reward,
            "high_speed_reward": high_speed_reward,
        }
    )
    config["initial_vehicle_count"] = len(controlled_cars)
    config["observation"]["vehicles_count"] = len(controlled_cars)
    if target_speeds is not None:
        config["action"]["target_speeds"] = list(target_speeds)
    return config


def make_intersection_env_config(
    collision_reward: int = -50,
    arrived_reward: int = 50,
    starvation_reward: float = 0,
    high_speed_reward: float = 5,
    target_speeds: list | None = None,
) -> dict:
    """Build a full env config for the 6-car single intersection."""

    return _build_env_config(
        controlled_cars=_INTERSECTION_CONTROLLED_CARS,
        collision_reward=collision_reward,
        arrived_reward=arrived_reward,
        starvation_reward=starvation_reward,
        high_speed_reward=high_speed_reward,
        target_speeds=target_speeds,
    )


def make_roundabout_env_config(
    collision_reward: int = -50,
    arrived_reward: int = 50,
    starvation_reward: float = 0,
    high_speed_reward: float = 5,
    target_speeds: list | None = None,
) -> dict:
    """Build a full env config for the roundabout."""

    return _build_env_config(
        controlled_cars=_ROUNDABOUT_CONTROLLED_CARS,
        collision_reward=collision_reward,
        arrived_reward=arrived_reward,
        starvation_reward=starvation_reward,
        high_speed_reward=high_speed_reward,
        target_speeds=target_speeds,
    )


def make_double_intersection_env_config(
    collision_reward: int = -50,
    arrived_reward: int = 50,
    starvation_reward: float = 0,
    high_speed_reward: float = 5,
    target_speeds: list | None = None,
) -> dict:
    """Build a full env config for the double intersection."""

    return _build_env_config(
        controlled_cars=_DOUBLE_INTERSECTION_CONTROLLED_CARS,
        collision_reward=collision_reward,
        arrived_reward=arrived_reward,
        starvation_reward=starvation_reward,
        high_speed_reward=high_speed_reward,
        target_speeds=target_speeds,
    )


def with_custom_scenarios(env_config: dict, scenarios: list[dict]) -> dict:
    """Clone an env config and force sampling only from the provided scenarios."""

    configured_env = deepcopy(env_config)
    configured_env["custom_regular_scenarios"] = list(scenarios)
    configured_env["custom_regular_only"] = True
    configured_env["conflict_ratio"] = 0.0
    configured_env["use_conflict_scenarios_only"] = False
    configured_env["use_held_out_scenarios"] = False
    return configured_env
