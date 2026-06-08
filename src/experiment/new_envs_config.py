"""
Environment configurations for the two new environments:
  - RELroundabout-v0        (node names identical to intersection: o0-o3, ir0-ir3)
  - RELdouble-intersection-v0 (A_/B_ prefixed node names)

Both use 6 controlled cars split as LM1: agents 0-2 | LM2: agents 3-5.
"""

from src.experiment.scenarios_config import create_full_environment_config


# ── Roundabout ────────────────────────────────────────────────────────────────
# Uses the same node naming as the single intersection, so the same
# controlled_cars layout works.  _reset re-positions cars from scenarios.

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


def make_roundabout_env_config(
    collision_reward: int   = -50,
    arrived_reward: int     = 50,
    starvation_reward: float = 0,
    high_speed_reward: float = 5,
    target_speeds: list     = None,
) -> dict:
    """Build a full env config for the roundabout."""
    base = {
        "controlled_cars": _ROUNDABOUT_CONTROLLED_CARS,
        "static_cars": {},
        "collision_reward":  collision_reward,
        "arrived_reward":    arrived_reward,
        "starvation_reward": starvation_reward,
        "high_speed_reward": high_speed_reward,
    }
    cfg = create_full_environment_config(base)
    if target_speeds is not None:
        cfg["action"]["target_speeds"] = list(target_speeds)
    return cfg


# ── Double Intersection ───────────────────────────────────────────────────────
# Cars 0-2 start at intersection A (LM1), Cars 3-5 start at intersection B (LM2).
# _reset re-positions cars from double_intersection_base_scenarios.

_DOUBLE_INTERSECTION_CONTROLLED_CARS = {
    # ── LM1: intersection A ───────────────────────────────────────────────────
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
    # ── LM2: intersection B ───────────────────────────────────────────────────
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


def make_double_intersection_env_config(
    collision_reward: int   = -50,
    arrived_reward: int     = 50,
    starvation_reward: float = 0,
    high_speed_reward: float = 5,
    target_speeds: list     = None,
) -> dict:
    """Build a full env config for the double intersection."""
    base = {
        "controlled_cars": _DOUBLE_INTERSECTION_CONTROLLED_CARS,
        "static_cars": {},
        "collision_reward":  collision_reward,
        "arrived_reward":    arrived_reward,
        "starvation_reward": starvation_reward,
        "high_speed_reward": high_speed_reward,
    }
    cfg = create_full_environment_config(base)
    if target_speeds is not None:
        cfg["action"]["target_speeds"] = list(target_speeds)
    return cfg
