"""
Scenario Configuration for QMIX Experiments
4-way intersection with 4 controlled vehicles (1 from each direction)
"""

from src.experiment.experiment_config import Experiment
from src.project_globals import after_is_arrived_flags


def create_full_environment_config(env_config):
    """
    Creates a full experiment configuration by combining a default configuration
    with user-specified experiment configurations.
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
            "type": "CustomMultiAgentAction",
            "action_config": {
                "type": "CustomDiscreteAction",
            },
            "target_speeds": [5, 10],  # speed is in [m/s] - 2 discrete actions
        },
        # Reward structure
        "collision_reward": Experiment.COLLISION_REWARD,
        "arrived_reward": Experiment.REACHED_TARGET_REWARD,
        "normalize_reward": False,
        "starvation_reward": Experiment.STARVATION_REWARD,
        "offroad_terminal": False,
        "duration": Experiment.EPISODE_MAX_TIME,
        "initial_vehicle_count": Experiment.CARS_AMOUNT,
        "spawn_probability": Experiment.SPAWN_PROBABILITY,
        "screen_width": Experiment.SCREEN_WIDTH,
        "screen_height": Experiment.SCREEN_HEIGHT,
        "centering_position": [0.5, 0.6],
        "scaling": Experiment.SCALING,
    }
    default_config.update(env_config)
    return default_config


# 4-way intersection: 1 controlled vehicle from each direction (matching paper)
CONFIG_QMIX_4_AGENTS = {
    "controlled_cars": {
        "car1": {
            "start_lane": Experiment.SOUTH_TO_NORTH,
            "destination": Experiment.OUTER_NORTH,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            },
            "color": (0, 204, 0)  # Green - from South
        },
        "car2": {
            "start_lane": Experiment.EAST_TO_WEST,
            "destination": Experiment.OUTER_WEST,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            },
            "color": (0, 0, 204)  # Blue - from East
        },
        "car3": {
            "start_lane": Experiment.WEST_TO_EAST,
            "destination": Experiment.OUTER_EAST,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            },
            "color": (204, 0, 0)  # Red - from West
        },
        "car4": {
            "start_lane": Experiment.NORTH_TO_SOUTH,
            "destination": Experiment.OUTER_SOUTH,
            "speed": Experiment.THROTTLE_SLOW,
            "init_location": {
                "longitudinal": Experiment.LONGITUDINAL,
                "lateral": Experiment.LATERAL
            },
            "color": (204, 204, 0)  # Yellow - from North
        },
    },
    "static_cars": {}  # No static cars for QMIX experiments
}


# Use the QMIX configuration
current_experiment = CONFIG_QMIX_4_AGENTS

# Create full configuration
full_env_config_exp5 = create_full_environment_config(current_experiment)

# Initialize arrival flags
after_is_arrived_flags.clear()  # Clear existing flags
for _ in current_experiment["controlled_cars"]:
    after_is_arrived_flags.append(False)

print(f"Scenario configured with {len(current_experiment['controlled_cars'])} controlled vehicles")