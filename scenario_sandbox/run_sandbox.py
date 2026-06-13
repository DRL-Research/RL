"""
run_sandbox.py — Double-intersection sandbox runner.

Usage:
    python run_sandbox.py --scenario_path C:\\path\\to\\scenarios\\exp_1
    python run_sandbox.py  # prompts for path
"""

import argparse
import sys
import os

from src.experiment.experiment_config import Experiment

# Ensure imports resolve from this project root
sys.path.insert(0, os.path.dirname(__file__))


def _looks_like_valid_double_config(cfg: dict) -> bool:
    """Basic shape check to avoid crashing on malformed metadata env_config."""
    if not isinstance(cfg, dict):
        return False
    controlled = cfg.get("controlled_cars")
    statics = cfg.get("static_cars")
    return isinstance(controlled, dict) and isinstance(statics, dict)


def _setup_from_folder(folder_path: str):
    """Load double-intersection scenarios + env config from a saved folder."""
    from src.scenario_io import load_scenarios_from_folder, load_metadata
    import my_scenarios

    try:
        metadata = load_metadata(folder_path)
    except FileNotFoundError:
        metadata = {}

    env_type = metadata.get("env_type", "double_intersection")
    if env_type != "double_intersection":
        raise ValueError(
            f"Only double_intersection is supported, but metadata has env_type='{env_type}'."
        )

    env_config = metadata.get("env_config")
    if not _looks_like_valid_double_config(env_config):
        env_config = my_scenarios.DOUBLE_INTERSECTION_ENV_CONFIG

    scenarios_list = load_scenarios_from_folder(folder_path)

    # Patch the scenario module so the env classes find our scenarios
    import src.experiment.scenarios as _scenarios_module

    _scenarios_module.double_intersection_base_scenarios = scenarios_list
    env_id = "RELdouble-intersection-v0"

    from highwayenv.utils import patch_intersection_env, register_double_intersection_env
    patch_intersection_env()
    register_double_intersection_env()

    return env_id, env_config, scenarios_list


def run_scenario(scenario_index: int, env_id, env_config, scenarios_list):
    """Run one full episode using the scenario at the given index."""
    import gymnasium as gym
    import src.project_globals as project_globals

    env = gym.make(
        env_id,
        render_mode="human",
        config=env_config,
    )

    # Override to a single scenario so every reset uses this scenario.
    import src.experiment.scenarios as sc
    sc.double_intersection_base_scenarios = [scenarios_list[scenario_index]]

    # Reset project globals (needed by the env)
    project_globals.after_is_arrived_flags = [False] * len(env_config["controlled_cars"])
    project_globals.rollout_buffers = []
    project_globals.episode_count = 0

    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0
    step = 0

    print(f"\n{'='*50}")
    print(f"  Scenario {scenario_index}: running...")
    print(f"{'='*50}")

    while not (terminated or truncated):
        # Apply FASTER to every controlled vehicle (action index 1)
        num_agents = len(env.unwrapped.controlled_vehicles)
        action = tuple(1 for _ in range(num_agents))  # 1 = FASTER

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += sum(reward) if hasattr(reward, '__iter__') else reward
        step += 1

    env.close()
    print(f"  Finished after {step} steps | total reward: {total_reward:.1f}")
    print(f"  crashed={info.get('crashed', '?')}  agents_arrived={info.get('agents_dones', '?')}")


def run_experiment(experiment: Experiment):
    scenario_path = experiment.SCENARIOS_FOLDER
    if not scenario_path:
        raise ValueError(
            f"Experiment '{experiment.EXPERIMENT_ID}' is missing SCENARIOS_FOLDER."
        )

    env_id, env_config, scenarios_list = _setup_from_folder(scenario_path)

    print(
        f"\nSendbox: {len(scenarios_list)} double_intersection scenario(s) loaded from {scenario_path}"
    )

    for i in range(len(scenarios_list)):
        run_scenario(i, env_id, env_config, scenarios_list)

    print("\nAll scenarios done.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run double-intersection experiment scenarios")
    parser.add_argument(
        "--scenario_path",
        type=str,
        default=None,
        help="Optional override for the selected experiment scenario folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    current_experiment = Experiment(
        EXPERIMENT_ID="exp_221",
        SCENARIOS_FOLDER=r"C:\Users\omrih\OneDrive\Desktop\ProjectVsCode\Sendbox\scenarios\exp_1",
    )

    if args.scenario_path:
        current_experiment.SCENARIOS_FOLDER = args.scenario_path

    experiments = [current_experiment]

    for experiment in experiments:
        print(f"Starting experiment: {experiment.EXPERIMENT_ID}")
        run_experiment(experiment)
        print(f"Experiment {experiment.EXPERIMENT_ID} completed.")
