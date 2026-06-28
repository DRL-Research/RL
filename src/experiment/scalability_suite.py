from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime
from typing import Any

from highwayenv.intersection_class import rotate_scenario_clockwise
from src.experiment.comparison_runner import run_multi_seed_comparison
from src.experiment.env_utils import (
    DOUBLE_INTERSECTION_ENV_ID,
    INTERSECTION_ENV_ID,
    ROUNDABOUT_ENV_ID,
    register_all_supported_envs,
)
from src.experiment.new_envs_config import (
    make_double_intersection_env_config,
    make_intersection_env_config,
    make_roundabout_env_config,
    with_custom_scenarios,
)
from src.experiment import scenarios as scenario_library
from src.plotting_utils.comparison_plotting import plot_multi_seed_algorithm_comparison


INTERSECTION_BASE_REGULAR_COUNT = 25
ROUNDABOUT_BASE_REGULAR_COUNT = 15
DOUBLE_INTERSECTION_BASE_REGULAR_COUNT = 20


def _expand_rotated_scenarios(base_scenarios: list[dict]) -> list[dict]:
    expanded_scenarios: list[dict] = []

    for base_scenario in base_scenarios:
        expanded_scenarios.append(base_scenario)
        for rotation in (1, 2, 3):
            expanded_scenarios.append(
                {
                    "agents": [
                        rotate_scenario_clockwise([agent], rotation)[0]
                        for agent in base_scenario["agents"]
                    ],
                    "static": [
                        rotate_scenario_clockwise([static_vehicle], rotation)[0]
                        for static_vehicle in base_scenario["static"]
                    ],
                }
            )

    return expanded_scenarios


def _selected_regular_scenarios(
    scenario_pool: list[dict],
    *,
    default_base_count: int,
    include_extra_regular: bool,
) -> list[dict]:
    if include_extra_regular:
        return list(scenario_pool)
    return list(scenario_pool[:default_base_count])


def build_requested_suite_envs(*, include_extra_regular: bool = False) -> dict[str, dict[str, Any]]:
    """Build env specs for the exact scenario groups requested by the user."""

    intersection_regular = _selected_regular_scenarios(
        scenario_library.base_complete_scenarios_6_cars,
        default_base_count=INTERSECTION_BASE_REGULAR_COUNT,
        include_extra_regular=include_extra_regular,
    )
    roundabout_regular = _selected_regular_scenarios(
        scenario_library.roundabout_base_scenarios,
        default_base_count=ROUNDABOUT_BASE_REGULAR_COUNT,
        include_extra_regular=include_extra_regular,
    )
    double_intersection_regular = _selected_regular_scenarios(
        scenario_library.double_intersection_base_scenarios,
        default_base_count=DOUBLE_INTERSECTION_BASE_REGULAR_COUNT,
        include_extra_regular=include_extra_regular,
    )

    intersection_scenarios = _expand_rotated_scenarios(intersection_regular) + _expand_rotated_scenarios(
        list(scenario_library.conflict_base_scenarios)
    )
    roundabout_scenarios = _expand_rotated_scenarios(roundabout_regular) + _expand_rotated_scenarios(
        list(scenario_library.roundabout_conflict_base_scenarios)
    )
    double_intersection_scenarios = list(double_intersection_regular) + list(
        scenario_library.double_intersection_conflict_base_scenarios
    )

    return {
        "single_intersection": {
            "label": "Single Intersection",
            "env_id": INTERSECTION_ENV_ID,
            "scenario_count": len(intersection_scenarios),
            "env_config": with_custom_scenarios(
                make_intersection_env_config(),
                intersection_scenarios,
            ),
        },
        "roundabout": {
            "label": "Roundabout",
            "env_id": ROUNDABOUT_ENV_ID,
            "scenario_count": len(roundabout_scenarios),
            "env_config": with_custom_scenarios(
                make_roundabout_env_config(),
                roundabout_scenarios,
            ),
        },
        "double_intersection": {
            "label": "Double Intersection",
            "env_id": DOUBLE_INTERSECTION_ENV_ID,
            "scenario_count": len(double_intersection_scenarios),
            "env_config": with_custom_scenarios(
                make_double_intersection_env_config(),
                double_intersection_scenarios,
            ),
        },
    }


def run_requested_suite_comparison(
    base_experiment,
    *,
    seeds: tuple[int, ...] = (11, 22, 33),
    algorithms: tuple[str, ...] = ("experiment", "vn_maddpg", "ma_ga_ddpg"),
    moving_avg_window: int = 50,
    show_plot: bool = False,
    include_extra_regular: bool = False,
) -> dict[str, Any]:
    """Run the 3-model comparison across all requested layouts and aggregate one overall plot."""

    register_all_supported_envs()

    suite_timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    suite_root = os.path.join(
        "experiments",
        f"{suite_timestamp}_{base_experiment.EXPERIMENT_ID}_requested_suite",
    )
    os.makedirs(os.path.join(suite_root, "plots"), exist_ok=True)

    overall_progress_csvs_by_algorithm: dict[str, list[str]] = {}
    per_environment_summary: dict[str, Any] = {}
    suite_envs = build_requested_suite_envs(include_extra_regular=include_extra_regular)

    for env_key, env_spec in suite_envs.items():
        env_experiment = replace(
            base_experiment,
            EXPERIMENT_ID=f"{base_experiment.EXPERIMENT_ID}_{env_key}",
            ENV_ID=env_spec["env_id"],
            CARS_AMOUNT=len(env_spec["env_config"]["controlled_cars"]),
            SHOW_PLOTS=False,
        )

        comparison_summary = run_multi_seed_comparison(
            base_experiment=env_experiment,
            env_config=env_spec["env_config"],
            seeds=seeds,
            algorithms=algorithms,
            moving_avg_window=moving_avg_window,
            show_plot=False,
        )

        per_environment_summary[env_key] = {
            "label": env_spec["label"],
            "env_id": env_spec["env_id"],
            "scenario_count": env_spec["scenario_count"],
            **comparison_summary,
        }

        for algorithm_name, run_rows in comparison_summary["runs"].items():
            overall_progress_csvs_by_algorithm.setdefault(algorithm_name, [])
            overall_progress_csvs_by_algorithm[algorithm_name].extend(
                run_row["progress_csv"] for run_row in run_rows
            )

    overall_plot_path = os.path.join(suite_root, "plots", "all_requested_scenarios_comparison.png")
    overall_plot_summary = plot_multi_seed_algorithm_comparison(
        progress_csvs_by_algorithm=overall_progress_csvs_by_algorithm,
        output_path=overall_plot_path,
        moving_avg_window=moving_avg_window,
        show_plot=show_plot,
    )

    summary = {
        "suite_root": suite_root,
        "overall_plot_path": overall_plot_path,
        "include_extra_regular": include_extra_regular,
        "moving_avg_window": moving_avg_window,
        "seeds": list(seeds),
        "algorithms": list(algorithms),
        "environments": per_environment_summary,
        "overall_plot_summary": overall_plot_summary,
    }
    summary_path = os.path.join(suite_root, "requested_suite_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return summary
