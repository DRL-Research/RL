import json
import os
from dataclasses import replace
from datetime import datetime
from typing import Any

from src.baseline.vn_maddpg import canonicalize_algorithm_name
from src.plotting_utils.comparison_plotting import (
    DEFAULT_ALGORITHM_LABELS,
    plot_multi_seed_algorithm_comparison,
    resolve_progress_csv_path,
)
from src.project_globals import rollout_buffers
from src.training.experiment_utils import set_global_seeds
from src.training.training_handler import run_experiment


def _build_comparison_run_config(base_experiment, algorithm_name: str, seed: int):
    """Create a fresh experiment config for one algorithm/seed run."""

    normalized_algorithm = canonicalize_algorithm_name(algorithm_name)
    timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    run_experiment_id = f"{base_experiment.EXPERIMENT_ID}_{normalized_algorithm}_seed_{seed}"

    run_config = replace(
        base_experiment,
        ALGORITHM=normalized_algorithm,
        EXPERIMENT_ID=run_experiment_id,
        EXPERIMENT_DATE_TIME=timestamp,
        SEED=seed,
        SHOW_PLOTS=False,
    )
    run_config.LOAD_PREVIOUS_WEIGHT = False
    run_config.ONLY_INFERENCE = False
    run_config.RENDER_MODE = None
    return run_config


def run_multi_seed_comparison(
    base_experiment,
    env_config: dict[str, Any],
    seeds: tuple[int, ...] = (11, 22, 33),
    algorithms: tuple[str, ...] = ("experiment", "vn_maddpg", "ma_ga_ddpg"),
    moving_avg_window: int = 50,
    show_plot: bool = False,
):
    """Run each algorithm across multiple seeds and create one aggregated comparison plot."""

    comparison_timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    comparison_root = os.path.join(
        "experiments",
        f"{comparison_timestamp}_{base_experiment.EXPERIMENT_ID}_comparison",
    )
    os.makedirs(os.path.join(comparison_root, "plots"), exist_ok=True)

    progress_csvs_by_algorithm: dict[str, list[str]] = {}
    run_manifest: dict[str, list[dict[str, Any]]] = {}

    for algorithm_name in algorithms:
        normalized_algorithm = canonicalize_algorithm_name(algorithm_name)
        progress_csvs_by_algorithm[normalized_algorithm] = []
        run_manifest[normalized_algorithm] = []

        for seed in seeds:
            run_config = _build_comparison_run_config(base_experiment, normalized_algorithm, seed)
            rollout_buffers.clear()
            set_global_seeds(seed)
            run_experiment(run_config, env_config)
            progress_csv_path = resolve_progress_csv_path(run_config.EXPERIMENT_PATH)

            progress_csvs_by_algorithm[normalized_algorithm].append(progress_csv_path)
            run_manifest[normalized_algorithm].append(
                {
                    "seed": seed,
                    "experiment_path": run_config.EXPERIMENT_PATH,
                    "progress_csv": progress_csv_path,
                    "label": DEFAULT_ALGORITHM_LABELS.get(normalized_algorithm, normalized_algorithm),
                }
            )

    comparison_plot_path = os.path.join(comparison_root, "plots", "multi_seed_algorithm_comparison.png")
    plot_summary = plot_multi_seed_algorithm_comparison(
        progress_csvs_by_algorithm=progress_csvs_by_algorithm,
        output_path=comparison_plot_path,
        moving_avg_window=moving_avg_window,
        show_plot=show_plot,
    )

    summary = {
        "comparison_root": comparison_root,
        "plot_path": comparison_plot_path,
        "moving_avg_window": moving_avg_window,
        "seeds": list(seeds),
        "algorithms": list(progress_csvs_by_algorithm.keys()),
        "runs": run_manifest,
        "plot_summary": plot_summary,
    }
    summary_path = os.path.join(comparison_root, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return summary
