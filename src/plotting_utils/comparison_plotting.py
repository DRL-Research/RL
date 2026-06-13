import csv
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ALGORITHM_LABELS = {
    "experiment": "MAPS",
    "vn_maddpg": "VN-MA-DDPG",
    "ma_ga_ddpg": "MA-GA-DDPG",
}

DEFAULT_ALGORITHM_COLORS = {
    "experiment": "#1f77b4",
    "vn_maddpg": "#9467bd",
    "ma_ga_ddpg": "#2ca02c",
}


def resolve_progress_csv_path(experiment_path: str) -> str:
    """Locate the comparable per-episode metrics CSV for one training run."""

    candidate_paths = [
        os.path.join(experiment_path, "comparison_logs", "progress.csv"),
        os.path.join(experiment_path, "baseline_logs", "progress.csv"),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(f"Could not find a comparable progress CSV under '{experiment_path}'.")


def _read_progress_csv(csv_path: str) -> dict[str, np.ndarray]:
    """Load one run's per-episode metrics from CSV."""

    episodes = []
    rewards = []
    successes = []
    collisions = []
    episode_lengths = []

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"episode", "reward", "success", "collision", "episode_length"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV '{csv_path}' is missing required columns: {sorted(required_columns)}."
            )

        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            successes.append(float(row["success"]))
            collisions.append(float(row["collision"]))
            episode_lengths.append(float(row["episode_length"]))

    episodes_array = np.asarray(episodes, dtype=np.int32)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    successes_array = np.asarray(successes, dtype=np.float32)
    collisions_array = np.asarray(collisions, dtype=np.float32)
    episode_lengths_array = np.asarray(episode_lengths, dtype=np.float32)

    successful_episode_lengths = np.where(successes_array > 0.0, episode_lengths_array, np.nan)
    return {
        "episodes": episodes_array,
        "success_rate": successes_array * 100.0,
        "collision_rate": collisions_array * 100.0,
        "reward": rewards_array,
        "avg_travel_time": successful_episode_lengths,
    }


def _trailing_moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """Compute a trailing moving average while ignoring NaNs."""

    if window_size <= 1:
        return values.astype(np.float32, copy=True)

    float_values = values.astype(np.float32, copy=False)
    valid_mask = ~np.isnan(float_values)
    safe_values = np.where(valid_mask, float_values, 0.0)

    cumulative_sums = np.concatenate(([0.0], np.cumsum(safe_values, dtype=np.float64)))
    cumulative_counts = np.concatenate(([0], np.cumsum(valid_mask.astype(np.int32))))

    moving_average = np.full(float_values.shape, np.nan, dtype=np.float32)
    for end_index in range(float_values.shape[0]):
        start_index = max(0, end_index - window_size + 1)
        total = cumulative_sums[end_index + 1] - cumulative_sums[start_index]
        count = cumulative_counts[end_index + 1] - cumulative_counts[start_index]
        if count > 0:
            moving_average[end_index] = float(total / count)

    return moving_average


def _aggregate_runs(run_series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad run series with NaNs and compute mean/std episode-wise."""

    max_length = max(series.shape[0] for series in run_series)
    padded_runs = np.full((len(run_series), max_length), np.nan, dtype=np.float32)
    for run_index, series in enumerate(run_series):
        padded_runs[run_index, : series.shape[0]] = series

    valid_mask = ~np.isnan(padded_runs)
    valid_counts = valid_mask.sum(axis=0)

    mean_curve = np.full(max_length, np.nan, dtype=np.float32)
    valid_columns = valid_counts > 0
    if np.any(valid_columns):
        mean_curve[valid_columns] = (
            np.nansum(padded_runs[:, valid_columns], axis=0) / valid_counts[valid_columns]
        ).astype(np.float32)

    std_curve = np.full(max_length, np.nan, dtype=np.float32)
    if np.any(valid_columns):
        centered_runs = np.where(valid_mask, padded_runs - mean_curve, 0.0)
        std_curve[valid_columns] = np.sqrt(
            np.sum(centered_runs[:, valid_columns] ** 2, axis=0) / valid_counts[valid_columns]
        ).astype(np.float32)

    return mean_curve, std_curve


def plot_multi_seed_algorithm_comparison(
    progress_csvs_by_algorithm: dict[str, list[str]],
    output_path: str,
    moving_avg_window: int = 50,
    show_plot: bool = False,
    algorithm_labels: dict[str, str] | None = None,
    algorithm_colors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Plot a 2x2 algorithm comparison with mean curves and variance bands."""

    if not progress_csvs_by_algorithm:
        raise ValueError("At least one algorithm with one CSV run is required.")

    algorithm_labels = {**DEFAULT_ALGORITHM_LABELS, **(algorithm_labels or {})}
    algorithm_colors = {**DEFAULT_ALGORITHM_COLORS, **(algorithm_colors or {})}

    metric_specs = [
        ("success_rate", "Success Rate (%) - Moving Avg", "Success Rate (%)"),
        ("collision_rate", "Collision Rate (%) - Moving Avg", "Collision Rate (%)"),
        ("reward", "Reward per Episode - Moving Avg", "Average Reward"),
        ("avg_travel_time", "Avg Travel Time (Steps) - Moving Avg", "Steps in Successful Episodes"),
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.style.use("ggplot")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_by_metric = {
        metric_key: axis
        for metric_key, axis in zip(
            [metric_key for metric_key, _, _ in metric_specs],
            axes.flatten(),
        )
    }

    plot_summary: dict[str, Any] = {"output_path": output_path, "moving_avg_window": moving_avg_window, "algorithms": {}}

    for algorithm_name, csv_paths in progress_csvs_by_algorithm.items():
        if not csv_paths:
            continue

        loaded_runs = [_read_progress_csv(csv_path) for csv_path in csv_paths]
        display_name = algorithm_labels.get(algorithm_name, algorithm_name)
        line_color = algorithm_colors.get(algorithm_name, None)

        plot_summary["algorithms"][algorithm_name] = {
            "label": display_name,
            "runs": csv_paths,
        }

        for metric_key, _, _ in metric_specs:
            smoothed_runs = [
                _trailing_moving_average(run_metrics[metric_key], moving_avg_window)
                for run_metrics in loaded_runs
            ]
            mean_curve, std_curve = _aggregate_runs(smoothed_runs)
            episode_axis = np.arange(1, mean_curve.shape[0] + 1)

            if metric_key in {"success_rate", "collision_rate"}:
                lower_band = np.clip(mean_curve - std_curve, 0.0, 100.0)
                upper_band = np.clip(mean_curve + std_curve, 0.0, 100.0)
            else:
                lower_band = mean_curve - std_curve
                upper_band = mean_curve + std_curve

            axis = axes_by_metric[metric_key]
            axis.plot(episode_axis, mean_curve, linewidth=2, color=line_color, label=display_name)
            if len(smoothed_runs) > 1:
                axis.fill_between(episode_axis, lower_band, upper_band, color=line_color, alpha=0.18)

    for metric_key, title, ylabel in metric_specs:
        axis = axes_by_metric[metric_key]
        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xlabel("Episode", fontsize=12)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.grid(True, alpha=0.3)
        if metric_key in {"success_rate", "collision_rate"}:
            axis.set_ylim(0.0, 100.0)
        axis.legend(fontsize=10)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    return plot_summary
