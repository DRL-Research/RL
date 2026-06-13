import csv
import os
import random
from typing import Any

import numpy as np
import torch


PROGRESS_CSV_HEADER = [
    "episode",
    "reward",
    "actor_loss",
    "critic_loss",
    "noise_scale",
    "success",
    "collision",
    "episode_length",
]


def set_global_seeds(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def build_episode_seed(base_seed: int | None, episode_index: int) -> int | None:
    """Derive a deterministic per-episode seed from a run seed."""

    if base_seed is None:
        return None
    return int(base_seed) + max(episode_index - 1, 0)


def write_progress_csv(
    experiment_path: str,
    progress_rows: list[dict[str, Any]],
    log_dir_name: str,
) -> str:
    """Write comparable per-episode metrics for later aggregation and plotting."""

    log_dir = os.path.join(experiment_path, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "progress.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PROGRESS_CSV_HEADER)
        writer.writeheader()
        for progress_row in progress_rows:
            writer.writerow({header: progress_row.get(header, "") for header in PROGRESS_CSV_HEADER})

    return csv_path
