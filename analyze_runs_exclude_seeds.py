"""
Re-aggregate training metrics from disk while excluding selected seeds.

Reads only ``episode_metrics.csv`` under::

    <run_dir>/<config>/W_MASTER/s<seed>/episode_metrics.csv

Writes:
  - ``arrival_mean_std_excluded.png`` — mean ± std of smoothed arrival curves
    (same rolling window as ``run_unified.SMOOTH_EP``).
  - ``summary_training_excluded.json`` — filtered ``per_seed`` rows plus
    recomputed mean/std for training fields.

Limitation: ``summary.json`` fields ``held_out_mean`` / ``conflict_test_mean``
are aggregated across all seeds at write time and are not stored per seed.
This script cannot reproduce those test bar plots without re-running tests
or having per-seed test dumps.

Example::

    py -3 analyze_runs_exclude_seeds.py experiment_runs/full_26_04_2026-11_40_39 --exclude 7
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match run_unified.py training arrival smoothing
SMOOTH_EP = 50

_SEED_DIR = re.compile(r"^s(\d+)$", re.IGNORECASE)


def _rolling_mean(lst: list, w: int) -> list:
    out = []
    for i in range(len(lst)):
        window = [v for v in lst[max(0, i - w + 1) : i + 1] if v is not None]
        out.append(float(np.mean(window)) if window else 0.0)
    return out


def _smooth_arr(arr: list, window: int) -> np.ndarray:
    return np.array(_rolling_mean(arr, window), dtype=np.float64)


def _compute_seed_bands(seed_data_list: list, key: str, window: int):
    curves = []
    for d in seed_data_list:
        raw = d["results"].get(key, [])
        if raw:
            curves.append(_smooth_arr(raw, window))
    if not curves:
        return None, None, None
    min_len = min(len(c) for c in curves)
    stacked = np.array([c[:min_len] for c in curves])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std, min_len


def _read_episode_metrics(csv_path: Path) -> dict:
    arrivals: list[float | None] = []
    collisions = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                arrivals.append(float(row["arrival_pct"]))
            except (KeyError, TypeError, ValueError):
                arrivals.append(None)
            try:
                collisions += int(float(row.get("collision", 0)))
            except (TypeError, ValueError):
                pass
    clean = [v for v in arrivals if v is not None]
    last50 = clean[-50:] if clean else []
    seed = None
    m = _SEED_DIR.match(csv_path.parent.name)
    if m:
        seed = int(m.group(1))
    return {
        "seed": seed,
        "results": {"arrival_rates": arrivals},
        "arrival_avg": round(float(np.mean(clean)), 2) if clean else 0.0,
        "arrival_last50": round(float(np.mean(last50)), 2) if last50 else 0.0,
        "total_collisions": int(collisions),
    }


def _discover_seed_dirs(run_dir: Path, config: str, condition: str) -> list[Path]:
    base = run_dir / config / condition
    if not base.is_dir():
        return []
    out = []
    for p in base.iterdir():
        if p.is_dir() and _SEED_DIR.match(p.name):
            out.append(p)
    return sorted(out, key=lambda x: int(_SEED_DIR.match(x.name).group(1)))


def _discover_configs(run_dir: Path, condition: str) -> list[str]:
    names = []
    for p in run_dir.iterdir():
        if p.is_dir() and (p / condition).is_dir():
            names.append(p.name)
    return sorted(names)


def _plot_arrival_bands(
    all_seeds_data: dict,
    out_path: Path,
    n_seeds: int,
    excluded: list[int],
) -> None:
    config_names = list(all_seeds_data.keys())
    n_configs = len(config_names)
    if n_configs == 0:
        return
    fig, axes = plt.subplots(1, n_configs, figsize=(7 * min(n_configs, 4), 5), squeeze=False)
    excl_str = ", ".join(str(x) for x in sorted(excluded)) if excluded else "none"
    for i, cfg_name in enumerate(config_names):
        ax = axes[0][i]
        for cond, color, ls in [("W_MASTER", "#2196F3", "-"), ("NO_MASTER", "#FF5722", "--")]:
            seed_list = all_seeds_data[cfg_name].get(cond, [])
            mean, std, n = _compute_seed_bands(seed_list, "arrival_rates", SMOOTH_EP)
            if mean is None:
                continue
            x = np.arange(n)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=2, label=cond)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(cfg_name, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Arrival %")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Training Arrival (mean ± std, {n_seeds} seeds; excluded: {excl_str})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=str, help="Experiment folder, e.g. experiment_runs/full_..._39")
    ap.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=[7],
        help="Seed values to drop (default: 7)",
    )
    ap.add_argument("--condition", type=str, default="W_MASTER", help="Subfolder name, default W_MASTER")
    ap.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=None,
        help="Config folder names under run_dir (default: all that contain <condition>/)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="summary_training_excluded.json",
        help="Output JSON filename inside run_dir",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default="arrival_mean_std_excluded.png",
        help="Output plot filename inside run_dir",
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    exclude = set(args.exclude)
    condition = args.condition

    configs = args.configs if args.configs else _discover_configs(run_dir, condition)
    if not configs:
        raise SystemExit(f"No configs found under {run_dir} with {condition}/")

    all_seeds_data: dict[str, dict[str, list]] = {}
    summary_block: dict = {
        "excluded_seeds": sorted(exclude),
        "condition": condition,
        "configs": configs,
        "note": (
            "Training metrics only, recomputed from episode_metrics.csv. "
            "held_out / conflict_test aggregates in summary.json are not per-seed."
        ),
    }

    for cfg in configs:
        all_seeds_data[cfg] = {condition: []}
        per_seed_rows = []
        included_seeds: list[int] = []

        for sdir in _discover_seed_dirs(run_dir, cfg, condition):
            m = _SEED_DIR.match(sdir.name)
            if not m:
                continue
            seed = int(m.group(1))
            if seed in exclude:
                continue
            csv_path = sdir / "episode_metrics.csv"
            if not csv_path.is_file():
                continue
            row = _read_episode_metrics(csv_path)
            all_seeds_data[cfg][condition].append(
                {"results": row["results"], "seed": row["seed"]}
            )
            included_seeds.append(seed)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "arrival_avg": row["arrival_avg"],
                    "arrival_last50": row["arrival_last50"],
                    "total_collisions": row["total_collisions"],
                }
            )

        if not per_seed_rows:
            summary_block[cfg] = {condition: {"error": "no seeds after filter"}}
            continue

        avgs = [r["arrival_avg"] for r in per_seed_rows]
        l50s = [r["arrival_last50"] for r in per_seed_rows]
        cols = [r["total_collisions"] for r in per_seed_rows]
        summary_block[cfg] = {
            condition: {
                "n_seeds": len(per_seed_rows),
                "included_seeds": sorted(included_seeds),
                "per_seed": per_seed_rows,
                "mean_arrival_avg": round(float(np.mean(avgs)), 2),
                "std_arrival_avg": round(float(np.std(avgs)), 2),
                "mean_arrival_l50": round(float(np.mean(l50s)), 2),
                "std_arrival_l50": round(float(np.std(l50s)), 2),
                "mean_collisions": round(float(np.mean(cols)), 2),
                "std_collisions": round(float(np.std(cols)), 2),
            }
        }

    n_plot = max(
        len(all_seeds_data[c].get(condition, [])) for c in all_seeds_data
    )
    _plot_arrival_bands(
        all_seeds_data,
        run_dir / args.out_png,
        n_plot,
        sorted(exclude),
    )

    out_json = run_dir / args.out_json
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary_block, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"Wrote {run_dir / args.out_png}")


if __name__ == "__main__":
    main()
