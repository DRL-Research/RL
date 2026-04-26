"""
Training curves: WITH_MASTER vs WITHOUT_MASTER on a common x-axis (0–100% of episodes).

**WITH (default):** all ``episode_metrics.csv`` under
``<with_run_dir>/<config>/<condition>/s*/`` (multi-seed), excluding listed seeds
— same sources as ``analyze_runs_exclude_seeds.py`` / ``arrival_mean_std_excluded.png``.
Rolling smooth uses a **fixed episode window** (default 50), matching ``run_unified.SMOOTH_EP``.

**WITHOUT (default):** mean ± std over the two legacy single-run CSVs
(unified_vs_ablation + unified_full A_baseline), same smoothing.

Outputs (under ``experiment_runs/`` by default):
  - master_training_with_vs_without.png
  - master_training_all_runs.png (with ``--all-runs``): each WITH seed + each WITHOUT run.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent

# Match run_unified.py / analyze_runs_exclude_seeds.py arrival smoothing
DEFAULT_SMOOTH_EP = 50

_SEED_DIR = re.compile(r"^s(\d+)$", re.IGNORECASE)


def _rolling_mean(arr: list | np.ndarray, window: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    w = max(1, int(window))
    out = np.empty_like(a)
    for i in range(len(a)):
        sl = a[max(0, i - w + 1) : i + 1]
        out[i] = float(np.mean(sl))
    return out


def _load_episode_metrics(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (arrival_pct, crash01, reward, n_episodes)."""
    arrivals: list[float] = []
    crashes: list[float] = []
    rewards: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                arrivals.append(float(row["arrival_pct"]))
            except (KeyError, ValueError):
                arrivals.append(0.0)
            c = row.get("crashed", row.get("collision", "0"))
            try:
                crashes.append(float(c))
            except ValueError:
                crashes.append(0.0)
            try:
                rewards.append(float(row["reward"]))
            except (KeyError, ValueError):
                rewards.append(0.0)
    n = len(arrivals)
    if n == 0:
        raise ValueError(f"No rows in {csv_path}")
    return (
        np.asarray(arrivals, dtype=np.float64),
        np.asarray(crashes, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        n,
    )


def _progress_pct(n: int) -> np.ndarray:
    if n == 1:
        return np.array([50.0])
    return np.linspace(0.0, 100.0, n)


def _interp_to_grid(x_pct: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x_pct, y)


def _smooth_training_curves(
    arrival: np.ndarray,
    crash01: np.ndarray,
    reward: np.ndarray,
    smooth_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = max(1, int(smooth_episodes))
    sr = _rolling_mean(arrival, w)
    cr = _rolling_mean(crash01 * 100.0, w)
    rw = _rolling_mean(reward, w)
    return sr, cr, rw


def _default_without_runs() -> list[tuple[str, Path]]:
    base = _REPO / "experiment_runs"
    return [
        (
            "unified_vs_ablation",
            base / "unified_vs_ablation_25_04_2026-21_47_36" / "WITHOUT_MASTER" / "episode_metrics.csv",
        ),
        (
            "unified_full_A_baseline",
            base / "unified_full_experiment_26_04_2026-08_09_00" / "A_baseline" / "WITHOUT_MASTER" / "episode_metrics.csv",
        ),
    ]


def _default_legacy_with_runs() -> list[tuple[str, Path]]:
    """Old behaviour (two single CSVs) — use only with ``--with-legacy``."""
    base = _REPO / "experiment_runs"
    return [
        (
            "unified_vs_ablation",
            base / "unified_vs_ablation_25_04_2026-21_47_36" / "WITH_MASTER" / "episode_metrics.csv",
        ),
        (
            "unified_full_A_baseline",
            base / "unified_full_experiment_26_04_2026-08_09_00" / "A_baseline" / "WITH_MASTER" / "episode_metrics.csv",
        ),
    ]


def discover_with_seed_csvs(
    run_dir: Path,
    config: str,
    condition: str,
    exclude_seeds: set[int],
) -> list[tuple[str, Path]]:
    """Paths like ``.../A_base/W_MASTER/s42/episode_metrics.csv``."""
    base = run_dir / config / condition
    if not base.is_dir():
        raise FileNotFoundError(f"Expected directory: {base}")
    rows: list[tuple[int, str, Path]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = _SEED_DIR.match(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        if seed in exclude_seeds:
            continue
        csv_p = p / "episode_metrics.csv"
        if csv_p.is_file():
            rows.append((seed, f"s{seed}", csv_p))
    rows.sort(key=lambda t: t[0])
    return [(lab, path) for _seed, lab, path in rows]


def _collect_series(
    runs: list[tuple[str, Path]],
    smooth_ep: int,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sr_s, cr_s, rw_s = [], [], []
    for _label, path in runs:
        arr, cr, rw, n = _load_episode_metrics(path)
        x = _progress_pct(n)
        sr, cr_pct, rw_m = _smooth_training_curves(arr, cr, rw, smooth_ep)
        sr_s.append(_interp_to_grid(x, sr, grid))
        cr_s.append(_interp_to_grid(x, cr_pct, grid))
        rw_s.append(_interp_to_grid(x, rw_m, grid))
    return np.stack(sr_s), np.stack(cr_s), np.stack(rw_s)


def _plot_with_vs_without(
    grid: np.ndarray,
    with_sr: np.ndarray,
    with_cr: np.ndarray,
    with_rw: np.ndarray,
    wo_sr: np.ndarray,
    wo_cr: np.ndarray,
    wo_rw: np.ndarray,
    out_path: Path,
    n_with_seeds: int,
) -> None:
    sr_m, sr_s = with_sr.mean(0), with_sr.std(0)
    cr_m, cr_s = with_cr.mean(0), with_cr.std(0)
    rw_m, rw_s = with_rw.mean(0), with_rw.std(0)
    wsr_m, wsr_s = wo_sr.mean(0), wo_sr.std(0)
    wcr_m, wcr_s = wo_cr.mean(0), wo_cr.std(0)
    wrw_m, wrw_s = wo_rw.mean(0), wo_rw.std(0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    def band(ax, x, m, s, color, ls="-"):
        ax.plot(x, m, color=color, lw=2.2, linestyle=ls)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)

    fig.suptitle(
        f"with and without master- {n_with_seeds} seeds",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    band(axes[0], grid, sr_m, sr_s, "#1565C0")
    band(axes[0], grid, wsr_m, wsr_s, "#C62828")
    axes[0].set_ylabel("Success (arrival %)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)

    band(axes[1], grid, cr_m, cr_s, "#1565C0", ls="-")
    band(axes[1], grid, wcr_m, wcr_s, "#C62828", ls="-")
    axes[1].set_ylabel("Crash rate (rolling %)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    band(axes[2], grid, rw_m, rw_s, "#1565C0", ls="-")
    band(axes[2], grid, wrw_m, wrw_s, "#C62828", ls="-")
    axes[2].set_ylabel("Reward (rolling mean)")
    axes[2].set_xlabel("Training progress (% of episodes)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_all_runs(
    grid: np.ndarray,
    with_stack: tuple[np.ndarray, np.ndarray, np.ndarray],
    wo_stack: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels_with: list[str],
    labels_wo: list[str],
    out_path: Path,
    smooth_ep: int,
) -> None:
    n_w = len(labels_with)
    colors_w = plt.cm.Blues(np.linspace(0.35, 0.95, max(n_w, 2)))[:n_w]
    colors_wo = plt.cm.Reds(np.linspace(0.35, 0.95, max(len(labels_wo), 2)))[: len(labels_wo)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    names = ["Success (arrival %)", "Crash rate (rolling %)", "Reward (rolling mean)"]
    for ax, idx, name in zip(axes, range(3), names):
        for i, lab in enumerate(labels_with):
            ax.plot(grid, with_stack[idx][i], color=colors_w[i], lw=1.8, label=f"WITH — {lab}")
        for i, lab in enumerate(labels_wo):
            ax.plot(
                grid,
                wo_stack[idx][i],
                color=colors_wo[i],
                lw=1.8,
                linestyle="--",
                label=f"WITHOUT — {lab}",
            )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_title(f"Per-seed / per-run curves (rolling window = {smooth_ep} episodes)", fontsize=11)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right", ncol=2)
        if "Success" in name:
            ax.set_ylim(0, 105)
        if "Crash" in name:
            ax.set_ylim(0, 105)
    axes[-1].set_xlabel("Training progress (% of episodes)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        type=str,
        default=str(_REPO / "experiment_runs" / "master_training_with_vs_without.png"),
        help="Output path for main WITH vs WITHOUT figure",
    )
    ap.add_argument(
        "--out-all-runs",
        type=str,
        default=str(_REPO / "experiment_runs" / "master_training_all_runs.png"),
        help="Output path for per-seed / per-run figure",
    )
    ap.add_argument(
        "--with-run-dir",
        type=str,
        default=str(_REPO / "experiment_runs" / "full_26_04_2026-11_40_39"),
        help="Multi-seed experiment folder (parent of A_base/...)",
    )
    ap.add_argument("--with-config", type=str, default="A_base", help="Config folder under with-run-dir")
    ap.add_argument("--with-condition", type=str, default="W_MASTER", help="Condition folder name")
    ap.add_argument(
        "--exclude-seeds",
        type=int,
        nargs="*",
        default=[7],
        help="Seeds to skip under s<seed>/ (default: 7)",
    )
    ap.add_argument(
        "--with-legacy",
        action="store_true",
        help="Use two legacy WITH_MASTER CSVs instead of multi-seed under --with-run-dir",
    )
    ap.add_argument(
        "--smooth-episodes",
        type=int,
        default=DEFAULT_SMOOTH_EP,
        help=f"Rolling-mean window in episodes (default {DEFAULT_SMOOTH_EP}, same as run_unified SMOOTH_EP)",
    )
    ap.add_argument("--grid-points", type=int, default=400, help="Interpolation resolution on 0–100%% axis")
    ap.add_argument("--all-runs", action="store_true", help="Also write per-seed WITH + per-run WITHOUT figure")
    args = ap.parse_args()

    exclude = set(args.exclude_seeds)
    if args.with_legacy:
        with_runs = _default_legacy_with_runs()
    else:
        with_runs = discover_with_seed_csvs(
            Path(args.with_run_dir).resolve(),
            args.with_config,
            args.with_condition,
            exclude,
        )
    if not with_runs:
        raise SystemExit("No WITH runs found (check paths and --exclude-seeds).")

    without_runs = _default_without_runs()
    for _lab, p in with_runs + without_runs:
        if not p.is_file():
            raise SystemExit(f"Missing CSV: {p}")

    grid = np.linspace(0.0, 100.0, args.grid_points)
    se = args.smooth_episodes

    with_sr, with_cr, with_rw = _collect_series(with_runs, se, grid)
    wo_sr, wo_cr, wo_rw = _collect_series(without_runs, se, grid)

    out_main = Path(args.out)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    _plot_with_vs_without(
        grid,
        with_sr,
        with_cr,
        with_rw,
        wo_sr,
        wo_cr,
        wo_rw,
        out_main,
        len(with_runs),
    )
    print(f"Wrote {out_main}")

    if args.all_runs:
        _plot_all_runs(
            grid,
            (with_sr, with_cr, with_rw),
            (wo_sr, wo_cr, wo_rw),
            [a for a, _ in with_runs],
            [a for a, _ in without_runs],
            Path(args.out_all_runs),
            se,
        )
        print(f"Wrote {args.out_all_runs}")


if __name__ == "__main__":
    main()
