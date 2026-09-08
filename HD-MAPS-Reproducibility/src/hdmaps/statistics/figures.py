from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_simultaneous(rows: list[dict], out_path: Path) -> Path:
    xs = [r["fleet"] for r in rows]
    arrivals = [r["arrival_pct"] for r in rows]
    sem = [r["arrival_sem"] for r in rows]
    crashes = [r["crash_pct"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].errorbar(xs, arrivals, yerr=sem, marker="o", color="#1b4f72")
    axes[0].set_xlabel("Fleet size")
    axes[0].set_ylabel("Arrival rate (%)")
    axes[0].set_title("Simultaneous arrival")
    axes[0].set_ylim(0, 105)
    axes[1].plot(xs, crashes, marker="o", color="#922b21")
    axes[1].set_xlabel("Fleet size")
    axes[1].set_ylabel("Crash rate (%)")
    axes[1].set_title("Simultaneous crash")
    axes[1].set_ylim(0, 25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
