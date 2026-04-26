import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================
CSV_DIR = Path(
    r"C:\Users\glebb\Downloads\RL-highway-feature-exp6_2_agents_100_scenarios\F21\RL-highway-feature-exp6_2_agents_100_scenarios\new_arch\embedding_experiment_20260205_213851\embedding_experiment_20260205_213851\csv_data"
)

DIMS = [2, 4, 8, 16]
WINDOWS = [50, 100, 200, 300]
ROLLING_WINDOW = 50  # smoothing for left plot

# Save to short local folder near this script (prevents long-path issues on Windows)
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR / "plots_success_windows"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    2: "#e41a1c",   # red
    4: "#377eb8",   # blue
    8: "#4daf4a",   # green
    16: "#984ea3",  # purple
}
MARKERS = {
    2: "o",
    4: "s",
    8: "^",
    16: "D",
}


def load_episode_df(csv_dir: Path, dim: int) -> pd.DataFrame:
    fp = csv_dir / f"episodes_dim{dim}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    df = pd.read_csv(fp)

    if "episode" not in df.columns:
        df["episode"] = np.arange(len(df))

    if "crashed" in df.columns:
        df["success"] = 1 - df["crashed"].astype(float)
    elif "success" in df.columns:
        df["success"] = df["success"].astype(float)
    else:
        raise ValueError(f"{fp.name} missing 'crashed' or 'success' column.")

    df = df.sort_values("episode").reset_index(drop=True)
    return df


def compute_final_success(df: pd.DataFrame, window: int) -> tuple[float, float]:
    tail = df.tail(window)["success"].astype(float)
    return float(tail.mean()), float(tail.std(ddof=0))


def safe_savefig(fig: plt.Figure, out_path: Path, dpi: int = 220):
    """
    Try saving to out_path. If fails (missing dir / long path), fallback to script dir.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)
        return out_path
    except Exception as e:
        fallback_dir = Path(__file__).resolve().parent / "plots_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / out_path.name
        fig.savefig(str(fallback_path), dpi=dpi)
        print(f"[WARN] Failed saving to:\n{out_path}\nReason: {e}\nSaved to fallback:\n{fallback_path}")
        return fallback_path


def make_plot_for_window(all_data: dict[int, pd.DataFrame], window: int, save_dir: Path) -> Path:
    fig = plt.figure(figsize=(16, 6))

    # Left panel: success over training (rolling)
    ax1 = plt.subplot(1, 2, 1)
    for dim in DIMS:
        df = all_data[dim]
        rolling = df["success"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        ax1.plot(
            df["episode"],
            rolling,
            color=COLORS[dim],
            linewidth=2.5,
            marker=MARKERS[dim],
            markevery=max(len(df) // 12, 1),
            alpha=0.95,
            label=f"dim={dim}",
        )

    ax1.set_title("Success Rate Over Training", fontsize=18, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Success Rate (No Crash)", fontsize=16, fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=13, framealpha=0.9)
    ax1.tick_params(labelsize=12)

    # Right panel: final success in selected window
    ax2 = plt.subplot(1, 2, 2)
    means, stds = [], []
    for dim in DIMS:
        m, s = compute_final_success(all_data[dim], window)
        means.append(m)
        stds.append(s)

    x = np.arange(len(DIMS))
    bars = ax2.bar(
        x,
        means,
        yerr=stds,
        capsize=8,
        color=[COLORS[d] for d in DIMS],
        alpha=0.8,
        edgecolor="none",
    )

    ax2.set_title(f"Success Rate [%]", fontsize=18, fontweight="bold")
    ax2.set_xlabel("Embedding Dimension", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Final Success Rate", fontsize=16, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in DIMS], fontsize=14)
    ax2.set_ylim(0.0, 1.1)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.tick_params(labelsize=12)

    for bar, m in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = save_dir / f"success_rate_last_{window}.png"
    saved_path = safe_savefig(fig, out_path, dpi=220)
    plt.close(fig)
    return saved_path


def main():
    print("Loading csv files...")
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV_DIR does not exist:\n{CSV_DIR}")

    all_data = {dim: load_episode_df(CSV_DIR, dim) for dim in DIMS}

    print("Generating plots...")
    generated = []
    for w in WINDOWS:
        out = make_plot_for_window(all_data, w, SAVE_DIR)
        generated.append(out)
        print(f"Saved: {out}")

    rows = []
    for w in WINDOWS:
        for dim in DIMS:
            mean_s, std_s = compute_final_success(all_data[dim], w)
            rows.append(
                {
                    "window": w,
                    "embedding_dim": dim,
                    "success_mean": mean_s,
                    "success_std": std_s,
                }
            )
    summary_df = pd.DataFrame(rows)

    summary_path = SAVE_DIR / "success_summary_windows.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
