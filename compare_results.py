"""
Comparison and Visualization Script (paper-aligned)
- X-axis = environment steps, including demonstration transitions
- Single-run curves + optional multi-seed 95% CI bands
- Three panels: success rate, episodic return, travel time (successful eps)
"""

import sys
import os
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


# ---------- I/O helpers ---------- #

def load_metrics(path: str):
    with open(path, "r") as f:
        return json.load(f)

def load_many(prefix: str, pattern: str = "metrics.json"):
    paths = sorted(glob.glob(f"{prefix}_seed*/{pattern}"))
    runs = []
    for p in paths:
        try:
            runs.append(load_metrics(p))
        except Exception as e:
            print(f"[WARN] Failed to load '{p}': {e}")
    return runs


# ---------- math helpers ---------- #

def smooth_curve(data, window=50):
    data = np.asarray(data, dtype=float)
    if data.size == 0 or data.size < window:
        return data
    out = []
    half = window // 2
    for i in range(len(data)):
        s = max(0, i - half)
        e = min(len(data), i + half)
        out.append(data[s:e].mean())
    return np.asarray(out, dtype=float)

def get_test_steps(metrics, fallback_interval=20000):
    if metrics is None:
        return np.array([], dtype=float)
    steps = metrics.get("test_total_steps", [])
    if steps:
        return np.asarray(steps, dtype=float)
    n = len(metrics.get("test_success_rates", []))
    return np.arange(1, n + 1, dtype=float) * float(fallback_interval)

def interpolate_on_steps(runs):
    all_steps = sorted(set(s for r in runs for s in r.get("test_total_steps", [])))
    if not all_steps:
        lens = [len(r.get("test_success_rates", [])) for r in runs]
        if not lens or min(lens) == 0:
            return np.array([]), {}
        S = max(lens)
        all_steps = list((np.arange(1, S + 1, dtype=float) * 20000))

    def interp_one(run, key):
        xs = np.asarray(run.get("test_total_steps", []), dtype=float)
        ys = np.asarray(run.get(key, []), dtype=float)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(len(all_steps), dtype=float)
        uniq_x, idx = np.unique(xs, return_index=True)
        uniq_y = ys[idx]
        return np.interp(all_steps, uniq_x, uniq_y)

    packed = {}
    for key in ["test_success_rates", "test_rewards", "test_travel_times"]:
        series = [interp_one(r, key) for r in runs]
        packed[key] = np.stack(series, axis=0) if series else np.zeros((0, len(all_steps)))
    return np.asarray(all_steps, dtype=float), packed

def mean_and_ci(arr, alpha=0.95):
    if arr.size == 0:
        return np.array([]), np.array([])
    N = arr.shape[0]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if N > 1 else np.zeros_like(mean)
    se = std / max(1, np.sqrt(N))
    try:
        from scipy.stats import t
        tcrit = t.ppf((1 + alpha) / 2.0, df=max(1, N - 1))
    except Exception:
        t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
        tcrit = t_table.get(N - 1, 1.96)
    return mean, tcrit * se


# ---------- plotting ---------- #

def plot_single(axs, qmix_metrics, qmixwd_metrics, master_metrics=None):
    ax1, ax2, ax3 = axs
    colors = {"qmix": "#1f77b4", "qmixwd": "#ff7f0e", "master": "#2ca02c"}

    # Success rate
    if qmix_metrics and qmix_metrics.get("test_success_rates", []):
        xs = get_test_steps(qmix_metrics) / 1e6
        ys = np.asarray(qmix_metrics["test_success_rates"], dtype=float)
        ax1.plot(xs, ys, color=colors["qmix"], label="QMIX", linewidth=2, marker='o', markersize=3)
    if qmixwd_metrics and qmixwd_metrics.get("test_success_rates", []):
        xs = get_test_steps(qmixwd_metrics) / 1e6
        ys = np.asarray(qmixwd_metrics["test_success_rates"], dtype=float)
        ax1.plot(xs, ys, color=colors["qmixwd"], label="QMIXwD", linewidth=2, marker='o', markersize=3)
    if master_metrics and master_metrics.get("test_success_rates", []):
        xs = get_test_steps(master_metrics) / 1e6
        ys = np.asarray(master_metrics["test_success_rates"], dtype=float)
        ax1.plot(xs, ys, color=colors["master"], label="Master+Agent", linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Test Success Rate")
    ax1.set_title("Test Success Rate")
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Episodic return
    if qmix_metrics and qmix_metrics.get("test_rewards", []):
        xs = get_test_steps(qmix_metrics) / 1e6
        ys = smooth_curve(qmix_metrics["test_rewards"])
        ax2.plot(xs, ys, color=colors["qmix"], label="QMIX", linewidth=2, marker='o', markersize=3)
    if qmixwd_metrics and qmixwd_metrics.get("test_rewards", []):
        xs = get_test_steps(qmixwd_metrics) / 1e6
        ys = smooth_curve(qmixwd_metrics["test_rewards"])
        ax2.plot(xs, ys, color=colors["qmixwd"], label="QMIXwD", linewidth=2, marker='o', markersize=3)
    if master_metrics and master_metrics.get("test_rewards", []):
        xs = get_test_steps(master_metrics) / 1e6
        ys = smooth_curve(master_metrics["test_rewards"])
        ax2.plot(xs, ys, color=colors["master"], label="Master+Agent", linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Test Episodic Return")
    ax2.set_title("Test Episodic Return")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Travel time
    if qmix_metrics and qmix_metrics.get("test_travel_times", []):
        xs = get_test_steps(qmix_metrics) / 1e6
        ys = smooth_curve(qmix_metrics["test_travel_times"])
        ax3.plot(xs, ys, color=colors["qmix"], label="QMIX", linewidth=2, marker='o', markersize=3)
    if qmixwd_metrics and qmixwd_metrics.get("test_travel_times", []):
        xs = get_test_steps(qmixwd_metrics) / 1e6
        ys = smooth_curve(qmixwd_metrics["test_travel_times"])
        ax3.plot(xs, ys, color=colors["qmixwd"], label="QMIXwD", linewidth=2, marker='o', markersize=3)
    if master_metrics and master_metrics.get("test_travel_times", []):
        xs = get_test_steps(master_metrics) / 1e6
        ys = smooth_curve(master_metrics["test_travel_times"])
        ax3.plot(xs, ys, color=colors["master"], label="Master+Agent", linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel("Steps (×10⁶)")
    ax3.set_ylabel("Test Travel Time (steps)")
    ax3.set_title("Test Travel Time (Successful Episodes)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)


def plot_aggregate_band(axs, prefix, label, color):
    runs = load_many(prefix)
    if len(runs) < 2:
        print(f"[INFO] No multi-seed runs found for '{prefix}_seed*' — skipping CI band.")
        return
    steps, packed = interpolate_on_steps(runs)
    if steps.size == 0:
        print(f"[WARN] Could not align steps for '{prefix}'.")
        return
    xs = steps / 1e6
    ax1, ax2, ax3 = axs

    m, h = mean_and_ci(packed["test_success_rates"])
    ax1.plot(xs, m, color=color, label=f"{label} (mean over seeds)", linewidth=2.5)
    ax1.fill_between(xs, m - h, m + h, color=color, alpha=0.2)

    m, h = mean_and_ci(packed["test_rewards"])
    ax2.plot(xs, m, color=color, label=f"{label} (mean over seeds)", linewidth=2.5)
    ax2.fill_between(xs, m - h, m + h, color=color, alpha=0.2)

    m, h = mean_and_ci(packed["test_travel_times"])
    ax3.plot(xs, m, color=color, label=f"{label} (mean over seeds)", linewidth=2.5)
    ax3.fill_between(xs, m - h, m + h, color=color, alpha=0.2)


# ---------- table printing ---------- #

def extract_final(metrics):
    if not metrics or not metrics.get("test_success_rates", []):
        return None
    n_last = min(5, len(metrics["test_success_rates"]))
    final_success = float(np.mean(metrics["test_success_rates"][-n_last:]))
    final_reward = float(np.mean(metrics.get("test_rewards", [0.0])[-n_last:]))
    last_travel = [t for t in metrics.get("test_travel_times", [])[-n_last:] if t > 0]
    final_travel = float(np.mean(last_travel)) if last_travel else 0.0
    total_episodes = len(metrics.get("episode_rewards", []))
    total_collisions = int(sum(metrics.get("collisions", [])))
    collision_rate = (total_collisions / total_episodes) if total_episodes > 0 else 0.0
    return {
        "success_rate": final_success,
        "reward": final_reward,
        "travel_time": final_travel,
        "collision_rate": collision_rate,
        "total_steps": int(metrics.get("total_steps", 0)),
        "demo_steps": int(metrics.get("demo_steps", 0)),
    }

def print_table(rows, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Method':<20} {'Success Rate':<15} {'Avg Reward':<15} "
          f"{'Avg Travel Time':<18} {'Collision Rate':<15} {'Steps':<10} {'Demo Steps':<10}")
    print("-" * 80)
    for name, d in rows:
        if d is None:
            continue
        print(f"{name:<20} {d['success_rate']:<15.2%} {d['reward']:<15.2f} "
              f"{d['travel_time']:<18.1f} {d['collision_rate']:<15.2%} "
              f"{d['total_steps']:<10} {d['demo_steps']:<10}")
    print("=" * 80)


# ---------- main ---------- #

def main():
    print("Loading metrics...")
    qmix_metrics = load_metrics("qmix_baseline_results/metrics.json") if os.path.exists("qmix_baseline_results/metrics.json") else None
    if qmix_metrics: print("Loaded QMIX metrics")
    qmixwd_metrics = load_metrics("qmixwd_results/metrics.json") if os.path.exists("qmixwd_results/metrics.json") else None
    if qmixwd_metrics: print("Loaded QMIXwD metrics")
    master_metrics = load_metrics("master_agent_results/metrics.json") if os.path.exists("master_agent_results/metrics.json") else None
    if master_metrics: print("Loaded Master+Agent metrics")

    os.makedirs("comparison_plots", exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    plot_single(axs, qmix_metrics, qmixwd_metrics, master_metrics)
    plot_aggregate_band(axs, "qmix_baseline_results", "QMIX", "#1f77b4")
    plot_aggregate_band(axs, "qmixwd_results", "QMIXwD", "#ff7f0e")

    plt.tight_layout()
    out = "comparison_plots/comparison_all_metrics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {out}")
    # Optional: show the figure interactively
    # plt.show()
    plt.close(fig)

    single_rows = [
        ("QMIX", extract_final(qmix_metrics)),
        ("QMIXwD", extract_final(qmixwd_metrics)),
        ("Master+Agent", extract_final(master_metrics)),
    ]
    print_table(single_rows, "FINAL PERFORMANCE COMPARISON (single runs)")

    print("\nAll visualizations complete!")
    print("Results saved to comparison_plots/")


if __name__ == "__main__":
    main()
