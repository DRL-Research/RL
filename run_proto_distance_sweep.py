from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_proto_action_sweep import BASE_CFG, SEED, run_config, write_json


def cfg(label: str, **overrides):
    base = {
        **BASE_CFG,
        "label": label,
        "embedding_dim": 2,
        "load_pretrained": False,
        "test_conflict_ratio": 0.5,
        "target_min_pairwise_dist": 8.0,
        "close_distance_threshold": 8.0,
        "critical_distance_threshold": 4.0,
    }
    base.update(overrides)
    return base


DISTANCE_CONFIGS = [
    cfg("D00_emb2_reference"),
    cfg("D01_emb2_low_std", master_log_std_init=-2.0),
    cfg("D02_emb2_group_reward", reward_mode="group"),
    cfg("D03_emb2_mean_proto", train_policy_mean=True, eval_policy_mean=True),
    cfg("D04_emb2_master_norm", normalize_master_inputs=True),
    cfg("D05_emb2_agent_norm", normalize_master_inputs=True, normalize_agent_obs=True),
    cfg("D06_emb2_dist_reward_002", distance_reward_weight=0.02),
    cfg("D07_emb2_dist_reward_005", distance_reward_weight=0.05),
    cfg("D08_emb2_dist_reward_group", reward_mode="group", distance_reward_weight=0.03),
    cfg(
        "D09_emb2_conflict_dist",
        conflict_schedule=[(0, 0.0), (800, 0.2), (1600, 0.5)],
        distance_reward_weight=0.03,
    ),
]


def safe_float(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_distance_summary(root: str, configs: list[dict]) -> list[dict]:
    rows = []
    for c in configs:
        label = c["label"]
        cfg_dir = os.path.join(root, label)
        episode_path = os.path.join(cfg_dir, "episode_metrics.csv")
        cf_path = os.path.join(cfg_dir, "counterfactual_summary.csv")
        phase_path = os.path.join(cfg_dir, "phase_metrics.csv")
        if not os.path.exists(episode_path) or not os.path.exists(cf_path):
            continue

        episodes = read_csv(episode_path)
        cf = read_csv(cf_path)
        phase = read_csv(phase_path) if os.path.exists(phase_path) else []
        cf_by = {r["condition"]: r for r in cf}
        normal = cf_by.get("normal", {})

        normal_min = safe_float(normal.get("min_pairwise_dist_mean"))
        normal_close = safe_float(normal.get("close_step_rate_mean"))
        normal_critical = safe_float(normal.get("critical_step_rate_mean"))

        ablation_mins = [
            safe_float(cf_by.get(cond, {}).get("min_pairwise_dist_mean"), normal_min)
            for cond in ("zero_master", "swap_local_masters", "negate_master")
        ]
        ablation_close = [
            safe_float(cf_by.get(cond, {}).get("close_step_rate_mean"), normal_close)
            for cond in ("zero_master", "swap_local_masters", "negate_master")
        ]
        gm = next((r for r in phase if r.get("role") == "GM"), {})
        lm1 = next((r for r in phase if r.get("role") == "LM1"), {})
        lm2 = next((r for r in phase if r.get("role") == "LM2"), {})

        train_arr = [float(r["arrival_pct"]) for r in episodes]
        train_cr = [int(r["crashed"]) for r in episodes]
        rows.append({
            "label": label,
            "train_arrival_last50": float(np.mean(train_arr[-50:])) if train_arr else 0.0,
            "train_arrival_last100": float(np.mean(train_arr[-100:])) if train_arr else 0.0,
            "train_crash_last50": int(sum(train_cr[-50:])),
            "test_arrival_normal": safe_float(normal.get("arrival_mean")),
            "test_crash_rate_normal": safe_float(normal.get("crash_rate")),
            "normal_min_pairwise_dist_mean": normal_min,
            "normal_p10_pairwise_dist_mean": safe_float(normal.get("p10_pairwise_dist_mean")),
            "normal_close_step_rate": normal_close,
            "normal_critical_step_rate": normal_critical,
            "min_distance_drop_worst_ablation": normal_min - min(ablation_mins) if ablation_mins else 0.0,
            "close_rate_increase_worst_ablation": max(ablation_close) - normal_close if ablation_close else 0.0,
            "gm_safe_danger_kl": safe_float(gm.get("safe_vs_danger_symmetric_kl")),
            "gm_danger_auc": safe_float(gm.get("danger_auc_linear_centroid")),
            "lm_best_danger_auc": max(
                safe_float(lm1.get("danger_auc_linear_centroid")),
                safe_float(lm2.get("danger_auc_linear_centroid")),
            ),
        })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_distance_summary(root: str, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots = [
        ("train_arrival_last50", "Train Arrival Last 50"),
        ("normal_min_pairwise_dist_mean", "Normal Mean Min Pairwise Distance"),
        ("min_distance_drop_worst_ablation", "Distance Drop Under Worst Ablation"),
        ("close_rate_increase_worst_ablation", "Close-Step Rate Increase Under Ablation"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.bar(range(len(rows)), [float(r[key]) for r in rows])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "distance_sweep_summary.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(rows)), [float(r["gm_danger_auc"]) for r in rows])
    axes[0].set_title("GM Danger AUROC")
    axes[1].bar(range(len(rows)), [float(r["lm_best_danger_auc"]) for r in rows])
    axes[1].set_title("Best LM Danger AUROC")
    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "distance_master_signal_summary.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--only-config", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 3
        args.test_episodes = 2
        if not args.only_config:
            args.only_config = "D00_emb2_reference"

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"proto_distance_sweep_{ts}")
    os.makedirs(root, exist_ok=True)

    configs = [dict(c) for c in DISTANCE_CONFIGS]
    if args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown config: {args.only_config}")

    write_json(os.path.join(root, "distance_sweep_config.json"), {"seed": SEED, "configs": configs})
    results = []
    for c in configs:
        results.append(run_config(c, root, episodes_override=args.episodes, test_override=args.test_episodes))
        write_json(os.path.join(root, "summary_partial.json"), results)

    write_json(os.path.join(root, "summary.json"), results)
    distance_rows = collect_distance_summary(root, configs)
    write_csv(os.path.join(root, "distance_config_ranking.csv"), distance_rows)
    plot_distance_summary(root, distance_rows)
    print(f"Proto-distance sweep complete: {root}")


if __name__ == "__main__":
    main()
