from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_proto_action_sweep import BASE_CFG, SEED, run_config, write_json
from run_proto_distance_sweep import collect_distance_summary, plot_distance_summary


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
        "distance_reward_weight": 0.0,
    }
    base.update(overrides)
    return base


# Anchored to the only configs from proto_sweep_26_04_2026-23_23_43 that kept
# training arrival around 85-91%: C06_embedding_dim_2 and C08_group_reward.
HIGH_ARRIVAL_CONFIGS = [
    cfg("H00_c06_emb2_reference"),
    cfg("H01_c06_emb2_low_std", master_log_std_init=-2.0),
    cfg("H02_c06_emb2_eval_mean", eval_policy_mean=True),
    cfg("H03_c06_emb2_train_eval_mean", train_policy_mean=True, eval_policy_mean=True),
    cfg("H04_c06_emb2_dist_002_t8", distance_reward_weight=0.02, target_min_pairwise_dist=8.0),
    cfg("H05_c06_emb2_dist_003_t10", distance_reward_weight=0.03, target_min_pairwise_dist=10.0),
    cfg("H06_c08_emb2_group_reference", reward_mode="group"),
    cfg("H07_c08_emb2_group_low_std", reward_mode="group", master_log_std_init=-2.0),
    cfg("H08_c08_emb2_group_dist_002_t10", reward_mode="group", distance_reward_weight=0.02, target_min_pairwise_dist=10.0),
    cfg(
        "H09_c08_emb2_group_conflict_dist",
        reward_mode="group",
        distance_reward_weight=0.02,
        target_min_pairwise_dist=10.0,
        conflict_schedule=[(0, 0.0), (1000, 0.15), (1800, 0.30)],
    ),
]


def safe_float(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank_for_paper(rows: list[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        train_last50 = safe_float(row.get("train_arrival_last50"))
        test_arrival = safe_float(row.get("test_arrival_normal"))
        test_crash = safe_float(row.get("test_crash_rate_normal"))
        dist_drop = safe_float(row.get("min_distance_drop_worst_ablation"))
        close_inc = safe_float(row.get("close_rate_increase_worst_ablation"))
        gm_auc = safe_float(row.get("gm_danger_auc"))
        lm_auc = safe_float(row.get("lm_best_danger_auc"))
        arrival_pass = train_last50 >= 85.0
        # Arrival is a gate. Distance/counterfactual effect is the claim we want.
        score = (
            (2.0 if arrival_pass else -2.0)
            + 0.02 * test_arrival
            - 0.01 * test_crash
            + 0.8 * max(0.0, dist_drop)
            + 8.0 * max(0.0, close_inc)
            + 2.0 * max(0.0, lm_auc - 0.5)
            + 1.0 * max(0.0, gm_auc - 0.5)
        )
        ranked.append({
            **row,
            "arrival_pass_85": int(arrival_pass),
            "paper_distance_score": score,
        })
    ranked.sort(key=lambda r: safe_float(r["paper_distance_score"]), reverse=True)
    return ranked


def plot_paper_summary(root: str, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots = [
        ("train_arrival_last50", "Train Arrival Last 50"),
        ("test_arrival_normal", "Normal Test Arrival"),
        ("min_distance_drop_worst_ablation", "Ablation Makes Min Distance Smaller"),
        ("close_rate_increase_worst_ablation", "Ablation Increases Close-Step Rate"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.bar(range(len(rows)), [safe_float(r.get(key)) for r in rows])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "high_arrival_distance_summary.png"), dpi=180)
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
            args.only_config = "H00_c06_emb2_reference"

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"proto_high_arrival_distance_{ts}")
    os.makedirs(root, exist_ok=True)

    configs = [dict(c) for c in HIGH_ARRIVAL_CONFIGS]
    if args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown config: {args.only_config}")

    write_json(os.path.join(root, "high_arrival_distance_config.json"), {"seed": SEED, "configs": configs})
    results = []
    for c in configs:
        results.append(run_config(c, root, episodes_override=args.episodes, test_override=args.test_episodes))
        write_json(os.path.join(root, "summary_partial.json"), results)

    write_json(os.path.join(root, "summary.json"), results)
    distance_rows = collect_distance_summary(root, configs)
    write_csv(os.path.join(root, "distance_config_ranking.csv"), distance_rows)
    plot_distance_summary(root, distance_rows)
    ranked = rank_for_paper(distance_rows)
    write_csv(os.path.join(root, "paper_candidate_ranking.csv"), ranked)
    plot_paper_summary(root, ranked)
    print(f"High-arrival distance sweep complete: {root}")


if __name__ == "__main__":
    main()
