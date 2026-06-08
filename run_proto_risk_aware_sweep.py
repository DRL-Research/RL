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
from run_proto_high_arrival_distance_sweep import rank_for_paper


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
        "risk_aux_weight": 0.0,
        "distance_aux_weight": 0.0,
        "risk_horizon_steps": 5,
        "master_dropout_prob": 0.0,
        "master_noise_std": 0.0,
    }
    base.update(overrides)
    return base


# These configs keep the learned latent proto-action, but add post-trajectory
# risk/distance supervision to make the master encode collision prevention.
RISK_AWARE_CONFIGS = [
    cfg("R00_emb2_baseline"),
    cfg("R01_risk_aux_005", risk_aux_weight=0.05),
    cfg("R02_risk_aux_010", risk_aux_weight=0.10),
    cfg("R03_dist_aux_005", distance_aux_weight=0.05, target_min_pairwise_dist=8.0),
    cfg("R04_risk_dist_005", risk_aux_weight=0.05, distance_aux_weight=0.05, target_min_pairwise_dist=8.0),
    cfg("R05_risk_dist_010_t10", risk_aux_weight=0.10, distance_aux_weight=0.05, target_min_pairwise_dist=10.0),
    cfg("R06_group_risk_dist", reward_mode="group", risk_aux_weight=0.05, distance_aux_weight=0.05, target_min_pairwise_dist=10.0),
    cfg("R07_low_std_risk_dist", master_log_std_init=-2.0, risk_aux_weight=0.05, distance_aux_weight=0.05),
    cfg("R08_dropout_risk_dist", risk_aux_weight=0.05, distance_aux_weight=0.05, master_dropout_prob=0.15),
    cfg(
        "R09_group_dropout_conflict",
        reward_mode="group",
        risk_aux_weight=0.05,
        distance_aux_weight=0.05,
        master_dropout_prob=0.15,
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


def risk_rank(rows: list[dict]) -> list[dict]:
    ranked = rank_for_paper(rows)
    for row in ranked:
        gm_auc = safe_float(row.get("gm_danger_auc"))
        lm_auc = safe_float(row.get("lm_best_danger_auc"))
        dist_drop = safe_float(row.get("min_distance_drop_worst_ablation"))
        close_inc = safe_float(row.get("close_rate_increase_worst_ablation"))
        arrival = safe_float(row.get("train_arrival_last50"))
        row["risk_proto_score"] = (
            0.03 * arrival
            + 3.0 * max(0.0, gm_auc - 0.5)
            + 3.0 * max(0.0, lm_auc - 0.5)
            + 1.0 * max(0.0, dist_drop)
            + 10.0 * max(0.0, close_inc)
        )
    ranked.sort(key=lambda r: safe_float(r.get("risk_proto_score")), reverse=True)
    return ranked


def plot_risk_summary(root: str, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots = [
        ("train_arrival_last50", "Train Arrival Last 50"),
        ("gm_danger_auc", "GM Danger AUROC"),
        ("lm_best_danger_auc", "Best LM Danger AUROC"),
        ("min_distance_drop_worst_ablation", "Ablation Distance Drop"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.bar(range(len(rows)), [safe_float(r.get(key)) for r in rows])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "risk_aware_proto_summary.png"), dpi=180)
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
            args.only_config = "R01_risk_aux_005"

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"proto_risk_aware_{ts}")
    os.makedirs(root, exist_ok=True)

    configs = [dict(c) for c in RISK_AWARE_CONFIGS]
    if args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown config: {args.only_config}")

    write_json(os.path.join(root, "risk_aware_config.json"), {"seed": SEED, "configs": configs})
    results = []
    for c in configs:
        results.append(run_config(c, root, episodes_override=args.episodes, test_override=args.test_episodes))
        write_json(os.path.join(root, "summary_partial.json"), results)

    write_json(os.path.join(root, "summary.json"), results)
    distance_rows = collect_distance_summary(root, configs)
    write_csv(os.path.join(root, "distance_config_ranking.csv"), distance_rows)
    plot_distance_summary(root, distance_rows)
    ranked = risk_rank(distance_rows)
    write_csv(os.path.join(root, "risk_proto_candidate_ranking.csv"), ranked)
    plot_risk_summary(root, ranked)
    print(f"Risk-aware proto sweep complete: {root}")


if __name__ == "__main__":
    main()
