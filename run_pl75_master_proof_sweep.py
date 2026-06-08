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
from run_proto_distance_sweep import collect_distance_summary


PROOF_CONDITIONS = [
    "normal",
    "zero_master",
    "zero_local_masters",
    "zero_global_master",
    "disconnect_global_master",
    "shuffle_global_master",
    "random_global_master",
    "delayed_global_master",
    "negate_master",
    "random_local_masters",
    "swap_local_masters",
]


def cfg(label: str, **overrides) -> dict:
    base = {
        **BASE_CFG,
        "label": label,
        "episodes": 3000,
        "test_episodes": 300,
        "embedding_dim": 4,
        "load_pretrained": True,
        "ent_coef": 0.005,
        "warmup_episodes": 200,
        "peak_arrival_threshold": 75.0,
        "test_conflict_ratio": 0.35,
        "target_min_pairwise_dist": 8.0,
        "close_distance_threshold": 8.0,
        "critical_distance_threshold": 4.0,
        "risk_aux_weight": 0.0,
        "distance_aux_weight": 0.0,
        "risk_aux_start_episode": 1200,
        "risk_aux_ramp_episodes": 600,
        "distance_aux_start_episode": 1200,
        "distance_aux_ramp_episodes": 600,
        "risk_horizon_steps": 5,
        "eval_conditions": PROOF_CONDITIONS,
    }
    base.update(overrides)
    return base


PL75_PROOF_CONFIGS = [
    cfg("P00_PL75_reference"),
    cfg("P01_PL75_risk_001", risk_aux_weight=0.01),
    cfg("P02_PL75_risk_002", risk_aux_weight=0.02),
    cfg("P03_PL75_risk_005_late", risk_aux_weight=0.05, risk_aux_start_episode=1800, risk_aux_ramp_episodes=600),
    cfg("P04_PL75_risk_005_slow", risk_aux_weight=0.05, risk_aux_start_episode=1000, risk_aux_ramp_episodes=1200),
    cfg("P05_PL75_distance_002", distance_aux_weight=0.02),
    cfg("P06_PL75_distance_005", distance_aux_weight=0.05),
    cfg("P07_PL75_risk001_dist002", risk_aux_weight=0.01, distance_aux_weight=0.02),
    cfg("P08_PL75_risk002_dist002", risk_aux_weight=0.02, distance_aux_weight=0.02),
    cfg("P09_PL75_risk005_dist002_late", risk_aux_weight=0.05, distance_aux_weight=0.02, risk_aux_start_episode=1800, distance_aux_start_episode=1400),
    cfg("P10_PL75_target10_risk002_dist002", target_min_pairwise_dist=10.0, risk_aux_weight=0.02, distance_aux_weight=0.02),
    cfg("P11_PL75_target10_dist003", target_min_pairwise_dist=10.0, distance_aux_weight=0.03),
    cfg("P12_PL75_conflict025", conflict_schedule=[(0, 0.0), (1200, 0.15), (2200, 0.25)]),
    cfg("P13_PL75_conflict025_risk002", risk_aux_weight=0.02, conflict_schedule=[(0, 0.0), (1200, 0.15), (2200, 0.25)]),
    cfg("P14_PL75_conflict035_risk_dist", risk_aux_weight=0.02, distance_aux_weight=0.02, conflict_schedule=[(0, 0.0), (1000, 0.15), (2000, 0.35)]),
    cfg("P15_PL75_low_std_late_risk", master_log_std_init=-2.0, risk_aux_weight=0.02, risk_aux_start_episode=1600),
    cfg("P16_PL75_eval_mean_risk", eval_policy_mean=True, risk_aux_weight=0.02),
    cfg("P17_PL75_train_eval_mean_light", train_policy_mean=True, eval_policy_mean=True, risk_aux_weight=0.01),
    cfg("P18_PL75_group_light_risk", reward_mode="group", risk_aux_weight=0.01),
    cfg("P19_PL75_group_risk_dist_conflict", reward_mode="group", risk_aux_weight=0.02, distance_aux_weight=0.02, conflict_schedule=[(0, 0.0), (1200, 0.15), (2200, 0.25)]),
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


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_for_proof(root: str, configs: list[dict]) -> list[dict]:
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
        cf = {r["condition"]: r for r in read_csv(cf_path)}
        phase = read_csv(phase_path) if os.path.exists(phase_path) else []
        normal = cf.get("normal", {})
        zero = cf.get("zero_master", {})
        zero_gm = cf.get("zero_global_master", {})
        disconnect_gm = cf.get("disconnect_global_master", {})
        random_gm = cf.get("random_global_master", {})
        negate = cf.get("negate_master", {})
        lm1 = next((r for r in phase if r.get("role") == "LM1"), {})
        lm2 = next((r for r in phase if r.get("role") == "LM2"), {})
        gm = next((r for r in phase if r.get("role") == "GM"), {})
        train_arr = [safe_float(r.get("arrival_pct")) for r in episodes]
        full_arrival = safe_float(normal.get("arrival_mean"))
        zero_arrival = safe_float(zero.get("arrival_mean"))
        no_gm_arrival = min(
            safe_float(zero_gm.get("arrival_mean"), full_arrival),
            safe_float(disconnect_gm.get("arrival_mean"), full_arrival),
            safe_float(random_gm.get("arrival_mean"), full_arrival),
        )
        full_crash = safe_float(normal.get("crash_rate"))
        zero_crash = safe_float(zero.get("crash_rate"))
        no_gm_crash = max(
            safe_float(zero_gm.get("crash_rate"), full_crash),
            safe_float(disconnect_gm.get("crash_rate"), full_crash),
            safe_float(random_gm.get("crash_rate"), full_crash),
        )
        lm_auc = max(safe_float(lm1.get("danger_auc_linear_centroid")), safe_float(lm2.get("danger_auc_linear_centroid")))
        gm_auc = safe_float(gm.get("danger_auc_linear_centroid"))
        proof_score = (
            0.04 * full_arrival
            - 0.02 * full_crash
            + 0.05 * max(0.0, full_arrival - zero_arrival)
            + 0.05 * max(0.0, full_arrival - no_gm_arrival)
            + 3.0 * max(0.0, lm_auc - 0.5)
            + 2.0 * max(0.0, gm_auc - 0.5)
            + (2.0 if full_arrival >= 85.0 else 0.0)
            + (2.0 if zero_arrival <= 60.0 else 0.0)
            + (2.0 if no_gm_arrival <= 65.0 else 0.0)
        )
        rows.append({
            "label": label,
            "train_arrival_last50": float(np.mean(train_arr[-50:])) if train_arr else 0.0,
            "train_arrival_last100": float(np.mean(train_arr[-100:])) if train_arr else 0.0,
            "full_arrival": full_arrival,
            "full_crash": full_crash,
            "zero_master_arrival": zero_arrival,
            "zero_master_crash": zero_crash,
            "no_gm_worst_arrival": no_gm_arrival,
            "no_gm_worst_crash": no_gm_crash,
            "full_minus_zero_arrival": full_arrival - zero_arrival,
            "full_minus_no_gm_arrival": full_arrival - no_gm_arrival,
            "negate_master_arrival": safe_float(negate.get("arrival_mean")),
            "normal_min_pairwise_dist": safe_float(normal.get("min_pairwise_dist_mean")),
            "zero_min_pairwise_dist": safe_float(zero.get("min_pairwise_dist_mean")),
            "lm_best_danger_auc": lm_auc,
            "gm_danger_auc": gm_auc,
            "gm_safe_danger_kl": safe_float(gm.get("safe_vs_danger_symmetric_kl")),
            "paper_pass_full_85": int(full_arrival >= 85.0),
            "paper_pass_zero_60": int(zero_arrival <= 60.0),
            "paper_pass_no_gm_65": int(no_gm_arrival <= 65.0),
            "proof_score": proof_score,
        })
    rows.sort(key=lambda r: safe_float(r["proof_score"]), reverse=True)
    return rows


def plot_proof_summary(root: str, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    plots = [
        ("full_arrival", "Full Master Test Arrival"),
        ("zero_master_arrival", "Zero-Master Test Arrival"),
        ("no_gm_worst_arrival", "Worst No-GM Test Arrival"),
        ("lm_best_danger_auc", "Best LM Safe/Danger AUROC"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.bar(range(len(rows)), [safe_float(r.get(key)) for r in rows])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "pl75_master_proof_summary.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(rows))
    ax.bar(x - 0.25, [safe_float(r["full_arrival"]) for r in rows], width=0.25, label="full")
    ax.bar(x, [safe_float(r["zero_master_arrival"]) for r in rows], width=0.25, label="zero master")
    ax.bar(x + 0.25, [safe_float(r["no_gm_worst_arrival"]) for r in rows], width=0.25, label="no/disconnected GM")
    ax.axhline(90, color="green", linestyle="--", linewidth=1, label="90% target")
    ax.axhline(60, color="red", linestyle="--", linewidth=1, label="60% ablation target")
    ax.set_title("Causal Master Proof: Full vs Zero vs No-GM")
    ax.set_ylabel("Arrival %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(root, "full_vs_zero_vs_no_gm_arrival.png"), dpi=180)
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
            args.only_config = "P01_PL75_risk_001"

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"PL75_MASTER_PROOF_{ts}")
    os.makedirs(root, exist_ok=True)
    config_dir = os.path.join(root, "00_run_config")
    experiments_dir = os.path.join(root, "01_experiments")
    rankings_dir = os.path.join(root, "02_rankings")
    plots_dir = os.path.join(root, "03_summary_plots")
    for folder in (config_dir, experiments_dir, rankings_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)

    configs = [dict(c) for c in PL75_PROOF_CONFIGS]
    if args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown config: {args.only_config}")

    write_json(os.path.join(config_dir, "pl75_master_proof_config.json"), {"seed": SEED, "configs": configs, "eval_conditions": PROOF_CONDITIONS})
    results = []
    for c in configs:
        results.append(run_config(c, experiments_dir, episodes_override=args.episodes, test_override=args.test_episodes))
        write_json(os.path.join(rankings_dir, "summary_partial.json"), results)

    write_json(os.path.join(rankings_dir, "summary.json"), results)
    distance_rows = collect_distance_summary(experiments_dir, configs)
    write_csv(os.path.join(rankings_dir, "distance_config_ranking.csv"), distance_rows)
    proof_rows = summarize_for_proof(experiments_dir, configs)
    write_csv(os.path.join(rankings_dir, "pl75_master_proof_ranking.csv"), proof_rows)
    plot_proof_summary(plots_dir, proof_rows)
    write_json(os.path.join(root, "README_OUTPUT_STRUCTURE.json"), {
        "run_config": "00_run_config/pl75_master_proof_config.json",
        "per_experiment_outputs": "01_experiments/<CONFIG_LABEL>/",
        "rankings_and_raw_summaries": "02_rankings/",
        "aggregate_plots": "03_summary_plots/",
    })
    print(f"PL75 master proof sweep complete: {root}")


if __name__ == "__main__":
    main()
