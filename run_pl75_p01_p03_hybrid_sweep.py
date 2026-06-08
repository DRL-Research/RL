"""
Small sweep: interpolate P01 (risk 0.01 @ start 1200) vs P03 (risk 0.05 @ start 1800)
to seek both strong LM PCA / danger AUC and a large zero_master test gap.
Same outputs layout as PL75_MASTER_PROOF sweeps.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

from run_pl75_master_proof_sweep import (
    PROOF_CONDITIONS,
    plot_proof_summary,
    summarize_for_proof,
    write_csv,
)
from run_proto_action_sweep import BASE_CFG, SEED, run_config, write_json
from run_proto_distance_sweep import collect_distance_summary


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


# P01 baseline: risk 0.01, start 1200, ramp 600
# P03 baseline: risk 0.05, start 1800, ramp 600
HYBRID_CONFIGS = [
    cfg("PH01_blend_003_s1200", risk_aux_weight=0.03, risk_aux_start_episode=1200, risk_aux_ramp_episodes=600),
    cfg("PH02_blend_003_s1500", risk_aux_weight=0.03, risk_aux_start_episode=1500, risk_aux_ramp_episodes=600),
    cfg("PH03_blend_003_s1800", risk_aux_weight=0.03, risk_aux_start_episode=1800, risk_aux_ramp_episodes=600),
    cfg("PH04_blend_004_s1800", risk_aux_weight=0.04, risk_aux_start_episode=1800, risk_aux_ramp_episodes=600),
    cfg("PH05_blend_025_s1200_slow", risk_aux_weight=0.025, risk_aux_start_episode=1200, risk_aux_ramp_episodes=900),
    cfg("PH06_blend_020_s1650", risk_aux_weight=0.02, risk_aux_start_episode=1650, risk_aux_ramp_episodes=750),
    cfg("PH07_blend_035_s1650", risk_aux_weight=0.035, risk_aux_start_episode=1650, risk_aux_ramp_episodes=750),
    cfg("PH08_p01_weight_late_start", risk_aux_weight=0.01, risk_aux_start_episode=1650, risk_aux_ramp_episodes=600),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--only-config", default="")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[],
        metavar="LABEL",
        help="Run several hybrid labels in one dated folder (overrides --only-config).",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 3
        args.test_episodes = 2
        if not args.only_config and not args.configs:
            args.only_config = "PH01_blend_003_s1200"

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"PL75_P01_P03_HYBRID_{ts}")
    os.makedirs(root, exist_ok=True)
    config_dir = os.path.join(root, "00_run_config")
    experiments_dir = os.path.join(root, "01_experiments")
    rankings_dir = os.path.join(root, "02_rankings")
    plots_dir = os.path.join(root, "03_summary_plots")
    for folder in (config_dir, experiments_dir, rankings_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)

    configs = [dict(c) for c in HYBRID_CONFIGS]
    if args.configs:
        labels_ordered = list(dict.fromkeys(args.configs))
        by_label = {c["label"]: dict(c) for c in HYBRID_CONFIGS}
        configs = []
        unknown = []
        for lbl in labels_ordered:
            c = by_label.get(lbl)
            if c is None:
                unknown.append(lbl)
            else:
                configs.append(c)
        if unknown:
            raise ValueError(f"Unknown config label(s): {unknown}")
        if not configs:
            raise ValueError("No matching configs after --configs filter.")
    elif args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown config: {args.only_config}")

    write_json(
        os.path.join(config_dir, "pl75_hybrid_config.json"),
        {"seed": SEED, "configs": configs, "eval_conditions": list(PROOF_CONDITIONS)},
    )
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
    write_json(
        os.path.join(root, "README_OUTPUT_STRUCTURE.json"),
        {
            "run_config": "00_run_config/pl75_hybrid_config.json",
            "per_experiment_outputs": "01_experiments/<CONFIG_LABEL>/",
            "rankings_and_raw_summaries": "02_rankings/",
            "aggregate_plots": "03_summary_plots/",
        },
    )
    print(f"PL75 P01/P03 hybrid sweep complete: {root}")


if __name__ == "__main__":
    main()
