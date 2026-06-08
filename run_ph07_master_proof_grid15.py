# Code and comments only in English.

"""
Focused 3000-episode sweep (15 configs) centred on hybrid PH07 to stress hierarchical
coordination signals for a paper-ready evidence bundle:

  * CT-style training rationale: extra global structure at train-time while controllers
    still execute through factored embeddings (survey: https://arxiv.org/abs/2409.03052).
  * Causal evaluation: paired counterfactuals (zero / disconnect / saturated GM, swaps,
    delay, negate, etc.).
  * Diagnostic geometry: manoeuvre mixtures on RELintersection (straight-heavy crossing
    patterns vs arcs that emulate roundabout-arm usage vs conflict/static-heavy pool).

Each run emits per-step master traces + PCA overlays, aggregated geometry CSVs/plots,
and `summary.json` with `best_ckpt_metric` (arrival rolling window by default,
optional composite arrival+reward snapshot for G15).
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
from run_pl75_p01_p03_hybrid_sweep import HYBRID_CONFIGS
from run_proto_action_sweep import SEED, run_config, write_json
from run_proto_distance_sweep import collect_distance_summary

# Extend PL75-proof counterfactuals with “saturated coordinator” GM replacement.
MASTER_PROOF_GRID15_EVAL_CONDITIONS: list[str] = list(PROOF_CONDITIONS) + [
    "large_const_global_master",
    "negate_global_master",
]

GEOMETRY_BUCKET_LEGEND = {
    "crossing_straight_heavy": (
        "Most controlled vehicles intend straight crossings through the junction "
        "(measured via origin/destination arm pairs)."
    ),
    "turn_arc_heavy": (
        "Many left/right turns that route through CircularLane arcs "
        '(analogous to “roundabout arm” manoeuvres inside the diamond).'
    ),
    "mixed_maneuvers": "No manoeuvre mode clearly dominates.",
    "static_conflict_heavy": (
        "Scenario sampled from the curated conflict/static-vehicle-heavy pool "
        "(harder occlusion / crossing geometry)."
    ),
}

_PH07_TEMPLATE = next(c for c in HYBRID_CONFIGS if c["label"] == "PH07_blend_035_s1650")


def _cfg(label: str, **overrides: object) -> dict:
    merged = dict(_PH07_TEMPLATE)
    merged["label"] = label
    merged["eval_conditions"] = MASTER_PROOF_GRID15_EVAL_CONDITIONS
    merged.update(overrides)
    return merged


PH07_MASTER_GRID15_CONFIGS = [
    _cfg("G01_PH07_proof_baseline", large_global_master_constant=28.0),
    _cfg("G02_proof_dist_aux015", distance_aux_weight=0.015, distance_aux_start_episode=1400, distance_aux_ramp_episodes=620),
    _cfg("G03_proof_dist_aux030", distance_aux_weight=0.030, distance_aux_start_episode=1350, distance_aux_ramp_episodes=700),
    _cfg("G04_proof_risk042_late1650", risk_aux_weight=0.042),
    _cfg("G05_proof_risk038_early1580", risk_aux_weight=0.038, risk_aux_start_episode=1580, risk_aux_ramp_episodes=780),
    _cfg(
        "G06_proof_master_low_std",
        master_log_std_init=-2.0,
    ),
    _cfg("G07_proof_group_credit", reward_mode="group"),
    _cfg(
        "G08_proof_conflict_curve",
        conflict_schedule=[(0, 0.0), (1100, 0.18), (2300, 0.34)],
        test_conflict_ratio=0.35,
    ),
    _cfg(
        "G09_proof_far_pairwise",
        target_min_pairwise_dist=10.0,
        distance_aux_weight=0.02,
        distance_aux_start_episode=1300,
        distance_aux_ramp_episodes=650,
    ),
    _cfg("G10_proof_master_dropout", master_dropout_prob=0.08),
    _cfg("G11_proof_master_noise", master_noise_std=0.025),
    _cfg(
        "G12_proof_mean_proto_eval",
        eval_policy_mean=True,
    ),
    _cfg(
        "G13_proof_mean_proto_train_eval",
        train_policy_mean=True,
        eval_policy_mean=True,
    ),
    _cfg(
        "G14_proof_group_plus_dist002",
        reward_mode="group",
        distance_aux_weight=0.02,
        distance_aux_start_episode=1400,
        distance_aux_ramp_episodes=600,
    ),
    _cfg(
        "G15_proof_composite_ckpt",
        distance_aux_weight=0.015,
        distance_aux_start_episode=1400,
        distance_aux_ramp_episodes=620,
        best_ckpt_metric="composite",
        best_ckpt_composite_reward_weight=0.42,
        best_ckpt_composite_reward_div=450.0,
    ),
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
        help="Run several grid labels inside one dated folder.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 5
        args.test_episodes = 3
        if not args.only_config and not args.configs:
            args.only_config = "G01_PH07_proof_baseline"

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"PH07_MASTER_PROOF_GRID15_{ts}")
    os.makedirs(root, exist_ok=True)
    config_dir = os.path.join(root, "00_run_config")
    experiments_dir = os.path.join(root, "01_experiments")
    rankings_dir = os.path.join(root, "02_rankings")
    plots_dir = os.path.join(root, "03_summary_plots")
    for folder in (config_dir, experiments_dir, rankings_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)

    configs = [dict(c) for c in PH07_MASTER_GRID15_CONFIGS]
    if args.configs:
        labels_ordered = list(dict.fromkeys(args.configs))
        by_label = {c["label"]: dict(c) for c in PH07_MASTER_GRID15_CONFIGS}
        configs = []
        unknown = []
        for lbl in labels_ordered:
            c = by_label.get(lbl)
            if c is None:
                unknown.append(lbl)
            else:
                configs.append(c)
        if unknown:
            raise ValueError(f"Unknown grid label(s): {unknown}")
        if not configs:
            raise ValueError("No configs after --configs.")
    elif args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown grid label: {args.only_config}")

    write_json(
        os.path.join(config_dir, "ph07_proof_grid15_config.json"),
        {
            "seed": SEED,
            "configs": configs,
            "eval_conditions": MASTER_PROOF_GRID15_EVAL_CONDITIONS,
            "geometry_bucket_legend": GEOMETRY_BUCKET_LEGEND,
            "notes": (
                "'best_rolling50' checkpoints maximise either rolling arrival (default) or "
                "a composite arrival+normalized reward score for G15. "
                "Per-episode columns scenario_pool/scenario_base/geometry_bucket appear "
                "in episode_metrics.csv and test_episode_metrics.csv."
            ),
        },
    )
    write_json(os.path.join(config_dir, "geometry_bucket_legend.json"), GEOMETRY_BUCKET_LEGEND)

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
            "run_config": "00_run_config/ph07_proof_grid15_config.json",
            "per_experiment_outputs": "01_experiments/<CONFIG_LABEL>/ (*_geometry_agg*.csv & breakdown plots)",
            "rankings_and_raw_summaries": "02_rankings/",
            "aggregate_plots": "03_summary_plots/",
        },
    )
    print(f"PH07 master-proof grid sweep complete (n={len(configs)} configs): {root}")


if __name__ == "__main__":
    main()
