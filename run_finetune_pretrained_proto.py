"""
For the **`run_unified.py` Phase-1 recipe** (e.g. A_base / W_MASTER / three-env mix), prefer
``continue_unified_training_from_ckpt.py`` instead of this script.

Fine-tune an existing hierarchical *proto* checkpoint (agent.pth + master.pth) on RELintersection-v0
with gradually harder conflict-heavy scenarios — intended continuation from checkpoints like ckpt_id6.

Goals (monitor in output summary.json — not enforced as a constrained optimiser):
  • Raise test `normal` arrival (target often ~≥90%).
  • Keep test `zero_master` meaningfully BELOW `normal` (your earlier gates used <60 arrival under
    zero-master — if this gap closes, Kinect-style agents may ignore LM broadcasts).

The training loop matches `run_proto_action_sweep.run_config` (warm-start via init_checkpoint_paths).
Post-hoc triple-env report: rerun `run_models_evaluation_suite.py` on the saved `trained_model/*.pth`
or `best/ckpt` pair.

Examples
--------
  cd <repo-root>

  py -3.11 run_finetune_pretrained_proto.py ^
    --agent models_to_check/agent/ckpt_agent6.pth ^
    --master models_to_check/master/ckpt_master6.pth ^
    --label FINETUNE_id6_v1

  py -3.11 run_finetune_pretrained_proto.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import run_g03_crossing_quad_bundle as bundle
import run_proto_action_sweep as rps
from run_models_evaluation_suite import resolve_embedding_dim


def _apply_finetune_overrides(
    cfg: dict[str, Any],
    *,
    lr_scale: float,
    collision_reward_scale: float,
    episodes: int,
    test_episodes: int,
    conflict_schedule_auto: bool,
    eval_conditions: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["label"] = label
    cfg["load_pretrained"] = False
    cfg.setdefault("master_broadcast_const_test", 9999.0)

    for k in ("agent_lr", "master_lr"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k]) * float(lr_scale)

    if "collision_reward" in cfg and cfg["collision_reward"] is not None:
        cfg["collision_reward"] = float(cfg["collision_reward"]) * float(collision_reward_scale)

    eps = max(300, int(episodes))
    cfg["episodes"] = eps
    cfg["test_episodes"] = max(40, int(test_episodes))
    cfg["eval_conditions"] = list(eval_conditions)
    cfg["test_conflict_ratio"] = float(cfg.get("test_conflict_ratio", 0.35))

    if conflict_schedule_auto or not cfg.get("conflict_schedule"):
        cfg["conflict_schedule"] = [
            (0, 0.12),
            (max(1, eps // 5), 0.20),
            (max(1, eps // 2), 0.28),
            (max(1, int(0.72 * eps)), 0.34),
        ]

    cfg["best_ckpt_metric"] = str(cfg.get("best_ckpt_metric") or "composite")
    cfg.setdefault("best_ckpt_composite_arrival_weight", 1.0)
    cfg.setdefault("best_ckpt_composite_reward_weight", 0.35)
    cfg.setdefault("best_ckpt_composite_reward_div", 400.0)
    return cfg


def main() -> None:
    default_ag = os.path.join(_REPO, "models_to_check", "agent", "ckpt_agent6.pth")
    default_ms = os.path.join(_REPO, "models_to_check", "master", "ckpt_master6.pth")
    parser = argparse.ArgumentParser(description="Proto fine-tune from agent/master checkpoints.")
    parser.add_argument("--agent", type=str, default=default_ag)
    parser.add_argument("--master", type=str, default=default_ms)
    parser.add_argument("--base-config", type=str, default=bundle.G03_DEFAULT_CONFIG)
    parser.add_argument("--label", type=str, default="", help="Run label subdirectory (default FINETUNE_<ts>).")
    parser.add_argument("--output-root", type=str, default="", help="experiment_runs/PARENT — default FINETUNE_proto_<datetime>.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=120)
    parser.add_argument("--lr-scale", type=float, default=0.45)
    parser.add_argument(
        "--collision-reward-scale",
        type=float,
        default=1.08,
        help="Multiply collision_reward (<0). >1 ⇒ stronger collisions penalty.",
    )
    parser.add_argument(
        "--no-auto-conflict-schedule",
        action="store_true",
        help="Keep conflict_schedule from base JSON verbatim (needs non-empty schedule).",
    )
    parser.add_argument(
        "--eval-conditions",
        type=str,
        default="",
        help='Comma-separated; default = G03 quartet + coordination probes (same as MODELS_EVALUATION suite).',
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny run for wiring check.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    label = (args.label or "").strip() or f"FINETUNE_{ts}"
    root_parent = (
        os.path.normpath(os.path.abspath(os.path.expanduser(args.output_root.strip())))
        if (args.output_root or "").strip()
        else os.path.join(_REPO, "experiment_runs", f"FINETUNE_proto_{ts}")
    )
    os.makedirs(root_parent, exist_ok=True)

    agent_pth = os.path.abspath(os.path.expanduser(args.agent.strip()))
    master_pth = os.path.abspath(os.path.expanduser(args.master.strip()))
    if not os.path.isfile(agent_pth) or not os.path.isfile(master_pth):
        print("Missing --agent or --master .pth file.", file=sys.stderr)
        sys.exit(2)

    base_cfg_path = os.path.abspath(os.path.expanduser(args.base_config.strip()))
    base_cfg = bundle.load_json(base_cfg_path)

    episodes = 80 if args.smoke else int(args.episodes)
    test_episodes = 8 if args.smoke else int(args.test_episodes)

    if (args.eval_conditions or "").strip():
        eval_t = tuple(
            bundle.normalize_eval_conditions(
                tuple(p.strip() for p in args.eval_conditions.split(",") if p.strip())
            )
        )
    else:
        eval_t = bundle.default_plus_coordination_probes_order()

    emb = resolve_embedding_dim(master_pth, agent_pth, base_cfg)
    merged = _apply_finetune_overrides(
        {**base_cfg, "embedding_dim": emb},
        lr_scale=float(args.lr_scale),
        collision_reward_scale=float(args.collision_reward_scale),
        episodes=episodes,
        test_episodes=test_episodes,
        conflict_schedule_auto=not args.no_auto_conflict_schedule,
        eval_conditions=eval_t,
        label=label,
    )

    meta = {
        "base_config_path": base_cfg_path,
        "agent_pth": agent_pth,
        "master_pth": master_pth,
        "embedding_dim_restored": emb,
        "label": label,
        "fine_tune_args": vars(args),
    }
    write_meta = os.path.join(root_parent, "finetune_meta.json")
    with open(write_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        json.dumps({"embedding_dim_restored": emb, "episode_plan": episodes, "test_episodes_plan": test_episodes}, indent=2)
    )
    result = rps.run_config(
        merged,
        root_parent,
        episodes_override=None,
        test_override=None,
        init_checkpoint_paths=(agent_pth, master_pth),
    )

    print("\nFine-tune finished. Keys to watch in summary.json under the label folder:")
    print("  test_arrival_normal         (want ~≥90 after enough episodes)")
    print("  test_arrival_zero_master   (want sufficiently below normal — if it climbs to ~normal, embeddings matter less)")
    print(f"\nArtifacts: {os.path.join(root_parent, label)}")
    print(f"Checkpoint prefix to load for re-eval: trained_model OR best/ckpt (see summary eval_checkpoint)")
    print(json.dumps({k: result.get(k) for k in ("test_arrival_normal", "test_arrival_zero_master", "eval_checkpoint") if k in result}, indent=2))


if __name__ == "__main__":
    main()
