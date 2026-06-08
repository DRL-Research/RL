"""
MODELS_EVALUATION-style coordination suite for a fine-tuned unified run dir.

**IDE:** use launch config ``Fine-tune: coordination suite (G03, 3 envs)`` or run this file (Play).

**CLI (optional):**

  py -3 play_finetuned_coordination_suite.py
  py -3 play_finetuned_coordination_suite.py --run-dir experiment_runs/fine_tune_id6/A_base/W_MASTER/s123
  py -3 play_finetuned_coordination_suite.py --eval-envs double_intersection

Writes under ``MODELS_EVALUATION/eval_<timestamp>_finetuned_coord/<pair_slug>/``
(intersection / roundabout / double_intersection dashboards, CSVs, pair_summary.json).

Each env evaluates **8 conditions** per scenario (G03 quartet + 4 coordination probes).
``dashboard_quartet_*.png`` = 4 conditions; ``dashboard_full_*.png`` = all 8.
Scenario count defaults to ``--n-scenarios 100`` (layout pool cap), unrelated to the 8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import eval_finetuned_unified_run as efu  # noqa: E402 — same default run-dir + ckpt naming
from eval_pretrained_on_custom_crossing import DEFAULT_SCENARIO_PKL  # noqa: E402
import run_g03_crossing_quad_bundle as bundle  # noqa: E402
from run_models_evaluation_suite import eval_pair_all_envs  # noqa: E402

_VALID_TAGS = frozenset({"intersection", "roundabout", "double_intersection"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=str,
        default="",
        help=f"Unified run dir containing best/ ckpts (default: {efu.DEFAULT_RUN_DIR} if present).",
    )
    p.add_argument(
        "--checkpoint",
        choices=("auto", "best", "final"),
        default="best",
        help="Checkpoint prefix under run-dir when not overriding .pth paths.",
    )
    p.add_argument("--agent-pth", type=str, default="", help="Override agent checkpoint path")
    p.add_argument("--master-pth", type=str, default="", help="Override master checkpoint path")
    p.add_argument("--g03-config", type=str, default=bundle.G03_DEFAULT_CONFIG)
    p.add_argument("--scenarios-pickle", type=str, default=DEFAULT_SCENARIO_PKL)
    p.add_argument("--n-scenarios", type=int, default=100)
    p.add_argument("--eval-run-id", type=int, default=0)
    p.add_argument(
        "--eval-envs",
        type=str,
        default="",
        help="Comma list: intersection,roundabout,double_intersection (empty = all three).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Batch folder (default: MODELS_EVALUATION/eval_<ts>_finetuned_coord).",
    )
    args = p.parse_args()

    env_tags: frozenset[str] | None = None
    raw = (args.eval_envs or "").strip()
    if raw:
        parts = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
        unknown = parts - _VALID_TAGS
        if unknown:
            print(f"Unknown --eval-envs tags {unknown}; allowed {_VALID_TAGS}", file=sys.stderr)
            sys.exit(2)
        env_tags = parts

    run_dir = efu.resolve_run_dir(args.run_dir)
    ap = (args.agent_pth or "").strip()
    mp = (args.master_pth or "").strip()
    if bool(ap) ^ bool(mp):
        p.error("Provide both --agent-pth and --master-pth, or neither.")

    if ap and mp:
        agent_pth = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
        master_pth = os.path.normpath(os.path.abspath(os.path.expanduser(mp)))
        if not os.path.isfile(agent_pth) or not os.path.isfile(master_pth):
            p.error("Explicit --agent-pth / --master-pth must be existing files.")
    else:
        mode = args.checkpoint if args.checkpoint != "auto" else "auto"
        prefix = efu._resolve_checkpoint_prefix(run_dir, mode)
        agent_pth = prefix + "_agent.pth"
        master_pth = prefix + "_master.pth"

    slug_tail = os.path.basename(os.path.normpath(run_dir))
    pair_slug = f"coord_{slug_tail}"

    if (args.output_root or "").strip():
        batch_root = os.path.normpath(os.path.abspath(os.path.expanduser(args.output_root.strip())))
    else:
        ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        batch_root = os.path.join(_REPO, "MODELS_EVALUATION", f"eval_{ts}_finetuned_coord")
    os.makedirs(batch_root, exist_ok=True)

    pair_out_dir = os.path.join(batch_root, pair_slug)

    print(
        "[coordination suite]\n"
        f"  run_dir:    {run_dir}\n"
        f"  agent:      {agent_pth}\n"
        f"  master:     {master_pth}\n"
        f"  out:        {pair_out_dir}\n"
        f"  env_filter: {sorted(env_tags) if env_tags else 'all'}\n",
        flush=True,
    )

    meta: dict = {
        "batch_root": batch_root,
        "run_dir": run_dir,
        "pair_slug": pair_slug,
        "agent_pth": agent_pth,
        "master_pth": master_pth,
        "eval_envs_filter": sorted(env_tags) if env_tags else "all",
    }
    ok = False
    err = ""
    try:
        summary = eval_pair_all_envs(
            agent_pth=agent_pth,
            master_pth=master_pth,
            pair_out_dir=pair_out_dir,
            g03_config=args.g03_config,
            scenarios_pickle=args.scenarios_pickle,
            n_scenarios_cap=args.n_scenarios,
            eval_run_id=args.eval_run_id,
            env_tags=env_tags,
        )
        meta["pair_summary_preview"] = {k: summary.get(k) for k in ("environments", "embedding_dim_used")}
        ok = True
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        print(f"[fail] {exc}", file=sys.stderr)

    meta["ok"] = ok
    meta["error"] = err
    out_js = os.path.join(batch_root, "batch_summary.json")
    with open(out_js, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if ok:
        print(f"[done]\n  {pair_out_dir}\n  batch_summary.json → {out_js}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
