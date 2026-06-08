"""
G03 coordination eval on **double-intersection held-out regular** scenarios only.

Loads fine-tuned (or any) agent/master checkpoints and writes the same plots as the full
MODELS_EVALUATION run: ``dashboard_full_arrival_and_crash.png`` and ``dashboard_quartet_*.png``
under ``<out>/double_intersection/``.

Held-out indices match training: ``DOUBLE_INTERSECTION_HELD_OUT_INDICES`` in
``src/experiment/scenarios.py`` (5 regular layouts by default — not the larger
train-pool slice used in older ``n=14`` dashboard exports).

**IDE:** Run this file or add a launch config pointing here.

  py -3 play_finetuned_double_intersection_held_out.py
  py -3 play_finetuned_double_intersection_held_out.py --agent-pth ... --master-pth ...
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

import eval_finetuned_unified_run as efu  # noqa: E402
from eval_pretrained_on_custom_crossing import DEFAULT_SCENARIO_PKL  # noqa: E402
import run_g03_crossing_quad_bundle as bundle  # noqa: E402
from run_models_evaluation_suite import eval_pair_double_intersection_held_out_only  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=str, default="", help=f"Fine-tune run dir (default: {efu.DEFAULT_RUN_DIR} if present).")
    p.add_argument("--checkpoint", choices=("auto", "best", "final"), default="best")
    p.add_argument("--agent-pth", type=str, default="")
    p.add_argument("--master-pth", type=str, default="")
    p.add_argument("--g03-config", type=str, default=bundle.G03_DEFAULT_CONFIG)
    p.add_argument("--scenarios-pickle", type=str, default=DEFAULT_SCENARIO_PKL)
    p.add_argument("--n-scenarios", type=int, default=100, help="Cap held-out list (default uses all held-out).")
    p.add_argument("--eval-run-id", type=int, default=0)
    p.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Batch folder (default: MODELS_EVALUATION/eval_<ts>_dbl_inter_held_out).",
    )
    args = p.parse_args()

    ap = (args.agent_pth or "").strip()
    mp = (args.master_pth or "").strip()
    if bool(ap) ^ bool(mp):
        p.error("Provide both --agent-pth and --master-pth, or neither.")

    run_dir = efu.resolve_run_dir(args.run_dir)

    if ap and mp:
        agent_pth = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
        master_pth = os.path.normpath(os.path.abspath(os.path.expanduser(mp)))
        if not os.path.isfile(agent_pth) or not os.path.isfile(master_pth):
            p.error("Explicit paths must be existing files.")
    else:
        mode = args.checkpoint if args.checkpoint != "auto" else "auto"
        prefix = efu._resolve_checkpoint_prefix(run_dir, mode)
        agent_pth = prefix + "_agent.pth"
        master_pth = prefix + "_master.pth"

    slug = "coord_" + os.path.basename(os.path.normpath(run_dir)) + "_dbl_inter_held_out"
    if (args.output_root or "").strip():
        batch_root = os.path.normpath(os.path.abspath(os.path.expanduser(args.output_root.strip())))
    else:
        ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        batch_root = os.path.join(_REPO, "MODELS_EVALUATION", f"eval_{ts}_dbl_inter_held_out")
    os.makedirs(batch_root, exist_ok=True)
    pair_out = os.path.join(batch_root, slug)

    print(
        "[double_intersection held-out G03]\n"
        f"  run_dir: {run_dir}\n"
        f"  agent:   {agent_pth}\n"
        f"  master:  {master_pth}\n"
        f"  out:     {pair_out}\n",
        flush=True,
    )

    summary = eval_pair_double_intersection_held_out_only(
        agent_pth=agent_pth,
        master_pth=master_pth,
        pair_out_dir=pair_out,
        g03_config=args.g03_config,
        scenarios_pickle=args.scenarios_pickle,
        n_scenarios_cap=args.n_scenarios,
        eval_run_id=args.eval_run_id,
    )

    meta_path = os.path.join(batch_root, "batch_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"batch_root": batch_root, "pair_out": pair_out, "summary": summary}, f, indent=2)

    di = (summary.get("environments") or {}).get("double_intersection") or {}
    n = di.get("n_scenarios", "?")
    print(f"[done] n_scenarios={n}\n  plots: {pair_out}/double_intersection/dashboard_full_arrival_and_crash.png\n  {meta_path}")


if __name__ == "__main__":
    main()
