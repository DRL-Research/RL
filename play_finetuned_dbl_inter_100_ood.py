"""
G03 coordination eval on **100 double-intersection OOD scenarios** (IDE Play, one click).

Runs **five** coordination conditions only (full hierarchy, ablations, probes):
  Full hierarchy, Zero master, Const broadcast, Swap LM1↔LM2, Zero global master (GM zeros).

Pool: 5 held-out regular bases + 95 deterministic jittered variants
(see ``src/experiment/double_intersection_100_ood_eval.py``).

Model paths are **hard-coded** below—edit ``_AGENT_PTH`` / ``_MASTER_PTH`` if your checkpoints move.

Outputs: ``MODELS_EVALUATION/eval_<timestamp>_dbl_inter_ood100/<slug>/double_intersection/``
         ``dashboard_full_arrival_and_crash.png``, second chart with same 5 conditions, CSV, ``pair_summary.json``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from eval_pretrained_on_custom_crossing import DEFAULT_SCENARIO_PKL  # noqa: E402
import run_g03_crossing_quad_bundle as bundle  # noqa: E402
from run_models_evaluation_suite import eval_pair_double_intersection_custom_pool  # noqa: E402
from src.experiment.double_intersection_100_ood_eval import (  # noqa: E402
    DOUBLE_INTERSECTION_100_OOD_EVAL_SCENARIOS,
)

# --- Edit these if needed -------------------------------------------------
_AGENT_PTH = os.path.join(
    _REPO,
    "experiment_runs",
    "fine_tune_id6",
    "A_base",
    "W_MASTER",
    "s123",
    "best",
    "ckpt_agent.pth",
)
_MASTER_PTH = os.path.join(
    _REPO,
    "experiment_runs",
    "fine_tune_id6",
    "A_base",
    "W_MASTER",
    "s123",
    "best",
    "ckpt_master.pth",
)
_G03_CONFIG = bundle.G03_DEFAULT_CONFIG
_SCENARIOS_PKL = DEFAULT_SCENARIO_PKL
_EVAL_RUN_ID = 0
# G03 condition keys (order = bar order). Labels on charts come from bundle presets.
# full hierarchy | zero master | const masters | swap LM1/LM2 | GM path zeros
_COORD_CONDITIONS = (
    "normal",
    "zero_master",
    "const_all_masters",
    "swap_local_masters",
    "zero_global_master",
)
# -------------------------------------------------------------------------


def main() -> None:
    agent_pth = os.path.normpath(_AGENT_PTH)
    master_pth = os.path.normpath(_MASTER_PTH)
    if not os.path.isfile(agent_pth) or not os.path.isfile(master_pth):
        sys.exit(f"Missing checkpoint files:\n  {agent_pth}\n  {master_pth}")

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    batch_root = os.path.join(_REPO, "MODELS_EVALUATION", f"eval_{ts}_dbl_inter_ood100")
    os.makedirs(batch_root, exist_ok=True)
    pair_out = os.path.join(batch_root, "coord_s123_dbl_inter_ood100")
    os.makedirs(pair_out, exist_ok=True)

    scenarios = DOUBLE_INTERSECTION_100_OOD_EVAL_SCENARIOS
    n = len(scenarios)
    print(
        "[double_intersection 100 OOD G03 — 5 conditions]\n"
        f"  agent:   {agent_pth}\n"
        f"  master:  {master_pth}\n"
        f"  n_scen:  {n}\n"
        f"  conditions: {list(_COORD_CONDITIONS)}\n"
        f"  out:     {pair_out}\n",
        flush=True,
    )

    summary = eval_pair_double_intersection_custom_pool(
        agent_pth=agent_pth,
        master_pth=master_pth,
        pair_out_dir=pair_out,
        g03_config=_G03_CONFIG,
        scenarios_pickle=_SCENARIOS_PKL,
        scenarios=list(scenarios),
        eval_run_id=_EVAL_RUN_ID,
        scenario_pool_key="ood_100_held_out_plus_jitter",
        metrics_csv_basename="coordination_metrics_double_intersection_ood100.csv",
        chart_suite_descriptor="100 OOD (5 held-out + 95 jittered)",
        pair_summary_extras={
            "n_scenarios_in_pool": n,
            "ood_eval_module": "src.experiment.double_intersection_100_ood_eval",
            "coordination_conditions": list(_COORD_CONDITIONS),
        },
        coordination_conditions=_COORD_CONDITIONS,
    )

    meta_path = os.path.join(batch_root, "batch_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"batch_root": batch_root, "pair_out": pair_out, "summary": summary}, f, indent=2)

    print(
        f"[done] plots: {pair_out}/double_intersection/dashboard_full_arrival_and_crash.png\n  {meta_path}"
    )


if __name__ == "__main__":
    main()
