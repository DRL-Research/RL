"""
Regional evaluation: normal vs zero_master on every REL environment (+ optional
fine-tuned checkpoint). Proves the unified fine-tune generalises.

Usage:
  py -3 test_all_environments_finetune.py --quick
  py -3 test_all_environments_finetune.py --agent path/to/agent.pth --master path/to/master.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np

import run_scalability_suite as rss
import run_proto_action_sweep as rps
import run_chain_scalability as rcs
import run_mixed_layer_experiment as mlx

CKPT_AGENT = os.path.join(_REPO, "models_to_check", "agent", "ckpt_agent6.pth")
CKPT_MASTER = os.path.join(_REPO, "models_to_check", "master", "ckpt_master6.pth")


def _eval_intersection(proto, master, agent, n_scen: int, cond: str) -> dict:
    ts = list(rps.BASE_CFG["target_speeds"])
    rng = np.random.default_rng(99)
    cell = rss.IntersectionCell(6, 3, ts)
    arr, cr = [], []
    for _ in range(n_scen):
        sc = rss._make_intersection_scenario(6, rng)
        r = rss.run_coordinated_episode([cell], [sc], master, agent, proto, cond, max_steps=80)
        arr.append(r["arrival_pct"])
        cr.append(r["any_crash"])
    cell.close()
    return {"arrival_pct": float(np.mean(arr)), "crash_pct": 100 * float(np.mean(cr))}


def _eval_chain(proto, master, agent, n_scen: int, cond: str) -> dict:
    from run_chain_scalability import ChainCell, run_chain_episode, CONDITIONS
    if cond not in CONDITIONS and cond != "normal":
        cond = "zero_master" if "zero" in cond else cond
    ts = list(rps.BASE_CFG["target_speeds"])
    rng = np.random.default_rng(99)
    cell = rcs.ChainCell(2, 6, ts)
    arr, cr = [], []
    for _ in range(n_scen):
        sc = rcs.generate_chain_scenario(2, 6, rng)
        r = run_chain_episode(cell, sc, master, agent, proto, cond)
        arr.append(r["arrival_pct"])
        cr.append(r.get("any_crash", r.get("crash", 0)))
    cell.close()
    return {"arrival_pct": float(np.mean(arr)), "crash_pct": 100 * float(np.mean(cr))}


def _eval_mixed(proto, master, agent, n_scen: int, cond: str) -> dict:
    ts = list(rps.BASE_CFG["target_speeds"])
    rng = np.random.default_rng(99)
    cell = rss.IntersectionCell(6, 3, ts)
    arr, cr = [], []
    for _ in range(n_scen):
        sc = rss._make_intersection_scenario(6, rng)
        r = mlx.run_mixed_episode(cell, sc, master, agent, proto, cond)
        arr.append(r["arrival_pct"])
        cr.append(r["crash"])
    cell.close()
    return {"arrival_pct": float(np.mean(arr)), "crash_pct": 100 * float(np.mean(cr))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--agent", default="")
    ap.add_argument("--master", default="")
    ap.add_argument("--out", default=os.path.join("PROFESSOR_RESULTS", "05_finetune_eval"))
    args = ap.parse_args()

    n_scen = 8 if args.quick else 25
    agent_p = args.agent.strip() or CKPT_AGENT
    master_p = args.master.strip() or CKPT_MASTER
    os.makedirs(args.out, exist_ok=True)

    proto, master, agent = rss.make_proto_and_models(agent_p, master_p)

    results = {}
    for env_name, fn in [
        ("intersection_6car", _eval_intersection),
        ("chain_2x3", _eval_chain),
        ("mixed_layer_6car", _eval_mixed),
    ]:
        results[env_name] = {}
        for cond in ("normal", "zero_master"):
            results[env_name][cond] = fn(proto, master, agent, n_scen, cond)
            s = results[env_name][cond]
            print(f"  {env_name:22s} {cond:12s}  arrival={s['arrival_pct']:5.1f}%  crash={s['crash_pct']:5.1f}%")

    out = {
        "checkpoints": {"agent": agent_p, "master": master_p},
        "n_scenarios": n_scen,
        "scenario_pool": rss.SCENARIO_POOL_NAME,
        "results": results,
        "note": "Roundabout/double use same 6-car hierarchical loop via intersection cell pattern in full suite.",
    }
    path = os.path.join(args.out, "regional_eval.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
