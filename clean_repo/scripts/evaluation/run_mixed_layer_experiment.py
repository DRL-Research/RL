"""
MIXED-LAYER scalability experiment: a master and raw vehicles share the SAME layer.

The hierarchy here is intentionally heterogeneous:

         GM  (top master)
       /  |  |  \
     LM   a3 a4 a5     <-- LM, a3, a4, a5 are SIBLINGS in the same layer
    / | \
  a0 a1 a2

The top master (GM) supervises children that are NOT all the same kind: one of
them is another master (the local master LM, over a0,a1,a2), the others are
ordinary agents (a3, a4, a5). This is possible because each master-input slot
carries an *identifier bit*: 1.0 marks a slot holding a (sub-)master embedding,
0.0 marks a slot holding a raw agent state. The network reads the same 25-D vector
regardless of whether a child is a master or a vehicle — so masters and vehicles
can be mixed freely on one layer. This experiment shows the pretrained model still
coordinates the scene under that heterogeneous structure.

The scene is the curated DENSE 6-car crossing (the master-dependent unit), so the
gap between conditions reflects real coordination, not an easy scenario:
  * normal      : full mixed hierarchy active.
  * zero_master : LM and GM embeddings forced to 0 (no coordination signal).
  * zero_top    : only the top (mixed) master's embedding forced to 0 — isolates
                  the contribution of the layer that mixes a master with vehicles.

Usage:
  py -3 run_mixed_layer_experiment.py --scenarios 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# repo root + every scripts/ sub-folder go on sys.path so cross-script imports
# (e.g. ``import run_scalability_suite``) keep working from any category folder.
for _p in (
    _REPO,
    os.path.join(_REPO, "scripts", "training"),
    os.path.join(_REPO, "scripts", "evaluation"),
    os.path.join(_REPO, "scripts", "visualization"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_scalability_suite as rss  # noqa: E402
import run_proto_action_sweep as rps  # noqa: E402

CKPT_AGENT = os.path.join(_REPO, "models_to_check", "agent", "ckpt_agent6.pth")
CKPT_MASTER = os.path.join(_REPO, "models_to_check", "master", "ckpt_master6.pth")

# Use the proven dense 6-car crossing (the curated, master-dependent unit where a
# coordinated policy reaches ~87% but an uncoordinated one collapses to ~55%). The
# 6 cars are split into the heterogeneous tree below so the scene stays IN the
# model's training distribution while still forcing genuine coordination.
N_CARS = 6
LM_AGENTS = [0, 1, 2]      # supervised by the child local master
TOP_AGENTS = [3, 4, 5]     # supervised DIRECTLY by the top (mixed) master
CONDITIONS = ["normal", "zero_master", "zero_top"]


def _zero(emb):
    return np.zeros_like(emb)


def run_mixed_episode(cell, scenario, master_model, agent_model, proto_exp, condition,
                      *, max_steps=80):
    """One episode of the GM -> [LM -> (a0,a1,a2), a3, a4, a5] mixed hierarchy."""
    states = cell.reset(scenario)
    emb_dim = proto_exp.embedding_dim
    top_prev = np.zeros(emb_dim, dtype=np.float32)
    steps = 0
    while steps < max_steps and not cell.done:
        steps += 1
        # 1) child local master over agents a0,a1,a2 (slot0 = top-master feedback).
        lm_in = rss.build_local_master_input(top_prev, states[LM_AGENTS], proto_exp)
        lm_emb = rss.master_embeddings_batch(master_model, [lm_in], proto_exp, False)[0]

        # 2) TOP master over a MIXED layer: [LM embedding (id=1), a2 (id=0), a3 (id=0)].
        #    build_local_master_input puts slot0=child-embedding(id=1), rest=agents(id=0).
        top_in = rss.build_local_master_input(lm_emb, states[TOP_AGENTS], proto_exp)
        top_emb = rss.master_embeddings_batch(master_model, [top_in], proto_exp, False)[0]

        # 3) apply the ablation condition to the broadcast embeddings.
        if condition == "zero_master":
            lm_used, top_used = _zero(lm_emb), _zero(top_emb)
        elif condition == "zero_top":
            lm_used, top_used = lm_emb, _zero(top_emb)
        else:
            lm_used, top_used = lm_emb, top_emb

        # 4) each agent observes ITS parent's embedding: a0,a1,a2 -> LM; a3,a4,a5 -> top master.
        obs = []
        for i in LM_AGENTS:
            obs.append(np.concatenate([states[i][:4], lm_used]).astype(np.float32))
        for i in TOP_AGENTS:
            obs.append(np.concatenate([states[i][:4], top_used]).astype(np.float32))
        order = LM_AGENTS + TOP_AGENTS
        acts, _, _ = rps.agent_actions(agent_model, obs, deterministic=True)
        action_by_car = {c: int(a) for c, a in zip(order, acts)}
        states = cell.step([action_by_car[c] for c in range(N_CARS)])
        top_prev = top_used  # feedback into the child LM next step

    return {
        "arrival_pct": 100.0 * cell.n_arrived() / N_CARS,
        "crash": int(cell.crashed),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, default=60)
    ap.add_argument("--out-dir", default=os.path.join("MODELS_EVALUATION", "mixed_layer"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    proto_exp, master_model, agent_model = rss.make_proto_and_models(CKPT_AGENT, CKPT_MASTER)
    target_speeds = list(rps.BASE_CFG["target_speeds"])

    rng = np.random.default_rng(2026)
    scenarios = [rss._make_intersection_scenario(N_CARS, rng) for _ in range(args.scenarios)]

    results = {c: {"arrival": [], "crash": []} for c in CONDITIONS}
    for cond in CONDITIONS:
        cell = rss.IntersectionCell(N_CARS, agents_per_lm=3, target_speeds=target_speeds)
        for sc in scenarios:
            r = run_mixed_episode(cell, sc, master_model, agent_model, proto_exp, cond)
            results[cond]["arrival"].append(r["arrival_pct"])
            results[cond]["crash"].append(r["crash"])
        cell.close()

    summary = {}
    for c in CONDITIONS:
        summary[c] = {
            "mean_arrival_pct": float(np.mean(results[c]["arrival"])),
            "crash_rate_pct": 100.0 * float(np.mean(results[c]["crash"])),
            "n_scenarios": args.scenarios,
        }

    out = {
        "experiment": "mixed_layer_master_and_vehicles_same_layer",
        "hierarchy": "GM -> [ LM -> (a0,a1,a2) , a3 , a4 , a5 ]",
        "n_cars": N_CARS,
        "lm_children": LM_AGENTS,
        "top_master_direct_children": TOP_AGENTS,
        "identifier_bit": "slot id=1.0 -> sub-master embedding, id=0.0 -> raw agent state",
        "checkpoint": {"agent": CKPT_AGENT, "master": CKPT_MASTER},
        "results": summary,
    }
    json_path = os.path.join(args.out_dir, "mixed_layer_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # bar plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["normal\n(mixed hierarchy)", "zero_master\n(all signals 0)", "zero_top\n(mixed master 0)"]
    arr = [summary[c]["mean_arrival_pct"] for c in CONDITIONS]
    crk = [summary[c]["crash_rate_pct"] for c in CONDITIONS]
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]
    axes[0].bar(labels, arr, color=colors)
    axes[0].set_title("Arrival rate"); axes[0].set_ylabel("%"); axes[0].set_ylim(0, 105)
    for i, v in enumerate(arr):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    axes[1].bar(labels, crk, color=colors)
    axes[1].set_title("Crash rate"); axes[1].set_ylabel("%"); axes[1].set_ylim(0, 105)
    for i, v in enumerate(crk):
        axes[1].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    fig.suptitle("Mixed-layer hierarchy  GM → [ LM → (a0,a1,a2) , a3 , a4 , a5 ]\n"
                 "master + vehicles on one layer, separated only by the identifier bit",
                 fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "mixed_layer_results.png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nMIXED-LAYER RESULTS  (GM -> [LM -> (a0,a1,a2), a3, a4, a5])")
    for c in CONDITIONS:
        s = summary[c]
        print(f"  {c:14s}  arrival={s['mean_arrival_pct']:5.1f}%   crash={s['crash_rate_pct']:5.1f}%")
    print(f"\n  json: {json_path}\n  plot: {png_path}")


if __name__ == "__main__":
    main()
