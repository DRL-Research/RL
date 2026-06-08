"""
Dynamic hierarchy evaluation.

The earlier suites used a fixed layout: a known number of agents per master and a
known number of masters per intersection, decided in advance. This script tests
the opposite case. The number of agents in a single intersection changes over
time (it grows and then shrinks), and the master hierarchy is rebuilt on the fly
to match.

The rule is capacity based. One local master (LM) can supervise at most 4 agents
(the master input has 5 slots, slot 0 is reserved for parent feedback, so 4 slots
are left for subordinates). So:

    1..4  agents  -> 1 LM,                no global master
    5..8  agents  -> 2 LMs               + 1 global master (GM)
    9..12 agents  -> 3 LMs               + 1 GM
    ...
    >20   agents  -> >5 LMs, grouped by intermediate masters (fan-in 5) under 1 GM

The same shared master weights and the same shared agent weights are used at every
size. Nothing is retrained. Adding or removing a master only changes how the input
is packed.

The experiment ramps the agent count UP from 1 to a maximum, then back DOWN to 1.
At every step it evaluates the current architecture on a batch of scenarios and
records arrival rate and crash rate. If the live architecture change is sound, the
up ramp and the down ramp should produce the same numbers at the same agent count
(no degradation introduced by adding or removing a master).

CLI:
    py -3 run_dynamic_hierarchy.py --smoke
    py -3 run_dynamic_hierarchy.py --max-agents 12 --scenarios 40
    py -3 run_dynamic_hierarchy.py --max-agents 16 --scenarios 60 --conditions normal,zero_master
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
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

LM_CAPACITY = 4   # agents one local master can supervise (5 slots - 1 feedback slot)
GM_FANOUT = 5     # masters one parent master can aggregate (all 5 slots are children)
CONDITIONS_DEFAULT = ("normal", "zero_master")


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchy builder: a pure function of the agent count.
# ──────────────────────────────────────────────────────────────────────────────
def build_hierarchy(n_agents: int, capacity: int = LM_CAPACITY, fanout: int = GM_FANOUT) -> dict[str, Any]:
    """Describe the master tree needed for ``n_agents`` in one intersection.

    Returns the leaf local-master groups (contiguous, filled to capacity, matching
    IntersectionCell._lm_groups) plus the number of master nodes and tree depth.
    """
    if n_agents < 1:
        return {"n_agents": 0, "n_lms": 0, "groups": [], "has_gm": False,
                "n_intermediate": 0, "total_masters": 0, "levels": 0}

    groups = [list(range(g, min(g + capacity, n_agents)))
              for g in range(0, n_agents, capacity)]
    n_lms = len(groups)

    # One LM and nothing above it.
    if n_lms == 1:
        return {"n_agents": n_agents, "n_lms": 1, "groups": groups, "has_gm": False,
                "n_intermediate": 0, "total_masters": 1, "levels": 1}

    # Aggregate LMs upward in chunks of ``fanout`` until a single root remains.
    total_masters = n_lms
    intermediate = 0
    levels = 1  # the LM level
    cur = n_lms
    while cur > 1:
        parents = math.ceil(cur / fanout)
        total_masters += parents
        if parents > 1:
            intermediate += parents
        cur = parents
        levels += 1
    # The final single parent is the GM; everything between LMs and GM is intermediate.
    return {"n_agents": n_agents, "n_lms": n_lms, "groups": groups, "has_gm": True,
            "n_intermediate": intermediate, "total_masters": total_masters, "levels": levels}


def describe_hierarchy(h: dict[str, Any]) -> str:
    """One-line human-readable summary of a hierarchy dict."""
    if h["n_agents"] == 0:
        return "empty"
    if not h["has_gm"]:
        return f"{h['n_agents']} agents -> 1 LM (no GM)"
    parts = [f"{h['n_agents']} agents", f"{h['n_lms']} LMs"]
    if h["n_intermediate"] > 0:
        parts.append(f"{h['n_intermediate']} intermediate masters")
    parts.append("1 GM")
    return " -> ".join(parts)


def architecture_change_points(max_agents: int, capacity: int = LM_CAPACITY) -> list[int]:
    """Agent counts at which the number of local masters increases (capacity+1,
    2*capacity+1, ...). These are the boundaries where a master is added."""
    pts = []
    n = capacity + 1
    while n <= max_agents:
        pts.append(n)
        n += capacity
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation of one agent count.
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_count(
    n_agents: int,
    master_model,
    agent_model,
    proto_exp,
    *,
    n_scenarios: int,
    base_seed: int,
    target_speeds: list[int],
    conditions: tuple[str, ...],
    cell_cache: dict[int, rss.IntersectionCell],
    max_steps: int = 80,
) -> dict[str, Any]:
    """Run ``n_scenarios`` episodes for a single intersection holding ``n_agents``.

    The IntersectionCell is built with agents_per_lm = LM_CAPACITY, so it already
    partitions the agents into the right local-master groups and the coordinated
    episode runner aggregates them through the shared master (recursively if there
    are more than 5 LMs). Cells are cached by agent count so the down ramp reuses
    the exact same environment that the up ramp built (a genuine live reuse).
    """
    h = build_hierarchy(n_agents)

    if n_agents not in cell_cache:
        cell_cache[n_agents] = rss.IntersectionCell(n_agents, LM_CAPACITY, target_speeds)
    cell = cell_cache[n_agents]

    # Deterministic per-count scenarios (same pool regardless of ramp direction).
    rng = np.random.default_rng(base_seed + 1000 * n_agents)
    scenarios = [rss._make_intersection_scenario(n_agents, rng) for _ in range(n_scenarios)]

    out: dict[str, Any] = {
        "n_agents": n_agents,
        "hierarchy": describe_hierarchy(h),
        "n_lms": h["n_lms"],
        "n_intermediate": h["n_intermediate"],
        "has_gm": h["has_gm"],
        "total_masters": h["total_masters"],
        "levels": h["levels"],
        "conditions": {},
    }
    for condition in conditions:
        arrivals, crashes, steps = [], [], []
        for ep in range(n_scenarios):
            rps.set_all_seeds(base_seed + ep)
            res = rss.run_coordinated_episode(
                [cell], [scenarios[ep]], master_model, agent_model, proto_exp,
                condition, max_steps=max_steps,
            )
            arrivals.append(res["arrival_pct"])
            crashes.append(res["any_crash"])
            steps.append(res["steps"])
        arr = np.asarray(arrivals, dtype=float)
        out["conditions"][condition] = {
            "arrival_pct_mean": float(np.mean(arr)),
            "crash_rate_pct": 100.0 * float(np.mean(crashes)),
            "mean_steps": float(np.mean(steps)),
            "n_scenarios": int(arr.size),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The ramp: up from 1 to max, then back down to 1.
# ──────────────────────────────────────────────────────────────────────────────
def run_ramp(
    max_agents: int,
    master_model,
    agent_model,
    proto_exp,
    *,
    n_scenarios: int,
    base_seed: int,
    target_speeds: list[int],
    conditions: tuple[str, ...],
) -> list[dict[str, Any]]:
    up = list(range(1, max_agents + 1))
    down = list(range(max_agents - 1, 0, -1))
    schedule = [("up", n) for n in up] + [("down", n) for n in down]

    cell_cache: dict[int, rss.IntersectionCell] = {}
    records: list[dict[str, Any]] = []
    try:
        for direction, n in schedule:
            rec = evaluate_count(
                n, master_model, agent_model, proto_exp,
                n_scenarios=n_scenarios, base_seed=base_seed,
                target_speeds=target_speeds, conditions=conditions,
                cell_cache=cell_cache,
            )
            rec["direction"] = direction
            records.append(rec)
            normal = rec["conditions"].get("normal", {})
            print(f"  [{direction:4s}] n={n:2d}  {rec['hierarchy']:42s}  "
                  f"arrival={normal.get('arrival_pct_mean', float('nan')):5.1f}%  "
                  f"crash={normal.get('crash_rate_pct', float('nan')):5.1f}%")
    finally:
        for c in cell_cache.values():
            c.close()
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Plots and tables.
# ──────────────────────────────────────────────────────────────────────────────
def _series(records: list[dict[str, Any]], direction: str, condition: str, key: str):
    xs, ys = [], []
    for r in records:
        if r["direction"] != direction:
            continue
        c = r["conditions"].get(condition)
        if c is None:
            continue
        xs.append(r["n_agents"])
        ys.append(c[key])
    return xs, ys


def make_plots(records: list[dict[str, Any]], conditions: tuple[str, ...],
               max_agents: int, out_dir: str) -> None:
    change_pts = architecture_change_points(max_agents)

    for metric_key, metric_label, fname in [
        ("crash_rate_pct", "Crash rate (%)", "dynamic_crash.png"),
        ("arrival_pct_mean", "Arrival rate (%)", "dynamic_arrival.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        styles = {"up": ("-o", 1.0), "down": ("--s", 0.6)}
        colors = {"normal": "#2ca02c", "zero_master": "#d62728"}
        for condition in conditions:
            col = colors.get(condition, None)
            for direction, (style, alpha) in styles.items():
                xs, ys = _series(records, direction, condition, metric_key)
                if not xs:
                    continue
                ax.plot(xs, ys, style, color=col, alpha=alpha,
                        label=f"{condition} ({direction})", markersize=5)
        for i, cp in enumerate(change_pts):
            ax.axvline(cp - 0.5, color="gray", linestyle=":", alpha=0.6,
                       label="master added" if i == 0 else None)
        ax.set_xlabel("Agents in the intersection")
        ax.set_ylabel(metric_label)
        ax.set_title("Dynamic hierarchy: ramp up then down\n"
                     "vertical lines mark where a local master is added")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot: {path}")


def write_summary(records: list[dict[str, Any]], conditions: tuple[str, ...],
                  max_agents: int, out_dir: str) -> None:
    lines = ["# Dynamic Hierarchy - live architecture change", "",
             "One intersection. Agent count ramps up from 1 to "
             f"{max_agents}, then back down to 1. The master tree is rebuilt for "
             "each count using the same shared weights (no retraining).", "",
             f"- Local-master capacity: {LM_CAPACITY} agents",
             f"- A new local master is added at: {architecture_change_points(max_agents)}",
             "",
             "## Up ramp vs down ramp (normal condition)", "",
             "| agents | hierarchy | crash up | crash down | arrival up | arrival down |",
             "|---:|---|---:|---:|---:|---:|"]

    def _val(direction, n, cond, key):
        for r in records:
            if r["direction"] == direction and r["n_agents"] == n:
                c = r["conditions"].get(cond)
                if c:
                    return c[key]
        return None

    for n in range(1, max_agents + 1):
        h = describe_hierarchy(build_hierarchy(n))
        cu = _val("up", n, "normal", "crash_rate_pct")
        cd = _val("down", n, "normal", "crash_rate_pct")
        au = _val("up", n, "normal", "arrival_pct_mean")
        ad = _val("down", n, "normal", "arrival_pct_mean")
        def fmt(x):
            return f"{x:.1f}%" if x is not None else "-"
        lines.append(f"| {n} | {h} | {fmt(cu)} | {fmt(cd)} | {fmt(au)} | {fmt(ad)} |")

    path = os.path.join(out_dir, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  summary: {path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-agents", type=int, default=12,
                    help="peak agent count for the ramp (default 12)")
    ap.add_argument("--scenarios", type=int, default=40,
                    help="scenarios evaluated per agent count (default 40)")
    ap.add_argument("--conditions", default="normal,zero_master",
                    help="comma list from {normal,zero_master} (default both)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--agent", default="")
    ap.add_argument("--master", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="quick run: max-agents 6, 8 scenarios")
    args = ap.parse_args()

    if args.smoke:
        args.max_agents = 6
        args.scenarios = 8

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())

    agent_pth = os.path.abspath(args.agent.strip() or os.path.join(
        _REPO, "models", "agent", "agent.pth"))
    master_pth = os.path.abspath(args.master.strip() or os.path.join(
        _REPO, "models", "master", "master.pth"))

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = args.out_dir.strip() or os.path.join(
        _REPO, "EVALUATION_RESULTS", f"dynamic_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[dynamic] agent  -> {agent_pth}")
    print(f"[dynamic] master -> {master_pth}")
    print(f"[dynamic] ramp 1..{args.max_agents}..1, {args.scenarios} scenarios/count, "
          f"conditions={conditions}")
    print(f"[dynamic] output -> {out_dir}\n")

    proto_exp, master_model, agent_model = rss.make_proto_and_models(agent_pth, master_pth)
    target_speeds = list(rps.BASE_CFG["target_speeds"])

    records = run_ramp(
        args.max_agents, master_model, agent_model, proto_exp,
        n_scenarios=args.scenarios, base_seed=args.seed,
        target_speeds=target_speeds, conditions=conditions,
    )

    payload = {
        "experiment": "dynamic_hierarchy_live_architecture_change",
        "lm_capacity": LM_CAPACITY,
        "gm_fanout": GM_FANOUT,
        "max_agents": args.max_agents,
        "n_scenarios": args.scenarios,
        "conditions": list(conditions),
        "architecture_change_points": architecture_change_points(args.max_agents),
        "checkpoint": {"agent": agent_pth, "master": master_pth},
        "records": records,
    }
    json_path = os.path.join(out_dir, "dynamic_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  json: {json_path}")

    make_plots(records, conditions, args.max_agents, out_dir)
    write_summary(records, conditions, args.max_agents, out_dir)
    print(f"\n[dynamic] done -> {out_dir}")


if __name__ == "__main__":
    main()
