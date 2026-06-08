"""
Unified, clean scalability evaluation — PARALLEL + CHAIN in one organized folder.

Runs the canonical coordination study with ONLY the two conditions that matter
for the research claim:
    - normal       (full hierarchy: GM -> LMs -> agents)
    - zero_master  (no master signal; agents drive on raw state alone)

Everything is written under ONE timestamped root:

    EVALUATION_RESULTS/eval_<timestamp>/
        parallel/
            scalability_results.json
            scalability_summary.png
            scalability_master_benefit.png
            scalability_table.csv / .tex
            scenarios/<MxK_N>/scenarios.json + diagram PNGs
        chain/
            chain_results.json
            chain_scalability.png
            scale_<L>LM_<N>agents/scenarios.json + scenario_plots/*.png
        SUMMARY.md           (combined human-readable tables)
        master_summary.json  (combined machine-readable highlights)

Defaults use the BASELINE checkpoint (ckpt_agent6 / ckpt_master6), which is the
validated, well-calibrated model (normal ~87%, zero_master ~60% — a real,
consistent coordination gap). Override with --agent / --master if needed.

CLI:
    py -3 run_full_evaluation.py                # full canonical sweep
    py -3 run_full_evaluation.py --smoke        # quick end-to-end sanity check
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

import run_proto_action_sweep as rps          # noqa: E402
import run_scalability_suite as rss           # noqa: E402
import run_chain_scalability as rcs           # noqa: E402

# Only the two conditions that carry the research claim.
CONDITIONS = ("normal", "zero_master")

# Canonical parallel scaling family: K=3 agents per local master, growing the
# number of local masters/intersections — 3, 6, 12, 18, 24, 36, 48 agents.
PARALLEL_LAYOUTS = [(1, 3), (2, 3), (4, 3), (6, 3), (8, 3), (12, 3), (16, 3)]

# Connected chain: 2..5 intersections (3 agents each) -> 6, 9, 12, 15 agents.
# A single isolated intersection (n=1) is trivially solvable without coordination
# (no multi-hop conflict), so the connected-topology story starts at 2 nodes.
CHAIN_SIZES = [2, 3, 4, 5]


# ──────────────────────────────────────────────────────────────────────────────
def run_parallel(out_dir: str, agent_pth: str, master_pth: str,
                 n_scenarios: int, base_seed: int,
                 layouts: list[tuple[int, int]], save_scenarios: bool) -> list[dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    proto_exp, master_model, agent_model = rss.make_proto_and_models(agent_pth, master_pth)
    target_speeds = list(rps.BASE_CFG["target_speeds"])

    results: list[dict[str, Any]] = []
    for (m, k) in layouts:
        print(f"\n[parallel] layout M={m} x K={k}  (N={m * k})", flush=True)
        r = rss.run_layout(m, k, master_model, agent_model, proto_exp,
                           n_scenarios, base_seed, target_speeds, CONDITIONS)
        for cond in CONDITIONS:
            cc = r["conditions"][cond]
            print(f"    {cond:<12s} arrival={cc['arrival_pct_mean']:6.1f}%  "
                  f"crash/cell={cc['crash_rate_per_cell_pct']:6.1f}%", flush=True)
        results.append(r)

        payload = {
            "meta": {
                "topology": "parallel_independent_intersections",
                "agent_pth": agent_pth, "master_pth": master_pth,
                "conditions": list(CONDITIONS), "n_scenarios": n_scenarios,
                "base_seed": base_seed,
                "note": "Same shared weights at every scale; no retraining.",
            },
            "layouts": results,
        }
        with open(os.path.join(out_dir, "scalability_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    rss.make_plot(results, os.path.join(out_dir, "scalability_summary.png"))
    rss.make_master_benefit_plot(results, os.path.join(out_dir, "scalability_master_benefit.png"))
    rss.make_compute_plot(results, os.path.join(out_dir, "scalability_compute_cost.png"))
    rss.write_tables(results, out_dir)
    if save_scenarios:
        rss.dump_scenarios(out_dir, layouts, base_seed, target_speeds, n_scenarios=n_scenarios)
    return results


# ──────────────────────────────────────────────────────────────────────────────
def run_chain(out_dir: str, agent_pth: str, master_pth: str,
              n_scenarios: int, base_seed: int,
              sizes: list[int], save_plots: bool) -> list[dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    proto_exp, master_model, agent_model = rcs.make_proto_and_models(agent_pth, master_pth)
    target_speeds = list(rps.BASE_CFG["target_speeds"])

    results: list[dict[str, Any]] = []
    for n_int in sizes:
        n_agents = n_int * rcs.AGENTS_PER_INTERSECTION
        n_lms = n_int * rcs.LMS_PER_INTERSECTION
        scale_dir = os.path.join(out_dir, f"scale_{n_lms}LM_{n_agents}agents")
        print(f"\n[chain] {n_int} intersections  ({n_agents} agents, {n_lms} LM)", flush=True)
        r = rcs.run_sweep(n_int, master_model, agent_model, proto_exp,
                          n_scenarios, base_seed, target_speeds, CONDITIONS,
                          scale_dir=scale_dir, save_plots=save_plots)
        for cond in CONDITIONS:
            cc = r["conditions"][cond]
            print(f"    {cond:<12s} arrival={cc['arrival_pct_mean']:6.1f}%  "
                  f"crash={cc['crash_rate_pct']:5.1f}%", flush=True)
        results.append(r)

        payload = {
            "meta": {
                "topology": "chain_connected", "agent_pth": agent_pth, "master_pth": master_pth,
                "conditions": list(CONDITIONS), "n_scenarios": n_scenarios, "base_seed": base_seed,
                "agents_per_intersection": rcs.AGENTS_PER_INTERSECTION,
                "lms_per_intersection": rcs.LMS_PER_INTERSECTION,
                "note": "All intersections in ONE connected env; agents route multi-hop.",
            },
            "layouts": results,
        }
        with open(os.path.join(out_dir, "chain_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    rcs.make_chain_plot(results, os.path.join(out_dir, "chain_scalability.png"))
    return results


# ──────────────────────────────────────────────────────────────────────────────
def _benefit(cc_normal: float, cc_zero: float) -> float:
    return round(cc_normal - cc_zero, 1)


def _benefit_str(cc_normal: float, cc_zero: float) -> str:
    b = _benefit(cc_normal, cc_zero)
    return f"{b:+.1f}%"


def write_summary(root: str, parallel: list[dict[str, Any]], chain: list[dict[str, Any]],
                  agent_pth: str, master_pth: str, n_scenarios: int) -> None:
    lines: list[str] = []
    lines.append("# Scalability Evaluation — Hierarchical Multi-Agent Coordination\n")
    lines.append(f"- Checkpoint (agent): `{agent_pth}`")
    lines.append(f"- Checkpoint (master): `{master_pth}`")
    lines.append(f"- Scenarios per scale: **{n_scenarios}**")
    lines.append(f"- Conditions: **normal** (full hierarchy) vs **zero_master** (no master signal)\n")
    lines.append("A meaningful master benefit means `normal` clearly beats `zero_master`. "
                 "If `zero_master` is high, scenarios are too easy (no coordination needed).\n")

    # ── Parallel table ──────────────────────────────────────────────────────
    lines.append("## Parallel (independent intersections)\n")
    lines.append("| N agents | M local masters | normal arrival | zero_master arrival | benefit | normal crash/cell | zero crash/cell |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(parallel, key=lambda x: x["total_agents"]):
        n = r["conditions"]["normal"]
        z = r["conditions"]["zero_master"]
        lines.append(
            f"| {r['total_agents']} | {r['n_local_masters']} | "
            f"{n['arrival_pct_mean']:.1f}% | {z['arrival_pct_mean']:.1f}% | "
            f"{_benefit_str(n['arrival_pct_mean'], z['arrival_pct_mean'])} | "
            f"{n['crash_rate_per_cell_pct']:.1f}% | {z['crash_rate_per_cell_pct']:.1f}% |"
        )
    lines.append("")

    # ── Chain table ─────────────────────────────────────────────────────────
    lines.append("## Chain (connected intersections, multi-hop routing)\n")
    lines.append("| Intersections | N agents | normal arrival | zero_master arrival | benefit | normal crash | zero crash |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(chain, key=lambda x: x["n_agents"]):
        n = r["conditions"]["normal"]
        z = r["conditions"]["zero_master"]
        lines.append(
            f"| {r['n_intersections']} | {r['n_agents']} | "
            f"{n['arrival_pct_mean']:.1f}% | {z['arrival_pct_mean']:.1f}% | "
            f"{_benefit_str(n['arrival_pct_mean'], z['arrival_pct_mean'])} | "
            f"{n['crash_rate_pct']:.1f}% | {z['crash_rate_pct']:.1f}% |"
        )
    lines.append("")
    lines.append("## Folder layout\n")
    lines.append("- `hierarchy_overview/` — one full-picture spec per scale (top: the GM -> "
                 "merge-masters -> LMs -> agents tree; bottom: all intersections together).")
    lines.append("- `parallel/` — results JSON, summary/benefit/compute plots, CSV+LaTeX tables, "
                 "`scenarios/<MxK_N>/` with per-layout scenario JSON.")
    lines.append("- `chain/` — results JSON, scalability plot, `scale_<L>LM_<N>agents/` with "
                 "scenario JSON and a per-scenario layout plot for every scale (all agent levels).")
    lines.append("- `master_summary.json` — combined machine-readable highlights.\n")

    with open(os.path.join(root, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    master = {
        "agent_pth": agent_pth, "master_pth": master_pth, "n_scenarios": n_scenarios,
        "conditions": list(CONDITIONS),
        "parallel": [
            {"total_agents": r["total_agents"], "n_local_masters": r["n_local_masters"],
             "normal_arrival": r["conditions"]["normal"]["arrival_pct_mean"],
             "zero_master_arrival": r["conditions"]["zero_master"]["arrival_pct_mean"],
             "benefit": _benefit(r["conditions"]["normal"]["arrival_pct_mean"],
                                 r["conditions"]["zero_master"]["arrival_pct_mean"])}
            for r in sorted(parallel, key=lambda x: x["total_agents"])
        ],
        "chain": [
            {"n_intersections": r["n_intersections"], "n_agents": r["n_agents"],
             "normal_arrival": r["conditions"]["normal"]["arrival_pct_mean"],
             "zero_master_arrival": r["conditions"]["zero_master"]["arrival_pct_mean"],
             "benefit": _benefit(r["conditions"]["normal"]["arrival_pct_mean"],
                                 r["conditions"]["zero_master"]["arrival_pct_mean"])}
            for r in sorted(chain, key=lambda x: x["n_agents"])
        ],
    }
    with open(os.path.join(root, "master_summary.json"), "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="", help="Agent checkpoint .pth (default: baseline ckpt6)")
    p.add_argument("--master", default="", help="Master checkpoint .pth (default: baseline ckpt6)")
    p.add_argument("--n-scenarios", type=int, default=30)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--output-root", default="")
    p.add_argument("--only", choices=("both", "parallel", "chain"), default="both")
    p.add_argument("--reuse-parallel", default="",
                   help="Path to an existing parallel/ folder (or its scalability_results.json) "
                        "to copy + include in the summary instead of recomputing it.")
    p.add_argument("--no-scenarios", action="store_true", help="Skip scenario JSON/diagrams.")
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end check.")
    args = p.parse_args()

    agent_pth = os.path.abspath(args.agent.strip() or rss.CKPT_BASELINE[0])
    master_pth = os.path.abspath(args.master.strip() or rss.CKPT_BASELINE[1])

    parallel_layouts = list(PARALLEL_LAYOUTS)
    chain_sizes = list(CHAIN_SIZES)
    n_scenarios = args.n_scenarios
    if args.smoke:
        parallel_layouts = [(1, 3), (2, 3)]
        chain_sizes = [2, 3]
        n_scenarios = min(n_scenarios, 4)

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    root = os.path.abspath(args.output_root.strip()
                           or os.path.join(_REPO, "EVALUATION_RESULTS", f"eval_{ts}"))
    os.makedirs(root, exist_ok=True)

    print(f"[full-eval] root          -> {root}")
    print(f"[full-eval] agent         -> {agent_pth}")
    print(f"[full-eval] master        -> {master_pth}")
    print(f"[full-eval] n_scenarios   -> {n_scenarios}")
    print(f"[full-eval] conditions    -> {list(CONDITIONS)}")
    print(f"[full-eval] parallel      -> {parallel_layouts if args.only != 'chain' else 'skipped'}")
    print(f"[full-eval] chain sizes   -> {chain_sizes if args.only != 'parallel' else 'skipped'}")

    parallel_results: list[dict[str, Any]] = []
    chain_results: list[dict[str, Any]] = []

    if args.reuse_parallel.strip():
        src = args.reuse_parallel.strip()
        src_dir = src if os.path.isdir(src) else os.path.dirname(src)
        dst_dir = os.path.join(root, "parallel")
        if os.path.abspath(src_dir) != os.path.abspath(dst_dir):
            import shutil
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        with open(os.path.join(dst_dir, "scalability_results.json"), encoding="utf-8") as f:
            parallel_results = json.load(f)["layouts"]
        print(f"[full-eval] reused parallel results from {src_dir} ({len(parallel_results)} layouts)")
    elif args.only in ("both", "parallel"):
        parallel_results = run_parallel(
            os.path.join(root, "parallel"), agent_pth, master_pth,
            n_scenarios, args.base_seed, parallel_layouts, save_scenarios=not args.no_scenarios)

    if args.only in ("both", "chain"):
        chain_results = run_chain(
            os.path.join(root, "chain"), agent_pth, master_pth,
            n_scenarios, args.base_seed, chain_sizes, save_plots=not args.no_scenarios)

    if parallel_results or chain_results:
        write_summary(root, parallel_results, chain_results, agent_pth, master_pth, n_scenarios)

    print(f"\n[done] everything organized under -> {root}")
    print(f"[done] read {os.path.join(root, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()
