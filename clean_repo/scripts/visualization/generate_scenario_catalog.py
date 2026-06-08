"""
Generate schematic scenario maps (style: diagram_intersection0.png) for every
agent count N in {3, 6, 12, 18, 24, 36, 48} — parallel and connected chain.

Output:
  <out_dir>/parallel/N3_agents/scenario_map.png  (+ scenarios.json)
  <out_dir>/chain/N12_agents/scenario_map.png

Usage:
  py -3 generate_scenario_catalog.py
  py -3 generate_scenario_catalog.py --out-dir PROFESSOR_RESULTS/04_scenario_maps
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

import run_scalability_suite as rss  # noqa: E402
import run_chain_scalability as rcs  # noqa: E402

AGENT_COUNTS = [3, 6, 12, 18, 24, 36, 48]
BASE_SEED = 42


def _draw_mini_crossing(ax, scenario: dict, title: str, *, offset_x: float = 0.0) -> None:
    """One 4-way schematic on axes ``ax`` (reuses rss geometry helpers)."""
    R = 55.0
    for idx in range(4):
        x, y = rss._approach_xy(idx, R)
        ax.plot([offset_x, offset_x + x], [0, y], color="#cccccc", lw=7,
                solid_capstyle="round", zorder=0)
        lx, ly = rss._approach_xy(idx, R * 1.15)
        ax.text(offset_x + lx, ly, f"o{idx}", ha="center", va="center", fontsize=7,
                color="#555", weight="bold")
    ax.add_patch(plt.Circle((offset_x, 0), 10, color="#eeeeee", zorder=1))
    recs = rss._scenario_to_record(scenario)
    for rec in recs:
        c = tuple(v / 255.0 for v in rec["color_rgb"])
        sx = offset_x + rec["schematic_start_xy"][0] * (55.0 / 60.0)
        sy = rec["schematic_start_xy"][1] * (55.0 / 60.0)
        ex = offset_x + rec["schematic_end_xy"][0] * (55.0 / 60.0)
        ey = rec["schematic_end_xy"][1] * (55.0 / 60.0)
        ax.add_patch(mpatches.FancyArrowPatch(
            (sx, sy), (ex, ey), connectionstyle="arc3,rad=0.25",
            arrowstyle="-|>", mutation_scale=12, lw=1.6, color=c, alpha=0.9, zorder=3))
        ax.scatter([sx], [sy], s=70, color=c, edgecolors="black", linewidths=0.5, zorder=4)
        ax.text(sx, sy, str(rec["agent_id"]), ha="center", va="center", fontsize=6,
                color="white", weight="bold", zorder=5)
    ax.set_title(title, fontsize=8)
    lim = R * 1.25
    ax.set_xlim(offset_x - lim, offset_x + lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def _layout_for_n(n: int) -> tuple[int, int]:
    """Return (M local masters, K agents per master) with M*K = N."""
    for k in (3, 4, 5):
        if n % k == 0:
            return n // k, k
    return 1, n


def generate_parallel_map(n: int, out_dir: str, rng: np.random.Generator) -> dict:
    m, k = _layout_for_n(n)
    cell_cars = rss.plan_cells(m, k, lm_per_cell=2)
    scenarios_per_cell = rss.gen_layout_scenarios(cell_cars, 1, BASE_SEED + n)[0]

    n_cells = len(cell_cars)
    fig_w = max(5.0, 3.8 * n_cells)
    fig, axes = plt.subplots(1, n_cells, figsize=(fig_w, 4.2))
    if n_cells == 1:
        axes = [axes]
    for ci, (ax, nc) in enumerate(zip(axes, cell_cars)):
        sc = rss._make_intersection_scenario(nc, rng) if ci >= len(scenarios_per_cell) else scenarios_per_cell[ci]
        _draw_mini_crossing(ax, sc, f"Int {ci} ({nc} cars)")
    fig.suptitle(
        f"Parallel — N={n} agents ({m} local masters × {k}/LM, {n_cells} intersections)\n"
        f"Example scenario: entry (coloured dot) → exit (arrow tip)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = os.path.join(out_dir, "scenario_map.png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    spec = {
        "topology": "parallel",
        "n_agents": n,
        "n_local_masters": m,
        "agents_per_lm": k,
        "cars_per_intersection": cell_cars,
        "agents": [],
    }
    for ci, sc in enumerate(scenarios_per_cell):
        for rec in rss._scenario_to_record(sc):
            rec["intersection_id"] = ci
            spec["agents"].append(rec)
    with open(os.path.join(out_dir, "scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return spec


def _chain_node_scenario(full_scenario: dict, node: int, agents_per_int: int = 3) -> dict:
    """Extract agents whose spawn lane belongs to intersection ``node``."""
    prefix = f"I{node}_"
    agents = [a for a in full_scenario["agents"] if str(a[0][0]).startswith(prefix)]
    if len(agents) >= agents_per_int:
        return {"agents": agents[:agents_per_int], "static": []}
    return {"agents": agents, "static": []}


def generate_chain_map(n: int, out_dir: str, rng: np.random.Generator) -> dict:
    n_int = max(1, n // rcs.AGENTS_PER_INTERSECTION)
    n_agents = n_int * rcs.AGENTS_PER_INTERSECTION
    full = rcs.generate_chain_scenario(n_int, n_agents, rng)

    fig_w = max(8.0, 2.2 * n_int)
    fig, axes = plt.subplots(1, n_int, figsize=(fig_w, 3.8))
    if n_int == 1:
        axes = [axes]
    for zi, ax in enumerate(axes):
        node_sc = _chain_node_scenario(full, zi)
        if not node_sc["agents"]:
            node_sc = {"agents": full["agents"][zi::n_int][:3], "static": []}
        _draw_mini_crossing(ax, node_sc, f"Zone I{zi} (LM{zi+1})")
        if zi < n_int - 1:
            ax.annotate("", xy=(1.05, 0.5), xytext=(0.95, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))
    fig.suptitle(
        f"Connected chain — N={n_agents} agents, {n_int} regional masters (3 cars/zone)\n"
        f"Cars hand off to the next zone's LM when crossing connectors →",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    png = os.path.join(out_dir, "scenario_map.png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    spec = {
        "topology": "chain",
        "n_agents": n_agents,
        "n_intersections": n_int,
        "regional_masters": n_int,
        "agents_per_zone": rcs.AGENTS_PER_INTERSECTION,
        "full_scenario_agents": [
            {"lane": a[0], "destination": a[1], "offset": a[2]} for a in full["agents"]
        ],
    }
    with open(os.path.join(out_dir, "scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join("PROFESSOR_RESULTS", "04_scenario_maps"))
    args = ap.parse_args()

    par_root = os.path.join(args.out_dir, "parallel")
    chain_root = os.path.join(args.out_dir, "chain")
    catalog = {"agent_counts": AGENT_COUNTS, "parallel": {}, "chain": {}}

    rng = np.random.default_rng(BASE_SEED)
    for n in AGENT_COUNTS:
        pdir = os.path.join(par_root, f"N{n}_agents")
        os.makedirs(pdir, exist_ok=True)
        catalog["parallel"][str(n)] = generate_parallel_map(n, pdir, rng)
        print(f"[parallel] N={n} -> {pdir}/scenario_map.png")

        cdir = os.path.join(chain_root, f"N{n}_agents")
        os.makedirs(cdir, exist_ok=True)
        catalog["chain"][str(n)] = generate_chain_map(n, cdir, rng)
        print(f"[chain]    N={n} -> {cdir}/scenario_map.png")

    with open(os.path.join(args.out_dir, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"\nCatalog index: {args.out_dir}/catalog_index.json")


if __name__ == "__main__":
    main()
