"""
Full-picture "spec" visualizations of one example scenario, showing the PHYSICAL
intersection space, every controlled agent (real position + heading arrow), and the
MASTER HIERARCHY overlaid (local-master groups + the global master on top).

Two views:
  * parallel : every intersection drawn side-by-side in one figure; up to 2 local
               masters share a dense crossing; a global master sits on top of all LMs.
  * chain    : one long connected corridor; each intersection is one regional local
               master's zone (shaded band); the global master sits on top.

Usage:
  py -3 visualize_hierarchy.py                       # both, default sizes
  py -3 visualize_hierarchy.py --m 8 --k 3 --n-int 5
  py -3 visualize_hierarchy.py --only chain --n-int 8
"""
from __future__ import annotations

import argparse
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
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import run_scalability_suite as rss  # noqa: E402
import run_chain_scalability as rcs  # noqa: E402

# Distinct colours for local-master groups.
_LM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
              "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f"] * 4
_GM_COLOR = "#111111"


def _draw_network(ax, net, ox=0.0, oy=0.0, color="#d2d2d2", lw=2.2):
    """Sample and draw every lane polyline of a highway-env road network."""
    for a in net.graph:
        for b in net.graph[a]:
            for lane in net.graph[a][b]:
                try:
                    length = float(lane.length)
                    ss = np.linspace(0.0, length, max(2, int(length // 4)))
                    pts = np.array([lane.position(s, 0) for s in ss], dtype=float)
                    ax.plot(pts[:, 0] + ox, pts[:, 1] + oy, color=color, lw=lw,
                            solid_capstyle="round", zorder=0)
                except Exception:
                    pass


def _agent_heading(net, lane_key, off, base_long=40.0):
    """Real spawn (x,y) and unit heading for one agent."""
    lane = net.get_lane(tuple(lane_key))
    p0 = np.array(lane.position(base_long + off, 0), dtype=float)
    p1 = np.array(lane.position(base_long + off + 6.0, 0), dtype=float)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    return p0, (d / n if n > 1e-6 else np.array([1.0, 0.0]))


def _group_box(ax, pts, color, label, pad=14.0):
    """Draw a rounded bounding box around a master's agents + a label."""
    pts = np.asarray(pts, dtype=float)
    x0, y0 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    w, h = np.ptp(pts[:, 0]) + 2 * pad, np.ptp(pts[:, 1]) + 2 * pad
    box = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=2,rounding_size=6",
                         fill=False, ec=color, lw=2.0, ls="-", zorder=4)
    ax.add_patch(box)
    ax.text(x0 + w / 2, y0 + h + 4, label, ha="center", va="bottom", fontsize=9,
            color=color, fontweight="bold", zorder=5)
    return (x0 + w / 2, y0 + h + 4)  # top-centre anchor for tree lines


# ──────────────────────────────────────────────────────────────────────────────
def plot_parallel_spec(m_local_masters: int, k_agents: int, out_png: str, seed: int = 7):
    """All intersections side-by-side, agents + LM groups + a global master on top."""
    cell_cars = rss.plan_cells(m_local_masters, k_agents, lm_per_cell=2)
    rng = np.random.default_rng(seed)
    target_speeds = list(rss.rps.BASE_CFG["target_speeds"])

    n_cells = len(cell_cars)
    cell_gap = 260.0  # horizontal spacing between intersections in the figure
    fig, ax = plt.subplots(figsize=(max(9.0, 4.6 * n_cells), 7.2))

    lm_counter = 0
    lm_anchors = []
    total_agents = 0
    for ci, n_cars in enumerate(cell_cars):
        cell = rss.IntersectionCell(n_cars, k_agents, target_speeds)
        scenario = rss._make_intersection_scenario(n_cars, rng)
        cell.reset(scenario)
        net = cell._inner().road.network
        ox = ci * cell_gap
        _draw_network(ax, net, ox=ox)

        # agents (real positions), coloured by their local-master group
        for g, grp in enumerate(cell.groups):
            color = _LM_COLORS[lm_counter % len(_LM_COLORS)]
            pts = []
            for a_idx in grp:
                lane_key, dest, off = scenario["agents"][a_idx]
                p0, d = _agent_heading(net, lane_key, off)
                p0 = p0 + np.array([ox, 0.0])
                pts.append(p0)
                ax.scatter([p0[0]], [p0[1]], color=color, s=46, zorder=6,
                           edgecolors="k", linewidths=0.5)
                ax.annotate("", xy=(p0[0] + d[0] * 16, p0[1] + d[1] * 16), xytext=(p0[0], p0[1]),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=5)
                total_agents += 1
            anchor = _group_box(ax, pts, color, f"LM{lm_counter + 1}")
            lm_anchors.append(anchor)
            lm_counter += 1
        ax.text(ox, -118, f"Intersection {ci}", ha="center", fontsize=9, color="#555")
        cell.close()

    # Global master node on top, connected to every local master.
    gm_x = (n_cells - 1) * cell_gap / 2.0
    gm_y = max(a[1] for a in lm_anchors) + 70
    for (lx, ly) in lm_anchors:
        ax.plot([gm_x, lx], [gm_y, ly + 6], color=_GM_COLOR, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter([gm_x], [gm_y], s=320, marker="s", color=_GM_COLOR, zorder=7)
    ax.text(gm_x, gm_y, "GM", ha="center", va="center", color="white", fontsize=10,
            fontweight="bold", zorder=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title(f"Parallel topology — {m_local_masters} local masters + 1 global master, "
                 f"{total_agents} agents ({n_cells} intersections, {k_agents}/LM)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[parallel] {out_png}  ({m_local_masters} LM, {total_agents} agents, {n_cells} intersections)")


# ──────────────────────────────────────────────────────────────────────────────
def plot_chain_spec(n_int: int, out_png: str, seed: int = 7):
    """Long connected corridor; each intersection = one regional LM zone; GM on top."""
    rcs.register_chain_intersection_env()
    n_agents = n_int * rcs.AGENTS_PER_INTERSECTION
    rng = np.random.default_rng(seed)
    target_speeds = list(rcs.rps.BASE_CFG["target_speeds"])

    cell = rcs.ChainCell(n_int, n_agents, target_speeds)
    scenario = rcs.generate_chain_scenario(n_int, n_agents, rng)
    cell.reset(scenario)
    net = cell._inner().road.network
    spacing = cell._get_spacing()
    geom = rcs.resolve_geometry(net, scenario)

    fig, ax = plt.subplots(figsize=(max(10.0, 2.6 * n_int), 6.4))
    _draw_network(ax, net)

    # Regional master zone bands (one local master per intersection).
    ymax = max(g["start"][1] for g in geom) + 30
    ymin = min(g["start"][1] for g in geom) - 30
    lm_anchors = []
    for z in range(n_int):
        color = _LM_COLORS[z % len(_LM_COLORS)]
        cx = z * spacing
        ax.axvspan(cx - spacing / 2 + 6, cx + spacing / 2 - 6, ymin=0.05, ymax=0.95,
                   color=color, alpha=0.07, zorder=0)
        ax.text(cx, ymax + 16, f"LM{z + 1}\n(zone I{z})", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")
        lm_anchors.append((cx, ymax + 14))

    # Agents (real positions + heading), coloured by their current zone's master.
    for k, g in enumerate(geom):
        sx, sy = g["start"]; dx, dy = g["dir"]
        zone = max(0, min(n_int - 1, round(sx / spacing)))
        color = _LM_COLORS[zone % len(_LM_COLORS)]
        ax.scatter([sx], [sy], color=color, s=44, zorder=6, edgecolors="k", linewidths=0.5)
        ax.annotate("", xy=(sx + dx * 15, sy + dy * 15), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=5)

    # Global master on top, connected to each regional local master.
    gm_x = (n_int - 1) * spacing / 2.0
    gm_y = ymax + 70
    for (lx, ly) in lm_anchors:
        ax.plot([gm_x, lx], [gm_y, ly + 6], color=_GM_COLOR, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter([gm_x], [gm_y], s=340, marker="s", color=_GM_COLOR, zorder=7)
    ax.text(gm_x, gm_y, "GM", ha="center", va="center", color="white", fontsize=10,
            fontweight="bold", zorder=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title(f"Connected chain — {n_int} intersections, {n_int} regional local masters "
                 f"+ 1 global master, {n_agents} agents (3/LM). Cars hand off to the next "
                 f"zone's master as they cross.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    cell.close()
    print(f"[chain] {out_png}  ({n_int} LM, {n_agents} agents)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=6, help="parallel: number of local masters")
    p.add_argument("--k", type=int, default=3, help="parallel: agents per local master")
    p.add_argument("--n-int", type=int, default=5, help="chain: number of intersections")
    p.add_argument("--only", choices=["parallel", "chain", "both"], default="both")
    p.add_argument("--out-dir", default=os.path.join("MODELS_EVALUATION", "hierarchy_spec"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.only in ("parallel", "both"):
        plot_parallel_spec(args.m, args.k, os.path.join(args.out_dir, "spec_parallel.png"))
    if args.only in ("chain", "both"):
        plot_chain_spec(args.n_int, os.path.join(args.out_dir, "spec_chain.png"))


if __name__ == "__main__":
    main()
