"""
Full-picture "spec" visualizations of one example scenario.

Each figure has TWO clearly separated panels:

  * TOP  — the MASTER TREE: a clean node-link diagram showing exactly who controls
           whom. Agents (circles) are grouped under their Local Master (LM, square);
           LMs are combined PAIRWISE by the shared master weights into intermediate
           merge-masters, all the way up to a single Global Master (GM) on top. This
           is the real recursive aggregation the code performs.
  * BOTTOM — the PHYSICAL space: every intersection of the layout drawn together in
           one panel, with each agent at its real spawn (heading arrow) coloured by
           its Local Master, and a labelled box per LM group. Colours match the tree.

Two topologies:
  * parallel : independent intersections side-by-side (up to 2 LMs share a crossing).
  * chain    : one connected corridor; each intersection is one regional LM zone.

Usage:
  py -3 visualize_hierarchy.py                              # both, default sizes
  py -3 visualize_hierarchy.py --m 16 --k 3 --n-int 5
  py -3 visualize_hierarchy.py --only parallel --m 8 --k 3
  py -3 visualize_hierarchy.py --catalog --out-dir SOME_DIR # one image per scale
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch         # noqa: E402
from matplotlib.gridspec import GridSpec              # noqa: E402

import run_scalability_suite as rss                   # noqa: E402
import run_chain_scalability as rcs                   # noqa: E402

# Distinct colours for local-master groups (cycled if there are many).
_LM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
              "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f", "#393b79", "#637939",
              "#8c6d31", "#843c39", "#7b4173", "#3182bd"] * 4
_GM_COLOR = "#111111"
_MID_COLOR = "#5b3a8c"   # intermediate master (a level below GM, above LM)
_MERGE_COLOR = "#888888"  # edges only


# ══════════════════════════════════════════════════════════════════════════════
# Master-tree panel (shared by both topologies)
# ══════════════════════════════════════════════════════════════════════════════
def _draw_master_tree(ax, n_lms: int, agents_per_lm: list[int], lm_labels: list[str]):
    """Draw GM -> (pairwise merge-masters) -> LMs -> agents as a clean node-link tree.

    ``agents_per_lm`` gives the number of agent leaves under each LM (may differ if a
    cell is not full). All nodes are the SAME shared master network; intermediate grey
    squares are pairwise aggregations, the black square on top is the Global Master.
    """
    ax.axis("off")
    ax.set_xlim(-0.04, 1.04)

    xs = np.linspace(0.06, 0.94, n_lms) if n_lms > 1 else np.array([0.5])
    lm_y = 0.34
    agent_y = 0.06
    half_slot = (xs[1] - xs[0]) / 2.0 if n_lms > 1 else 0.4

    # ── agents under each LM ────────────────────────────────────────────────
    for i, x in enumerate(xs):
        color = _LM_COLORS[i % len(_LM_COLORS)]
        k = max(1, agents_per_lm[i])
        spread = min(0.7 * half_slot, 0.02 * max(1, k - 1) + 0.012)
        axs = np.linspace(x - spread, x + spread, k) if k > 1 else np.array([x])
        for ax_x in axs:
            ax.plot([x, ax_x], [lm_y, agent_y], color=color, lw=0.9, alpha=0.6, zorder=1)
            ax.scatter([ax_x], [agent_y], s=34, color=color, edgecolors="k",
                       linewidths=0.4, zorder=3)
        # LM node
        ax.scatter([x], [lm_y], marker="s", s=150, color=color, edgecolors="k",
                   linewidths=0.6, zorder=4)
        ax.text(x, lm_y - 0.055, lm_labels[i], ha="center", va="top", fontsize=7.5,
                color=color, fontweight="bold")

    # ── 5-way intermediate masters up to the GM (matches global_master_embedding) ─
    # Every group of up to 5 masters is combined by ONE master a level above; those
    # intermediate masters are themselves grouped in 5s, recursively, until a single
    # global master remains. Branching factor = the master's 5 input slots.
    #
    # Intermediate masters are drawn as clearly-labelled master nodes ("M1", "M2", ...)
    # in a distinct master colour — NOT anonymous dots — so it is obvious that a real
    # (shared-weights) master sits at every one of these nodes.
    num_slots = 5
    cur = list(xs)
    y = lm_y
    dy = 0.20
    mid_counter = [0]

    def _draw_edges(children_x, parent_x, y0, y1):
        for cx in children_x:
            ax.plot([cx, parent_x], [y0, y1], color=_MERGE_COLOR, lw=1.2, zorder=1)

    def _draw_mid(parent_x, y1):
        mid_counter[0] += 1
        label = f"M{mid_counter[0]}"
        ax.scatter([parent_x], [y1], marker="s", s=560, color=_MID_COLOR,
                   edgecolors="k", linewidths=1.0, zorder=5)
        ax.text(parent_x, y1, label, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=6)

    def _draw_gm(parent_x, y1):
        ax.scatter([parent_x], [y1], marker="s", s=620, color=_GM_COLOR,
                   edgecolors="k", linewidths=1.0, zorder=5)
        ax.text(parent_x, y1, "GM", ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=6)

    # Intermediate levels while more than 5 nodes remain.
    while len(cur) > num_slots:
        y_next = y + dy
        nxt = []
        for i in range(0, len(cur), num_slots):
            chunk = cur[i:i + num_slots]
            px = float(np.mean(chunk))
            _draw_edges(chunk, px, y, y_next)
            _draw_mid(px, y_next)
            nxt.append(px)
        cur = nxt
        y = y_next

    # Final level: the remaining <=5 nodes are managed directly by the global master.
    y_next = y + dy
    gx = float(np.mean(cur))
    _draw_edges(cur, gx, y, y_next)
    _draw_gm(gx, y_next)
    y = y_next

    ax.set_ylim(agent_y - 0.06, y + 0.10)
    # tiny legend
    ax.scatter([], [], marker="s", s=90, color=_GM_COLOR, label="Global master (GM)")
    ax.scatter([], [], marker="s", s=80, color=_MID_COLOR,
               label="Intermediate master (Mk, manages <=5)")
    ax.scatter([], [], marker="s", s=70, color=_LM_COLORS[0], label="Local master (LM)")
    ax.scatter([], [], s=34, color=_LM_COLORS[0], label="Agent (vehicle)")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), fontsize=7.5,
              frameon=False, handletextpad=0.4, labelspacing=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Physical-space helpers
# ══════════════════════════════════════════════════════════════════════════════
def _draw_network(ax, net, ox=0.0, oy=0.0, color="#d2d2d2", lw=2.2):
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
    lane = net.get_lane(tuple(lane_key))
    p0 = np.array(lane.position(base_long + off, 0), dtype=float)
    p1 = np.array(lane.position(base_long + off + 6.0, 0), dtype=float)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    return p0, (d / n if n > 1e-6 else np.array([1.0, 0.0]))


def _group_box(ax, pts, color, label, pad=14.0):
    pts = np.asarray(pts, dtype=float)
    x0, y0 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    w, h = np.ptp(pts[:, 0]) + 2 * pad, np.ptp(pts[:, 1]) + 2 * pad
    box = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=2,rounding_size=6",
                         fill=False, ec=color, lw=1.8, zorder=4)
    ax.add_patch(box)
    ax.text(x0 + w / 2, y0 + h + 5, label, ha="center", va="bottom", fontsize=8,
            color=color, fontweight="bold", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Parallel topology
# ══════════════════════════════════════════════════════════════════════════════
def plot_parallel_spec(m_local_masters: int, k_agents: int, out_png: str, seed: int = 7):
    """Two-panel spec: master tree on top, all intersections together below."""
    cell_cars = rss.plan_cells(m_local_masters, k_agents, lm_per_cell=2)
    rng = np.random.default_rng(seed)
    target_speeds = list(rss.rps.BASE_CFG["target_speeds"])
    n_cells = len(cell_cars)

    fig = plt.figure(figsize=(max(11.0, 2.7 * m_local_masters), 9.4))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.25], hspace=0.12)
    ax_tree = fig.add_subplot(gs[0])
    ax_phys = fig.add_subplot(gs[1])

    lm_counter = 0
    agents_per_lm: list[int] = []
    lm_labels: list[str] = []
    total_agents = 0
    cell_gap = 260.0
    for ci, n_cars in enumerate(cell_cars):
        cell = rss.IntersectionCell(n_cars, k_agents, target_speeds)
        scenario = rss._make_intersection_scenario(n_cars, rng)
        cell.reset(scenario)
        net = cell._inner().road.network
        ox = ci * cell_gap
        _draw_network(ax_phys, net, ox=ox)

        for grp in cell.groups:
            color = _LM_COLORS[lm_counter % len(_LM_COLORS)]
            pts = []
            for a_idx in grp:
                lane_key, dest, off = scenario["agents"][a_idx]
                p0, d = _agent_heading(net, lane_key, off)
                p0 = p0 + np.array([ox, 0.0])
                pts.append(p0)
                ax_phys.scatter([p0[0]], [p0[1]], color=color, s=46, zorder=6,
                                edgecolors="k", linewidths=0.5)
                ax_phys.annotate("", xy=(p0[0] + d[0] * 16, p0[1] + d[1] * 16),
                                 xytext=(p0[0], p0[1]),
                                 arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=5)
                total_agents += 1
            if pts:
                _group_box(ax_phys, pts, color, f"LM{lm_counter + 1}")
            agents_per_lm.append(len(grp))
            lm_labels.append(f"LM{lm_counter + 1}")
            lm_counter += 1
        ax_phys.text(ox, -120, f"Intersection {ci}", ha="center", fontsize=9, color="#555")
        cell.close()

    _draw_master_tree(ax_tree, lm_counter, agents_per_lm, lm_labels)
    ax_tree.set_title(
        f"Master hierarchy:  {total_agents} agents  ->  {lm_counter} local masters  "
        f"->  merge-masters (groups of <=5)  ->  1 global master", fontsize=12, pad=8)

    ax_phys.set_aspect("equal", adjustable="datalim")
    ax_phys.axis("off")
    ax_phys.set_title(
        f"Physical layout: {n_cells} independent intersections side-by-side "
        f"({k_agents} agents per local master, up to 2 LMs per crossing)", fontsize=11, pad=6)

    fig.suptitle(f"PARALLEL topology — {total_agents} agents, {m_local_masters} local masters, "
                 f"1 global master", fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"[parallel] {out_png}  ({m_local_masters} LM, {total_agents} agents, {n_cells} intersections)")


# ══════════════════════════════════════════════════════════════════════════════
# Chain topology
# ══════════════════════════════════════════════════════════════════════════════
def plot_chain_spec(n_int: int, out_png: str, seed: int = 7):
    """Two-panel spec: master tree on top, connected corridor below."""
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

    fig = plt.figure(figsize=(max(11.0, 2.9 * n_int), 8.8))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.15], hspace=0.1)
    ax_tree = fig.add_subplot(gs[0])
    ax_phys = fig.add_subplot(gs[1])

    # one regional LM per intersection, each owning 3 agents
    agents_per_lm = [rcs.AGENTS_PER_INTERSECTION] * n_int
    lm_labels = [f"LM{z + 1}\n(zone I{z})" for z in range(n_int)]
    _draw_master_tree(ax_tree, n_int, agents_per_lm, lm_labels)
    ax_tree.set_title(
        f"Master hierarchy:  {n_agents} agents  ->  {n_int} regional local masters  "
        f"->  merge-masters (groups of <=5)  ->  1 global master  "
        f"(cars hand off to the next zone's LM as they cross)", fontsize=11, pad=8)

    _draw_network(ax_phys, net)
    ymax = max(g["start"][1] for g in geom) + 30
    for z in range(n_int):
        color = _LM_COLORS[z % len(_LM_COLORS)]
        cx = z * spacing
        ax_phys.axvspan(cx - spacing / 2 + 6, cx + spacing / 2 - 6, ymin=0.05, ymax=0.9,
                        color=color, alpha=0.08, zorder=0)
        ax_phys.text(cx, ymax + 12, f"LM{z + 1}", ha="center", va="bottom",
                     fontsize=9, color=color, fontweight="bold")
    for g in geom:
        sx, sy = g["start"]; dx, dy = g["dir"]
        zone = max(0, min(n_int - 1, round(sx / spacing)))
        color = _LM_COLORS[zone % len(_LM_COLORS)]
        ax_phys.scatter([sx], [sy], color=color, s=44, zorder=6, edgecolors="k", linewidths=0.5)
        ax_phys.annotate("", xy=(sx + dx * 15, sy + dy * 15), xytext=(sx, sy),
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=5)

    ax_phys.set_aspect("equal", adjustable="datalim")
    ax_phys.axis("off")
    ax_phys.set_title(
        f"Physical layout: {n_int} connected intersections (one corridor); shaded bands "
        f"are the per-zone local-master regions", fontsize=11, pad=6)

    fig.suptitle(f"CHAIN topology — {n_agents} agents, {n_int} regional local masters, "
                 f"1 global master", fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    cell.close()
    print(f"[chain] {out_png}  ({n_int} LM, {n_agents} agents)")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=6, help="parallel: number of local masters")
    p.add_argument("--k", type=int, default=3, help="parallel: agents per local master")
    p.add_argument("--n-int", type=int, default=5, help="chain: number of intersections")
    p.add_argument("--only", choices=["parallel", "chain", "both"], default="both")
    p.add_argument("--catalog", action="store_true",
                   help="Generate one spec image per scale (parallel: 1..16 LM; chain: 2..5).")
    p.add_argument("--out-dir", default=os.path.join("MODELS_EVALUATION", "hierarchy_spec"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.catalog:
        # Parallel scales: M=1,2,4,6,8,12,16 local masters (K=3) -> 3..48 agents.
        for m in [1, 2, 4, 6, 8, 12, 16]:
            plot_parallel_spec(m, 3, os.path.join(args.out_dir, f"parallel_M{m:02d}_N{m * 3}.png"))
        # Chain scales: 2..5 intersections -> 6..15 agents.
        for n in [2, 3, 4, 5]:
            plot_chain_spec(n, os.path.join(args.out_dir, f"chain_int{n}_N{n * 3}.png"))
        print(f"\n[done] catalog -> {args.out_dir}")
        return

    if args.only in ("parallel", "both"):
        plot_parallel_spec(args.m, args.k, os.path.join(args.out_dir, "spec_parallel.png"))
    if args.only in ("chain", "both"):
        plot_chain_spec(args.n_int, os.path.join(args.out_dir, "spec_chain.png"))


if __name__ == "__main__":
    main()
