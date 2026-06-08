"""
Connected-topology scalability evaluation.

Unlike the parallel-cells suite, here ALL intersections live in ONE shared
highway-env world connected by bidirectional roads, and agents route across
multiple intersections to reach their destination.

This directly tests the hierarchy's ability to coordinate multi-hop routing
in a physically connected road network.

CLI:
  py -3 run_chain_scalability.py --smoke                          # quick sanity
  py -3 run_chain_scalability.py --n-intersections 2,3,4 --n-scenarios 30
  py -3 run_chain_scalability.py --n-intersections 2,3,4,6,8 --n-scenarios 30 --conditions full,zero
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any

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

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.3,
})

import gymnasium as gym  # noqa: E402
import run_proto_action_sweep as rps  # noqa: E402
from src import project_globals  # noqa: E402
from src.experiment.scenarios_config import create_full_environment_config  # noqa: E402
from src.model.model_handler import load_models_from_paths  # noqa: E402
from highwayenv.utils import register_chain_intersection_env  # noqa: E402

register_chain_intersection_env()

# ── Checkpoints ──────────────────────────────────────────────────────────────
CKPT_BASELINE = (
    os.path.join(_REPO, "models", "agent", "ckpt_agent6.pth"),
    os.path.join(_REPO, "models", "master", "ckpt_master6.pth"),
)

CONDITIONS = ("normal", "zero_master", "const_all_masters", "swap_local_masters", "zero_global_master")
CONDITION_ALIASES = {
    "full": "normal", "normal": "normal", "zero": "zero_master", "zero_master": "zero_master",
    "const": "const_all_masters", "const_all_masters": "const_all_masters",
    "swap": "swap_local_masters", "swap_local_masters": "swap_local_masters",
    "zero_global": "zero_global_master", "zero_global_master": "zero_global_master",
}

CONST_BROADCAST = 9999.0

# Agents per local master — matches the trained unit (3 cars per LM).
AGENTS_PER_LM = 3
# REGIONAL master layout: each intersection is one region owned by ONE local master
# that coordinates the 3 cars currently in its zone. As cars traverse the corridor
# they hand off to the next zone's master. Scale axis = number of intersections =
# number of local masters: 1, 2, 5, 15 -> 3, 6, 15, 45 agents.
AGENTS_PER_INTERSECTION = 3
LMS_PER_INTERSECTION = 1


# ──────────────────────────────────────────────────────────────────────────────
# Scenario generation for the chain
# ──────────────────────────────────────────────────────────────────────────────
def _approach_lanes(n_int: int) -> list[tuple[tuple, int, int]]:
    """All valid approach lanes across the chain. Returns (lane_key, intersection_idx, corner_idx).
    Excludes internal connections (east of non-last, west of non-first)."""
    approaches = []
    for i in range(n_int):
        for corner in range(4):
            if corner == 3 and i < n_int - 1:
                continue  # east is connector, not an entry
            if corner == 1 and i > 0:
                continue  # west is connector, not an entry
            lane_key = (f"I{i}_o{corner}", f"I{i}_ir{corner}", 0)
            approaches.append((lane_key, i, corner))
    return approaches


def _outer_exits(n_int: int) -> list[str]:
    """All valid outer exits where agents can finish (only exits with physical lanes)."""
    exits = []
    for i in range(n_int):
        exits.append(f"I{i}_o0")  # south — always exists
        exits.append(f"I{i}_o2")  # north — always exists
    exits.append("I0_o1")           # leftmost west
    exits.append(f"I{n_int - 1}_o3")  # rightmost east
    return exits


def generate_chain_scenario(
    n_int: int, n_agents: int, rng: np.random.Generator, *,
    jitter: float = 4.0, hop_window: int = 2, **_,
) -> dict:
    """Generate a varied perpendicular-crossing scenario for a connected chain.

    REGIONAL layout with genuine conflict + master hand-off. Each intersection has
    exactly 3 home cars (one local master's worth):
      * 2 local vertical cars (S->N, N->S) that finish at this node, and
      * 1 EASTBOUND THROUGH car that drives along the main road and passes straight
        through the next ~`hop_window` nodes before exiting at a north ramp (or the
        far-east exit). As it passes straight through each interior node it crosses
        that node's vertical stream head-on — the genuine perpendicular conflict the
        master must time — and it hands off to each zone's master in turn.

    This is the windowed version of a full-corridor traverse: it preserves the strong
    crossing conflict (so the master adds real value) but, unlike a traverse that
    spans all N nodes, it finishes within the step budget at any scale.

    Per intersection (3 cars):
      * Node 0:            S->N, N->S, eastbound through-car (W approach -> exit east)
      * Node N-1 (last):   S->N, N->S, westbound through-car (E approach -> exit west)
      * Interior node i:   S->N, N->S, eastbound through-car (S approach, turn east)
      * Single node:       S->N, N->S, W->E (4-way local crossing)

    Edge nodes inject a horizontal stream from a real W/E approach; interior nodes
    inject theirs from the vertical approach turning onto the main road. Either way the
    through-car drives straight through the next `hop_window` nodes, crossing each
    node's vertical stream head-on, then exits at a north ramp (or the far end).

    `jitter` adds a small random longitudinal offset (+/- a few metres) to every spawn
    so each generated scenario shows the network slightly different positions.
    """
    far_east = f"I{n_int - 1}_o3"
    far_west = "I0_o1"

    def _approach(i, corner):
        return (f"I{i}_o{corner}", f"I{i}_ir{corner}", 0)

    def _jit() -> float:
        return float(rng.uniform(-jitter, jitter))

    agents: list[tuple] = []
    for i in range(n_int):
        is_last = (i == n_int - 1)

        # Local vertical stream (always valid, finishes at this node)
        agents.append((_approach(i, 0), f"I{i}_o2", _jit()))        # S -> N
        agents.append((_approach(i, 2), f"I{i}_o0", _jit()))        # N -> S

        # Third car: a windowed horizontal through-car (the perpendicular crossing).
        # The exact spawn that the pretrained master coordinates best differs slightly
        # between a 2-node chain (no interior node) and longer chains:
        #
        #  n == 1 : a single 4-way W->E crossing.
        #  n == 2 : BOTH edges inject an un-staggered traverse (I0 east, I1 west) so the
        #           lone node pair hosts a full bidirectional crossing.
        #  n >= 3 : every node EXCEPT the last injects an EASTBOUND through-car staggered
        #           35 m back; the last node is the eastbound sink (2nd local vertical).
        #           A westbound stream is deliberately avoided here — head-on traffic on
        #           the connectors is what makes long chains uncoordinatable.
        if n_int == 1:
            agents.append((_approach(i, 1), "I0_o3", _jit()))       # W -> E (4-way)
        elif n_int == 2:
            if i == 0:
                agents.append((_approach(i, 1), far_east, _jit()))  # eastbound (W appr)
            else:
                agents.append((_approach(i, 3), far_west, _jit()))  # westbound (E appr)
        elif not is_last:
            j = min(i + hop_window, n_int - 1)                     # eastbound
            dest = far_east if j == n_int - 1 else f"I{j}_o2"
            origin = _approach(i, 1) if i == 0 else _approach(i, 0)
            agents.append((origin, dest, _jit() - 35.0))
        else:
            agents.append((_approach(i, 2), f"I{i}_o0", _jit() - 35.0))  # 2nd N->S sink

    return {"agents": agents, "static": []}


# ──────────────────────────────────────────────────────────────────────────────
# Master / agent helpers (reuse from the main scalability suite)
# ──────────────────────────────────────────────────────────────────────────────
def _pad_vec(vec, size):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[:min(size, len(arr))] = arr[:min(size, len(arr))]
    return out


def _slot(vec, identifier, slot_vec_dim):
    return np.concatenate([_pad_vec(vec, slot_vec_dim), np.asarray([identifier], dtype=np.float32)])


def build_local_master_input(global_emb, group_states, proto_exp, *, intersection_center_x: float = 0.0):
    """Build LM input with positions normalized relative to local intersection center.
    This ensures the master always sees positions in its trained range (~[-100,100])."""
    num_slots = 5
    svd = proto_exp.slot_vec_dim
    slots = [_slot(global_emb, 1.0, svd)]
    for st in np.asarray(group_states, dtype=np.float32):
        local_st = st[:4].copy()
        local_st[0] -= intersection_center_x  # shift x relative to local intersection
        slots.append(_slot(local_st, 0.0, svd))
    while len(slots) < num_slots:
        slots.append(np.zeros(svd + 1, dtype=np.float32))
    return np.concatenate(slots[:num_slots]).astype(np.float32)


def master_embeddings_batch(master_model, inputs, proto_exp, deterministic):
    """One batched forward through the shared master — IDENTICAL to the proven
    parallel scalability suite (run_scalability_suite.master_embeddings_batch)."""
    import torch
    if not inputs:
        return []
    obs_t = torch.as_tensor(np.asarray(inputs, dtype=np.float32))
    with torch.no_grad():
        actions, _, _ = master_model.model.policy.forward(obs_t, deterministic=deterministic)
    arr = actions.detach().cpu().numpy().astype(np.float32)
    ed = proto_exp.embedding_dim
    return [arr[i].reshape(-1)[:ed] for i in range(arr.shape[0])]


def _pack_gm_input(embs, proto_exp):
    num_slots = 5
    svd = proto_exp.slot_vec_dim
    slots = [_slot(e, 1.0, svd) for e in embs[:num_slots]]
    while len(slots) < num_slots:
        slots.append(np.zeros(svd + 1, dtype=np.float32))
    return np.concatenate(slots[:num_slots]).astype(np.float32)


def global_master_embedding(master_model, lm_embs, proto_exp, deterministic):
    """Pairwise recursive aggregation: always present exactly 2 LM embeddings to GM.
    This matches training where GM always saw [LM1, LM2, pad, pad, pad]."""
    cur = [np.asarray(e, dtype=np.float32).reshape(-1) for e in lm_embs]
    if not cur:
        return np.zeros(proto_exp.embedding_dim, dtype=np.float32)
    if len(cur) == 1:
        return cur[0]
    # Pairwise reduction: combine pairs until one embedding remains. All pairs at a
    # given tree level are packed and pushed through the master in ONE batched forward
    # (instead of one call per pair), which is the dominant speed-up at large scale.
    while len(cur) > 1:
        pair_inputs = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                pair_inputs.append(_pack_gm_input([cur[i], cur[i + 1]], proto_exp))
            else:
                pair_inputs.append(_pack_gm_input([cur[i]], proto_exp))
        cur = master_embeddings_batch(master_model, pair_inputs, proto_exp, deterministic)
    return cur[0]


# ──────────────────────────────────────────────────────────────────────────────
# Coordination conditions
# ──────────────────────────────────────────────────────────────────────────────
def _apply_lm_condition(condition, lm_embs):
    if condition == "zero_master":
        return [np.zeros_like(e) for e in lm_embs]
    if condition == "const_all_masters":
        return [np.full_like(e, CONST_BROADCAST) for e in lm_embs]
    if condition == "swap_local_masters":
        if len(lm_embs) <= 1:
            return list(lm_embs)
        return lm_embs[-1:] + lm_embs[:-1]
    return list(lm_embs)


def _apply_gm_condition(condition, gm_emb):
    if condition in ("zero_master", "zero_global_master"):
        return np.zeros_like(gm_emb)
    if condition == "const_all_masters":
        return np.full_like(gm_emb, CONST_BROADCAST)
    return gm_emb


# ──────────────────────────────────────────────────────────────────────────────
# ChainCell — wraps the chain env and exposes per-intersection agent grouping
# ──────────────────────────────────────────────────────────────────────────────
class ChainCell:
    """One connected chain of N intersections with K agents, each assigned to a
    local master based on their CURRENT intersection zone."""

    def __init__(self, n_int: int, n_agents: int, target_speeds: list[int]):
        self.n_int = n_int
        self.n_agents = n_agents
        self.n_lms = n_int * LMS_PER_INTERSECTION

        controlled = {}
        for i in range(n_agents):
            controlled[f"car{i + 1}"] = {
                "start_lane": ("I0_o0", "I0_ir0", 0),
                "destination": "I0_o2",
                "speed": 5,
                "init_location": {"longitudinal": 40, "lateral": 0},
                "color": [0, 204, 0],
            }
        cfg = {
            "controlled_cars": controlled,
            "static_cars": {},
            "collision_reward": -50,
            "arrived_reward": 50,
            "starvation_reward": 0,
            "high_speed_reward": 5,
            "initial_vehicle_count": 0,
            "n_intersections": n_int,
            "connector_length": 80,
            "chain_scenarios": [],
            "chain_scenarios_only": True,
        }
        full_cfg = create_full_environment_config(cfg)
        full_cfg["action"]["target_speeds"] = list(target_speeds)
        full_cfg["n_intersections"] = n_int
        full_cfg["connector_length"] = 80
        full_cfg["chain_scenarios"] = []
        full_cfg["chain_scenarios_only"] = True
        full_cfg["initial_vehicle_count"] = 0
        full_cfg["duration"] = 120  # short single-connector hops finish well within this
        full_cfg["policy_frequency"] = 1  # ensure 1 step = 1 second

        self.env = gym.make("RELchain-intersection-v0", render_mode=None, config=full_cfg)
        self.done = False
        self.crashed = False

    def _inner(self):
        env = self.env
        while hasattr(env, "env") and not hasattr(env, "controlled_vehicles"):
            env = env.env
        return env

    def read_state(self) -> np.ndarray:
        inner = self._inner()
        out = []
        for v in inner.controlled_vehicles:
            if getattr(v, "is_arrived", False):
                out.append([0.0, 0.0, 0.0, 0.0])
                continue
            vel = v.velocity if hasattr(v, "velocity") else np.zeros(2)
            out.append([float(v.position[0]), float(v.position[1]), float(vel[0]), float(vel[1])])
        return np.asarray(out, dtype=np.float32)

    def agent_intersection_zone(self) -> list[int]:
        """Determine which intersection zone each agent is currently in (by x-position).
        This implements the LM hand-off: LM assignment follows the agent's physical location."""
        inner = self._inner()
        spacing = self._get_spacing()
        zones = []
        for v in inner.controlled_vehicles:
            x = float(v.position[0])
            zone = max(0, min(self.n_int - 1, round(x / spacing)))
            zones.append(zone)
        return zones

    def _get_spacing(self) -> float:
        from highway_env.road.lane import AbstractLane
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        outer_distance = right_turn_radius + lane_width / 2
        connector_length = 80
        return 2 * outer_distance + connector_length

    def lm_groups(self) -> list[list[int]]:
        """Dynamic LM hand-off: agents are coordinated by the local masters of the
        intersection they are CURRENTLY in. Each intersection owns LMS_PER_INTERSECTION
        local masters; cars in a zone are split round-robin across that zone's LMs so
        each LM sees <= AGENTS_PER_LM cars (the trained group size)."""
        zones = self.agent_intersection_zone()
        zone_agents: dict[int, list[int]] = {}
        for agent_idx, z in enumerate(zones):
            zone_agents.setdefault(z, []).append(agent_idx)

        groups: list[list[int]] = [[] for _ in range(self.n_lms)]
        for z, ags in zone_agents.items():
            base_lm = z * LMS_PER_INTERSECTION
            for j, a in enumerate(ags):
                lm_idx = base_lm + (j % LMS_PER_INTERSECTION)
                if lm_idx < self.n_lms:
                    groups[lm_idx].append(a)
        return groups

    def agent_zone_for(self, agent_idx: int) -> int:
        """Current intersection zone of one agent (by x-position)."""
        return self.agent_intersection_zone()[agent_idx]

    def reset(self, scenario: dict) -> np.ndarray:
        inner = self._inner()
        inner.config["chain_scenarios"] = [scenario]
        inner.config["chain_scenarios_only"] = True
        project_globals.after_is_arrived_flags = [False] * self.n_agents
        self.env.reset()
        self.done = False
        self.crashed = False
        return self.read_state()

    def step(self, actions: list[int]) -> np.ndarray:
        if self.done:
            return self.read_state()
        project_globals.after_is_arrived_flags = project_globals.after_is_arrived_flags[:self.n_agents]
        _, _, done, truncated, _ = self.env.step(tuple(int(a) for a in actions))
        inner = self._inner()
        if any(getattr(v, "crashed", False) for v in inner.controlled_vehicles):
            self.crashed = True
        if done or truncated:
            self.done = True
        return self.read_state()

    def n_arrived(self) -> int:
        return sum(1 for v in self._inner().controlled_vehicles if getattr(v, "is_arrived", False))

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────
def run_chain_episode(
    cell: ChainCell,
    scenario: dict,
    master_model,
    agent_model,
    proto_exp,
    condition: str,
    *,
    max_steps: int = 120,
) -> dict[str, Any]:
    state = cell.reset(scenario)
    emb_dim = proto_exp.embedding_dim
    gm_prev = np.zeros(emb_dim, dtype=np.float32)
    steps = 0

    spacing = cell._get_spacing()

    while steps < max_steps and not cell.done:
        steps += 1
        groups = cell.lm_groups()
        zones = cell.agent_intersection_zone()  # current zone per agent

        # Local master embeddings (one per LM). Each LM belongs to one intersection
        # zone; positions are shifted into that zone's local frame so the master
        # always sees coordinates in its trained range (~[-100, 100]).
        lm_inputs = []
        for lm_idx, grp in enumerate(groups):
            zone = lm_idx // LMS_PER_INTERSECTION
            center_x = zone * spacing
            if grp:
                grp_states = state[grp].copy()
                grp_states[:, 0] -= center_x
            else:
                grp_states = np.zeros((0, 4), dtype=np.float32)
            lm_inputs.append(build_local_master_input(gm_prev, grp_states, proto_exp))
        lm_embs = master_embeddings_batch(master_model, lm_inputs, proto_exp, False)

        # Global master
        gm_emb = global_master_embedding(master_model, lm_embs, proto_exp, False)
        gm_feedback = _apply_gm_condition(condition, gm_emb)
        lm_used = _apply_lm_condition(condition, lm_embs)

        # Agent observations + actions (each car in its CURRENT zone's local frame)
        obs_list = []
        agent_order = []
        for lm_idx, grp in enumerate(groups):
            emb = lm_used[lm_idx]
            for a_idx in grp:
                center_x = zones[a_idx] * spacing
                local_state = state[a_idx][:4].copy()
                local_state[0] -= center_x
                obs_list.append(np.concatenate([local_state, emb]).astype(np.float32))
                agent_order.append(a_idx)
        if obs_list:
            actions_raw, _, _ = rps.agent_actions(agent_model, obs_list, deterministic=True)
        else:
            actions_raw = []

        # Reassemble actions in original agent order
        full_actions = [0] * cell.n_agents
        for a_idx, act in zip(agent_order, actions_raw):
            full_actions[a_idx] = int(act)

        state = cell.step(full_actions)
        gm_prev = gm_feedback

    return {
        "arrival_pct": 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        "crashed": int(cell.crashed),
        "steps": steps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────
def make_proto_and_models(agent_pth: str, master_pth: str):
    cfg = dict(rps.BASE_CFG)
    cfg["label"] = "chain_scalability"
    cfg["embedding_dim"] = 4
    cfg["load_pretrained"] = False
    rps.set_all_seeds(rps.SEED)
    work = os.path.join(_REPO, "MODELS_EVALUATION", "_chain_work")
    os.makedirs(work, exist_ok=True)
    proto_exp = rps.ProtoExperiment(cfg, work)
    master_model, agent_model = rps.make_models(proto_exp)
    if not load_models_from_paths(agent_model, master_model, agent_pth, master_pth):
        raise RuntimeError(f"Failed to load:\n  {agent_pth}\n  {master_pth}")
    return proto_exp, master_model, agent_model


def resolve_geometry(net, scenario: dict, base_long: float = 40.0) -> list[dict]:
    """Resolve each agent's spawn (x,y), heading direction, and destination from the
    chain road network, so scenarios can be saved/plotted without re-simulating."""
    geom = []
    for (lane_key, destination, off) in scenario["agents"]:
        lk = tuple(lane_key)
        try:
            lane = net.get_lane(lk)
            p0 = np.asarray(lane.position(base_long + off, 0), dtype=float)
            p1 = np.asarray(lane.position(base_long + off + 6.0, 0), dtype=float)
            d = p1 - p0
            nrm = float(np.linalg.norm(d))
            d = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
        except Exception:
            p0 = np.array([0.0, 0.0]); d = np.array([1.0, 0.0])
        geom.append({
            "start": [round(float(p0[0]), 2), round(float(p0[1]), 2)],
            "dir": [round(float(d[0]), 3), round(float(d[1]), 3)],
            "lane_key": [str(lane_key[0]), str(lane_key[1]), int(lane_key[2])],
            "destination": str(destination),
            "offset": round(float(off), 2),
        })
    return geom


def plot_scenario(geom: list[dict], n_int: int, spacing: float, out_png: str, title: str) -> None:
    """Top-down layout: each car as a coloured dot + arrow (heading) at its spawn."""
    fig, ax = plt.subplots(figsize=(max(6.0, n_int * 2.0), 4.2))
    for z in range(n_int):
        cx = z * spacing
        ax.add_patch(plt.Rectangle((cx - 9, -9), 18, 18, fill=False, ec="#bbbbbb", lw=1.0, ls="--", zorder=0))
        ax.text(cx, 13, f"I{z}", ha="center", va="bottom", fontsize=8, color="#666")
    if n_int > 1:
        ax.plot([0, (n_int - 1) * spacing], [0, 0], color="#e0e0e0", lw=1.5, zorder=0)
    cmap = plt.cm.tab20
    for k, g in enumerate(geom):
        sx, sy = g["start"]; dx, dy = g["dir"]
        c = cmap(k % 20)
        ax.scatter([sx], [sy], color=c, s=28, zorder=3, edgecolors="k", linewidths=0.4)
        ax.annotate("", xy=(sx + dx * 13, sy + dy * 13), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4), zorder=2)
        ax.text(sx, sy - 4, str(k), fontsize=6, ha="center", va="top", color=c)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def run_sweep(
    n_int: int,
    master_model,
    agent_model,
    proto_exp,
    n_scenarios: int,
    base_seed: int,
    target_speeds: list[int],
    conditions: tuple[str, ...],
    scale_dir: str | None = None,
    save_plots: bool = True,
) -> dict[str, Any]:
    n_agents = n_int * AGENTS_PER_INTERSECTION
    rng = np.random.default_rng(base_seed)
    scenarios = [generate_chain_scenario(n_int, n_agents, rng) for _ in range(n_scenarios)]

    cell = ChainCell(n_int, n_agents, target_speeds)
    spacing = cell._get_spacing()
    out: dict[str, Any] = {
        "n_intersections": n_int,
        "n_agents": n_agents,
        "n_lms": n_int * LMS_PER_INTERSECTION,
        "topology": "chain",
        "n_scenarios": n_scenarios,
        "conditions": {},
    }

    # ── Save every scenario (geometry) + a per-scenario layout plot ──────────────
    geoms: list[list[dict]] = []
    if scale_dir is not None:
        os.makedirs(scale_dir, exist_ok=True)
        cell.reset(scenarios[0])               # build the road network once
        net = cell._inner().road.network
        plots_dir = os.path.join(scale_dir, "scenario_plots")
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
        for ep in range(n_scenarios):
            g = resolve_geometry(net, scenarios[ep])
            geoms.append(g)
            if save_plots:
                plot_scenario(g, n_int, spacing,
                              os.path.join(plots_dir, f"scenario_{ep:03d}.png"),
                              f"{n_int} intersections / {n_agents} agents — scenario {ep}")
        with open(os.path.join(scale_dir, "scenarios.json"), "w", encoding="utf-8") as f:
            json.dump({"n_intersections": n_int, "n_agents": n_agents, "spacing": round(spacing, 2),
                       "n_scenarios": n_scenarios, "scenarios": [
                           {"index": i, "agents": geoms[i]} for i in range(n_scenarios)]},
                      f, indent=2)

    # ── Run every condition, recording per-scenario outcomes ────────────────────
    per_scenario: dict[str, list[dict]] = {}
    t0 = time.time()
    try:
        for cond in conditions:
            arrivals, crashes, step_counts, rows = [], [], [], []
            for ep in range(n_scenarios):
                rps.set_all_seeds(base_seed + ep)
                res = run_chain_episode(cell, scenarios[ep], master_model, agent_model, proto_exp, cond)
                arrivals.append(res["arrival_pct"])
                crashes.append(res["crashed"])
                step_counts.append(res["steps"])
                rows.append({"scenario": ep, "arrival_pct": round(res["arrival_pct"], 2),
                             "crashed": int(res["crashed"]), "steps": int(res["steps"])})
            per_scenario[cond] = rows
            arr = np.asarray(arrivals, dtype=float)
            out["conditions"][cond] = {
                "arrival_pct_mean": float(np.mean(arr)),
                "arrival_pct_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "arrival_pct_sem": float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
                "crash_rate_pct": 100.0 * float(np.mean(crashes)),
                "mean_steps": float(np.mean(step_counts)),
                "arrival_pct_samples": [round(float(x), 2) for x in arrivals],
            }
    finally:
        cell.close()
    out["wall_time_sec"] = round(time.time() - t0, 2)

    if scale_dir is not None:
        with open(os.path.join(scale_dir, "per_scenario_results.json"), "w", encoding="utf-8") as f:
            json.dump({"n_intersections": n_int, "n_agents": n_agents,
                       "conditions": {c: per_scenario[c] for c in conditions}}, f, indent=2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def make_chain_plot(results: list[dict], out_png: str) -> None:
    """Connected-chain scaling curves. Saved as TWO separate images:
    ``*_arrival.png`` and ``*_crash.png`` (one chart per file)."""
    res = sorted(results, key=lambda r: r["n_agents"])
    xs = [r["n_agents"] for r in res]
    colors = {"normal": "#2E7D32", "zero_master": "#C62828", "const_all_masters": "#6A1B9A",
              "swap_local_masters": "#1565C0", "zero_global_master": "#EF6C00"}
    conds = [c for c in CONDITIONS if c in res[0]["conditions"]]
    xlabel = "Total agents (3 per local master, connected chain)"
    base, ext = os.path.splitext(out_png)
    ext = ext or ".png"

    # ── arrival ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cond in conds:
        arr = [r["conditions"][cond]["arrival_pct_mean"] for r in res]
        sem = [r["conditions"][cond].get("arrival_pct_sem", 0.0) for r in res]
        ax.errorbar(xs, arr, yerr=sem, marker="o", capsize=3, lw=2, label=cond, color=colors.get(cond))
    ax.set_ylabel("Mean arrival % (+/- SEM)")
    ax.set_title("Arrival % — connected chain topology (multi-hop routing)")
    ax.set_ylim(-5, 105)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{base}_arrival{ext}", dpi=300)
    plt.close(fig)

    # ── crash ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cond in conds:
        cr = [r["conditions"][cond]["crash_rate_pct"] for r in res]
        ax.plot(xs, cr, marker="o", lw=2, label=cond, color=colors.get(cond))
    ax.set_ylabel("Crash rate (%)")
    ax.set_title("Crash rate — connected chain topology")
    ax.set_ylim(-5, 105)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{base}_crash{ext}", dpi=300)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-intersections", default="2,3,4",
                   help="Chain sizes to test, e.g. '2,3,4,6'")
    p.add_argument("--n-scenarios", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--conditions", default="",
                   help="e.g. 'full,zero'. Default: all five.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="Skip per-scenario layout plots")
    p.add_argument("--output-root", default="")
    p.add_argument("--agent", default="", help="Agent checkpoint .pth (default: pretrained baseline)")
    p.add_argument("--master", default="", help="Master checkpoint .pth (default: pretrained baseline)")
    args = p.parse_args()

    if args.smoke:
        chain_sizes = [2, 3]
        args.n_scenarios = min(args.n_scenarios, 5)
    else:
        chain_sizes = [int(x.strip()) for x in args.n_intersections.split(",")]

    if args.conditions.strip():
        conditions = tuple(
            CONDITION_ALIASES[t.strip().lower()]
            for t in args.conditions.split(",") if t.strip().lower() in CONDITION_ALIASES
        )
    else:
        conditions = CONDITIONS

    agent_pth = args.agent.strip() or CKPT_BASELINE[0]
    master_pth = args.master.strip() or CKPT_BASELINE[1]
    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_root = args.output_root.strip() or os.path.join(_REPO, "MODELS_EVALUATION", f"chain_scalability_{ts}")
    os.makedirs(out_root, exist_ok=True)

    print(f"[chain] sizes={chain_sizes}  n_scenarios={args.n_scenarios}  conditions={list(conditions)}")
    print(f"[chain] output -> {out_root}")

    proto_exp, master_model, agent_model = make_proto_and_models(agent_pth, master_pth)
    target_speeds = list(rps.BASE_CFG["target_speeds"])

    results = []
    for n_int in chain_sizes:
        n_agents = n_int * AGENTS_PER_INTERSECTION
        n_lms = n_int * LMS_PER_INTERSECTION
        scale_dir = os.path.join(out_root, f"scale_{n_lms}LM_{n_agents}agents")
        print(f"\n=== Chain: {n_int} intersections, {n_agents} agents, {n_lms} LM ===", flush=True)
        r = run_sweep(n_int, master_model, agent_model, proto_exp,
                      args.n_scenarios, args.base_seed, target_speeds, conditions,
                      scale_dir=scale_dir, save_plots=not args.no_plots)
        for cond in conditions:
            cc = r["conditions"][cond]
            print(f"  {cond:<20s} arrival={cc['arrival_pct_mean']:6.1f}%  crash={cc['crash_rate_pct']:5.1f}%")
        results.append(r)

        payload = {"meta": {"topology": "chain_connected", "conditions": list(conditions),
                            "n_scenarios": args.n_scenarios, "base_seed": args.base_seed,
                            "timestamp": ts, "agents_per_intersection": AGENTS_PER_INTERSECTION,
                            "lms_per_intersection": LMS_PER_INTERSECTION,
                            "note": "All intersections in ONE env, agents route multi-hop."},
                   "layouts": results}
        with open(os.path.join(out_root, "chain_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    make_chain_plot(results, os.path.join(out_root, "chain_scalability.png"))
    print(f"\n[done] results -> {os.path.join(out_root, 'chain_results.json')}")
    print(f"[done] plot    -> {os.path.join(out_root, 'chain_scalability.png')}")


if __name__ == "__main__":
    main()
