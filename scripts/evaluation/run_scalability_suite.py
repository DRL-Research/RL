"""
NEXT PHASE — Scalability suite for the 3-level hierarchy.

Goal (research claim):
  The hierarchy can scale to more agents and more masters and still coordinate
  the environment very well and prevent crashes.

Why this works WITHOUT retraining:
  Both the MasterModel (shared PPO + ResNet, fixed 25-D input → 4-D embedding)
  and the agent PPO (shared, obs = local state + LM embedding) use SHARED,
  role-agnostic weights with FIXED-dimension input packing. Roles (GM / LM /
  agent) differ only by how inputs are packed, never by parameters. Therefore
  the exact same best validated checkpoint (id6) can be deployed at any layout.

Layout abstraction (faithful generalisation of the double-intersection design):
  * Each LOCAL MASTER governs one independent intersection "cell" with K agents.
  * Adding masters  == adding intersection cells (M cells).
  * Adding agents   == denser cell (K agents per cell, capped by master slots).
  * A single GLOBAL MASTER aggregates the M local-master embeddings every step
    (recursively in chunks of <= NUM_MASTER_SLOTS so M can exceed the slot count).
  * The double-intersection baseline is exactly M=2, K=3.

Master input slot budget (must match the pretrained packing, 25-D = 5 slots):
  * Local master  : slot0 = global-feedback embedding (id=1) + up to 4 agent
                    slots (id=0)  -> capacity 4 agents per LM.
  * Global master : up to 5 local-master embedding slots (id=1).

Coordination counterfactuals (same scenarios, same weights, only signal path):
  normal | zero_master | const_all_masters | swap_local_masters | zero_global_master

Outputs (deliverables) under MODELS_EVALUATION/scalability_<ts>/ :
  * scalability_results.json        — arrival %, crash, compute cost per layout × condition
                                       (incl. per-episode samples + std/SEM)
  * scalability_summary.png         — scaling curves (arrival ±SEM & crash vs total agents)
  * scalability_master_benefit.png  — grouped bars: every layout × condition
  * scalability_compute_cost.png    — wall-time & env steps vs scale
  * scalability_table.csv / .tex    — paper tables (CSV + LaTeX booktabs)
  * scenarios/<MxK_N>/scenarios.json — every agent's entry/exit + real spawn (start/end)
  * scenarios/<MxK_N>/diagram_*.png  — crossing schematics with per-agent arrows

CLI:
  py -3 run_scalability_suite.py --smoke
  py -3 run_scalability_suite.py --high-scale --n-scenarios 30     # up to 50 agents
  py -3 run_scalability_suite.py --layouts "2x3,10x5" --n-scenarios 30
  py -3 run_scalability_suite.py --checkpoint finetuned --n-scenarios 30
"""

from __future__ import annotations

import argparse
import json
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

# Publication-quality defaults for the paper figures.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

import gymnasium as gym  # noqa: E402

# Importing run_proto_action_sweep registers the custom envs and gives us the
# proven ProtoExperiment / model factory / master+agent forward helpers.
import run_proto_action_sweep as rps  # noqa: E402
from src import project_globals  # noqa: E402
from src.experiment.scenarios_config import create_full_environment_config  # noqa: E402
from src.model.model_handler import load_models_from_paths  # noqa: E402
from highwayenv.intersection_class import rotate_scenario_clockwise  # noqa: E402
from src.experiment.scenarios import (  # noqa: E402
    base_complete_scenarios_6_cars,
    EXCLUDED_SCENARIO_INDICES,
    HELD_OUT_SCENARIO_INDICES,
)


def _build_active_regular_pool() -> list[dict]:
    """Reproduce IntersectionEnv._reset's active regular pool (base × 4 rotations,
    minus excluded/held-out indices). These are the curated, solvable crossing
    scenarios the id6 checkpoint was trained and evaluated on."""
    all_scen = []
    for base in base_complete_scenarios_6_cars:
        all_scen.append(base)
        for rot in (1, 2, 3):
            all_scen.append({
                "agents": [rotate_scenario_clockwise([a], rot)[0] for a in base["agents"]],
                "static": [rotate_scenario_clockwise([s], rot)[0] for s in base["static"]],
            })
    excl = EXCLUDED_SCENARIO_INDICES | HELD_OUT_SCENARIO_INDICES
    return [s for i, s in enumerate(all_scen) if i not in excl]


def _build_scenario_pool() -> tuple[list[dict], str]:
    """Prefer the crossing-heavy G03 pool (the conflict-dense scenarios that
    produced the published 83% / 55% / 8% numbers and where coordination is
    genuinely required). Fall back to the curated regular pool if absent."""
    import pickle
    pkl = os.path.join(_REPO, "experiment_runs", "G03_CROSSING_QUAD_2026_05_03-23_39_21",
                       "custom_crossing_test_scenarios.pkl")
    if os.path.isfile(pkl):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            pool = [{"agents": [tuple(a) for a in s["agents"]], "static": list(s.get("static", []))}
                    for s in data if s.get("agents")]
            if pool:
                return pool, "crossing_heavy_g03"
        except Exception:
            pass
    return _build_active_regular_pool(), "active_regular"


ACTIVE_REGULAR_POOL, SCENARIO_POOL_NAME = _build_scenario_pool()


# ── Checkpoints (best validated id6 pair; fine-tuned continuation optional) ───
CKPT_BASELINE = (
    os.path.join(_REPO, "models", "agent", "ckpt_agent6.pth"),
    os.path.join(_REPO, "models", "master", "ckpt_master6.pth"),
)
CKPT_FINETUNED = (
    os.path.join(_REPO, "experiment_runs", "fine_tune_id6", "A_base", "W_MASTER", "s123", "best", "ckpt_agent.pth"),
    os.path.join(_REPO, "experiment_runs", "fine_tune_id6", "A_base", "W_MASTER", "s123", "best", "ckpt_master.pth"),
)

CONDITIONS = (
    "normal",
    "zero_master",
    "const_all_masters",
    "swap_local_masters",
    "zero_global_master",
)

# Friendly aliases accepted by --conditions.
CONDITION_ALIASES = {
    "full": "normal", "normal": "normal", "all_masters": "normal",
    "zero": "zero_master", "zero_master": "zero_master", "zero_lm": "zero_master",
    "const": "const_all_masters", "const_all_masters": "const_all_masters",
    "swap": "swap_local_masters", "swap_local_masters": "swap_local_masters",
    "zero_global": "zero_global_master", "zero_global_master": "zero_global_master", "zero_gm": "zero_global_master",
}


def _present_conditions(results: list[dict[str, Any]]) -> list[str]:
    """Conditions actually evaluated, in canonical order."""
    seen: set[str] = set()
    for r in results:
        seen |= set(r.get("conditions", {}).keys())
    return [c for c in CONDITIONS if c in seen]

# Default layouts to sweep: (n_local_masters M, agents_per_master K).
# Up to 2 local masters share one dense intersection (the baseline crossing
# unit). (M=2,K=3) == the original 6-car / 2-LM / 1-GM baseline.
DEFAULT_LAYOUTS = [
    (2, 3),   # 6  agents  — BASELINE: 1 intersection, 2 LMs (reproduces id6)
    (4, 3),   # 12 agents  — 2 intersections, 4 LMs
    (6, 3),   # 18 agents  — 3 intersections, 6 LMs
    (8, 3),   # 24 agents  — 4 intersections, 8 LMs (GM aggregates >5 → recursive)
    (10, 3),  # 30 agents  — 5 intersections, 10 LMs
    (2, 4),   # 8  agents  — 1 intersection, 2 LMs × 4 (denser per master, OOD slot)
    (2, 5),   # 10 agents  — 1 intersection, 2 LMs × 5 (densest per master)
    (4, 4),   # 16 agents  — 2 intersections, 4 LMs × 4
    (1, 3),   # 3  agents  — single sparse local master (low-density reference)
]

# High-scale sweep up to ~50 agents. Two clean families for the paper:
#   * Master-scaling (fix K=3): grow the number of local masters / intersections.
#   * Agent-density  (fix M=2): grow agents per master (incl. out-of-distribution
#     5th slot) on a single intersection.
# Plus a 50-agent capstone point (10 masters × 5 agents).
HIGH_SCALE_LAYOUTS = [
    (2, 3),    # 6   — baseline crossing unit
    (4, 3),    # 12
    (6, 3),    # 18
    (8, 3),    # 24  (GM recursive aggregation kicks in beyond 5 LMs)
    (12, 3),   # 36
    (16, 3),   # 48  — master-scaling endpoint
    (2, 4),    # 8   — density: 4 agents per master
    (2, 5),    # 10  — density: 5 agents per master (OOD 5th slot)
    (10, 5),   # 50  — CAPSTONE: 50 agents, 10 masters, 5 intersections
]

APPROACH_LANES = [
    (("o0", "ir0", 0), 0),
    (("o1", "ir1", 0), 1),
    (("o2", "ir2", 0), 2),
    (("o3", "ir3", 0), 3),
]
CAR_COLORS = [
    (0, 204, 0), (0, 0, 204), (204, 0, 0), (204, 204, 0),
    (0, 204, 204), (204, 0, 204), (120, 120, 120), (255, 140, 0),
]
CONST_BROADCAST = 9999.0


# ──────────────────────────────────────────────────────────────────────────────
# Master / agent input packing (generalised but matching the pretrained packing)
# ──────────────────────────────────────────────────────────────────────────────
def _pad_vec(vec: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, len(arr))] = arr[: min(size, len(arr))]
    return out


def _slot(vec: np.ndarray, identifier: float, slot_vec_dim: int) -> np.ndarray:
    return np.concatenate([_pad_vec(vec, slot_vec_dim), np.asarray([identifier], dtype=np.float32)])


def build_local_master_input(global_emb: np.ndarray, group_states: np.ndarray, proto_exp) -> np.ndarray:
    """slot0 = global feedback (id=1), then one slot per subordinate (id=0)."""
    num_slots = 5
    svd = proto_exp.slot_vec_dim
    slots = [_slot(global_emb, 1.0, svd)]
    for st in np.asarray(group_states, dtype=np.float32):
        slots.append(_slot(st[:4], 0.0, svd))
    while len(slots) < num_slots:
        slots.append(np.zeros(svd + 1, dtype=np.float32))
    return np.concatenate(slots[:num_slots]).astype(np.float32)


def master_embeddings_batch(master_model, inputs: list[np.ndarray], proto_exp, deterministic: bool) -> list[np.ndarray]:
    """One batched forward through the shared master for many packed inputs."""
    import torch
    if not inputs:
        return []
    obs_t = torch.as_tensor(np.asarray(inputs, dtype=np.float32))
    with torch.no_grad():
        actions, _, _ = master_model.model.policy.forward(obs_t, deterministic=deterministic)
    arr = actions.detach().cpu().numpy().astype(np.float32)
    ed = proto_exp.embedding_dim
    return [arr[i].reshape(-1)[:ed] for i in range(arr.shape[0])]


def _pack_embeddings_as_master_input(embs: list[np.ndarray], proto_exp) -> np.ndarray:
    """Pack up to NUM_MASTER_SLOTS embeddings (each id=1) — the GM packing."""
    num_slots = 5
    svd = proto_exp.slot_vec_dim
    slots = [_slot(e, 1.0, svd) for e in embs[:num_slots]]
    while len(slots) < num_slots:
        slots.append(np.zeros(svd + 1, dtype=np.float32))
    return np.concatenate(slots[:num_slots]).astype(np.float32)


def global_master_embedding(master_model, lm_embs: list[np.ndarray], proto_exp, deterministic: bool) -> np.ndarray:
    """
    Aggregate M local-master embeddings into ONE global embedding through the
    shared master. If M exceeds the slot budget, aggregate recursively in chunks
    so the hierarchy keeps scaling past 5 masters (two-level global aggregation).
    """
    num_slots = 5
    cur = [np.asarray(e, dtype=np.float32).reshape(-1) for e in lm_embs]
    if not cur:
        return np.zeros(proto_exp.embedding_dim, dtype=np.float32)
    while len(cur) > num_slots:
        # >5 masters: aggregate in chunks of <=5 (one batched forward), then recurse.
        chunk_inputs = [
            _pack_embeddings_as_master_input(cur[i : i + num_slots], proto_exp)
            for i in range(0, len(cur), num_slots)
        ]
        cur = master_embeddings_batch(master_model, chunk_inputs, proto_exp, deterministic)
    gm_in = _pack_embeddings_as_master_input(cur, proto_exp)
    return master_embeddings_batch(master_model, [gm_in], proto_exp, deterministic)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Intersection cell — one DENSE crossing governed by >=1 local masters.
#
# Crucial: the master's value only emerges when the intersection is dense enough
# that agents genuinely conflict. The baseline unit is 6 cars / 2 local masters /
# 1 GM (normal ~83% arrival, zero_master ~55%). A lone 3-car group barely
# conflicts and is solved by the agent alone — so we keep each intersection dense
# and scale the NUMBER of intersections (and the cars per master).
# ──────────────────────────────────────────────────────────────────────────────
def _make_cell_config(n_cars: int, target_speeds: list[int]) -> dict:
    """Build a full intersection env config with N controlled cars."""
    controlled = {}
    for i in range(n_cars):
        lane, c = APPROACH_LANES[i % 4]
        back = (i // 4) * 30
        controlled[f"car{i + 1}"] = {
            "start_lane": lane,
            "destination": f"o{(c + 2) % 4}",
            "speed": 5,
            "init_location": {"longitudinal": 40 - back, "lateral": 0},
            "color": CAR_COLORS[i % len(CAR_COLORS)],
        }
    default_scenario = _make_intersection_scenario(n_cars, np.random.default_rng(0))
    base = {
        "controlled_cars": controlled,
        "static_cars": {},
        "collision_reward": -50,
        "arrived_reward": 50,
        "starvation_reward": 0,
        "high_speed_reward": 5,
        "custom_regular_scenarios": [default_scenario],
        "custom_regular_only": True,
    }
    cfg = create_full_environment_config(base)
    cfg["action"]["target_speeds"] = list(target_speeds)
    return cfg


def _make_intersection_scenario(n_cars: int, rng: np.random.Generator) -> dict:
    """A crossing scenario with exactly N controlled cars.

    * N == 6 : a real curated 6-car crossing (the dense, master-dependent unit).
    * N <  6 : first N agents of a curated scenario (sparser).
    * N >  6 : a curated 6-car crossing densified with extra staggered cars so
               the conflict (and the master's job) grows with density.
    """
    base = ACTIVE_REGULAR_POOL[int(rng.integers(len(ACTIVE_REGULAR_POOL)))]
    agents = [tuple(a) for a in base["agents"]]
    if n_cars <= len(agents):
        return {"agents": agents[:n_cars], "static": []}
    # Densify: track per-lane occupancy so extra cars sit safely behind.
    lane_min_off: dict[Any, float] = {}
    for lane, _dest, off in agents:
        lane_min_off[lane] = min(lane_min_off.get(lane, 0.0), float(off))
    out = list(agents)
    j = 0
    while len(out) < n_cars:
        lane, c = APPROACH_LANES[j % 4]
        prev = lane_min_off.get(lane, 0.0)
        off = prev - 30.0  # >=30 gap keeps spawns collision-free
        lane_min_off[lane] = off
        turn = int(rng.choice([1, 2, 3]))
        out.append((lane, f"o{(c + turn) % 4}", off))
        j += 1
    return {"agents": out[:n_cars], "static": []}


def _gen_cell_scenario(n_cars: int, rng: np.random.Generator) -> dict:
    return _make_intersection_scenario(n_cars, rng)


def _lm_groups(n_cars: int, agents_per_lm: int) -> list[list[int]]:
    """Partition car indices of one intersection into local-master groups."""
    return [list(range(g, min(g + agents_per_lm, n_cars))) for g in range(0, n_cars, agents_per_lm)]


class IntersectionCell:
    """One RELintersection env with N controlled cars, internally partitioned
    into local-master groups of size ``agents_per_lm``. Reads controlled-vehicle
    state directly and isolates the shared after_is_arrived_flags so multiple
    cells can coexist in one process."""

    def __init__(self, n_cars: int, agents_per_lm: int, target_speeds: list[int]):
        self.k = n_cars            # cars in this intersection
        self.agents_per_lm = agents_per_lm
        self.groups = _lm_groups(n_cars, agents_per_lm)
        self.cfg = _make_cell_config(n_cars, target_speeds)
        self.env = gym.make("RELintersection-v0", render_mode=None, config=self.cfg)
        self.flags = [False] * n_cars
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

    def reset(self, scenario: dict) -> np.ndarray:
        inner = self._inner()
        inner.config["custom_regular_scenarios"] = [scenario]
        inner.config["custom_regular_only"] = True
        inner.config["conflict_ratio"] = 0.0
        inner.config["use_conflict_scenarios_only"] = False
        inner.config["use_held_out_scenarios"] = False
        project_globals.after_is_arrived_flags = [False] * self.k
        self.env.reset()
        self.flags = project_globals.after_is_arrived_flags
        self.done = False
        self.crashed = False
        return self.read_state()

    def step(self, actions: list[int]) -> np.ndarray:
        if self.done:
            return self.read_state()
        project_globals.after_is_arrived_flags = self.flags
        _, _reward, done, truncated, _info = self.env.step(tuple(int(a) for a in actions))
        self.flags = project_globals.after_is_arrived_flags
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
# One coordinated episode across M cells under a coordination condition
# ──────────────────────────────────────────────────────────────────────────────
def _apply_condition_to_lm(condition: str, lm_embs: list[np.ndarray]) -> list[np.ndarray]:
    if condition == "zero_master":
        return [np.zeros_like(e) for e in lm_embs]
    if condition == "const_all_masters":
        return [np.full_like(e, CONST_BROADCAST) for e in lm_embs]
    if condition == "swap_local_masters":
        # Generalised swap: roll group→group by one (LM_i used by group i+1).
        if len(lm_embs) <= 1:
            return list(lm_embs)
        return lm_embs[-1:] + lm_embs[:-1]
    return list(lm_embs)  # normal, zero_global_master (LM path unchanged)


def _apply_condition_to_global(condition: str, gm_emb: np.ndarray) -> np.ndarray:
    if condition in ("zero_master", "zero_global_master"):
        return np.zeros_like(gm_emb)
    if condition == "const_all_masters":
        return np.full_like(gm_emb, CONST_BROADCAST)
    return gm_emb


def run_coordinated_episode(
    cells: list[IntersectionCell],
    scenarios: list[dict],
    master_model,
    agent_model,
    proto_exp,
    condition: str,
    *,
    max_steps: int = 80,
    deterministic_master: bool = False,
) -> dict[str, Any]:
    states = [c.reset(scenarios[m]) for m, c in enumerate(cells)]
    emb_dim = proto_exp.embedding_dim
    global_emb_prev = np.zeros(emb_dim, dtype=np.float32)
    steps = 0
    while steps < max_steps and not all(c.done for c in cells):
        steps += 1
        # 1) One local-master embedding PER GROUP, across every intersection.
        #    Pack all group inputs and run a single batched master forward.
        lm_inputs: list[np.ndarray] = []
        lm_index: list[list[int]] = []  # lm_index[m][g] -> position in lm_embs
        for m, c in enumerate(cells):
            idxs = []
            for grp in c.groups:
                idxs.append(len(lm_inputs))
                lm_inputs.append(build_local_master_input(global_emb_prev, states[m][grp], proto_exp))
            lm_index.append(idxs)
        lm_embs = master_embeddings_batch(master_model, lm_inputs, proto_exp, deterministic_master)
        # 2) Global master aggregates ALL local-master embeddings (recursive if >5).
        gm_emb = global_master_embedding(master_model, lm_embs, proto_exp, deterministic_master)
        gm_feedback = _apply_condition_to_global(condition, gm_emb)
        # 3) Embeddings actually broadcast to agents (condition-dependent).
        lm_used = _apply_condition_to_lm(condition, lm_embs)
        # 4) Build all agent observations, query the shared agent policy once.
        obs_all, owners = [], []
        for m, c in enumerate(cells):
            if c.done:
                continue
            for g, grp in enumerate(c.groups):
                emb = lm_used[lm_index[m][g]]
                for i in grp:
                    obs_all.append(np.concatenate([states[m][i][:4], emb]).astype(np.float32))
                    owners.append(m)
        if obs_all:
            actions, _, _ = rps.agent_actions(agent_model, obs_all, deterministic=True)
        else:
            actions = []
        # 5) Step each active cell with its agents' actions (group order == car order).
        per_cell_actions: dict[int, list[int]] = {m: [] for m in range(len(cells))}
        for a, m in zip(actions, owners):
            per_cell_actions[m].append(int(a))
        for m, c in enumerate(cells):
            if c.done:
                continue
            states[m] = c.step(per_cell_actions[m])
        # 6) Feedback loop.
        global_emb_prev = gm_feedback

    total_agents = sum(c.k for c in cells)
    total_arrived = sum(c.n_arrived() for c in cells)
    n_cells_crashed = sum(1 for c in cells if c.crashed)
    return {
        "arrival_pct": 100.0 * total_arrived / max(1, total_agents),
        "any_crash": int(n_cells_crashed > 0),
        "cells_crashed": n_cells_crashed,
        "n_cells": len(cells),
        "steps": steps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────
def make_proto_and_models(agent_pth: str, master_pth: str):
    cfg = dict(rps.BASE_CFG)
    cfg["label"] = "scalability"
    cfg["embedding_dim"] = 4
    cfg["load_pretrained"] = False
    rps.set_all_seeds(rps.SEED)
    work = os.path.join(_REPO, "MODELS_EVALUATION", "_scalability_work")
    os.makedirs(work, exist_ok=True)
    proto_exp = rps.ProtoExperiment(cfg, work)
    master_model, agent_model = rps.make_models(proto_exp)
    if not load_models_from_paths(agent_model, master_model, agent_pth, master_pth):
        raise RuntimeError(f"Failed to load checkpoints:\n  {agent_pth}\n  {master_pth}")
    return proto_exp, master_model, agent_model


def plan_cells(m: int, k: int, lm_per_cell: int = 2) -> list[int]:
    """Map M local masters onto intersections (up to ``lm_per_cell`` LMs share one
    dense crossing). Returns the controlled-car count for each intersection."""
    cell_cars: list[int] = []
    rem = m
    while rem > 0:
        lms_here = min(lm_per_cell, rem)
        cell_cars.append(lms_here * k)
        rem -= lms_here
    return cell_cars


def gen_layout_scenarios(cell_cars: list[int], n_scenarios: int, base_seed: int) -> list[list[dict]]:
    """Deterministically generate the per-episode scenarios for a layout."""
    rng = np.random.default_rng(base_seed)
    return [[_make_intersection_scenario(nc, rng) for nc in cell_cars] for _ in range(n_scenarios)]


def run_layout(
    m: int,
    k: int,
    master_model,
    agent_model,
    proto_exp,
    n_scenarios: int,
    base_seed: int,
    target_speeds: list[int],
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, Any]:
    cap = 5 - 1  # local-master subordinate capacity given the reserved global slot
    note = "" if k <= cap else f"K={k} exceeds per-master slot capacity {cap}; extra subordinates are truncated"

    cell_cars = plan_cells(m, k)
    n_cells = len(cell_cars)
    scenarios_per_ep = gen_layout_scenarios(cell_cars, n_scenarios, base_seed)

    cells = [IntersectionCell(nc, k, target_speeds) for nc in cell_cars]
    out: dict[str, Any] = {"n_local_masters": m, "agents_per_master": k, "total_agents": m * k,
                           "n_intersections": n_cells, "cars_per_intersection": cell_cars,
                           "n_scenarios": n_scenarios, "capacity_note": note, "conditions": {}}
    t0 = time.time()
    total_env_steps = 0
    try:
        for condition in conditions:
            arrivals, any_crashes, cell_crashes, cell_counts, step_counts = [], [], [], [], []
            for ep in range(n_scenarios):
                rps.set_all_seeds(base_seed + ep)
                res = run_coordinated_episode(
                    cells, scenarios_per_ep[ep], master_model, agent_model, proto_exp, condition
                )
                arrivals.append(res["arrival_pct"])
                any_crashes.append(res["any_crash"])
                cell_crashes.append(res["cells_crashed"])
                cell_counts.append(res["n_cells"])
                step_counts.append(res["steps"])
                total_env_steps += res["steps"] * n_cells
            arr = np.asarray(arrivals, dtype=float)
            n = max(1, arr.size)
            out["conditions"][condition] = {
                "arrival_pct_mean": float(np.mean(arr)),
                "arrival_pct_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "arrival_pct_sem": (float(np.std(arr, ddof=1) / np.sqrt(n)) if arr.size > 1 else 0.0),
                "crash_rate_per_cell_pct": 100.0 * float(np.sum(cell_crashes)) / max(1, float(np.sum(cell_counts))),
                "crash_rate_any_episode_pct": 100.0 * float(np.mean(any_crashes)),
                "mean_steps": float(np.mean(step_counts)),
                "n_scenarios": int(arr.size),
                "arrival_pct_samples": [round(float(x), 4) for x in arrivals],
            }
    finally:
        for c in cells:
            c.close()
    out["wall_time_sec"] = round(time.time() - t0, 2)
    out["total_env_steps"] = int(total_env_steps)
    return out


def _suffix_path(out_png: str, suffix: str) -> str:
    base, ext = os.path.splitext(out_png)
    return f"{base}_{suffix}{ext or '.png'}"


def make_plot(results: list[dict[str, Any]], out_png: str) -> None:
    """Headline scaling curves. Saved as TWO separate images:
    ``*_arrival.png`` and ``*_crash.png`` (one chart per file)."""
    # Use the master-scaling family (fixed K=3) for the headline scaling curves;
    # fall back to all layouts if that family is absent.
    fam = [r for r in results if r["agents_per_master"] == 3]
    if len(fam) < 2:
        fam = results
    fam = sorted(fam, key=lambda r: r["total_agents"])
    xs = [r["total_agents"] for r in fam]
    conds = _present_conditions(results)
    colors = {
        "normal": "#2E7D32",
        "zero_master": "#C62828",
        "const_all_masters": "#6A1B9A",
        "swap_local_masters": "#1565C0",
        "zero_global_master": "#EF6C00",
    }
    xlabel = "Total agents  (M local masters × K agents,  K=3)"

    # ── arrival vs scale ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cond in conds:
        arr = [r["conditions"][cond]["arrival_pct_mean"] for r in fam]
        sem = [r["conditions"][cond].get("arrival_pct_sem", 0.0) for r in fam]
        ax.errorbar(xs, arr, yerr=sem, marker="o", capsize=3, lw=2,
                    label=cond, color=colors.get(cond))
    ax.set_title("Arrival % vs scale (master signal coordinated)")
    ax.set_ylabel("Mean arrival % (±SEM)")
    ax.set_ylim(-5, 105)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    rps.safe_figure_save_png(fig, _suffix_path(out_png, "arrival"), dpi=170)
    plt.close(fig)

    # ── crash vs scale ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cond in conds:
        cr = [r["conditions"][cond]["crash_rate_per_cell_pct"] for r in fam]
        ax.plot(xs, cr, marker="o", lw=2, label=cond, color=colors.get(cond))
    ax.set_title("Per-intersection crash % vs scale")
    ax.set_ylabel("Crash rate per cell (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    rps.safe_figure_save_png(fig, _suffix_path(out_png, "crash"), dpi=170)
    plt.close(fig)


def make_master_benefit_plot(results: list[dict[str, Any]], out_png: str) -> None:
    """Grouped bars: normal vs each ablation, across ALL layouts — shows the
    coordination (master) benefit persists as agents/masters grow."""
    res = sorted(results, key=lambda r: (r["total_agents"], r["agents_per_master"]))
    labels = [f"{r['n_local_masters']}x{r['agents_per_master']}\n(N={r['total_agents']})" for r in res]
    x = np.arange(len(res))
    conds = _present_conditions(results)
    width = 0.8 / max(1, len(conds))
    colors = {
        "normal": "#2E7D32", "zero_master": "#C62828", "const_all_masters": "#6A1B9A",
        "swap_local_masters": "#1565C0", "zero_global_master": "#EF6C00",
    }
    figw = max(8, len(res) * 1.1)

    def _bars(metric_key, title, ylabel, suffix):
        fig, ax = plt.subplots(figsize=(figw, 5.5))
        for j, cond in enumerate(conds):
            vals = [r["conditions"][cond][metric_key] for r in res]
            ax.bar(x + j * width, vals, width=width * 0.95, label=cond,
                   color=colors.get(cond), alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.set_xticks(x + width * (len(conds) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=len(conds))
        fig.tight_layout()
        rps.safe_figure_save_png(fig, _suffix_path(out_png, suffix), dpi=170)
        plt.close(fig)

    _bars("arrival_pct_mean", "Arrival % per layout × coordination condition",
          "Mean arrival %", "arrival")
    _bars("crash_rate_per_cell_pct", "Per-intersection crash % per layout × coordination condition",
          "Crash rate per cell (%)", "crash")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario serialisation — JSON specs (start/end per agent) + schematic diagrams
# ──────────────────────────────────────────────────────────────────────────────
_TURN_BY_DIFF = {0: "u_turn", 1: "right", 2: "straight", 3: "left"}


def _interpret_agent(agent: tuple) -> dict[str, Any]:
    """Decode a scenario agent tuple (start_lane, destination, offset) into a
    human/paper-readable entry→exit spec."""
    start_lane = agent[0]
    entry_id = start_lane[0] if isinstance(start_lane, (tuple, list)) else str(start_lane)
    dest_id = agent[1]
    offset = float(agent[2]) if len(agent) > 2 else 0.0
    try:
        ei = int(str(entry_id)[1:])
    except Exception:
        ei = 0
    try:
        di = int(str(dest_id)[1:])
    except Exception:
        di = (ei + 2) % 4
    return {
        "entry_approach": str(entry_id),
        "exit_approach": str(dest_id),
        "entry_index": ei,
        "exit_index": di,
        "lane_offset": offset,
        "turn": _TURN_BY_DIFF[(di - ei) % 4],
    }


def _approach_xy(idx: int, radius: float, offset: float = 0.0) -> tuple[float, float]:
    """Place approach ``idx`` on a circle; positive offset pushes the spawn
    further out along the approach (cars queued behind each other)."""
    ang = np.deg2rad(idx * 90.0)
    r = radius + abs(offset) * 0.25
    return float(r * np.cos(ang)), float(r * np.sin(ang))


def _scenario_to_record(scenario: dict, real_start: np.ndarray | None = None) -> list[dict]:
    """Full per-agent record: symbolic entry/exit, schematic start/end coords, and
    (if available) the real (x, y) spawn read from the environment after reset."""
    recs = []
    for i, agent in enumerate(scenario["agents"]):
        info = _interpret_agent(tuple(agent))
        sx, sy = _approach_xy(info["entry_index"], radius=60.0, offset=info["lane_offset"])
        ex, ey = _approach_xy(info["exit_index"], radius=60.0)
        rec = {
            "agent_id": i,
            "color_rgb": list(CAR_COLORS[i % len(CAR_COLORS)]),
            **info,
            "schematic_start_xy": [round(sx, 2), round(sy, 2)],
            "schematic_end_xy": [round(ex, 2), round(ey, 2)],
        }
        if real_start is not None and i < len(real_start):
            v = real_start[i]
            rec["env_start_xy"] = [round(float(v[0]), 3), round(float(v[1]), 3)]
            rec["env_start_velocity"] = [round(float(v[2]), 3), round(float(v[3]), 3)]
        recs.append(rec)
    return recs


def render_scenario_schematic(scenario: dict, title: str, out_png: str) -> None:
    """Draw a 4-way crossing schematic: each agent as a colored dot at its entry
    with a curved arrow to its destination exit."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    R = 70.0
    for idx in range(4):
        x, y = _approach_xy(idx, R)
        ax.plot([0, x], [0, y], color="#cccccc", lw=10, solid_capstyle="round", zorder=0)
        lx, ly = _approach_xy(idx, R * 1.18)
        ax.text(lx, ly, f"o{idx}", ha="center", va="center", fontsize=11, color="#555555", weight="bold")
    ax.add_patch(plt.Circle((0, 0), 14, color="#eeeeee", zorder=1))
    recs = _scenario_to_record(scenario)
    for rec in recs:
        c = tuple(v / 255.0 for v in rec["color_rgb"])
        sx, sy = rec["schematic_start_xy"]
        ex, ey = rec["schematic_end_xy"]
        ax.add_patch(mpatches.FancyArrowPatch(
            (sx, sy), (ex, ey), connectionstyle="arc3,rad=0.25",
            arrowstyle="-|>", mutation_scale=16, lw=2.0, color=c, alpha=0.9, zorder=3))
        ax.scatter([sx], [sy], s=120, color=c, edgecolors="black", linewidths=0.8, zorder=4)
        ax.text(sx, sy, str(rec["agent_id"]), ha="center", va="center", fontsize=8,
                color="white", weight="bold", zorder=5)
    lim = R * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    rps.safe_figure_save_png(fig, out_png, dpi=150)
    plt.close(fig)


def dump_scenarios(
    out_root: str,
    layouts: list[tuple[int, int]],
    base_seed: int,
    target_speeds: list[int],
    *,
    n_scenarios: int,
    max_diagrams_per_layout: int = 3,
    capture_env_start: bool = True,
) -> None:
    """Persist the exact scenarios evaluated for each layout: a JSON file with the
    full entry/exit spec of every agent, plus a few schematic PNGs. Scenarios are
    regenerated with the SAME seed as the sweep, so they match what was scored."""
    scen_root = os.path.join(out_root, "scenarios")
    os.makedirs(scen_root, exist_ok=True)
    for (m, k) in layouts:
        cell_cars = plan_cells(m, k)
        eps = gen_layout_scenarios(cell_cars, n_scenarios, base_seed)
        tag = f"{m}x{k}_N{m * k}"
        layout_dir = os.path.join(scen_root, tag)
        os.makedirs(layout_dir, exist_ok=True)

        # Capture real spawn coordinates for episode 0 (one cell per unique size).
        real_start_ep0: list[np.ndarray | None] = [None] * len(cell_cars)
        if capture_env_start:
            cache: dict[int, IntersectionCell] = {}
            try:
                for ci, nc in enumerate(cell_cars):
                    cell = cache.get(nc) or IntersectionCell(nc, k, target_speeds)
                    cache[nc] = cell
                    try:
                        real_start_ep0[ci] = cell.reset(eps[0][ci])
                    except Exception:
                        real_start_ep0[ci] = None
            finally:
                for cell in cache.values():
                    cell.close()

        payload = {
            "layout": {"n_local_masters": m, "agents_per_master": k, "total_agents": m * k,
                       "n_intersections": len(cell_cars), "cars_per_intersection": cell_cars,
                       "lm_per_intersection": 2},
            "base_seed": base_seed,
            "scenario_pool": SCENARIO_POOL_NAME,
            "legend": {"turn": _TURN_BY_DIFF,
                       "note": "schematic_*_xy are normalized diagram coords; "
                               "env_start_xy (episode 0 only) is the real spawn read from highway-env."},
            "episodes": [],
        }
        for ep_idx, ep in enumerate(eps):
            ep_rec = {"episode": ep_idx, "intersections": []}
            for ci, scenario in enumerate(ep):
                rs = real_start_ep0[ci] if ep_idx == 0 else None
                ep_rec["intersections"].append({
                    "intersection_id": ci,
                    "n_cars": cell_cars[ci],
                    "agents": _scenario_to_record(scenario, rs),
                })
            payload["episodes"].append(ep_rec)
        with open(os.path.join(layout_dir, "scenarios.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        for ci, scenario in enumerate(eps[0][:max_diagrams_per_layout]):
            render_scenario_schematic(
                scenario,
                title=f"Layout {m}x{k} (N={m * k}) — intersection {ci} ({cell_cars[ci]} cars)",
                out_png=os.path.join(layout_dir, f"diagram_intersection{ci}.png"),
            )
    print(f"[done] scenarios -> {scen_root}")


# ──────────────────────────────────────────────────────────────────────────────
# Paper tables (CSV for analysis + LaTeX booktabs for the manuscript)
# ──────────────────────────────────────────────────────────────────────────────
def write_tables(results: list[dict[str, Any]], out_root: str) -> None:
    import csv
    res = sorted(results, key=lambda r: (r["agents_per_master"], r["total_agents"]))
    conds = _present_conditions(results)

    csv_path = os.path.join(out_root, "scalability_table.csv")
    cols = ["M_local_masters", "K_agents_per_master", "N_total_agents", "n_intersections",
            "condition", "arrival_pct_mean", "arrival_pct_std", "crash_rate_per_cell_pct",
            "crash_rate_any_episode_pct", "mean_steps", "wall_time_sec", "total_env_steps",
            "n_scenarios"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in res:
            for cond in conds:
                cc = r["conditions"][cond]
                w.writerow([
                    r["n_local_masters"], r["agents_per_master"], r["total_agents"],
                    r["n_intersections"], cond,
                    f"{cc['arrival_pct_mean']:.2f}", f"{cc.get('arrival_pct_std', 0.0):.2f}",
                    f"{cc['crash_rate_per_cell_pct']:.2f}", f"{cc['crash_rate_any_episode_pct']:.2f}",
                    f"{cc['mean_steps']:.1f}", r.get("wall_time_sec", 0.0),
                    r.get("total_env_steps", 0), cc.get("n_scenarios", r.get("n_scenarios", 0)),
                ])

    def _fmt(cc, key="arrival_pct_mean", std_key="arrival_pct_std"):
        return f"{cc[key]:.1f}\\,$\\pm$\\,{cc.get(std_key, 0.0):.1f}"

    cond_hdr = {"normal": "Normal", "zero_master": "Zero LM", "const_all_masters": "Const",
                "swap_local_masters": "Swap LM", "zero_global_master": "Zero GM"}
    tex_lines = [
        "% Auto-generated by run_scalability_suite.py",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Coordination ablations across scale (shared id6 weights, no retraining). "
        "Arrival \\% is mean\\,$\\pm$\\,std over scenarios; crash is per-intersection rate.}",
        "\\label{tab:scalability}", "\\small", "\\begin{tabular}{l" + "c" * len(conds) + "}",
        "\\toprule",
        "Layout (N agents) & " + " & ".join(cond_hdr[c] for c in conds) + " \\\\",
        "\\midrule",
        "\\multicolumn{" + str(1 + len(conds)) + "}{l}{\\emph{Arrival rate (\\%)}} \\\\",
    ]
    for r in res:
        row = f"{r['n_local_masters']}$\\times${r['agents_per_master']} (N={r['total_agents']})"
        row += " & " + " & ".join(_fmt(r["conditions"][c]) for c in conds) + " \\\\"
        tex_lines.append(row)
    tex_lines += ["\\midrule",
                  "\\multicolumn{" + str(1 + len(conds)) + "}{l}{\\emph{Crash rate per intersection (\\%)}} \\\\"]
    for r in res:
        row = f"{r['n_local_masters']}$\\times${r['agents_per_master']} (N={r['total_agents']})"
        row += " & " + " & ".join(f"{r['conditions'][c]['crash_rate_per_cell_pct']:.1f}" for c in conds) + " \\\\"
        tex_lines.append(row)
    tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(os.path.join(out_root, "scalability_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"[done] tables  -> {csv_path}")
    print(f"[done] tables  -> {os.path.join(out_root, 'scalability_table.tex')}")


def make_compute_plot(results: list[dict[str, Any]], out_png: str) -> None:
    """Compute cost vs scale: wall-time and environment steps per layout."""
    res = sorted(results, key=lambda r: r["total_agents"])
    xs = [r["total_agents"] for r in res]
    wall = [r.get("wall_time_sec", 0.0) for r in res]
    steps = [r.get("total_env_steps", 0) for r in res]
    n_cond = len(_present_conditions(results))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, wall, marker="o", color="#1565C0", label="Wall time (s)")
    ax1.set_xlabel("Total agents (N)")
    ax1.set_ylabel(f"Wall time (s, all {n_cond} conditions)", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, steps, marker="s", color="#C62828", label="Env steps")
    ax2.set_ylabel("Total environment steps", color="#C62828")
    ax2.tick_params(axis="y", labelcolor="#C62828")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    fig.suptitle("Compute cost vs scale", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    rps.safe_figure_save_png(fig, out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", choices=("baseline", "finetuned"), default="baseline")
    p.add_argument("--agent-pth", default="")
    p.add_argument("--master-pth", default="")
    p.add_argument("--n-scenarios", type=int, default=30)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--output-root", default="")
    p.add_argument("--smoke", action="store_true", help="2 small layouts × 4 scenarios for a quick check.")
    p.add_argument("--high-scale", action="store_true",
                   help="Sweep the up-to-50-agent layouts (master-scaling, density, 50-agent capstone).")
    p.add_argument("--layouts", default="", help="Override layouts, e.g. '1x3,2x3,3x3,2x4'.")
    p.add_argument("--conditions", default="",
                   help="Subset of conditions to run, e.g. 'full,zero' or "
                        "'normal,zero_master'. Default: all five.")
    p.add_argument("--no-scenario-dump", action="store_true", help="Skip saving scenario JSON/diagrams.")
    args = p.parse_args()

    if args.conditions.strip():
        sel = []
        for tok in args.conditions.split(","):
            key = tok.strip().lower()
            if not key:
                continue
            if key not in CONDITION_ALIASES:
                raise SystemExit(f"Unknown condition '{tok}'. Valid: {sorted(set(CONDITION_ALIASES))}")
            c = CONDITION_ALIASES[key]
            if c not in sel:
                sel.append(c)
        conditions = tuple(c for c in CONDITIONS if c in sel)
    else:
        conditions = CONDITIONS

    if args.agent_pth and args.master_pth:
        agent_pth, master_pth = args.agent_pth, args.master_pth
    else:
        agent_pth, master_pth = CKPT_FINETUNED if args.checkpoint == "finetuned" else CKPT_BASELINE
    agent_pth = os.path.abspath(agent_pth)
    master_pth = os.path.abspath(master_pth)

    if args.layouts.strip():
        layouts = []
        for tok in args.layouts.split(","):
            mm, kk = tok.lower().split("x")
            layouts.append((int(mm), int(kk)))
    elif args.smoke:
        layouts = [(1, 3), (2, 3), (3, 3)]
        args.n_scenarios = min(args.n_scenarios, 4)
    elif args.high_scale:
        layouts = list(HIGH_SCALE_LAYOUTS)
    else:
        layouts = list(DEFAULT_LAYOUTS)

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.output_root.strip():
        out_root = os.path.abspath(args.output_root.strip())
    else:
        out_root = os.path.join(_REPO, "MODELS_EVALUATION", f"scalability_{ts}")
    os.makedirs(out_root, exist_ok=True)

    print(f"[scalability] checkpoint={args.checkpoint}\n  agent={agent_pth}\n  master={master_pth}")
    print(f"[scalability] layouts={layouts}  n_scenarios={args.n_scenarios}  out={out_root}")
    print(f"[scalability] conditions={list(conditions)}")

    proto_exp, master_model, agent_model = make_proto_and_models(agent_pth, master_pth)
    target_speeds = list(rps.BASE_CFG["target_speeds"])

    results = []
    for (m, k) in layouts:
        print(f"\n=== layout M={m} local masters × K={k} agents (N={m*k}) ===", flush=True)
        r = run_layout(m, k, master_model, agent_model, proto_exp,
                       args.n_scenarios, args.base_seed, target_speeds, conditions)
        for cond in conditions:
            cc = r["conditions"][cond]
            print(f"  {cond:<20s} arrival={cc['arrival_pct_mean']:6.1f}%  "
                  f"crash/cell={cc['crash_rate_per_cell_pct']:6.1f}%", flush=True)
        results.append(r)
        # Persist progressively so partial runs are still useful.
        payload = {
            "meta": {
                "checkpoint_kind": args.checkpoint,
                "agent_pth": agent_pth,
                "master_pth": master_pth,
                "embedding_dim": proto_exp.embedding_dim,
                "num_master_slots": 5,
                "per_master_subordinate_capacity": 4,
                "global_master_slot_capacity": 5,
                "conditions": list(conditions),
                "scenario_pool": SCENARIO_POOL_NAME,
                "scenario_pool_size": len(ACTIVE_REGULAR_POOL),
                "n_scenarios": args.n_scenarios,
                "base_seed": args.base_seed,
                "timestamp": ts,
                "note": "Same shared weights deployed at every scale; no retraining. "
                        "Each intersection hosts up to 2 local masters on a dense crossing.",
            },
            "layouts": results,
        }
        with open(os.path.join(out_root, "scalability_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    make_plot(results, os.path.join(out_root, "scalability_summary.png"))
    make_master_benefit_plot(results, os.path.join(out_root, "scalability_master_benefit.png"))
    make_compute_plot(results, os.path.join(out_root, "scalability_compute_cost.png"))
    write_tables(results, out_root)
    if not args.no_scenario_dump:
        dump_scenarios(out_root, layouts, args.base_seed, target_speeds,
                       n_scenarios=args.n_scenarios)
    print(f"\n[done] results -> {os.path.join(out_root, 'scalability_results.json')}")
    print(f"[done] plot    -> {os.path.join(out_root, 'scalability_summary.png')}")
    print(f"[done] plot    -> {os.path.join(out_root, 'scalability_master_benefit.png')}")
    print(f"[done] plot    -> {os.path.join(out_root, 'scalability_compute_cost.png')}")


if __name__ == "__main__":
    main()
