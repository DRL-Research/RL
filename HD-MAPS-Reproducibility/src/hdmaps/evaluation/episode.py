from __future__ import annotations

import numpy as np

from hdmaps.env.corridor import Corridor
from hdmaps.hierarchy.packing import agent_obs, pack_children, pack_local
from hdmaps.hierarchy.tree import chunk_indices
from hdmaps.hierarchy.zones import in_radius
from hdmaps.policies.agent import agent_actions
from hdmaps.protocol import (
    DETERMINISTIC_AGENT,
    DETERMINISTIC_MASTER,
    EMBEDDING_DIM,
    FANOUT,
    STAG_COORD_RADIUS_M,
)


def _aggregate_pairwise(master, embeddings: list[np.ndarray], deterministic: bool) -> np.ndarray:
    cur = [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings]
    if not cur:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if len(cur) == 1:
        return cur[0]
    while len(cur) > 1:
        packed = []
        for i in range(0, len(cur), 2):
            kids = [cur[i], cur[i + 1]] if i + 1 < len(cur) else [cur[i]]
            packed.append(pack_children(kids))
        cur = master.embeddings(packed, deterministic)
    return cur[0]


def _aggregate_fanout(master, embeddings: list[np.ndarray], deterministic: bool) -> np.ndarray:
    cur = [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings]
    if not cur:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if len(cur) == 1:
        return cur[0]
    while len(cur) > FANOUT:
        packed = [pack_children([cur[i] for i in g]) for g in chunk_indices(len(cur), FANOUT)]
        cur = master.embeddings(packed, deterministic)
    return master.embeddings([pack_children(cur)], deterministic)[0]


def run_simultaneous_episode(cell: Corridor, scenario: dict, master, agent, *, max_steps: int, condition: str = "normal") -> dict:
    state = cell.reset(scenario)
    gm_prev = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    steps = 0
    spacing = cell.spacing()
    while steps < max_steps and not cell.done:
        steps += 1
        groups = cell.lm_groups()
        zones = cell.agent_zones()
        lm_inputs = []
        for lm_idx, grp in enumerate(groups):
            center_x = lm_idx * spacing
            if grp:
                shifted = state[grp].copy()
                shifted[:, 0] -= center_x
            else:
                shifted = np.zeros((0, 4), dtype=np.float32)
            lm_inputs.append(pack_local(gm_prev, shifted, center_x=0.0))
        lm_embs = master.embeddings(lm_inputs, DETERMINISTIC_MASTER)
        gm_emb = _aggregate_pairwise(master, lm_embs, DETERMINISTIC_MASTER)
        if condition == "zero_master":
            lm_used = [np.zeros_like(e) for e in lm_embs]
            gm_feedback = np.zeros_like(gm_emb)
        else:
            lm_used = lm_embs
            gm_feedback = gm_emb
        obs_list = []
        order = []
        for lm_idx, grp in enumerate(groups):
            for a_idx in grp:
                local = state[a_idx][:4].copy()
                local[0] -= zones[a_idx] * spacing
                obs_list.append(agent_obs(local, lm_used[lm_idx]))
                order.append(a_idx)
        acts = agent_actions(agent, obs_list, DETERMINISTIC_AGENT)
        full = [0] * cell.n_agents
        for a_idx, act in zip(order, acts):
            full[a_idx] = int(act)
        state = cell.step(full)
        gm_prev = gm_feedback
    return {
        "arrival_pct": 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        "crashed": int(cell.crashed),
        "steps": steps,
        "n_lms": cell.n_lms,
    }


def run_staggered_episode(
    cell: Corridor,
    scenario: dict,
    master,
    agent,
    *,
    max_steps: int,
    condition: str = "normal",
    coord_radius: float = STAG_COORD_RADIUS_M,
) -> dict:
    state = cell.reset(scenario)
    gm_prev = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    steps = 0
    spacing = cell.spacing()
    peak_lms = 0
    while steps < max_steps and not cell.done:
        steps += 1
        zones = cell.agent_zones()
        pos = cell.positions()
        inner = cell._inner()
        active = []
        for i, v in enumerate(inner.controlled_vehicles):
            if getattr(v, "is_arrived", False) or getattr(v, "crashed", False):
                continue
            center = (zones[i] * spacing, 0.0)
            if in_radius(pos[i], center, coord_radius):
                active.append(i)
        zone_agents: dict[int, list[int]] = {}
        for i in active:
            zone_agents.setdefault(zones[i], []).append(i)
        lm_zones = sorted(zone_agents)
        peak_lms = max(peak_lms, len(lm_zones))
        lm_embs = []
        lm_of = {}
        if lm_zones:
            inputs = []
            for z in lm_zones:
                grp = zone_agents[z]
                grp_states = state[grp].copy()
                grp_states[:, 0] -= z * spacing
                inputs.append(pack_local(gm_prev, grp_states, center_x=0.0))
            lm_embs = master.embeddings(inputs, DETERMINISTIC_MASTER)
            for z, e in zip(lm_zones, lm_embs):
                lm_of[z] = e
            gm_emb = _aggregate_fanout(master, lm_embs, DETERMINISTIC_MASTER)
        else:
            gm_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        if condition == "zero_master":
            lm_of = {z: np.zeros(EMBEDDING_DIM, dtype=np.float32) for z in lm_zones}
            gm_feedback = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        else:
            gm_feedback = gm_emb
        obs_list = []
        order = []
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for i in range(cell.n_agents):
            local = state[i][:4].copy()
            local[0] -= zones[i] * spacing
            emb = lm_of.get(zones[i], zero) if i in active else zero
            obs_list.append(agent_obs(local, emb))
            order.append(i)
        acts = agent_actions(agent, obs_list, DETERMINISTIC_AGENT)
        state = cell.step([int(a) for a in acts])
        gm_prev = gm_feedback
    return {
        "arrival_pct": 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        "crashed": int(cell.crashed),
        "steps": steps,
        "peak_lms": peak_lms,
        "n_lms": cell.n_lms,
    }
