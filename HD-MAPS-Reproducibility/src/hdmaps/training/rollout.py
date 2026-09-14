from __future__ import annotations

import numpy as np

from hdmaps.env.corridor import Corridor
from hdmaps.hierarchy.packing import agent_obs, pack_children, pack_local
from hdmaps.policies.agent import agent_actions
from hdmaps.protocol import (
    ARRIVED_REWARD,
    COLLISION_REWARD,
    DETERMINISTIC_AGENT,
    EMBEDDING_DIM,
    HIGH_SPEED_REWARD,
    TRAIN_MAX_AGENTS_PER_MASTER,
)


def _group_reward(group, arrived_now, arrived_prev, crashed, speeds) -> float:
    reward = 0.0
    for idx in group:
        if arrived_now[idx] and not arrived_prev[idx]:
            reward += ARRIVED_REWARD
        if crashed[idx]:
            reward += COLLISION_REWARD
        elif (not arrived_now[idx]) and speeds[idx] >= 7.5:
            reward += HIGH_SPEED_REWARD
    return float(reward)


def train_lm_groups(cell: Corridor) -> list[list[int]]:
    zones = cell.agent_zones()
    zone_agents: dict[int, list[int]] = {}
    for idx, z in enumerate(zones):
        zone_agents.setdefault(int(z), []).append(idx)
    groups: list[list[int]] = []
    cap = TRAIN_MAX_AGENTS_PER_MASTER
    for z in sorted(zone_agents):
        ags = zone_agents[z]
        for i in range(0, len(ags), cap):
            chunk = ags[i : i + cap]
            if chunk:
                groups.append(chunk)
    return groups


def _pairwise_train(master, embeddings: list[np.ndarray]) -> tuple[np.ndarray, dict[str, dict]]:
    cur = [np.asarray(e, dtype=np.float32).reshape(-1)[:EMBEDDING_DIM] for e in embeddings]
    recs: dict[str, dict] = {}
    if not cur:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32), recs
    if len(cur) == 1:
        return cur[0], recs
    level = 0
    while len(cur) > 1:
        packed = []
        for i in range(0, len(cur), 2):
            kids = [cur[i], cur[i + 1]] if i + 1 < len(cur) else [cur[i]]
            packed.append(pack_children(kids))
        acts, vals, lps = master.act(packed, deterministic=False)
        nxt = []
        for j, obs in enumerate(packed):
            emb = np.asarray(acts[j], dtype=np.float32).reshape(-1)[:EMBEDDING_DIM]
            recs['gm_L%d_%d' % (level, j)] = {
                'obs': np.asarray(obs, dtype=np.float32),
                'action': emb,
                'value': float(vals[j]),
                'log_prob': float(lps[j]),
                'reward': 0.0,
            }
            nxt.append(emb)
        cur = nxt
        level += 1
    return cur[0], recs


def run_training_episode(cell: Corridor, scenario: dict, master, agent, *, max_steps: int) -> tuple[dict, list[list[dict]]]:
    state = cell.reset(scenario)
    gm_prev = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    arrived_prev = cell.arrived_mask()
    steps = 0
    spacing = cell.spacing()
    streams: dict[str, list[dict]] = {}
    episode_reward = 0.0

    while steps < max_steps and not cell.done:
        steps += 1
        groups = train_lm_groups(cell)
        zones = cell.agent_zones()
        if not groups:
            groups = [[i for i in range(cell.n_agents)]]
        lm_inputs = []
        for lm_idx, grp in enumerate(groups):
            shifted = state[grp].copy()
            shifted[:, 0] -= zones[grp[0]] * spacing
            lm_inputs.append(pack_local(gm_prev, shifted, center_x=0.0))
        lm_act, lm_val, lm_lp = master.act(lm_inputs, deterministic=False)
        lm_embs = [np.asarray(lm_act[i], dtype=np.float32).reshape(-1)[:EMBEDDING_DIM] for i in range(len(lm_inputs))]
        gm_emb, gm_recs = _pairwise_train(master, lm_embs)

        obs_list = []
        order = []
        for lm_idx, grp in enumerate(groups):
            for a_idx in grp:
                local = state[a_idx][:4].copy()
                local[0] -= zones[a_idx] * spacing
                obs_list.append(agent_obs(local, lm_embs[lm_idx]))
                order.append(a_idx)
        acts = agent_actions(agent, obs_list, DETERMINISTIC_AGENT)
        full = [0] * cell.n_agents
        for a_idx, act in zip(order, acts):
            full[a_idx] = int(act)
        state = cell.step(full)
        arrived_now = cell.arrived_mask()
        crashed = cell.crashed_mask()
        speeds = cell.speeds()

        group_rewards = []
        for lm_idx, grp in enumerate(groups):
            reward = _group_reward(grp, arrived_now, arrived_prev, crashed, speeds)
            group_rewards.append(reward)
            episode_reward += reward
            streams.setdefault('lm%d' % lm_idx, []).append(
                {
                    'obs': np.asarray(lm_inputs[lm_idx], dtype=np.float32),
                    'action': lm_embs[lm_idx],
                    'value': float(lm_val[lm_idx]),
                    'log_prob': float(lm_lp[lm_idx]),
                    'reward': reward,
                }
            )
        gm_reward = float(np.mean(group_rewards) if group_rewards else 0.0)
        for name, rec in gm_recs.items():
            rec = dict(rec)
            rec['reward'] = gm_reward
            streams.setdefault(name, []).append(rec)

        gm_prev = gm_emb
        arrived_prev = arrived_now

    result = {
        'arrival_pct': 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        'crashed': int(cell.crashed),
        'arrived': int(cell.n_arrived()),
        'reward': float(episode_reward),
        'steps': steps,
        'n_lms': len(train_lm_groups(cell)) if not cell.done else len(groups),
        'n_int': cell.n_int,
        'n_agents': cell.n_agents,
    }
    return result, list(streams.values())
