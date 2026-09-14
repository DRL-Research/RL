from __future__ import annotations

import numpy as np

from hdmaps.protocol import EMBEDDING_DIM, NUM_MASTER_SLOTS, SLOT_VEC_DIM


def pad_vec(vec, size: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, len(arr))] = arr[: min(size, len(arr))]
    return out


def slot(vec, identifier: float, slot_vec_dim: int = SLOT_VEC_DIM) -> np.ndarray:
    return np.concatenate(
        [pad_vec(vec, slot_vec_dim), np.asarray([identifier], dtype=np.float32)]
    )


def pack_local(parent_emb, group_states, *, center_x: float = 0.0) -> np.ndarray:
    slots = [slot(parent_emb, 1.0)]
    states = np.asarray(group_states, dtype=np.float32).reshape(-1, 4) if len(np.asarray(group_states).reshape(-1)) else np.zeros((0, 4), dtype=np.float32)
    for st in states:
        local = np.asarray(st[:4], dtype=np.float32).copy()
        local[0] -= float(center_x)
        slots.append(slot(local, 0.0))
    while len(slots) < NUM_MASTER_SLOTS:
        slots.append(np.zeros(SLOT_VEC_DIM + 1, dtype=np.float32))
    return np.concatenate(slots[:NUM_MASTER_SLOTS]).astype(np.float32)


def pack_children(child_embs) -> np.ndarray:
    slots = [slot(e, 1.0) for e in list(child_embs)[:NUM_MASTER_SLOTS]]
    while len(slots) < NUM_MASTER_SLOTS:
        slots.append(np.zeros(SLOT_VEC_DIM + 1, dtype=np.float32))
    return np.concatenate(slots[:NUM_MASTER_SLOTS]).astype(np.float32)


def agent_obs(local_state, embedding) -> np.ndarray:
    st = np.asarray(local_state, dtype=np.float32).reshape(-1)[:4]
    emb = pad_vec(embedding, EMBEDDING_DIM)
    return np.concatenate([st, emb]).astype(np.float32)
