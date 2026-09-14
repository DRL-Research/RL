from __future__ import annotations

import math

from hdmaps.protocol import FANOUT


def hierarchy_counts(n_local_masters: int, fanout: int = FANOUT) -> tuple[int, int, int]:
    n_lm = int(n_local_masters)
    if n_lm <= 0:
        return 0, 0, 0
    if n_lm == 1:
        return 1, 0, 0
    n_im = 0 if n_lm <= fanout else int(math.ceil(n_lm / fanout))
    return n_lm, n_im, 1


def hierarchy_label(n_local_masters: int, fanout: int = FANOUT) -> str:
    n_lm, n_im, n_gm = hierarchy_counts(n_local_masters, fanout)
    return f"{n_lm} / {n_im} / {n_gm}"


def chunk_indices(n_items: int, fanout: int = FANOUT) -> list[list[int]]:
    if n_items <= 0:
        return []
    return [list(range(i, min(i + fanout, n_items))) for i in range(0, n_items, fanout)]
