from __future__ import annotations

EXTRA_INTERSECTION_6CAR_COUNT = 50
EXTRA_ROUNDABOUT_COUNT = 50
EXTRA_DOUBLE_INTERSECTION_COUNT = 50


def build_extra_intersection_6car() -> list[dict]:
    out: list[dict] = []
    for k in range(EXTRA_INTERSECTION_6CAR_COUNT):
        d = 14 + (k % 4) * 2
        tail = 5 * d + 8 + (k // 4) * 2
        out.append({
            "agents": [
                (('o0', 'ir0', 0), "o3", 0),
                (('o1', 'ir1', 0), "o0", -d),
                (('o2', 'ir2', 0), "o1", -2 * d),
                (('o3', 'ir3', 0), "o2", -3 * d),
                (('o0', 'ir0', 0), "o2", -4 * d - (k % 11)),
                (('o1', 'ir1', 0), "o3", -tail),
            ],
            "static": [],
        })
    return out


def build_extra_roundabout() -> list[dict]:
    out: list[dict] = []
    for k in range(EXTRA_ROUNDABOUT_COUNT):
        s = -18 - (k % 7) * 7
        out.append({
            "agents": [
                (('o0', 'ir0', 0), "o3", 0),
                (('o2', 'ir2', 0), "o1", 0),
                (('o1', 'ir1', 0), "o0", max(-88, s - 12 - k // 3)),
                (('o3', 'ir3', 0), "o2", max(-88, s - 18 - k // 3)),
                (('o0', 'ir0', 0), "o2", max(-90, -48 - k)),
                (('o2', 'ir2', 0), "o0", max(-90, -52 - k)),
            ],
            "static": [],
        })
    return out


def build_extra_double_intersection() -> list[dict]:
    out: list[dict] = []
    for k in range(EXTRA_DOUBLE_INTERSECTION_COUNT):
        gap = -22 - (k % 6) * 4
        tail = max(-78, gap - (k // 6) * 3)
        out.append({
            "agents": [
                (('A_o0', 'A_ir0', 0), "A_o2", 0),
                (('A_o0', 'A_ir0', 0), "A_o1", tail),
                (('A_o2', 'A_ir2', 0), "A_o0", 0),
                (('B_o0', 'B_ir0', 0), "B_o2", 0),
                (('B_o2', 'B_ir2', 0), "B_o0", 0),
                (('B_o3', 'B_ir3', 0), "B_o2", max(-76, -20 - k // 4)),
            ],
            "static": [],
        })
    return out


EXTRA_INTERSECTION_6CAR_50 = build_extra_intersection_6car()
EXTRA_ROUNDABOUT_50 = build_extra_roundabout()
EXTRA_DOUBLE_INTERSECTION_50 = build_extra_double_intersection()
