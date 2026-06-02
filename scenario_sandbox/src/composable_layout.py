from __future__ import annotations

MODULE_TYPE_OPTIONS = ["empty", "intersection", "roundabout"]
SLOT_ORDER = ("left", "right")

SLOT_PREFIX = {
    "left": "L_",
    "right": "R_",
}

SLOT_LABEL = {
    "left": "Left",
    "right": "Right",
}

CARDINAL_LABELS = {
    0: "South",
    1: "West",
    2: "North",
    3: "East",
}

CONNECTED_OUTER_CORNER = {
    "left": 3,
    "right": 1,
}


def normalize_layout_config(layout_config: dict | None) -> dict[str, str]:
    layout = {
        "left": "intersection",
        "right": "intersection",
    }
    if layout_config:
        for slot in SLOT_ORDER:
            value = layout_config.get(slot, layout[slot])
            if value in MODULE_TYPE_OPTIONS:
                layout[slot] = value

    if layout["left"] == "empty" and layout["right"] == "empty":
        layout["left"] = "intersection"

    return layout


def slot_prefix(slot: str) -> str:
    return SLOT_PREFIX[slot]


def approach_name(slot: str, corner: int) -> str:
    return f"{slot_prefix(slot)}o{corner}"


def lane_name(slot: str, lane_kind: str, corner: int) -> str:
    return f"{slot_prefix(slot)}{lane_kind}{corner}"


def slot_has_module(layout_config: dict, slot: str) -> bool:
    return normalize_layout_config(layout_config)[slot] != "empty"


def slot_is_connected(layout_config: dict, slot: str) -> bool:
    layout = normalize_layout_config(layout_config)
    if slot == "left":
        return layout["left"] != "empty" and layout["right"] != "empty"
    return layout["left"] != "empty" and layout["right"] != "empty"


def visible_outer_corners(layout_config: dict, slot: str) -> list[int]:
    layout = normalize_layout_config(layout_config)
    if layout[slot] == "empty":
        return []

    corners = [0, 1, 2, 3]
    if slot_is_connected(layout, slot):
        hidden = CONNECTED_OUTER_CORNER[slot]
        corners = [corner for corner in corners if corner != hidden]
    return corners


def visible_approaches(layout_config: dict) -> list[str]:
    layout = normalize_layout_config(layout_config)
    approaches: list[str] = []
    for slot in SLOT_ORDER:
        for corner in visible_outer_corners(layout, slot):
            approaches.append(approach_name(slot, corner))
    return approaches


def visible_approach_labels(layout_config: dict) -> dict[str, str]:
    layout = normalize_layout_config(layout_config)
    labels: dict[str, str] = {}
    for slot in SLOT_ORDER:
        if layout[slot] == "empty":
            continue
        for corner in visible_outer_corners(layout, slot):
            labels[approach_name(slot, corner)] = f"{SLOT_LABEL[slot]} {CARDINAL_LABELS[corner]}"
    return labels


def layout_title(layout_config: dict) -> str:
    layout = normalize_layout_config(layout_config)
    left = layout["left"].title()
    right = layout["right"].title()
    return f"{left} | {right}"


def approach_to_lane_tuple(approach: str) -> tuple[str, str, int]:
    return (approach, approach.replace("_o", "_ir"), 0)


def lane_tuple_to_approach(lane_tuple) -> str:
    return lane_tuple[0]
