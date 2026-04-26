"""Scenario registry focused on double-intersection flow.

This project now runs experiments from saved scenario folders and only uses
``double_intersection_base_scenarios`` at runtime.

Legacy names are still exposed as empty lists so old imports do not break.
"""

from src.scenario_io import load_scenarios_from_folder


def get_scenarios(cfg=None, scenario_type: str = "double_intersection") -> list:
    """Return scenarios from folder when configured, else from in-memory list."""
    if cfg is not None and getattr(cfg, "SCENARIOS_FOLDER", ""):
        return load_scenarios_from_folder(cfg.SCENARIOS_FOLDER)

    # Keep compatibility with callers that still pass scenario_type.
    if scenario_type != "double_intersection":
        return []
    return double_intersection_base_scenarios


# ---------------------------------------------------------------------------
# Runtime scenario lists
# ---------------------------------------------------------------------------

# Actively used list for the current project flow.
double_intersection_base_scenarios = []

# Legacy placeholders kept only for import compatibility.
roundabout_base_scenarios = []
base_complete_scenarios_3_cars = []
base_complete_scenarios_2_cars = []
