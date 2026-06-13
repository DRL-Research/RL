"""Scenario registry focused on saved-scenario runtime flow.

This project now runs experiments from saved scenario folders and only uses
runtime scenario lists at execution time.

Legacy names are still exposed as empty lists so old imports do not break.
"""

from src.scenario_io import load_scenarios_from_folder


def get_scenarios(cfg=None, scenario_type: str = "double_intersection") -> list:
    """Return scenarios from folder when configured, else from in-memory list."""
    if cfg is not None and getattr(cfg, "SCENARIOS_FOLDER", ""):
        return load_scenarios_from_folder(cfg.SCENARIOS_FOLDER)

    if scenario_type == "double_intersection":
        return double_intersection_base_scenarios
    if scenario_type == "roundabout":
        return roundabout_base_scenarios
    if scenario_type in {"intersection", "single_intersection"}:
        return base_complete_scenarios_3_cars
    if scenario_type in {"composable", "composable_layout"}:
        return composable_base_scenarios
    return []


# ---------------------------------------------------------------------------
# Runtime scenario lists
# ---------------------------------------------------------------------------

# Actively used list for the current project flow.
double_intersection_base_scenarios = []
composable_base_scenarios = []

# Legacy placeholders kept only for import compatibility.
roundabout_base_scenarios = []
base_complete_scenarios_3_cars = []
base_complete_scenarios_2_cars = []
