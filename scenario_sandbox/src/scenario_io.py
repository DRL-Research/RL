"""
src/scenario_io.py — Scenario persistence utilities
=====================================================
Save scenario dicts to individual JSON files and reload them later so
experiments can be driven entirely from a folder of saved configurations.

Scenario dict format (in Python):
    {
        "agents": [
            (("A_o0", "A_ir0", 0), "B_o3", -15),
            ...
        ],
        "static": [
            (("B_o3", "B_ir3", 0), "B_o2", -55),
            ...
        ]
    }

JSON representation (tuples serialised as arrays):
    {
        "agents": [
            [["A_o0", "A_ir0", 0], "B_o3", -15],
            ...
        ],
        "static": [
            [["B_o3", "B_ir3", 0], "B_o2", -55],
            ...
        ],
        "meta": {
            "env_type": "double_intersection",
            "saved_at": "2026-04-17T12:00:00",
            "description": ""
        }
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scenario_to_json(scenario: dict, env_type: str = "", description: str = "") -> dict:
    """Convert a scenario dict (with tuple lane keys) to a JSON-serialisable dict."""
    def encode_car(entry):
        lane_key, dest, offset = entry
        return [list(lane_key), dest, offset]

    return {
        "agents": [encode_car(e) for e in scenario["agents"]],
        "static": [encode_car(e) for e in scenario["static"]],
        "meta": {
            "env_type": env_type,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "description": description,
        },
    }


def _json_to_scenario(data: dict) -> dict:
    """Convert a JSON-loaded dict back to a scenario dict with tuple lane keys."""
    def decode_car(entry):
        lane_key, dest, offset = entry
        return (tuple(lane_key), dest, int(offset))

    return {
        "agents": [decode_car(e) for e in data["agents"]],
        "static": [decode_car(e) for e in data["static"]],
    }


def _restore_config_types(value: Any, parent_key: str | None = None) -> Any:
    """Restore tuple-backed config fields that JSON turns into lists."""
    if isinstance(value, list):
        if parent_key == "start_lane":
            return tuple(value)
        if parent_key == "color":
            return tuple(value)
        return [_restore_config_types(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _restore_config_types(item, parent_key=key)
            for key, item in value.items()
        }
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_scenario(
    scenario: dict,
    folder: str | os.PathLike,
    filename: str,
    env_type: str = "",
    description: str = "",
) -> str:
    """
    Save a single scenario dict as *filename* inside *folder*.

    Returns the absolute path of the written file.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    filepath = folder / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_scenario_to_json(scenario, env_type, description), f, indent=2)

    return str(filepath.resolve())


def save_scenarios_list(
    scenarios: List[dict],
    folder: str | os.PathLike,
    prefix: str = "scenario",
    env_type: str = "",
) -> List[str]:
    """
    Save a list of scenario dicts to *folder* as individual JSON files.

    Files are named  <prefix>_000.json, <prefix>_001.json, …

    Returns a list of absolute paths for the written files.
    """
    written = []
    for i, sc in enumerate(scenarios):
        filename = f"{prefix}_{i:03d}.json"
        path = save_scenario(sc, folder, filename, env_type=env_type)
        written.append(path)
    return written


def save_metadata(
    folder: str | os.PathLike,
    env_type: str,
    env_config: dict,
    scenario_count: int | None = None,
) -> str:
    """Write metadata.json for a saved scenario folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    payload = {
        "env_type": env_type,
        "env_config": env_config,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if scenario_count is not None:
        payload["scenario_count"] = scenario_count

    filepath = folder / "metadata.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(filepath.resolve())


def load_metadata(folder: str | os.PathLike) -> dict:
    """Load metadata.json from a saved scenario folder."""
    folder = Path(folder)
    filepath = folder / "metadata.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath.resolve()}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "env_config" in data:
        data["env_config"] = _restore_config_types(data["env_config"])
    return data


def load_scenario_file(filepath: str | os.PathLike) -> dict:
    """Load a single scenario JSON file and return the scenario dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _json_to_scenario(data)


def load_scenarios_from_folder(folder: str | os.PathLike) -> List[dict]:
    """
    Load every ``*.json`` file from *folder* (sorted by filename) and return
    a list of scenario dicts ready to be used as a drop-in replacement for
    ``base_complete_scenarios_3_cars`` or any other scenario list.

    Raises FileNotFoundError if the folder does not exist.
    Raises ValueError if the folder contains no JSON files.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Scenario folder not found: {folder.resolve()}")

    json_files = sorted(
        path for path in folder.glob("*.json") if path.name != "metadata.json"
    )
    if not json_files:
        raise ValueError(f"No .json scenario files found in: {folder.resolve()}")

    return [load_scenario_file(p) for p in json_files]
