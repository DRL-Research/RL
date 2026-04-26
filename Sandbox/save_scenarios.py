"""
save_scenarios.py — Export sandbox scenarios to a folder of JSON files
=======================================================================
Run this script manually once you are happy with a set of scenarios
configured in my_scenarios.py.  Each scenario is written as a separate
JSON file so the experiment runner can load them by folder path.

Usage
-----
    # Save all scenario types to their default sub-folders
    python save_scenarios.py

    # Save only one type
    python save_scenarios.py --type double_intersection

    # Save to a custom folder
    python save_scenarios.py --type double_intersection --folder experiments/run_01/scenarios

Options
-------
    --type      intersection | roundabout | double_intersection | all   (default: all)
    --folder    destination folder  (default: saved_scenarios/<type>/)
    --prefix    filename prefix     (default: scenario)
    --dry-run   print what would be saved without writing files
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import my_scenarios
from src.scenario_io import save_scenarios_list


SCENARIO_TYPES = {
    "intersection": {
        "scenarios": my_scenarios.SANDBOX_SCENARIOS,
        "default_folder": "saved_scenarios/intersection",
    },
    "roundabout": {
        "scenarios": my_scenarios.ROUNDABOUT_SCENARIOS,
        "default_folder": "saved_scenarios/roundabout",
    },
    "double_intersection": {
        "scenarios": my_scenarios.DOUBLE_INTERSECTION_SCENARIOS,
        "default_folder": "saved_scenarios/double_intersection",
    },
}


def save_type(type_name: str, folder: str | None, prefix: str, dry_run: bool):
    cfg = SCENARIO_TYPES[type_name]
    scenarios = cfg["scenarios"]
    dest = folder or cfg["default_folder"]

    print(f"\n[{type_name}]  {len(scenarios)} scenario(s)  →  {dest}/")

    if dry_run:
        for i in range(len(scenarios)):
            print(f"  (dry-run) would write  {dest}/{prefix}_{i:03d}.json")
        return

    written = save_scenarios_list(scenarios, dest, prefix=prefix, env_type=type_name)
    for path in written:
        print(f"  saved  {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export my_scenarios.py scenarios to JSON files."
    )
    parser.add_argument(
        "--type",
        choices=list(SCENARIO_TYPES.keys()) + ["all"],
        default="all",
        help="Which scenario type to save (default: all)",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Destination folder (overrides the default per-type folder)",
    )
    parser.add_argument(
        "--prefix",
        default="scenario",
        help="Filename prefix for saved files (default: scenario)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be saved without writing any files",
    )
    args = parser.parse_args()

    types_to_save = list(SCENARIO_TYPES.keys()) if args.type == "all" else [args.type]

    for t in types_to_save:
        # When --folder is given with --type all, it applies to each type separately
        # (the default sub-folder is used unless an explicit folder was given with
        #  a single --type selection).
        folder = args.folder if args.type != "all" else None
        save_type(t, folder, args.prefix, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
