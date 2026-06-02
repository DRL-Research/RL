"""
one_element_scenario_ui.py - Tkinter GUI for one-element scenario building
==========================================================================
Build scenarios for:
  - one intersection
  - roundabout
  - double intersection

The UI intentionally stays close to double_intersection_scenario_ui.py,
but keeps the original builder flow untouched.

Usage:
    python one_element_scenario_ui.py
"""

from __future__ import annotations

import copy
import os
import random
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

# Ensure imports resolve from this project root
sys.path.insert(0, os.path.dirname(__file__))

import my_scenarios


AGENT_ROW_COUNT = 6
STATIC_ROW_COUNT = 2

AGENT_COLORS_RGB = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 255),
]

AGENT_COLOR_NAMES = ["Red", "Green", "Blue", "Orange", "Cyan", "Magenta"]

ENV_DEFINITIONS = {
    "intersection": {
        "label": "One Intersection",
        "header": "One Intersection Scenario Builder",
        "window_title": "One Intersection - Scenario Builder",
        "env_id": "RELintersection-v0",
        "scenario_attr": "base_complete_scenarios_3_cars",
        "approaches": ["o0", "o1", "o2", "o3"],
        "approach_labels": {
            "o0": "South",
            "o1": "West",
            "o2": "North",
            "o3": "East",
        },
    },
    "roundabout": {
        "label": "Roundabout",
        "header": "Roundabout Scenario Builder",
        "window_title": "Roundabout - Scenario Builder",
        "env_id": "RELroundabout-v0",
        "scenario_attr": "roundabout_base_scenarios",
        "approaches": ["o0", "o1", "o2", "o3"],
        "approach_labels": {
            "o0": "South",
            "o1": "West",
            "o2": "North",
            "o3": "East",
        },
    },
    "double_intersection": {
        "label": "Double Intersection",
        "header": "Double Intersection Scenario Builder",
        "window_title": "Double Intersection - Scenario Builder",
        "env_id": "RELdouble-intersection-v0",
        "scenario_attr": "double_intersection_base_scenarios",
        "approaches": ["A_o0", "A_o1", "A_o2", "B_o0", "B_o2", "B_o3"],
        "approach_labels": {
            "A_o0": "A South",
            "A_o1": "A West",
            "A_o2": "A North",
            "B_o0": "B South",
            "B_o2": "B North",
            "B_o3": "B East",
        },
    },
}

ENV_TYPE_OPTIONS = list(ENV_DEFINITIONS.keys())


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def approach_to_lane_tuple(approach: str) -> tuple[str, str, int]:
    if "_o" in approach:
        return (approach, approach.replace("_o", "_ir"), 0)
    return (approach, approach.replace("o", "ir", 1), 0)


def lane_tuple_to_approach(lane_tuple) -> str:
    return lane_tuple[0]


def get_env_definition(env_type: str) -> dict:
    return ENV_DEFINITIONS[env_type]


def default_row_values(
    env_type: str,
    row_index: int,
    *,
    is_static: bool,
) -> tuple[str, str, int]:
    approaches = get_env_definition(env_type)["approaches"]
    offset_values = [-35, -60] if is_static else [0, -15, -30, -45, -60, -75]

    if is_static:
        start = approaches[(row_index + len(approaches) - 1) % len(approaches)]
        dest = approaches[(row_index + 1) % len(approaches)]
    else:
        start = approaches[row_index % len(approaches)]
        dest = approaches[(row_index + 2) % len(approaches)]

    if dest == start:
        dest = approaches[(approaches.index(start) + 1) % len(approaches)]

    offset = offset_values[row_index % len(offset_values)]
    return start, dest, offset


def normalize_entry(
    entry,
    env_type: str,
    row_index: int,
    *,
    is_static: bool,
):
    default_start, default_dest, default_offset = default_row_values(
        env_type,
        row_index,
        is_static=is_static,
    )
    approaches = set(get_env_definition(env_type)["approaches"])

    if not entry or len(entry) != 3:
        start = default_start
        dest = default_dest
        offset = default_offset
    else:
        lane_key, dest, offset = entry
        start = lane_tuple_to_approach(lane_key)
        if start not in approaches:
            start = default_start
        if dest not in approaches or dest == start:
            dest = default_dest
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            offset = default_offset

    return (approach_to_lane_tuple(start), dest, offset)


def normalize_scenario(scenario: dict, env_type: str) -> dict:
    scenario = scenario or {}
    agents_in = list(scenario.get("agents", []))
    static_in = list(scenario.get("static", []))

    agents = []
    for i in range(AGENT_ROW_COUNT):
        source = agents_in[i] if i < len(agents_in) else None
        agents.append(normalize_entry(source, env_type, i, is_static=False))

    statics = []
    for i in range(STATIC_ROW_COUNT):
        source = static_in[i] if i < len(static_in) else None
        statics.append(normalize_entry(source, env_type, i, is_static=True))

    return {
        "agents": agents,
        "static": statics,
    }


def build_env_config(env_type: str) -> dict:
    base_config_map = {
        "intersection": my_scenarios.SANDBOX_ENV_CONFIG,
        "roundabout": my_scenarios.ROUNDABOUT_ENV_CONFIG,
        "double_intersection": my_scenarios.DOUBLE_INTERSECTION_ENV_CONFIG,
    }

    config = copy.deepcopy(base_config_map[env_type])
    approaches = get_env_definition(env_type)["approaches"]
    agent_longitudes = [20, 32, 44, 56, 72, 90]
    static_longitudes = [28, 68]

    controlled_cars = {}
    for i in range(AGENT_ROW_COUNT):
        start, dest, _ = default_row_values(env_type, i, is_static=False)
        controlled_cars[f"agent_{i}"] = {
            "start_lane": approach_to_lane_tuple(start),
            "init_location": {
                "longitudinal": agent_longitudes[i],
                "lateral": 0,
            },
            "speed": 8,
            "color": AGENT_COLORS_RGB[i],
            "destination": dest,
        }

    static_cars = {}
    for i in range(STATIC_ROW_COUNT):
        start, dest, _ = default_row_values(env_type, i, is_static=True)
        static_cars[f"static_{i}"] = {
            "start_lane": approach_to_lane_tuple(start),
            "init_location": {
                "longitudinal": static_longitudes[i],
                "lateral": 0,
            },
            "speed": 8,
            "destination": dest,
        }

    config["controlled_cars"] = controlled_cars
    config["static_cars"] = static_cars
    config["initial_vehicle_count"] = AGENT_ROW_COUNT + STATIC_ROW_COUNT

    if env_type in {"intersection", "roundabout"}:
        x_range = 120 if env_type == "intersection" else 110
        config["observation"]["features_range"]["x"] = [-x_range, x_range]
        config["observation"]["features_range"]["y"] = [-x_range, x_range]

    return config


def activate_runtime_scenarios(env_type: str, scenarios: list[dict]) -> None:
    import src.experiment.scenarios as sc

    sc.base_complete_scenarios_3_cars = []
    sc.roundabout_base_scenarios = []
    sc.double_intersection_base_scenarios = []

    setattr(sc, get_env_definition(env_type)["scenario_attr"], list(scenarios))


def prepare_runtime(env_type: str, scenarios: list[dict]) -> None:
    from highwayenv.utils import (
        patch_intersection_env,
        register_double_intersection_env,
        register_intersection_env,
        register_roundabout_env,
    )

    patch_intersection_env()

    register_map = {
        "intersection": register_intersection_env,
        "roundabout": register_roundabout_env,
        "double_intersection": register_double_intersection_env,
    }

    try:
        register_map[env_type]()
    except Exception:
        pass

    activate_runtime_scenarios(env_type, scenarios)


class VehicleRow:
    """One row of the UI: label + start dropdown + destination dropdown + offset."""

    def __init__(
        self,
        parent,
        row: int,
        label: str,
        approaches: list[str],
        color_hex: str | None = None,
    ):
        self.label_text = label
        self._approaches = list(approaches)

        if color_hex:
            swatch = tk.Label(parent, text="  ", bg=color_hex, relief="solid", width=2)
            swatch.grid(row=row, column=0, padx=(10, 2), pady=4)
        else:
            tk.Label(parent, text="  ", width=2).grid(row=row, column=0, padx=(10, 2), pady=4)

        tk.Label(parent, text=label, anchor="w", width=18).grid(
            row=row,
            column=1,
            padx=4,
            pady=4,
            sticky="w",
        )

        self.start_var = tk.StringVar(value=self._approaches[0])
        self.start_combo = ttk.Combobox(
            parent,
            textvariable=self.start_var,
            values=self._approaches,
            state="readonly",
            width=10,
        )
        self.start_combo.grid(row=row, column=2, padx=4, pady=4)
        self.start_combo.bind("<<ComboboxSelected>>", self._on_start_changed)

        self.dest_var = tk.StringVar(value=self._approaches[min(1, len(self._approaches) - 1)])
        self.dest_combo = ttk.Combobox(
            parent,
            textvariable=self.dest_var,
            values=self._approaches,
            state="readonly",
            width=10,
        )
        self.dest_combo.grid(row=row, column=3, padx=4, pady=4)

        self.offset_var = tk.StringVar(value="0")
        self.offset_entry = ttk.Entry(parent, textvariable=self.offset_var, width=8)
        self.offset_entry.grid(row=row, column=4, padx=4, pady=4)

    def _on_start_changed(self, _event=None):
        if self.dest_var.get() == self.start_var.get():
            for candidate in self._approaches:
                if candidate != self.start_var.get():
                    self.dest_var.set(candidate)
                    break

    def configure_choices(
        self,
        approaches: list[str],
        *,
        default_start: str,
        default_dest: str,
        reset_values: bool = False,
    ) -> None:
        previous_start = self.start_var.get()
        previous_dest = self.dest_var.get()
        self._approaches = list(approaches)

        self.start_combo["values"] = self._approaches
        self.dest_combo["values"] = self._approaches

        if reset_values or previous_start not in self._approaches:
            self.start_var.set(default_start)
        else:
            self.start_var.set(previous_start)

        if reset_values or previous_dest not in self._approaches or previous_dest == self.start_var.get():
            self.dest_var.set(default_dest)
        else:
            self.dest_var.set(previous_dest)

    def get(self):
        start = self.start_var.get()
        dest = self.dest_var.get()
        try:
            offset = int(self.offset_var.get())
        except ValueError:
            offset = 0
        return approach_to_lane_tuple(start), dest, offset

    def set(self, start_approach: str, dest: str, offset: int):
        self.start_var.set(start_approach)
        self.dest_var.set(dest)
        self.offset_var.set(str(offset))


class ScenarioBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_env_type = "double_intersection"
        self.scenario_queue: list[dict] = []
        self._current_experiment_name: str | None = None

        definition = get_env_definition(self.current_env_type)
        self.root.title(definition["window_title"])
        self.root.resizable(False, False)

        header = ttk.Frame(root, padding=10)
        header.pack(fill="x")
        self.header_var = tk.StringVar(value=definition["header"])
        ttk.Label(
            header,
            textvariable=self.header_var,
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left")

        toolbar = ttk.Frame(root, padding=(10, 0, 10, 5))
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="New Experiment", command=self._on_new_experiment).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Load Experiment Folder", command=self._on_load_folder).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Randomize", command=self._on_randomize).pack(side="left", padx=4)

        options_frame = ttk.LabelFrame(root, text="Scenario Type", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(options_frame, text="Road Layout").pack(side="left")
        self.env_type_var = tk.StringVar(value=self.current_env_type)
        env_combo = ttk.Combobox(
            options_frame,
            textvariable=self.env_type_var,
            values=ENV_TYPE_OPTIONS,
            state="readonly",
            width=22,
        )
        env_combo.pack(side="left", padx=(8, 12))
        env_combo.bind("<<ComboboxSelected>>", self._on_env_type_selected)

        self.layout_note_var = tk.StringVar()
        ttk.Label(
            options_frame,
            textvariable=self.layout_note_var,
            foreground="gray",
        ).pack(side="left")

        grid_frame = ttk.LabelFrame(root, text="Vehicle Placements", padding=10)
        grid_frame.pack(fill="x", padx=10, pady=5)

        headers = ["", "Vehicle", "Start Approach", "Destination", "Offset (m)"]
        for c, text in enumerate(headers):
            ttk.Label(grid_frame, text=text, font=("Segoe UI", 9, "bold")).grid(
                row=0,
                column=c,
                padx=4,
                pady=(0, 6),
            )

        initial_approaches = definition["approaches"]
        self.agent_rows: list[VehicleRow] = []
        for i in range(AGENT_ROW_COUNT):
            row = VehicleRow(
                grid_frame,
                row=i + 1,
                label=f"Agent {i} ({AGENT_COLOR_NAMES[i]})",
                approaches=initial_approaches,
                color_hex=rgb_to_hex(*AGENT_COLORS_RGB[i]),
            )
            self.agent_rows.append(row)

        ttk.Separator(grid_frame, orient="horizontal").grid(
            row=AGENT_ROW_COUNT + 1,
            column=0,
            columnspan=5,
            sticky="ew",
            pady=6,
        )

        self.static_rows: list[VehicleRow] = []
        for i in range(STATIC_ROW_COUNT):
            row = VehicleRow(
                grid_frame,
                row=AGENT_ROW_COUNT + 2 + i,
                label=f"Static {i}",
                approaches=initial_approaches,
            )
            self.static_rows.append(row)

        add_frame = ttk.Frame(root, padding=(10, 5))
        add_frame.pack(fill="x")
        self.add_btn = ttk.Button(add_frame, text="+ Add Scenario to Queue", command=self._on_add)
        self.add_btn.pack(side="left", ipadx=10, ipady=4)

        queue_frame = ttk.LabelFrame(root, text="Scenario Queue", padding=10)
        queue_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.queue_listbox = tk.Listbox(queue_frame, height=6, font=("Consolas", 9))
        self.queue_listbox.pack(side="left", fill="both", expand=True)

        queue_scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.queue_listbox.yview)
        queue_scroll.pack(side="left", fill="y")
        self.queue_listbox.config(yscrollcommand=queue_scroll.set)

        queue_btn_frame = ttk.Frame(queue_frame, padding=(10, 0, 0, 0))
        queue_btn_frame.pack(side="left", fill="y")
        ttk.Button(queue_btn_frame, text="Remove", command=self._on_remove).pack(pady=2)
        ttk.Button(queue_btn_frame, text="Clear All", command=self._on_clear).pack(pady=2)

        bottom_frame = ttk.Frame(root, padding=10)
        bottom_frame.pack(fill="x")

        self.status_var = tk.StringVar()
        ttk.Label(
            bottom_frame,
            textvariable=self.status_var,
            foreground="gray",
            wraplength=560,
        ).pack(pady=(0, 5))

        btn_row = ttk.Frame(bottom_frame)
        btn_row.pack()

        self.run_btn = ttk.Button(btn_row, text="Run Experiment", command=self._on_run_all)
        self.run_btn.pack(side="left", ipadx=20, ipady=6, padx=(0, 8))

        self.save_btn = ttk.Button(btn_row, text="Save Queue to Folder", command=self._on_save_queue)
        self.save_btn.pack(side="left", ipadx=12, ipady=6)

        self.queue_listbox.bind("<Double-1>", self._on_queue_double_click)
        self._apply_env_type(self.current_env_type, reset_rows=True)
        self.status_var.set("Choose a layout, configure vehicles, then add scenarios to the queue.")

    def _apply_env_type(self, env_type: str, *, reset_rows: bool) -> None:
        self.current_env_type = env_type
        definition = get_env_definition(env_type)
        approaches = definition["approaches"]

        self.root.title(definition["window_title"])
        self.header_var.set(definition["header"])
        self.layout_note_var.set(
            f"{len(approaches)} approaches available. Queue stays at {AGENT_ROW_COUNT} agents and {STATIC_ROW_COUNT} static cars."
        )

        for i, row in enumerate(self.agent_rows):
            start, dest, offset = default_row_values(env_type, i, is_static=False)
            row.configure_choices(
                approaches,
                default_start=start,
                default_dest=dest,
                reset_values=reset_rows,
            )
            if reset_rows:
                row.offset_var.set(str(offset))

        for i, row in enumerate(self.static_rows):
            start, dest, offset = default_row_values(env_type, i, is_static=True)
            row.configure_choices(
                approaches,
                default_start=start,
                default_dest=dest,
                reset_values=reset_rows,
            )
            if reset_rows:
                row.offset_var.set(str(offset))

        self._refresh_queue_listbox()

    def _on_env_type_selected(self, _event=None):
        new_env_type = self.env_type_var.get()
        if new_env_type == self.current_env_type:
            return

        if self.scenario_queue:
            should_clear = messagebox.askyesno(
                "Change Layout",
                "Changing the road layout will clear the current queue because each saved experiment uses one env_type.\n\nContinue?",
            )
            if not should_clear:
                self.env_type_var.set(self.current_env_type)
                return
            self.scenario_queue.clear()
            self._current_experiment_name = None

        self._apply_env_type(new_env_type, reset_rows=True)
        self.status_var.set(
            f"Layout changed to {get_env_definition(new_env_type)['label']}. Configure vehicles and add scenarios."
        )

    def _on_new_experiment(self):
        self.scenario_queue.clear()
        self._current_experiment_name = None
        self._apply_env_type(self.current_env_type, reset_rows=True)
        self.status_var.set("New experiment. Configure vehicles and add scenarios.")

    def _on_load_folder(self):
        from src.scenario_io import load_metadata, load_scenarios_from_folder

        project_root = os.path.dirname(os.path.abspath(__file__))
        scenarios_dir = os.path.join(project_root, "scenarios")

        folder = filedialog.askdirectory(
            title="Select an experiment folder",
            initialdir=scenarios_dir if os.path.isdir(scenarios_dir) else project_root,
        )
        if not folder:
            return

        try:
            loaded = load_scenarios_from_folder(folder)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        env_type = self.current_env_type
        try:
            metadata = load_metadata(folder)
            maybe_env_type = metadata.get("env_type", env_type)
            if maybe_env_type in ENV_DEFINITIONS:
                env_type = maybe_env_type
        except FileNotFoundError:
            pass

        self.env_type_var.set(env_type)
        self._apply_env_type(env_type, reset_rows=False)

        self.scenario_queue = [normalize_scenario(scenario, env_type) for scenario in loaded]
        self._current_experiment_name = os.path.basename(folder)
        self._refresh_queue_listbox()

        if self.scenario_queue:
            self._load_scenario(self.scenario_queue[0])

        self.status_var.set(
            f"Loaded {len(self.scenario_queue)} scenario(s) from: {self._current_experiment_name}/"
        )

    def _on_queue_double_click(self, _event=None):
        selection = self.queue_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        self._load_scenario(self.scenario_queue[index])
        self.status_var.set(f"Editing scenario {index + 1}. Modify it and click Add to re-queue.")

    def _load_scenario(self, scenario: dict):
        scenario = normalize_scenario(scenario, self.current_env_type)

        for i, row in enumerate(self.agent_rows):
            lane, dest, offset = scenario["agents"][i]
            row.set(lane_tuple_to_approach(lane), dest, offset)

        for i, row in enumerate(self.static_rows):
            lane, dest, offset = scenario["static"][i]
            row.set(lane_tuple_to_approach(lane), dest, offset)

    def _on_randomize(self):
        approaches = get_env_definition(self.current_env_type)["approaches"]
        offsets = [0, -10, -15, -20, -25, -30, -35, -40, -50, -60, -70]

        for row in self.agent_rows + self.static_rows:
            start = random.choice(approaches)
            possible_dests = [candidate for candidate in approaches if candidate != start]
            row.set(start, random.choice(possible_dests), random.choice(offsets))

    def _build_scenario(self) -> dict:
        scenario = {
            "agents": [row.get() for row in self.agent_rows],
            "static": [row.get() for row in self.static_rows],
        }
        return normalize_scenario(scenario, self.current_env_type)

    def _on_add(self):
        if not self._validate():
            return

        scenario = self._build_scenario()
        self.scenario_queue.append(scenario)
        self._refresh_queue_listbox()
        self.status_var.set(
            f"{len(self.scenario_queue)} scenario(s) queued for {get_env_definition(self.current_env_type)['label']}."
        )

    def _on_remove(self):
        selection = self.queue_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        self.scenario_queue.pop(index)
        self._refresh_queue_listbox()
        self.status_var.set(f"{len(self.scenario_queue)} scenario(s) queued.")

    def _on_clear(self):
        self.scenario_queue.clear()
        self._refresh_queue_listbox()
        self.status_var.set("Queue cleared. Add scenarios to the queue.")

    def _refresh_queue_listbox(self):
        self.queue_listbox.delete(0, tk.END)
        for i, scenario in enumerate(self.scenario_queue):
            description = self._scenario_short_desc(scenario)
            self.queue_listbox.insert(tk.END, f"  [{i + 1}]  {description}")

    def _scenario_short_desc(self, scenario: dict) -> str:
        labels = get_env_definition(self.current_env_type)["approach_labels"]
        parts = [get_env_definition(self.current_env_type)["label"]]
        for index, (lane, dest, _offset) in enumerate(scenario["agents"]):
            start_text = labels.get(lane[0], lane[0])
            dest_text = labels.get(dest, dest)
            parts.append(f"A{index}:{start_text}->{dest_text}")
        return "  ".join(parts)

    def _validate(self) -> bool:
        all_rows = list(enumerate(self.agent_rows)) + [
            (AGENT_ROW_COUNT + i, row) for i, row in enumerate(self.static_rows)
        ]

        placements = []
        for index, row in all_rows:
            lane, dest, offset = row.get()
            start = lane[0]

            if start == dest:
                label = f"Agent {index}" if index < AGENT_ROW_COUNT else f"Static {index - AGENT_ROW_COUNT}"
                messagebox.showwarning(
                    "Invalid Placement",
                    f"{label}: start and destination are the same ({start}).",
                )
                return False

            placements.append((start, offset, index))

        placements.sort(key=lambda item: (item[0], item[1]))
        for i in range(len(placements) - 1):
            current_start, current_offset, current_index = placements[i]
            next_start, next_offset, next_index = placements[i + 1]

            if current_start != next_start:
                continue

            gap = abs(current_offset - next_offset)
            if gap >= 10:
                continue

            current_label = (
                f"Agent {current_index}"
                if current_index < AGENT_ROW_COUNT
                else f"Static {current_index - AGENT_ROW_COUNT}"
            )
            next_label = (
                f"Agent {next_index}"
                if next_index < AGENT_ROW_COUNT
                else f"Static {next_index - AGENT_ROW_COUNT}"
            )

            should_continue = messagebox.askyesno(
                "Potential Collision",
                f"{current_label} and {next_label} start on the same approach "
                f"({current_start}) with only {gap}m offset gap.\n\nContinue anyway?",
            )
            if not should_continue:
                return False

        return True

    def _on_save_queue(self):
        from src.scenario_io import save_metadata, save_scenarios_list

        if not self.scenario_queue:
            messagebox.showinfo("Empty Queue", "Add at least one scenario to the queue first.")
            return

        experiment_name = simpledialog.askstring(
            "Experiment Name",
            "Enter a name for this experiment:",
            parent=self.root,
            initialvalue=self._current_experiment_name or "",
        )
        if not experiment_name or not experiment_name.strip():
            return

        experiment_name = experiment_name.strip().replace(" ", "_")
        self._current_experiment_name = experiment_name

        project_root = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(project_root, "scenarios", experiment_name)

        env_type = self.current_env_type
        env_config = build_env_config(env_type)

        save_metadata(
            folder,
            env_type=env_type,
            env_config=env_config,
            scenario_count=len(self.scenario_queue),
        )

        written = save_scenarios_list(
            self.scenario_queue,
            folder,
            prefix="scenario",
            env_type=env_type,
        )
        self.status_var.set(f"Saved {len(written)} scenario(s) to: scenarios/{experiment_name}/")
        messagebox.showinfo(
            "Saved",
            f"Saved {len(written)} scenario(s) to:\nscenarios/{experiment_name}/",
        )

    def _on_run_all(self):
        if not self.scenario_queue:
            messagebox.showinfo("Empty Queue", "Add at least one scenario to the queue first.")
            return

        env_type = self.current_env_type

        try:
            prepare_runtime(env_type, self.scenario_queue)
        except Exception as exc:
            messagebox.showerror("Runtime Setup Error", str(exc))
            return

        total = len(self.scenario_queue)
        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.add_btn.config(state="disabled")

        for i, scenario in enumerate(self.scenario_queue):
            self.queue_listbox.selection_clear(0, tk.END)
            self.queue_listbox.selection_set(i)
            self.queue_listbox.see(i)
            self.status_var.set(f"Running scenario {i + 1}/{total}... close the pygame window for the next run.")
            self.root.update()

            try:
                run_simulation(scenario, env_type)
            except Exception as exc:
                messagebox.showerror("Simulation Error", f"Scenario {i + 1} failed:\n{exc}")
                break

        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.add_btn.config(state="normal")
        self.queue_listbox.selection_clear(0, tk.END)
        self.status_var.set(f"Finished running {total} scenario(s).")


def run_simulation(scenario: dict, env_type: str) -> None:
    import gymnasium as gym
    import src.project_globals as project_globals

    prepare_runtime(env_type, [scenario])
    env_config = build_env_config(env_type)

    project_globals.after_is_arrived_flags = [False] * len(env_config["controlled_cars"])
    project_globals.rollout_buffers = []
    project_globals.episode_count = 0

    env = gym.make(
        get_env_definition(env_type)["env_id"],
        render_mode="human",
        config=env_config,
    )

    _obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0
    step = 0

    print(f"\n{'=' * 50}")
    print(f"  UI Scenario: {get_env_definition(env_type)['label']} running...")
    print(f"{'=' * 50}")

    while not (terminated or truncated):
        num_agents = len(env.unwrapped.controlled_vehicles)
        action = tuple(1 for _ in range(num_agents))
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += sum(reward) if hasattr(reward, "__iter__") else reward
        step += 1

    env.close()
    print(f"  Finished after {step} steps | total reward: {total_reward:.1f}")
    print(f"  crashed={info.get('crashed', '?')}  agents_arrived={info.get('agents_dones', '?')}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ScenarioBuilderApp(root)
    root.mainloop()
