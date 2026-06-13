"""
composer_scenario_ui.py - Tkinter GUI for left/right road composition
=====================================================================
Choose which road element appears on the left and right side of the map,
then create scenarios for the resulting combined layout.

Supported module types:
  - empty
  - intersection
  - roundabout

Usage:
    python composer_scenario_ui.py
"""

from __future__ import annotations

import copy
import os
import random
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

sys.path.insert(0, os.path.dirname(__file__))

import my_scenarios
from src.composable_layout import (
    MODULE_TYPE_OPTIONS,
    SLOT_LABEL,
    approach_to_lane_tuple,
    lane_tuple_to_approach,
    layout_title,
    normalize_layout_config,
    visible_approach_labels,
    visible_approaches,
)


COMPOSER_ENV_TYPE = "composable_layout"
COMPOSER_ENV_ID = "RELcomposable-layout-v0"
AGENT_ROW_COUNT = 6
STATIC_ROW_COUNT = 2
ROW_LABEL_WIDTH = 15
APPROACH_COMBO_WIDTH = 8
LAYOUT_COMBO_WIDTH = 13
OFFSET_ENTRY_WIDTH = 6
QUEUE_HEIGHT = 4

AGENT_COLORS_RGB = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 255),
]

AGENT_COLOR_NAMES = ["Red", "Green", "Blue", "Orange", "Cyan", "Magenta"]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def default_row_values(
    layout_config: dict,
    row_index: int,
    *,
    is_static: bool,
) -> tuple[str, str, int]:
    approaches = visible_approaches(layout_config)
    offset_values = [-35, -60] if is_static else [0, -15, -30, -45, -60, -75]

    start = approaches[row_index % len(approaches)]
    dest = approaches[(row_index + 2) % len(approaches)]
    if dest == start:
        dest = approaches[(row_index + 1) % len(approaches)]

    offset = offset_values[row_index % len(offset_values)]
    return start, dest, offset


def normalize_entry(
    entry,
    layout_config: dict,
    row_index: int,
    *,
    is_static: bool,
):
    default_start, default_dest, default_offset = default_row_values(
        layout_config,
        row_index,
        is_static=is_static,
    )
    approaches = set(visible_approaches(layout_config))

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


def normalize_scenario(scenario: dict, layout_config: dict) -> dict:
    scenario = scenario or {}
    agents_in = list(scenario.get("agents", []))
    static_in = list(scenario.get("static", []))

    agents = []
    for i in range(AGENT_ROW_COUNT):
        entry = agents_in[i] if i < len(agents_in) else None
        agents.append(normalize_entry(entry, layout_config, i, is_static=False))

    statics = []
    for i in range(STATIC_ROW_COUNT):
        entry = static_in[i] if i < len(static_in) else None
        statics.append(normalize_entry(entry, layout_config, i, is_static=True))

    return {
        "agents": agents,
        "static": statics,
    }


def build_env_config(layout_config: dict) -> dict:
    layout = normalize_layout_config(layout_config)
    config = copy.deepcopy(my_scenarios.DOUBLE_INTERSECTION_ENV_CONFIG)
    approaches = visible_approaches(layout)

    if layout["left"] != "empty" and layout["right"] != "empty":
        config["screen_width"] = 1500
        config["screen_height"] = 800
        config["scaling"] = 2.6
        config["centering_position"] = [0.5, 0.5]
        config["observation"]["features_range"]["x"] = [-260, 260]
        config["observation"]["features_range"]["y"] = [-120, 120]
    else:
        config["screen_width"] = 1200
        config["screen_height"] = 800
        config["scaling"] = 3.2
        config["centering_position"] = [0.5, 0.55]
        config["observation"]["features_range"]["x"] = [-180, 180]
        config["observation"]["features_range"]["y"] = [-120, 120]

    config["layout_config"] = layout
    config["connector_length"] = 80
    config["single_slot_offset"] = 65
    config["initial_vehicle_count"] = AGENT_ROW_COUNT + STATIC_ROW_COUNT

    controlled_cars = {}
    for i in range(AGENT_ROW_COUNT):
        start, dest, _offset = default_row_values(layout, i, is_static=False)
        controlled_cars[f"agent_{i}"] = {
            "start_lane": approach_to_lane_tuple(start),
            "init_location": {"longitudinal": 32 + i * 12, "lateral": 0},
            "speed": 8,
            "color": AGENT_COLORS_RGB[i],
            "destination": dest,
        }

    static_cars = {}
    for i in range(STATIC_ROW_COUNT):
        start, dest, _offset = default_row_values(layout, i, is_static=True)
        static_cars[f"static_{i}"] = {
            "start_lane": approach_to_lane_tuple(start),
            "init_location": {"longitudinal": 32 + i * 28, "lateral": 0},
            "speed": 8,
            "destination": dest,
        }

    config["controlled_cars"] = controlled_cars
    config["static_cars"] = static_cars
    return config


def activate_runtime_scenarios(scenarios: list[dict]) -> None:
    import src.experiment.scenarios as sc

    sc.composable_base_scenarios = list(scenarios)


def prepare_runtime(scenarios: list[dict]) -> None:
    from highwayenv.utils import patch_intersection_env, register_composable_env

    patch_intersection_env()
    try:
        register_composable_env()
    except Exception:
        pass

    activate_runtime_scenarios(scenarios)


class VehicleRow:
    def __init__(
        self,
        parent,
        row: int,
        label: str,
        approaches: list[str],
        color_hex: str | None = None,
    ):
        self._approaches = list(approaches)

        if color_hex:
            swatch = tk.Label(parent, text="  ", bg=color_hex, relief="solid", width=2)
            swatch.grid(row=row, column=0, padx=(10, 2), pady=4)
        else:
            tk.Label(parent, text="  ", width=2).grid(row=row, column=0, padx=(10, 2), pady=4)

        tk.Label(parent, text=label, anchor="w", width=ROW_LABEL_WIDTH).grid(
            row=row,
            column=1,
            padx=3,
            pady=3,
            sticky="w",
        )

        self.start_var = tk.StringVar(value=self._approaches[0])
        self.start_combo = ttk.Combobox(
            parent,
            textvariable=self.start_var,
            values=self._approaches,
            state="readonly",
            width=APPROACH_COMBO_WIDTH,
        )
        self.start_combo.grid(row=row, column=2, padx=3, pady=3)
        self.start_combo.bind("<<ComboboxSelected>>", self._on_start_changed)

        self.dest_var = tk.StringVar(value=self._approaches[min(1, len(self._approaches) - 1)])
        self.dest_combo = ttk.Combobox(
            parent,
            textvariable=self.dest_var,
            values=self._approaches,
            state="readonly",
            width=APPROACH_COMBO_WIDTH,
        )
        self.dest_combo.grid(row=row, column=3, padx=3, pady=3)

        self.offset_var = tk.StringVar(value="0")
        self.offset_entry = ttk.Entry(parent, textvariable=self.offset_var, width=OFFSET_ENTRY_WIDTH)
        self.offset_entry.grid(row=row, column=4, padx=3, pady=3)

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
        reset_values: bool,
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
        try:
            offset = int(self.offset_var.get())
        except ValueError:
            offset = 0
        return approach_to_lane_tuple(self.start_var.get()), self.dest_var.get(), offset

    def set(self, start_approach: str, dest: str, offset: int) -> None:
        self.start_var.set(start_approach)
        self.dest_var.set(dest)
        self.offset_var.set(str(offset))


class ComposerScenarioBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.layout_config = normalize_layout_config({"left": "roundabout", "right": "intersection"})
        self.scenario_queue: list[dict] = []
        self._current_experiment_name: str | None = None

        root.title("Composable Road Scenario Builder")
        root.resizable(False, False)

        header = ttk.Frame(root, padding=8)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Composable Road Scenario Builder",
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")

        toolbar = ttk.Frame(root, padding=(8, 0, 8, 4))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="New Experiment", command=self._on_new_experiment).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Load Folder", command=self._on_load_folder).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Randomize", command=self._on_randomize).pack(side="left", padx=4)

        layout_frame = ttk.LabelFrame(root, text="Road Layout", padding=8)
        layout_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(layout_frame, text="Left Element").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
        ttk.Label(layout_frame, text="Right Element").grid(row=0, column=2, padx=(12, 6), pady=4, sticky="w")

        self.left_var = tk.StringVar(value=self.layout_config["left"])
        self.right_var = tk.StringVar(value=self.layout_config["right"])

        left_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.left_var,
            values=MODULE_TYPE_OPTIONS,
            state="readonly",
            width=LAYOUT_COMBO_WIDTH,
        )
        left_combo.grid(row=0, column=1, padx=4, pady=4)
        left_combo.bind("<<ComboboxSelected>>", self._on_layout_changed)

        right_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.right_var,
            values=MODULE_TYPE_OPTIONS,
            state="readonly",
            width=LAYOUT_COMBO_WIDTH,
        )
        right_combo.grid(row=0, column=3, padx=4, pady=4)
        right_combo.bind("<<ComboboxSelected>>", self._on_layout_changed)

        self.layout_note_var = tk.StringVar()
        ttk.Label(layout_frame, textvariable=self.layout_note_var, foreground="gray").grid(
            row=1,
            column=0,
            columnspan=4,
            padx=4,
            pady=(6, 0),
            sticky="w",
        )

        grid_frame = ttk.LabelFrame(root, text="Vehicle Placements", padding=8)
        grid_frame.pack(fill="x", padx=8, pady=4)

        headers = ["", "Vehicle", "Start Approach", "Destination", "Offset (m)"]
        for c, text in enumerate(headers):
            ttk.Label(grid_frame, text=text, font=("Segoe UI", 9, "bold")).grid(
                row=0,
                column=c,
                padx=3,
                pady=(0, 5),
            )

        initial_approaches = visible_approaches(self.layout_config)
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

        add_frame = ttk.Frame(root, padding=(8, 4))
        add_frame.pack(fill="x")
        self.add_btn = ttk.Button(add_frame, text="+ Add to Queue", command=self._on_add)
        self.add_btn.pack(side="left", ipadx=8, ipady=3)

        queue_frame = ttk.LabelFrame(root, text="Scenario Queue", padding=8)
        queue_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.queue_listbox = tk.Listbox(queue_frame, height=QUEUE_HEIGHT, font=("Consolas", 9))
        self.queue_listbox.pack(side="left", fill="both", expand=True)

        queue_scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.queue_listbox.yview)
        queue_scroll.pack(side="left", fill="y")
        self.queue_listbox.config(yscrollcommand=queue_scroll.set)

        queue_btn_frame = ttk.Frame(queue_frame, padding=(10, 0, 0, 0))
        queue_btn_frame.pack(side="left", fill="y")
        ttk.Button(queue_btn_frame, text="Remove", command=self._on_remove).pack(pady=2)
        ttk.Button(queue_btn_frame, text="Clear All", command=self._on_clear).pack(pady=2)

        bottom_frame = ttk.Frame(root, padding=8)
        bottom_frame.pack(fill="x")

        self.status_var = tk.StringVar()
        ttk.Label(bottom_frame, textvariable=self.status_var, foreground="gray", wraplength=500).pack(pady=(0, 4))

        btn_row = ttk.Frame(bottom_frame)
        btn_row.pack()
        self.run_btn = ttk.Button(btn_row, text="Run Experiment", command=self._on_run_all)
        self.run_btn.pack(side="left", ipadx=16, ipady=5, padx=(0, 8))

        self.save_btn = ttk.Button(btn_row, text="Save Queue to Folder", command=self._on_save_queue)
        self.save_btn.pack(side="left", ipadx=10, ipady=5)

        self.queue_listbox.bind("<Double-1>", self._on_queue_double_click)
        self._apply_layout(self.layout_config, reset_rows=True)
        self.status_var.set("Choose the left and right road elements, then configure vehicles and queue scenarios.")

    def _compose_layout_from_vars(self) -> dict:
        return normalize_layout_config(
            {
                "left": self.left_var.get(),
                "right": self.right_var.get(),
            }
        )

    def _apply_layout(self, layout_config: dict, *, reset_rows: bool) -> None:
        self.layout_config = normalize_layout_config(layout_config)
        self.left_var.set(self.layout_config["left"])
        self.right_var.set(self.layout_config["right"])

        approaches = visible_approaches(self.layout_config)
        note = (
            f"Layout: {layout_title(self.layout_config)}. "
            f"{len(approaches)} outer approaches available. "
            f"Queue stays at {AGENT_ROW_COUNT} agents and {STATIC_ROW_COUNT} static cars."
        )
        self.layout_note_var.set(note)

        for i, row in enumerate(self.agent_rows):
            start, dest, offset = default_row_values(self.layout_config, i, is_static=False)
            row.configure_choices(approaches, default_start=start, default_dest=dest, reset_values=reset_rows)
            if reset_rows:
                row.offset_var.set(str(offset))

        for i, row in enumerate(self.static_rows):
            start, dest, offset = default_row_values(self.layout_config, i, is_static=True)
            row.configure_choices(approaches, default_start=start, default_dest=dest, reset_values=reset_rows)
            if reset_rows:
                row.offset_var.set(str(offset))

        self._refresh_queue_listbox()

    def _on_layout_changed(self, _event=None):
        new_layout = self._compose_layout_from_vars()
        if new_layout == self.layout_config:
            return

        if self.scenario_queue:
            should_clear = messagebox.askyesno(
                "Change Layout",
                "Changing the layout will clear the current queue because each saved experiment uses one layout.\n\nContinue?",
            )
            if not should_clear:
                self.left_var.set(self.layout_config["left"])
                self.right_var.set(self.layout_config["right"])
                return
            self.scenario_queue.clear()
            self._current_experiment_name = None

        self._apply_layout(new_layout, reset_rows=True)
        self.status_var.set(f"Layout changed to {layout_title(self.layout_config)}. Configure vehicles and add scenarios.")

    def _on_new_experiment(self):
        self.scenario_queue.clear()
        self._current_experiment_name = None
        self._apply_layout(self.layout_config, reset_rows=True)
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

        layout = self.layout_config
        try:
            metadata = load_metadata(folder)
            env_config = metadata.get("env_config", {})
            if metadata.get("env_type") == COMPOSER_ENV_TYPE or "layout_config" in env_config:
                layout = normalize_layout_config(env_config.get("layout_config"))
        except FileNotFoundError:
            pass

        self._apply_layout(layout, reset_rows=False)
        self.scenario_queue = [normalize_scenario(scenario, self.layout_config) for scenario in loaded]
        self._current_experiment_name = os.path.basename(folder)
        self._refresh_queue_listbox()

        if self.scenario_queue:
            self._load_scenario(self.scenario_queue[0])

        self.status_var.set(f"Loaded {len(self.scenario_queue)} scenario(s) from: {self._current_experiment_name}/")

    def _on_queue_double_click(self, _event=None):
        selection = self.queue_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self._load_scenario(self.scenario_queue[index])
        self.status_var.set(f"Editing scenario {index + 1}. Modify it and click Add to re-queue.")

    def _load_scenario(self, scenario: dict) -> None:
        scenario = normalize_scenario(scenario, self.layout_config)

        for i, row in enumerate(self.agent_rows):
            lane, dest, offset = scenario["agents"][i]
            row.set(lane_tuple_to_approach(lane), dest, offset)

        for i, row in enumerate(self.static_rows):
            lane, dest, offset = scenario["static"][i]
            row.set(lane_tuple_to_approach(lane), dest, offset)

    def _on_randomize(self):
        approaches = visible_approaches(self.layout_config)
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
        return normalize_scenario(scenario, self.layout_config)

    def _on_add(self):
        if not self._validate():
            return
        scenario = self._build_scenario()
        self.scenario_queue.append(scenario)
        self._refresh_queue_listbox()
        self.status_var.set(
            f"{len(self.scenario_queue)} scenario(s) queued for layout {layout_title(self.layout_config)}."
        )

    def _on_remove(self):
        selection = self.queue_listbox.curselection()
        if not selection:
            return
        self.scenario_queue.pop(selection[0])
        self._refresh_queue_listbox()
        self.status_var.set(f"{len(self.scenario_queue)} scenario(s) queued.")

    def _on_clear(self):
        self.scenario_queue.clear()
        self._refresh_queue_listbox()
        self.status_var.set("Queue cleared. Add scenarios to the queue.")

    def _refresh_queue_listbox(self):
        self.queue_listbox.delete(0, tk.END)
        for i, scenario in enumerate(self.scenario_queue):
            self.queue_listbox.insert(tk.END, f"  [{i + 1}]  {self._scenario_short_desc(scenario)}")

    def _scenario_short_desc(self, scenario: dict) -> str:
        labels = visible_approach_labels(self.layout_config)
        parts = [layout_title(self.layout_config)]
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

        env_config = build_env_config(self.layout_config)
        save_metadata(
            folder,
            env_type=COMPOSER_ENV_TYPE,
            env_config=env_config,
            scenario_count=len(self.scenario_queue),
        )

        written = save_scenarios_list(
            self.scenario_queue,
            folder,
            prefix="scenario",
            env_type=COMPOSER_ENV_TYPE,
        )
        self.status_var.set(f"Saved {len(written)} scenario(s) to: scenarios/{experiment_name}/")
        messagebox.showinfo("Saved", f"Saved {len(written)} scenario(s) to:\nscenarios/{experiment_name}/")

    def _on_run_all(self):
        if not self.scenario_queue:
            messagebox.showinfo("Empty Queue", "Add at least one scenario to the queue first.")
            return

        try:
            prepare_runtime(self.scenario_queue)
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
                run_simulation(scenario, self.layout_config)
            except Exception as exc:
                messagebox.showerror("Simulation Error", f"Scenario {i + 1} failed:\n{exc}")
                break

        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.add_btn.config(state="normal")
        self.queue_listbox.selection_clear(0, tk.END)
        self.status_var.set(f"Finished running {total} scenario(s).")


def run_simulation(scenario: dict, layout_config: dict) -> None:
    import gymnasium as gym
    import src.project_globals as project_globals

    prepare_runtime([scenario])
    env_config = build_env_config(layout_config)

    project_globals.after_is_arrived_flags = [False] * len(env_config["controlled_cars"])
    project_globals.rollout_buffers = []
    project_globals.episode_count = 0

    env = gym.make(
        COMPOSER_ENV_ID,
        render_mode="human",
        config=env_config,
    )

    _obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0
    step = 0

    print(f"\n{'=' * 50}")
    print(f"  UI Scenario: {layout_title(layout_config)} running...")
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
    app = ComposerScenarioBuilderApp(root)
    root.mainloop()
