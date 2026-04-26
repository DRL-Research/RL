"""
scenario_ui.py — Tkinter GUI for Double Intersection Scenario Builder
======================================================================
Configure all 8 vehicle placements (6 agents + 2 static), then click
"Run Scenario" to launch the simulation in a pygame window.

Usage:
    python scenario_ui.py
"""

import sys
import os
import random
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

# Ensure imports resolve from this project root
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid approach nodes for the double intersection
APPROACHES = ["A_o0", "A_o1", "A_o2", "B_o0", "B_o2", "B_o3"]

# Map approach name → lane tuple used in scenario dicts
APPROACH_TO_LANE = {
    "A_o0": ("A_o0", "A_ir0", 0),
    "A_o1": ("A_o1", "A_ir1", 0),
    "A_o2": ("A_o2", "A_ir2", 0),
    "B_o0": ("B_o0", "B_ir0", 0),
    "B_o2": ("B_o2", "B_ir2", 0),
    "B_o3": ("B_o3", "B_ir3", 0),
}

# Friendly labels for approaches
APPROACH_LABELS = {
    "A_o0": "A South",
    "A_o1": "A West",
    "A_o2": "A North",
    "B_o0": "B South",
    "B_o2": "B North",
    "B_o3": "B East",
}

# Agent colors (RGB) matching DOUBLE_INTERSECTION_ENV_CONFIG
AGENT_COLORS_RGB = [
    (255, 100, 100),  # agent 0 — red
    (100, 255, 100),  # agent 1 — green
    (100, 100, 255),  # agent 2 — blue
    (255, 165,   0),  # agent 3 — orange
    (  0, 255, 255),  # agent 4 — cyan
    (255,   0, 255),  # agent 5 — magenta
]

AGENT_COLOR_NAMES = ["Red", "Green", "Blue", "Orange", "Cyan", "Magenta"]


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


import my_scenarios


# ---------------------------------------------------------------------------
# Vehicle row widget
# ---------------------------------------------------------------------------

class VehicleRow:
    """One row of the UI: label + start dropdown + destination dropdown + offset."""

    def __init__(self, parent, row: int, label: str, color_hex: str | None = None):
        self.label_text = label

        # Color swatch
        if color_hex:
            swatch = tk.Label(parent, text="  ", bg=color_hex, relief="solid", width=2)
            swatch.grid(row=row, column=0, padx=(10, 2), pady=4)
        else:
            tk.Label(parent, text="  ", width=2).grid(row=row, column=0, padx=(10, 2), pady=4)

        # Label
        tk.Label(parent, text=label, anchor="w", width=12).grid(
            row=row, column=1, padx=4, pady=4, sticky="w"
        )

        # Start approach
        self.start_var = tk.StringVar(value=APPROACHES[0])
        self.start_combo = ttk.Combobox(
            parent, textvariable=self.start_var, values=APPROACHES,
            state="readonly", width=10
        )
        self.start_combo.grid(row=row, column=2, padx=4, pady=4)

        # Destination
        self.dest_var = tk.StringVar(value=APPROACHES[2])
        self.dest_combo = ttk.Combobox(
            parent, textvariable=self.dest_var, values=APPROACHES,
            state="readonly", width=10
        )
        self.dest_combo.grid(row=row, column=3, padx=4, pady=4)

        # Offset
        self.offset_var = tk.StringVar(value="0")
        self.offset_entry = ttk.Entry(parent, textvariable=self.offset_var, width=8)
        self.offset_entry.grid(row=row, column=4, padx=4, pady=4)

    def get(self):
        """Return (lane_tuple, destination_str, offset_int)."""
        start = self.start_var.get()
        dest = self.dest_var.get()
        try:
            offset = int(self.offset_var.get())
        except ValueError:
            offset = 0
        return APPROACH_TO_LANE[start], dest, offset

    def set(self, start_approach: str, dest: str, offset: int):
        self.start_var.set(start_approach)
        self.dest_var.set(dest)
        self.offset_var.set(str(offset))


def lane_tuple_to_approach(lane_tuple) -> str:
    """('A_o0', 'A_ir0', 0) → 'A_o0'"""
    return lane_tuple[0]


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

class ScenarioBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Double Intersection — Scenario Builder")
        root.resizable(False, False)

        # ── Header ──
        header = ttk.Frame(root, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Double Intersection Scenario Builder",
                  font=("Segoe UI", 14, "bold")).pack(side="left")

        # ── Toolbar ──
        toolbar = ttk.Frame(root, padding=(10, 0, 10, 5))
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="New Experiment",
                   command=self._on_new_experiment).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Load Experiment Folder",
                   command=self._on_load_folder).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Randomize",
                   command=self._on_randomize).pack(side="left", padx=4)

        # Track which experiment folder is currently loaded (None = new)
        self._current_experiment_name: str | None = None

        # ── Vehicle grid ──
        grid_frame = ttk.LabelFrame(root, text="Vehicle Placements", padding=10)
        grid_frame.pack(fill="x", padx=10, pady=5)

        # Column headers
        headers = ["", "Vehicle", "Start Approach", "Destination", "Offset (m)"]
        for c, h in enumerate(headers):
            ttk.Label(grid_frame, text=h, font=("Segoe UI", 9, "bold")).grid(
                row=0, column=c, padx=4, pady=(0, 6)
            )

        # Agent rows
        self.agent_rows: list[VehicleRow] = []
        for i in range(6):
            hex_color = rgb_to_hex(*AGENT_COLORS_RGB[i])
            vr = VehicleRow(grid_frame, row=i + 1,
                            label=f"Agent {i} ({AGENT_COLOR_NAMES[i]})",
                            color_hex=hex_color)
            self.agent_rows.append(vr)

        # Separator
        ttk.Separator(grid_frame, orient="horizontal").grid(
            row=7, column=0, columnspan=5, sticky="ew", pady=6
        )

        # Static rows
        self.static_rows: list[VehicleRow] = []
        for i in range(2):
            vr = VehicleRow(grid_frame, row=8 + i, label=f"Static {i}")
            self.static_rows.append(vr)

        # ── Add to queue button ──
        add_frame = ttk.Frame(root, padding=(10, 5))
        add_frame.pack(fill="x")
        self.add_btn = ttk.Button(add_frame, text="+ Add Scenario to Queue",
                                  command=self._on_add)
        self.add_btn.pack(side="left", ipadx=10, ipady=4)

        # ── Scenario queue ──
        queue_frame = ttk.LabelFrame(root, text="Scenario Queue", padding=10)
        queue_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.queue_listbox = tk.Listbox(queue_frame, height=6, font=("Consolas", 9))
        self.queue_listbox.pack(side="left", fill="both", expand=True)

        queue_scroll = ttk.Scrollbar(queue_frame, orient="vertical",
                                     command=self.queue_listbox.yview)
        queue_scroll.pack(side="left", fill="y")
        self.queue_listbox.config(yscrollcommand=queue_scroll.set)

        queue_btn_frame = ttk.Frame(queue_frame, padding=(10, 0, 0, 0))
        queue_btn_frame.pack(side="left", fill="y")
        ttk.Button(queue_btn_frame, text="Remove", command=self._on_remove).pack(pady=2)
        ttk.Button(queue_btn_frame, text="Clear All", command=self._on_clear).pack(pady=2)

        # ── Status + Run All button ──
        bottom_frame = ttk.Frame(root, padding=10)
        bottom_frame.pack(fill="x")

        self.status_var = tk.StringVar(value="Add scenarios to the queue, then click Run Experiment.")
        ttk.Label(bottom_frame, textvariable=self.status_var, foreground="gray",
                  wraplength=500).pack(pady=(0, 5))

        btn_row = ttk.Frame(bottom_frame)
        btn_row.pack()

        self.run_btn = ttk.Button(btn_row, text="▶  Run Experiment",
                                  command=self._on_run_all)
        self.run_btn.pack(side="left", ipadx=20, ipady=6, padx=(0, 8))

        self.save_btn = ttk.Button(btn_row, text="💾  Save Queue to Folder",
                                   command=self._on_save_queue)
        self.save_btn.pack(side="left", ipadx=12, ipady=6)

        # Internal queue storage
        self.scenario_queue: list[dict] = []

        # Double-click a queued scenario to load it into the editor
        self.queue_listbox.bind("<Double-1>", self._on_queue_double_click)

    # ── New experiment ──────────────────────────────────────────────────

    def _on_new_experiment(self):
        self.scenario_queue.clear()
        self._current_experiment_name = None
        self._refresh_queue_listbox()
        # Reset all vehicle rows to defaults
        for vr in self.agent_rows + self.static_rows:
            vr.set(APPROACHES[0], APPROACHES[2], 0)
        self.status_var.set("New experiment. Configure vehicles and add scenarios.")

    # ── Load experiment folder ──────────────────────────────────────────

    def _on_load_folder(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        scenarios_dir = os.path.join(project_root, "scenarios")

        folder = filedialog.askdirectory(
            title="Select an experiment folder",
            initialdir=scenarios_dir if os.path.isdir(scenarios_dir) else project_root,
        )
        if not folder:
            return

        from src.scenario_io import load_scenarios_from_folder

        try:
            loaded = load_scenarios_from_folder(folder)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Load Error", str(e))
            return

        self.scenario_queue = loaded
        self._current_experiment_name = os.path.basename(folder)
        self._refresh_queue_listbox()

        # Load the first scenario into the editor
        if loaded:
            self._load_scenario(loaded[0])

        self.status_var.set(
            f"Loaded {len(loaded)} scenario(s) from: {self._current_experiment_name}/"
        )

    # ── Load scenario into editor ───────────────────────────────────────

    def _on_queue_double_click(self, _event=None):
        sel = self.queue_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._load_scenario(self.scenario_queue[idx])
        self.status_var.set(f"Editing scenario {idx + 1}. Modify and click 'Add' to re-queue.")

    def _load_scenario(self, scenario: dict):
        agents = scenario["agents"]
        statics = scenario["static"]
        for i, vr in enumerate(self.agent_rows):
            if i < len(agents):
                lane, dest, offset = agents[i]
                vr.set(lane_tuple_to_approach(lane), dest, offset)
        for i, vr in enumerate(self.static_rows):
            if i < len(statics):
                lane, dest, offset = statics[i]
                vr.set(lane_tuple_to_approach(lane), dest, offset)

    # ── Randomize ───────────────────────────────────────────────────────

    def _on_randomize(self):
        for vr in self.agent_rows + self.static_rows:
            start = random.choice(APPROACHES)
            # Pick a destination different from start
            possible_dests = [a for a in APPROACHES if a != start]
            dest = random.choice(possible_dests)
            offset = random.choice([0, -10, -15, -20, -25, -30, -35, -40, -50, -60])
            vr.set(start, dest, offset)

    # ── Build scenario dict from UI ─────────────────────────────────────

    def _build_scenario(self) -> dict:
        return {
            "agents": [vr.get() for vr in self.agent_rows],
            "static": [vr.get() for vr in self.static_rows],
        }

    # ── Queue management ────────────────────────────────────────────────

    def _on_add(self):
        if not self._validate():
            return
        scenario = self._build_scenario()
        self.scenario_queue.append(scenario)
        self._refresh_queue_listbox()
        self.status_var.set(
            f"{len(self.scenario_queue)} scenario(s) queued. "
            "Add more or click Run All."
        )

    def _on_remove(self):
        sel = self.queue_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.scenario_queue.pop(idx)
        self._refresh_queue_listbox()
        self.status_var.set(f"{len(self.scenario_queue)} scenario(s) queued.")

    def _on_clear(self):
        self.scenario_queue.clear()
        self._refresh_queue_listbox()
        self.status_var.set("Queue cleared. Add scenarios to the queue.")

    def _refresh_queue_listbox(self):
        self.queue_listbox.delete(0, tk.END)
        for i, sc in enumerate(self.scenario_queue):
            desc = self._scenario_short_desc(sc)
            self.queue_listbox.insert(tk.END, f"  [{i + 1}]  {desc}")

    @staticmethod
    def _scenario_short_desc(scenario: dict) -> str:
        parts = []
        for j, (lane, dest, offset) in enumerate(scenario["agents"]):
            s = APPROACH_LABELS.get(lane[0], lane[0])
            d = APPROACH_LABELS.get(dest, dest)
            parts.append(f"A{j}:{s}→{d}")
        return "  ".join(parts)

    # ── Validation ──────────────────────────────────────────────────────

    def _validate(self) -> bool:
        all_rows = list(enumerate(self.agent_rows)) + [
            (6 + i, sr) for i, sr in enumerate(self.static_rows)
        ]
        # Check for same start with close offsets
        placements = []
        for idx, vr in all_rows:
            lane, dest, offset = vr.get()
            start = lane[0]
            if start == dest:
                label = f"Agent {idx}" if idx < 6 else f"Static {idx - 6}"
                messagebox.showwarning(
                    "Invalid Placement",
                    f"{label}: start and destination are the same ({start})."
                )
                return False
            placements.append((start, offset, idx))

        # Warn about vehicles on same approach with close offsets
        placements.sort(key=lambda x: (x[0], x[1]))
        for i in range(len(placements) - 1):
            if placements[i][0] == placements[i + 1][0]:
                diff = abs(placements[i][1] - placements[i + 1][1])
                if diff < 10:
                    a_idx = placements[i][2]
                    b_idx = placements[i + 1][2]
                    a_lbl = f"Agent {a_idx}" if a_idx < 6 else f"Static {a_idx - 6}"
                    b_lbl = f"Agent {b_idx}" if b_idx < 6 else f"Static {b_idx - 6}"
                    if not messagebox.askyesno(
                        "Potential Collision",
                        f"{a_lbl} and {b_lbl} start on the same approach "
                        f"({placements[i][0]}) with only {diff}m offset gap.\n\n"
                        "This may cause a collision at spawn. Continue anyway?"
                    ):
                        return False
        return True

    # ── Save queue to folder ─────────────────────────────────────────────

    def _on_save_queue(self):
        if not self.scenario_queue:
            messagebox.showinfo("Empty Queue", "Add at least one scenario to the queue first.")
            return

        exp_name = simpledialog.askstring(
            "Experiment Name",
            "Enter a name for this experiment:",
            parent=self.root,
            initialvalue=self._current_experiment_name or "",
        )
        if not exp_name or not exp_name.strip():
            return  # user cancelled or entered empty string

        # Sanitise: strip whitespace, replace spaces with underscores
        exp_name = exp_name.strip().replace(" ", "_")
        self._current_experiment_name = exp_name

        project_root = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(project_root, "scenarios", exp_name)

        from src.scenario_io import save_scenarios_list, save_metadata

        save_metadata(
            folder,
            env_type="double_intersection",
            env_config=my_scenarios.DOUBLE_INTERSECTION_ENV_CONFIG,
            scenario_count=len(self.scenario_queue),
        )

        written = save_scenarios_list(
            self.scenario_queue,
            folder,
            prefix="scenario",
            env_type="double_intersection",
        )
        self.status_var.set(
            f"Saved {len(written)} scenario(s) to: scenarios/{exp_name}/"
        )
        messagebox.showinfo(
            "Saved",
            f"Saved {len(written)} scenario(s) to:\nscenarios/{exp_name}/"
        )

    # ── Run all queued scenarios ─────────────────────────────────────────

    def _on_run_all(self):
        if not self.scenario_queue:
            messagebox.showinfo("Empty Queue", "Add at least one scenario to the queue first.")
            return

        try:
            self._prepare_runtime()
        except Exception as e:
            messagebox.showerror("Runtime Setup Error", str(e))
            return

        total = len(self.scenario_queue)
        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.add_btn.config(state="disabled")

        for i, scenario in enumerate(self.scenario_queue):
            self.queue_listbox.selection_clear(0, tk.END)
            self.queue_listbox.selection_set(i)
            self.queue_listbox.see(i)
            self.status_var.set(f"Running scenario {i + 1}/{total}… (close pygame window for next)")
            self.root.update()

            try:
                _run_simulation(scenario)
            except Exception as e:
                messagebox.showerror("Simulation Error", f"Scenario {i + 1} failed:\n{e}")
                break

        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.add_btn.config(state="normal")
        self.queue_listbox.selection_clear(0, tk.END)
        self.status_var.set(f"Finished running {total} scenario(s). Add more or Run again.")

    def _prepare_runtime(self):
        """Prepare gym env registration and globals once before running queued scenarios."""
        from highwayenv.utils import patch_intersection_env, register_double_intersection_env
        import src.experiment.scenarios as sc

        patch_intersection_env()
        try:
            register_double_intersection_env()
        except Exception:
            # Already registered in this process; safe to ignore.
            pass

        # Keep only the queue scenarios active for this run.
        sc.double_intersection_base_scenarios = list(self.scenario_queue)


# ---------------------------------------------------------------------------
# Simulation runner (reuses run_sandbox.py logic)
# ---------------------------------------------------------------------------

def _run_simulation(scenario: dict):
    """Launch one episode with the given scenario dict."""
    import my_scenarios

    import src.experiment.scenarios as sc
    import src.project_globals as project_globals
    import gymnasium as gym

    env_config = my_scenarios.DOUBLE_INTERSECTION_ENV_CONFIG

    # Inject our single scenario
    sc.double_intersection_base_scenarios = [scenario]

    # Reset globals
    project_globals.after_is_arrived_flags = [False] * len(env_config["controlled_cars"])
    project_globals.rollout_buffers = []
    project_globals.episode_count = 0

    env = gym.make(
        "RELdouble-intersection-v0",
        render_mode="human",
        config=env_config,
    )

    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0
    step = 0

    print(f"\n{'=' * 50}")
    print(f"  UI Scenario: running…")
    print(f"{'=' * 50}")

    while not (terminated or truncated):
        num_agents = len(env.unwrapped.controlled_vehicles)
        action = tuple(1 for _ in range(num_agents))  # FASTER
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += sum(reward) if hasattr(reward, '__iter__') else reward
        step += 1

    env.close()
    print(f"  Finished after {step} steps | total reward: {total_reward:.1f}")
    print(f"  crashed={info.get('crashed', '?')}  agents_arrived={info.get('agents_dones', '?')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ScenarioBuilderApp(root)
    root.mainloop()
