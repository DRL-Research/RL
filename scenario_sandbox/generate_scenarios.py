"""
generate_scenarios.py - Scenario dataset generator
==================================================
Generate saved scenario folders for:
  - the original `double_intersection_scenario_ui.py` flow
  - `one_element_scenario_ui.py`
  - `composer_scenario_ui.py`

When run without CLI arguments, this file opens a small Tkinter generator UI
so the user can choose:
  - which UI family to target
  - how many scenarios to create
  - whether distribution is equal or level-heavy

Run:
    python generate_scenarios.py
    python generate_scenarios.py --count 300 --builder one_element_scenario_ui --env-type roundabout
    python generate_scenarios.py --count 300 --builder composer_scenario_ui --left-element roundabout --right-element intersection
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

import my_scenarios
import composer_scenario_ui
import one_element_scenario_ui
from src.composable_layout import MODULE_TYPE_OPTIONS, layout_title, normalize_layout_config, visible_approaches
from src.scenario_io import save_metadata, save_scenarios_list


TIERS = ["easy", "medium", "hard"]
TARGET_AGENT_COUNT = 6
TARGET_STATIC_COUNT = 2
DEFAULT_SEED = 42

DISTRIBUTIONS = {
    "equal": [1, 1, 1],
    "balanced": [1, 1, 1],  # CLI compatibility with the older generator
    "easy-heavy": [3, 1, 1],
    "medium-heavy": [1, 3, 1],
    "hard-heavy": [1, 1, 3],
    "hard-only": [0, 1, 4],
}

DISTRIBUTION_LABELS = {
    "equal": "Equal",
    "easy-heavy": "Easy-heavy",
    "medium-heavy": "Medium-heavy",
    "hard-heavy": "Hard-heavy",
    "hard-only": "Hard-only",
}

BUILDER_LABELS = {
    "double_intersection_scenario_ui": "double_intersection_scenario_ui.py",
    "one_element_scenario_ui": "one_element_scenario_ui.py",
    "composer_scenario_ui": "composer_scenario_ui.py",
}

BUILDER_ALIASES = {
    "scenario_ui": "double_intersection_scenario_ui",
    "scenario_ui_multi": "one_element_scenario_ui",
    "scenario_ui_composer": "composer_scenario_ui",
}

AGENT_OFFSET_POOLS = {
    "easy": [-90, -75, -60, -45, -30, -15],
    "medium": [-70, -55, -40, -25, -10, 0],
    "hard": [-45, -35, -25, -15, -10, -5, 0, 5, 10],
}

STATIC_OFFSET_POOLS = {
    "easy": [-110, -95, -80, -65, -50, -35],
    "medium": [-95, -80, -65, -50, -35, -20, -5],
    "hard": [-80, -65, -50, -35, -20, -10, 0, 10],
}

MIN_GAP_BY_DIFFICULTY = {
    "easy": 18,
    "medium": 10,
    "hard": 4,
}


@dataclass
class GenerationTarget:
    builder: str
    env_type: str
    label: str
    approaches: list[str]
    env_config: dict
    output_folder: str
    layout_config: dict | None = None


def parse_approach(approach: str) -> tuple[str, int]:
    if "_o" in approach:
        module_name, corner = approach.split("_o")
        return module_name or "single", int(corner)
    if approach.startswith("o"):
        return "single", int(approach[1:])
    raise ValueError(f"Unrecognized approach name: {approach}")


def module_of(approach: str) -> str:
    return parse_approach(approach)[0]


def corner_of(approach: str) -> int:
    return parse_approach(approach)[1]


def route_delta(start: str, dest: str) -> int | None:
    if module_of(start) != module_of(dest):
        return None
    return (corner_of(dest) - corner_of(start)) % 4


def resolve_generation_target(
    builder: str,
    *,
    env_type: str = "double_intersection",
    layout_config: dict | None = None,
) -> GenerationTarget:
    builder = BUILDER_ALIASES.get(builder, builder)

    if builder == "double_intersection_scenario_ui":
        target_env_type = "double_intersection"
        return GenerationTarget(
            builder=builder,
            env_type=target_env_type,
            label="Original UI - Double Intersection",
            approaches=list(one_element_scenario_ui.get_env_definition(target_env_type)["approaches"]),
            env_config=one_element_scenario_ui.build_env_config(target_env_type),
            output_folder=os.path.join("saved_scenarios", "double_intersection_scenario_ui_generated"),
        )

    if builder == "one_element_scenario_ui":
        target_env_type = env_type
        env_definition = one_element_scenario_ui.get_env_definition(target_env_type)
        return GenerationTarget(
            builder=builder,
            env_type=target_env_type,
            label=f"Multi UI - {env_definition['label']}",
            approaches=list(env_definition["approaches"]),
            env_config=one_element_scenario_ui.build_env_config(target_env_type),
            output_folder=os.path.join("saved_scenarios", f"one_element_scenario_ui_{target_env_type}_generated"),
        )

    if builder == "composer_scenario_ui":
        normalized_layout = normalize_layout_config(layout_config)
        layout_name = layout_title(normalized_layout).replace(" | ", "_").replace(" ", "_").lower()
        return GenerationTarget(
            builder=builder,
            env_type=composer_scenario_ui.COMPOSER_ENV_TYPE,
            label=f"Composer UI - {layout_title(normalized_layout)}",
            approaches=visible_approaches(normalized_layout),
            env_config=composer_scenario_ui.build_env_config(normalized_layout),
            output_folder=os.path.join("saved_scenarios", f"composer_scenario_ui_{layout_name}_generated"),
            layout_config=normalized_layout,
        )

    raise ValueError(f"Unsupported builder: {builder}")


def choose_starts(approaches: list[str], difficulty: str, count: int) -> list[str]:
    if difficulty == "easy":
        starts = []
        while len(starts) < count:
            starts.extend(random.sample(approaches, len(approaches)))
        starts = starts[:count]
        random.shuffle(starts)
        return starts

    if difficulty == "medium":
        unique_count = min(len(approaches), max(3, count - 2))
        starts = random.sample(approaches, unique_count)
        while len(starts) < count:
            starts.append(random.choice(approaches))
        random.shuffle(starts)
        return starts

    hotspot_count = min(2, len(approaches))
    hotspots = random.sample(approaches, hotspot_count)
    starts = []
    for _ in range(count):
        if random.random() < 0.7:
            starts.append(random.choice(hotspots))
        else:
            starts.append(random.choice(approaches))
    return starts


def choose_destination(start: str, approaches: list[str], difficulty: str) -> str:
    candidates = [approach for approach in approaches if approach != start]
    local_candidates = [dest for dest in candidates if module_of(dest) == module_of(start)]
    cross_candidates = [dest for dest in candidates if module_of(dest) != module_of(start)]

    use_cross = False
    if cross_candidates:
        if difficulty == "easy":
            use_cross = random.random() < 0.20
        elif difficulty == "medium":
            use_cross = random.random() < 0.45
        else:
            use_cross = random.random() < 0.75

    active_candidates = cross_candidates if use_cross else (local_candidates or candidates)

    if active_candidates is cross_candidates:
        return random.choice(active_candidates)

    weights = []
    for dest in active_candidates:
        delta = route_delta(start, dest)
        if delta == 1:
            weights.append({"easy": 4, "medium": 2, "hard": 1}[difficulty])
        elif delta == 2:
            weights.append({"easy": 3, "medium": 3, "hard": 2}[difficulty])
        elif delta == 3:
            weights.append({"easy": 1, "medium": 2, "hard": 4}[difficulty])
        else:
            weights.append(1)
    return random.choices(active_candidates, weights=weights, k=1)[0]


def assign_offsets(starts: list[str], difficulty: str, *, is_static: bool) -> list[int]:
    pool = STATIC_OFFSET_POOLS[difficulty] if is_static else AGENT_OFFSET_POOLS[difficulty]
    min_gap = MIN_GAP_BY_DIFFICULTY[difficulty]
    used_by_start: dict[str, list[int]] = {}
    offsets = []

    for start in starts:
        selected = None
        for candidate in random.sample(pool, len(pool)):
            if all(abs(candidate - existing) >= min_gap for existing in used_by_start.get(start, [])):
                selected = candidate
                break
        if selected is None:
            selected = random.choice(pool)
        used_by_start.setdefault(start, []).append(selected)
        offsets.append(selected)

    return offsets


def approach_to_lane_tuple(approach: str) -> tuple[str, str, int]:
    if "_o" in approach:
        return (approach, approach.replace("_o", "_ir"), 0)
    return (approach, approach.replace("o", "ir", 1), 0)


def build_scenario(
    agent_starts: list[str],
    agent_dests: list[str],
    agent_offsets: list[int],
    static_starts: list[str],
    static_dests: list[str],
    static_offsets: list[int],
) -> dict:
    return {
        "agents": [
            (approach_to_lane_tuple(start), dest, offset)
            for start, dest, offset in zip(agent_starts, agent_dests, agent_offsets)
        ],
        "static": [
            (approach_to_lane_tuple(start), dest, offset)
            for start, dest, offset in zip(static_starts, static_dests, static_offsets)
        ],
    }


def route_complexity(start: str, dest: str) -> float:
    if module_of(start) != module_of(dest):
        return 0.95
    delta = route_delta(start, dest)
    return {
        1: 0.35,
        2: 0.60,
        3: 0.80,
    }.get(delta, 0.45)


def pair_conflict_score(route_a, route_b) -> float:
    start_a, dest_a, offset_a = route_a
    start_b, dest_b, offset_b = route_b

    base = 0.05
    if start_a == start_b:
        base += 0.55
    elif module_of(start_a) == module_of(start_b):
        base += 0.15

    if dest_a == dest_b:
        base += 0.18
    elif module_of(dest_a) == module_of(dest_b):
        base += 0.07

    if module_of(start_a) != module_of(dest_a):
        base += 0.10
    if module_of(start_b) != module_of(dest_b):
        base += 0.10

    base += 0.15 * ((route_complexity(start_a, dest_a) + route_complexity(start_b, dest_b)) / 2.0)

    delta_a = route_delta(start_a, dest_a)
    delta_b = route_delta(start_b, dest_b)
    if delta_a in {2, 3} and delta_b in {2, 3} and module_of(start_a) == module_of(start_b):
        base += 0.12

    offset_gap = abs(offset_a - offset_b)
    timing_risk = max(0.0, 1.0 - (offset_gap / 80.0))
    return min(1.0, base * (0.55 + 0.45 * timing_risk))


def scenario_difficulty_score(scenario: dict) -> float:
    agent_routes = [(lane[0], dest, offset) for lane, dest, offset in scenario["agents"]]
    static_routes = [(lane[0], dest, offset) for lane, dest, offset in scenario["static"]]

    if len(agent_routes) < 2:
        return 0.0

    pair_scores = []
    for i in range(len(agent_routes)):
        for j in range(i + 1, len(agent_routes)):
            pair_scores.append(pair_conflict_score(agent_routes[i], agent_routes[j]))

    avg_pair_score = sum(pair_scores) / len(pair_scores)

    cross_module_rate = 0.0
    if agent_routes:
        cross_module_rate = sum(
            1 for start, dest, _ in agent_routes if module_of(start) != module_of(dest)
        ) / len(agent_routes)

    static_pressure = 0.0
    if static_routes:
        static_pressure = sum(
            0.4 if any(static_start == agent_start for agent_start, _agent_dest, _agent_offset in agent_routes) else 0.1
            for static_start, _static_dest, _static_offset in static_routes
        ) / len(static_routes)

    score = (avg_pair_score * 0.70) + (cross_module_rate * 0.20) + (static_pressure * 0.10)
    return max(0.0, min(1.0, score))


def classify_difficulty(score: float) -> str:
    if score < 0.38:
        return "easy"
    if score < 0.68:
        return "medium"
    return "hard"


def generate_candidate_scenario(target: GenerationTarget, difficulty: str) -> dict:
    agent_starts = choose_starts(target.approaches, difficulty, TARGET_AGENT_COUNT)
    agent_dests = [choose_destination(start, target.approaches, difficulty) for start in agent_starts]
    agent_offsets = assign_offsets(agent_starts, difficulty, is_static=False)

    static_starts = choose_starts(target.approaches, difficulty, TARGET_STATIC_COUNT)
    static_dests = [choose_destination(start, target.approaches, difficulty) for start in static_starts]
    static_offsets = assign_offsets(static_starts, difficulty, is_static=True)

    return build_scenario(
        agent_starts,
        agent_dests,
        agent_offsets,
        static_starts,
        static_dests,
        static_offsets,
    )


def scenario_signature(scenario: dict) -> tuple:
    signature = []
    for lane, dest, offset in scenario["agents"]:
        signature.append((lane[0], lane[1], lane[2], dest, offset))
    for lane, dest, offset in scenario["static"]:
        signature.append((lane[0], lane[1], lane[2], dest, offset))
    return tuple(signature)


def compute_targets(total_count: int, distribution: str) -> dict[str, int]:
    weights = DISTRIBUTIONS[distribution]
    total_weight = sum(weights)
    easy_count = int(total_count * weights[0] / total_weight)
    medium_count = int(total_count * weights[1] / total_weight)
    hard_count = total_count - easy_count - medium_count
    return {
        "easy": easy_count,
        "medium": medium_count,
        "hard": hard_count,
    }


def generate_scenario_set(
    target: GenerationTarget,
    *,
    total_count: int,
    distribution: str,
    seed: int,
) -> Dict[str, List[dict]]:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")

    random.seed(seed)
    targets = compute_targets(total_count, distribution)
    scenarios_by_tier = {tier: [] for tier in TIERS}
    seen_signatures = {tier: set() for tier in TIERS}
    fallback_candidates = {tier: [] for tier in TIERS}

    print(f"\n{'=' * 72}")
    print(f"SCENARIO GENERATOR - {target.label}")
    print(f"{'=' * 72}")
    print(
        f"Generating {total_count} scenarios "
        f"({targets['easy']} easy, {targets['medium']} medium, {targets['hard']} hard) "
        f"with distribution '{distribution}'."
    )
    if target.layout_config:
        print(f"  Layout: {target.layout_config}")
    print(f"  Approaches: {', '.join(target.approaches)}")
    print(f"  Seed: {seed}")

    for difficulty in TIERS:
        desired = targets[difficulty]
        if desired == 0:
            continue

        print(f"\n[{difficulty.upper()}] Generating {desired} scenario(s)...")
        attempts = 0
        max_attempts = max(5000, desired * 250)

        while len(scenarios_by_tier[difficulty]) < desired and attempts < max_attempts:
            candidate = generate_candidate_scenario(target, difficulty)
            signature = scenario_signature(candidate)
            if signature in seen_signatures[difficulty]:
                attempts += 1
                continue

            score = scenario_difficulty_score(candidate)
            classified = classify_difficulty(score)
            if classified != difficulty:
                fallback_candidates[difficulty].append((score, candidate))
                attempts += 1
                continue

            scenarios_by_tier[difficulty].append(candidate)
            seen_signatures[difficulty].add(signature)

        if len(scenarios_by_tier[difficulty]) < desired:
            fallback_fill = [
                (score, candidate)
                for score, candidate in fallback_candidates[difficulty]
                if scenario_signature(candidate) not in seen_signatures[difficulty]
            ]

            if difficulty == "easy":
                fallback_fill.sort(key=lambda item: item[0])
            elif difficulty == "hard":
                fallback_fill.sort(key=lambda item: item[0], reverse=True)
            else:
                fallback_fill.sort(key=lambda item: abs(item[0] - 0.53))

            for score, candidate in fallback_fill:
                if len(scenarios_by_tier[difficulty]) >= desired:
                    break
                signature = scenario_signature(candidate)
                if signature in seen_signatures[difficulty]:
                    continue
                scenarios_by_tier[difficulty].append(candidate)
                seen_signatures[difficulty].add(signature)

        if len(scenarios_by_tier[difficulty]) < desired:
            print(
                f"  Warning: generated {len(scenarios_by_tier[difficulty])}/{desired} "
                f"{difficulty} scenarios after {attempts} attempts."
            )

    return scenarios_by_tier


def save_generated_scenarios(
    target: GenerationTarget,
    scenarios_by_tier: dict[str, list[dict]],
    *,
    output_folder: str,
    difficulty: str,
    dry_run: bool,
) -> dict[str, int]:
    tiers_to_save = [difficulty] if difficulty != "all" else list(TIERS)
    saved_counts: dict[str, int] = {}

    for tier in tiers_to_save:
        tier_scenarios = scenarios_by_tier[tier]
        folder = os.path.join(output_folder, tier)

        print(f"\n[{tier.upper()}] {len(tier_scenarios)} scenarios -> {folder}/")
        if dry_run:
            print(f"  (dry-run) would write {len(tier_scenarios)} scenario files")
            saved_counts[tier] = len(tier_scenarios)
            continue

        written = save_scenarios_list(
            tier_scenarios,
            folder,
            prefix="scenario",
            env_type=target.env_type,
        )
        save_metadata(
            folder,
            env_type=target.env_type,
            env_config=copy.deepcopy(target.env_config),
            scenario_count=len(tier_scenarios),
        )

        saved_counts[tier] = len(written)
        print(f"  Saved {len(written)} scenario files")
        print(f"  Metadata saved to {folder}/metadata.json")

    return saved_counts


def generate_and_save(
    target: GenerationTarget,
    *,
    total_count: int,
    distribution: str,
    difficulty: str,
    output_folder: str,
    seed: int,
    dry_run: bool,
) -> dict[str, list[dict]]:
    scenarios_by_tier = generate_scenario_set(
        target,
        total_count=total_count,
        distribution=distribution,
        seed=seed,
    )
    save_generated_scenarios(
        target,
        scenarios_by_tier,
        output_folder=output_folder,
        difficulty=difficulty,
        dry_run=dry_run,
    )

    print(f"\n{'=' * 72}")
    print("Generation complete!")
    print(f"  easy:   {len(scenarios_by_tier['easy'])} scenarios")
    print(f"  medium: {len(scenarios_by_tier['medium'])} scenarios")
    print(f"  hard:   {len(scenarios_by_tier['hard'])} scenarios")
    print(f"  TOTAL:  {sum(len(items) for items in scenarios_by_tier.values())} scenarios")
    print(f"{'=' * 72}\n")
    return scenarios_by_tier


class ScenarioGeneratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Scenario Generator")
        root.resizable(False, False)

        self.builder_var = tk.StringVar(value="double_intersection_scenario_ui")
        self.multi_env_var = tk.StringVar(value="double_intersection")
        self.left_element_var = tk.StringVar(value="roundabout")
        self.right_element_var = tk.StringVar(value="intersection")
        self.count_var = tk.StringVar(value="300")
        self.distribution_var = tk.StringVar(value="equal")
        self.folder_var = tk.StringVar()
        self.seed_var = tk.StringVar(value=str(DEFAULT_SEED))

        header = ttk.Frame(root, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Scenario Generator", font=("Segoe UI", 14, "bold")).pack(side="left")

        source_frame = ttk.LabelFrame(root, text="Source UI", padding=10)
        source_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(source_frame, text="Scenario Builder").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        builder_combo = ttk.Combobox(
            source_frame,
            textvariable=self.builder_var,
            values=list(BUILDER_LABELS.keys()),
            state="readonly",
            width=24,
        )
        builder_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        builder_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        self.source_note_var = tk.StringVar()
        ttk.Label(source_frame, textvariable=self.source_note_var, foreground="gray").grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(2, 0),
        )

        self.multi_frame = ttk.Frame(source_frame)
        self.multi_frame.grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Label(self.multi_frame, text="Scenario Type").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        multi_env_combo = ttk.Combobox(
            self.multi_frame,
            textvariable=self.multi_env_var,
            values=one_element_scenario_ui.ENV_TYPE_OPTIONS,
            state="readonly",
            width=24,
        )
        multi_env_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        multi_env_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        self.composer_frame = ttk.Frame(source_frame)
        self.composer_frame.grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Label(self.composer_frame, text="Left Element").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        composer_left_combo = ttk.Combobox(
            self.composer_frame,
            textvariable=self.left_element_var,
            values=MODULE_TYPE_OPTIONS,
            state="readonly",
            width=14,
        )
        composer_left_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        composer_left_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        ttk.Label(self.composer_frame, text="Right Element").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        composer_right_combo = ttk.Combobox(
            self.composer_frame,
            textvariable=self.right_element_var,
            values=MODULE_TYPE_OPTIONS,
            state="readonly",
            width=14,
        )
        composer_right_combo.grid(row=0, column=3, sticky="w", padx=4, pady=4)
        composer_right_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        settings_frame = ttk.LabelFrame(root, text="Generation Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(settings_frame, text="Scenario Count").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(settings_frame, textvariable=self.count_var, width=10).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(settings_frame, text="Distribution").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        distribution_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.distribution_var,
            values=["equal", "easy-heavy", "medium-heavy", "hard-heavy", "hard-only"],
            state="readonly",
            width=14,
        )
        distribution_combo.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(settings_frame, text="Seed").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(settings_frame, textvariable=self.seed_var, width=10).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(settings_frame, text="Output Folder").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(settings_frame, textvariable=self.folder_var, width=52).grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="w",
            padx=4,
            pady=4,
        )

        action_frame = ttk.Frame(root, padding=10)
        action_frame.pack(fill="x")
        self.generate_btn = ttk.Button(action_frame, text="Generate Scenarios", command=self._on_generate)
        self.generate_btn.pack(side="left", ipadx=14, ipady=5)

        self.status_var = tk.StringVar(value="Choose the target UI, scenario count, and distribution.")
        ttk.Label(root, textvariable=self.status_var, foreground="gray", wraplength=520).pack(
            padx=10,
            pady=(0, 10),
            anchor="w",
        )

        self._refresh_source_controls()

    def _current_layout_config(self) -> dict:
        return normalize_layout_config(
            {
                "left": self.left_element_var.get(),
                "right": self.right_element_var.get(),
            }
        )

    def _refresh_source_controls(self):
        builder = self.builder_var.get()

        if builder == "double_intersection_scenario_ui":
            self.multi_frame.grid_remove()
            self.composer_frame.grid_remove()
            self.source_note_var.set("Original UI uses the double-intersection layout.")
        elif builder == "one_element_scenario_ui":
            self.multi_frame.grid()
            self.composer_frame.grid_remove()
            env_label = one_element_scenario_ui.get_env_definition(self.multi_env_var.get())["label"]
            self.source_note_var.set(f"Multi UI target: {env_label}.")
        else:
            self.multi_frame.grid_remove()
            self.composer_frame.grid()
            self.source_note_var.set(f"Composer layout: {layout_title(self._current_layout_config())}.")

        self.folder_var.set(self._default_folder())

    def _default_folder(self) -> str:
        target = resolve_generation_target(
            self.builder_var.get(),
            env_type=self.multi_env_var.get(),
            layout_config=self._current_layout_config(),
        )
        return target.output_folder

    def _on_source_changed(self, _event=None):
        self._refresh_source_controls()

    def _on_generate(self):
        try:
            count = int(self.count_var.get())
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Scenario count and seed must be integers.")
            return

        if count <= 0:
            messagebox.showerror("Invalid Input", "Scenario count must be greater than zero.")
            return

        distribution = self.distribution_var.get()
        if distribution not in DISTRIBUTIONS:
            messagebox.showerror("Invalid Input", "Choose a valid distribution option.")
            return

        folder = self.folder_var.get().strip() or self._default_folder()

        target = resolve_generation_target(
            self.builder_var.get(),
            env_type=self.multi_env_var.get(),
            layout_config=self._current_layout_config(),
        )

        self.generate_btn.config(state="disabled")
        self.status_var.set(f"Generating {count} scenarios for {target.label}...")
        self.root.update()

        try:
            scenarios_by_tier = generate_and_save(
                target,
                total_count=count,
                distribution=distribution,
                difficulty="all",
                output_folder=folder,
                seed=seed,
                dry_run=False,
            )
        except Exception as exc:
            self.generate_btn.config(state="normal")
            self.status_var.set("Generation failed.")
            messagebox.showerror("Generation Error", str(exc))
            return

        total_saved = sum(len(items) for items in scenarios_by_tier.values())
        self.generate_btn.config(state="normal")
        self.status_var.set(f"Saved {total_saved} scenarios to {folder}/")
        messagebox.showinfo(
            "Generation Complete",
            f"Saved {total_saved} scenarios to:\n{folder}/\n\n"
            f"easy: {len(scenarios_by_tier['easy'])}\n"
            f"medium: {len(scenarios_by_tier['medium'])}\n"
            f"hard: {len(scenarios_by_tier['hard'])}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate saved scenarios for the available scenario UIs")
    parser.add_argument("--count", type=int, default=300, help="Total scenarios to generate (default: 300)")
    parser.add_argument(
        "--builder",
        choices=list(BUILDER_LABELS.keys()) + list(BUILDER_ALIASES.keys()),
        default="double_intersection_scenario_ui",
        help="Which scenario builder family to target",
    )
    parser.add_argument(
        "--env-type",
        choices=one_element_scenario_ui.ENV_TYPE_OPTIONS,
        default="double_intersection",
        help="Used with one_element_scenario_ui",
    )
    parser.add_argument(
        "--left-element",
        choices=MODULE_TYPE_OPTIONS,
        default="roundabout",
        help="Used with composer_scenario_ui",
    )
    parser.add_argument(
        "--right-element",
        choices=MODULE_TYPE_OPTIONS,
        default="intersection",
        help="Used with composer_scenario_ui",
    )
    parser.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS.keys()),
        default="equal",
        help="Difficulty distribution for the generated dataset",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which tier(s) to save (default: all)",
    )
    parser.add_argument(
        "--folder",
        default="",
        help="Base output folder. If empty, a default folder is chosen from the selected UI target.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible generation")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be saved without writing files")
    parser.add_argument("--gui", action="store_true", help="Force the Tkinter generator UI")
    return parser


def run_cli(args) -> None:
    target = resolve_generation_target(
        args.builder,
        env_type=args.env_type,
        layout_config={
            "left": args.left_element,
            "right": args.right_element,
        },
    )
    output_folder = args.folder.strip() or target.output_folder
    generate_and_save(
        target,
        total_count=args.count,
        distribution=args.distribution,
        difficulty=args.difficulty,
        output_folder=output_folder,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    no_cli_args = len(sys.argv) == 1
    if args.gui or no_cli_args:
        root = tk.Tk()
        app = ScenarioGeneratorApp(root)
        root.mainloop()
        return

    run_cli(args)


if __name__ == "__main__":
    main()
