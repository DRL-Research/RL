# Code and comments only in English.

"""
Per-step dataset: all controlled agents' positions/velocities, crash flags,
polygon-intersection collision pairs, and whether new crashes occurred this step.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src import project_globals
from src.model.agent_handler import Driver
from src.training.episode_utils import _build_global_master_input, _build_local_master_input
from src.training.general_utils import get_scaler_action_and_action_array


def _inner_env(wrapped_env):
    return wrapped_env.env._get_unwrapped_env()


def _vehicle_label(inner, v) -> str:
    if v in inner.controlled_vehicles:
        return f"C{inner.controlled_vehicles.index(v)}"
    try:
        idx = inner.road.vehicles.index(v)
    except ValueError:
        idx = id(v) % 1_000_000
    return f"{v.__class__.__name__}_{idx}"


def all_intersecting_pairs(inner, dt: float = 0.0) -> List[str]:
    """Polygon-intersection pairs among collidable road vehicles (same test as HighwayEnv)."""
    vehs = list(inner.road.vehicles)
    seen = set()
    out: List[str] = []
    for i, a in enumerate(vehs):
        for b in vehs[i + 1 :]:
            if not (getattr(a, "collidable", True) and getattr(b, "collidable", True)):
                continue
            if not (getattr(a, "check_collisions", True) or getattr(b, "check_collisions", True)):
                continue
            try:
                inter, _, _ = a._is_colliding(b, dt)
            except Exception:
                continue
            if not inter:
                continue
            la, lb = _vehicle_label(inner, a), _vehicle_label(inner, b)
            key = tuple(sorted((la, lb)))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{key[0]}+{key[1]}")
    return out


def _controlled_state_row(inner, n_agents: int) -> Dict[str, Any]:
    row = {}
    for i in range(n_agents):
        if i < len(inner.controlled_vehicles):
            v = inner.controlled_vehicles[i]
            pos = np.asarray(v.position, dtype=np.float64).ravel()
            vel = np.asarray(v.velocity, dtype=np.float64).ravel()
            row[f"agent{i}_x"] = float(pos[0]) if len(pos) > 0 else 0.0
            row[f"agent{i}_y"] = float(pos[1]) if len(pos) > 1 else 0.0
            row[f"agent{i}_vx"] = float(vel[0]) if len(vel) > 0 else 0.0
            row[f"agent{i}_vy"] = float(vel[1]) if len(vel) > 1 else 0.0
            row[f"agent{i}_crashed"] = int(bool(v.crashed))
        else:
            row[f"agent{i}_x"] = float("nan")
            row[f"agent{i}_y"] = float("nan")
            row[f"agent{i}_vx"] = float("nan")
            row[f"agent{i}_vy"] = float("nan")
            row[f"agent{i}_crashed"] = 0
    return row


def collect_episodes_to_csv(
    experiment,
    wrapped_env,
    master_model,
    agent_model,
    *,
    num_episodes: int,
    csv_path: str,
    episode_summary_json: str,
    stochastic_agent: bool = False,
    verbose_every: int = 5,
    render_delay_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Run ``num_episodes`` with the same control stack as training, append one CSV row per step.
    """
    n_agents = experiment.CARS_AMOUNT
    agents_per_lm = experiment.AGENTS_PER_LOCAL_MASTER
    emb_size = experiment.EMBEDDING_SIZE

    fieldnames = (
        [
            "episode",
            "step",
            "scenario_index",
            "base_scenario",
            "rotation",
            "any_new_crash",
            "n_intersecting_pairs",
            "intersecting_pairs",
            "pairs_with_newly_crashed",
            "step_reward",
            "done",
            "truncated",
        ]
        + [f"agent{i}_{k}" for i in range(n_agents) for k in ("x", "y", "vx", "vy", "crashed")]
    )

    episode_summaries: List[Dict[str, Any]] = []
    pair_counter: Counter = Counter()
    new_crash_steps = 0
    total_steps = 0

    master_model.freeze()
    train_both = False
    training_local_master = False
    training_agent = bool(stochastic_agent)
    training_global_master = False

    global_step = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for ep in range(1, num_episodes + 1):
            wrapped_env.reset()
            wrapped_env.render()
            if render_delay_sec > 0:
                time.sleep(render_delay_sec)
            inner = _inner_env(wrapped_env)
            scenario_index = int(getattr(inner, "last_scenario_index", -1))
            base_sc = int(getattr(inner, "last_base_scenario", -1))
            rot = int(getattr(inner, "last_rotation", -1))

            done, truncated = False, False
            global_emb_prev = np.zeros(emb_size, dtype=np.float32)
            steps = 0
            ep_first_collision_step: Optional[int] = None
            ep_first_pairs: Optional[str] = None
            terminal_crashed = False

            while not done and not truncated:
                steps += 1
                total_steps += 1
                before_crashed = {id(v): bool(v.crashed) for v in inner.road.vehicles}

                all_states = wrapped_env.env.current_state
                lm1_input = _build_local_master_input(global_emb_prev, all_states[0:agents_per_lm])
                lm2_input = _build_local_master_input(global_emb_prev, all_states[agents_per_lm:n_agents])
                lm1_emb, _, _ = master_model.get_proto_action(lm1_input)
                lm2_emb, _, _ = master_model.get_proto_action(lm2_input)
                gm_input = _build_global_master_input(lm1_emb, lm2_emb)
                gm_emb, _, _ = master_model.get_proto_action(gm_input)
                local_embeddings = [lm1_emb] * agents_per_lm + [lm2_emb] * agents_per_lm
                car_observations = wrapped_env.env.build_full_obs(local_embeddings)

                policy_stochastic = bool(train_both or training_agent)
                actions = Driver.get_action(
                    agent_model,
                    car_observations,
                    global_step,
                    experiment.EXPLORATION_EXPLOITATION_THRESHOLD,
                    policy_stochastic=policy_stochastic,
                )
                cars_scalar_action = []
                for action in actions:
                    s, _ = get_scaler_action_and_action_array(action)
                    cars_scalar_action.append(s)

                _, reward, done, truncated, _ = wrapped_env.step(tuple(cars_scalar_action))
                global_step += 1
                wrapped_env.render()
                if render_delay_sec > 0:
                    time.sleep(render_delay_sec)
                inner = _inner_env(wrapped_env)

                newly = [v for v in inner.road.vehicles if v.crashed and not before_crashed.get(id(v), False)]
                any_new = len(newly) > 0
                if any_new:
                    new_crash_steps += 1

                pairs = all_intersecting_pairs(inner, dt=0.0)
                for p in pairs:
                    pair_counter[p] += 1

                new_ids = {id(v) for v in newly}
                pairs_new = []
                for p in pairs:
                    a, b = p.split("+", 1)
                    if _pair_touches_new_ids(inner, a, b, new_ids):
                        pairs_new.append(p)

                if any_new and ep_first_collision_step is None:
                    ep_first_collision_step = steps
                    ep_first_pairs = ";".join(pairs) if pairs else ""

                row = {
                    "episode": ep,
                    "step": steps,
                    "scenario_index": scenario_index,
                    "base_scenario": base_sc,
                    "rotation": rot,
                    "any_new_crash": int(any_new),
                    "n_intersecting_pairs": len(pairs),
                    "intersecting_pairs": ";".join(pairs),
                    "pairs_with_newly_crashed": ";".join(pairs_new),
                    "step_reward": float(reward),
                    "done": int(bool(done)),
                    "truncated": int(bool(truncated)),
                }
                row.update(_controlled_state_row(inner, n_agents))
                writer.writerow(row)

                global_emb_prev = np.asarray(gm_emb, dtype=np.float32).flatten()

            if done:
                terminal_crashed = any(v.crashed for v in inner.controlled_vehicles)

            episode_summaries.append(
                {
                    "episode": ep,
                    "steps": steps,
                    "scenario_index": scenario_index,
                    "base_scenario": base_sc,
                    "rotation": rot,
                    "terminal_crashed": terminal_crashed,
                    "first_collision_step": ep_first_collision_step,
                    "first_collision_pairs": ep_first_pairs,
                }
            )

            if verbose_every > 0 and (ep % verbose_every == 0 or ep == 1):
                print(
                    f"Episode {ep}/{num_episodes}  steps={steps}  terminal_crashed={terminal_crashed}  "
                    f"scenario={scenario_index}"
                )

    analysis = {
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "steps_with_any_new_crash": new_crash_steps,
        "pair_occurrences_across_steps": dict(pair_counter.most_common(50)),
        "episode_summaries": episode_summaries,
    }
    with open(episode_summary_json, "w", encoding="utf-8") as jf:
        json.dump(analysis, jf, indent=2)

    return analysis


def _pair_touches_new_ids(inner, la: str, lb: str, new_ids: set) -> bool:
    """True if either endpoint of the pair is a vehicle that newly crashed this step."""

    def label_in_new(lab: str) -> bool:
        if lab.startswith("C") and lab[1:].isdigit():
            i = int(lab[1:])
            if 0 <= i < len(inner.controlled_vehicles):
                return id(inner.controlled_vehicles[i]) in new_ids
        for v in inner.road.vehicles:
            if _vehicle_label(inner, v) == lab:
                return id(v) in new_ids
        return False

    return label_in_new(la) or label_in_new(lb)


def analyze_steps_csv(csv_path: str, out_analysis_json: str) -> Dict[str, Any]:
    """Re-read CSV and compute aggregates (no need to re-simulate)."""
    pair_counter: Counter = Counter()
    new_crash_rows = 0
    rows = 0
    first_collision_by_ep: Dict[int, int] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += 1
            if int(r.get("any_new_crash", 0)):
                new_crash_rows += 1
            pairs = (r.get("intersecting_pairs") or "").strip()
            if pairs:
                for p in pairs.split(";"):
                    if p:
                        pair_counter[p] += 1
            ep = int(r["episode"])
            st = int(r["step"])
            if int(r.get("any_new_crash", 0)) and ep not in first_collision_by_ep:
                first_collision_by_ep[ep] = st

    rep = {
        "csv_rows": rows,
        "rows_with_new_crash": new_crash_rows,
        "top_intersecting_pairs": pair_counter.most_common(30),
        "first_collision_step_by_episode": first_collision_by_ep,
    }
    with open(out_analysis_json, "w", encoding="utf-8") as jf:
        json.dump(rep, jf, indent=2)
    return rep
