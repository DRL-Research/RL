from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hdmaps.env.corridor import Corridor
from hdmaps.evaluation.episode import run_simultaneous_episode, run_staggered_episode
from hdmaps.hierarchy.tree import hierarchy_label
from hdmaps.paths import agent_checkpoint, master_checkpoint
from hdmaps.policies.master import load_models
from hdmaps.protocol import (
    SCALE_CONFIGS,
    SIM_DURATION,
    SIM_MAX_STEPS,
    SIM_UNIFORM_SEED,
    SIM_VARIED_SEED,
    STAG_COORD_RADIUS_M,
    STAG_EPISODES,
    STAG_SEED,
)
from hdmaps.scenarios.regional import generate_suite
from hdmaps.scenarios.staggered import generate_staggered_scenario, horizon_for_connector
from hdmaps.statistics.metrics import crash_rate, mean_sem, round_wilson, wilson_interval


def _summarize(rows: list[dict], n_int: int, n_agents: int) -> dict:
    arrivals = np.array([r["arrival_pct"] for r in rows], dtype=float)
    crashed = np.array([r["crashed"] for r in rows], dtype=int)
    crash_pct, k, n = crash_rate(crashed)
    lo, hi = round_wilson(*wilson_interval(k, n))
    arr_mean, arr_sem = mean_sem(arrivals)
    return {
        "fleet": n_agents,
        "n_int": n_int,
        "hierarchy": hierarchy_label(n_int),
        "n_episodes": n,
        "crash_pct": crash_pct,
        "crash_k": k,
        "crash_lo": lo,
        "crash_hi": hi,
        "arrival_pct": arr_mean,
        "arrival_sem": arr_sem,
        "episodes": rows,
    }


def evaluate_simultaneous(*, smoke: bool = False, out_dir: Path | None = None) -> list[dict]:
    master, agent = load_models(str(agent_checkpoint()), str(master_checkpoint()))
    configs = SCALE_CONFIGS[:1] if smoke else SCALE_CONFIGS
    n_ep = 1 if smoke else 100
    results = []
    for n_int, n_agents in configs:
        cell = Corridor(n_int, n_agents, connector_length=150, duration=SIM_DURATION)
        rows = []
        for varied, seed in ((True, SIM_VARIED_SEED), (False, SIM_UNIFORM_SEED)):
            rng = np.random.default_rng(seed)
            scenarios = generate_suite(n_int, n_agents, n_ep, rng, varied=varied)
            for sc in scenarios:
                rows.append(run_simultaneous_episode(cell, sc, master, agent, max_steps=SIM_MAX_STEPS))
                if smoke:
                    break
            if smoke:
                break
        cell.close()
        results.append(_summarize(rows, n_int, n_agents))
        if smoke:
            break
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "simultaneous_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def evaluate_staggered(*, smoke: bool = False, out_dir: Path | None = None) -> list[dict]:
    master, agent = load_models(str(agent_checkpoint()), str(master_checkpoint()))
    configs = SCALE_CONFIGS[:1] if smoke else SCALE_CONFIGS
    n_ep = 1 if smoke else STAG_EPISODES
    results = []
    for n_int, n_agents in configs:
        rng = np.random.default_rng(STAG_SEED)
        scenarios = [generate_staggered_scenario(n_int, n_agents, rng, varied=True) for _ in range(n_ep)]
        connector = int(scenarios[0]["connector_length"])
        duration = int(scenarios[0]["max_steps"])
        cell = Corridor(n_int, n_agents, connector_length=connector, duration=max(SIM_DURATION, duration))
        full_rows = []
        zero_rows = []
        for sc in scenarios:
            horizon = int(sc.get("max_steps") or horizon_for_connector(connector))
            full_rows.append(
                run_staggered_episode(
                    cell, sc, master, agent, max_steps=horizon, condition="normal", coord_radius=STAG_COORD_RADIUS_M
                )
            )
            zero_rows.append(
                run_staggered_episode(
                    cell, sc, master, agent, max_steps=horizon, condition="zero_master", coord_radius=STAG_COORD_RADIUS_M
                )
            )
            if smoke:
                break
        cell.close()
        summary = _summarize(full_rows, n_int, n_agents)
        zero = _summarize(zero_rows, n_int, n_agents)
        summary["zero_crash_pct"] = zero["crash_pct"]
        summary["zero_crash_k"] = zero["crash_k"]
        results.append(summary)
        if smoke:
            break
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "staggered_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
