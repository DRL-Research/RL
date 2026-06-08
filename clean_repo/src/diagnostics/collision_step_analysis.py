# Code and comments only in English.

"""
Explain and quantify the common "first collision at step ~14" pattern.

In HighwayEnv with default policy_frequency=1 Hz, each env.step() advances simulation time by
1 second. So "step 14" ≈ 14 s into the episode — often when vehicles from BASE_LONG=40 m
reach the conflict zone, not a magic off-by-one bug.

This module reads the CSV from run_collision_dataset.py and the episode summary JSON.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any, Dict, List, Optional


def analyze_collision_step_pattern(
    csv_path: str,
    episode_summary_path: Optional[str] = None,
    *,
    highlight_steps: tuple = (14,),
) -> Dict[str, Any]:
    """
    Build histograms of (a) which *step index* first shows any_new_crash per episode,
    (b) all steps where any_new_crash=1 across dataset,
    (c) for highlight steps, which scenario_index values co-occur.
    """
    # Per-episode: first step with any_new_crash
    first_new_crash: Dict[int, int] = {}
    all_new_crash_steps: Counter = Counter()
    scenario_at_highlight: Counter = Counter()
    rows = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += 1
            ep = int(r["episode"])
            st = int(r["step"])
            if int(r.get("any_new_crash", 0)):
                all_new_crash_steps[st] += 1
                if ep not in first_new_crash:
                    first_new_crash[ep] = st
                if st in highlight_steps:
                    scenario_at_highlight[int(r.get("scenario_index", -1))] += 1

    first_hist = Counter(first_new_crash.values())
    total_eps = len(first_new_crash)
    at_14 = first_hist.get(14, 0)
    frac_14 = (at_14 / total_eps) if total_eps else 0.0

    report: Dict[str, Any] = {
        "csv_rows_read": rows,
        "episodes_with_at_least_one_new_crash": total_eps,
        "histogram_first_new_crash_step": dict(sorted(first_hist.items())),
        "fraction_of_those_episodes_first_crash_at_step_14": round(frac_14, 4),
        "count_first_crash_at_step_14": at_14,
        "histogram_all_rows_with_any_new_crash_by_step": dict(
            sorted(all_new_crash_steps.items())
        ),
        "scenario_index_counts_on_highlight_steps": {
            str(k): v for k, v in scenario_at_highlight.most_common(25)
        },
        "note": (
            "With policy_frequency=1 (HighwayEnv default), each row step ≈ 1 s sim time. "
            "Many collisions around step 12–16 often mean vehicles meet at the intersection "
            "after ~12–16 s of travel from typical spawn distances — compare EPISODE_MAX_TIME "
            "(duration) and initial offsets (e.g. BASE_LONG=40 in intersection_class)."
        ),
    }

    if episode_summary_path:
        try:
            with open(episode_summary_path, encoding="utf-8") as jf:
                data = json.load(jf)
            summ = data.get("episode_summaries") or []
            fc = [e["first_collision_step"] for e in summ if e.get("first_collision_step")]
            c2 = Counter(fc)
            report["from_episode_summary_first_collision_step"] = dict(sorted(c2.items()))
        except OSError:
            pass

    return report


def print_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))
