from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hdmaps.hierarchy.tree import hierarchy_label
from hdmaps.paths import simultaneous_dir
from hdmaps.protocol import PUBLISHED_TABLE, SCALE_CONFIGS
from hdmaps.statistics.metrics import round_wilson, wilson_interval


def _suite_configs(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["n_agents"]): row for row in data["configs"]}


def load_simultaneous_scale(n_agents: int) -> dict:
    varied = _suite_configs(simultaneous_dir() / "varied" / "results.json")[n_agents]
    uniform = _suite_configs(simultaneous_dir() / "uniform" / "results.json")[n_agents]
    n1 = n2 = 100
    crash_k = int(round(varied["crash_rate_pct"] / 100.0 * n1 + uniform["crash_rate_pct"] / 100.0 * n2))
    n = n1 + n2
    crash_pct = 100.0 * crash_k / n
    lo, hi = round_wilson(*wilson_interval(crash_k, n))
    arr_mean = 0.5 * (float(varied["arrival_pct_mean"]) + float(uniform["arrival_pct_mean"]))
    std1 = float(varied["arrival_pct_std"])
    std2 = float(uniform["arrival_pct_std"])
    pooled_std = float(np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n - 2))) if n > 2 else 0.0
    arr_sem = pooled_std / np.sqrt(n) if n else 0.0
    n_int = next(ni for ni, na in SCALE_CONFIGS if na == n_agents)
    return {
        "fleet": n_agents,
        "n_int": n_int,
        "hierarchy": hierarchy_label(n_int),
        "n_episodes": n,
        "crash_pct": crash_pct,
        "crash_k": crash_k,
        "crash_lo": lo,
        "crash_hi": hi,
        "arrival_pct": arr_mean,
        "arrival_sem": float(arr_sem),
        "steps_mean": 0.5 * (float(varied["steps_mean"]) + float(uniform["steps_mean"])),
    }


def simultaneous_table() -> list[dict]:
    return [load_simultaneous_scale(n) for _, n in SCALE_CONFIGS]


def published_staggered_table() -> list[dict]:
    rows = []
    for row in PUBLISHED_TABLE:
        rows.append(
            {
                "fleet": row.fleet,
                "hierarchy": row.hierarchy,
                "crash_pct": row.stag_crash_pct,
                "crash_lo": row.stag_crash_lo,
                "crash_hi": row.stag_crash_hi,
                "source": "published_reference",
            }
        )
    return rows


def format_table(sim_rows: list[dict], stag_rows: list[dict] | None = None) -> str:
    lines = [
        "Fleet  Hierarchy    Sim crash              Sim arrival         Stag crash",
        "-" * 88,
    ]
    stag_by_fleet = {r["fleet"]: r for r in (stag_rows or [])}
    for row in sim_rows:
        stag = stag_by_fleet.get(row["fleet"])
        stag_txt = ""
        if stag is not None:
            stag_txt = f"{stag['crash_pct']:4.1f} [{stag['crash_lo']:4.1f}, {stag['crash_hi']:4.1f}]"
        lines.append(
            f"{row['fleet']:>5}  {row['hierarchy']:<12} "
            f"{row['crash_pct']:4.1f} [{row['crash_lo']:4.1f}, {row['crash_hi']:4.1f}]   "
            f"{row['arrival_pct']:5.1f} +/- {row['arrival_sem']:3.1f}     "
            f"{stag_txt}"
        )
    if not sim_rows and stag_rows:
        lines = ["Fleet  Hierarchy    Stag crash", "-" * 40]
        for stag in stag_rows:
            lines.append(
                f"{stag['fleet']:>5}  {stag.get('hierarchy',''):<12} "
                f"{stag['crash_pct']:4.1f} [{stag['crash_lo']:4.1f}, {stag['crash_hi']:4.1f}]"
            )
    return "\n".join(lines)
