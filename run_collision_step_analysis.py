"""
Analyze the "step 14" collision pattern from a collision dataset CSV.

Run after collision_dataset (CLI or IDE Run).

Edit paths below.
"""

from __future__ import annotations

import json
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.diagnostics.collision_step_analysis import analyze_collision_step_pattern, print_report

# ── CONFIG ───────────────────────────────────────────────────────────────
CSV_PATH = "experiments/collision_steps_dataset.csv"
EPISODE_SUMMARY_JSON = "experiments/collision_episode_summary.json"
OUT_JSON = "experiments/collision_step14_analysis.json"
# Also report scenarios co-occurring with new crashes on these steps:
HIGHLIGHT_STEPS = (12, 13, 14, 15, 16)
# ───────────────────────────────────────────────────────────────────────────


def main() -> None:
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(
            f"Missing {CSV_PATH!r}. Run python run_collision_dataset.py first."
        )
    summ = EPISODE_SUMMARY_JSON if os.path.isfile(EPISODE_SUMMARY_JSON) else None
    rep = analyze_collision_step_pattern(
        CSV_PATH,
        summ,
        highlight_steps=HIGHLIGHT_STEPS,
    )
    print_report(rep)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
