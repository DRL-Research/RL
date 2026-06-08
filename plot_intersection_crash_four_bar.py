"""
Rebuild crash-rate bar chart from ckpt_id6 intersection eval with custom condition labels.

Reads:
  MODELS_EVALUATION/eval_2026_05_10-14_31_15/ckpt_id6/intersection/env_eval_summary.json

Writes (repo root):
  intersection_crash_rate_four_conditions.png

Standalone (matplotlib + numpy only; no project env imports).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_SUMMARY = os.path.join(
    _REPO,
    "MODELS_EVALUATION",
    "eval_2026_05_10-14_31_15",
    "ckpt_id6",
    "intersection",
    "env_eval_summary.json",
)
_OUT_PNG = os.path.join(_REPO, "intersection_crash_rate_four_conditions.png")

_CONDITION_ORDER = (
    "normal",
    "zero_master",
    "const_all_masters",
    "swap_local_masters",
)

_LABEL_OVERRIDE: dict[str, str] = {
    "normal": "Full architecture",
    "zero_master": "Zero proto action",
    "const_all_masters": "Const 9999 proto action",
    "swap_local_masters": "Swap LM1–LM2",
}


def main() -> None:
    summary_path = os.path.join(_REPO, os.path.normpath(_DEFAULT_SUMMARY))
    if len(sys.argv) > 1:
        summary_path = os.path.normpath(os.path.abspath(sys.argv[1]))
    if not os.path.isfile(summary_path):
        sys.exit(f"Missing {summary_path}")

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    crash = data.get("mean_crash_episode_pct_by_condition") or {}
    means = {k: float(crash[k]) for k in _CONDITION_ORDER if k in crash}
    if len(means) != len(_CONDITION_ORDER):
        sys.exit(f"Expected crash % for all conditions; got keys {list(means.keys())}")

    keys = list(_CONDITION_ORDER)
    vals = [means[k] for k in keys]
    labels = [_LABEL_OVERRIDE[k] for k in keys]

    title = "n=100"

    try:
        cmap = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab10")
    cols = [cmap(i % 10) for i in range(len(keys))]

    wide = float(max(8.5, min(16.0, 1.4 * len(keys) + 2.0)))
    fig, ax = plt.subplots(figsize=(wide, 5.2))
    x = np.arange(len(keys))
    ax.bar(x, vals, color=cols)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Episodes with crash % (mean 0/1)")
    ax.set_title(title)
    ax.set_ylim(0, max(100.0, max(vals) * 1.08))
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(_OUT_PNG, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {_OUT_PNG}")


if __name__ == "__main__":
    main()
