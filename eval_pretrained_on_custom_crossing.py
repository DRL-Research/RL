"""
Evaluate pretrained agent+master checkpoints on the same custom crossing-heavy test pool
used by run_g03_crossing_quad_bundle (four coordination ablations).

Usage (pick one):
  1) Set PASTE_BEST_MODEL_PATH below to your .../best_model folder, then run:
       py -3.11 eval_pretrained_on_custom_crossing.py
  2) Pass the folder path as the only required argument:
       py -3.11 eval_pretrained_on_custom_crossing.py "C:\\...\\Q02_...\\best_model"

You can paste either the best_model directory itself or the experiment folder that contains it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

# Windows: first `import torch` loads many DLLs (LoadLibraryExW); can take 10–30+ s —
# Ctrl+C during that pause looks like a crash but is KeyboardInterrupt only.
print(
    "Loading PyTorch and project modules (wait 10–30 s on first import; avoid Ctrl+C)...",
    file=sys.stderr,
    flush=True,
)
import run_proto_action_sweep as rps  # noqa: E402
import run_g03_crossing_quad_bundle as bundle  # noqa: E402

# Optional: paste Windows path here and run the script with no arguments.
PASTE_BEST_MODEL_PATH = r"experiment_runs\grid_23_04_2026-13_51_18\Q03_ep1_ent001_PL75\best_model"

DEFAULT_SCENARIO_PKL = os.path.join(
    _REPO,
    "experiment_runs",
    "G03_CROSSING_QUAD_2026_05_03-23_39_21",
    "custom_crossing_test_scenarios.pkl",
)


def _prefix_if_pair_ok(prefix: str) -> str | None:
    if os.path.isfile(f"{prefix}_agent.pth") and os.path.isfile(f"{prefix}_master.pth"):
        return prefix
    return None


def resolve_checkpoint_prefix(user_path: str) -> str:
    """Resolve P such that P_agent.pth and P_master.pth exist.

    Accepts either the experiment root (contains best_model/) or the best_model folder directly.
    """
    base = os.path.normpath(os.path.abspath(os.path.expanduser(user_path)))
    candidates: list[str] = []
    for name in ("checkpoint", "ckpt", "trained_model"):
        candidates.append(os.path.join(base, name))
    candidates.extend(
        [
            os.path.join(base, "best_model", "checkpoint"),
            os.path.join(base, "best", "ckpt"),
            os.path.join(base, "trained_model"),
        ]
    )
    seen: set[str] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        ok = _prefix_if_pair_ok(p)
        if ok is not None:
            return ok
    raise FileNotFoundError(
        f"No agent+master checkpoint pair found under or next to:\n  {base}\n"
        f"Expected e.g. .../best_model/checkpoint_agent.pth and checkpoint_master.pth"
    )


def experiment_home_from_input(user_path: str) -> str:
    """Directory that represents the run (parent of best_model if user pasted best_model)."""
    ap = os.path.normpath(os.path.abspath(os.path.expanduser(user_path)))
    if os.path.basename(ap).lower() == "best_model":
        return os.path.dirname(ap)
    return ap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints from a best_model (or experiment) folder on G03 custom crossing pickle.",
    )
    parser.add_argument(
        "best_model_or_experiment",
        nargs="?",
        default=(os.environ.get("BEST_MODEL_OR_EXPERIMENT_PATH", "").strip() or PASTE_BEST_MODEL_PATH.strip()),
        metavar="PATH",
        help='Folder: .../best_model or experiment root. Or set PASTE_BEST_MODEL_PATH / env BEST_MODEL_OR_EXPERIMENT_PATH.',
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="",
        help="Override: path without _agent.pth / _suffix (skip auto-discovery).",
    )
    parser.add_argument(
        "--scenarios-pickle",
        type=str,
        default=DEFAULT_SCENARIO_PKL,
        help="custom_crossing_test_scenarios.pkl from a G03 crossing bundle.",
    )
    parser.add_argument(
        "--g03-config",
        type=str,
        default=bundle.G03_DEFAULT_CONFIG,
        help="Proto config JSON (hyperparams + obs sizes).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Where to write CSV/JSON/PNG; default: <model-dir>/custom_crossing_eval",
    )
    parser.add_argument("--eval-run-id", type=int, default=0, help="Seed offset tag (same as bundle run_id).")
    parser.add_argument(
        "--suite",
        type=str,
        choices=("default", "coordination_extended"),
        default="default",
        help=(
            "coordination_extended adds swap_local_masters, random_local_masters, "
            "zero_global_master, negate_master (after the default quartet)."
        ),
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="",
        metavar="CSV",
        help="Comma-separated run_episode conditions (overrides --suite when non-empty).",
    )
    args = parser.parse_args()
    user_path = (args.best_model_or_experiment or "").strip()
    if not user_path:
        exe = os.path.basename(sys.executable) if sys.executable else "python"
        print(
            "Missing path to best_model (or experiment folder).\n\n"
            "  Option A — edit this file: set PASTE_BEST_MODEL_PATH = r\"...\\\\best_model\"\n"
            f"  Option B — run: {exe} eval_pretrained_on_custom_crossing.py \"<PATH>\"\n"
            "  Option C — env: set BEST_MODEL_OR_EXPERIMENT_PATH then run with no args.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    scenarios_path = os.path.normpath(os.path.abspath(os.path.expanduser(args.scenarios_pickle)))
    scenarios = bundle.load_scenarios_pickle(scenarios_path)
    n_ev = len(scenarios)

    cfg_path = os.path.normpath(os.path.abspath(os.path.expanduser(args.g03_config)))
    cfg = bundle.load_json(cfg_path)
    cfg.setdefault("master_broadcast_const_test", 9999.0)

    if (args.checkpoint_prefix or "").strip():
        ckpt_prefix = os.path.normpath(os.path.abspath(os.path.expanduser(args.checkpoint_prefix.strip())))
        experiment_home = experiment_home_from_input(user_path)
    else:
        ckpt_prefix = resolve_checkpoint_prefix(user_path)
        experiment_home = experiment_home_from_input(user_path)

    out_dir = args.out_dir or os.path.join(experiment_home, "custom_crossing_eval")
    out_dir = os.path.normpath(os.path.abspath(os.path.expanduser(out_dir)))
    os.makedirs(out_dir, exist_ok=True)

    rps.set_all_seeds(123)
    proto_exp = rps.ProtoExperiment(cfg, out_dir)
    master_model, agent_model = rps.make_models(proto_exp)
    if not rps.load_models(agent_model, master_model, ckpt_prefix):
        print(f"Failed to load from {ckpt_prefix}", file=sys.stderr)
        sys.exit(1)

    chosen = bundle.parse_conditions_csv(args.conditions) if (args.conditions or "").strip() else (
        bundle.default_plus_coordination_probes_order()
        if args.suite == "coordination_extended"
        else bundle.CUSTOM_TEST_CONDITIONS
    )

    eval_pm_backup = bool(proto_exp.cfg.get("eval_policy_mean", False))
    rows = bundle.run_custom_eval(
        proto_exp,
        master_model,
        agent_model,
        scenarios,
        run_dir=out_dir,
        run_id=int(args.eval_run_id),
        eval_policy_mean_backup=eval_pm_backup,
        n_eval_scenarios=n_ev,
        conditions=chosen,
    )

    means: dict[str, float] = {}
    for cond in chosen:
        sub = [r for r in rows if r["condition"] == cond]
        means[cond] = float(np.mean([float(x["arrival_pct"]) for x in sub])) if sub else 0.0

    tag = os.path.basename(experiment_home.rstrip(os.sep))
    chart_path = (
        os.path.join(out_dir, "crossing_ablation_bars.png")
        if len(chosen) > len(bundle.CUSTOM_TEST_CONDITIONS)
        else os.path.join(out_dir, "four_bar_custom_crossing_pretrained.png")
    )
    bundle.plot_crossing_condition_bars(
        means,
        chart_path,
        condition_order=list(chosen),
        chart_title=f"Custom crossing test — {tag}",
        y_axis_label=f"Mean arrival % ({n_ev} scenarios)",
    )

    summary: dict[str, Any] = {
        "best_model_or_experiment_input": os.path.normpath(os.path.abspath(os.path.expanduser(user_path))),
        "experiment_home": experiment_home,
        "checkpoint_prefix": ckpt_prefix,
        "scenarios_pickle": scenarios_path,
        "g03_config": cfg_path,
        "n_scenarios": n_ev,
        "eval_run_id": int(args.eval_run_id),
        "suite": ("custom_csv" if (args.conditions or "").strip() else args.suite),
        "conditions": list(chosen),
        "mean_arrival_pct_by_condition": means,
        "coordination_probe_rationale_ref": (
            "run_g03_crossing_quad_bundle.py — COORDINATION_PROBE_CONDITIONS docstring "
            "(swap / random_local / zero_global_master / negate_master)."
        ),
    }
    with open(os.path.join(out_dir, "custom_crossing_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote metrics + plot under: {out_dir}")
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
