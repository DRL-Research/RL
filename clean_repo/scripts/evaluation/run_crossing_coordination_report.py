"""
One-shot full crossing coordination report: eval all standard + probe conditions,
write CSV, JSON, and paired arrival / crash-rate dashboards (8-bar + 4-bar quartet).

Run from repo root (edit path or set env COORDINATION_REPORT_MODEL):

  py -3.11 run_crossing_coordination_report.py "experiment_runs/G03_CROSSING_QUAD_2026_05_03-23_39_21/run_04/best"

Batch (master_dir + agent_dir, matching stems foo_master.pth ↔ foo_agent.pth):

  py -3.11 run_crossing_coordination_report.py ^
    --batch-master-dir PATH/TO/masters ^
    --batch-agent-dir PATH/TO/agents ^
    --batch-output-root PATH/OUT

Outputs under <experiment_home>/crossing_coordination_report/ (single run),
or under <batch-output-root>/<stem>/crossing_coordination_report/ (batch).
Batch also writes batch_summary.json at --batch-output-root.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# repo root + every scripts/ sub-folder go on sys.path so cross-script imports
# (e.g. ``import run_scalability_suite``) keep working from any category folder.
for _p in (
    _REPO,
    os.path.join(_REPO, "scripts", "training"),
    os.path.join(_REPO, "scripts", "evaluation"),
    os.path.join(_REPO, "scripts", "visualization"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO)

from eval_pretrained_on_custom_crossing import (  # noqa: E402
    DEFAULT_SCENARIO_PKL,
    PASTE_BEST_MODEL_PATH,
    experiment_home_from_input,
    resolve_checkpoint_prefix,
)
import run_proto_action_sweep as rps  # noqa: E402
import run_g03_crossing_quad_bundle as bundle  # noqa: E402
from src.model.model_handler import load_models_from_paths  # noqa: E402

_INVALID_FS = re.compile(r'[<>:"/\\|?*]')


def _safe_batch_subdir(stem: str) -> str:
    s = (stem or "model").strip() or "model"
    s = _INVALID_FS.sub("_", s)
    s = s.rstrip(" .")
    return s[:200] if len(s) > 200 else s


def run_coordination_report(
    *,
    scenarios_path: str,
    cfg_path: str,
    eval_run_id: int,
    out_dir: str,
    checkpoint_prefix: str | None,
    agent_pth: str | None,
    master_pth: str | None,
    experiment_home: str | None,
    chart_tag: str,
) -> dict[str, Any]:
    """Run eval + plots; load from checkpoint_prefix XOR (agent_pth, master_pth)."""
    cp = (checkpoint_prefix or "").strip()
    ap = (agent_pth or "").strip()
    mp = (master_pth or "").strip()
    has_prefix = bool(cp)
    has_split = bool(ap and mp)
    if has_prefix == has_split:
        raise ValueError("Provide exactly one of: checkpoint_prefix, or agent_pth+master_pth")

    full_order = bundle.default_plus_coordination_probes_order()
    quartet_order = list(bundle.CUSTOM_TEST_CONDITIONS)

    scenarios_path = os.path.normpath(os.path.abspath(os.path.expanduser(scenarios_path)))
    scenarios = bundle.load_scenarios_pickle(scenarios_path)
    n_ev = len(scenarios)

    cfg_path = os.path.normpath(os.path.abspath(os.path.expanduser(cfg_path)))
    cfg = bundle.load_json(cfg_path)
    cfg.setdefault("master_broadcast_const_test", 9999.0)

    out_dir = os.path.normpath(os.path.abspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    rps.set_all_seeds(123)
    proto_exp = rps.ProtoExperiment(cfg, out_dir)
    master_model, agent_model = rps.make_models(proto_exp)

    if has_prefix:
        ok = rps.load_models(agent_model, master_model, cp)
    else:
        ok = load_models_from_paths(agent_model, master_model, ap, mp)
    if not ok:
        raise RuntimeError("Failed to load model weights")

    eval_pm_backup = bool(proto_exp.cfg.get("eval_policy_mean", False))
    rows = bundle.run_custom_eval(
        proto_exp,
        master_model,
        agent_model,
        scenarios,
        run_dir=out_dir,
        run_id=int(eval_run_id),
        eval_policy_mean_backup=eval_pm_backup,
        n_eval_scenarios=n_ev,
        conditions=full_order,
    )

    arrival = bundle.rows_mean_numeric_by_condition(rows, "arrival_pct")
    crash_frac = bundle.rows_mean_numeric_by_condition(rows, "crashed", scale=1.0)
    crash_pct = {k: 100.0 * v for k, v in crash_frac.items()}

    tag = chart_tag
    bundle.plot_crossing_dashboard_two_metrics(
        arrival,
        crash_pct,
        os.path.join(out_dir, "dashboard_full_arrival_and_crash.png"),
        condition_order=full_order,
        primary_ylabel="Mean arrival %",
        secondary_ylabel="Episodes with crash % (mean 0/1)",
        chart_title=f"Full suite — {tag} (n={n_ev} scenarios × {len(full_order)} conditions)",
    )
    bundle.plot_crossing_dashboard_two_metrics(
        arrival,
        crash_pct,
        os.path.join(out_dir, "dashboard_quartet_arrival_and_crash.png"),
        condition_order=quartet_order,
        primary_ylabel="Mean arrival %",
        secondary_ylabel="Episodes with crash % (mean 0/1)",
        chart_title=f"Canonical quartet — {tag}",
    )

    eh = experiment_home or out_dir
    summary: dict[str, Any] = {
        "experiment_home": eh,
        "checkpoint_prefix": cp,
        "agent_pth": ap,
        "master_pth": mp,
        "scenarios_pickle": scenarios_path,
        "g03_config": cfg_path,
        "n_scenarios": n_ev,
        "eval_run_id": int(eval_run_id),
        "condition_order_full": list(full_order),
        "condition_order_quartet": quartet_order,
        "mean_arrival_pct_by_condition": arrival,
        "mean_crash_episode_fraction_by_condition": crash_frac,
        "mean_crash_episode_pct_by_condition": crash_pct,
    }
    with open(os.path.join(out_dir, "coordination_report_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _iter_batch_pairs(master_dir: str, agent_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (stem, master_path, agent_path) for files matching *_master.pth."""
    master_dir = os.path.normpath(os.path.abspath(os.path.expanduser(master_dir)))
    agent_dir = os.path.normpath(os.path.abspath(os.path.expanduser(agent_dir)))
    pattern = os.path.join(master_dir, "*_master.pth")
    pairs: list[tuple[str, str, str]] = []
    for mp in sorted(glob.glob(pattern)):
        base = os.path.basename(mp)
        if not base.endswith("_master.pth"):
            continue
        stem = base[: -len("_master.pth")]
        ap = os.path.join(agent_dir, f"{stem}_agent.pth")
        pairs.append((stem, mp, ap))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full coordination ablation report (8 conditions + charts). "
            "Pass checkpoint folder or experiment root; set COORDINATION_REPORT_MODEL to skip arg. "
            "Or use --batch-master-dir / --batch-agent-dir / --batch-output-root."
        )
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=(os.environ.get("COORDINATION_REPORT_MODEL", "").strip() or PASTE_BEST_MODEL_PATH.strip()),
        metavar="PATH",
        help="best_model/, .../best, or experiment root containing checkpoints.",
    )
    parser.add_argument("--scenarios-pickle", type=str, default=DEFAULT_SCENARIO_PKL)
    parser.add_argument("--g03-config", type=str, default=bundle.G03_DEFAULT_CONFIG)
    parser.add_argument("--eval-run-id", type=int, default=0)
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="",
        help="Explicit stem without _agent.pth (optional).",
    )
    parser.add_argument("--batch-master-dir", type=str, default="", metavar="DIR")
    parser.add_argument("--batch-agent-dir", type=str, default="", metavar="DIR")
    parser.add_argument("--batch-output-root", type=str, default="", metavar="DIR")
    args = parser.parse_args()

    batch_master = (args.batch_master_dir or "").strip()
    batch_agent = (args.batch_agent_dir or "").strip()
    batch_out = (args.batch_output_root or "").strip()
    batch_mode = bool(batch_master or batch_agent or batch_out)
    if batch_mode:
        if not (batch_master and batch_agent and batch_out):
            print(
                "Batch mode requires all three: --batch-master-dir, --batch-agent-dir, --batch-output-root",
                file=sys.stderr,
            )
            sys.exit(2)
        batch_out = os.path.normpath(os.path.abspath(os.path.expanduser(batch_out)))
        os.makedirs(batch_out, exist_ok=True)
        pairs = _iter_batch_pairs(batch_master, batch_agent)
        batch_summary: dict[str, Any] = {
            "batch_master_dir": batch_master,
            "batch_agent_dir": batch_agent,
            "batch_output_root": batch_out,
            "n_master_files": len(pairs),
            "runs": [],
        }
        for stem, master_pth, agent_pth in pairs:
            sub = _safe_batch_subdir(stem)
            pair_out = os.path.join(batch_out, sub, "crossing_coordination_report")
            entry: dict[str, Any] = {
                "stem": stem,
                "master_pth": master_pth,
                "agent_pth": agent_pth,
                "out_dir": pair_out,
                "ok": False,
                "error": "",
            }
            if not os.path.isfile(agent_pth):
                entry["error"] = "missing_agent_file"
                batch_summary["runs"].append(entry)
                print(f"[skip] {stem}: no {agent_pth}", file=sys.stderr)
                continue
            try:
                run_coordination_report(
                    scenarios_path=args.scenarios_pickle,
                    cfg_path=args.g03_config,
                    eval_run_id=int(args.eval_run_id),
                    out_dir=pair_out,
                    checkpoint_prefix=None,
                    agent_pth=agent_pth,
                    master_pth=master_pth,
                    experiment_home=os.path.join(batch_out, sub),
                    chart_tag=sub,
                )
                entry["ok"] = True
            except Exception as exc:  # noqa: BLE001
                entry["error"] = str(exc)
                print(f"[fail] {stem}: {exc}", file=sys.stderr)
            batch_summary["runs"].append(entry)

        summary_path = os.path.join(batch_out, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2)
        n_ok = sum(1 for r in batch_summary["runs"] if r.get("ok"))
        print(f"Batch done. {n_ok}/{len(batch_summary['runs'])} OK. Summary:\n  {summary_path}")
        return

    user_path = (args.model_path or "").strip()
    if not user_path:
        print(
            "Missing model path. Example:\n"
            '  py -3.11 run_crossing_coordination_report.py "experiment_runs/.../run_04/best"\n'
            "Or: set env COORDINATION_REPORT_MODEL\n"
            "Or: --batch-master-dir ... --batch-agent-dir ... --batch-output-root ...",
            file=sys.stderr,
        )
        sys.exit(2)

    if (args.checkpoint_prefix or "").strip():
        ckpt_prefix = os.path.normpath(os.path.abspath(os.path.expanduser(args.checkpoint_prefix.strip())))
    else:
        ckpt_prefix = resolve_checkpoint_prefix(user_path)
    experiment_home = experiment_home_from_input(user_path)
    out_dir = os.path.join(experiment_home, "crossing_coordination_report")
    tag = os.path.basename(experiment_home.rstrip(os.sep))

    try:
        summary = run_coordination_report(
            scenarios_path=args.scenarios_pickle,
            cfg_path=args.g03_config,
            eval_run_id=int(args.eval_run_id),
            out_dir=out_dir,
            checkpoint_prefix=ckpt_prefix,
            agent_pth=None,
            master_pth=None,
            experiment_home=experiment_home,
            chart_tag=tag,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(f"Done. Report directory:\n  {out_dir}")
    print(
        json.dumps(
            {
                "arrival_pct": summary["mean_arrival_pct_by_condition"],
                "crash_episode_pct": summary["mean_crash_episode_pct_by_condition"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
