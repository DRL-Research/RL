"""
Evaluate every matched agent/master checkpoint pair on three environments:
  RELintersection-v0, RELroundabout-v0, RELdouble-intersection-v0

Each environment uses its own native scenario pool (topology-correct). The same
coordination condition suite as G03 (quartet + probes) runs on every env.

Checkpoint pairing:
  • Prefer numeric suffixes: foo_agent7.pth ↔ foo_master7.pth (same prefix + ID)
  • Fallback: identical stem keys from ..._agent.pth / ..._master.pth

Usage (repo root):
  py -3.11 run_models_evaluation_suite.py

Outputs under MODELS_EVALUATION/<timestamp>/<pair_slug>/...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from eval_pretrained_on_custom_crossing import DEFAULT_SCENARIO_PKL  # noqa: E402

import run_g03_crossing_quad_bundle as bundle  # noqa: E402 — pulls registered envs via rps
import run_proto_action_sweep as rps  # noqa: E402
from highwayenv.intersection_class import rotate_scenario_clockwise  # noqa: E402
from src.model.model_handler import load_models_from_paths  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from src.experiment.scenarios import (  # noqa: E402
    DOUBLE_INTERSECTION_EXCLUDED_INDICES,
    DOUBLE_INTERSECTION_HELD_OUT_INDICES,
    roundabout_base_scenarios,
    ROUNDABOUT_HELD_OUT_INDICES,
    double_intersection_base_scenarios,
)

_INVALID = re.compile(r'[<>:"/\\|?*]')
RE_DIGIT_AGENT = re.compile(r"^(.+?)(agent)(\d+)\.pth$", re.IGNORECASE)
RE_DIGIT_MASTER = re.compile(r"^(.+?)(master)(\d+)\.pth$", re.IGNORECASE)


def _safe(s: str) -> str:
    x = _INVALID.sub("_", (s or "").strip()) or "pair"
    return x[:180]


def build_intersection_scenario_list(pickle_path: str, cap: int) -> list[dict[str, Any]]:
    data = bundle.load_scenarios_pickle(pickle_path)
    if cap > 0:
        return data[: min(cap, len(data))]
    return data


def build_roundabout_scenario_list(cap: int) -> list[dict[str, Any]]:
    all_regular: list[dict[str, Any]] = []
    for base in roundabout_base_scenarios:
        all_regular.append(base)
        for rotation in (1, 2, 3):
            rotated_agents = [
                rotate_scenario_clockwise([agent], rotation)[0]
                for agent in base["agents"]
            ]
            rotated_static = [
                rotate_scenario_clockwise([st], rotation)[0]
                for st in base["static"]
            ]
            all_regular.append({"agents": rotated_agents, "static": rotated_static})
    active = [
        s
        for i, s in enumerate(all_regular)
        if i not in ROUNDABOUT_HELD_OUT_INDICES
    ]
    if cap > 0:
        return active[:cap]
    return active


def build_double_intersection_scenario_list(cap: int) -> list[dict[str, Any]]:
    excl = DOUBLE_INTERSECTION_HELD_OUT_INDICES | DOUBLE_INTERSECTION_EXCLUDED_INDICES
    active = [s for i, s in enumerate(double_intersection_base_scenarios) if i not in excl]
    if cap > 0:
        return active[:cap]
    return active


def build_double_intersection_held_out_scenario_list(cap: int) -> list[dict[str, Any]]:
    """Regular-scenario indices held out from training (see DOUBLE_INTERSECTION_HELD_OUT_INDICES)."""
    idx = sorted(DOUBLE_INTERSECTION_HELD_OUT_INDICES)
    held = [double_intersection_base_scenarios[i] for i in idx]
    if cap > 0:
        return held[: min(cap, len(held))]
    return held


def discover_digit_pairs(agent_dir: Path, master_dir: Path) -> dict[tuple[str, int], tuple[Path, Path]]:
    agents: dict[tuple[str, int], Path] = {}
    masters: dict[tuple[str, int], Path] = {}
    for fp in agent_dir.glob("*.pth"):
        m = RE_DIGIT_AGENT.match(fp.name)
        if not m:
            continue
        key = (m.group(1).lower(), int(m.group(3)))
        agents[key] = fp
    for fp in master_dir.glob("*.pth"):
        m = RE_DIGIT_MASTER.match(fp.name)
        if not m:
            continue
        key = (m.group(1).lower(), int(m.group(3)))
        masters[key] = fp
    pairs: dict[tuple[str, int], tuple[Path, Path]] = {}
    for key, ap in agents.items():
        mp = masters.get(key)
        if mp:
            pairs[key] = (ap, mp)
    return pairs


def discover_stem_pairs(agent_dir: Path, master_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Match foo_agent.pth with foo_master.pth (same stem foo)."""
    agents: dict[str, Path] = {}
    masters: dict[str, Path] = {}
    for fp in agent_dir.glob("*.pth"):
        if RE_DIGIT_AGENT.match(fp.name):
            continue
        low = fp.name.lower()
        if low.endswith("_agent.pth"):
            stem = fp.name[: -len("_agent.pth")]
            agents[stem.lower()] = fp
    for fp in master_dir.glob("*.pth"):
        if RE_DIGIT_MASTER.match(fp.name):
            continue
        low = fp.name.lower()
        if low.endswith("_master.pth"):
            stem = fp.name[: -len("_master.pth")]
            masters[stem.lower()] = fp
    out: dict[str, tuple[Path, Path]] = {}
    for stem, ap in agents.items():
        mp = masters.get(stem)
        if mp:
            out[stem] = (ap, mp)
    return out


def pair_slug_from_digit(prefix: str, num: int) -> str:
    return _safe(f"{prefix.rstrip('_')}_id{num}")


def resolve_embedding_dim(master_pth: str, agent_pth: str, base_cfg: dict[str, Any]) -> int:
    """Match ProtoExperiment embedding_dim to saved weights (G03 config alone may be wrong)."""
    import torch

    master_abs = os.path.abspath(master_pth)
    legacy = f"{master_abs}_custom_params.pt"
    short = os.path.join(os.path.dirname(master_abs), "master_custom_params.pt")
    for pth in (legacy, short):
        if os.path.isfile(pth):
            try:
                cp = torch.load(pth, map_location="cpu")
                if isinstance(cp, dict) and "embedding_size" in cp:
                    return int(cp["embedding_size"])
            except Exception:
                pass
    try:
        ag = PPO.load(agent_pth, device="cpu")
        od = int(ag.observation_space.shape[0])
        if od >= 5:
            return od - 4
    except Exception:
        pass
    return int(base_cfg.get("embedding_dim", 4))


def eval_pair_all_envs(
    *,
    agent_pth: str,
    master_pth: str,
    pair_out_dir: str,
    g03_config: str,
    scenarios_pickle: str,
    n_scenarios_cap: int,
    eval_run_id: int,
    env_tags: frozenset[str] | None = None,
) -> dict[str, Any]:
    full_order = bundle.default_plus_coordination_probes_order()
    quartet_order = list(bundle.CUSTOM_TEST_CONDITIONS)

    scenarios_path = os.path.normpath(os.path.abspath(os.path.expanduser(scenarios_pickle)))
    cfg_path = os.path.normpath(os.path.abspath(os.path.expanduser(g03_config)))
    base_cfg = bundle.load_json(cfg_path)
    base_cfg.setdefault("master_broadcast_const_test", 9999.0)
    emb = resolve_embedding_dim(master_pth, agent_pth, base_cfg)
    cfg = dict(base_cfg)
    cfg["embedding_dim"] = emb

    cap = max(1, int(n_scenarios_cap))
    pools: list[tuple[str, str, list[dict[str, Any]], str]] = [
        (
            "RELintersection-v0",
            "intersection",
            build_intersection_scenario_list(scenarios_path, cap),
            "coordination_metrics_intersection.csv",
        ),
        (
            "RELroundabout-v0",
            "roundabout",
            build_roundabout_scenario_list(cap),
            "coordination_metrics_roundabout.csv",
        ),
        (
            "RELdouble-intersection-v0",
            "double_intersection",
            build_double_intersection_scenario_list(cap),
            "coordination_metrics_double_intersection.csv",
        ),
    ]

    pair_out_dir = os.path.normpath(pair_out_dir)
    os.makedirs(pair_out_dir, exist_ok=True)

    rps.set_all_seeds(123)
    proto_base_dir = os.path.join(pair_out_dir, "_proto_work")
    os.makedirs(proto_base_dir, exist_ok=True)
    proto_exp = rps.ProtoExperiment(cfg, proto_base_dir)
    master_model, agent_model = rps.make_models(proto_exp)
    if not load_models_from_paths(agent_model, master_model, agent_pth, master_pth):
        raise RuntimeError(f"Failed loading\n  {agent_pth}\n  {master_pth}")

    eval_pm_backup = bool(proto_exp.cfg.get("eval_policy_mean", False))

    pair_summary: dict[str, Any] = {
        "agent_pth": agent_pth,
        "master_pth": master_pth,
        "g03_config": cfg_path,
        "embedding_dim_used": emb,
        "scenarios_pickle_intersection": scenarios_path,
        "n_scenarios_requested_cap": cap,
        "env_tags_filter": sorted(env_tags) if env_tags is not None else None,
        "condition_order_full": list(full_order),
        "condition_order_quartet": quartet_order,
        "environments": {},
    }

    for env_id, tag, scenarios, csv_name in pools:
        if env_tags is not None and tag not in env_tags:
            continue
        n_ev = len(scenarios)
        if n_ev < 1:
            pair_summary["environments"][tag] = {"error": "empty_scenario_pool", "env_id": env_id}
            continue
        env_run_dir = os.path.join(pair_out_dir, tag)
        os.makedirs(env_run_dir, exist_ok=True)

        rows = bundle.run_custom_eval(
            proto_exp,
            master_model,
            agent_model,
            scenarios,
            run_dir=env_run_dir,
            run_id=int(eval_run_id),
            eval_policy_mean_backup=eval_pm_backup,
            n_eval_scenarios=n_ev,
            conditions=full_order,
            env_id=env_id,
            metrics_csv_basename=csv_name,
        )
        arrival = bundle.rows_mean_numeric_by_condition(rows, "arrival_pct")
        crash_frac = bundle.rows_mean_numeric_by_condition(rows, "crashed", scale=1.0)
        crash_pct = {k: 100.0 * v for k, v in crash_frac.items()}

        bundle.plot_crossing_dashboard_two_metrics(
            arrival,
            crash_pct,
            os.path.join(env_run_dir, "dashboard_full_arrival_and_crash.png"),
            condition_order=full_order,
            primary_ylabel="Mean arrival %",
            secondary_ylabel="Episodes with crash % (mean 0/1)",
            chart_title=f"{tag} — full suite (n={n_ev} scenarios × {len(full_order)} conditions)",
        )
        bundle.plot_crossing_dashboard_two_metrics(
            arrival,
            crash_pct,
            os.path.join(env_run_dir, "dashboard_quartet_arrival_and_crash.png"),
            condition_order=quartet_order,
            primary_ylabel="Mean arrival %",
            secondary_ylabel="Episodes with crash % (mean 0/1)",
            chart_title=f"{tag} — canonical quartet",
        )

        js = {
            "env_id": env_id,
            "tag": tag,
            "n_scenarios": n_ev,
            "metrics_csv": os.path.join(env_run_dir, csv_name),
            "mean_arrival_pct_by_condition": arrival,
            "mean_crash_episode_fraction_by_condition": crash_frac,
            "mean_crash_episode_pct_by_condition": crash_pct,
        }
        with open(os.path.join(env_run_dir, "env_eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2)

        pair_summary["environments"][tag] = js

    with open(os.path.join(pair_out_dir, "pair_summary.json"), "w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2)

    return pair_summary


def eval_pair_double_intersection_custom_pool(
    *,
    agent_pth: str,
    master_pth: str,
    pair_out_dir: str,
    g03_config: str,
    scenarios_pickle: str,
    scenarios: list[dict[str, Any]],
    eval_run_id: int,
    scenario_pool_key: str,
    metrics_csv_basename: str,
    chart_suite_descriptor: str,
    pair_summary_extras: dict[str, Any] | None = None,
    coordination_conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    G03 coordination eval on an explicit double-intersection scenario list.

    ``coordination_conditions``: optional subset/order of condition keys (e.g. ``normal``,
    ``zero_master``). Default: full 8-condition suite from ``default_plus_coordination_probes_order``.
    """
    if coordination_conditions is not None:
        full_order = bundle.normalize_eval_conditions(tuple(coordination_conditions))
        quartet_order = list(full_order)
        quartet_title_qualifier = "selected conditions"
    else:
        full_order = bundle.default_plus_coordination_probes_order()
        quartet_order = list(bundle.CUSTOM_TEST_CONDITIONS)
        quartet_title_qualifier = "canonical quartet"

    scenarios_path = os.path.normpath(os.path.abspath(os.path.expanduser(scenarios_pickle)))
    cfg_path = os.path.normpath(os.path.abspath(os.path.expanduser(g03_config)))
    base_cfg = bundle.load_json(cfg_path)
    base_cfg.setdefault("master_broadcast_const_test", 9999.0)
    emb = resolve_embedding_dim(master_pth, agent_pth, base_cfg)
    cfg = dict(base_cfg)
    cfg["embedding_dim"] = emb

    tag = "double_intersection"
    env_id = "RELdouble-intersection-v0"

    pair_out_dir = rps._resolve_long_path(str(pair_out_dir).strip())
    os.makedirs(pair_out_dir, exist_ok=True)

    rps.set_all_seeds(123)
    proto_base_dir = os.path.join(pair_out_dir, "_proto_work")
    os.makedirs(proto_base_dir, exist_ok=True)
    proto_exp = rps.ProtoExperiment(cfg, proto_base_dir)
    master_model, agent_model = rps.make_models(proto_exp)
    if not load_models_from_paths(agent_model, master_model, agent_pth, master_pth):
        raise RuntimeError(f"Failed loading\n  {agent_pth}\n  {master_pth}")

    eval_pm_backup = bool(proto_exp.cfg.get("eval_policy_mean", False))

    pair_summary: dict[str, Any] = {
        "agent_pth": agent_pth,
        "master_pth": master_pth,
        "g03_config": cfg_path,
        "embedding_dim_used": emb,
        "scenarios_pickle_intersection": scenarios_path,
        "double_intersection_pool": scenario_pool_key,
        "condition_order_full": list(full_order),
        "condition_order_quartet": quartet_order,
        "environments": {},
    }
    if pair_summary_extras:
        pair_summary.update(pair_summary_extras)

    n_ev = len(scenarios)
    if n_ev < 1:
        pair_summary["environments"][tag] = {"error": "empty_scenario_pool", "env_id": env_id}
        with open(os.path.join(pair_out_dir, "pair_summary.json"), "w", encoding="utf-8") as f:
            json.dump(pair_summary, f, indent=2)
        return pair_summary

    env_run_dir = rps._resolve_long_path(os.path.join(pair_out_dir, tag))
    os.makedirs(env_run_dir, exist_ok=True)

    rows = bundle.run_custom_eval(
        proto_exp,
        master_model,
        agent_model,
        scenarios,
        run_dir=env_run_dir,
        run_id=int(eval_run_id),
        eval_policy_mean_backup=eval_pm_backup,
        n_eval_scenarios=n_ev,
        conditions=full_order,
        env_id=env_id,
        metrics_csv_basename=metrics_csv_basename,
    )
    arrival = bundle.rows_mean_numeric_by_condition(rows, "arrival_pct")
    crash_frac = bundle.rows_mean_numeric_by_condition(rows, "crashed", scale=1.0)
    crash_pct = {k: 100.0 * v for k, v in crash_frac.items()}

    bundle.plot_crossing_dashboard_two_metrics(
        arrival,
        crash_pct,
        os.path.join(env_run_dir, "dashboard_full_arrival_and_crash.png"),
        condition_order=full_order,
        primary_ylabel="Mean arrival %",
        secondary_ylabel="Episodes with crash % (mean 0/1)",
        chart_title=(
            f"{tag} — {chart_suite_descriptor} (n={n_ev} scenarios × {len(full_order)} conditions)"
        ),
    )
    bundle.plot_crossing_dashboard_two_metrics(
        arrival,
        crash_pct,
        os.path.join(env_run_dir, "dashboard_quartet_arrival_and_crash.png"),
        condition_order=quartet_order,
        primary_ylabel="Mean arrival %",
        secondary_ylabel="Episodes with crash % (mean 0/1)",
        chart_title=f"{tag} — {chart_suite_descriptor} ({quartet_title_qualifier})",
    )

    js = {
        "env_id": env_id,
        "tag": tag,
        "scenario_pool": scenario_pool_key,
        "n_scenarios": n_ev,
        "metrics_csv": os.path.join(env_run_dir, metrics_csv_basename),
        "mean_arrival_pct_by_condition": arrival,
        "mean_crash_episode_fraction_by_condition": crash_frac,
        "mean_crash_episode_pct_by_condition": crash_pct,
    }
    with open(os.path.join(env_run_dir, "env_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2)

    pair_summary["environments"][tag] = js

    with open(os.path.join(pair_out_dir, "pair_summary.json"), "w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2)

    return pair_summary


def eval_pair_double_intersection_held_out_only(
    *,
    agent_pth: str,
    master_pth: str,
    pair_out_dir: str,
    g03_config: str,
    scenarios_pickle: str,
    n_scenarios_cap: int,
    eval_run_id: int,
) -> dict[str, Any]:
    """
    Same G03 coordination suite (8 conditions) as the full MODELS_EVALUATION run, but scenarios are
    **only** the double-intersection **held-out regular** set (training never sees these indices).
    """
    cap = max(1, int(n_scenarios_cap))
    scenarios = build_double_intersection_held_out_scenario_list(cap)
    return eval_pair_double_intersection_custom_pool(
        agent_pth=agent_pth,
        master_pth=master_pth,
        pair_out_dir=pair_out_dir,
        g03_config=g03_config,
        scenarios_pickle=scenarios_pickle,
        scenarios=scenarios,
        eval_run_id=eval_run_id,
        scenario_pool_key="held_out_regular",
        metrics_csv_basename="coordination_metrics_double_intersection_held_out.csv",
        chart_suite_descriptor="held-out regular",
        pair_summary_extras={
            "n_scenarios_requested_cap": cap,
            "held_out_indices": sorted(DOUBLE_INTERSECTION_HELD_OUT_INDICES),
        },
    )


def main() -> None:
    default_agent = os.path.join(_REPO, "models_to_check", "agent")
    default_master = os.path.join(_REPO, "models_to_check", "master")
    parser = argparse.ArgumentParser(description="Multi-env coordination eval for checkpoint pairs.")
    parser.add_argument("--agent-dir", type=str, default=default_agent)
    parser.add_argument("--master-dir", type=str, default=default_master)
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Parent folder; default: <repo>/MODELS_EVALUATION/<timestamp>",
    )
    parser.add_argument("--scenarios-pickle", type=str, default=DEFAULT_SCENARIO_PKL)
    parser.add_argument("--g03-config", type=str, default=bundle.G03_DEFAULT_CONFIG)
    parser.add_argument("--n-scenarios", type=int, default=100, help="Cap per environment (native pool sliced).")
    parser.add_argument("--eval-run-id", type=int, default=0)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If >0, only evaluate the first K pairs after sorting (smoke / debug).",
    )
    parser.add_argument(
        "--no-stem-pairs",
        action="store_true",
        help="Only pair numeric checkpoints (foo_agent3.pth ↔ foo_master3.pth); skip foo_agent.pth pairs.",
    )
    parser.add_argument(
        "--agent-pth",
        type=str,
        default="",
        help="Single-agent checkpoint (.pth). With --master-pth, skips directory pairing.",
    )
    parser.add_argument(
        "--master-pth",
        type=str,
        default="",
        help="Single-master checkpoint (.pth). With --agent-pth, skips directory pairing.",
    )
    parser.add_argument(
        "--pair-slug",
        type=str,
        default="fine_tuned_pair",
        help="Output subfolder under --output-root when using --agent-pth / --master-pth.",
    )
    parser.add_argument(
        "--eval-envs",
        type=str,
        default="",
        help="Comma list: intersection,roundabout,double_intersection (omit = all three).",
    )
    args = parser.parse_args()

    valid_tags = frozenset({"intersection", "roundabout", "double_intersection"})
    env_tags: frozenset[str] | None = None
    ev_raw = (args.eval_envs or "").strip()
    if ev_raw:
        parts = frozenset(p.strip().lower() for p in ev_raw.split(",") if p.strip())
        unknown = parts - valid_tags
        if unknown:
            print(f"--eval-envs unknown tags: {unknown}. Allowed: {sorted(valid_tags)}", file=sys.stderr)
            sys.exit(2)
        env_tags = parts

    agent_single = (args.agent_pth or "").strip()
    master_single = (args.master_pth or "").strip()
    if bool(agent_single) ^ bool(master_single):
        print("Provide both --agent-pth and --master-pth together.", file=sys.stderr)
        sys.exit(2)

    if agent_single and master_single:
        ap = os.path.normpath(os.path.abspath(os.path.expanduser(agent_single)))
        mp = os.path.normpath(os.path.abspath(os.path.expanduser(master_single)))
        if not os.path.isfile(ap) or not os.path.isfile(mp):
            print("Single-pair paths must be existing files.", file=sys.stderr)
            sys.exit(2)

    if (args.output_root or "").strip():
        batch_root = os.path.normpath(os.path.abspath(os.path.expanduser(args.output_root.strip())))
    else:
        ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        batch_root = os.path.join(_REPO, "MODELS_EVALUATION", f"eval_{ts}")
    os.makedirs(batch_root, exist_ok=True)

    digit_pairs: dict = {}
    agent_dir = master_dir = Path(".")

    if agent_single and master_single:
        worklist = [(ap, mp, _safe(args.pair_slug))]
    else:
        agent_dir = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.agent_dir))))
        master_dir = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.master_dir))))
        if not agent_dir.is_dir() or not master_dir.is_dir():
            print("agent-dir and master-dir must exist", file=sys.stderr)
            sys.exit(2)

        digit_pairs = discover_digit_pairs(agent_dir, master_dir)
        worklist = []
        for (prefix, num), (apath, mpath) in sorted(digit_pairs.items(), key=lambda x: (x[0][0], x[0][1])):
            slug = pair_slug_from_digit(prefix, num)
            worklist.append((str(apath), str(mpath), slug))

        used_agent = {t[0] for t in worklist}
        used_master = {t[1] for t in worklist}

        if not args.no_stem_pairs:
            stem_map = discover_stem_pairs(agent_dir, master_dir)
            for stem, (apath, mpath) in sorted(stem_map.items()):
                sa, sm = str(apath), str(mpath)
                if sa in used_agent or sm in used_master:
                    continue
                worklist.append((sa, sm, _safe(stem)))
                used_agent.add(sa)
                used_master.add(sm)

    if int(args.max_pairs or 0) > 0:
        worklist = worklist[: int(args.max_pairs)]

    meta = {
        "batch_root": batch_root,
        "agent_dir": str(agent_dir),
        "master_dir": str(master_dir),
        "n_digit_pairs_found": len(digit_pairs),
        "n_pairs_to_run": len(worklist),
        "eval_envs_filter": sorted(env_tags) if env_tags else "all",
        "g03_config": os.path.abspath(args.g03_config),
        "scenarios_pickle": os.path.abspath(args.scenarios_pickle),
        "n_scenarios_cap": int(args.n_scenarios),
        "pairs": [],
    }

    if not worklist:
        print("No matching checkpoint pairs found. Check filenames (e.g. foo_agent5.pth / foo_master5.pth).")
        with open(os.path.join(batch_root, "batch_summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        sys.exit(1)

    for agent_pth, master_pth, slug in worklist:
        pair_dir = os.path.join(batch_root, slug)
        entry: dict[str, Any] = {
            "slug": slug,
            "agent_pth": agent_pth,
            "master_pth": master_pth,
            "out_dir": pair_dir,
            "ok": False,
            "error": "",
        }
        try:
            eval_pair_all_envs(
                agent_pth=agent_pth,
                master_pth=master_pth,
                pair_out_dir=pair_dir,
                g03_config=args.g03_config,
                scenarios_pickle=args.scenarios_pickle,
                n_scenarios_cap=args.n_scenarios,
                eval_run_id=args.eval_run_id,
                env_tags=env_tags,
            )
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
            print(f"[fail] {slug}: {exc}", file=sys.stderr)
        meta["pairs"].append(entry)

    meta["n_ok"] = int(sum(1 for p in meta["pairs"] if p["ok"]))
    with open(os.path.join(batch_root, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {meta['n_ok']}/{len(worklist)} pairs OK.\n{batch_root}\nbatch_summary.json")


if __name__ == "__main__":
    main()
