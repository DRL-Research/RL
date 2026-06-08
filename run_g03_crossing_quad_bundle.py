"""
Five sequential G03-style experiments: identical hyperparameters, episode counts 2000→6000,

custom crossing-heavy scenario pool evaluation, aggregate 4-bar ablation chart.

Alternative: `--train-episodes K` repeats K episodes for each of the five runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Sequence

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import run_proto_action_sweep as rps  # noqa: E402 — registers env + patches

from highwayenv.intersection_class import rotate_scenario_clockwise  # noqa: E402
from src.experiment.scenario_geometry import regular_base_geometry_bucket  # noqa: E402
from src.experiment.scenarios import base_complete_scenarios_6_cars  # noqa: E402
from src.experiment.scenarios_config import make_env_config_exp7  # noqa: E402

G03_DEFAULT_CONFIG = os.path.join(
    _REPO,
    "experiment_runs",
    "PH07_MASTER_PROOF_GRID15_2026_05_03-14_30_25",
    "01_experiments",
    "G03_proof_dist_aux030",
    "config.json",
)

# One run per entry: same cfg, differing training length unless --train-episodes overrides all runs uniformly.
FULL_EPISODE_SCHEDULE: tuple[int, ...] = (2000, 3000, 4000, 5000, 6000)
N_TRAIN_RUNS = len(FULL_EPISODE_SCHEDULE)
TRAIN_EPISODES = FULL_EPISODE_SCHEDULE[0]
N_CUSTOM_TEST = 100
OFFSET_JITTER = 3
MIN_SAME_LANE_GAP = 22
CUSTOM_TEST_CONDITIONS = ("normal", "zero_master", "const_all_masters", "disconnect_global_master")
TRAIN_TRACE_EVERY = 50

# Strings accepted as `condition` in `run_proto_action_sweep.run_episode`.
RUN_EPISODE_SUPPORTED_CONDITIONS = frozenset(
    {
        "normal",
        "const_all_masters",
        "zero_global_master",
        "disconnect_global_master",
        "large_const_global_master",
        "negate_global_master",
        "random_global_master",
        "shuffle_global_master",
        "delayed_global_master",
        "zero_master",
        "zero_local_masters",
        "swap_local_masters",
        "negate_master",
        "random_local_masters",
    }
)

# Extra stress tests targeting interpretable embedding misuse (beyond the default quartet).
COORDINATION_PROBE_CONDITIONS: tuple[str, ...] = (
    "swap_local_masters",
    "random_local_masters",
    "zero_global_master",
    "negate_master",
)


def default_plus_coordination_probes_order() -> tuple[str, ...]:
    """Default G03 quartet followed by coordination probe conditions (unique, stable order)."""
    out: list[str] = []
    seen: set[str] = set()
    for c in list(CUSTOM_TEST_CONDITIONS) + list(COORDINATION_PROBE_CONDITIONS):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def normalize_eval_conditions(seq: Sequence[str] | tuple[str, ...]) -> tuple[str, ...]:
    cleaned = tuple(x.strip() for x in seq if (x or "").strip())
    if not cleaned:
        raise ValueError("Empty condition list.")
    unk = set(cleaned) - RUN_EPISODE_SUPPORTED_CONDITIONS
    if unk:
        raise ValueError(f"Unknown conditions {sorted(unk)}.")
    return cleaned


def parse_conditions_csv(s: str) -> tuple[str, ...]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return normalize_eval_conditions(tuple(parts))


def default_condition_chart_labels(condition_keys: Sequence[str]) -> dict[str, str]:
    preset = {
        "normal": "Full hierarchy",
        "zero_master": "Zero LM to agents\n(+ zero global loop)",
        "const_all_masters": "Const broadcast\n(agent + gm loop)",
        "disconnect_global_master": "Disconnect GM\n(agent LM intact)",
        "swap_local_masters": "Swap LM1↔LM2\n(wrong group signal)",
        "random_local_masters": "Random LM each step\n(Gaussian)",
        "zero_global_master": "Zero global loop only\n(LM obs intact)",
        "negate_master": "Negate LM + loop",
        "zero_local_masters": "Zero LM (alias)",
    }
    return {k: preset.get(k, k.replace("_", "\n")) for k in condition_keys}


def plot_crossing_condition_bars(
    means: dict[str, float],
    out_path: str,
    *,
    condition_order: Sequence[str],
    chart_title: str | None = None,
    y_axis_label: str | None = None,
    label_override: dict[str, str] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(condition_order)
    vals = [float(means[k]) if k in means else float("nan") for k in keys]
    default_l = default_condition_chart_labels(keys)
    labels = [(label_override or {}).get(k) or default_l[k] for k in keys]
    try:
        cmap = plt.colormaps["tab10"]
    except (AttributeError, KeyError):  # older matplotlib
        cmap = plt.cm.get_cmap("tab10")
    cols = [cmap(i % 10) for i in range(len(keys))]
    wide = float(max(8.5, min(18.0, 1.1 * len(keys) + 2.5)))
    fig_h = 5.0 if len(keys) <= 6 else 5.5
    fig, ax = plt.subplots(figsize=(wide, fig_h))
    x = np.arange(len(keys))
    ax.bar(x, vals, color=cols)
    ax.set_xticks(x)
    if len(keys) <= 6:
        ax.set_xticklabels(labels, fontsize=9)
    else:
        ax.set_xticklabels(labels, fontsize=7, rotation=28, ha="right")
    ax.set_ylabel(y_axis_label or "Mean arrival % (custom crossing scenarios)")
    ax.set_title(chart_title or "Custom crossing coordination ablations")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    rps.safe_figure_save_png(fig, out_path, dpi=175)
    plt.close(fig)


def rows_mean_numeric_by_condition(
    rows: list[dict[str, Any]],
    field: str,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    """Mean of a numeric CSV/row field keyed by episode `condition`."""
    buckets: dict[str, list[float]] = {}
    for r in rows:
        c = str(r.get("condition", ""))
        if not c:
            continue
        v = float(r[field]) * float(scale)
        buckets.setdefault(c, []).append(v)
    return {c: float(np.mean(vs)) if vs else float("nan") for c, vs in buckets.items()}


def plot_crossing_dashboard_two_metrics(
    primary: dict[str, float],
    secondary: dict[str, float],
    out_path: str,
    *,
    condition_order: Sequence[str],
    primary_ylabel: str,
    secondary_ylabel: str,
    chart_title: str | None = None,
) -> None:
    """Stacked bar charts (same condition order): e.g. arrival % on top, crash rate on bottom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(condition_order)
    y1 = [float(primary.get(k, float("nan"))) for k in keys]
    y2 = [float(secondary.get(k, float("nan"))) for k in keys]
    labels = [default_condition_chart_labels(keys)[k] for k in keys]
    wide = float(max(9.0, min(18.0, 1.05 * len(keys) + 3.0)))

    fig, axes = plt.subplots(2, 1, figsize=(wide, 8.8), sharex=True)
    try:
        cmap = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab10")
    cols = [cmap(i % 10) for i in range(len(keys))]
    xa = np.arange(len(keys))

    axes[0].bar(xa, y1, color=cols)
    axes[0].set_ylabel(primary_ylabel)
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].set_title(chart_title or "Custom crossing coordination report")

    axes[1].bar(xa, y2, color=cols)
    axes[1].set_ylabel(secondary_ylabel)
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[1].set_xticks(xa)
    if len(keys) <= 6:
        axes[1].set_xticklabels(labels, fontsize=9)
    else:
        axes[1].set_xticklabels(labels, fontsize=7, rotation=26, ha="right")
    fig.tight_layout()
    rps.safe_figure_save_png(fig, out_path, dpi=175)
    plt.close(fig)


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rotate_complete_dict(scenario: dict[str, Any], rotation: int) -> dict[str, Any]:
    if rotation == 0:
        return {
            "agents": copy.deepcopy(scenario["agents"]),
            "static": copy.deepcopy(scenario["static"]),
        }
    return {
        "agents": rotate_scenario_clockwise(scenario["agents"], rotation),
        "static": rotate_scenario_clockwise(scenario["static"], rotation),
    }


def crossing_straight_heavy_templates() -> list[dict[str, Any]]:
    table = base_complete_scenarios_6_cars
    out: list[dict[str, Any]] = []
    for bi, base in enumerate(table):
        if regular_base_geometry_bucket(bi, table) != "crossing_straight_heavy":
            continue
        for rot in (0, 1, 2, 3):
            out.append(rotate_complete_dict(base, rot))
    if not out:
        raise RuntimeError("No crossing_straight_heavy scenario templates found — check scenario_geometry buckets.")
    return out


def perturb_offset(off: int, rng: np.random.Generator, span: int) -> int:
    return int(off) + int(rng.integers(-span, span + 1))


def same_lane_spacing_ok(scenario: dict[str, Any]) -> bool:
    by_lane: dict[tuple[Any, ...], list[int]] = {}
    for lane_key, _dest, off in list(scenario["agents"]) + list(scenario["static"]):
        lk = tuple(lane_key) if isinstance(lane_key, tuple) else (lane_key,)
        by_lane.setdefault(lk, []).append(int(off))
    for offs in by_lane.values():
        if len(offs) < 2:
            continue
        s = sorted(offs)
        for a, b in zip(s, s[1:]):
            if abs(b - a) < MIN_SAME_LANE_GAP:
                return False
    return True


def perturb_scenario(template: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return {
        "agents": [(lk, d, perturb_offset(off, rng, OFFSET_JITTER)) for lk, d, off in template["agents"]],
        "static": [(lk, d, perturb_offset(off, rng, OFFSET_JITTER)) for lk, d, off in template["static"]],
    }


def validator_env_config(proto_cfg: dict[str, Any], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = make_env_config_exp7(
        collision_reward=int(proto_cfg["collision_reward"]),
        arrived_reward=int(proto_cfg["arrived_reward"]),
        starvation_reward=float(proto_cfg["starvation_reward"]),
        high_speed_reward=float(proto_cfg["high_speed_reward"]),
        target_speeds=list(proto_cfg["target_speeds"]),
    )
    cfg["conflict_ratio"] = 0.0
    cfg["use_conflict_scenarios_only"] = False
    cfg["use_held_out_scenarios"] = False
    cfg["custom_regular_scenarios"] = scenarios
    cfg["custom_regular_only"] = True
    return cfg


def spawn_valid_for_scenario(proto_cfg: dict[str, Any], scenario: dict[str, Any]) -> bool:
    import gymnasium as gym

    env = gym.make(rps.ENV_ID, render_mode=None, config=validator_env_config(proto_cfg, [scenario]))
    try:
        env.reset(seed=0)
        inner = env.unwrapped
        while hasattr(inner, "env") and not hasattr(inner, "controlled_vehicles"):
            inner = inner.env
        if any(getattr(v, "crashed", False) for v in inner.controlled_vehicles):
            return False
        return True
    finally:
        env.close()


def build_custom_test_pool(proto_cfg: dict[str, Any], rng: np.random.Generator, templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = 50_000
    while len(candidates) < N_CUSTOM_TEST and attempts < max_attempts:
        tmpl = templates[len(candidates) % len(templates)]
        cand = perturb_scenario(tmpl, rng)
        attempts += 1
        if not same_lane_spacing_ok(cand):
            continue
        if not spawn_valid_for_scenario(proto_cfg, cand):
            continue
        candidates.append(cand)
    if len(candidates) < N_CUSTOM_TEST:
        raise RuntimeError(
            f"Failed to generate {N_CUSTOM_TEST} valid perturbed scenarios after {attempts} attempts "
            f"(got {len(candidates)})."
        )
    return candidates


def train_one_run(
    base_cfg: dict[str, Any],
    run_dir: str,
    run_seed: int,
    *,
    train_trace_every: int,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    cfg = dict(base_cfg)
    cfg["episodes"] = TRAIN_EPISODES
    rps.write_json(os.path.join(run_dir, "config.json"), cfg)

    rps.set_all_seeds(run_seed)
    proto_exp = rps.ProtoExperiment(cfg, run_dir)
    master_model, agent_model = rps.make_models(proto_exp)
    aux_head = rps.RiskAuxHead(proto_exp.embedding_dim)
    agent_model, pretrained_loaded = rps.maybe_load_pretrained(cfg, master_model, agent_model)
    train_env = rps.ProtoHighwayWrapper(proto_exp, conflict_ratio=0.0, conflict_only=False)

    train_rows: list[dict[str, Any]] = []
    train_trace_rows: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    best_selection_score = float("-inf")
    best_ckpt_metric_mode = str(cfg.get("best_ckpt_metric") or "arrival")

    try:
        for ep in range(1, int(cfg["episodes"]) + 1):
            train_env.set_conflict_ratio(rps.schedule_value(cfg["conflict_schedule"], ep))
            trace_this = train_trace_every > 0 and (ep % train_trace_every == 0 or ep == 1)
            summary, master_transitions, agent_transitions = rps.run_episode(
                proto_exp=proto_exp,
                env=train_env,
                master_model=master_model,
                agent_model=agent_model,
                episode=ep,
                train=True,
                trace_rows=train_trace_rows if trace_this else None,
            )
            train_rows.append(summary)
            master_loss = rps.ppo_update(
                master_model.model.policy,
                master_transitions,
                discrete=False,
                cfg=cfg,
                aux_head=aux_head,
                episode=ep,
            )
            agent_loss = rps.ppo_update(agent_model.policy, agent_transitions, discrete=True, cfg=cfg, episode=ep)
            losses.append(
                {"episode": ep, **{f"master_{k}": v for k, v in master_loss.items()}, **{f"agent_{k}": v for k, v in agent_loss.items()}}
            )

            if ep >= 50:
                win = min(50, ep)
                window_rows = train_rows[-win:]
                rolling_arr = float(np.mean([float(r["arrival_pct"]) for r in window_rows]))
                if best_ckpt_metric_mode == "composite":
                    rolling_rew = float(np.mean([float(r["reward"]) for r in window_rows]))
                    div = float(cfg.get("best_ckpt_composite_reward_div") or 400.0)
                    arrival_w = float(cfg.get("best_ckpt_composite_arrival_weight", 1.0))
                    rew_w = float(cfg.get("best_ckpt_composite_reward_weight", 0.35))
                    score = arrival_w * rolling_arr + rew_w * (rolling_rew / max(1e-6, div))
                else:
                    score = rolling_arr
                if score > best_selection_score:
                    best_selection_score = score
                    rps.save_models(agent_model, master_model, os.path.join(run_dir, "best", "ckpt"))

            if ep % 50 == 0 or ep == 1:
                print(
                    f"[{cfg['label']}] seed={run_seed} ep={ep}/{cfg['episodes']} "
                    f"arrival={summary['arrival_pct']:.1f} crashed={summary['crashed']} pretrained={pretrained_loaded}"
                )
    finally:
        train_env.close()

    rps.save_models(agent_model, master_model, os.path.join(run_dir, "trained_model"))
    rps.write_csv(os.path.join(run_dir, "episode_metrics.csv"), train_rows)
    rps.write_csv(os.path.join(run_dir, "losses.csv"), losses)
    rps.plot_training(run_dir, train_rows)
    rps.write_csv(os.path.join(run_dir, "train_master_step_trace.csv"), train_trace_rows)
    metrics = rps.phase_metrics(train_trace_rows, run_dir)
    for role in rps.ROLES:
        rps.plot_pca(train_trace_rows, run_dir, role)
    rps.plot_pca_lm_combined(train_trace_rows, run_dir)
    rps.plot_proto_distance_timeline(train_trace_rows, run_dir)
    rps.write_json(
        os.path.join(run_dir, "train_trace_meta.json"),
        {
            "trace_every_episodes": train_trace_every,
            "n_trace_rows": len(train_trace_rows),
            "phase_metrics": metrics,
        },
    )


def load_best_for_eval(run_dir: str, agent_model, master_model) -> str:
    best_ckpt_dir = os.path.join(run_dir, "best", "ckpt")
    if os.path.isfile(f"{best_ckpt_dir}_agent.pth") and os.path.isfile(f"{best_ckpt_dir}_master.pth"):
        if rps.load_models(agent_model, master_model, best_ckpt_dir):
            return "best_rolling50"
    trained_ckpt_dir = os.path.join(run_dir, "trained_model")
    if os.path.isfile(f"{trained_ckpt_dir}_agent.pth") and os.path.isfile(f"{trained_ckpt_dir}_master.pth"):
        if rps.load_models(agent_model, master_model, trained_ckpt_dir):
            return "trained_model_final"
    raise FileNotFoundError(f"No checkpoints under {best_ckpt_dir} or {trained_ckpt_dir}")


def run_custom_eval(
    proto_exp: rps.ProtoExperiment,
    master_model,
    agent_model,
    scenarios: list[dict[str, Any]],
    *,
    run_dir: str,
    run_id: int,
    eval_policy_mean_backup: bool,
    n_eval_scenarios: int | None = None,
    conditions: tuple[str, ...] | None = None,
    env_id: str | None = None,
    metrics_csv_basename: str = "custom_crossing_test_metrics.csv",
) -> list[dict[str, Any]]:
    n_ep = int(n_eval_scenarios if n_eval_scenarios is not None else N_CUSTOM_TEST)
    cond_tuple = normalize_eval_conditions(conditions) if conditions is not None else CUSTOM_TEST_CONDITIONS
    env_extra = {
        "custom_regular_scenarios": scenarios,
        "custom_regular_only": True,
        "conflict_ratio": 0.0,
        "use_conflict_scenarios_only": False,
        "use_held_out_scenarios": False,
    }
    run_dir = rps._resolve_long_path(str(run_dir))
    os.makedirs(run_dir, exist_ok=True)
    test_env = rps.ProtoHighwayWrapper(
        proto_exp,
        conflict_ratio=0.0,
        conflict_only=False,
        env_extra=env_extra,
        env_id=env_id,
    )
    eval_env = env_id or rps.ENV_ID
    proto_exp.cfg["eval_policy_mean"] = True
    rows: list[dict[str, Any]] = []
    try:
        for ep_ix in range(n_ep):
            base_seed = 8_000_000 + int(run_id) * 100_000 + ep_ix
            for condition in cond_tuple:
                summary, _, _ = rps.run_episode(
                    proto_exp=proto_exp,
                    env=test_env,
                    master_model=master_model,
                    agent_model=agent_model,
                    episode=ep_ix + 1,
                    train=False,
                    condition=condition,
                    replay_seed=base_seed,
                    custom_scenario_index=ep_ix,
                )
                summary = dict(summary)
                summary["run_id"] = run_id
                summary["scenario_slot"] = ep_ix
                summary["condition"] = condition
                summary["eval_env_id"] = eval_env
                rows.append(summary)
    finally:
        test_env.close()
        proto_exp.cfg["eval_policy_mean"] = eval_policy_mean_backup
    out_csv = os.path.join(run_dir, metrics_csv_basename)
    rps.write_csv(out_csv, rows)
    return rows


def plot_four_bar(
    means: dict[str, float],
    out_path: str,
    *,
    chart_title: str | None = None,
    y_axis_label: str | None = None,
) -> None:
    plot_crossing_condition_bars(
        means,
        out_path,
        condition_order=list(CUSTOM_TEST_CONDITIONS),
        chart_title=chart_title
        or "G03 five-run episode sweep — crossing-only custom test ablations",
        y_axis_label=y_axis_label or "Mean arrival % (100 crossing-heavy scenarios)",
    )


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scenarios_pickle(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) < 1:
        raise ValueError(f"Invalid scenarios pickle: {path}")
    return data


def default_run_seeds() -> list[int]:
    return [12_345 + i * 9973 for i in range(N_TRAIN_RUNS)]


def parse_seeds_arg(s: str | None, fallback: list[int]) -> list[int]:
    if not (s or "").strip():
        return list(fallback)
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(out) != N_TRAIN_RUNS:
        raise ValueError(f"--seeds must list exactly {N_TRAIN_RUNS} integers, got {out!r}")
    return out


def main() -> None:
    global TRAIN_EPISODES, N_CUSTOM_TEST, TRAIN_TRACE_EVERY, OFFSET_JITTER

    parser = argparse.ArgumentParser(
        description=(
            "G03 five-run episode sweep (2000…6000 by default). "
            "Use --reproduce-from to reuse a scenario pickle; --train-episodes N for five runs of length N each."
        )
    )
    parser.add_argument("--g03-config", type=str, default=G03_DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--smoke", action="store_true", help="2 episodes / 4 test scenarios for a quick sanity check.")
    parser.add_argument(
        "--reproduce-from",
        type=str,
        default="",
        help="Existing bundle directory; loads its scenario pickle and bundle_meta. "
        "Uses that bundle's run_00/config.json when present (overrides --g03-config).",
    )
    parser.add_argument(
        "--scenarios-pickle",
        type=str,
        default="",
        help="Load the test scenario list from this pickle (skip random pool generation).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help=f"Comma-separated {N_TRAIN_RUNS} training seeds; default formula 12345+run_id*9973.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=0,
        help=(
            "Uniform training episodes for EVERY run (0 = use default schedule "
            f"{list(FULL_EPISODE_SCHEDULE)}; ignored with --smoke)."
        ),
    )
    parser.add_argument(
        "--n-custom-test",
        type=int,
        default=0,
        help=f"Custom eval scenarios count (0 = module default {N_CUSTOM_TEST} except --smoke).",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help=f"Train PCA trace stride (0 = module default {TRAIN_TRACE_EVERY} except --smoke).",
    )
    parser.add_argument(
        "--ignore-bundle-run-config",
        action="store_true",
        help="With --reproduce-from: keep --g03-config instead of substituting run_00/config.json.",
    )
    args = parser.parse_args()

    if args.smoke:
        TRAIN_EPISODES = 2
        N_CUSTOM_TEST = 4
        TRAIN_TRACE_EVERY = 1

    reuse_meta: dict[str, Any] | None = None
    source_pickle_hint = ""
    run_00_cfg_path = ""

    reproduce_root = os.path.normpath(os.path.abspath(os.path.expanduser(args.reproduce_from))) if args.reproduce_from else ""
    if reproduce_root:
        if not os.path.isdir(reproduce_root):
            raise FileNotFoundError(f"--reproduce-from not a directory: {reproduce_root}")
        meta_fp = os.path.join(reproduce_root, "bundle_meta.json")
        if os.path.isfile(meta_fp):
            reuse_meta = load_json(meta_fp)
        pickle_src = os.path.join(reproduce_root, "custom_crossing_test_scenarios.pkl")
        if not os.path.isfile(pickle_src):
            raise FileNotFoundError(f"Missing {pickle_src}")
        source_pickle_hint = pickle_src
        roc = os.path.join(reproduce_root, "run_00", "config.json")
        if os.path.isfile(roc) and not args.ignore_bundle_run_config:
            run_00_cfg_path = roc

    config_path_abs = (
        os.path.normpath(os.path.abspath(os.path.expanduser(run_00_cfg_path)))
        if run_00_cfg_path
        else os.path.normpath(os.path.abspath(os.path.expanduser(args.g03_config)))
    )
    base_cfg = load_json(config_path_abs)
    base_cfg.setdefault("master_broadcast_const_test", 9999.0)

    if reuse_meta and not args.smoke and reuse_meta.get("offset_jitter_pm") is not None:
        OFFSET_JITTER = int(reuse_meta["offset_jitter_pm"])

    uniform_train_eps: int | None = None
    # Full experiments: never inherit train_episodes / n_custom_test / trace_every from bundle_meta —
    # a previous smoke bundle would otherwise force tiny runs (episodes=2 etc.).
    if not args.smoke:
        if args.train_episodes > 0:
            uniform_train_eps = int(args.train_episodes)
        if args.n_custom_test > 0:
            N_CUSTOM_TEST = int(args.n_custom_test)
        # else leave N_CUSTOM_TEST at module default (100)
        if args.trace_every > 0:
            TRAIN_TRACE_EVERY = int(args.trace_every)

    seed_fallback = default_run_seeds()
    if reuse_meta and isinstance(reuse_meta.get("training_run_seeds"), list):
        try:
            parsed_fb = [int(x) for x in reuse_meta["training_run_seeds"]]
            if len(parsed_fb) == N_TRAIN_RUNS:
                seed_fallback = parsed_fb
        except (TypeError, ValueError):
            pass
    run_seeds = parse_seeds_arg(args.seeds, seed_fallback)

    if args.smoke:
        episode_plan = [2] * N_TRAIN_RUNS
    elif uniform_train_eps is not None:
        episode_plan = [uniform_train_eps] * N_TRAIN_RUNS
    else:
        episode_plan = list(FULL_EPISODE_SCHEDULE)

    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    bundle_root = args.out_root or os.path.join(_REPO, "experiment_runs", f"G03_CROSSING_QUAD_{ts}")
    bundle_root = os.path.normpath(os.path.abspath(bundle_root))
    os.makedirs(bundle_root, exist_ok=True)

    scenario_gen_seed: int | None = 20260501
    templates_catalog: list[dict[str, Any]] = []
    if args.scenarios_pickle:
        pickle_load = os.path.normpath(os.path.abspath(os.path.expanduser(args.scenarios_pickle)))
        scenarios = load_scenarios_pickle(pickle_load)
        scenario_gen_seed = None
    elif reproduce_root:
        scenarios = load_scenarios_pickle(source_pickle_hint)
        scenario_gen_seed = reuse_meta.get("scenario_pool_generator_seed") if reuse_meta else None
    else:
        rng_templates = np.random.default_rng(scenario_gen_seed)
        templates_catalog = crossing_straight_heavy_templates()
        scenarios = build_custom_test_pool(base_cfg, rng_templates, templates_catalog)

    if args.smoke and len(scenarios) > N_CUSTOM_TEST:
        scenarios = scenarios[:N_CUSTOM_TEST]

    if len(scenarios) != N_CUSTOM_TEST:
        raise ValueError(
            f"Scenario count mismatch: got {len(scenarios)} scenarios, N_CUSTOM_TEST={N_CUSTOM_TEST}. "
            "Use --n-custom-test to match the pickle or regenerate the pool."
        )

    scenarios_path = os.path.join(bundle_root, "custom_crossing_test_scenarios.pkl")
    with open(scenarios_path, "wb") as f:
        pickle.dump(scenarios, f)
    if reproduce_root and source_pickle_hint and os.path.isfile(source_pickle_hint):
        try:
            shutil.copy2(
                source_pickle_hint,
                os.path.join(bundle_root, "custom_crossing_test_scenarios.source_copy.pkl"),
            )
        except OSError:
            pass

    meta: dict[str, Any] = {
        "source_g03_config": config_path_abs,
        "source_g03_config_sha256": _sha256_file(config_path_abs) if os.path.isfile(config_path_abs) else "",
        "bundle_root": bundle_root,
        "reproduced_from_bundle": reproduce_root or None,
        "n_train_runs": N_TRAIN_RUNS,
        "episode_schedule": episode_plan,
        "train_episodes_max": max(episode_plan),
        "uniform_train_episodes_override": uniform_train_eps,
        "full_episode_schedule_default": list(FULL_EPISODE_SCHEDULE),
        "n_custom_test_scenarios": N_CUSTOM_TEST,
        "offset_jitter_pm": OFFSET_JITTER,
        "geometry_filter": "crossing_straight_heavy only (templates + rotations)",
        "test_conditions": list(CUSTOM_TEST_CONDITIONS),
        "scenarios_pickle": scenarios_path,
        "scenario_pool_generator_seed": scenario_gen_seed,
        "training_run_seeds": run_seeds,
        "train_trace_every": TRAIN_TRACE_EVERY,
        "templates_count": len(templates_catalog) if templates_catalog else (reuse_meta or {}).get("templates_count"),
    }
    rps.write_json(os.path.join(bundle_root, "bundle_meta.json"), meta)

    all_eval_rows: list[dict[str, Any]] = []
    run_means: dict[int, dict[str, float]] = {}

    for run_id in range(N_TRAIN_RUNS):
        TRAIN_EPISODES = int(episode_plan[run_id])
        label = f"{base_cfg.get('label', 'G03')}_ser_run{run_id}_{TRAIN_EPISODES}eps"
        run_cfg = dict(base_cfg)
        run_cfg["label"] = label
        run_dir = os.path.join(bundle_root, f"run_{run_id:02d}")
        run_seed = run_seeds[run_id]

        train_one_run(run_cfg, run_dir, run_seed, train_trace_every=TRAIN_TRACE_EVERY)

        rps.set_all_seeds(run_seed)
        proto_exp = rps.ProtoExperiment(run_cfg, run_dir)
        master_model, agent_model = rps.make_models(proto_exp)
        ckpt_used = load_best_for_eval(run_dir, agent_model, master_model)
        rps.write_json(os.path.join(run_dir, "custom_eval_ckpt_used.json"), {"checkpoint": ckpt_used})

        eval_pm_backup = bool(proto_exp.cfg.get("eval_policy_mean", False))
        rows = run_custom_eval(
            proto_exp,
            master_model,
            agent_model,
            scenarios,
            run_dir=run_dir,
            run_id=run_id,
            eval_policy_mean_backup=eval_pm_backup,
        )
        all_eval_rows.extend(rows)

        rm: dict[str, float] = {}
        for cond in CUSTOM_TEST_CONDITIONS:
            sub = [r for r in rows if r["condition"] == cond]
            rm[cond] = float(np.mean([float(x["arrival_pct"]) for x in sub])) if sub else 0.0
        run_means[run_id] = rm

    rps.write_csv(os.path.join(bundle_root, "custom_crossing_test_metrics_all_runs.csv"), all_eval_rows)

    grand_means = {}
    for cond in CUSTOM_TEST_CONDITIONS:
        grand_means[cond] = float(np.mean([run_means[r][cond] for r in range(N_TRAIN_RUNS)]))

    rps.write_json(
        os.path.join(bundle_root, "custom_crossing_test_summary.json"),
        {
            "episode_schedule": episode_plan,
            "per_run_mean_arrival_pct": run_means,
            "mean_of_run_means_arrival_pct": grand_means,
        },
    )

    plot_four_bar(grand_means, os.path.join(bundle_root, "four_bar_custom_crossing_test.png"))
    print(f"Done. Bundle: {bundle_root}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed_s={time.time() - t0:.1f}")
