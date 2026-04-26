from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

import run_unified as unified
from src import project_globals
from src.diagnostics.collision_audit import build_step_trace_row
from src.model.agent_handler import Driver, DummyVecEnv
from src.model.model_handler import load_models
from src.training.episode_utils import _build_global_master_input, _build_local_master_input
from src.training.general_utils import get_scaler_action_and_action_array, initialize_models, setup_experiment_dirs


EVAL_MODES = [
    ("held_out", dict(use_held_out=True, use_conflict_only=False)),
    ("conflict", dict(use_held_out=False, use_conflict_only=True)),
]

CONDITIONS = ("normal", "zero_master", "swap_local_masters", "negate_master")
ROLE_ORDER = ("LM1", "LM2", "GM")


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json(values: Any) -> str:
    return json.dumps(np.asarray(values).tolist(), separators=(",", ":"))


def _current_scenario_idx(wrapped_env) -> int:
    try:
        inner = wrapped_env.env._get_unwrapped_env()
        return int(getattr(inner, "last_scenario_index", -1))
    except Exception:
        return -1


def _policy_stats(master_model, master_input: np.ndarray, embedding: np.ndarray) -> dict[str, Any]:
    obs = torch.as_tensor(master_input, dtype=torch.float32).reshape(1, -1)
    act = torch.as_tensor(embedding, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        value = master_model.model.policy.predict_values(obs)
        dist = master_model.model.policy.get_distribution(obs)
        log_prob = dist.log_prob(act)
        base = getattr(dist, "distribution", None)
        if base is not None and hasattr(base, "mean") and hasattr(base, "stddev"):
            mean = base.mean.detach().cpu().numpy().reshape(-1)
            std = base.stddev.detach().cpu().numpy().reshape(-1)
        else:
            mean = np.asarray(embedding, dtype=np.float32).reshape(-1)
            std = np.ones_like(mean, dtype=np.float32)
    return {
        "value": float(value.detach().cpu().reshape(-1)[0]),
        "log_prob": float(log_prob.detach().cpu().reshape(-1)[0]),
        "mean": mean.astype(float),
        "std": np.maximum(std.astype(float), 1e-6),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _entropy_binary(actions: list[int]) -> float:
    if not actions:
        return 0.0
    p = float(sum(actions)) / len(actions)
    if p <= 1e-12 or p >= 1.0 - 1e-12:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _disagreement(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    return float(min(sum(actions), len(actions) - sum(actions)) / len(actions))


def _build_models_for_checkpoint(checkpoint_path: str, out_dir: str):
    exp = unified._make_experiment(out_dir, 1, env_id="RELintersection-v0")
    setup_experiment_dirs(out_dir)
    inter_cfg = unified._make_env_config("RELintersection-v0")
    master_model, agent_model, _ = initialize_models(exp, inter_cfg)
    if not load_models(agent_model, master_model, checkpoint_path):
        raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")
    master_model.freeze()
    return exp, master_model, agent_model


def _make_wrapped_env(env_id: str, out_dir: str, mode_kwargs: dict[str, bool]):
    exp = unified._make_experiment(out_dir, 1, env_id=env_id)
    exp.WARMUP_EPISODES = 0
    exp.CONFIG = unified._make_env_config(env_id, **mode_kwargs)
    project_globals.after_is_arrived_flags = [False] * 6

    def _env_fn(ec=exp):
        return Driver(ec)

    return exp, DummyVecEnv([_env_fn])


def _counterfactual_embeddings(
    condition: str,
    lm1_emb: np.ndarray,
    lm2_emb: np.ndarray,
    agents_per_lm: int,
) -> list[np.ndarray]:
    zero = np.zeros_like(lm1_emb)
    if condition == "normal":
        return [lm1_emb] * agents_per_lm + [lm2_emb] * agents_per_lm
    if condition == "zero_master":
        return [zero] * (2 * agents_per_lm)
    if condition == "swap_local_masters":
        return [lm2_emb] * agents_per_lm + [lm1_emb] * agents_per_lm
    if condition == "negate_master":
        return [-lm1_emb] * agents_per_lm + [-lm2_emb] * agents_per_lm
    raise ValueError(f"Unknown condition: {condition}")


def _next_global_embedding(condition: str, gm_emb: np.ndarray) -> np.ndarray:
    if condition == "zero_master":
        return np.zeros_like(gm_emb)
    if condition == "negate_master":
        return -np.asarray(gm_emb, dtype=np.float32).reshape(-1)
    return np.asarray(gm_emb, dtype=np.float32).reshape(-1)


def _run_one_episode(
    *,
    model_seed: int,
    env_label: str,
    env_id: str,
    eval_mode: str,
    mode_kwargs: dict[str, bool],
    episode_idx: int,
    replay_seed: int,
    condition: str,
    out_dir: str,
    master_model,
    agent_model,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _set_all_seeds(replay_seed)
    exp, wrapped_env = _make_wrapped_env(env_id, out_dir, mode_kwargs)
    n_agents = exp.CARS_AMOUNT
    agents_per_lm = exp.AGENTS_PER_LOCAL_MASTER
    emb_size = exp.EMBEDDING_SIZE

    rows: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []
    total_reward = 0.0
    global_emb_prev = np.zeros(emb_size, dtype=np.float32)
    done = False
    truncated = False
    step = 0

    try:
        wrapped_env.reset()
        scenario_idx = _current_scenario_idx(wrapped_env)
        while not done and not truncated:
            step += 1
            all_states = np.asarray(wrapped_env.env.current_state, dtype=np.float32)

            lm1_input = _build_local_master_input(global_emb_prev, all_states[0:agents_per_lm])
            lm2_input = _build_local_master_input(global_emb_prev, all_states[agents_per_lm:n_agents])
            lm1_emb, _, _ = master_model.get_proto_action(lm1_input)
            lm2_emb, _, _ = master_model.get_proto_action(lm2_input)
            lm1_emb = np.asarray(lm1_emb, dtype=np.float32).reshape(-1)
            lm2_emb = np.asarray(lm2_emb, dtype=np.float32).reshape(-1)

            gm_input = _build_global_master_input(lm1_emb, lm2_emb)
            gm_emb, _, _ = master_model.get_proto_action(gm_input)
            gm_emb = np.asarray(gm_emb, dtype=np.float32).reshape(-1)

            stats = {
                "LM1": _policy_stats(master_model, lm1_input, lm1_emb),
                "LM2": _policy_stats(master_model, lm2_input, lm2_emb),
                "GM": _policy_stats(master_model, gm_input, gm_emb),
            }

            local_embeddings = _counterfactual_embeddings(condition, lm1_emb, lm2_emb, agents_per_lm)
            car_observations = wrapped_env.env.build_full_obs(local_embeddings)
            actions = Driver.get_action(
                agent_model,
                car_observations,
                episode_idx,
                0,
                policy_stochastic=False,
            )
            scalar_actions = []
            for action in actions:
                scalar, _ = get_scaler_action_and_action_array(action)
                scalar_actions.append(int(scalar))

            _, reward, done, truncated, info = wrapped_env.step(tuple(scalar_actions))
            total_reward += float(reward)
            inner = wrapped_env.env._get_unwrapped_env()
            trace_row = build_step_trace_row(step, scalar_actions, float(reward), info, inner, exp)
            step_trace.append(trace_row)

            lm_l2 = float(np.linalg.norm(lm1_emb - lm2_emb))
            lm_cos = _cosine(lm1_emb, lm2_emb)
            action_entropy = _entropy_binary(scalar_actions)
            action_disagreement = _disagreement(scalar_actions)

            shared = {
                "model_seed": model_seed,
                "eval_mode": eval_mode,
                "env": env_label,
                "condition": condition,
                "episode": episode_idx,
                "replay_seed": replay_seed,
                "scenario_idx": scenario_idx,
                "step": step,
                "step_reward": float(reward),
                "actions": _json(scalar_actions),
                "action_sum": int(sum(scalar_actions)),
                "action_entropy": action_entropy,
                "action_disagreement": action_disagreement,
                "group0_disagreement": _disagreement(scalar_actions[:agents_per_lm]),
                "group1_disagreement": _disagreement(scalar_actions[agents_per_lm:]),
                "min_pairwise_dist_active_m": trace_row.get("min_pairwise_dist_active_m"),
                "min_dist_active_to_uncontrolled_m": trace_row.get("min_dist_active_to_uncontrolled_m"),
                "n_active_controlled": trace_row.get("n_active_controlled"),
                "positions_xy": _json(trace_row.get("positions_xy", [])),
                "speeds": _json(trace_row.get("speeds", [])),
                "lm_l2_distance": lm_l2,
                "lm_cosine": lm_cos,
                "gm_norm": float(np.linalg.norm(gm_emb)),
            }
            role_payload = {
                "LM1": (lm1_input, lm1_emb),
                "LM2": (lm2_input, lm2_emb),
                "GM": (gm_input, gm_emb),
            }
            for role in ROLE_ORDER:
                inp, emb = role_payload[role]
                row_stats = stats[role]
                rows.append({
                    **shared,
                    "role": role,
                    "embedding": _json(emb),
                    "policy_mean": _json(row_stats["mean"]),
                    "policy_std": _json(row_stats["std"]),
                    "master_value": float(row_stats["value"]),
                    "master_log_prob": float(row_stats["log_prob"]),
                    "input_norm": float(np.linalg.norm(inp)),
                    "embedding_norm": float(np.linalg.norm(emb)),
                })

            global_emb_prev = _next_global_embedding(condition, gm_emb)
    finally:
        try:
            wrapped_env.close()
        except Exception:
            pass

    crashed = any(any(r.get("crashed_flags", [])) for r in step_trace)
    first_crash_step = None
    for r in step_trace:
        if any(r.get("crashed_flags", [])):
            first_crash_step = int(r["step"])
            break

    min_pair_vals = [
        float(r["min_pairwise_dist_active_m"])
        for r in step_trace
        if r.get("min_pairwise_dist_active_m") is not None
    ]
    near_miss_steps = sum(1 for v in min_pair_vals if v < 8.0)
    for row in rows:
        row["crashed_episode"] = int(crashed)
        row["first_crash_step"] = first_crash_step if first_crash_step is not None else ""
        if first_crash_step is not None:
            row["steps_to_crash"] = first_crash_step - int(row["step"])
        else:
            row["steps_to_crash"] = ""

    arrived = 0
    try:
        inner = wrapped_env.env._get_unwrapped_env()
        arrived = sum(1 for v in inner.controlled_vehicles if hasattr(v, "is_arrived") and v.is_arrived)
    except Exception:
        pass

    episode_summary = {
        "model_seed": model_seed,
        "eval_mode": eval_mode,
        "env": env_label,
        "condition": condition,
        "episode": episode_idx,
        "replay_seed": replay_seed,
        "scenario_idx": scenario_idx if "scenario_idx" in locals() else -1,
        "steps": step,
        "reward": total_reward,
        "arrival_pct": 100.0 * arrived / max(1, n_agents),
        "crashed": int(crashed),
        "first_crash_step": first_crash_step if first_crash_step is not None else "",
        "min_pairwise_dist_m": min(min_pair_vals) if min_pair_vals else "",
        "mean_pairwise_dist_m": float(np.mean(min_pair_vals)) if min_pair_vals else "",
        "near_miss_steps": int(near_miss_steps),
        "near_miss_rate": float(near_miss_steps / max(1, step)),
    }
    return rows, episode_summary


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return float(np.mean(vals)) if vals else 0.0


def _summarize(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(row["model_seed"], row["eval_mode"], row["env"], row["condition"])].append(row)
    out = []
    for (model_seed, eval_mode, env, condition), rows in sorted(grouped.items()):
        out.append({
            "model_seed": model_seed,
            "eval_mode": eval_mode,
            "env": env,
            "condition": condition,
            "n_episodes": len(rows),
            "arrival_mean": _mean(rows, "arrival_pct"),
            "crash_rate": 100.0 * sum(int(r["crashed"]) for r in rows) / max(1, len(rows)),
            "reward_mean": _mean(rows, "reward"),
            "min_pairwise_dist_mean": _mean(rows, "min_pairwise_dist_m"),
            "near_miss_rate_mean": _mean(rows, "near_miss_rate"),
            "steps_mean": _mean(rows, "steps"),
        })
    return out


def _plot_summary(summary_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    conditions = list(CONDITIONS)
    cond_colors = {
        "normal": "#2E7D32",
        "zero_master": "#D32F2F",
        "swap_local_masters": "#F9A825",
        "negate_master": "#6A1B9A",
    }
    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        overall[r["condition"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    metrics = [
        ("arrival_mean", "Arrival (%)", True),
        ("crash_rate", "Crash Rate (%)", False),
        ("near_miss_rate_mean", "Near-Miss Step Rate", False),
    ]
    for ax, (key, title, higher_better) in zip(axes, metrics):
        means = [float(np.mean([row[key] for row in overall[c]])) if overall[c] else 0.0 for c in conditions]
        stds = [float(np.std([row[key] for row in overall[c]])) if overall[c] else 0.0 for c in conditions]
        ax.bar(range(len(conditions)), means, yerr=stds, color=[cond_colors[c] for c in conditions], alpha=0.85)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if higher_better:
            ax.set_ylim(bottom=0)
    fig.suptitle("Counterfactual Master Ablation on Fixed Trained Agents", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "counterfactual_performance.png"), dpi=180)
    plt.close(fig)

    gm_rows = [r for r in step_rows if r["role"] == "GM" and r["condition"] == "normal"]
    by_crash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in gm_rows:
        label = "crashed" if int(r["crashed_episode"]) else "success"
        by_crash[label].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, key, title in [
        (axes[0], "master_value", "GM Value by Outcome"),
        (axes[1], "lm_l2_distance", "LM1-LM2 Embedding Distance"),
        (axes[2], "action_disagreement", "Agent Action Disagreement"),
    ]:
        data = [[float(r[key]) for r in by_crash[label]] for label in ["success", "crashed"]]
        ax.boxplot(data, labels=["success", "crashed"], showfliers=False)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Master/Coordination Signals Under Normal Master", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "normal_master_signals_by_outcome.png"), dpi=180)
    plt.close(fig)

    rows_ttc = [
        r for r in gm_rows
        if r["steps_to_crash"] not in ("", None) and -5 <= int(r["steps_to_crash"]) <= 10
    ]
    if rows_ttc:
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows_ttc:
            stc = int(r["steps_to_crash"])
            if 0 <= stc <= 10:
                buckets[stc].append(r)
        xs = sorted(buckets)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, key, title in [
            (axes[0], "master_value", "GM Value"),
            (axes[1], "lm_l2_distance", "LM Distance"),
            (axes[2], "min_pairwise_dist_active_m", "Min Pairwise Distance"),
        ]:
            means = [float(np.mean([float(r[key]) for r in buckets[x] if r.get(key) not in ("", None)])) for x in xs]
            ax.plot(xs, means, marker="o")
            ax.invert_xaxis()
            ax.set_xlabel("Steps to crash (0 = crash step)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Signals Approaching Crash Under Normal Master", fontsize=14, weight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "time_to_crash_signals.png"), dpi=180)
        plt.close(fig)


def run_counterfactual(args: argparse.Namespace) -> str:
    Driver.NORMALIZE_AGENT_OBS = bool(args.normalize_agent_obs)

    source_root = os.path.abspath(os.path.normpath(args.source_root))
    out_dir = os.path.abspath(args.output_dir or os.path.join(
        source_root,
        f"master_coord_counterfactual_{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}",
    ))
    os.makedirs(out_dir, exist_ok=True)

    all_steps: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []

    for model_seed in args.model_seeds:
        run_dir = os.path.join(source_root, "A_base", "W_MASTER", f"s{model_seed}")
        ckpt = os.path.join(run_dir, "best", "ckpt")
        if not os.path.exists(ckpt + "_agent.pth"):
            ckpt = os.path.join(run_dir, "trained_model")
        model_out = os.path.join(out_dir, f"load_s{model_seed}")
        _, master_model, agent_model = _build_models_for_checkpoint(ckpt, model_out)

        for eval_mode, mode_kwargs in EVAL_MODES:
            for env_id, env_label in unified.ENV_DEFS:
                for ep in range(1, args.episodes_per_env + 1):
                    replay_seed = args.replay_seed_base + model_seed * 100000 + ep
                    for condition in CONDITIONS:
                        rows, summary = _run_one_episode(
                            model_seed=model_seed,
                            env_label=env_label,
                            env_id=env_id,
                            eval_mode=eval_mode,
                            mode_kwargs=mode_kwargs,
                            episode_idx=ep,
                            replay_seed=replay_seed,
                            condition=condition,
                            out_dir=out_dir,
                            master_model=master_model,
                            agent_model=agent_model,
                        )
                        all_steps.extend(rows)
                        all_episodes.append(summary)
                    print(f"s{model_seed} {eval_mode}/{env_label} ep={ep}/{args.episodes_per_env}")

    summary_rows = _summarize(all_episodes)
    _write_csv(os.path.join(out_dir, "step_traces.csv"), all_steps)
    _write_csv(os.path.join(out_dir, "episode_summary.csv"), all_episodes)
    _write_csv(os.path.join(out_dir, "condition_summary.csv"), summary_rows)
    _plot_summary(summary_rows, all_steps, out_dir)

    metadata = {
        "source_root": source_root,
        "model_seeds": args.model_seeds,
        "episodes_per_env_per_mode": args.episodes_per_env,
        "conditions": list(CONDITIONS),
        "eval_modes": [m for m, _ in EVAL_MODES],
        "envs": [label for _, label in unified.ENV_DEFS],
        "n_step_rows": len(all_steps),
        "n_episode_rows": len(all_episodes),
        "summary": summary_rows,
        "known_caveat": (
            "Existing checkpoints were trained with unnormalized master inputs; "
            "raw embeddings may be saturated. Counterfactual behavioral deltas are "
            "therefore more reliable than PCA of final clipped embeddings."
        ),
        "normalize_agent_obs": bool(args.normalize_agent_obs),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved counterfactual master coordination analysis to: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual master coordination analysis.")
    parser.add_argument(
        "--source-root",
        default=os.path.join("experiment_runs", "full_26_04_2026-11_40_39"),
    )
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[123, 42])
    parser.add_argument("--episodes-per-env", type=int, default=20)
    parser.add_argument("--replay-seed-base", type=int, default=20260426)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Use a short path on Windows.",
    )
    parser.add_argument(
        "--normalize-agent-obs",
        action="store_true",
        help="Enable this for checkpoints trained with Driver.NORMALIZE_AGENT_OBS=True.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_counterfactual(parse_args())
