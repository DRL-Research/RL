from __future__ import annotations

import argparse
import csv
import json
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


ENV_MODES = [
    ("held_out", dict(use_held_out=True, use_conflict_only=False)),
    ("conflict", dict(use_held_out=False, use_conflict_only=True)),
]

ROLE_ORDER = ("LM1", "LM2", "GM")


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


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
        "value": _scalar(value),
        "log_prob": _scalar(log_prob),
        "mean": mean.astype(float),
        "std": np.maximum(std.astype(float), 1e-6),
    }


def _kl_diag_gaussian(mu0: np.ndarray, sd0: np.ndarray, mu1: np.ndarray, sd1: np.ndarray) -> float:
    var0 = np.square(sd0)
    var1 = np.square(sd1)
    return float(np.sum(np.log(sd1 / sd0) + (var0 + np.square(mu0 - mu1)) / (2.0 * var1) - 0.5))


def _symmetric_kl(a: dict[str, Any], b: dict[str, Any]) -> float:
    return 0.5 * (
        _kl_diag_gaussian(a["mean"], a["std"], b["mean"], b["std"])
        + _kl_diag_gaussian(b["mean"], b["std"], a["mean"], a["std"])
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _json_list(values: Any) -> str:
    return json.dumps(np.asarray(values, dtype=float).reshape(-1).tolist(), separators=(",", ":"))


def _current_scenario_idx(wrapped_env) -> int:
    try:
        inner = wrapped_env.env._get_unwrapped_env()
        return int(getattr(inner, "last_scenario_index", -1))
    except Exception:
        return -1


def _run_traced_episode(
    *,
    env_label: str,
    mode: str,
    episode_idx: int,
    wrapped_env,
    exp,
    master_model,
    agent_model,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    emb_size = exp.EMBEDDING_SIZE
    n_agents = exp.CARS_AMOUNT
    agents_per_lm = exp.AGENTS_PER_LOCAL_MASTER
    gamma = float(getattr(exp, "GAMMA", 0.9))

    wrapped_env.reset()
    scenario_idx = _current_scenario_idx(wrapped_env)
    done = False
    truncated = False
    step = 0
    episode_reward = 0.0
    global_emb_prev = np.zeros(emb_size, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []

    while not done and not truncated:
        step += 1
        all_states = np.asarray(wrapped_env.env.current_state, dtype=np.float32)

        lm1_input = _build_local_master_input(global_emb_prev, all_states[0:agents_per_lm])
        lm2_input = _build_local_master_input(global_emb_prev, all_states[agents_per_lm:n_agents])
        lm1_emb, _, _ = master_model.get_proto_action(lm1_input)
        lm2_emb, _, _ = master_model.get_proto_action(lm2_input)
        gm_input = _build_global_master_input(lm1_emb, lm2_emb)
        gm_emb, _, _ = master_model.get_proto_action(gm_input)

        role_payload = {
            "LM1": (lm1_input, np.asarray(lm1_emb, dtype=np.float32).reshape(-1)),
            "LM2": (lm2_input, np.asarray(lm2_emb, dtype=np.float32).reshape(-1)),
            "GM": (gm_input, np.asarray(gm_emb, dtype=np.float32).reshape(-1)),
        }
        role_stats = {
            role: _policy_stats(master_model, inp, emb)
            for role, (inp, emb) in role_payload.items()
        }

        local_embeddings = [lm1_emb] * agents_per_lm + [lm2_emb] * agents_per_lm
        car_observations = wrapped_env.env.build_full_obs(local_embeddings)
        actions = Driver.get_action(
            agent_model,
            car_observations,
            episode_idx,
            0,
            policy_stochastic=False,
        )
        cars_scalar_action = []
        for action in actions:
            scalar, _ = get_scaler_action_and_action_array(action)
            cars_scalar_action.append(scalar)

        _, reward, done, truncated, info = wrapped_env.step(tuple(cars_scalar_action))
        episode_reward += float(reward)

        inner = wrapped_env.env._get_unwrapped_env()
        trace_row = build_step_trace_row(step, list(cars_scalar_action), float(reward), info, inner, exp)
        step_trace.append(trace_row)

        per_agent_rewards = info.get("agents_rewards", None)
        if per_agent_rewards is None or len(per_agent_rewards) < n_agents:
            per_agent_rewards = [reward] * n_agents

        reward_scale = 1.0 / max(
            1.0,
            abs(float(getattr(exp, "COLLISION_REWARD", -50.0))),
            abs(float(getattr(exp, "REACHED_TARGET_REWARD", 50.0))),
        )
        lm1_reward = min(per_agent_rewards[0:agents_per_lm]) * reward_scale
        lm2_reward = min(per_agent_rewards[agents_per_lm:n_agents]) * reward_scale
        gm_reward = float(reward) * reward_scale
        if getattr(exp, "AGENT_REWARD_MODE", "global") == "global":
            lm1_reward = gm_reward
            lm2_reward = gm_reward

        kl_lm1_lm2 = _symmetric_kl(role_stats["LM1"], role_stats["LM2"])
        kl_lm1_gm = _symmetric_kl(role_stats["LM1"], role_stats["GM"])
        kl_lm2_gm = _symmetric_kl(role_stats["LM2"], role_stats["GM"])
        cosine_lm = _cosine(role_payload["LM1"][1], role_payload["LM2"][1])

        shared = {
            "episode": episode_idx,
            "env": env_label,
            "mode": mode,
            "scenario_idx": scenario_idx,
            "step": step,
            "step_reward": float(reward),
            "actions_scalar": _json_list(cars_scalar_action),
            "min_pairwise_dist_active_m": trace_row.get("min_pairwise_dist_active_m"),
            "min_dist_active_to_uncontrolled_m": trace_row.get("min_dist_active_to_uncontrolled_m"),
            "n_active_controlled": trace_row.get("n_active_controlled"),
            "kl_lm1_lm2": kl_lm1_lm2,
            "kl_lm1_gm": kl_lm1_gm,
            "kl_lm2_gm": kl_lm2_gm,
            "cosine_lm1_lm2": cosine_lm,
        }
        role_rewards = {"LM1": float(lm1_reward), "LM2": float(lm2_reward), "GM": float(gm_reward)}
        for role in ROLE_ORDER:
            inp, emb = role_payload[role]
            stats = role_stats[role]
            rows.append({
                **shared,
                "role": role,
                "embedding": _json_list(emb),
                "input": _json_list(inp),
                "policy_mean": _json_list(stats["mean"]),
                "policy_std": _json_list(stats["std"]),
                "value_pred": float(stats["value"]),
                "log_prob": float(stats["log_prob"]),
                "role_reward_scaled": role_rewards[role],
                "discounted_return_scaled": None,
                "value_loss_proxy": None,
            })

        global_emb_prev = np.asarray(gm_emb, dtype=np.float32).reshape(-1)

    crashed = any(any(r.get("crashed_flags", [])) for r in step_trace)
    first_crash_step = None
    for r in step_trace:
        if any(r.get("crashed_flags", [])):
            first_crash_step = int(r["step"])
            break

    for role in ROLE_ORDER:
        idxs = [i for i, row in enumerate(rows) if row["episode"] == episode_idx and row["role"] == role]
        running_return = 0.0
        for i in reversed(idxs):
            running_return = float(rows[i]["role_reward_scaled"]) + gamma * running_return
            rows[i]["discounted_return_scaled"] = running_return
            rows[i]["value_loss_proxy"] = float((rows[i]["value_pred"] - running_return) ** 2)

    for row in rows:
        if row["episode"] != episode_idx:
            continue
        row["crashed_episode"] = int(crashed)
        row["first_crash_step"] = first_crash_step if first_crash_step is not None else ""
        if crashed and first_crash_step is not None:
            steps_to_crash = first_crash_step - int(row["step"])
            row["steps_to_crash"] = steps_to_crash
            if steps_to_crash == 0:
                phase = "crash_step"
            elif 1 <= steps_to_crash <= 3:
                phase = "pre_crash_3"
            else:
                phase = "crash_episode_other"
        else:
            row["steps_to_crash"] = ""
            min_pair = row.get("min_pairwise_dist_active_m")
            phase = "near_miss_safe" if min_pair is not None and min_pair != "" and float(min_pair) < 8.0 else "safe"
        row["phase"] = phase

    arrived = sum(
        1 for v in wrapped_env.env._get_unwrapped_env().controlled_vehicles
        if hasattr(v, "is_arrived") and v.is_arrived
    )
    episode_summary = {
        "episode": episode_idx,
        "env": env_label,
        "mode": mode,
        "scenario_idx": scenario_idx,
        "steps": step,
        "reward": episode_reward,
        "arrival_pct": 100.0 * arrived / max(1, n_agents),
        "crashed": int(crashed),
        "first_crash_step": first_crash_step if first_crash_step is not None else "",
    }
    return rows, episode_summary


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_pca(rows: list[dict[str, Any]], out_dir: str) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    os.makedirs(out_dir, exist_ok=True)

    embs = np.asarray([json.loads(r["embedding"]) for r in rows], dtype=float)
    roles = np.asarray([r["role"] for r in rows])
    phases = np.asarray([r["phase"] for r in rows])
    envs = np.asarray([r["env"] for r in rows])

    pca = PCA(n_components=2)
    xy = pca.fit_transform(embs)
    for row, coords in zip(rows, xy):
        row["pc1"] = float(coords[0])
        row["pc2"] = float(coords[1])

    phase_colors = {
        "safe": "#2E7D32",
        "near_miss_safe": "#90A4AE",
        "crash_episode_other": "#FFB300",
        "pre_crash_3": "#E53935",
        "crash_step": "#7B1FA2",
    }
    markers = {"LM1": "o", "LM2": "^", "GM": "s"}

    fig, ax = plt.subplots(figsize=(11, 8))
    for phase in phase_colors:
        for role in ROLE_ORDER:
            mask = (phases == phase) & (roles == role)
            if np.any(mask):
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    c=phase_colors[phase],
                    marker=markers[role],
                    s=18,
                    alpha=0.48,
                    label=f"{phase} {role}",
                )
    ax.set_title("Master Embedding PCA by Crash Phase and Role")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_by_crash_phase_and_role.png"), dpi=180)
    plt.close(fig)

    env_colors = {"intersection": "#1E88E5", "roundabout": "#43A047", "double_intersection": "#FB8C00"}
    fig, ax = plt.subplots(figsize=(11, 8))
    for env in env_colors:
        for role in ROLE_ORDER:
            mask = (envs == env) & (roles == role)
            if np.any(mask):
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    c=env_colors[env],
                    marker=markers[role],
                    s=16,
                    alpha=0.42,
                    label=f"{env} {role}",
                )
    ax.set_title("Master Embedding PCA by Environment and Role")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_by_environment_and_role.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    phase_order = [p for p in phase_colors if np.any(phases == p)]
    axes[0].boxplot(
        [[float(r["value_loss_proxy"]) for r in rows if r["phase"] == p] for p in phase_order],
        labels=phase_order,
        showfliers=False,
    )
    axes[0].set_title("Master Value Loss Proxy by Phase")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, alpha=0.3)
    axes[1].boxplot(
        [[float(r["kl_lm1_lm2"]) for r in rows if r["phase"] == p and r["role"] == "GM"] for p in phase_order],
        labels=phase_order,
        showfliers=False,
    )
    axes[1].set_title("LM1-LM2 Policy KL by Phase")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_loss_and_kl_by_phase.png"), dpi=180)
    plt.close(fig)

    summary: dict[str, Any] = {
        "n_points": int(len(rows)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_explained_variance_2d_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    for label_name, labels in [("role", roles), ("phase", phases), ("env", envs)]:
        unique = sorted(set(labels.tolist()))
        if len(unique) > 1 and min(np.sum(labels == u) for u in unique) >= 2:
            summary[f"silhouette_{label_name}_embedding"] = float(silhouette_score(embs, labels))
            summary[f"silhouette_{label_name}_pca2"] = float(silhouette_score(xy, labels))

    safe_mask = phases == "safe"
    danger_mask = np.isin(phases, ["pre_crash_3", "crash_step"])
    if np.any(safe_mask) and np.any(danger_mask):
        safe_centroid = xy[safe_mask].mean(axis=0)
        danger_centroid = xy[danger_mask].mean(axis=0)
        summary["safe_vs_danger_pca_centroid_distance"] = float(np.linalg.norm(safe_centroid - danger_centroid))
        summary["safe_points"] = int(np.sum(safe_mask))
        summary["danger_points"] = int(np.sum(danger_mask))

    return summary


def _summarize_rows(rows: list[dict[str, Any]], episodes: list[dict[str, Any]], out_dir: str) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    def mean_for(key: str, pred) -> float | None:
        vals = [float(r[key]) for r in rows if pred(r) and r.get(key) not in ("", None)]
        return float(np.mean(vals)) if vals else None

    phase_summary = []
    phases = sorted(set(r["phase"] for r in rows))
    for phase in phases:
        subset = [r for r in rows if r["phase"] == phase]
        phase_summary.append({
            "phase": phase,
            "n": len(subset),
            "mean_value_loss_proxy": mean_for("value_loss_proxy", lambda r, p=phase: r["phase"] == p),
            "mean_kl_lm1_lm2": mean_for("kl_lm1_lm2", lambda r, p=phase: r["phase"] == p and r["role"] == "GM"),
            "mean_cosine_lm1_lm2": mean_for("cosine_lm1_lm2", lambda r, p=phase: r["phase"] == p and r["role"] == "GM"),
        })
    _write_csv(os.path.join(out_dir, "phase_summary.csv"), phase_summary)

    return {
        "episodes": len(episodes),
        "crash_episodes": int(sum(e["crashed"] for e in episodes)),
        "arrival_pct_mean": float(np.mean([e["arrival_pct"] for e in episodes])) if episodes else 0.0,
        "phase_summary": phase_summary,
    }


def run_deep_dive(args: argparse.Namespace) -> str:
    from src.model.master_model import MasterModel

    _set_all_seeds(args.seed)
    MasterModel.NORMALIZE_INPUTS = bool(args.normalize_master_inputs)
    Driver.NORMALIZE_AGENT_OBS = bool(getattr(args, "normalize_agent_obs", False))

    source_run = os.path.abspath(os.path.normpath(args.source_run))
    ckpt = os.path.join(source_run, "best", "ckpt")
    if not os.path.exists(ckpt + "_agent.pth"):
        ckpt = os.path.join(source_run, "trained_model")
    if not os.path.exists(ckpt + "_agent.pth"):
        raise FileNotFoundError(f"Could not find checkpoint under {source_run}")

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    if args.output_dir:
        out_dir = os.path.abspath(os.path.normpath(args.output_dir))
    else:
        run_root = os.path.abspath(os.path.join(source_run, "..", "..", ".."))
        out_dir = os.path.join(run_root, f"master_deep_s{args.seed}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    exp = unified._make_experiment(out_dir, args.episodes_per_env, env_id="RELintersection-v0")
    setup_experiment_dirs(out_dir)
    inter_cfg = unified._make_env_config("RELintersection-v0")
    master_model, agent_model, _ = initialize_models(exp, inter_cfg)
    load_models(agent_model, master_model, ckpt)
    master_model.freeze()

    all_rows: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    episode_idx = 0

    for mode, mode_kwargs in ENV_MODES:
        for env_id, env_label in unified.ENV_DEFS:
            project_globals.after_is_arrived_flags = [False] * 6
            exp_test = unified._make_experiment(out_dir, args.episodes_per_env, env_id=env_id)
            exp_test.WARMUP_EPISODES = 0
            exp_test.CONFIG = unified._make_env_config(env_id, **mode_kwargs)

            def _env_fn(ec=exp_test):
                return Driver(ec)

            wrapped_env = DummyVecEnv([_env_fn])
            for _ in range(args.episodes_per_env):
                episode_idx += 1
                rows, summary = _run_traced_episode(
                    env_label=env_label,
                    mode=mode,
                    episode_idx=episode_idx,
                    wrapped_env=wrapped_env,
                    exp=exp_test,
                    master_model=master_model,
                    agent_model=agent_model,
                )
                all_rows.extend(rows)
                all_episodes.append(summary)
                print(
                    f"  {mode}/{env_label} ep={summary['episode']} "
                    f"arrival={summary['arrival_pct']:.0f}% crashed={summary['crashed']}"
                )
            try:
                wrapped_env.close()
            except Exception:
                pass

    pca_summary = _plot_pca(all_rows, out_dir)
    aggregate_summary = _summarize_rows(all_rows, all_episodes, out_dir)

    _write_csv(os.path.join(out_dir, "master_embedding_steps.csv"), all_rows)
    _write_csv(os.path.join(out_dir, "episode_summary.csv"), all_episodes)
    np.savez_compressed(
        os.path.join(out_dir, "embeddings_raw.npz"),
        embeddings=np.asarray([json.loads(r["embedding"]) for r in all_rows], dtype=np.float32),
        roles=np.asarray([r["role"] for r in all_rows]),
        phases=np.asarray([r["phase"] for r in all_rows]),
        envs=np.asarray([r["env"] for r in all_rows]),
        modes=np.asarray([r["mode"] for r in all_rows]),
    )

    metadata = {
        "source_run": source_run,
        "checkpoint": ckpt,
        "seed": args.seed,
        "episodes_per_env_per_mode": args.episodes_per_env,
        "modes": [m for m, _ in ENV_MODES],
        "envs": [label for _, label in unified.ENV_DEFS],
        **aggregate_summary,
        **pca_summary,
    }
    with open(os.path.join(out_dir, "deep_dive_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved master embedding deep dive to: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master embedding deep-dive diagnostics for a single run.")
    parser.add_argument(
        "--source-run",
        default=os.path.join("experiment_runs", "full_26_04_2026-11_40_39", "A_base", "W_MASTER", "s123"),
        help="Path to an A_base/W_MASTER/s<seed> run directory.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes-per-env", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional short output directory. Useful on Windows to avoid MAX_PATH failures.",
    )
    parser.add_argument(
        "--normalize-master-inputs",
        action="store_true",
        help="Use the normalized master-input convention required by normalized-master checkpoints.",
    )
    parser.add_argument(
        "--normalize-agent-obs",
        action="store_true",
        help="Use the normalized agent-observation convention required by normalized-agent checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_deep_dive(parse_args())
