from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from highwayenv.utils import (
    patch_intersection_env,
    register_double_intersection_env,
    register_intersection_env,
    register_roundabout_env,
)
from src import project_globals
from src.diagnostics.collision_audit import build_step_trace_row
from src.experiment.experiment_config import Experiment
from src.experiment.scenario_geometry import episode_geometry_fields
from src.experiment.scenarios import base_complete_scenarios_6_cars
from src.experiment.scenarios_config import make_env_config_exp7
from src.experiment.new_envs_config import (
    make_double_intersection_env_config,
    make_roundabout_env_config,
)
from src.model.master_model import MasterModel
from src.model.model_handler import load_models, load_models_from_paths, save_models

patch_intersection_env()
register_intersection_env()
register_roundabout_env()
register_double_intersection_env()


SEED = 123
ENV_ID = "RELintersection-v0"
PRETRAINED_CHECKPOINT = os.path.join(
    "experiment_runs",
    "grid_23_04_2026-13_51_18",
    "Q02_ep1_ent0005_PL75",
    "best_model",
    "checkpoint",
)


BASE_CFG = dict(
    label="C00_legacy_baseline",
    embedding_dim=4,
    episodes=2500,
    test_episodes=100,
    collision_reward=-50,
    arrived_reward=50,
    starvation_reward=0,
    high_speed_reward=5,
    reward_mode="global",
    master_lr=3e-4,
    agent_lr=3e-3,
    gamma=0.9,
    clip_range=0.2,
    vf_coef=1.0,
    ent_coef=0.005,
    n_ppo_epochs=5,
    warmup_episodes=200,
    peak_arrival_threshold=75.0,
    target_speeds=[5, 10],
    train_policy_mean=False,
    eval_policy_mean=False,
    master_log_std_init=None,
    normalize_master_inputs=False,
    normalize_agent_obs=False,
    conflict_schedule=[],
    test_conflict_ratio=0.5,
    distance_reward_weight=0.0,
    target_min_pairwise_dist=8.0,
    close_distance_threshold=8.0,
    critical_distance_threshold=4.0,
    risk_aux_weight=0.0,
    distance_aux_weight=0.0,
    risk_aux_start_episode=1,
    risk_aux_ramp_episodes=0,
    distance_aux_start_episode=1,
    distance_aux_ramp_episodes=0,
    risk_horizon_steps=5,
    master_dropout_prob=0.0,
    master_noise_std=0.0,
    eval_conditions=list(("normal", "zero_master", "swap_local_masters", "negate_master")),
    load_pretrained=True,
)


SWEEP_CONFIGS = [
    BASE_CFG,
    {**BASE_CFG, "label": "C01_mean_proto_eval", "eval_policy_mean": True},
    {**BASE_CFG, "label": "C02_mean_proto_train_eval", "train_policy_mean": True, "eval_policy_mean": True, "load_pretrained": False},
    {**BASE_CFG, "label": "C03_low_master_std", "master_log_std_init": -2.0, "load_pretrained": False},
    {**BASE_CFG, "label": "C04_master_input_norm", "normalize_master_inputs": True, "load_pretrained": False},
    {**BASE_CFG, "label": "C05_master_agent_norm", "normalize_master_inputs": True, "normalize_agent_obs": True, "load_pretrained": False},
    {**BASE_CFG, "label": "C06_embedding_dim_2", "embedding_dim": 2, "load_pretrained": False},
    {**BASE_CFG, "label": "C07_embedding_dim_8", "embedding_dim": 8, "load_pretrained": False},
    {**BASE_CFG, "label": "C08_group_reward", "reward_mode": "group", "load_pretrained": False},
    {
        **BASE_CFG,
        "label": "C09_conflict_curriculum",
        "conflict_schedule": [(0, 0.0), (800, 0.2), (1600, 0.5)],
        "load_pretrained": False,
    },
]

CONDITIONS = ("normal", "zero_master", "swap_local_masters", "negate_master")
ROLES = ("LM1", "LM2", "GM")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_vec(values: Any) -> str:
    return json.dumps(np.asarray(values, dtype=float).reshape(-1).tolist(), separators=(",", ":"))


def schedule_value(schedule: list[tuple[int, float]], episode: int) -> float:
    value = 0.0
    for start, ratio in sorted(schedule):
        if episode >= int(start):
            value = float(ratio)
    return value


def ramped_weight(base_weight: float, start_episode: int, ramp_episodes: int, episode: int) -> float:
    base_weight = float(base_weight or 0.0)
    if base_weight <= 0.0:
        return 0.0
    if episode < int(start_episode):
        return 0.0
    ramp_episodes = int(ramp_episodes or 0)
    if ramp_episodes <= 0:
        return base_weight
    progress = min(1.0, max(0.0, (episode - int(start_episode) + 1) / float(ramp_episodes)))
    return base_weight * progress


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


class StaticPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, observation_dim: int, action_space):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        self.action_space = action_space

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, True, {}


class RiskAuxHead(nn.Module):
    def __init__(self, proto_dim: int):
        super().__init__()
        hidden = max(8, proto_dim * 4)
        self.net = nn.Sequential(
            nn.Linear(proto_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, proto_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(proto_mean)
        risk_logit = out[:, 0]
        distance_pred = torch.sigmoid(out[:, 1])
        return risk_logit, distance_pred


@dataclass
class ProtoExperiment:
    cfg: dict
    out_dir: str
    exp: Experiment = field(init=False)
    embedding_dim: int = field(init=False)
    slot_vec_dim: int = field(init=False)
    master_obs_dim: int = field(init=False)
    agent_obs_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.embedding_dim = int(self.cfg["embedding_dim"])
        self.slot_vec_dim = max(4, self.embedding_dim)
        self.master_obs_dim = 5 * (self.slot_vec_dim + 1)
        self.agent_obs_dim = 4 + self.embedding_dim
        self.exp = Experiment(
            RENDER_MODE=None,
            EXPERIMENT_ID=f"proto_{self.cfg['label']}",
            EPOCHS=1,
            CYCLES=1,
            EPISODES_PER_CYCLE=int(self.cfg["episodes"]),
            COLLISION_REWARD=float(self.cfg["collision_reward"]),
            REACHED_TARGET_REWARD=float(self.cfg["arrived_reward"]),
            STARVATION_REWARD=float(self.cfg["starvation_reward"]),
            HIGH_SPEED_REWARD=float(self.cfg["high_speed_reward"]),
            AGENT_REWARD_MODE=str(self.cfg["reward_mode"]),
            MASTER_LR=float(self.cfg["master_lr"]),
            AGENT_LR=float(self.cfg["agent_lr"]),
            GAMMA=float(self.cfg["gamma"]),
            CLIP_RANGE=float(self.cfg["clip_range"]),
            VF_COEF=float(self.cfg["vf_coef"]),
            ENT_COEF=float(self.cfg["ent_coef"]),
            EMBEDDING_SIZE=self.embedding_dim,
            STATE_INPUT_SIZE=self.agent_obs_dim,
            MASTER_OBS_DIM=self.master_obs_dim,
            WARMUP_EPISODES=int(self.cfg["warmup_episodes"]),
            PEAK_ARRIVAL_THRESHOLD=float(self.cfg["peak_arrival_threshold"]),
            N_STEPS=4096,
        )
        self.exp.EXPERIMENT_PATH = self.out_dir
        self.exp.SAVE_MODEL_DIRECTORY = os.path.join(self.out_dir, "trained_model")


class ProtoHighwayWrapper:
    def __init__(
        self,
        proto_exp: ProtoExperiment,
        *,
        conflict_ratio: float = 0.0,
        conflict_only: bool = False,
        env_extra: dict[str, Any] | None = None,
        render_mode: str | None = None,
        step_sleep_seconds: float = 0.0,
        env_id: str | None = None,
    ):
        self.proto_exp = proto_exp
        self.cfg = proto_exp.cfg
        self.env_id = env_id or ENV_ID
        self.step_sleep_seconds = float(step_sleep_seconds)
        kw = dict(
            collision_reward=self.cfg["collision_reward"],
            arrived_reward=self.cfg["arrived_reward"],
            starvation_reward=self.cfg["starvation_reward"],
            high_speed_reward=self.cfg["high_speed_reward"],
            target_speeds=self.cfg["target_speeds"],
        )
        if self.env_id == "RELintersection-v0":
            self.env_config = make_env_config_exp7(**kw)
        elif self.env_id == "RELroundabout-v0":
            self.env_config = make_roundabout_env_config(**kw)
        elif self.env_id == "RELdouble-intersection-v0":
            self.env_config = make_double_intersection_env_config(**kw)
        else:
            raise ValueError(f"ProtoHighwayWrapper: unsupported env_id {self.env_id!r}")
        self.env_config["conflict_ratio"] = float(conflict_ratio)
        self.env_config["use_conflict_scenarios_only"] = bool(conflict_only)
        self.env_config["use_held_out_scenarios"] = False
        if env_extra:
            self.env_config.update(env_extra)
        self.env = gym.make(self.env_id, render_mode=render_mode, config=self.env_config)
        self.current_state = np.zeros((6, 4), dtype=np.float32)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def set_conflict_ratio(self, ratio: float) -> None:
        try:
            self.env.unwrapped.config["conflict_ratio"] = float(ratio)
        except Exception:
            pass

    def reset(self, seed: int | None = None, *, custom_scenario_index: int | None = None) -> np.ndarray:
        project_globals.after_is_arrived_flags = [False] * 6
        ic = self.inner()
        if custom_scenario_index is not None:
            ic.config["custom_regular_episode_index"] = int(custom_scenario_index)
        else:
            ic.config.pop("custom_regular_episode_index", None)
        if seed is None:
            self.env.reset()
        else:
            self.env.reset(seed=seed)
        self.current_state = self.read_state()
        return self.current_state

    def step(self, actions: list[int]):
        _, reward, done, truncated, info = self.env.step(tuple(int(a) for a in actions))
        self.current_state = self.read_state()
        return self.current_state, float(reward), bool(done), bool(truncated), info

    def inner(self):
        env = self.env
        while hasattr(env, "env") and not hasattr(env, "controlled_vehicles"):
            env = env.env
        return env

    def read_state(self) -> np.ndarray:
        inner = self.inner()
        states = []
        for vehicle in inner.controlled_vehicles:
            if hasattr(vehicle, "is_arrived") and vehicle.is_arrived:
                states.append([0.0, 0.0, 0.0, 0.0])
                continue
            vel = vehicle.velocity if hasattr(vehicle, "velocity") else np.zeros(2)
            states.append([float(vehicle.position[0]), float(vehicle.position[1]), float(vel[0]), float(vel[1])])
        return np.asarray(states, dtype=np.float32)


def pad_vec(vec: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, len(arr))] = arr[: min(size, len(arr))]
    return out


def normalize_state4(state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.float32).copy()
    out[0:2] /= 120.0
    out[2:4] /= 15.0
    return out


def master_slot(vec: np.ndarray, identifier: float, slot_vec_dim: int) -> np.ndarray:
    return np.concatenate([pad_vec(vec, slot_vec_dim), np.asarray([identifier], dtype=np.float32)])


def build_local_master_input(global_emb: np.ndarray, states: np.ndarray, proto_exp: ProtoExperiment) -> np.ndarray:
    slots = [master_slot(global_emb, 1.0, proto_exp.slot_vec_dim)]
    for state in np.asarray(states, dtype=np.float32):
        slot_state = normalize_state4(state) if proto_exp.cfg["normalize_master_inputs"] else state
        slots.append(master_slot(slot_state, 0.0, proto_exp.slot_vec_dim))
    while len(slots) < 5:
        slots.append(np.zeros(proto_exp.slot_vec_dim + 1, dtype=np.float32))
    return np.concatenate(slots[:5]).astype(np.float32)


def build_global_master_input(lm1_emb: np.ndarray, lm2_emb: np.ndarray, proto_exp: ProtoExperiment) -> np.ndarray:
    slots = [
        master_slot(lm1_emb, 1.0, proto_exp.slot_vec_dim),
        master_slot(lm2_emb, 1.0, proto_exp.slot_vec_dim),
    ]
    while len(slots) < 5:
        slots.append(np.zeros(proto_exp.slot_vec_dim + 1, dtype=np.float32))
    return np.concatenate(slots[:5]).astype(np.float32)


def build_agent_obs(states: np.ndarray, lm1_emb: np.ndarray, lm2_emb: np.ndarray, proto_exp: ProtoExperiment) -> list[np.ndarray]:
    out = []
    agents_per_lm = 3
    for i, raw_state in enumerate(np.asarray(states, dtype=np.float32)):
        car_state = normalize_state4(raw_state) if proto_exp.cfg["normalize_agent_obs"] else raw_state
        emb = lm1_emb if i < agents_per_lm else lm2_emb
        out.append(np.concatenate([car_state.astype(np.float32), np.asarray(emb, dtype=np.float32)]).astype(np.float32))
    return out


def policy_dist_stats(policy, obs: np.ndarray) -> dict[str, Any]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        value = policy.predict_values(obs_t)
        dist = policy.get_distribution(obs_t)
        base = getattr(dist, "distribution", None)
        if base is not None and hasattr(base, "mean"):
            mean = base.mean.detach().cpu().numpy().reshape(-1)
            std = np.maximum(base.stddev.detach().cpu().numpy().reshape(-1), 1e-6)
        else:
            probs = base.probs.detach().cpu().numpy().reshape(-1)
            mean = probs
            std = np.maximum(np.sqrt(probs * (1.0 - probs)), 1e-6)
    return {"value": value.reshape(-1), "mean": mean, "std": std}


def master_action(master_model: MasterModel, obs: np.ndarray, deterministic: bool) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict[str, Any]]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        actions, values, log_prob = master_model.model.policy.forward(obs_t, deterministic=deterministic)
        dist = master_model.model.policy.get_distribution(obs_t)
        base = getattr(dist, "distribution", None)
        if base is not None and hasattr(base, "mean"):
            mean = base.mean.detach().cpu().numpy().reshape(-1)
            std = np.maximum(base.stddev.detach().cpu().numpy().reshape(-1), 1e-6)
        else:
            mean = actions.detach().cpu().numpy().reshape(-1)
            std = np.ones_like(mean)
    action_np = actions.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return action_np, values.reshape(-1), log_prob.reshape(-1), {"mean": mean, "std": std}


def agent_actions(agent_model: PPO, observations: list[np.ndarray], deterministic: bool):
    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    with torch.no_grad():
        actions, values, log_prob = agent_model.policy.forward(obs_t, deterministic=deterministic)
    actions_np = actions.detach().cpu().numpy().reshape(-1).astype(int).tolist()
    return actions_np, values.reshape(-1), log_prob.reshape(-1)


def agent_values_log_probs(agent_model: PPO, observations: list[np.ndarray], actions: list[int]):
    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    actions_t = torch.as_tensor(np.asarray(actions, dtype=np.int64))
    with torch.no_grad():
        values = agent_model.policy.predict_values(obs_t).reshape(-1)
        _, log_probs, _ = agent_model.policy.evaluate_actions(obs_t, actions_t)
    return values, log_probs.reshape(-1)


def append_transition(store: list[dict[str, Any]], obs, action, reward, value, log_prob, done, **extra) -> None:
    store.append({
        "obs": np.asarray(obs, dtype=np.float32),
        "action": np.asarray(action),
        "reward": float(reward),
        "value": float(value.detach().cpu().reshape(-1)[0] if isinstance(value, torch.Tensor) else value),
        "log_prob": float(log_prob.detach().cpu().reshape(-1)[0] if isinstance(log_prob, torch.Tensor) else log_prob),
        "done": bool(done),
        **extra,
    })


def label_master_risk_targets(transitions: list[dict[str, Any]], step_trace: list[dict[str, Any]], cfg: dict) -> None:
    if not transitions or not step_trace:
        return
    horizon = int(cfg.get("risk_horizon_steps", 5))
    target_dist = max(1e-6, float(cfg.get("target_min_pairwise_dist", 8.0)))
    for t in transitions:
        step_idx = int(t.get("step", 1))
        future = [r for r in step_trace if step_idx <= int(r["step"]) <= step_idx + horizon]
        crashed_future = any(any(r.get("crashed_flags", [])) for r in future)
        dists = [
            safe_float(r.get("min_pairwise_dist_active_m"), default=target_dist)
            for r in future
            if r.get("min_pairwise_dist_active_m") not in ("", None)
        ]
        min_future = min(dists) if dists else target_dist
        t["risk_target"] = 1.0 if crashed_future else 0.0
        t["distance_target"] = min(1.0, max(0.0, min_future / target_dist))


def compute_returns(transitions: list[dict[str, Any]], gamma: float) -> tuple[np.ndarray, np.ndarray]:
    returns = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(transitions))):
        if transitions[i]["done"]:
            running = 0.0
        running = transitions[i]["reward"] + gamma * running
        returns[i] = running
    values = np.asarray([t["value"] for t in transitions], dtype=np.float32)
    advantages = returns - values
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def ppo_update(
    policy,
    transitions: list[dict[str, Any]],
    *,
    discrete: bool,
    cfg: dict,
    aux_head: RiskAuxHead | None = None,
    episode: int | None = None,
) -> dict[str, float]:
    if not transitions:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
    returns, advantages = compute_returns(transitions, float(cfg["gamma"]))
    obs = torch.as_tensor(np.asarray([t["obs"] for t in transitions], dtype=np.float32))
    if discrete:
        actions = torch.as_tensor(np.asarray([int(np.asarray(t["action"]).reshape(-1)[0]) for t in transitions]), dtype=torch.long)
    else:
        actions = torch.as_tensor(np.asarray([t["action"] for t in transitions], dtype=np.float32))
    old_log_probs = torch.as_tensor(np.asarray([t["log_prob"] for t in transitions], dtype=np.float32))
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)

    last = {}
    for _ in range(int(cfg["n_ppo_epochs"])):
        values, log_probs, entropy = policy.evaluate_actions(obs, actions)
        values = values.reshape(-1)
        log_probs = log_probs.reshape(-1)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = advantages_t * ratio
        clipped = advantages_t * torch.clamp(ratio, 1.0 - float(cfg["clip_range"]), 1.0 + float(cfg["clip_range"]))
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns_t)
        entropy_loss = -entropy.mean() if entropy is not None else torch.tensor(0.0)
        loss = policy_loss + float(cfg["vf_coef"]) * value_loss + float(cfg["ent_coef"]) * entropy_loss
        risk_loss = torch.tensor(0.0)
        distance_loss = torch.tensor(0.0)
        ep = int(episode or 1)
        risk_weight = ramped_weight(
            cfg.get("risk_aux_weight", 0.0),
            cfg.get("risk_aux_start_episode", 1),
            cfg.get("risk_aux_ramp_episodes", 0),
            ep,
        )
        distance_weight = ramped_weight(
            cfg.get("distance_aux_weight", 0.0),
            cfg.get("distance_aux_start_episode", 1),
            cfg.get("distance_aux_ramp_episodes", 0),
            ep,
        )
        if aux_head is not None and not discrete and risk_weight + distance_weight > 0:
            dist = policy.get_distribution(obs)
            base = getattr(dist, "distribution", None)
            proto_mean = base.mean if base is not None and hasattr(base, "mean") else actions.float()
            risk_logit, distance_pred = aux_head(proto_mean)
            risk_targets = torch.as_tensor(
                np.asarray([float(t.get("risk_target", 0.0)) for t in transitions], dtype=np.float32)
            )
            dist_targets = torch.as_tensor(
                np.asarray([float(t.get("distance_target", 1.0)) for t in transitions], dtype=np.float32)
            )
            risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(risk_logit.reshape(-1), risk_targets)
            distance_loss = torch.nn.functional.mse_loss(distance_pred.reshape(-1), dist_targets)
            loss = (
                loss
                + risk_weight * risk_loss
                + distance_weight * distance_loss
            )
        policy.optimizer.zero_grad()
        if aux_head is not None:
            aux_head.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()
        if aux_head is not None:
            aux_head.optimizer.step()
        last = {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float((-entropy_loss).detach().cpu()),
            "risk_aux_loss": float(risk_loss.detach().cpu()),
            "distance_aux_loss": float(distance_loss.detach().cpu()),
            "risk_aux_weight": float(risk_weight),
            "distance_aux_weight": float(distance_weight),
            "total_loss": float(loss.detach().cpu()),
        }
    return last


def make_models(proto_exp: ProtoExperiment):
    cfg = proto_exp.cfg
    MasterModel.NORMALIZE_INPUTS = False
    MasterModel.DETERMINISTIC_PROTO = False
    MasterModel.LOG_STD_INIT = cfg["master_log_std_init"]
    master_model = MasterModel(
        embedding_size=proto_exp.embedding_dim,
        experiment=proto_exp.exp,
        observation_dim=proto_exp.master_obs_dim,
    )
    MasterModel.LOG_STD_INIT = None

    arch = [256, 256]
    agent_env = StaticPPOEnv(proto_exp.agent_obs_dim, spaces.Discrete(2))
    agent_model = PPO(
        "MlpPolicy",
        agent_env,
        learning_rate=float(cfg["agent_lr"]),
        n_steps=2048,
        batch_size=128,
        gamma=float(cfg["gamma"]),
        clip_range=float(cfg["clip_range"]),
        ent_coef=0.0,
        vf_coef=float(cfg["vf_coef"]),
        policy_kwargs=dict(net_arch=[dict(pi=arch, vf=arch)]),
        verbose=0,
        device="cpu",
    )
    return master_model, agent_model


def maybe_load_pretrained(cfg: dict, master_model: MasterModel, agent_model: PPO) -> tuple[PPO, bool]:
    if not cfg.get("load_pretrained", False):
        return agent_model, False
    if int(cfg["embedding_dim"]) != 4:
        return agent_model, False
    if not os.path.exists(PRETRAINED_CHECKPOINT + "_agent.pth"):
        return agent_model, False
    try:
        loaded_agent = PPO.load(PRETRAINED_CHECKPOINT + "_agent.pth", env=agent_model.get_env(), device="cpu")
        agent_model.set_parameters(loaded_agent.get_parameters())
        master_model.load(PRETRAINED_CHECKPOINT + "_master.pth")
        print(f"Loaded pretrained weights from {PRETRAINED_CHECKPOINT}")
        return agent_model, True
    except Exception as exc:
        print(f"Failed to load pretrained weights: {exc}")
        return agent_model, False


def counterfactual_embeddings(condition: str, lm1: np.ndarray, lm2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if condition == "normal":
        return lm1, lm2
    if condition in ("zero_master", "zero_local_masters"):
        return np.zeros_like(lm1), np.zeros_like(lm2)
    if condition == "swap_local_masters":
        return lm2, lm1
    if condition == "negate_master":
        return -lm1, -lm2
    if condition in (
        "zero_global_master",
        "disconnect_global_master",
        "large_const_global_master",
        "shuffle_global_master",
        "random_global_master",
        "delayed_global_master",
        "negate_global_master",
        "const_all_masters",
    ):
        return lm1, lm2
    if condition == "random_local_masters":
        return (
            np.random.normal(0.0, 1.0, size=lm1.shape).astype(np.float32),
            np.random.normal(0.0, 1.0, size=lm2.shape).astype(np.float32),
        )
    raise ValueError(condition)


def run_episode(
    *,
    proto_exp: ProtoExperiment,
    env: ProtoHighwayWrapper,
    master_model: MasterModel,
    agent_model: PPO,
    episode: int,
    train: bool,
    condition: str = "normal",
    trace_rows: list[dict[str, Any]] | None = None,
    replay_seed: int | None = None,
    custom_scenario_index: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if replay_seed is not None:
        set_all_seeds(replay_seed)
    states = env.reset(seed=replay_seed, custom_scenario_index=custom_scenario_index)
    _inner0 = env.inner()
    _geom = episode_geometry_fields(
        scenario_pool=str(getattr(_inner0, "last_scenario_pool", "unknown")),
        scenario_index=int(getattr(_inner0, "last_scenario_index", -1)),
        scenario_base=int(getattr(_inner0, "last_base_scenario", -1)),
        base_scenarios_table=base_complete_scenarios_6_cars,
    )
    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    global_emb_prev = np.zeros(proto_exp.embedding_dim, dtype=np.float32)
    master_transitions: list[dict[str, Any]] = []
    agent_transitions: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []
    deterministic_master = (not train and proto_exp.cfg["eval_policy_mean"]) or (train and proto_exp.cfg["train_policy_mean"])
    deterministic_agent = not train
    warmup = int(proto_exp.cfg["warmup_episodes"])
    delayed_global_emb = np.zeros(proto_exp.embedding_dim, dtype=np.float32)

    while not done and not truncated:
        step += 1
        lm1_input = build_local_master_input(global_emb_prev, states[:3], proto_exp)
        lm2_input = build_local_master_input(global_emb_prev, states[3:6], proto_exp)
        lm1_emb, lm1_value, lm1_lp, lm1_stats = master_action(master_model, lm1_input, deterministic_master)
        lm2_emb, lm2_value, lm2_lp, lm2_stats = master_action(master_model, lm2_input, deterministic_master)
        gm_input = build_global_master_input(lm1_emb, lm2_emb, proto_exp)
        gm_emb, gm_value, gm_lp, gm_stats = master_action(master_model, gm_input, deterministic_master)
        gm_emb_for_next = gm_emb
        if condition == "const_all_masters":
            gv_c = float(proto_exp.cfg.get("master_broadcast_const_test", 9999.0))
            gm_emb_for_next = np.full_like(gm_emb, gv_c)
        elif condition in ("zero_global_master", "disconnect_global_master"):
            gm_emb_for_next = np.zeros_like(gm_emb)
        elif condition == "large_const_global_master":
            gv = float(proto_exp.cfg.get("large_global_master_constant", 24.0))
            gm_emb_for_next = np.full_like(gm_emb, gv)
        elif condition == "negate_global_master":
            gm_emb_for_next = -gm_emb
        elif condition == "random_global_master":
            gm_emb_for_next = np.random.normal(0.0, 1.0, size=gm_emb.shape).astype(np.float32)
        elif condition == "shuffle_global_master":
            gm_emb_for_next = gm_emb[::-1].copy()
        elif condition == "delayed_global_master":
            gm_emb_for_next = delayed_global_emb.copy()
            delayed_global_emb = gm_emb.copy()

        lm1_used, lm2_used = counterfactual_embeddings(condition, lm1_emb, lm2_emb)
        if condition == "const_all_masters":
            gv_c = float(proto_exp.cfg.get("master_broadcast_const_test", 9999.0))
            lm1_used = np.full_like(lm1_emb, gv_c)
            lm2_used = np.full_like(lm2_emb, gv_c)
        if train and float(proto_exp.cfg.get("master_dropout_prob", 0.0) or 0.0) > 0 and random.random() < float(proto_exp.cfg["master_dropout_prob"]):
            lm1_used = np.zeros_like(lm1_used)
            lm2_used = np.zeros_like(lm2_used)
        if train and float(proto_exp.cfg.get("master_noise_std", 0.0) or 0.0) > 0:
            noise_std = float(proto_exp.cfg["master_noise_std"])
            lm1_used = lm1_used + np.random.normal(0.0, noise_std, size=lm1_used.shape).astype(np.float32)
            lm2_used = lm2_used + np.random.normal(0.0, noise_std, size=lm2_used.shape).astype(np.float32)
        agent_obs = build_agent_obs(states, lm1_used, lm2_used, proto_exp)
        if train and warmup > 0 and episode <= warmup:
            actions = [random.choice([0, 1]) for _ in agent_obs]
            agent_values, agent_lps = agent_values_log_probs(agent_model, agent_obs, actions)
        else:
            actions, agent_values, agent_lps = agent_actions(agent_model, agent_obs, deterministic_agent)

        next_states, reward, done, truncated, info = env.step(actions)
        ss = getattr(env, "step_sleep_seconds", 0.0) or 0.0
        if ss > 0:
            time.sleep(float(ss))
        total_reward += reward
        terminal = bool(done or truncated)
        inner = env.inner()
        trace = build_step_trace_row(step, actions, reward, info, inner, proto_exp.exp)
        step_trace.append(trace)
        min_pair = trace.get("min_pairwise_dist_active_m")
        min_pair_val = safe_float(min_pair, default=float(proto_exp.cfg.get("target_min_pairwise_dist", 8.0)))
        target_dist = max(1e-6, float(proto_exp.cfg.get("target_min_pairwise_dist", 8.0)))
        dist_score = min(1.0, max(0.0, min_pair_val / target_dist))
        distance_reward = float(proto_exp.cfg.get("distance_reward_weight", 0.0)) * dist_score

        per_agent_rewards = info.get("agents_rewards") or [reward] * 6
        reward_scale = 1.0 / max(1.0, abs(float(proto_exp.cfg["collision_reward"])), abs(float(proto_exp.cfg["arrived_reward"])))
        gm_reward = float(reward) * reward_scale + distance_reward
        lm1_reward = min(float(x) for x in per_agent_rewards[:3]) * reward_scale + distance_reward
        lm2_reward = min(float(x) for x in per_agent_rewards[3:6]) * reward_scale + distance_reward
        if proto_exp.cfg["reward_mode"] == "global":
            lm1_reward = gm_reward
            lm2_reward = gm_reward

        if train:
            append_transition(master_transitions, lm1_input, lm1_emb, lm1_reward, lm1_value, lm1_lp, terminal, step=step, role="LM1")
            append_transition(master_transitions, lm2_input, lm2_emb, lm2_reward, lm2_value, lm2_lp, terminal, step=step, role="LM2")
            append_transition(master_transitions, gm_input, gm_emb, gm_reward, gm_value, gm_lp, terminal, step=step, role="GM")
            for i, obs in enumerate(agent_obs):
                agent_reward = gm_reward if proto_exp.cfg["reward_mode"] == "global" else (
                    lm1_reward if i < 3 else lm2_reward
                )
                append_transition(agent_transitions, obs, np.asarray([actions[i]]), agent_reward, agent_values[i], agent_lps[i], terminal)

        if trace_rows is not None:
            shared = {
                "config": proto_exp.cfg["label"],
                "episode": episode,
                "condition": condition,
                "step": step,
                "step_reward": reward,
                "actions": json_vec(actions),
                "min_pairwise_dist_active_m": trace.get("min_pairwise_dist_active_m"),
                "min_dist_active_to_uncontrolled_m": trace.get("min_dist_active_to_uncontrolled_m"),
                "n_active_controlled": trace.get("n_active_controlled"),
            }
            payload = {
                "LM1": (lm1_input, lm1_emb, lm1_stats, lm1_value, lm1_lp),
                "LM2": (lm2_input, lm2_emb, lm2_stats, lm2_value, lm2_lp),
                "GM": (gm_input, gm_emb, gm_stats, gm_value, gm_lp),
            }
            for role, (inp, emb, stats, value, lp) in payload.items():
                trace_rows.append({
                    **shared,
                    "role": role,
                    "input_norm": float(np.linalg.norm(inp)),
                    "sampled_embedding": json_vec(emb),
                    "policy_mean": json_vec(stats["mean"]),
                    "policy_std": json_vec(stats["std"]),
                    "value_pred": float(value.detach().cpu().reshape(-1)[0]),
                    "log_prob": float(lp.detach().cpu().reshape(-1)[0]),
                    "embedding_norm": float(np.linalg.norm(emb)),
                })

        global_emb_prev = gm_emb_for_next
        if condition == "zero_master":
            global_emb_prev = np.zeros_like(gm_emb)
        if condition == "negate_master":
            global_emb_prev = -gm_emb
        states = next_states

    crashed = any(any(r.get("crashed_flags", [])) for r in step_trace)
    first_crash_step = ""
    for row in step_trace:
        if any(row.get("crashed_flags", [])):
            first_crash_step = int(row["step"])
            break

    if trace_rows is not None:
        for row in trace_rows:
            if row["config"] != proto_exp.cfg["label"] or row["episode"] != episode or row["condition"] != condition:
                continue
            if first_crash_step != "":
                stc = int(first_crash_step) - int(row["step"])
                row["steps_to_crash"] = stc
                if stc == 0:
                    phase = "crash_step"
                elif 1 <= stc <= 3:
                    phase = "pre_crash_3"
                else:
                    phase = "crash_episode_other"
            else:
                row["steps_to_crash"] = ""
                min_pair = safe_float(row.get("min_pairwise_dist_active_m"), 999.0)
                phase = "near_miss_safe" if min_pair < 8.0 else "safe"
            row["phase"] = phase
            row["crashed_episode"] = int(crashed)
            row["first_crash_step"] = first_crash_step

    arrived = 0
    try:
        arrived = sum(1 for v in env.inner().controlled_vehicles if hasattr(v, "is_arrived") and v.is_arrived)
    except Exception:
        pass
    min_pair_vals = [
        safe_float(row.get("min_pairwise_dist_active_m"), default=float("nan"))
        for row in step_trace
        if row.get("min_pairwise_dist_active_m") not in ("", None)
    ]
    min_pair_vals = [v for v in min_pair_vals if not math.isnan(v)]
    close_thr = float(proto_exp.cfg.get("close_distance_threshold", 8.0))
    critical_thr = float(proto_exp.cfg.get("critical_distance_threshold", 4.0))
    summary = {
        "config": proto_exp.cfg["label"],
        "episode": episode,
        "condition": condition,
        **{k: _geom[k] for k in ("scenario_pool", "scenario_index", "scenario_base", "geometry_bucket")},
        "steps": step,
        "reward": total_reward,
        "arrival_pct": 100.0 * arrived / 6.0,
        "crashed": int(crashed),
        "first_crash_step": first_crash_step,
        "min_pairwise_dist_m": min(min_pair_vals) if min_pair_vals else "",
        "mean_pairwise_dist_m": float(np.mean(min_pair_vals)) if min_pair_vals else "",
        "p10_pairwise_dist_m": float(np.percentile(min_pair_vals, 10)) if min_pair_vals else "",
        "close_step_rate": float(np.mean([v < close_thr for v in min_pair_vals])) if min_pair_vals else "",
        "critical_step_rate": float(np.mean([v < critical_thr for v in min_pair_vals])) if min_pair_vals else "",
    }
    if train:
        label_master_risk_targets(master_transitions, step_trace, proto_exp.cfg)
    return summary, master_transitions, agent_transitions


def _resolve_long_path(path: str) -> str:
    """Absolute path; on Windows prepend \\\\?\\ for long paths so mkdir/open succeed under MAX_PATH."""
    p = os.path.normpath(os.path.abspath(os.path.expanduser(str(path).strip())))
    if os.name != "nt" or p.startswith("\\\\?\\"):
        return p
    if len(p) >= 200:
        return "\\\\?\\" + p
    return p


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve_long_path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_figure_save_png(fig: plt.Figure, full_path: str, *, dpi: int = 175) -> None:
    """Persist PNG via system TEMP then copy — avoids flaky writes under OneDrive / synced Downloads."""
    dst = Path(os.path.normpath(os.path.abspath(Path(str(full_path).strip()).expanduser())))
    parent = dst.parent

    fd, tmppath = tempfile.mkstemp(prefix="safefig_", suffix=".png")
    os.close(fd)
    tmppath = os.path.normpath(tmppath)
    try:
        fig.savefig(tmppath, dpi=dpi, format="png")
        last_exc: BaseException | None = None
        for attempt in range(24):
            parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(tmppath, os.fspath(dst))
                return
            except OSError as exc:
                last_exc = exc
                if getattr(exc, "errno", None) not in (None, 2, 13, 22):
                    raise
            time.sleep(0.04 * min(attempt + 1, 12))

        direct_last: BaseException | None = None
        for attempt in range(8):
            parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(os.fspath(dst), dpi=dpi, format="png")
                return
            except OSError as exc:
                direct_last = exc
                time.sleep(0.03 * min(attempt + 1, 10))

        raise RuntimeError(
            f"PNG save failed — copy from temp failed ({last_exc!r}); direct save ({direct_last!r}); "
            f"dst={os.fspath(dst)!r}. Move experiment_runs out of synced Downloads or pause sync."
        ) from (last_exc or direct_last)
    finally:
        try:
            if os.path.isfile(tmppath):
                os.unlink(tmppath)
        except OSError:
            pass


def plot_training(config_dir: str, episodes: list[dict[str, Any]]) -> None:
    xs = [int(r["episode"]) for r in episodes]
    arr = np.asarray([float(r["arrival_pct"]) for r in episodes], dtype=float)
    crashes = np.asarray([100.0 * int(r["crashed"]) for r in episodes], dtype=float)
    window = min(50, max(1, len(xs)))
    kernel = np.ones(window) / window
    arr_s = np.convolve(arr, kernel, mode="same")
    crash_s = np.convolve(crashes, kernel, mode="same")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, arr_s, label="arrival rolling")
    ax.plot(xs, crash_s, label="crash rolling")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Episode")
    ax.set_ylabel("%")
    ax.set_title("Training Arrival / Crash")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(config_dir, "training_curves.png"), dpi=160)
    plt.close(fig)


def gaussian_kl(mu0, sd0, mu1, sd1) -> float:
    sd0 = np.maximum(np.asarray(sd0, dtype=float), 1e-6)
    sd1 = np.maximum(np.asarray(sd1, dtype=float), 1e-6)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    return float(np.sum(np.log(sd1 / sd0) + (sd0**2 + (mu0 - mu1) ** 2) / (2.0 * sd1**2) - 0.5))


def phase_metrics(rows: list[dict[str, Any]], out_dir: str) -> list[dict[str, Any]]:
    out = []
    danger = {"pre_crash_3", "crash_step"}
    for role in ROLES:
        role_rows = [r for r in rows if r["role"] == role and r["condition"] == "normal"]
        safe = [np.asarray(json.loads(r["policy_mean"]), dtype=float) for r in role_rows if r["phase"] == "safe"]
        bad = [np.asarray(json.loads(r["policy_mean"]), dtype=float) for r in role_rows if r["phase"] in danger]
        if len(safe) >= 2 and len(bad) >= 2:
            s = np.vstack(safe)
            b = np.vstack(bad)
            kl = 0.5 * (
                gaussian_kl(s.mean(axis=0), s.std(axis=0), b.mean(axis=0), b.std(axis=0))
                + gaussian_kl(b.mean(axis=0), b.std(axis=0), s.mean(axis=0), s.std(axis=0))
            )
            centroid = float(np.linalg.norm(s.mean(axis=0) - b.mean(axis=0)))
        else:
            kl = None
            centroid = None
        auc = None
        try:
            from sklearn.metrics import roc_auc_score
            samples = []
            labels = []
            for r in role_rows:
                if r["phase"] in ("safe", "near_miss_safe", "pre_crash_3", "crash_step"):
                    samples.append(np.asarray(json.loads(r["policy_mean"]), dtype=float))
                    labels.append(1 if r["phase"] in danger else 0)
            if len(set(labels)) == 2:
                x = np.vstack(samples)
                safe_center = x[np.asarray(labels) == 0].mean(axis=0)
                danger_center = x[np.asarray(labels) == 1].mean(axis=0)
                direction = danger_center - safe_center
                scores = x @ direction
                auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = None
        out.append({
            "role": role,
            "safe_points": len(safe),
            "danger_points": len(bad),
            "safe_vs_danger_symmetric_kl": kl,
            "safe_vs_danger_centroid_distance": centroid,
            "danger_auc_linear_centroid": auc,
        })
    write_csv(os.path.join(out_dir, "phase_metrics.csv"), out)
    return out


def plot_pca(rows: list[dict[str, Any]], out_dir: str, role: str) -> None:
    role_rows = [r for r in rows if r["condition"] == "normal" and r["role"] == role]
    if len(role_rows) < 3:
        return
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return
    x = np.asarray([json.loads(r["policy_mean"]) for r in role_rows], dtype=float)
    xy = PCA(n_components=2).fit_transform(x)
    phases = [r["phase"] for r in role_rows]
    colors = {
        "safe": "#2E7D32",
        "near_miss_safe": "#90A4AE",
        "crash_episode_other": "#FFB300",
        "pre_crash_3": "#E53935",
        "crash_step": "#7B1FA2",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for phase, color in colors.items():
        mask = np.asarray([p == phase for p in phases])
        if np.any(mask):
            ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.55, c=color, label=phase)
    ax.set_title(f"{role} Policy Mean PCA by Crash Phase")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(out_dir, f"pca_policy_mean_by_phase_{role}.png"), dpi=180)
    plt.close(fig)


def plot_pca_lm_combined(rows: list[dict[str, Any]], out_dir: str) -> None:
    lm_rows = [r for r in rows if r["condition"] == "normal" and r["role"] in ("LM1", "LM2")]
    if len(lm_rows) < 3:
        return
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return
    x = np.asarray([json.loads(r["policy_mean"]) for r in lm_rows], dtype=float)
    xy = PCA(n_components=2).fit_transform(x)
    phases = [r["phase"] for r in lm_rows]
    roles = [r["role"] for r in lm_rows]
    colors = {
        "safe": "#2E7D32",
        "near_miss_safe": "#90A4AE",
        "crash_episode_other": "#FFB300",
        "pre_crash_3": "#E53935",
        "crash_step": "#7B1FA2",
    }
    markers = {"LM1": "o", "LM2": "^"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for phase, color in colors.items():
        for role, marker in markers.items():
            mask = np.asarray([(p == phase and r == role) for p, r in zip(phases, roles)])
            if np.any(mask):
                ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.55, c=color, marker=marker, label=f"{phase} {role}")
    ax.set_title("LM Policy Mean PCA by Crash Phase")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(out_dir, "pca_policy_mean_by_phase_LM1_LM2.png"), dpi=180)
    plt.close(fig)


def plot_proto_distance_timeline(rows: list[dict[str, Any]], out_dir: str) -> None:
    gm = [r for r in rows if r["condition"] == "normal" and r["role"] == "GM"]
    danger = [np.asarray(json.loads(r["policy_mean"]), dtype=float) for r in gm if r["phase"] in ("pre_crash_3", "crash_step")]
    if len(danger) < 2:
        return
    center = np.vstack(danger).mean(axis=0)
    buckets = defaultdict(list)
    for r in gm:
        if r.get("steps_to_crash") in ("", None):
            continue
        stc = int(r["steps_to_crash"])
        if 0 <= stc <= 12:
            emb = np.asarray(json.loads(r["policy_mean"]), dtype=float)
            buckets[stc].append(float(np.linalg.norm(emb - center)))
    if not buckets:
        return
    xs = sorted(buckets)
    ys = [float(np.mean(buckets[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.invert_xaxis()
    ax.set_xlabel("Steps to crash")
    ax.set_ylabel("Distance to crash prototype")
    ax.set_title("GM Approaches Crash Prototype Before Collision")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(out_dir, "steps_to_crash_proto_distance.png"), dpi=180)
    plt.close(fig)


def summarize_counterfactual(episodes: list[dict[str, Any]], out_dir: str, conditions: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["condition"]].append(row)
    out = []
    for condition in tuple(conditions or CONDITIONS):
        rows = grouped.get(condition, [])
        if not rows:
            continue
        out.append({
            "condition": condition,
            "n": len(rows),
            "arrival_mean": float(np.mean([float(r["arrival_pct"]) for r in rows])),
            "crash_rate": 100.0 * float(np.mean([int(r["crashed"]) for r in rows])),
            "reward_mean": float(np.mean([float(r["reward"]) for r in rows])),
            "min_pairwise_dist_mean": float(np.mean([safe_float(r.get("min_pairwise_dist_m")) for r in rows if r.get("min_pairwise_dist_m") not in ("", None)])) if any(r.get("min_pairwise_dist_m") not in ("", None) for r in rows) else "",
            "p10_pairwise_dist_mean": float(np.mean([safe_float(r.get("p10_pairwise_dist_m")) for r in rows if r.get("p10_pairwise_dist_m") not in ("", None)])) if any(r.get("p10_pairwise_dist_m") not in ("", None) for r in rows) else "",
            "close_step_rate_mean": float(np.mean([safe_float(r.get("close_step_rate")) for r in rows if r.get("close_step_rate") not in ("", None)])) if any(r.get("close_step_rate") not in ("", None) for r in rows) else "",
            "critical_step_rate_mean": float(np.mean([safe_float(r.get("critical_step_rate")) for r in rows if r.get("critical_step_rate") not in ("", None)])) if any(r.get("critical_step_rate") not in ("", None) for r in rows) else "",
        })
    write_csv(os.path.join(out_dir, "counterfactual_summary.csv"), out)
    return out


def plot_counterfactual(summary: list[dict[str, Any]], out_dir: str) -> None:
    if not summary:
        return
    labels = [r["condition"] for r in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(len(labels)), [r["arrival_mean"] for r in summary])
    axes[0].set_title("Arrival")
    axes[1].bar(range(len(labels)), [r["crash_rate"] for r in summary])
    axes[1].set_title("Crash Rate")
    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(out_dir, "counterfactual_performance.png"), dpi=180)
    plt.close(fig)

    if any(row.get("min_pairwise_dist_mean") not in ("", None) for row in summary):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].bar(range(len(labels)), [safe_float(r.get("min_pairwise_dist_mean")) for r in summary])
        axes[0].set_title("Mean Min Pairwise Distance")
        axes[1].bar(range(len(labels)), [safe_float(r.get("p10_pairwise_dist_mean")) for r in summary])
        axes[1].set_title("P10 Pairwise Distance")
        axes[2].bar(range(len(labels)), [safe_float(r.get("close_step_rate_mean")) for r in summary])
        axes[2].set_title("Close-Step Rate")
        for ax in axes:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        safe_figure_save_png(fig, os.path.join(out_dir, "counterfactual_distance.png"), dpi=180)
        plt.close(fig)


def aggregate_geometry_batches(rows: list[dict[str, Any]], *, has_condition: bool) -> list[dict[str, Any]]:
    """Means per geometry_bucket (+ condition for test stratification)."""
    keydims: list[str] = ["geometry_bucket"]
    if has_condition:
        keydims.insert(0, "condition")
    agg: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if str(r.get("geometry_bucket") or "").strip() == "":
            continue
        key = tuple(r.get(k) for k in keydims)
        if key not in agg:
            agg[key] = {k: r.get(k) for k in keydims}
            agg[key]["n_episodes"] = 0
            agg[key]["arrival_pct_sum"] = 0.0
            agg[key]["crashed_sum"] = 0.0
            agg[key]["reward_sum"] = 0.0
        b = agg[key]
        b["n_episodes"] += 1
        b["arrival_pct_sum"] += float(r.get("arrival_pct", 0.0))
        b["crashed_sum"] += int(r.get("crashed", 0))
        b["reward_sum"] += float(r.get("reward", 0.0))
    out = []
    for b in agg.values():
        n = max(1, int(b["n_episodes"]))
        out.append({
            **{k: b[k] for k in keydims},
            "n_episodes": n,
            "arrival_mean": b["arrival_pct_sum"] / n,
            "crash_rate_pct": 100.0 * b["crashed_sum"] / n,
            "reward_mean": b["reward_sum"] / n,
        })
    out.sort(key=lambda x: (x.get("condition", ""), x.get("geometry_bucket", "")))
    return out


def plot_test_geometry_breakdown(rows: list[dict[str, Any]], out_dir: str, *, focus_conditions: list[str] | None = None) -> None:
    if not rows:
        return
    out_base = os.path.normpath(os.path.abspath(os.path.expanduser(str(out_dir))))
    conds = focus_conditions or ["normal", "zero_master", "disconnect_global_master", "large_const_global_master", "zero_global_master"]
    buckets = sorted({str(r.get("geometry_bucket")) for r in rows if r.get("geometry_bucket")})
    if not buckets:
        return
    xs = np.arange(len(buckets))
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(max(10, len(buckets) * 1.3), 5.5))
    for i, cnd in enumerate(conds):
        series = []
        for b in buckets:
            matching = [
                r for r in rows
                if r.get("condition") == cnd and str(r.get("geometry_bucket")) == b
            ]
            series.append(float(np.mean([float(x["arrival_pct"]) for x in matching])) if matching else float("nan"))
        ax.bar(xs + i * width, series, width=width * 0.95, label=cnd[:18], alpha=0.85)
    ax.set_xticks(xs + width * (len(conds) - 1) / 2.0)
    ax.set_xticklabels(buckets, rotation=22, ha="right")
    ax.set_ylabel("Mean arrival %")
    ax.set_title("Test arrivals by manoeuvre/stratification bucket (episode mean)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(out_base, "test_geometry_arrival_breakdown.png"), dpi=175)
    plt.close(fig)


def plot_train_geometry_breakdown(rows: list[dict[str, Any]], out_dir: str) -> None:
    if not rows:
        return
    out_base = os.path.normpath(os.path.abspath(os.path.expanduser(str(out_dir))))
    buckets = sorted({str(r.get("geometry_bucket")) for r in rows if r.get("geometry_bucket")})
    if not buckets:
        return
    arr_means = []
    crash_rates = []
    for b in buckets:
        matching = [r for r in rows if str(r.get("geometry_bucket")) == b]
        arr_means.append(float(np.mean([float(x["arrival_pct"]) for x in matching])) if matching else 0.0)
        crash_rates.append(100.0 * float(np.mean([int(x["crashed"]) for x in matching])) if matching else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    x = np.arange(len(buckets))
    axes[0].bar(x, arr_means, color="#1565C0")
    axes[0].set_title("Train mean arrival % by geometry bucket")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets, rotation=22, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, crash_rates, color="#C62828")
    axes[1].set_title("Train crash-rate % by geometry bucket")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buckets, rotation=22, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(out_base, "train_geometry_breakdown.png"), dpi=175)
    plt.close(fig)


def run_config(
    cfg: dict,
    root: str,
    *,
    episodes_override: int | None = None,
    test_override: int | None = None,
    init_checkpoint_paths: tuple[str, str] | None = None,
) -> dict[str, Any]:
    label = cfg["label"]
    root_abs = os.path.normpath(os.path.abspath(os.path.expanduser(root)))
    config_dir = os.path.normpath(os.path.join(root_abs, str(label).strip()))
    os.makedirs(config_dir, exist_ok=True)
    cfg = dict(cfg)
    if episodes_override is not None:
        cfg["episodes"] = int(episodes_override)
    if test_override is not None:
        cfg["test_episodes"] = int(test_override)
    write_json(os.path.join(config_dir, "config.json"), cfg)

    set_all_seeds(SEED)
    proto_exp = ProtoExperiment(cfg, config_dir)
    master_model, agent_model = make_models(proto_exp)
    aux_head = RiskAuxHead(proto_exp.embedding_dim)
    pretrained_loaded = False
    if init_checkpoint_paths is not None:
        ap_raw, mp_raw = init_checkpoint_paths
        ap_init = os.path.normpath(os.path.abspath(os.path.expanduser(str(ap_raw).strip())))
        mp_init = os.path.normpath(os.path.abspath(os.path.expanduser(str(mp_raw).strip())))
        if load_models_from_paths(agent_model, master_model, ap_init, mp_init):
            pretrained_loaded = True
            print(f"[{label}] Fine-tune init load OK:\n  agent={ap_init}\n  master={mp_init}")
        else:
            print(f"[{label}] Fine-tune init load FAILED → falling back to maybe_load_pretrained", file=sys.stderr)
            agent_model, pretrained_loaded = maybe_load_pretrained(cfg, master_model, agent_model)
    else:
        agent_model, pretrained_loaded = maybe_load_pretrained(cfg, master_model, agent_model)
    train_env = ProtoHighwayWrapper(proto_exp, conflict_ratio=0.0, conflict_only=False)

    train_rows = []
    best_selection_score = float("-inf")
    best_ckpt_metric_mode = str(cfg.get("best_ckpt_metric") or "arrival")
    losses = []
    try:
        for ep in range(1, int(cfg["episodes"]) + 1):
            train_env.set_conflict_ratio(schedule_value(cfg["conflict_schedule"], ep))
            summary, master_transitions, agent_transitions = run_episode(
                proto_exp=proto_exp,
                env=train_env,
                master_model=master_model,
                agent_model=agent_model,
                episode=ep,
                train=True,
            )
            train_rows.append(summary)
            master_loss = ppo_update(
                master_model.model.policy,
                master_transitions,
                discrete=False,
                cfg=cfg,
                aux_head=aux_head,
                episode=ep,
            )
            agent_loss = ppo_update(agent_model.policy, agent_transitions, discrete=True, cfg=cfg, episode=ep)
            losses.append({"episode": ep, **{f"master_{k}": v for k, v in master_loss.items()}, **{f"agent_{k}": v for k, v in agent_loss.items()}})
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
                    save_models(agent_model, master_model, os.path.join(config_dir, "best", "ckpt"))
            if ep % 25 == 0 or ep == 1:
                print(f"[{label}] ep={ep}/{cfg['episodes']} arrival={summary['arrival_pct']:.1f} crashed={summary['crashed']}")
    finally:
        train_env.close()

    save_models(agent_model, master_model, os.path.join(config_dir, "trained_model"))
    write_csv(os.path.join(config_dir, "episode_metrics.csv"), train_rows)
    write_csv(os.path.join(config_dir, "losses.csv"), losses)
    plot_training(config_dir, train_rows)

    # Evaluation uses in-memory weights. Training may collapse at the end while an
    # earlier rolling-50 peak was saved under best/. Load that for test by default.
    best_ckpt_dir = os.path.join(config_dir, "best", "ckpt")
    eval_checkpoint = "trained_model_final"
    if os.path.isfile(f"{best_ckpt_dir}_agent.pth") and os.path.isfile(f"{best_ckpt_dir}_master.pth"):
        if load_models(agent_model, master_model, best_ckpt_dir):
            eval_checkpoint = "best_rolling50"
            score_frag = ""
            if best_selection_score != float("-inf"):
                score_frag = f" (peak {best_ckpt_metric_mode} score≈{float(best_selection_score):.4f})"
            print(f"[{label}] Loaded best rolling-window checkpoint for evaluation: {best_ckpt_dir}{score_frag}")

    trace_rows = []
    eval_rows = []
    test_env = ProtoHighwayWrapper(
        proto_exp,
        conflict_ratio=float(cfg.get("test_conflict_ratio", 0.5)),
        conflict_only=False,
    )
    try:
        eval_conditions = tuple(cfg.get("eval_conditions") or CONDITIONS)
        for condition in eval_conditions:
            for ep in range(1, int(cfg["test_episodes"]) + 1):
                summary, _, _ = run_episode(
                    proto_exp=proto_exp,
                    env=test_env,
                    master_model=master_model,
                    agent_model=agent_model,
                    episode=ep,
                    train=False,
                    condition=condition,
                    trace_rows=trace_rows,
                    replay_seed=SEED + ep,
                )
                eval_rows.append(summary)
    finally:
        test_env.close()

    write_csv(os.path.join(config_dir, "test_episode_metrics.csv"), eval_rows)
    write_csv(os.path.join(config_dir, "master_step_trace.csv"), trace_rows)
    tg_train_rows = aggregate_geometry_batches(train_rows, has_condition=False)
    tg_test_rows = aggregate_geometry_batches(eval_rows, has_condition=True)
    write_csv(os.path.join(config_dir, "train_geometry_agg.csv"), tg_train_rows)
    write_csv(os.path.join(config_dir, "test_geometry_agg.csv"), tg_test_rows)
    plot_train_geometry_breakdown(train_rows, config_dir)
    _conds_plot = list(tuple(cfg.get("eval_conditions") or CONDITIONS))
    plot_test_geometry_breakdown(eval_rows, config_dir, focus_conditions=_conds_plot[:12])
    metrics = phase_metrics(trace_rows, config_dir)
    for role in ROLES:
        plot_pca(trace_rows, config_dir, role)
    plot_pca_lm_combined(trace_rows, config_dir)
    plot_proto_distance_timeline(trace_rows, config_dir)
    cf_summary = summarize_counterfactual(eval_rows, config_dir, tuple(cfg.get("eval_conditions") or CONDITIONS))
    plot_counterfactual(cf_summary, config_dir)

    normal_eval = [r for r in eval_rows if r["condition"] == "normal"]
    zero_master_eval = [r for r in eval_rows if r["condition"] == "zero_master"]
    train_arr_all = [float(r["arrival_pct"]) for r in train_rows] if train_rows else []
    rolling50_at_end = (
        float(np.mean(train_arr_all[-50:])) if len(train_arr_all) >= 50 else (float(np.mean(train_arr_all)) if train_arr_all else 0.0)
    )
    best_roll = -1.0
    win = min(50, len(train_arr_all))
    if win > 0:
        for start in range(0, len(train_arr_all) - win + 1):
            best_roll = max(best_roll, float(np.mean(train_arr_all[start : start + win])))
    result = {
        "label": label,
        "pretrained_loaded": pretrained_loaded,
        "eval_checkpoint": eval_checkpoint,
        "best_ckpt_metric": best_ckpt_metric_mode,
        "best_ckpt_peak_selection_score": (
            float(best_selection_score)
            if best_selection_score != float("-inf") and math.isfinite(best_selection_score)
            else None
        ),
        "train_arrival_best_rolling50": float(best_roll),
        "train_arrival_last50": rolling50_at_end,
        "train_crashes": int(sum(int(r["crashed"]) for r in train_rows)),
        "test_arrival_normal": float(np.mean([float(r["arrival_pct"]) for r in normal_eval])) if normal_eval else 0.0,
        "test_crash_rate_normal": 100.0 * float(np.mean([int(r["crashed"]) for r in normal_eval])) if normal_eval else 0.0,
        "test_arrival_zero_master": (
            float(np.mean([float(r["arrival_pct"]) for r in zero_master_eval])) if zero_master_eval else None
        ),
        "test_min_pairwise_normal": float(np.mean([safe_float(r.get("min_pairwise_dist_m")) for r in normal_eval if r.get("min_pairwise_dist_m") not in ("", None)])) if any(r.get("min_pairwise_dist_m") not in ("", None) for r in normal_eval) else None,
        "test_close_step_rate_normal": float(np.mean([safe_float(r.get("close_step_rate")) for r in normal_eval if r.get("close_step_rate") not in ("", None)])) if any(r.get("close_step_rate") not in ("", None) for r in normal_eval) else None,
        "phase_metrics": metrics,
        "counterfactual": cf_summary,
    }
    write_json(os.path.join(config_dir, "summary.json"), result)
    return result


def plot_sweep_summary(results: list[dict[str, Any]], root: str) -> None:
    labels = [r["label"] for r in results]
    gm_kl = []
    gm_auc = []
    for r in results:
        gm = next((m for m in r["phase_metrics"] if m["role"] == "GM"), {})
        gm_kl.append(float(gm.get("safe_vs_danger_symmetric_kl") or 0.0))
        gm_auc.append(float(gm.get("danger_auc_linear_centroid") or 0.0))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(range(len(labels)), [r["test_arrival_normal"] for r in results])
    axes[0].set_title("Normal Test Arrival")
    axes[1].bar(range(len(labels)), gm_kl)
    axes[1].set_title("GM Safe vs Danger KL")
    axes[2].bar(range(len(labels)), gm_auc)
    axes[2].set_title("GM Danger AUROC")
    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(root, "proto_sweep_summary.png"), dpi=180)
    plt.close(fig)

    for key, title, filename in [
        ("gm_kl", "GM Safe vs Pre-Crash KL", "kl_safe_vs_precrash_by_config.png"),
        ("gm_auc", "GM Danger AUROC", "danger_auc_by_config.png"),
    ]:
        values = gm_kl if key == "gm_kl" else gm_auc
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(labels)), values)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        safe_figure_save_png(fig, os.path.join(root, filename), dpi=180)
        plt.close(fig)

    drops = []
    for r in results:
        cf = {row["condition"]: row for row in r.get("counterfactual", [])}
        normal = cf.get("normal", {}).get("arrival_mean", 0.0)
        worst_ablated = min(
            [cf.get(c, {}).get("arrival_mean", normal) for c in ("zero_master", "swap_local_masters", "negate_master")],
            default=normal,
        )
        drops.append(float(normal - worst_ablated))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(labels)), drops)
    ax.set_title("Counterfactual Arrival Drop")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    safe_figure_save_png(fig, os.path.join(root, "counterfactual_drop_by_config.png"), dpi=180)
    plt.close(fig)

    rank_rows = []
    for r, kl, auc in zip(results, gm_kl, gm_auc):
        score = kl + 5.0 * max(0.0, auc - 0.5) + 0.01 * r["test_arrival_normal"] - 0.01 * r["test_crash_rate_normal"]
        rank_rows.append({
            "label": r["label"],
            "proto_purity_score": score,
            "gm_kl": kl,
            "gm_auc": auc,
            "test_arrival_normal": r["test_arrival_normal"],
            "test_crash_rate_normal": r["test_crash_rate_normal"],
            "train_arrival_last50": r["train_arrival_last50"],
        })
    rank_rows.sort(key=lambda x: x["proto_purity_score"], reverse=True)
    write_csv(os.path.join(root, "proto_config_ranking.csv"), rank_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--only-config", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 3
        args.test_episodes = 2
        if not args.only_config:
            args.only_config = "C00_legacy_baseline"

    ts = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    root = args.output_root or os.path.join("experiment_runs", f"proto_sweep_{ts}")
    os.makedirs(root, exist_ok=True)

    configs = [dict(c) for c in SWEEP_CONFIGS]
    if args.only_config:
        configs = [c for c in configs if c["label"] == args.only_config]
        if not configs:
            raise ValueError(f"Unknown config: {args.only_config}")

    write_json(os.path.join(root, "sweep_config.json"), {"seed": SEED, "configs": configs})
    results = []
    for cfg in configs:
        results.append(run_config(cfg, root, episodes_override=args.episodes, test_override=args.test_episodes))
        write_json(os.path.join(root, "summary_partial.json"), results)
    write_json(os.path.join(root, "summary.json"), results)
    plot_sweep_summary(results, root)
    print(f"Proto-action sweep complete: {root}")


if __name__ == "__main__":
    main()
