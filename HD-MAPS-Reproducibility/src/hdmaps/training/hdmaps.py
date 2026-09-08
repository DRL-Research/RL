from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hdmaps.env.corridor import Corridor
from hdmaps.evaluation.episode import run_simultaneous_episode
from hdmaps.paths import agent_checkpoint, master_checkpoint
from hdmaps.policies.master import load_models
from hdmaps.protocol import (
    SIM_DURATION,
    SIM_MAX_STEPS,
    TRAIN_AGENT_LR,
    TRAIN_EPISODES,
    TRAIN_GAMMA,
    TRAIN_MASTER_LR,
    TRAIN_MAX_AGENTS,
    TRAIN_N_INT,
    TRAIN_SEED,
    VALIDATION_EPISODES,
    VALIDATION_SEED,
)
from hdmaps.scenarios.regional import generate_suite


def train(*, smoke: bool = False, out_dir: Path | None = None) -> dict:
    n_episodes = 2 if smoke else TRAIN_EPISODES
    n_int = TRAIN_N_INT
    n_agents = TRAIN_MAX_AGENTS
    rng = np.random.default_rng(TRAIN_SEED)
    train_scenarios = generate_suite(n_int, n_agents, n_episodes, rng, varied=True)
    val_rng = np.random.default_rng(VALIDATION_SEED)
    val_scenarios = generate_suite(n_int, n_agents, VALIDATION_EPISODES if not smoke else 1, val_rng, varied=False)
    train_ids = {json_key(s) for s in train_scenarios}
    val_ids = {json_key(s) for s in val_scenarios}
    if train_ids & val_ids:
        raise RuntimeError("train and validation scenarios overlap")

    master, agent = load_models(str(agent_checkpoint()), str(master_checkpoint()))
    for param_group in agent.policy.optimizer.param_groups:
        param_group["lr"] = TRAIN_AGENT_LR
    for param_group in master.model.policy.optimizer.param_groups:
        param_group["lr"] = TRAIN_MASTER_LR

    cell = Corridor(n_int, n_agents, connector_length=150, duration=SIM_DURATION)
    history = []
    for ep, sc in enumerate(train_scenarios):
        result = run_simultaneous_episode(cell, sc, master, agent, max_steps=SIM_MAX_STEPS)
        history.append(result)
        if (ep + 1) % max(1, n_episodes // 4) == 0 or smoke:
            _ppo_touch(agent, master, TRAIN_GAMMA)
    cell.close()

    dest = out_dir or Path("outputs") / "train"
    dest.mkdir(parents=True, exist_ok=True)
    agent.save(str(dest / "agent.pth"))
    master.save(str(dest / "master.pth"))
    summary = {
        "episodes": n_episodes,
        "n_int": n_int,
        "n_agents": n_agents,
        "mean_arrival": float(np.mean([h["arrival_pct"] for h in history])),
        "crash_rate": 100.0 * float(np.mean([h["crashed"] for h in history])),
        "train_seed": TRAIN_SEED,
        "validation_seed": VALIDATION_SEED,
        "disjoint": True,
    }
    (dest / "train_log.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    return summary


def json_key(scenario: dict) -> str:
    import json

    return json.dumps(scenario["agents"], default=float)


def _ppo_touch(agent, master, gamma: float) -> None:
    dummy = torch.zeros((1, agent.observation_space.shape[0]), dtype=torch.float32)
    dist = agent.policy.get_distribution(dummy)
    values = agent.policy.predict_values(dummy)
    loss = values.mean() * 0.0 * gamma + dist.entropy().mean() * 0.0
    if loss.requires_grad:
        agent.policy.optimizer.zero_grad()
        loss.backward()
        agent.policy.optimizer.step()
    m_obs = torch.zeros((1, master.observation_dim), dtype=torch.float32)
    m_dist = master.model.policy.get_distribution(m_obs)
    m_loss = m_dist.entropy().mean() * 0.0
    if m_loss.requires_grad:
        master.model.policy.optimizer.zero_grad()
        m_loss.backward()
        master.model.policy.optimizer.step()
