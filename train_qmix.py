"""
QMIXwD Training - 5 Minute Quick Test
"""

import sys
import os
import json
import numpy as np
import argparse
import random

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from highwayenv.utils import patch_intersection_env, register_intersection_env
patch_intersection_env()
register_intersection_env()

from src.qmix.qmix_agent import QMIXAgent, QMIXwDAgent
from src.qmix.qmix_env_wrapper import create_qmix_env
from src.qmix.qmix_buffer import EpisodeData

try:
    from scenarios_config_qmix import full_env_config_exp5
except ImportError:
    from src.experiment.scenarios_config import full_env_config_exp5


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.success_flags = []
        self.travel_times = []
        self.collisions = []
        self.training_losses = []
        self.test_success_rates = []
        self.test_rewards = []
        self.test_travel_times = []
        self.test_total_steps = []
        self.total_steps = 0
        self.demo_steps = 0

    def add_episode(self, reward, success, travel_time, crashed, steps_taken):
        self.episode_rewards.append(float(reward))
        self.success_flags.append(1.0 if success else 0.0)
        self.travel_times.append(float(travel_time) if success else 0.0)
        self.collisions.append(1 if crashed else 0)
        self.total_steps += int(steps_taken)

    def add_test_point(self, success_rate, avg_reward, avg_travel_time):
        self.test_success_rates.append(float(success_rate))
        self.test_rewards.append(float(avg_reward))
        self.test_travel_times.append(float(avg_travel_time))
        self.test_total_steps.append(int(self.total_steps))

    def add_demo_steps(self, n_transitions):
        n = int(n_transitions)
        self.demo_steps += n
        self.total_steps += n

    def recent_success_rate(self, window=100):
        if not self.success_flags:
            return 0.0
        w = min(window, len(self.success_flags))
        return float(np.mean(self.success_flags[-w:]))

    def save_json(self, path):
        data = {
            "episode_rewards": self.episode_rewards,
            "success_rates": self.success_flags,
            "travel_times": self.travel_times,
            "collisions": self.collisions,
            "training_losses": self.training_losses,
            "test_success_rates": self.test_success_rates,
            "test_rewards": self.test_rewards,
            "test_travel_times": self.test_travel_times,
            "test_total_steps": self.test_total_steps,
            "total_steps": self.total_steps,
            "demo_steps": self.demo_steps,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def run_episode(env, agent, epsilon, max_steps=100, render=False):
    obs, state, info = env.reset()
    ep = EpisodeData()
    total_reward = 0.0
    crashed = False
    steps = 0
    done = False
    truncated = False

    while not done and not truncated and steps < max_steps:
        actions = agent.select_actions(obs, epsilon=epsilon)
        next_obs, next_state, reward, done, truncated, info = env.step(actions)
        ep.add(obs, state, actions, reward, bool(done or truncated))
        total_reward += float(reward)
        steps += 1
        obs, state = next_obs, next_state
        if done and info.get("crashed", False):
            crashed = True

    all_exited = bool(info.get("all_exited", False) or info.get("success", False))
    success = (all_exited and not crashed)
    travel_time = steps if success else 0
    return ep, total_reward, success, travel_time, crashed, steps


def test_agent(env, agent, n_episodes=20, render=False):
    succ, rew, tts = [], [], []
    for _ in range(n_episodes):
        _, r, s, tt, _, _ = run_episode(env, agent, epsilon=0.0, max_steps=100, render=render)
        succ.append(s)
        rew.append(r)
        if s:
            tts.append(tt)
    sr = float(np.mean(succ)) if succ else 0.0
    ar = float(np.mean(rew)) if rew else 0.0
    att = float(np.mean(tts)) if tts else 0.0
    return sr, ar, att


def generate_expert_demos(env, agent, n_episodes=100):
    demos = []
    kept = 0
    attempts = 0
    max_attempts = n_episodes * 10

    while kept < n_episodes and attempts < max_attempts:
        ep, _, success, _, _, _ = run_episode(env, agent, epsilon=0.0, max_steps=100, render=False)
        attempts += 1
        if success and len(ep) > 0:
            demos.append(ep)
            kept += 1

    return demos


def count_transitions(episodes):
    return int(sum(len(e) for e in episodes)) if episodes else 0


def train_expert_qmix(env, config, seed):
    agent = QMIXAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=0.1,
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
    )

    for ep_idx in range(config["expert_train_episodes"]):
        ep_data, reward, success, _, _, steps = run_episode(
            env, agent, epsilon=0.1, max_steps=100, render=False
        )
        agent.replay_buffer.add_episode(ep_data)
        agent.episode_count += 1

        if agent.replay_buffer.can_sample(agent.batch_size):
            agent.train()

    return agent


def train_qmixwd(env, config, save_dir, seed):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Quick test - Seed: {seed}")

    expert_agent = train_expert_qmix(env, config, seed)

    expert_eps = generate_expert_demos(env, expert_agent, n_episodes=config["n_expert_demos"])
    expert_steps = count_transitions(expert_eps)

    agent = QMIXwDAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=0.1,
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        expert_ratio=config["expert_ratio"],
    )

    inter_eps = []
    for i in range(config["n_interaction_demos"]):
        ep, _, _, _, _, _ = run_episode(env, agent, epsilon=0.2, max_steps=100, render=False)
        if len(ep) > 0:
            inter_eps.append(ep)

    inter_steps = count_transitions(inter_eps)

    for ep in expert_eps:
        agent.demo_buffer.add_expert_episode(ep)
    for ep in inter_eps:
        agent.demo_buffer.add_interaction_episode(ep)

    metrics = MetricsTracker()
    metrics.add_demo_steps(expert_steps + inter_steps)

    sr_d, ar_d, att_d = test_agent(env, agent, n_episodes=config["n_test_episodes"], render=False)
    metrics.add_test_point(sr_d, ar_d, att_d)
    metrics.save_json(f"{save_dir}/metrics.json")

    agent.pretrain(config["n_pretrain_steps"])

    sr_p, ar_p, att_p = test_agent(env, agent, n_episodes=config["n_test_episodes"], render=False)
    metrics.add_test_point(sr_p, ar_p, att_p)
    metrics.save_json(f"{save_dir}/metrics.json")

    epsilon = 0.1
    next_test_at = metrics.total_steps + config["test_interval_steps"]

    for ep_idx in range(config["n_online_episodes"]):
        ep_data, reward, success, travel_time, crashed, steps = run_episode(
            env, agent, epsilon=epsilon, max_steps=100, render=False
        )
        agent.replay_buffer.add_episode(ep_data)
        agent.episode_count += 1
        metrics.add_episode(reward, success, travel_time, crashed, steps)

        if agent.replay_buffer.can_sample(agent.batch_size):
            loss = agent.train()
            if loss is not None:
                metrics.training_losses.append(float(loss))

        while metrics.total_steps >= next_test_at:
            sr, ar, att = test_agent(env, agent, n_episodes=config["n_test_episodes"], render=False)
            metrics.add_test_point(sr, ar, att)
            metrics.save_json(f"{save_dir}/metrics.json")
            next_test_at += config["test_interval_steps"]

    agent.save(f"{save_dir}/final_model.pth")
    metrics.save_json(f"{save_dir}/metrics.json")
    print(f"Done. Steps: {metrics.total_steps}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_suffix', type=str, default='_quick')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except:
        pass

    env = create_qmix_env(full_env_config_exp5, render_mode=None)

    config = {
        "n_expert_demos": 20,
        "n_interaction_demos": 50,
        "n_pretrain_steps": 100,
        "n_online_episodes": 500,
        "test_interval_steps": 5000,
        "n_test_episodes": 10,
        "expert_train_episodes": 200,
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 5000,
        "expert_ratio": 0.1,
    }

    save_dir = f"qmixwd_results{args.save_suffix}"
    train_qmixwd(env, config, save_dir, args.seed)
    env.close()


if __name__ == "__main__":
    main()