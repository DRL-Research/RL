"""
Training Script for QMIX and QMIXwD with Full CSV Logging
- Logs EVERY episode with all details to CSV
- Logs test evaluations to separate CSV
- Multiple comprehensive plots
"""

import sys
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

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


class CSVLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Episode-level CSV
        self.episode_csv_path = f"{save_dir}/episodes.csv"
        self.episode_csv = open(self.episode_csv_path, 'w', newline='')
        self.episode_writer = csv.writer(self.episode_csv)
        self.episode_writer.writerow([
            'episode_number',
            'total_steps',
            'episode_steps',
            'episode_reward',
            'success',
            'crashed',
            'travel_time',
            'timestamp'
        ])

        # Test evaluation CSV
        self.test_csv_path = f"{save_dir}/test_evaluations.csv"
        self.test_csv = open(self.test_csv_path, 'w', newline='')
        self.test_writer = csv.writer(self.test_csv)
        self.test_writer.writerow([
            'evaluation_number',
            'total_steps',
            'success_rate',
            'collision_rate',
            'avg_reward',
            'avg_travel_time',
            'timestamp'
        ])

        # Training loss CSV
        self.loss_csv_path = f"{save_dir}/training_losses.csv"
        self.loss_csv = open(self.loss_csv_path, 'w', newline='')
        self.loss_writer = csv.writer(self.loss_csv)
        self.loss_writer.writerow([
            'step_number',
            'loss_value',
            'timestamp'
        ])

        self.episode_count = 0
        self.eval_count = 0
        self.loss_count = 0

    def log_episode(self, total_steps, episode_steps, reward, success, crashed, travel_time):
        self.episode_count += 1
        self.episode_writer.writerow([
            self.episode_count,
            total_steps,
            episode_steps,
            reward,
            1 if success else 0,
            1 if crashed else 0,
            travel_time,
            datetime.now().isoformat()
        ])
        self.episode_csv.flush()

    def log_test(self, total_steps, success_rate, collision_rate, avg_reward, avg_travel_time):
        self.eval_count += 1
        self.test_writer.writerow([
            self.eval_count,
            total_steps,
            success_rate,
            collision_rate,
            avg_reward,
            avg_travel_time,
            datetime.now().isoformat()
        ])
        self.test_csv.flush()

    def log_loss(self, loss_value):
        self.loss_count += 1
        self.loss_writer.writerow([
            self.loss_count,
            loss_value,
            datetime.now().isoformat()
        ])
        self.loss_csv.flush()

    def close(self):
        self.episode_csv.close()
        self.test_csv.close()
        self.loss_csv.close()


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.success_flags = []
        self.travel_times = []
        self.collisions = []
        self.training_losses = []
        self.episode_lengths = []

        self.test_success_rates = []
        self.test_rewards = []
        self.test_travel_times = []
        self.test_collision_rates = []
        self.test_total_steps = []

        self.running_avg_loss = []
        self.running_avg_reward = []
        self.running_collision_rate = []

        self.total_steps = 0
        self.demo_steps = 0
        self.episodes_completed = 0

    def add_episode(self, reward, success, travel_time, crashed, steps_taken):
        self.episode_rewards.append(float(reward))
        self.success_flags.append(1.0 if success else 0.0)
        self.travel_times.append(float(travel_time) if success else 0.0)
        self.collisions.append(1 if crashed else 0)
        self.episode_lengths.append(int(steps_taken))
        self.total_steps += int(steps_taken)
        self.episodes_completed += 1

    def add_test_point(self, success_rate, avg_reward, avg_travel_time, collision_rate):
        self.test_success_rates.append(float(success_rate))
        self.test_rewards.append(float(avg_reward))
        self.test_travel_times.append(float(avg_travel_time))
        self.test_collision_rates.append(float(collision_rate))
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

    def recent_collision_rate(self, window=100):
        if not self.collisions:
            return 0.0
        w = min(window, len(self.collisions))
        return float(np.mean(self.collisions[-w:]))

    def recent_avg_length(self, window=100):
        if not self.episode_lengths:
            return 0.0
        w = min(window, len(self.episode_lengths))
        return float(np.mean(self.episode_lengths[-w:]))

    def update_running_metrics(self, window=100):
        if len(self.training_losses) > 0:
            w = min(window, len(self.training_losses))
            self.running_avg_loss.append(float(np.mean(self.training_losses[-w:])))

        if len(self.episode_rewards) > 0:
            w = min(window, len(self.episode_rewards))
            self.running_avg_reward.append(float(np.mean(self.episode_rewards[-w:])))

        if len(self.collisions) > 0:
            self.running_collision_rate.append(self.recent_collision_rate(window))

    def save_json(self, path):
        data = {
            "episode_rewards": self.episode_rewards,
            "success_rates": self.success_flags,
            "travel_times": self.travel_times,
            "collisions": self.collisions,
            "training_losses": self.training_losses,
            "episode_lengths": self.episode_lengths,
            "test_success_rates": self.test_success_rates,
            "test_rewards": self.test_rewards,
            "test_travel_times": self.test_travel_times,
            "test_collision_rates": self.test_collision_rates,
            "test_total_steps": self.test_total_steps,
            "running_avg_loss": self.running_avg_loss,
            "running_avg_reward": self.running_avg_reward,
            "running_collision_rate": self.running_collision_rate,
            "total_steps": self.total_steps,
            "demo_steps": self.demo_steps,
            "episodes_completed": self.episodes_completed,
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
        if render:
            env.render()
        ep.add(obs, state, actions, reward, bool(done or truncated))
        total_reward += float(reward)
        steps += 1
        obs, state = next_obs, next_state
        if done and info.get("crashed", False):
            crashed = True

    all_exited_flag = bool(info.get("all_exited", False) or info.get("success", False))
    crashed_flag = bool(info.get("crashed", False))
    success = (all_exited_flag and not crashed_flag)
    travel_time = steps if success else 0
    return ep, total_reward, success, travel_time, crashed, steps


def test_agent(env, agent, n_episodes=20, render=False):
    succ, rew, tts, cols = [], [], [], []
    for _ in range(n_episodes):
        _, r, s, tt, crashed, _ = run_episode(env, agent, epsilon=0.0, max_steps=100, render=render)
        succ.append(s)
        rew.append(r)
        cols.append(1 if crashed else 0)
        if s:
            tts.append(tt)
    sr = float(np.mean(succ)) if succ else 0.0
    ar = float(np.mean(rew)) if rew else 0.0
    att = float(np.mean(tts)) if tts else 0.0
    cr = float(np.mean(cols)) if cols else 0.0
    return sr, ar, att, cr


def train_qmix(env, agent, max_steps, test_interval_steps, n_test_episodes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metrics = MetricsTracker()
    csv_logger = CSVLogger(save_dir)
    epsilon = 0.1

    sr0, ar0, att0, cr0 = test_agent(env, agent, n_episodes=n_test_episodes, render=False)
    metrics.add_test_point(sr0, ar0, att0, cr0)
    csv_logger.log_test(0, sr0, cr0, ar0, att0)
    metrics.save_json(f"{save_dir}/metrics.json")

    print(f"[QMIX] Start | max_steps={max_steps} | epsilon={epsilon}")
    next_test_at = test_interval_steps
    log_every = 100

    while metrics.total_steps < max_steps:
        ep_data, reward, success, travel_time, crashed, steps = run_episode(
            env, agent, epsilon=epsilon, max_steps=100, render=False
        )
        agent.replay_buffer.add_episode(ep_data)
        agent.episode_count += 1
        metrics.add_episode(reward, success, travel_time, crashed, steps)
        csv_logger.log_episode(metrics.total_steps, steps, reward, success, crashed, travel_time)

        if agent.replay_buffer.can_sample(agent.batch_size):
            loss = agent.train()
            if loss is not None:
                metrics.training_losses.append(float(loss))
                csv_logger.log_loss(float(loss))

        if metrics.episodes_completed % log_every == 0:
            metrics.update_running_metrics()
            print(f"[QMIX] Ep {metrics.episodes_completed} | Steps={metrics.total_steps}/{max_steps} | "
                  f"SR={metrics.recent_success_rate():.2%} | CR={metrics.recent_collision_rate():.2%}")

        while metrics.total_steps >= next_test_at and metrics.total_steps < max_steps:
            print(f"[QMIX] Test @ {metrics.total_steps} steps")
            sr, ar, att, cr = test_agent(env, agent, n_episodes=n_test_episodes, render=False)
            metrics.add_test_point(sr, ar, att, cr)
            csv_logger.log_test(metrics.total_steps, sr, cr, ar, att)
            print(f"  SR={sr:.2%} | CR={cr:.2%}")
            agent.save(f"{save_dir}/checkpoint_steps_{metrics.total_steps}.pth")
            metrics.save_json(f"{save_dir}/metrics.json")
            next_test_at += test_interval_steps

    agent.save(f"{save_dir}/final_model.pth")
    metrics.save_json(f"{save_dir}/metrics.json")
    csv_logger.close()
    print("[QMIX] Training done.")
    return metrics


def generate_expert_demos(env, agent, n_episodes=100):
    demos = []
    kept = 0
    attempts = 0
    max_attempts = n_episodes * 20

    print(f"Generating {n_episodes} expert episodes...")
    while kept < n_episodes and attempts < max_attempts:
        ep, _, success, _, _, _ = run_episode(env, agent, epsilon=0.0, max_steps=100, render=False)
        attempts += 1
        if success and len(ep) > 0:
            demos.append(ep)
            kept += 1
        if attempts % 50 == 0:
            print(f"  attempts={attempts}, kept={kept}/{n_episodes}")

    print(f"Kept {kept} successful expert episodes from {attempts} attempts.")
    return demos


def count_transitions(episodes):
    return int(sum(len(e) for e in episodes)) if episodes else 0


def train_qmixwd(env, agent, expert_episodes, n_pretrain_steps, max_steps,
                 test_interval_steps, n_test_episodes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_logger = CSVLogger(save_dir)

    for ep in expert_episodes:
        agent.demo_buffer.add_expert_episode(ep)
    expert_steps = count_transitions(expert_episodes)

    print("Generating 900 interaction demos (ε=0.2)...")
    inter_eps = []
    for i in range(900):
        ep, _, _, _, _, _ = run_episode(env, agent, epsilon=0.2, max_steps=100, render=False)
        if len(ep) > 0:
            inter_eps.append(ep)
            agent.demo_buffer.add_interaction_episode(ep)
        if (i + 1) % 100 == 0:
            print(f"  generated {i+1}/900")

    inter_steps = count_transitions(inter_eps)

    metrics = MetricsTracker()
    metrics.add_demo_steps(expert_steps + inter_steps)
    print(f"[QMIXwD] Demo steps counted: {metrics.total_steps}")

    sr_d, ar_d, att_d, cr_d = test_agent(env, agent, n_episodes=n_test_episodes, render=False)
    metrics.add_test_point(sr_d, ar_d, att_d, cr_d)
    csv_logger.log_test(metrics.total_steps, sr_d, cr_d, ar_d, att_d)
    print(f"Before pre-train: SR={sr_d:.2%}, CR={cr_d:.2%}")
    metrics.save_json(f"{save_dir}/metrics.json")
    print(
        f"[QMIXwD] Ep {metrics.episodes_completed} | Steps={metrics.total_steps}/{max_steps} ({100 * metrics.total_steps / max_steps:.1f}%) | SR={metrics.recent_success_rate():.2%} | CR={metrics.recent_collision_rate():.2%}")
    print(f"[QMIXwD] Pre-train for {n_pretrain_steps} gradient steps")
    _ = agent.pretrain(n_pretrain_steps)

    sr_p, ar_p, att_p, cr_p = test_agent(env, agent, n_episodes=n_test_episodes, render=False)
    metrics.add_test_point(sr_p, ar_p, att_p, cr_p)
    csv_logger.log_test(metrics.total_steps, sr_p, cr_p, ar_p, att_p)
    print(f"After pre-train: SR={sr_p:.2%}, CR={cr_p:.2%}")
    metrics.save_json(f"{save_dir}/metrics.json")

    print(f"[QMIXwD] Online training (ε=0.1) until {max_steps} steps")
    epsilon = 0.1
    next_test_at = metrics.total_steps + test_interval_steps
    log_every = 100

    while metrics.total_steps < max_steps:
        ep_data, reward, success, travel_time, crashed, steps = run_episode(
            env, agent, epsilon=epsilon, max_steps=100, render=False
        )
        agent.replay_buffer.add_episode(ep_data)
        agent.episode_count += 1
        metrics.add_episode(reward, success, travel_time, crashed, steps)
        csv_logger.log_episode(metrics.total_steps, steps, reward, success, crashed, travel_time)

        if agent.replay_buffer.can_sample(agent.batch_size):
            loss = agent.train()
            if loss is not None:
                metrics.training_losses.append(float(loss))
                csv_logger.log_loss(float(loss))

        if metrics.episodes_completed % log_every == 0:
            metrics.update_running_metrics()
            print(f"[QMIXwD] Ep {metrics.episodes_completed} | Steps={metrics.total_steps}/{max_steps} | "
                  f"SR={metrics.recent_success_rate():.2%} | CR={metrics.recent_collision_rate():.2%}")

        while metrics.total_steps >= next_test_at and metrics.total_steps < max_steps:
            print(f"[QMIXwD] Test @ {metrics.total_steps} steps")
            sr, ar, att, cr = test_agent(env, agent, n_episodes=n_test_episodes, render=False)
            metrics.add_test_point(sr, ar, att, cr)
            csv_logger.log_test(metrics.total_steps, sr, cr, ar, att)
            print(f"  SR={sr:.2%} | CR={cr:.2%}")
            agent.save(f"{save_dir}/checkpoint_steps_{metrics.total_steps}.pth")
            metrics.save_json(f"{save_dir}/metrics.json")
            next_test_at += test_interval_steps

    agent.save(f"{save_dir}/final_model.pth")
    metrics.save_json(f"{save_dir}/metrics.json")
    csv_logger.close()
    print("[QMIXwD] Training done.")

    return metrics


def print_results(metrics):
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total Steps: {metrics.total_steps:,}")
    print(f"Demo Steps: {metrics.demo_steps:,}")
    print(f"Episodes Completed: {metrics.episodes_completed:,}")
    print(f"Total Collisions: {sum(metrics.collisions):,}")
    print(f"Overall Collision Rate: {np.mean(metrics.collisions):.2%}")

    if metrics.test_success_rates:
        print(f"\nFinal Test Success Rate: {metrics.test_success_rates[-1]:.2%}")
        print(f"Best Test Success Rate: {max(metrics.test_success_rates):.2%}")

    print("="*80)


def plot_comprehensive_results(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Test Success Rate
    ax1 = fig.add_subplot(gs[0, 0])
    if metrics.test_total_steps:
        steps = np.array(metrics.test_total_steps) / 1e6
        sr = np.array(metrics.test_success_rates)
        ax1.plot(steps, sr, 'o-', color='#2ca02c', linewidth=2, markersize=6)
        ax1.set_xlabel("Steps (×10⁶)", fontsize=11)
        ax1.set_ylabel("Success Rate", fontsize=11)
        ax1.set_title("Test Success Rate", fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3)

    # 2. Test Collision Rate
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.test_total_steps:
        steps = np.array(metrics.test_total_steps) / 1e6
        cr = np.array(metrics.test_collision_rates)
        ax2.plot(steps, cr, 'o-', color='#d62728', linewidth=2, markersize=6)
        ax2.set_xlabel("Steps (×10⁶)", fontsize=11)
        ax2.set_ylabel("Collision Rate", fontsize=11)
        ax2.set_title("Test Collision Rate", fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)

    # 3. Training Loss
    ax3 = fig.add_subplot(gs[1, 0])
    if metrics.running_avg_loss:
        ax3.plot(metrics.running_avg_loss, color='#ff7f0e', linewidth=1.5)
        ax3.set_xlabel("Episode (×100)", fontsize=11)
        ax3.set_ylabel("Loss (100-ep avg)", fontsize=11)
        ax3.set_title("Training Loss", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Episode Rewards
    ax4 = fig.add_subplot(gs[1, 1])
    if metrics.running_avg_reward:
        ax4.plot(metrics.running_avg_reward, color='#1f77b4', linewidth=1.5)
        ax4.set_xlabel("Episode (×100)", fontsize=11)
        ax4.set_ylabel("Reward (100-ep avg)", fontsize=11)
        ax4.set_title("Episode Reward", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # 5. Collision Rate (training)
    ax5 = fig.add_subplot(gs[2, 0])
    if metrics.running_collision_rate:
        ax5.plot(metrics.running_collision_rate, color='#d62728', linewidth=1.5)
        ax5.set_xlabel("Episode (×100)", fontsize=11)
        ax5.set_ylabel("Collision Rate (100-ep avg)", fontsize=11)
        ax5.set_title("Training Collision Rate", fontsize=12, fontweight='bold')
        ax5.set_ylim([0, 1.05])
        ax5.grid(True, alpha=0.3)

    # 6. Episode Length
    ax6 = fig.add_subplot(gs[2, 1])
    if metrics.episode_lengths:
        window = 100
        lengths_smoothed = []
        for i in range(len(metrics.episode_lengths)):
            start = max(0, i - window//2)
            end = min(len(metrics.episode_lengths), i + window//2)
            lengths_smoothed.append(np.mean(metrics.episode_lengths[start:end]))
        ax6.plot(lengths_smoothed, color='#9467bd', linewidth=1.5)
        ax6.set_xlabel("Episode", fontsize=11)
        ax6.set_ylabel("Episode Length (100-ep avg)", fontsize=11)
        ax6.set_title("Episode Length", fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

    plt.suptitle("QMIXwD Training Metrics", fontsize=16, fontweight='bold', y=0.995)

    save_path = f"{save_dir}/comprehensive_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


def main():
    env = create_qmix_env(full_env_config_exp5, render_mode=None)
    print("=" * 60)
    print("QMIX / QMIXwD Training with Full CSV Logging")
    print("=" * 60)
    print(f"Agents: {env.n_agents}")

    cfg = {
        "max_training_steps": 30000,
        "test_interval_steps": 100,
        "n_test_episodes": 20,
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 5000,
        "expert_ratio": 0.1,
        "n_pretrain_steps": 1000,
        "n_expert_demos": 100,
    }
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("1) Train QMIX baseline")
    print("=" * 60)
    qmix_agent = QMIXAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon=0.1,
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
    )
    _ = train_qmix(
        env, qmix_agent,
        max_steps=cfg["max_training_steps"],
        test_interval_steps=cfg["test_interval_steps"],
        n_test_episodes=cfg["n_test_episodes"],
        save_dir="qmix_baseline_results",
    )

    print("\n" + "=" * 60)
    print("2) Generate expert demonstrations")
    print("=" * 60)
    expert_eps = generate_expert_demos(env, qmix_agent, n_episodes=cfg["n_expert_demos"])

    print("\n" + "=" * 60)
    print("3) Train QMIXwD (with demos)")
    print("=" * 60)
    qmixwd_agent = QMIXwDAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon=0.1,
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        expert_ratio=cfg["expert_ratio"],
    )
    metrics = train_qmixwd(
        env, qmixwd_agent, expert_eps,
        n_pretrain_steps=cfg["n_pretrain_steps"],
        max_steps=cfg["max_training_steps"],
        test_interval_steps=cfg["test_interval_steps"],
        n_test_episodes=cfg["n_test_episodes"],
        save_dir="qmixwd_results",
    )

    env.close()

    print_results(metrics)
    plot_comprehensive_results(metrics, "qmixwd_results")

    print("\n" + "="*80)
    print("CSV FILES CREATED:")
    print("="*80)
    print("  qmixwd_results/episodes.csv          - Every episode logged")
    print("  qmixwd_results/test_evaluations.csv  - Test results")
    print("  qmixwd_results/training_losses.csv   - Loss per step")
    print("  qmixwd_results/comprehensive_metrics.png")
    print("="*80)


if __name__ == "__main__":
    main()