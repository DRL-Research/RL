# plot_3graphs.py
"""
Generate 3 plots matching original style: Loss Convergence, Reward Curves, Crash Rate
Only for dimensions 2, 4, 8
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
CSV_DIR = r"C:\Users\glebb\Downloads\RL-highway-feature-exp6_2_agents_100_scenarios\F21\RL-highway-feature-exp6_2_agents_100_scenarios\new_arch\embedding_experiment_20260205_213851\csv_data"
OUTPUT_DIR = r"C:\Users\glebb\Downloads\RL-highway-feature-exp6_2_agents_100_scenarios\F21\RL-highway-feature-exp6_2_agents_100_scenarios\new_arch\embedding_experiment_20260205_213851\plots"

DIMS = [2, 4, 8]
COLORS = {2: '#E41A1C', 4: '#377EB8', 8: '#4DAF4A'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_loss_convergence():
    """Plot 1: Loss Convergence - RAW DATA"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

    # Master Policy Loss
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'master_loss_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            ax1.plot(df['episode'], df['policy_loss'],
                     color=COLORS[dim], linewidth=1.5, label=f'dim={dim}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Policy Loss')
    ax1.set_title('Master Policy Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Master Value Loss
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'master_loss_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            ax2.plot(df['episode'], df['value_loss'],
                     color=COLORS[dim], linewidth=1.5, label=f'dim={dim}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Master Value Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Agent Policy Loss
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'agent_loss_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            ax3.plot(df['episode'], df['policy_loss'],
                     color=COLORS[dim], linewidth=1.5, label=f'dim={dim}')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Policy Loss')
    ax3.set_title('Agent Policy Loss', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Agent Value Loss
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'agent_loss_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            ax4.plot(df['episode'], df['value_loss'],
                     color=COLORS[dim], linewidth=1.5, label=f'dim={dim}')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Value Loss')
    ax4.set_title('Agent Value Loss', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'loss_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_reward_curves():
    """Plot 2: Training Reward Curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Reward Curves (Rolling Average)', fontsize=14, fontweight='bold')

    # Left: Raw + Rolling Average
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'episodes_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            # Raw data (very transparent)
            ax1.plot(df['episode'], df['reward'],
                     color=COLORS[dim], linewidth=0.3, alpha=0.1)
            # Rolling average
            rolling = df['reward'].rolling(window=50, min_periods=1).mean()
            ax1.plot(df['episode'], rolling,
                     color=COLORS[dim], linewidth=2.5, label=f'dim={dim}')

    ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5, linestyle='-')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Episode Reward', fontsize=11)
    ax1.set_title('Training Reward Curves (Rolling Average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Boxplot
    reward_data = []
    labels = []
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'episodes_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            reward_data.append(df['reward'].values)
            labels.append(str(dim))

    bp = ax2.boxplot(reward_data, labels=labels, patch_artist=True, widths=0.5)
    for patch, dim in zip(bp['boxes'], DIMS):
        patch.set_facecolor(COLORS[dim])
        patch.set_alpha(0.6)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Embedding Dimension', fontsize=11)
    ax2.set_ylabel('Episode Reward', fontsize=11)
    ax2.set_title('Reward Distribution by Embedding Dimension')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'reward_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_crash_rate():
    """Plot 3: Success Rate"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Success Rate Over Training', fontsize=14, fontweight='bold')

    # Left: Success Rate Over Training
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'episodes_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df['success'] = 1 - df['crashed']
            rolling = df['success'].rolling(window=50, min_periods=1).mean()
            ax1.plot(df['episode'], rolling,
                     color=COLORS[dim], linewidth=2.5, label=f'dim={dim}')

    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Success Rate (No Crash)', fontsize=11)
    ax1.set_title('Success Rate Over Training')
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Final Success Rate
    success_rates = []
    for dim in DIMS:
        fp = os.path.join(CSV_DIR, f'episodes_dim{dim}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            last_100 = df.tail(100)
            success_rate = 1 - last_100['crashed'].mean()
            success_rates.append(success_rate)

    bars = ax2.bar(range(len(DIMS)), success_rates,
                   color=[COLORS[d] for d in DIMS], alpha=0.7, width=0.5,
                   edgecolor='black', linewidth=1.2)

    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{rate * 100:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(range(len(DIMS)))
    ax2.set_xticklabels([str(d) for d in DIMS])
    ax2.set_xlabel('Embedding Dimension', fontsize=11)
    ax2.set_ylabel('Final Success Rate', fontsize=11)
    ax2.set_title('Final Success Rate (Last 100 Episodes)')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'crash_rate.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Generating plots for dims 2, 4, 8...")
    print(f"Reading from: {CSV_DIR}")
    print(f"Saving to: {OUTPUT_DIR}\n")

    plot_loss_convergence()
    plot_reward_curves()
    plot_crash_rate()

    print("\nDone!")