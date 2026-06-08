import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── helpers ────────────────────────────────────────────────────────────────────

def _smooth(values, window=20):
    """Moving-average smoothing. Returns array same length as input."""
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < window:
        window = max(1, len(arr))
    kernel = np.ones(window) / window
    # 'valid' shrinks the output; pad with nan on both sides to keep length
    pad = window // 2
    padded = np.concatenate([np.full(pad, np.nan), arr, np.full(pad, np.nan)])
    smoothed = np.convolve(np.where(np.isnan(padded), 0, padded), kernel, mode='valid')
    counts   = np.convolve((~np.isnan(padded)).astype(float), kernel, mode='valid')
    smoothed = smoothed / np.where(counts > 0, counts, 1)
    smoothed[counts == 0] = np.nan
    return smoothed[:len(arr)]


def _loss_subplot(ax, values, color, title):
    """Draw raw (light) + smoothed (dark) loss curve on ax."""
    x = np.arange(1, len(values) + 1)
    raw = np.array([v if v is not None else np.nan for v in values], dtype=float)
    smoothed = _smooth(values, window=20)

    ax.plot(x, raw,      color=color, alpha=0.25, linewidth=0.8)
    ax.plot(x, smoothed, color=color, alpha=1.0,  linewidth=2.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel("Loss",    fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


# ── main entry point ───────────────────────────────────────────────────────────

def plot_training_results(experiment, results, show_plots=True):
    """
    Generate three plots matching the target style:
      1. Training Loss – All Network Levels  (3 subplots)
      2. Arrival Rate
      3. Agent Rewards
    """
    plots_dir = os.path.join(experiment.EXPERIMENT_PATH, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print("Generating training plots...")

    episode_rewards        = results.get("episode_rewards", [])
    arrival_rates          = results.get("arrival_rates", [])
    master_total_losses    = results.get("master_total_losses", [])
    group0_total_losses    = results.get("agent_group0_total_losses", [])
    group1_total_losses    = results.get("agent_group1_total_losses", [])

    plt.style.use('seaborn-v0_8-whitegrid')

    # ── 1. Training Loss – All Network Levels ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Training Loss – All Network Levels", fontsize=13, fontweight='bold', y=1.01)

    _loss_subplot(axes[0], master_total_losses, color='#4472C4', title='Shared Master')
    _loss_subplot(axes[1], group0_total_losses, color='#C0504D', title='Agent Group0')
    _loss_subplot(axes[2], group1_total_losses, color='#4F9A50', title='Agent Group1')

    plt.tight_layout()
    path = os.path.join(plots_dir, 'training_loss_all_levels.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ── 2. Arrival Rate ────────────────────────────────────────────────────────
    if arrival_rates:
        x = np.arange(1, len(arrival_rates) + 1)
        raw      = np.array(arrival_rates, dtype=float)
        smoothed = _smooth(arrival_rates, window=20)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, raw,      color='#4F9A50', alpha=0.25, linewidth=0.8)
        ax.plot(x, smoothed, color='#2E7D32', alpha=1.0,  linewidth=2.2,
                label='Arrival Rate (smoothed)')
        ax.set_title('Arrival Rate', fontsize=14)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Arrival Rate (%)', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(plots_dir, 'arrival_rate.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

    # ── 3. Agent Rewards ───────────────────────────────────────────────────────
    if episode_rewards:
        x = np.arange(1, len(episode_rewards) + 1)
        raw      = np.array(episode_rewards, dtype=float)
        smoothed = _smooth(episode_rewards, window=20)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, raw,      color='#5B9BD5', alpha=0.25, linewidth=0.8)
        ax.plot(x, smoothed, color='#2E75B6', alpha=1.0,  linewidth=2.2,
                label='Total reward')
        ax.set_title('Agent Rewards', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(plots_dir, 'agent_rewards.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

    print(f"All plots saved to: {plots_dir}")
