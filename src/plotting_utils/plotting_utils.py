import numpy as np
import matplotlib.pyplot as plt
import os
import re

def sanitize_filename(s):
    """Sanitize string to create a safe filename."""
    return re.sub(r"[^\w\-\.]", "_", str(s))


def plot_training_results(experiment, results, show_plots=True):
    """
    Generate detailed training visualization with reward and loss curves.

    Args:
        experiment: Experiment configuration object.
        results: Dictionary with training results.
        show_plots: If True, display plots interactively; otherwise, save them only.
    """
    print("Generating detailed training plots...")

    # Set up plot directory
    plots_dir = os.path.join(experiment.EXPERIMENT_PATH, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Sanitize label
    label_suffix = getattr(experiment, "GRID_SEARCH_LABEL", "")
    safe_label = sanitize_filename(label_suffix)

    def save_plot(filename):
        path = os.path.join(plots_dir, f"{safe_label}{filename}")
        plt.savefig(path)

    # Unpack results
    episode_rewards = results["episode_rewards"]
    master_policy_losses = results["master_policy_losses"]
    master_value_losses = results["master_value_losses"]
    master_total_losses = results["master_total_losses"]
    agent_policy_losses = results["agent_policy_losses"]
    agent_value_losses = results["agent_value_losses"]
    agent_total_losses = results["agent_total_losses"]

    episodes = np.arange(1, len(episode_rewards) + 1)
    plt.style.use('ggplot')

    # ---------- 1. Episode Rewards ----------
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.title('Episode Rewards', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.tight_layout()
    save_plot('episode_rewards.png')
    plt.show() if show_plots else plt.close()

    # ---------- Helper for loss plotting ----------
    def plot_loss(title, data, color, filename):
        plt.figure(figsize=(10, 6))
        valid_indices = [i for i, val in enumerate(data) if val is not None]
        if valid_indices:
            x = [episodes[i] for i in valid_indices]
            y = [data[i] for i in valid_indices]
            plt.plot(x, y, 'o-', color=color, linewidth=2, markersize=6)
            plt.grid(True, alpha=0.3)
            plt.title(title, fontsize=16)
            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.yscale('log')
            plt.tight_layout()
            save_plot(filename)
            plt.show() if show_plots else plt.close()

    # ---------- 2–5. Loss Plots ----------
    plot_loss('Master Value Loss', master_value_losses, '#D95F02', 'master_value_loss.png')
    plot_loss('Master Total Loss', master_total_losses, '#7570B3', 'master_total_loss.png')
    plot_loss('Agent Value Loss', agent_value_losses, '#1B9E77', 'agent_value_loss.png')
    plot_loss('Agent Total Loss', agent_total_losses, '#E7298A', 'agent_total_loss.png')

    # ---------- 6. Combined Losses Plot ----------
    plt.figure(figsize=(12, 8))

    def extract_valid(data):
        idx = [i for i, val in enumerate(data) if val is not None]
        return [episodes[i] for i in idx], [data[i] for i in idx]

    master_ep, master_total = extract_valid(master_total_losses)
    _, master_value = extract_valid(master_value_losses)
    agent_ep, agent_total = extract_valid(agent_total_losses)
    _, agent_value = extract_valid(agent_value_losses)

    if master_total:
        plt.plot(master_ep, master_total, 'o-', label='Master Total Loss', color='#7570B3', linewidth=2, markersize=6)
        plt.plot(master_ep, master_value, 's-', label='Master Value Loss', color='#D95F02', linewidth=2, markersize=6)

    if agent_total:
        plt.plot(agent_ep, agent_total, 'o-', label='Agent Total Loss', color='#E7298A', linewidth=2, markersize=6)
        plt.plot(agent_ep, agent_value, 's-', label='Agent Value Loss', color='#1B9E77', linewidth=2, markersize=6)

    plt.grid(True, alpha=0.3)
    plt.title('Training Losses', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_plot('combined_losses.png')
    plt.show() if show_plots else plt.close()

    # ---------- 7. Combined Rewards + Losses ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Rewards
    ax1.plot(episodes, episode_rewards, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('Episode Rewards', fontsize=16)
    ax1.set_ylabel('Reward', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Losses
    if master_total:
        ax2.plot(master_ep, master_total, 'o-', label='Master Loss', color='#7570B3', linewidth=2, markersize=6)
    if agent_total:
        ax2.plot(agent_ep, agent_total, 'o-', label='Agent Loss', color='#E7298A', linewidth=2, markersize=6)

    ax2.set_title('Training Losses', fontsize=16)
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Loss (log scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    save_plot('training_summary.png')
    plt.show() if show_plots else plt.close()

    if show_plots:
        print("Plots displayed. Close all plot windows to continue.")
    print(f"All plots saved to: {plots_dir}")
