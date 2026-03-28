import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_results(experiment, results, show_plots=True):
    """
    Generate improved training visualization with loss curves
    Args:
        experiment: the experiment configuration
        results: dictionary with training results
        show_plots: if True, displays the plots interactively
    """
    print("Generating detailed training plots...")

    # Create directory for saving plots
    plots_dir = os.path.join(experiment.EXPERIMENT_PATH, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Unpack the results
    episode_rewards = results["episode_rewards"]
    master_policy_losses = results["master_policy_losses"]
    master_value_losses = results["master_value_losses"]
    master_total_losses = results["master_total_losses"]
    agent_policy_losses = results["agent_policy_losses"]
    agent_value_losses = results["agent_value_losses"]
    agent_total_losses = results["agent_total_losses"]

    # Create x-axis for episodes
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Set up the figure style
    plt.style.use('ggplot')

    # 1. Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.title('Episode Rewards', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2. Plot Master value loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(master_value_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [master_value_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#D95F02', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Master Value Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'master_value_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 3. Plot Master total loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(master_total_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [master_total_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#7570B3', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Master Total Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'master_total_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 4. Plot Agent value loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(agent_value_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [agent_value_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#1B9E77', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Agent Value Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'agent_value_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 5. Plot Agent total loss
    plt.figure(figsize=(10, 6))
    valid_indices = [i for i, val in enumerate(agent_total_losses) if val is not None]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_losses = [agent_total_losses[i] for i in valid_indices]

    if valid_losses:
        plt.plot(valid_episodes, valid_losses, 'o-', color='#E7298A', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Agent Total Loss', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.yscale('log')  # Use log scale since losses can vary greatly
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'agent_total_loss.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    plt.figure(figsize=(12, 8))

    # Master losses
    master_valid_indices = [i for i, val in enumerate(master_total_losses) if val is not None]
    master_valid_episodes = [episodes[i] for i in master_valid_indices]
    master_valid_total_losses = [master_total_losses[i] for i in master_valid_indices]
    master_valid_value_losses = [master_value_losses[i] for i in master_valid_indices]

    # Agent losses
    agent_valid_indices = [i for i, val in enumerate(agent_total_losses) if val is not None]
    agent_valid_episodes = [episodes[i] for i in agent_valid_indices]
    agent_valid_total_losses = [agent_total_losses[i] for i in agent_valid_indices]
    agent_valid_value_losses = [agent_value_losses[i] for i in agent_valid_indices]

    if master_valid_total_losses:
        plt.plot(master_valid_episodes, master_valid_total_losses, 'o-', label='Master Total Loss',
                 color='#7570B3', linewidth=2, markersize=6)
        plt.plot(master_valid_episodes, master_valid_value_losses, 's-', label='Master Value Loss',
                 color='#D95F02', linewidth=2, markersize=6)

    if agent_valid_total_losses:
        plt.plot(agent_valid_episodes, agent_valid_total_losses, 'o-', label='Agent Total Loss',
                 color='#E7298A', linewidth=2, markersize=6)
        plt.plot(agent_valid_episodes, agent_valid_value_losses, 's-', label='Agent Value Loss',
                 color='#1B9E77', linewidth=2, markersize=6)

    plt.grid(True, alpha=0.3)
    plt.title('Training Losses', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'combined_losses.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 7. Plot combined training metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot rewards
    ax1.plot(episodes, episode_rewards, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('Episode Rewards', fontsize=16)
    ax1.set_ylabel('Reward', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot losses on second axis
    if master_valid_total_losses:
        ax2.plot(master_valid_episodes, master_valid_total_losses, 'o-', label='Master Loss',
                 color='#7570B3', linewidth=2, markersize=6)

    if agent_valid_total_losses:
        ax2.plot(agent_valid_episodes, agent_valid_total_losses, 'o-', label='Agent Loss',
                 color='#E7298A', linewidth=2, markersize=6)

    ax2.set_title('Training Losses', fontsize=16)
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Loss (log scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_summary.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()
    if show_plots:
        print("Plots displayed. Close all plot windows to continue.")

    print(f"All plots saved to: {plots_dir}")















# def monitor_rollout_buffers(master_model, agent_model):
#     """Prints statistics about the rollout buffers for debugging"""
#     print("\n--- Rollout Buffer Statistics ---")
#
#     # Master buffer stats
#     print("Master rollout buffer:")
#     print(f"  Buffer size: {master_model.rollout_buffer.buffer_size}")
#     print(f"  Current position: {master_model.rollout_buffer.pos}")
#     print(f"  Is full: {master_model.rollout_buffer.full}")
#
#     if hasattr(master_model.rollout_buffer, 'observations') and master_model.rollout_buffer.observations is not None:
#         print(f"  Observations shape: {master_model.rollout_buffer.observations.shape}")
#
#     if hasattr(master_model.rollout_buffer, 'actions') and master_model.rollout_buffer.actions is not None:
#         print(f"  Actions shape: {master_model.rollout_buffer.actions.shape}")
#
#     if hasattr(master_model.rollout_buffer, 'rewards') and master_model.rollout_buffer.rewards is not None:
#         rewards = master_model.rollout_buffer.rewards[:master_model.rollout_buffer.pos]
#         if len(rewards) > 0:
#             print(f"  Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
#             print(f"  Rewards: {rewards}")
#
#     # Agent buffer stats
#     print("\nAgent rollout buffer:")
#     print(f"  Buffer size: {agent_model.rollout_buffer.buffer_size}")
#     print(f"  Current position: {agent_model.rollout_buffer.pos}")
#     print(f"  Is full: {agent_model.rollout_buffer.full}")
#
#     if hasattr(agent_model.rollout_buffer, 'observations') and agent_model.rollout_buffer.observations is not None:
#         print(f"  Observations shape: {agent_model.rollout_buffer.observations.shape}")
#
#     if hasattr(agent_model.rollout_buffer, 'actions') and agent_model.rollout_buffer.actions is not None:
#         print(f"  Actions shape: {agent_model.rollout_buffer.actions.shape}")
#
#     if hasattr(agent_model.rollout_buffer, 'rewards') and agent_model.rollout_buffer.rewards is not None:
#         rewards = agent_model.rollout_buffer.rewards[:agent_model.rollout_buffer.pos]
#         if len(rewards) > 0:
#             print(f"  Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
#             print(f"  Rewards: {rewards}")
#
#     print("-------------------------------\n")