import matplotlib.pyplot as plt
import pandas as pd


class PlottingUtils:

    @staticmethod
    def plot_losses(path):
        loss_data = pd.read_csv(f"{path}/progress.csv")
        losses = loss_data["train/value_loss"]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss')
        plt.legend()

    @staticmethod
    def plot_rewards(all_rewards):
        plt.subplot(1, 2, 2)
        plt.plot(all_rewards, label='Cumulative Rewards')
        plt.title('Cumulative Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()

    @staticmethod
    def show_plots():
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_actions(all_actions):
        plt.subplot(1, 3, 3)
        for i, actions in enumerate(all_actions):
            plt.plot(actions, label=f'Episode {i + 1}')
        plt.title('Actions of Car 1 Over Episodes')
        plt.xlabel('step')
        plt.ylabel('Action')
        plt.legend()

    @staticmethod
    def plot_vehicle_speeds(speeds_car1, speeds_car2, episode_steps, collision_steps=None):
        """
        Plot speed changes of two vehicles over time, with episode and collision markers.
        """

        # --- Validate input ---
        assert len(speeds_car1) == len(speeds_car2), "Speed lists must have the same length."

        if episode_steps is None:
            episode_steps = list(range(0, len(speeds_car1), max(1, len(speeds_car1) // 10)))  # Default: every 10% steps

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # --- Plot vehicle speeds ---
        ax1.plot(speeds_car1, label='Car 1 Speed', linestyle='-', marker='o', markersize=3, alpha=0.7)
        ax1.plot(speeds_car2, label='Car 2 Speed', linestyle='-', marker='s', markersize=3, alpha=0.7)

        # --- Plot collisions ---
        if collision_steps:
            # Add collision label only once for the legend
            ax1.plot([], [], 'rx', markersize=10, markeredgewidth=2, label='Collision')
            for step in collision_steps:
                if 0 <= step < len(speeds_car1):
                    ax1.plot(step, speeds_car1[step], 'rx', markersize=10, markeredgewidth=2)
                    ax1.plot(step, speeds_car2[step], 'rx', markersize=10, markeredgewidth=2)
                    ax1.vlines(x=step,
                               ymin=min(speeds_car1[step], speeds_car2[step]),
                               ymax=max(speeds_car1[step], speeds_car2[step]),
                               color='r', linestyle='--', alpha=0.5)
                    ax1.axvline(x=step, color='r', linestyle='--', alpha=0.3)

        # --- Plot episode markers (optional vertical lines) ---
        for ep in episode_steps:
            ax1.axvline(x=ep, color='gray', linestyle=':', alpha=0.2)

        # --- Configure main axis ---
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Speeds (m/s)')
        ax1.set_title('Speed Changes of Car 1 and Car 2 with Collision Points')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # Format episode ticks
        ax1.set_xticks(episode_steps)
        ax1.set_xticklabels([f'Ep {i + 1}' for i in range(len(episode_steps))], rotation=45)

        # --- Secondary X-axis for time steps ---
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Time Steps')

        fig.tight_layout()
        return fig

def plot_metrics(metrics_path):
    # Load metrics
    metrics = pd.read_csv(metrics_path)
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["loss"], label="Loss")
    plt.title("Loss over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["entropy"], label="Entropy", color="orange")
    plt.title("Entropy over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Return
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["return"], label="Return", color="green")
    plt.title("Return over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(path, all_rewards, all_actions):
    PlottingUtils.plot_losses(path)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)



