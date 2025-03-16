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
