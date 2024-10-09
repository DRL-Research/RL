import matplotlib.pyplot as plt
import pandas as pd

class PlottingUtils:

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
