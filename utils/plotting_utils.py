import matplotlib.pyplot as plt
import pandas as pd


class PlottingUtils:
    @staticmethod
    def plot_losses(path, experiment_id):
        loss_data = pd.read_csv(f"{path}/progress.csv")
        losses = loss_data["train/value_loss"]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.title(f'{experiment_id} - Model Loss Over Time')
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss')
        plt.legend()

    @staticmethod
    def plot_rewards(all_rewards, experiment_id):
        plt.subplot(1, 2, 2)
        plt.plot(all_rewards, label='Cumulative Rewards')
        plt.title(f'{experiment_id} - Cumulative Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()

    # @staticmethod
    # def plot_actions(all_actions):
    #     plt.subplot(1, 3, 3)
    #     for i, actions in enumerate(all_actions):
    #         plt.plot(actions, label=f'Episode {i + 1}')
    #     plt.title('Actions of Car 1 Over Episodes')
    #     plt.xlabel('step')
    #     plt.ylabel('Action')
    #     plt.legend()

    @staticmethod
    def plot_actions(all_actions, experiment_id):
        # Create subplots with one column and a number of rows equal to the number of episodes
        num_episodes = len(all_actions)
        fig, axes = plt.subplots(num_episodes, 1, figsize=(10, 6), sharex=True)

        # Iterate over each episode in reverse order and plot its actions
        for i, actions in enumerate(reversed(all_actions)):
            ax = axes[i] if num_episodes > 1 else axes  # Handle single subplot case
            ax.plot(actions, label=f'Episode {num_episodes - i}', color=f"C{i % 10}")  # Reverse numbering
            ax.set_title(f'Episode {num_episodes - i}')
            ax.set_ylabel('Action (0 = Slower, 1 = Faster)')
            ax.grid(True)
            ax.legend(loc='upper right')

        fig.suptitle(f'Controlled Car Actions Across Episodes - {experiment_id}', fontsize=14)
        # Set shared x-axis label
        plt.xlabel('Step')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    @staticmethod
    def show_plots():
        plt.tight_layout()
        plt.show()
