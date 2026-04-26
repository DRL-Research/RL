# plot_master_reward_comparison_scientific.py
"""
Scientific-style bar plot comparing episode rewards for different Master modes.
Mean ± std reward only.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    labels = [
        "Trained Master",
        "Random Master",
        "Zero Master",
    ]

    mean_rewards = np.array([56.6, 25.7, -304.7])
    std_rewards = np.array([5.0, 57.5, 20.9])

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        x,
        mean_rewards,
        yerr=std_rewards,
        capsize=6,
    )

    # Grid (scientific style)
    plt.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.7,
    )

    # Zero reference line
    plt.axhline(0, linewidth=1)

    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Episode Rewards")
    plt.title("Effect of Master Policy on Episode Rewards")

    # Annotate bars
    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{mean_rewards[i]:.1f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
