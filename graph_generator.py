import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def smooth(y, n=15):
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    return lfilter(b, 1, y)


if __name__ == '__main__':
    data = []
    colors = []

    # for i in range(4):
    #     try:
    #         data.append(pd.read_csv('old_logs/mcc/log-{}.csv'.format(i + 1)))
    #     except:
    #         continue
    #     colors.append('g')
    #
    # for i in range(6):
    #     try:
    #         data.append(pd.read_csv('old_logs/cartpole/log-at2c-{}.csv'.format(i + 1)))
    #     except:
    #         continue
    #     colors.append('b')

    # for i in range(6):
    try:
        data.append(pd.read_csv('final_logs/rewards_{}.csv'.format(1)))
    except:
        print("error 2")
    colors.append('r')

    # for i in range(4):
    #     data.append(pd.read_csv('old_logs/ac-episode-log-{}.csv'.format(i + 1)))
    #     colors.append('g')

    # Plot mean episode score per 100 episode - basic RL
    labels = ['No Exp Replay - Run 1', 'No Exp Replay - Run 2', 'Cartpole - TL - EX3', 'Cartpole - TL - EX4', 'Cartpole - Prog - EX1', 'Cartpole - Prog - EX2', 'Cartpole - Prog - EX3', 'Cartpole - Prog - EX4']
    # labels = ['MountainCar - TL - EX1', 'MountainCar - TL - EX2', 'MountainCar - TL - EX3', 'MountainCar - TL - EX4', 'MountainCar - Prog -  EX1', 'MountainCar - Prog - EX2', 'MountainCar - Prog - EX3', 'MountainCar - Prog - EX4']
    # labels = ['MountainCar - EX1', 'MountainCar - EX2', 'MountainCar - EX3', 'MountainCar - EX4']
    # labels = ['TL - EX1', 'TL - EX2', 'TL - EX3', 'TL - EX4']
    for i, df in enumerate(data):
        x = df.iloc[:, 1].values
        y = df.iloc[:, 2].values
        # y = smooth(y, n=5)
        # plt.plot(x, y, '.-', markersize=1, color=colors[i])
        # plt.plot(x, y, '.-', markersize=1)
        plt.scatter(x, y)
    plt.ylabel('Reward In Episode')
    plt.xlabel('Episode')
    plt.grid(linestyle='dashed')
    plt.legend(labels)
    plt.show()
    # #
    # Plot episode score - basic RL
    # labels = ['EX1', 'EX2', 'EX3', 'EX4']
    # for i, df in enumerate(data):
    #     x = df.iloc[:, 1].values
    #     y = df.iloc[:, 2].values
    #     plt.subplot(round(len(data)/2), 2, i + 1)
    #     plt.plot(x, y)
    #     plt.ylabel('Episode score')
    #     plt.xlabel('Episode')
    #     plt.grid(linestyle='dashed')
    #     # plt.legend([labels[i]])
    # # plt.savefig('figures/figure_ac_rl_episode_score.png', bbox_inches='tight', dpi=700)
    # plt.show()




