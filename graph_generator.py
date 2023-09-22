import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def smooth(y, n=15):
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    return lfilter(b, 1, y)


def create_graph_of_local_experiment():
    data = []
    colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]

    # upload iteration 1:
    data1 = "Local_Experiment/rewards_1_first_left_20230425-143409"
    data.append(pd.read_csv('Results/' + data1 + '.csv'.format(1)))

    # upload iteration 5:
    data2 = "Local_Experiment/rewards_5_second_right_20230425-145747"
    data.append(pd.read_csv('Results/' + data2 + '.csv'.format(1)))

    # upload iteration 11:
    data3 = "Local_Experiment/rewards_11_fifth_right_20230425-153707"
    data.append(pd.read_csv('Results/' + data3 + '.csv'.format(1)))

    # upload iteration 12:
    data4 = "Local_Experiment/rewards_12_sixth_right_20230425-154127"
    data.append(pd.read_csv('Results/' + data4 + '.csv'.format(1)))

    # define the labels:
    labels = ['No Exp Replay - Run 1', 'No Exp Replay - Run 2', 'Cartpole - TL - EX3', 'Cartpole - TL - EX4',
              'Cartpole - Prog - EX1', 'Cartpole - Prog - EX2', 'Cartpole - Prog - EX3', 'Cartpole - Prog - EX4']

    # one graph only:
    # for i, df in enumerate(data):
    #     x = df.iloc[:, 1].values
    #     y = df.iloc[:, 2].values
    #     # y = smooth(y, n=5)  # smooth the graph, comment if not needed
    #     plt.plot(x, y, '.-', markersize=1, color=colors[i])  # create lines, comment if not needed
    #     # plt.scatter(x, y)  # create scatter plot, comment if not needed
    # plt.ylabel('Reward In Episode')
    # plt.xlabel('Episode')
    # plt.grid(linestyle='dashed')
    # plt.legend(labels)
    # plt.show()

    # using subplots:
    labels = ['Iteration 1', 'Iteration 5', 'Iteration 11', 'Iteration 12']
    for i, df in enumerate(data):
        x = df.iloc[:, 1].values
        y = df.iloc[:, 2].values
        plt.subplot(round(len(data) / 2), 2, i + 1)
        plt.plot(x, y, color=colors[i])
        plt.ylabel('Reward In Episode')
        plt.xlabel('Episode')
        plt.grid(linestyle='dashed')
        plt.legend([labels[i]])
    plt.show()


def create_graph_of_global_experiment():
    data = []
    colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]

    # upload iteration 1:
    data1 = "Global_Experiment/rewards_1_first_left_20230520-193845"
    data.append(pd.read_csv('Results/' + data1 + '.csv'.format(1)))

    # upload iteration 5:
    data2 = "Global_Experiment/rewards_5_second_right_20230520-200640"
    data.append(pd.read_csv('Results/' + data2 + '.csv'.format(1)))

    # upload iteration 11:
    data3 = "Global_Experiment/rewards_11_fifth_right_20230520-203915"
    data.append(pd.read_csv('Results/' + data3 + '.csv'.format(1)))

    # upload iteration 12:
    data4 = "Global_Experiment/rewards_12_sixth_right_20230520-204438"
    data.append(pd.read_csv('Results/' + data4 + '.csv'.format(1)))

    # define the labels:
    labels = ['No Exp Replay - Run 1', 'No Exp Replay - Run 2', 'Cartpole - TL - EX3', 'Cartpole - TL - EX4',
              'Cartpole - Prog - EX1', 'Cartpole - Prog - EX2', 'Cartpole - Prog - EX3', 'Cartpole - Prog - EX4']

    # one graph only:
    # for i, df in enumerate(data):
    #     x = df.iloc[:, 1].values
    #     y = df.iloc[:, 2].values
    #     # y = smooth(y, n=5)  # smooth the graph, comment if not needed
    #     plt.plot(x, y, '.-', markersize=1, color=colors[i])  # create lines, comment if not needed
    #     # plt.scatter(x, y)  # create scatter plot, comment if not needed
    # plt.ylabel('Reward In Episode')
    # plt.xlabel('Episode')
    # plt.grid(linestyle='dashed')
    # plt.legend(labels)
    # plt.show()

    # using subplots:
    labels = ['Iteration 1', 'Iteration 5', 'Iteration 11', 'Iteration 12']
    for i, df in enumerate(data):
        x = df.iloc[:, 1].values
        y = df.iloc[:, 2].values
        plt.subplot(round(len(data) / 2), 2, i + 1)
        plt.plot(x, y, color=colors[i])
        plt.ylabel('Reward In Episode')
        plt.xlabel('Episode')
        plt.grid(linestyle='dashed')
        plt.legend([labels[i]])
    plt.show()


if __name__ == '__main__':
    # create_graph_of_local_experiment()
    create_graph_of_global_experiment()