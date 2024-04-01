from tensorflow import keras
import os
from RL.config import EXPERIMENT_ID, WEIGHTS_TO_SAVE_ID


def init_local_network(optimizer):
    """
    input of network: (x_c1, y_c1, Vx_c1, Vy_c1, x_c2, y_c2, Vx_c2, Vy_c2, dist_c1_c2, 5 neurons of global network)
    output of network: (q_value1, q_value2)
    """
    network = keras.Sequential([
        keras.layers.InputLayer(input_shape=(14,)),
        keras.layers.Normalization(axis=-1),
        keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        keras.layers.Dense(units=2, activation='linear')
    ])
    network.compile(optimizer=optimizer, loss="mse")
    return network


def init_global_network(optimizer):
    """
    input of network: (x_c1, y_c1, Vx_c1, Vy_c1, proto-plan1 (5 neurons),
                       x_c2, y_c2, Vx_c2, Vy_c2, proto-plan2 (5 neurons), dist_c1_c2)
    output of network: proto_plan_output_network -> outputs the proto-plan of car 1/2 (5 neurons)
                       network -> outputs the expected reward (used for training (compared to actual reward))
    """
    network = keras.Sequential([
        keras.layers.InputLayer(input_shape=(19,)),
        keras.layers.Normalization(axis=-1),
        keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        keras.layers.Dense(units=5, activation='linear'),  # embeddings
        keras.layers.Dense(units=1, activation='linear')  # reward prediction for training
    ])

    proto_plan_output_network = keras.Model(inputs=network.input, outputs=network.layers[-2].output)

    network.compile(optimizer=optimizer, loss="mse")

    return proto_plan_output_network, network

def copy_network(network):
    # for alternate training purpose
    return keras.models.clone_model(network)


def save_network_weights(rl_agent):
    # Create the directory if it doesn't exist
    save_dir = f"experiments/{EXPERIMENT_ID}/weights"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the weights to the specified directory
    save_path = f"{save_dir}/{WEIGHTS_TO_SAVE_ID}"
    rl_agent.local_network_car1.save_weights(save_path)

