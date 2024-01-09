from tensorflow import keras
import os


def init_local_network(learning_rate):
    """
    input of network: (x_car, y_car, v_car1, v_car2, up_car2, down_car2, right_car2, left_car2, dist_c1_c2)
    output of network: (q_value1, q_value2)
    """
    network = keras.Sequential([
        keras.layers.InputLayer(input_shape=(9,)),
        keras.layers.Normalization(axis=-1),
        keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        keras.layers.Dense(units=2, activation='linear')
    ])
    network.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss="mse")
    return network


def copy_network(network):
    # for alternate training purpose
    return keras.models.clone_model(network)


def save_network_weights(experiment_params, rl_agent):
    # Create the directory if it doesn't exist
    save_dir = f"experiments/{experiment_params.experiment_id}/weights"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the weights to the specified directory
    save_path = f"{save_dir}/{experiment_params.weights_to_save_id}"
    rl_agent.local_network.save_weights(save_path)


