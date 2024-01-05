from tensorflow import keras


def init_local_network(learning_rate):
    """
    input of network: (x_car, y_car, v_car1, v_car2, up_car2,down_car2,right_car2,left_car2, dist_c1_c2)
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
    return keras.models.clone_model(network)
