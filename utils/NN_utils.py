import os

from tensorflow import keras

from RL.config import EXPERIMENT_ID, WEIGHTS_TO_SAVE_NAME, LOSS_FUNCTION


def init_network(optimizer):
    """
        Master Input (for embedding): (x_c1, y_c1, Vx_c1, Vy_c1, x_c2, y_c2, Vx_c2, Vy_c2) - size 8
        Agent Input: (state of the car - x, y, Vx, Vy) - size 4
        Output: (q_value1, q_value2)
    """
    # Define the two separate inputs
    master_input = keras.Input(shape=(18,))
    agent_input = keras.Input(shape=(9,))

    # Process the first input
    layer_1 = keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform())(master_input)

    # Combine the first processed input with the second input
    combined = keras.layers.Concatenate()([layer_1, agent_input])

    # Additional processing after combining inputs
    layer_2 = keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform())(combined)
    layer_3 = keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform())(layer_2)

    # Output layer
    outputs = keras.layers.Dense(units=2, activation='linear')(layer_3)

    # Create the model
    model = keras.Model(inputs=[master_input, agent_input], outputs=outputs)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    return model


def copy_network(network):
    # for alternate training purpose
    return keras.models.clone_model(network)


def save_network_weights(network):
    # Create the directory if it doesn't exist
    save_dir = f"experiments/{EXPERIMENT_ID}/weights"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the weights to the specified directory
    save_path = f"{save_dir}/{WEIGHTS_TO_SAVE_NAME}.h5"
    network.save_weights(save_path)
