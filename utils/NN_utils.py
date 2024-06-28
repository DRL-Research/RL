import os

from tensorflow import keras

from RL.config import EXPERIMENT_ID, WEIGHTS_TO_SAVE_NAME, LOSS_FUNCTION


def init_network(optimizer):
    """
        Master Input (for embedding): (x_c1, y_c1, Vx_c1, Vy_c1, x_c2, y_c2, Vx_c2, Vy_c2) - size 8  TODO: update values
        Agent Input: (state of the car - x, y, Vx, Vy) - size 4  TODO: update values
        Output: (q_value1, q_value2)
    """
    # Define master_input and agent_input
    master_input = keras.Input(shape=(10,), name="master_input")
    agent_input = keras.Input(shape=(5,), name="agent_input")

    # Master layer(s)
    master_layer_1 = keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="master_layer_1")(master_input)

    # Combine master embedding with input of agent
    combined = keras.layers.Concatenate(name="combined")([master_layer_1, agent_input])

    # Agent layers
    agent_layer_2 = keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_2")(combined)
    agent_layer_3 = keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_3")(agent_layer_2)

    # Output layer
    outputs = keras.layers.Dense(units=2, activation='linear', name="outputs")(agent_layer_3)

    # Create the model
    model = keras.Model(inputs=[master_input, agent_input], outputs=outputs)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    return model


def alternate_master_and_agent_training(network, freeze_master):
    for layer in network.layers:
        if layer.name in ["master_input", "master_layer_1", "combined"]:
            layer.trainable = not freeze_master
        else:  # agent_input, agent_layer_2, agent_layer_3, outputs
            layer.trainable = freeze_master


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
