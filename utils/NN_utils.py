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
    # Pay attention: input layers do not have weights.
    # Concat layers simply concatenates the outputs of the previous layers.
    # The Concatenate layer itself does not have any weights or parameters to be trained
    for layer in network.layers:
        if layer.name in ["master_layer_1"]:
            layer.trainable = not freeze_master
        else:  # agent_layer_2, agent_layer_3, outputs
            layer.trainable = freeze_master


def create_network_using_agent_only_from_original(original_model):

    agent_input = keras.Input(shape=(5,), name="agent_input")
    original_agent_layer_2 = original_model.get_layer("agent_layer_2")
    original_agent_layer_3 = original_model.get_layer("agent_layer_3")
    original_output_layer = original_model.get_layer("outputs")

    # Get weights of agent_layer_2 from the original model
    agent_layer_2_weights = original_agent_layer_2.get_weights()

    # Extract the part of the weights related to the agent_input (last 5 units)
    new_agent_layer_2_weights = [
        agent_layer_2_weights[0][-5:, :],  # Kernel weights
        agent_layer_2_weights[1]  # Bias weights (unchanged)
    ]

    # Define the agent layers using the trimmed weights
    agent_layer_2 = keras.layers.Dense(units=16, activation='relu',
                                       kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_2")
    agent_layer_2.build((None, 5))
    agent_layer_2.set_weights(new_agent_layer_2_weights)

    agent_layer_3 = keras.layers.Dense(units=8, activation='relu',
                                       kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_3")
    agent_layer_3.build((None, 16))
    agent_layer_3.set_weights(original_agent_layer_3.get_weights())

    # Create the output layer
    outputs = keras.layers.Dense(units=2, activation='linear', name="outputs")
    outputs.build((None, 8))
    outputs.set_weights(original_output_layer.get_weights())

    # Create the new model
    agent_only_model = keras.Model(inputs=agent_input, outputs=outputs(agent_layer_3(agent_layer_2(agent_input))))
    agent_only_model.compile(optimizer='adam', loss='mse')
    return agent_only_model


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
