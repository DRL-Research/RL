import os
from tensorflow import keras

from RL.config import EXPERIMENT_ID, WEIGHTS_TO_SAVE_NAME, LOSS_FUNCTION


def create_master_network(master_input_shape):
    master_input_layer = keras.layers.Input(shape=master_input_shape, name="master_input")
    master_layer_1 = keras.layers.Dense(units=32, kernel_initializer='he_uniform', name="master_layer_1")(master_input_layer)
    master_layer_1 = keras.layers.BatchNormalization()(master_layer_1)
    master_layer_1 = keras.layers.LeakyReLU()(master_layer_1)
    master_layer_1 = keras.layers.Dropout(0.3)(master_layer_1)

    return master_input_layer, master_layer_1


def create_agent_network(combined_input_shape):
    combined = keras.layers.Input(shape=combined_input_shape, name="combined")
    agent_layer_2 = keras.layers.Dense(units=32, kernel_initializer='he_uniform', name="agent_layer_2")(combined)
    agent_layer_2 = keras.layers.BatchNormalization()(agent_layer_2)
    agent_layer_2 = keras.layers.LeakyReLU()(agent_layer_2)
    agent_layer_2 = keras.layers.Dropout(0.3)(agent_layer_2)

    agent_layer_3 = keras.layers.Dense(units=16, kernel_initializer='he_uniform', name="agent_layer_3")(agent_layer_2)
    agent_layer_3 = keras.layers.BatchNormalization()(agent_layer_3)
    agent_layer_3 = keras.layers.LeakyReLU()(agent_layer_3)

    outputs = keras.layers.Dense(units=2, activation='linear', name="outputs")(agent_layer_3)

    return combined, outputs


def create_full_network(master_input_shape, agent_input_shape, optimizer, loss_function):
    master_input_layer, master_layer_1 = create_master_network(master_input_shape)
    agent_input = keras.layers.Input(shape=agent_input_shape, name="agent_input")

    combined = keras.layers.Concatenate(name="combined")([master_layer_1, agent_input])
    combined_input_shape = combined.shape[1:]

    combined, outputs = create_agent_network(combined_input_shape)

    model = keras.Model(inputs=[master_input_layer, agent_input], outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model




























def init_network_master_and_agent(optimizer):

    # Define master_input and agent_input
    master_input = keras.layers.Input(shape=(10,), name="master_input")
    agent_input = keras.layers.Input(shape=(5,), name="agent_input")

    # Master layer(s)
    master_layer_1 = keras.layers.Dense(units=32, kernel_initializer='he_uniform', name="master_layer_1")(master_input)
    master_layer_1 = keras.layers.BatchNormalization()(master_layer_1)
    master_layer_1 = keras.layers.LeakyReLU()(master_layer_1)
    master_layer_1 = keras.layers.Dropout(0.3)(master_layer_1)

    # Combine master embedding with input of agent
    combined = keras.layers.Concatenate(name="combined")([master_layer_1, agent_input])

    # Agent layers
    agent_layer_2 = keras.layers.Dense(units=32, kernel_initializer='he_uniform', name="agent_layer_2")(combined)
    agent_layer_2 = keras.layers.BatchNormalization()(agent_layer_2)
    agent_layer_2 = keras.layers.LeakyReLU()(agent_layer_2)
    agent_layer_2 = keras.layers.Dropout(0.3)(agent_layer_2)

    agent_layer_3 = keras.layers.Dense(units=16, kernel_initializer='he_uniform', name="agent_layer_3")(agent_layer_2)
    agent_layer_3 = keras.layers.BatchNormalization()(agent_layer_3)
    agent_layer_3 = keras.layers.LeakyReLU()(agent_layer_3)

    # Output layer
    outputs = keras.layers.Dense(units=2, activation='linear', name="outputs")(agent_layer_3)

    # Create the model
    model = keras.Model(inputs=[master_input, agent_input], outputs=outputs)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    return model


def init_network_agent_only(optimizer):

    # Define master_input and agent_input
    agent_input = keras.Input(shape=(5,), name="agent_input")

    # Agent layers
    agent_layer_2 = keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_2")(agent_input)
    agent_layer_3 = keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_3")(agent_layer_2)

    # Output layer
    outputs = keras.layers.Dense(units=2, activation='linear', name="outputs")(agent_layer_3)

    # Create the model
    model = keras.Model(inputs=agent_input, outputs=outputs)
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
