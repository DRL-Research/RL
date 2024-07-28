import os
from tensorflow import keras
import numpy as np


class NN_handler:
    
    def __init__(self, config):
        self.config = config

    def init_network_master_and_agent(self, optimizer):
    
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
        model.compile(optimizer=optimizer, loss=self.config.LOSS_FUNCTION)
    
        return model

    def init_network_agent_only(self, optimizer):

        # Define agent_input
        agent_input = keras.Input(shape=(self.config.AGENT_INPUT_SIZE,), name="agent_input")

        # Normalization layer
        agent_input = keras.layers.BatchNormalization()(agent_input)

        # Agent layers
        agent_layer_2 = keras.layers.Dense(units=16, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_2")(agent_input)
        agent_layer_3 = keras.layers.Dense(units=8, activation='relu', kernel_initializer=keras.initializers.HeUniform(), name="agent_layer_3")(agent_layer_2)

        # Output layer
        outputs = keras.layers.Dense(units=2, activation='linear', name="outputs")(agent_layer_3)

        # Create the model
        model = keras.Model(inputs=agent_input, outputs=outputs)
        model.compile(optimizer=optimizer, loss=self.config.LOSS_FUNCTION)

        return model

    @staticmethod
    def alternate_master_and_agent_training(network, freeze_master):
        # Pay attention: input layers do not have weights.
        # Concat layers simply concatenates the outputs of the previous layers.
        # The Concatenate layer itself does not have any weights or parameters to be trained
        for layer in network.layers:
            if layer.name in ["master_layer_1"]:
                layer.trainable = not freeze_master
            else:  # agent_layer_2, agent_layer_3, outputs
                layer.trainable = freeze_master

    @staticmethod
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
    
    @staticmethod
    def create_network_copy(network):
        network_copy = keras.models.clone_model(network)
        network_copy.set_weights(network.get_weights())
        return network_copy

    def save_network_weights(self, network):
        # Create the directory if it doesn't exist
        save_dir = f"experiments/{self.config.EXPERIMENT_ID}/weights"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        # Save the weights to the specified directory
        save_path = f"{save_dir}/{self.config.WEIGHTS_TO_SAVE_NAME}.h5"
        network.save_weights(save_path)

    def load_weights_to_network(self, network):

        weight_directory = self.config.LOAD_WEIGHT_DIRECTORY

        if not os.path.exists(weight_directory):
            raise FileNotFoundError(f"Weight directory {weight_directory} does not exist.")

        network.load_weights(weight_directory)
        print("Weights were loaded successfully.")
        return network

    @staticmethod
    def are_weights_identical(model1, model2) -> bool:
        """Check if two Keras models have identical weights."""
        result = True
    
        # Ensure both models have the same number of layers
        if len(model1.layers) != len(model2.layers):
            result = False
    
        # Iterate over the layers of both models
        for layer1, layer2 in zip(model1.layers, model2.layers):
            # Ensure both layers have the same configuration
            if layer1.get_config() != layer2.get_config():
                result = False
    
            # Get weights of both layers
            weights1 = layer1.get_weights()
            weights2 = layer2.get_weights()
    
            # Check if the number of weight matrices are the same
            if len(weights1) != len(weights2):
                result = False
    
            # Check if all weight matrices are identical
            for w1, w2 in zip(weights1, weights2):
                if not np.array_equal(w1, w2):
                    result = False
    
        # check that copy process was done correctly
        if result:
            print(f"Successfully copied network car1 to network car2")
        else:
            print(f"Error in copying network car1 to network car2...")
    
        return result

