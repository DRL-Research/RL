
import numpy as np
from tensorflow import keras


def check_effect_of_master_on_selected_actions(network, car1_state, car2_state, car1_action_from_master, car2_action_from_master):
    agent_only_model = create_agent_model_from_original(network)
    car1_state = np.reshape(car1_state, (1, -1))
    car2_state = np.reshape(car2_state, (1, -1))
    car1_action_from_agent_only = agent_only_model.predict([car1_state], verbose=0).argmax()
    car2_action_from_agent_only = agent_only_model.predict([car2_state], verbose=0).argmax()
    print(f"Actions with agent only: {(car1_action_from_agent_only, car2_action_from_agent_only)}")
    #
    print(f"Actions with master:     {(car1_action_from_master, car2_action_from_master)}")
    print()


def create_agent_model_from_original(original_model):

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
