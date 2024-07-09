import tensorflow as tf
from config import LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES
import numpy as np

from utils.NN_utils import create_network_using_agent_only_from_original


class Logger:

    def __init__(self, experiment_id, experiment_date_time):
        self.rewards_and_losses_log_dir = f"experiments/{experiment_id}/{experiment_date_time}/rewards_and_losses/"
        self.weights_and_gradients_log_dir = f"experiments/{experiment_id}/{experiment_date_time}/weights_and_gradients/"
        self.rewards_and_losses_logger = tf.summary.create_file_writer(self.rewards_and_losses_log_dir)
        self.weights_and_gradients_logger = tf.summary.create_file_writer(self.weights_and_gradients_log_dir)

    def log_scaler(self, log_name, log_x_value, log_y_value):
        with self.rewards_and_losses_logger.as_default():
            tf.summary.scalar(log_name, step=log_x_value, data=log_y_value)

    def log_weights_and_gradients(self, gradients, episode_counter, network):
        """ Log gradients and weights to TensorBoard. """
        if episode_counter % LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES == 0:
            with self.weights_and_gradients_logger.as_default():

                # weights
                for var in network.trainable_variables:
                    if "bias" not in var.name:
                        tf.summary.histogram(var.name, var, step=episode_counter)
                # gradients
                for grad, var in zip(gradients, network.trainable_variables):
                    if "bias" not in var.name:
                        tf.summary.histogram(f'{var.name}_gradient', grad, step=episode_counter)

    def log_actions_selected_random(self):
        print("random action")
        print("-" * 10)

    def log_actions_selected(self, network, car1_state, car2_state, car1_action_using_master, car2_action_using_master):
        network_agent_only = create_network_using_agent_only_from_original(network)
        actions_from_agent_only = [network_agent_only.predict(np.reshape(state, (1, -1)), verbose=0).argmax()
                                   for state in [car1_state, car2_state]]
        print(f"Actions with agent only: {tuple(actions_from_agent_only)}")
        print(f"Actions with master:     {(car1_action_using_master, car2_action_using_master)}")
        print("-"*10)


