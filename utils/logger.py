import tensorflow as tf


class Logger:

    def __init__(self, config):
        self.config = config
        self.rewards_and_losses_log_dir = f"experiments/{self.config.EXPERIMENT_ID}/{self.config.EXPERIMENT_DATE_TIME}/rewards_and_losses/"
        self.weights_and_gradients_log_dir = f"experiments/{self.config.EXPERIMENT_ID}/{config.EXPERIMENT_DATE_TIME}/weights_and_gradients/"
        self.rewards_and_losses_logger = tf.summary.create_file_writer(self.rewards_and_losses_log_dir)
        self.weights_and_gradients_logger = tf.summary.create_file_writer(self.weights_and_gradients_log_dir)
        self.same_action_selected_list = []

    def log_scaler(self, log_name, log_x_value, log_y_value):
        with self.rewards_and_losses_logger.as_default():
            tf.summary.scalar(log_name, step=log_x_value, data=log_y_value)

    def log_weights_and_gradients(self, gradients, episode_counter, network):
        """ Log gradients and weights to TensorBoard. """
        if episode_counter % self.config.LOG_WEIGHTS_AND_GRADIENTS_EVERY_X_EPISODES == 0:
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
        # network_agent_only = create_network_using_agent_only_from_original(network)
        # actions_from_agent_only = [network_agent_only.predict(np.reshape(state, (1, -1)), verbose=0).argmax()
        #                            for state in [car1_state, car2_state]]
        # print(f"Actions with agent only: {tuple(actions_from_agent_only)}")
        print(f"Actions with master:     {(car1_action_using_master, car2_action_using_master)}")
        print("-" * 10)

        if self.config.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY:
            if car1_action_using_master == car2_action_using_master:
                self.same_action_selected_list.append(True)
            else:
                self.same_action_selected_list.append(False)

    def log_same_action_selected_in_trajectory(self):
        if all(self.same_action_selected_list):
            print("All (non random) actions in this trajectory are the same.")
        else:
            print("not all (non random) actions in this trajectory are the same.")
        self.same_action_selected_list = []  # clear list before next trajectory

    def log_state(self, car_state, car_name):
        x, y, Vx, Vy = car_state[0], car_state[1], car_state[2], car_state[3]
        print(f"{car_name}_state: x={x:.2f}, y={y:.2f}, Vx={Vx:.2f}, Vy={Vy:.2f}")

    def log_q_values(self, q_values, car_name):
        print(f"{car_name}_q_values: {q_values}")
