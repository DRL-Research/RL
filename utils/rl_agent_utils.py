import tensorflow as tf
from RL.utils.environment_utils import *
from RL.utils.NN_utils import *
from RL.utils.airsim_utils import *


class RLAgent:

    def __init__(self, learning_rate, verbose, experiment_id, alternate_training, alternate_car, current_date_time, tensorboard):
        self.step_counter = 0
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        self.env_state = None
        self.c1_state = None
        self.c2_state = None
        self.c1_desire = np.array([10, 0])
        self.local_network = init_local_network(self.learning_rate)
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        ##
        self.alternate_training = alternate_training
        if not alternate_training:
            self.alternate_training_network = None
        else:
            self.alternate_training_network = init_local_network(self.learning_rate)
        self.alternate_car = alternate_car  # The car which continues to train
        # log_dir_for_tensorboard = "experiments/" + experiment_id + "/loss/" + current_date_time
        # self.tensorboard = tf.summary.create_file_writer(log_dir_for_tensorboard)
        self.tensorboard = tensorboard

    def update_targets_and_train(self, airsim_client, action_car1, action_car2, steps_counter, collision):

        if self.alternate_training:
            if self.alternate_car == 1:
                target = self.local_network.predict(self.c1_state, verbose=self.verbose)
                self.env_state = get_env_state(airsim_client, "Car1")
                # get new state

                # TODO: input for local: x_c1, y_c1, x_c2, y_c2, Vx_c1, Vy_c1, Vx_c2, Vy_c2, dist (for both cars)
                # TODO: input for global: same but reward function is considering all cars.

                self.c1_state = np.array([[self.env_state["x_c1"],
                                           self.env_state["y_c1"],
                                           self.env_state["v_c1"],
                                           self.env_state["v_c2"],
                                           self.env_state["dist_c1_c2"],
                                           self.env_state["right"],
                                           self.env_state["left"],
                                           self.env_state["forward"],
                                           self.env_state["backward"]
                                           ]])  # has to be [[]] to enter as input to the DNN

                reward, reached_target = self.calc_reward(collision)

                # update q values - train the local network so it will continue to train.
                q_future = np.max(self.local_network.predict(self.c1_state, verbose=self.verbose)[0])
                target[0][action_car1] += self.learning_rate * (reward + (q_future * 0.95) - target[0][action_car1])

                loss_local = self.local_network.fit(self.c1_state, target, epochs=1, verbose=0)
                # print(f'Loss = {loss_local.history["loss"][-1]}')

                if not reached_target:
                    with self.tensorboard.as_default():
                        tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

            if self.alternate_car == 2:
                target = self.local_network.predict(self.c2_state, verbose=self.verbose)
                self.env_state = get_env_state(airsim_client, "Car2")
                self.c2_state = np.array([[self.env_state["x_c2"],
                                           self.env_state["y_c2"],
                                           self.env_state["v_c2"],
                                           self.env_state["v_c1"],
                                           self.env_state["dist_c1_c2"],
                                           self.env_state["right"],
                                           self.env_state["left"],
                                           self.env_state["forward"],
                                           self.env_state["backward"]
                                           ]])  # has to be [[]] to enter as input to the DNN

                reward, reached_target = self.calc_reward(collision)

                # update q values - train the local network, so it will continue to train.
                q_future = np.max(self.local_network.predict(self.c2_state, verbose=self.verbose)[0])
                target[0][action_car2] += self.learning_rate * (reward + (q_future * 0.95) - target[0][action_car2])

                loss_local = self.local_network.fit(self.c2_state, target, epochs=1, verbose=0)
                # print(f'Loss = {loss_local.history["loss"][-1]}')

                if not reached_target:
                    with self.tensorboard.as_default():
                        tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

        return reward, reached_target

    def step_local_2_cars(self, airsim_client, steps_counter):
        """
        Perform a step for two cars in the local environment.

        Args:
            airsim_client: The AirSim client object for interaction with the environment.
            steps_counter: The current count of steps.

        Returns:
            A tuple containing collision status, whether the target is reached, updated controls for both cars, and the reward.
        """

        self.step_counter += 1

        # Process state and control for each car
        car1_info = process_car_state_and_control(airsim_client, "Car1")
        car2_info = process_car_state_and_control(airsim_client, "Car2")
        self.c1_state, self.c2_state = car1_info['state'], car2_info['state']

        # Detect Collision and handle consequences
        collision, collision_reward = detect_and_handle_collision(airsim_client)
        if collision:
            return collision, None, None, None, collision_reward

        # Sample actions and update targets based on alternate training flag
        action_car1, action_car2 = self.sample_action()
        reward, reached_target = self.update_targets_and_train(airsim_client, action_car1, action_car2, steps_counter, collision)

        # Epsilon decay for exploration-exploitation balance
        if (self.step_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        # Translate actions to car controls
        current_controls_car1 = airsim_client.getCarControls("Car1")
        updated_controls_car1 = self.action_to_controls(current_controls_car1, action_car1)

        current_controls_car2 = airsim_client.getCarControls("Car2")
        updated_controls_car2 = self.action_to_controls(current_controls_car2, action_car2)

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

    def sample_action(self):
        if np.random.binomial(1, p=self.epsilon):
            rand_action = np.random.randint(2, size=(1, 1))[0][0]
            return rand_action, rand_action
        else:
            if not self.alternate_training:
                # pick the action based on the highest q value
                action_selected_car1 = self.local_network.predict(self.c1_state, verbose=self.verbose).argmax()
                action_selected_car2 = self.local_network.predict(self.c2_state, verbose=self.verbose).argmax()
                # print(f"Selected Action: {action_selected}")
                return action_selected_car1, action_selected_car2
            else:
                if self.alternate_car == 1:
                    action_selected_car1 = self.local_network.predict(self.c1_state, verbose=self.verbose).argmax()
                    action_selected_car2 = self.alternate_training_network.predict(self.c2_state,
                                                                                   verbose=self.verbose).argmax()
                    # print(f"Selected Action: {action_selected}")
                    return action_selected_car1, action_selected_car2
                if self.alternate_car == 2:
                    action_selected_car1 = self.alternate_training_network.predict(self.c1_state,
                                                                                   verbose=self.verbose).argmax()
                    action_selected_car2 = self.local_network.predict(self.c2_state, verbose=self.verbose).argmax()
                    # print(f"Selected Action: {action_selected}")
                    return action_selected_car1, action_selected_car2

    def action_to_controls(self, current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls

    def calc_reward(self, collision):
        reward = -0.1
        if collision:
            reward -= 1000

        if self.env_state["dist_c1_c2"] < 70:
            reward -= 150

        if self.env_state["dist_c1_c2"] > 100:
            reward += 60

        reached_target = False

        if self.c1_state[0][0] > self.c1_desire[0]:
            reward += 1000
            reached_target = True

        return reward, reached_target

