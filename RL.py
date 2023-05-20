from keras import Input, Model
from keras.layers import Dense, concatenate

from experience import experience
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time


class RL:

    def __init__(self, learning_rate, verbose, with_per, two_cars_local, log_directory, alternate_training, alternate_car):
        self.step_counter = 0
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        self.env_state = None
        self.c1_state = None
        self.c2_state = None
        self.c1_desire = np.array([10, 0])
        self.global_state = None
        self.local_network = self.init_network()
        self.two_cars_local = two_cars_local
        self.local_and_global_network = self.init_local_and_global_network()
        # experience replay
        self.experiences_size = 10000
        self.exp_batch_size = 10
        self.with_per = with_per  # per = prioritized experience replay (if 1 is on, if 0 off and use TD error only)
        self.experiences = experience.Experiences(self.experiences_size, self.with_per)
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        ###
        self.alternate_training = alternate_training
        if not alternate_training:
            self.alternate_training_network = None
        else:
            self.alternate_training_network = self.init_network()
        self.alternate_car = alternate_car  # The car which continues to train

        log_dir = log_directory+"/loss/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = tf.summary.create_file_writer(log_dir)
        self.train_global_counter = 0
        self.train_global_loss_sum = 0

    def init_local_and_global_network(self):
        # source of building the network:
        # https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

        # define the input of global network:
        input_global = Input(shape=(15,))
        # define the global network layers:
        x = Dense(16, activation="relu", name="global_16_layer")(input_global)
        x = Dense(8, activation="relu", name="global_8_layer")(x)
        x = Dense(5, activation="relu", name="global_5_layer")(x)
        x = Model(inputs=input_global, outputs=x)  # output of global

        input_local = Input(shape=(9,), name="input_local_9_layer")  # (x_car, y_car, v_car1, v_car2, up_car2,down_car2,right_car2,left_car2, dist_c1_c2)
        combined = concatenate([x.output, input_local])  # combine embedding of global & input of local

        z = Dense(16, activation="relu", name="local_16_layer")(combined)
        z = Dense(8, activation="relu", name="local_8_layer")(z)
        z = Dense(2, activation="linear", name="local_2_layer")(z)
        model = Model(inputs=[x.input, input_local], outputs=z)  # (q_value1, q_value2) = output of whole network

        # model.compile(self.opt, loss="mse")
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate), loss="mse")
        print(model.summary())
        return model

    def init_network(self):
        network = keras.Sequential([
            # keras.layers.InputLayer(input_shape=(4,)),  # (x_car, y_car, v_car1, dist_c1_c2)
            keras.layers.InputLayer(input_shape=(9,), name="input_9_layer"),  # (x_car, y_car, v_car1, v_car2, up_car2,down_car2,right_car2,left_car2, dist_c1_c2)
            keras.layers.Normalization(axis=-1),
            keras.layers.Dense(units=16, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), name="16_layer"),
            keras.layers.Dense(units=8, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), name="8_layer"),
            keras.layers.Dense(units=2, activation='linear', name="2_layer")  # (q_value1, q_value2)
        ])
        network.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate), loss="mse")

        return network

    def copy_network(self, network):
        return keras.models.clone_model(network)

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        td_error = 0
        if (self.with_per):
            q_val = self.local_network.predict(state, verbose=0)
            q_val_t = self.local_network.predict(new_state, verbose=0)
            next_best_action = np.argmax(q_val_t)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0][next_best_action]

        self.experiences.memorize(state, action, reward, done, new_state, td_error)

    def step_only_local(self, airsim_client, steps_counter):

        self.step_counter += 1
        # get state from environment of car1 and car2
        self.env_state = self.get_state_from_env(airsim_client)

        # update state of car1 (to be fed as input to the DNN):
        self.set_new_states()

        # update state of car2 (not relevant in this version of code): # has to be [[]] to enter as input to the DNN
        self.c2_state = np.array([[self.env_state["x_c2"], self.env_state["y_c2"], self.env_state["v_c2"]]])

        # Detect Collision:
        collision = self.detect_collision(airsim_client)
        if collision:
            return collision, None, None, -1000

        # sample action: for the first 10 steps make it go fast to avoid convergence by luck on first try
        action = self.sample_action_by_epsilon_greedy()
        target = self.local_network.predict(self.c1_state, verbose=self.verbose)

        # time.sleep(0.3)  # in seconds.
        # this makes sure that significant amount of time passed between state and new_state, for the DQN formula.
        # it seems that it is sufficient to not use sleep. (& it makes the finish intersection problematic)

        self.env_state = self.get_state_from_env(airsim_client)
        self.set_new_states()
        # get new state of car1 after performing the :
        new_state = self.c1_state
        reward, reached_target = self.calc_reward(collision)

        # add experience to experience replay:
        self.memorize(self.c1_state, action, reward, collision, new_state)

        # get experience from experience replay & train batch:
        if (self.step_counter % 50) == 0:
            print("exp replay")
            # states, action, reward, done, new_state, idx
            states, _, _, _, _, _ = self.experiences.sample(batch_size=self.exp_batch_size)
            predictions = []
            for state in list(states):
                predictions.append(self.local_network.predict(state))

            for i in range(0, self.exp_batch_size):
                self.local_network.fit(states[i], predictions[i])

        # update q values
        q_future = np.max(self.local_network.predict(new_state, verbose=self.verbose)[0])
        target[0][action] += self.learning_rate*(reward + (q_future * 0.95) - target[0][action])

        loss_local = self.local_network.fit(self.c1_state, target, epochs=1, verbose=0)
        print(f'Loss = {loss_local.history["loss"][-1]}')

        if not reached_target:
            with self.tensorboard.as_default():
                tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

        if (self.step_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay
            print(f"Epsilon = {self.epsilon}")

        # Set controls based action selected:
        current_controls = airsim_client.getCarControls()
        updated_controls = self.action_to_controls(current_controls, action)
        return collision, reached_target, updated_controls, reward

    def step_only_local_2_cars(self, airsim_client, steps_counter):

        self.step_counter += 1
        # get state from environment of car1 and car2
        self.env_state = self.get_state_from_env(airsim_client)

        # update state of car1 (to be fed as input to the DNN):
        self.set_new_states()

        # Detect Collision:
        collision = self.detect_collision(airsim_client)
        if collision:
            return collision, None, None, -1000

        # sample actions from network
        action_car1, action_car2 = self.sample_action_by_epsilon_greedy()

        print(self.alternate_car)

        if self.alternate_training:
            if self.alternate_car == 1:
                target = self.local_network.predict(self.c1_state, verbose=self.verbose)
                self.env_state = self.get_state_from_env(airsim_client)
                # get new state
                self.set_new_states()

                reward, reached_target = self.calc_reward(collision)

                # update q values - train the local network so it will continue to train.
                q_future = np.max(self.local_network.predict(self.c1_state, verbose=self.verbose)[0])
                target[0][action_car1] += self.learning_rate * (reward + (q_future * 0.95) - target[0][action_car1])

                loss_local = self.local_network.fit(self.c1_state, target, epochs=1, verbose=0)
                print(f'Loss = {loss_local.history["loss"][-1]}')

                if not reached_target:
                    with self.tensorboard.as_default():
                        tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

            if self.alternate_car == 2:
                target = self.local_network.predict(self.c2_state, verbose=self.verbose)
                self.env_state = self.get_state_from_env(airsim_client)
                self.set_new_states()

                reward, reached_target = self.calc_reward(collision)

                # update q values - train the local network so it will continue to train.
                q_future = np.max(self.local_network.predict(self.c2_state, verbose=self.verbose)[0])
                target[0][action_car2] += self.learning_rate * (reward + (q_future * 0.95) - target[0][action_car2])

                loss_local = self.local_network.fit(self.c2_state, target, epochs=1, verbose=0)
                print(f'Loss = {loss_local.history["loss"][-1]}')

                if not reached_target:
                    with self.tensorboard.as_default():
                        tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)



        if (self.step_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay
            print(f"Epsilon = {self.epsilon}")

        # Set controls based action selected:
        current_controls_car1 = airsim_client.getCarControls("Car1")
        updated_controls_car1 = self.action_to_controls(current_controls_car1, action_car1)

        current_controls_car2 = airsim_client.getCarControls("Car2")
        updated_controls_car2 = self.action_to_controls(current_controls_car2, action_car2)

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

    def step_with_global(self, airsim_client, steps_counter):

        self.step_counter += 1
        # get state from environment of car1 and car2
        self.env_state = self.get_state_from_env(airsim_client)

        self.set_new_states()

        # Detect Collision:
        collision = self.detect_collision(airsim_client)
        if collision:
            reward, reached_target = self.calc_reward(collision)
            return collision, None,None, None, reward

        # sample actions get target:
        action_c1, action_c2 = self.sample_action_global()
        target_c1 = self.local_and_global_network.predict([self.global_state, self.c1_state], verbose=self.verbose)
        print(f"Target Car1: {target_c1}")
        target_c2 = self.local_and_global_network.predict([self.global_state, self.c2_state], verbose=self.verbose)
        print(f"Target Car2: {target_c2}")

        self.env_state = self.get_state_from_env(airsim_client)
        self.set_new_states()

        reward, reached_target = self.calc_reward(collision)

        # update q values
        q_future_c1 = np.max(self.local_and_global_network.predict([self.global_state, self.c1_state], verbose=self.verbose)[0])
        target_c1[0][action_c1] += self.learning_rate * (reward + (q_future_c1 * 0.95) - target_c1[0][action_c1])
        q_future_c2 = np.max(self.local_and_global_network.predict([self.global_state, self.c2_state], verbose=self.verbose)[0])
        target_c2[0][action_c2] += self.learning_rate * (reward + (q_future_c2 * 0.95) - target_c2[0][action_c2])

        # compute loss:
        loss_local_c1 = self.local_and_global_network.fit([self.global_state, self.c1_state], target_c1, epochs=1, verbose=0)
        print(f'Loss Car1: {loss_local_c1.history["loss"][-1]}')
        loss_local_c2 = self.local_and_global_network.fit([self.global_state, self.c2_state], target_c2, epochs=1, verbose=0)
        print(f'Loss Car2: {loss_local_c2.history["loss"][-1]}')
        # average the loss of car1 and car2:
        average_loss = (loss_local_c1.history["loss"][-1] + loss_local_c2.history["loss"][-1]) / 2

        if not reached_target:
            with self.tensorboard.as_default():
                tf.summary.scalar('loss', average_loss, step=steps_counter)

        # Set controls based action selected:
        current_controls_c1 = airsim_client.getCarControls("Car1")
        updated_controls_c1 = self.action_to_controls(current_controls_c1, action_c1)
        current_controls_c2 = airsim_client.getCarControls("Car1")
        updated_controls_c2 = self.action_to_controls(current_controls_c2, action_c2)
        return collision, reached_target, updated_controls_c1, updated_controls_c2, reward

    def sample_action_by_epsilon_greedy(self):
        if not self.two_cars_local:
            if np.random.binomial(1, p=self.epsilon):
                # pick random action
                rand_action = np.random.randint(2, size=(1,1))[0][0]
                print(f"Random Action: {rand_action}")
                return rand_action
            else:
                # pick the action based on the highest q value
                action_selected = self.local_network.predict(self.c1_state, verbose=self.verbose).argmax()
                print(f"Selected Action: {action_selected}")
                return action_selected
        else:
            if np.random.binomial(1, p=self.epsilon):
                # pick random action
                rand_action = np.random.randint(2, size=(1,1))[0][0]
                print(f"Random Action: {rand_action}")
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
                        action_selected_car2 = self.alternate_training_network.predict(self.c2_state, verbose=self.verbose).argmax()
                        # print(f"Selected Action: {action_selected}")
                        return action_selected_car1, action_selected_car2
                    if self.alternate_car == 2:
                        action_selected_car1 = self.alternate_training_network.predict(self.c1_state, verbose=self.verbose).argmax()
                        action_selected_car2 = self.local_network.predict(self.c2_state, verbose=self.verbose).argmax()
                        # print(f"Selected Action: {action_selected}")
                        return action_selected_car1, action_selected_car2

    def sample_action_global(self):
        if np.random.binomial(1, p=self.epsilon):
            # pick random action
            rand_action_c1 = np.random.randint(2, size=(1, 1))[0][0]
            print(f"Random Action Car1: {rand_action_c1}")
            rand_action_c2 = np.random.randint(2, size=(1, 1))[0][0]
            print(f"Random Action Car2: {rand_action_c2}")
            return rand_action_c1, rand_action_c2
        else:
            # pick the action based on the highest q value
            action_selected_c1 = self.local_and_global_network.predict([self.global_state, self.c1_state],
                                                                    verbose=self.verbose).argmax()
            action_selected_c2 = self.local_and_global_network.predict([self.global_state, self.c2_state],
                                                                    verbose=self.verbose).argmax()
            print(f"Selected Action Car1: {action_selected_c1}")
            print(f"Selected Action Car2: {action_selected_c2}")
            return action_selected_c1, action_selected_c2

    def action_to_controls(self, current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls

    # TODO: Incorporate a reward mechanism for higher speeds
    def calc_reward(self, collision):
        # constant punish for every step taken, big punish for collision, big reward for reaching the target.
        # reward = 0
        reward = -0.1
        if collision:
            reward -= 1000
        # reward += self.env_state["dist_c1_c2"]**2

        if self.env_state["dist_c1_c2"] < 70:  # maybe the more punish- he wants to finish faster
            print("too close!!")
            reward -= 150

        if self.env_state["dist_c1_c2"] > 100:
            print("bonus!!")
            reward += 60


        reached_target = False
        # c1_dist_to_destination = np.sum(np.square(np.array([[self.c1_state[0][0], self.c1_state[0][1]]]) - self.c1_desire))
        # if c1_dist_to_destination <= 150:
        #     reward += 500 / c1_dist_to_destination


        if self.c1_state[0][0] > self.c1_desire[0]:  # if went over the desired
            reward += 1000
            reached_target = True
            print("Reached Target!!!")

        return reward, reached_target

    def get_state_from_env(self, airsim_client):

        env_state = {"x_c1": airsim_client.simGetObjectPose("Car1").position.x_val,
                     "y_c1": airsim_client.simGetObjectPose("Car1").position.y_val,
                     "v_c1": airsim_client.getCarState("Car1").speed,
                     "x_c2": airsim_client.simGetObjectPose("Car2").position.x_val,
                     "y_c2": airsim_client.simGetObjectPose("Car2").position.y_val,
                     "v_c2": airsim_client.getCarState("Car2").speed}

        env_state["dist_c1_c2"] = np.sum(np.square(
            np.array([[env_state["x_c1"], env_state["y_c1"]]]) - np.array([[env_state["x_c2"], env_state["y_c2"]]])))

        velocity_c2 = airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity
        # print("=================================")
        right = max(velocity_c2.x_val, 0)
        left = -1 * min(velocity_c2.x_val, 0)
        forward = max(velocity_c2.y_val, 0)
        backward = -1 * min(velocity_c2.y_val, 0)
        # print(right,left,forward,backward)
        # print("=================================")
        env_state["right_c2"] = right
        env_state["left_c2"] = left
        env_state["forward_c2"] = forward
        env_state["backward_c2"] = backward

        velocity_c1 = airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity
        # print("=================================")
        right = max(velocity_c1.x_val, 0)
        left = -1 * min(velocity_c1.x_val, 0)
        forward = max(velocity_c1.y_val, 0)
        backward = -1 * min(velocity_c1.y_val, 0)
        # print(right,left,forward,backward)
        # print("=================================")
        env_state["right_c1"] = right
        env_state["left_c1"] = left
        env_state["forward_c1"] = forward
        env_state["backward_c1"] = backward

        return env_state

    def detect_collision(self, airsim_client):
        collision_info = airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            print("Collided!")
            return True
        return False

    def set_new_states(self):
        self.c1_state = np.array([[self.env_state["x_c1"],
                                   self.env_state["y_c1"],
                                   self.env_state["v_c1"],
                                   self.env_state["v_c2"],
                                   self.env_state["dist_c1_c2"],
                                   self.env_state["right_c2"],
                                   self.env_state["left_c2"],
                                   self.env_state["forward_c2"],
                                   self.env_state["backward_c2"]
                                   ]])  # has to be [[]] to enter as input to the DNN

        self.c2_state = np.array([[self.env_state["x_c2"],
                                   self.env_state["y_c2"],
                                   self.env_state["v_c2"],
                                   self.env_state["v_c1"],
                                   self.env_state["dist_c1_c2"],
                                   self.env_state["right_c1"],
                                   self.env_state["left_c1"],
                                   self.env_state["forward_c1"],
                                   self.env_state["backward_c1"]
                                   ]])  # has to be [[]] to enter as input to the DNN

        self.global_state = np.array([[self.env_state["x_c1"],
                                       self.env_state["y_c1"],
                                       self.env_state["x_c2"],
                                       self.env_state["y_c2"],
                                       self.env_state["v_c1"],
                                       self.env_state["v_c2"],
                                       self.env_state["dist_c1_c2"],
                                       self.env_state["right_c1"],
                                       self.env_state["left_c1"],
                                       self.env_state["forward_c1"],
                                       self.env_state["backward_c1"],
                                       self.env_state["right_c2"],
                                       self.env_state["left_c2"],
                                       self.env_state["forward_c2"],
                                       self.env_state["backward_c2"]
                                       ]])  # has to be [[]] to enter as input to the DNN



