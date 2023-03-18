from keras import Input, Model
from keras.layers import Dense, concatenate

from experience import experience
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.compat.v1.disable_eager_execution() # if turned on - the old_logs does not work.
from keras.losses import categorical_crossentropy


class RL:

    def __init__(self, learning_rate, verbose, with_per):
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
        self.local_and_global_network = self.init_local_and_global_network()
        # experience replay
        self.experiences_size = 10000
        self.exp_batch_size = 10
        self.with_per = with_per  # per = prioritized experience replay (if 1 is on, if 0 off and use TD error only)
        self.experiences = experience.Experiences(self.experiences_size, self.with_per)
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99

        log_dir = "logs/loss/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = tf.summary.create_file_writer(log_dir)
        self.train_global_counter = 0
        self.train_global_loss_sum = 0

    def init_local_and_global_network(self):
        # source of building the network:
        # https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

        # define the input of global network:
        #input_global = Input(shape=(4,))  # (x_car1, y_car1, x_car2, y_car2) = input of global
        input_global = Input(shape=(9,))  # (x_car, y_car, v_car1, v_car2, up_car2,down_car2,right_car2,left_car2, dist_c1_c2)
        # define the global network layers:
        x = Dense(16, activation="relu")(input_global)
        x = Dense(8, activation="relu")(x)
        x = Dense(2, activation="relu")(x)
        x = Model(inputs=input_global, outputs=x)  # (emb1, emb2) = output of global

        #input_local = Input(shape=(4,))  # (x_car, y_car, v_car1, dist_c1_c2) = input of local
        input_local = Input(shape=(9,))  # (x_car, y_car, v_car1, v_car2, up_car2,down_car2,right_car2,left_car2, dist_c1_c2)
        combined = concatenate([x.output, input_local])  # combine embedding of global & input of local

        z = Dense(16, activation="relu")(combined)
        z = Dense(8, activation="relu")(z)
        z = Dense(2, activation="linear")(z)
        model = Model(inputs=[x.input, input_local], outputs=z)  # (q_value1, q_value2) = output of whole network

        # model.compile(self.opt, loss="mse")
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate), loss="mse")
        return model


    def init_network(self):
        network = keras.Sequential([
            #keras.layers.InputLayer(input_shape=(4,)),  # (x_car, y_car, v_car1, dist_c1_c2)
            keras.layers.InputLayer(input_shape=(9,)),  # (x_car, y_car, v_car1, v_car2, up_car2,down_car2,right_car2,left_car2, dist_c1_c2)
            keras.layers.Normalization(axis=-1),
            keras.layers.Dense(units=16, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=8, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=2, activation='linear')  # (q_value1, q_value2)
        ])
        network.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate), loss="mse")
        return network


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

        # update state of car2 (not relevant in this version of code): # has to be [[]] to enter as input to the DNN
        self.c2_state = np.array([[self.env_state["x_c2"], self.env_state["y_c2"], self.env_state["v_c2"]]])

        # Detect Collision:
        collision = False
        collision_info = airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            collision = True
            return collision, None, None, -1000

        # sample action: for the first 10 steps make it go fast to avoid convergence by luck on first try
        action = self.sample_action_by_epsilon_greedy()
        target = self.local_network.predict(self.c1_state, verbose=self.verbose)

        # time.sleep(0.3)  # in seconds.
        # this makes sure that significant amount of time passed between state and new_state, for the DQN formula.
        # it seems that it is sufficient to not use sleep. (& it makes the finish intersection problematic)

        self.env_state = self.get_state_from_env(airsim_client)
        # get new state of car1 after performing the :
        new_state = np.array([[self.env_state["x_c1"],
                              self.env_state["y_c1"],
                              self.env_state["v_c1"],
                              self.env_state["v_c2"],
                              self.env_state["dist_c1_c2"]
                              ,self.env_state["right"],
                              self.env_state["left"],
                              self.env_state["forward"],
                              self.env_state["backward"]
                               ]])
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
        print(loss_local.history["loss"][-1])

        if not reached_target:
            with self.tensorboard.as_default():
                tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

        if (self.step_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay
            print(self.epsilon)

        # Set controls based action selected:
        current_controls = airsim_client.getCarControls()
        updated_controls = self.action_to_controls(current_controls, action)
        return collision, reached_target, updated_controls, reward

    def step_with_global(self, airsim_client, steps_counter):

        self.step_counter += 1
        # get state from environment of car1 and car2
        self.env_state = self.get_state_from_env(airsim_client)

        # update state of car1 (to be fed as input to the DNN):
        self.c1_state = np.array([[self.env_state["x_c1"],
                                   self.env_state["y_c1"],
                                   self.env_state["v_c1"],
                                   self.env_state["dist_c1_c2"]
                                   ]])  # has to be [[]] to enter as input to the DNN

        self.global_state = np.array([[self.env_state["x_c1"],
                                       self.env_state["y_c1"],
                                       self.env_state["x_c2"],
                                       self.env_state["y_c2"]
                                       ]])  # has to be [[]] to enter as input to the DNN

        # Detect Collision:

        collision = False
        collision_info = airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            collision = True
            reward, reached_target = self.calc_reward(collision)
            return collision, None, None, reward

        # sample action:
        action = self.sample_action_global()
        target = self.local_and_global_network.predict([self.global_state, self.c1_state], verbose=self.verbose)
        print("target:")
        print(target)

        # time.sleep(0.3)  # in seconds.
        # this makes sure that significant amount of time passed between state and new_state, for the DQN formula.
        # it seems that it is sufficient to not use sleep. (& it makes the finish intersection problematic)

        self.env_state = self.get_state_from_env(airsim_client)
        # get new state of car1 after performing the :
        new_state = np.array([[self.env_state["x_c1"],
                               self.env_state["y_c1"],
                               self.env_state["v_c1"],
                               self.env_state["dist_c1_c2"]
                               ]])
        new_global = np.array([[self.env_state["x_c1"],
                                self.env_state["y_c1"],
                                self.env_state["x_c2"],
                                self.env_state["y_c2"]
                                ]])


        reward, reached_target = self.calc_reward(collision)
        # add experience to experience replay:
        self.memorize(self.c1_state, action, reward, collision, new_state)

        # get experience from experience replay:
        state, action, reward, done, new_state, idx = self.experiences.sample(batch_size=1)
        print("idx")
        print(idx)
        print()
        # if not using batch:
        state = state[0]
        action = action[0]
        reward = reward[0]
        done = done[0]
        new_state = new_state[0]

        # update q values
        q_future = np.max(self.local_and_global_network.predict([new_global, new_state], verbose=self.verbose)[0])
        target[0][action] += self.learning_rate * (reward + (q_future * 0.95) - target[0][action])

        loss_local = self.local_and_global_network.fit([self.global_state, self.c1_state], target, epochs=1, verbose=0)
        print(loss_local.history["loss"][-1])

        if not reached_target:
            with self.tensorboard.as_default():
                tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

        # Set controls based action selected:
        current_controls = airsim_client.getCarControls()
        updated_controls = self.action_to_controls(current_controls, action)
        return collision, reached_target, updated_controls, reward

        # # with tf.GradientTape() as tape:
        # #     pred = self.global_network(input)
        # #     print("omg:")
        # #     print(pred)
        # #     loss = categorical_crossentropy(np.array([[emb1,emb2]]), pred)
        # #     loss = tf.constant(0.05)
        # #     loss = tf.Variable(tf.constant(0.05))
        # #     print(type(loss))
        # #     print(type(tf.constant(0.05)))
        # # grads = tape.gradient(loss, self.global_network.trainable_variables)
        # # print(grads)
        # # self.opt.apply_gradients(zip(grads, self.global_network.trainable_variables))
        # # print("success!!")
        # # loss_global = self.global_network.fit(input, output_of_global, epochs=1, verbose=0)
        # # print("loss global: " + str(loss_global.history["loss"][-1]))
        # # updates = optimizer.get_updates(model1.trainable_weights, [], grads)
        #
        # # # every 10 steps - update the global network
        # # self.train_global_counter += 1
        # # self.train_global_loss_sum += loss
        # # if self.train_global_counter == 10:
        # #     self.train_global_counter = 0
        # #     self.train_global_loss_sum = 0
        # #     #
        # #     self.global_network.



    def sample_action_by_epsilon_greedy(self):
        if np.random.binomial(1, p=self.epsilon):
            # pick random action
            rand_action = np.random.randint(2, size=(1,1))[0][0]
            print(rand_action)
            return rand_action
        else:
            # pick the action based on the highest q value
            action_selected = self.local_network.predict(self.c1_state, verbose=self.verbose).argmax()
            print(action_selected)
            return action_selected

    def sample_action_global(self):
        if np.random.binomial(1, p=self.epsilon):
            # pick random action
            rand_action = np.random.randint(2, size=(1,1))[0][0]
            print(rand_action)
            return rand_action
        else:
            # pick the action based on the highest q value
            action_selected = self.local_and_global_network.predict([self.global_state,self.c1_state], verbose=self.verbose).argmax()
            print(action_selected)
            return action_selected

    # translate index of action to controls in car:
    def action_to_controls(self, current_controls, action):
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls

    # TODO: Icorporate a reward mechanism for higher speeds
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

        # print("=================================")
        velocity = airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity
        right = max(velocity.x_val, 0)
        left = -1 * min(velocity.x_val, 0)
        forward = max(velocity.y_val, 0)
        backward = -1 * min(velocity.y_val, 0)
        # print(right,left,forward,backward)
        # print("=================================")
        env_state["right"] = right
        env_state["left"] = left
        env_state["forward"] = forward
        env_state["backward"] = backward
        env_state["dist_c1_c2"] = np.sum(np.square(np.array([[env_state["x_c1"],env_state["y_c1"]]]) - np.array([[env_state["x_c2"],env_state["y_c2"]]])))
        return env_state

    def check_gradient_zero(self, model):
        return True

