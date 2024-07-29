import random
import time
from collections import deque

import tensorflow as tf
import numpy as np


class RL:

    def __init__(self, config, logger, airsim, nn_handler):
        self.config = config
        self.logger = logger
        self.airsim = airsim
        self.nn_handler = nn_handler
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.config.LEARNING_RATE)
        # self.discount_factor = 0.95
        if self.config.AGENT_ONLY:
            self.network = self.nn_handler.init_network_agent_only(self.optimizer)
        else:
            self.network = self.nn_handler.init_network_master_and_agent(self.optimizer) # TODO: change name network and network_car2
            self.network_car2 = self.nn_handler.create_network_copy(self.network)  # TODO: change name network and network_car2
        self.current_trajectory = []
        self.trajectories = []
        self.freeze_master = False
        ############################################################
        self.memory = deque(maxlen=100000)
        self.gamma = 0.999  # discount rate
        self.epsilon_min = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.TAU = 0.1
        self.train_start = 10  # from this size of memory we start to train
        self.ddqn = True
        self.Soft_Update = True
        self.distribution = True
        self.model = self.nn_handler.init_network_agent_only(self.optimizer)
        self.target_model = self.nn_handler.init_network_agent_only(self.optimizer)

    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(2)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        # minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))
        # trajectories = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        # trajectories = random.sample(self.memory, min(len(self.memory), len(self.memory)))
        # trajectories = list(self.memory)[-self.batch_size:]
        trajectories = list(self.memory)

        state, next_state, action, reward, done = [], [], [], [], []

        # for i in range(self.batch_size):
        #     # print(f"minibatch[{i}]: {minibatch[i]}")
        #     state.append(minibatch[i][0])
        #     action.append(minibatch[i][1])
        #     reward.append(minibatch[i][2])
        #     next_state.append(minibatch[i][3])
        #     done.append(minibatch[i][4])
        for trajectory in trajectories:
            for transition in trajectory:
                state.append(transition[0])
                action.append(transition[1])
                reward.append(transition[2])
                next_state.append(transition[3])
                done.append(transition[4])

        state = np.array(state)
        next_state = np.array(next_state)

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        # for i in range(len(minibatch)):
        #     # correction on the Q value for the action used
        #     if done[i]:
        #         target[i][action[i]] = reward[i]
        #     else:
        #         if self.ddqn:  # Double - DQN
        #             # current Q Network selects the action
        #             # a'_max = argmax_a' Q(s', a')
        #             a = np.argmax(target_next[i])
        #             # target Q Network evaluates the action
        #             # Q_max = Q_target(s', a'_max)
        #             target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
        #         else:  # Standard - DQN
        #             # DQN chooses the max Q value among next actions
        #             # selection and evaluation of action is on the target Q Network
        #             # Q_max = max_a' Q_target(s', a')
        #             target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        index = 0
        for trajectory in trajectories:
            for _ in trajectory:
                if done[index]:
                    target[index][action[index]] = reward[index]
                else:
                    if self.ddqn:  # Double - DQN
                        a = np.argmax(target_next[index])
                        target[index][action[index]] = reward[index] + self.gamma * target_val[index][a]
                    else:  # Standard - DQN
                        target[index][action[index]] = reward[index] + self.gamma * np.amax(target_next[index])
                index += 1

        # Train the Neural Network with batches
        # history = self.model.fit(state, target, epochs=1, batch_size=self.batch_size, verbose=0)
        # history = self.model.fit(state, target, epochs=1, batch_size=len(self.memory), verbose=0)
        print("self.config.EPOCHS")
        print(self.config.EPOCHS)
        history = self.model.fit(state, target, epochs=self.config.EPOCHS, verbose=0)

        return history.history['loss'][0]


    def updateEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################
    #############################################################################

    def step_agent_only(self):
        # get current state
        car1_state = self.airsim.get_car1_state(self.logger)

        # sample actions
        car1_action = self.sample_action_agent_only(car1_state)
        car2_action = self.config.CAR2_CONSTANT_ACTION
        print(f"car2 action: {self.config.CAR2_CONSTANT_ACTION}")

        if car1_action == car2_action:
            car1_state[1] = 1
        else:
            car1_state[1] = 0

        if self.config.LOG_CAR_STATES:
            self.logger.log_state(car1_state, self.config.CAR1_NAME)

        # set updated controls according to sampled action + car2 is constant speed
        self.set_controls_according_to_sampled_action(self.config.CAR1_NAME, car1_action)
        self.set_controls_according_to_sampled_action(self.config.CAR2_NAME, car2_action)

        # delay code in order to physically get the next state in the simulator
        time.sleep(self.config.TIME_BETWEEN_STEPS)

        # get next state
        car1_next_state = self.airsim.get_car1_state(self.logger)

        if car1_action == car2_action:
            car1_next_state[1] = 1
        else:
            car1_next_state[1] = 0

        # calculate reward
        collision_occurred = self.airsim.collision_occurred()
        reached_target = self.airsim.has_reached_target(car1_next_state)
        reward = self.calculate_reward(car1_next_state, collision_occurred, reached_target, car1_action, car2_action)

        # organize output
        # current_state = [[master_input, car1_state], [master_input, car2_state]]
        # cars_actions = [car1_action, car2_action]
        # next_state = [[master_input_of_next_state, car1_next_state], [master_input_of_next_state, car2_next_state]]

        return car1_state, car1_action, car1_next_state, collision_occurred, reached_target, reward

    def step(self):
        """
        Main Idea: get current state, sample action, get next state, get reward, detect collision_occurred or reached_target
        returns: current_state, cars_actions, next_state, collision_occurred, reached_target, reward
        """

        # get current state
        car1_state = self.airsim.get_car1_state(self.logger)
        car2_state = self.airsim.get_car2_state(self.logger)

        # sample actions
        car1_action, car2_action = self.sample_action(car1_state, car2_state)

        # set updated controls according to sampled action
        self.set_controls_according_to_sampled_action(self.config.CAR1_NAME, car1_action)
        self.set_controls_according_to_sampled_action(self.config.CAR2_NAME, car2_action)

        # delay code in order to physically get the next state in the simulator
        time.sleep(self.config.TIME_BETWEEN_STEPS)

        # get next state
        car1_next_state = self.airsim.get_car1_state(self.logger)
        car2_next_state = self.airsim.get_car2_state(self.logger)

        # calculate reward
        collision_occurred = self.airsim.collision_occurred()
        reached_target = self.airsim.has_reached_target(car1_next_state)
        reward = self.calculate_reward(car1_next_state, collision_occurred, reached_target, car1_action, car2_action)

        # Put together master input
        master_input = [car1_state, car2_state]
        master_input_of_next_state = [car1_next_state, car2_next_state]

        # organize output
        current_state = [[master_input, car1_state], [master_input, car2_state]]
        cars_actions = [car1_action, car2_action]
        next_state = [[master_input_of_next_state, car1_next_state], [master_input_of_next_state, car2_next_state]]

        return current_state, cars_actions, next_state, collision_occurred, reached_target, reward

    def train_trajectory(self, train_only_last_step, episode_counter):
        """
        train_only_last_step = True -> train on the last 2 items (these are the items that describe the last step)
        train_only_last_step = False -> train on whole trajectory
        """

        if train_only_last_step:
            current_trajectory = self.current_trajectory[-2:]
        else:
            current_trajectory = self.current_trajectory

        states, actions, next_states, rewards = self.process_trajectory(current_trajectory)

        current_state_q_values = self.predict_q_values_of_trajectory(states)
        next_state_q_values = self.predict_q_values_of_trajectory(next_states)

        updated_q_values = self.update_q_values(actions, rewards, current_state_q_values, next_state_q_values)

        with tf.GradientTape(persistent=True) as tape:
            loss, gradients = self.apply_gradients(tape, states, updated_q_values)
        self.logger.log_weights_and_gradients(gradients, episode_counter, self.network)

        return loss.numpy().mean()

    @staticmethod
    def process_trajectory(trajectory):
        """ Convert the trajectory into separate lists for each component. """
        states, actions, next_states, rewards = zip(*trajectory)
        return states, actions, next_states, rewards

    @staticmethod
    def prepare_state_inputs_agent_only(states):
        """ Assemble the master and agent inputs from states.
            output: [np.array(stacked arrays)] """
        agent_inputs = [np.array(np.vstack(states))]
        return agent_inputs

    @staticmethod
    def prepare_state_inputs(states, separate_state_for_each_car):
        """ Assemble the master and agent inputs from states.
            output: [np.array(stacked arrays)] """
        if separate_state_for_each_car:
            states_car1 = states[::2]
            states_car2 = states[1::2]
            master_inputs_car1 = np.array([np.concatenate(state[0]) for state in states_car1])
            master_inputs_car2 = np.array([np.concatenate(state[0]) for state in states_car2])
            agent_inputs_car1 = np.array([state[1] for state in states_car1])
            agent_inputs_car2 = np.array([state[1] for state in states_car2])
            return [master_inputs_car1, agent_inputs_car1], [master_inputs_car2, agent_inputs_car2]
        else:
            master_inputs = np.array([np.concatenate(state[0]) for state in states])
            agent_inputs = np.array([state[1] for state in states])
            return [master_inputs, agent_inputs]

    def predict_q_values_of_trajectory(self, states):
        """ Predict Q-values for the given states (according to the network of each car)
            for now, the code is only predicting and updating network of car1 (see commented code) """
        if self.config.AGENT_ONLY:
            car1_inputs = self.prepare_state_inputs_agent_only(states)
        else:
            car1_inputs, car2_inputs = self.prepare_state_inputs(states, separate_state_for_each_car=True)

        car1_q_values_of_trajectory = self.network.predict(car1_inputs, verbose=0)

        if self.config.LOG_Q_VALUES:
            self.logger.log_q_values(self.config.CAR1_NAME, car1_q_values_of_trajectory)

        """ This code is for calculating q_values for each car with different networks """
        # car1_q_values_of_trajectory = self.network.predict(car1_inputs, verbose=0)
        # car2_q_values_of_trajectory = self.network.predict(car2_inputs, verbose=0)
        # q_values_of_trajectory = np.empty((car1_q_values_of_trajectory.shape[0] +
        #                                    car2_q_values_of_trajectory.shape[0],
        #                                    car1_q_values_of_trajectory.shape[1]))
        # q_values_of_trajectory[::2] = car1_q_values_of_trajectory
        # q_values_of_trajectory[1::2] = car2_q_values_of_trajectory
        return car1_q_values_of_trajectory

    def update_q_values(self, actions, rewards, current_q_values, next_q_values):
        """ Update Q-values using the DQN update rule for each step in the trajectory. """
        if not self.config.AGENT_ONLY:
            actions = actions[::2]  # gather actions only from car1
            rewards = rewards[::2]  # gather rewards only from car1

        """ Update Q-values using the DQN update rule for each step in the trajectory. """
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + self.discount_factor * max_next_q_values
        for i, action in enumerate(actions):
            current_q_values[i][action] += self.config.LEARNING_RATE * (targets[i] - current_q_values[i][action])
        return current_q_values  # these are the updated q-values

        # # Calculate max Q-value for the next state
        # max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        # targets = rewards + self.discount_factor * max_next_q_values
        #
        # # Gather the Q-values corresponding to the taken actions
        # indices = tf.stack([tf.range(len(actions)), actions], axis=1)
        # gathered_q_values = tf.gather_nd(current_q_values, indices)
        #
        # # Update Q-values using the DQN update rule
        # updated_q_values = gathered_q_values + self.config.LEARNING_RATE * (targets - gathered_q_values)
        #
        # # Update the current Q-values tensor with the updated values
        # updated_q_values_tensor = tf.tensor_scatter_nd_update(current_q_values, indices, updated_q_values)
        # return updated_q_values_tensor

    def apply_gradients(self, tape, states, updated_q_values):
        """ Calculate and apply gradients to the network.
            Only compute loss and apply gradients on states & q_values of network_Car because this is the one
            that keeps training (network car2 is frozen)
        """
        if self.config.AGENT_ONLY:
            car1_inputs = self.prepare_state_inputs_agent_only(states)
        else:
            car1_inputs, car2_inputs = self.prepare_state_inputs(states, separate_state_for_each_car=True)
        current_state_q_values_car1 = self.network(car1_inputs, training=True)  # keep this format
        updated_q_values_car1 = updated_q_values

        loss = tf.keras.losses.mean_squared_error(current_state_q_values_car1, updated_q_values_car1)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss, gradients

    def sample_action(self, car1_state, car2_state):
        if np.random.binomial(1, p=self.epsilon):
            if self.config.LOG_ACTIONS_SELECTED:
                self.logger.log_actions_selected_random()
            return np.random.randint(2), np.random.randint(2)
        else:
            master_input = np.concatenate((car1_state, car2_state), axis=0).reshape(1, -1)
            car1_state = np.reshape(car1_state, (1, -1))
            car2_state = np.reshape(car2_state, (1, -1))
            car1_action = self.predict_q_values([master_input, car1_state], self.config.CAR1_NAME, self.network)
            car2_action = self.predict_q_values([master_input, car2_state], self.config.CAR2_NAME, self.network)
            if self.config.LOG_ACTIONS_SELECTED:
                self.logger.log_actions_selected(self.network, car1_state, car2_state, car1_action, car2_action)
            return car1_action, car2_action

    def sample_action_agent_only(self, car1_state):

        if np.random.binomial(1, p=self.epsilon) and not self.config.ONLY_INFERENCE:
            random_action = np.random.randint(2)
            if self.config.LOG_ACTIONS_SELECTED:
                self.logger.log_actions_selected_random(random_action)
            return random_action
        else:
            car1_state = np.reshape(car1_state, (1, -1))
            car1_action = self.predict_q_values(car1_state, self.config.CAR1_NAME, self.network)
            if self.config.LOG_ACTIONS_SELECTED:
                print(f"Action selected: {car1_action}")
            return car1_action

    def set_controls_according_to_sampled_action(self, car_name, sampled_action):
        current_controls = self.airsim.get_car_controls(car_name)
        updated_controls = self.action_to_controls(current_controls, sampled_action)
        self.airsim.set_car_controls(updated_controls, car_name)

    def calculate_reward(self, car1_state, collision_occurred, reached_target, car1_action, car2_action):

        x_car1 = car1_state[0]  # TODO: make it more generic
        cars_distance = car1_state[-1]  # TODO: make it more generic

        # avoid starvation
        reward = self.config.STARVATION_REWARD

        # # too close
        # # x_car1 < 2 is for not punishing after passing without collision (TODO: make it more generic)
        # if x_car1 < 2 and cars_distance < self.config.SAFETY_DISTANCE_FOR_PUNISH:
        #     # print(f"too close: {car1_state[-1]}")
        #     reward = self.config.NOT_KEEPING_SAFETY_DISTANCE_REWARD

        # # keeping safety distance
        # if cars_distance > self.config.SAFETY_DISTANCE_FOR_BONUS:
        #     # print(f"keeping safety distance: {car1_state[-1]}")
        #     reward = self.config.KEEPING_SAFETY_DISTANCE_REWARD

        # reached target
        if reached_target:
            reward = self.config.REACHED_TARGET_REWARD

        # collision occurred
        if collision_occurred:
            reward = self.config.COLLISION_REWARD

        # if self.config.AGENT_ONLY:
        #     if car1_action != car2_action:
        #         print("different actions")
        #         reward = 15
        #     else:
        #         reward = -10

        return reward

    @staticmethod
    def action_to_controls(current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.4
        elif action == 1:
            current_controls.throttle = 0.75
        return current_controls  # called current_controls - but it is updated controls

    def predict_q_values(self, car_input, car_name, car_network):
        q_values = self.network.predict(car_input, verbose=0)
        action_selected = q_values.argmax()
        return action_selected

    def copy_network(self):
        network_copy = self.nn_handler.create_network_copy(self.network)
        self.network_car2 = network_copy

        if self.config.LOG_WEIGHTS_ARE_IDENTICAL:
            self.nn_handler.are_weights_identical(self.network, self.network_car2)
