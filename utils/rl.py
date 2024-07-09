import tensorflow as tf
from RL.config import LEARNING_RATE, CAR1_NAME, CAR2_NAME, COLLISION_REWARD, REACHED_TARGET_REWARD, STARVATION_REWARD, \
    NOT_KEEPING_SAFETY_DISTANCE_REWARD, KEEPING_SAFETY_DISTANCE_REWARD, SAFETY_DISTANCE_FOR_PUNISH, \
    SAFETY_DISTANCE_FOR_BONUS, EPSILON_DECAY, LOG_ACTIONS_SELECTED
from RL.utils.NN_utils import *
import numpy as np


class RL:

    def __init__(self, logger, airsim):
        self.logger = logger
        self.airsim = airsim
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        self.discount_factor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = EPSILON_DECAY
        self.network = init_network(self.optimizer)
        self.current_trajectory = []
        self.trajectories = []
        # self.batch_size = BATCH_SIZE_FOR_TRAJECTORY_BATCH  # relevant for train_batch_of_trajectories function
        self.freeze_master = False

    def step(self):
        """
        Main Idea: get current state, sample action, get next state, get reward, detect collision_occurred or reached_target
        returns: current_state, cars_actions, next_state, collision_occurred, reached_target, reward
        """

        # get current state
        car1_state = self.airsim.get_car1_state()
        car2_state = self.airsim.get_car2_state()

        # sample actions
        car1_action, car2_action = self.sample_action(car1_state, car2_state)

        # set updated controls according to sampled action
        self.set_controls_according_to_sampled_action(CAR1_NAME, car1_action)
        self.set_controls_according_to_sampled_action(CAR2_NAME, car2_action)

        # delay code in order to physically get the next state in the simulator
        # TODO: this sleep time effects how many times in an episode the speed changes (maybe log that)
        # time.sleep(0.1)

        # get next state
        car1_next_state = self.airsim.get_car1_state()
        car2_next_state = self.airsim.get_car2_state()

        # calculate reward
        collision_occurred = self.airsim.collision_occurred()
        reached_target = self.airsim.has_reached_target(car1_next_state)
        reward = self.calculate_reward(car1_next_state, collision_occurred, reached_target)

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

        # TODO: go over this function, compare with commented code
        # TODO: check that it works the same as code before.. compare the graients? or loss?

        if train_only_last_step:
            current_trajectory = self.current_trajectory[-2:]
        else:
            current_trajectory = self.current_trajectory

        states, actions, next_states, rewards = self.process_trajectory(current_trajectory)

        current_state_q_values = self.predict_q_values(states)
        next_state_q_values = self.predict_q_values(next_states)

        updated_q_values = self.update_q_values(actions, rewards, current_state_q_values, next_state_q_values)

        with tf.GradientTape(persistent=True) as tape:
            loss, gradients = self.apply_gradients(tape, updated_q_values, self.prepare_state_inputs(states))

        self.logger.log_weights_and_gradients(gradients, episode_counter, self.network)

        return loss.numpy().mean()

        # if train_only_last_step:
        #     current_trajectory = self.current_trajectory[-2:] # train on the last 2 items (these are the items that describe the last step)
        # else:
        #     current_trajectory = self.current_trajectory
        #
        # # Convert the trajectory into separate lists for each component
        # states, actions, next_states, rewards = zip(*current_trajectory)
        #
        # # Assemble current_state_of_all_trajectory + current_state_q_values_of_all_trajectory
        # TODO: add this line to the functions I built of predict
        # # predict expects: [array(x, 18),array(x, 9)]=[array of arrays of master input, array of arrays of agent input]
        # master_inputs_of_all_trajectory = np.array([np.concatenate(state[0]) for state in states])
        # agent_inputs_of_all_trajectory = np.array([state[1] for state in states])
        # current_state_of_all_trajectory = [master_inputs_of_all_trajectory, agent_inputs_of_all_trajectory]
        # current_state_q_values_of_all_trajectory = self.network.predict(current_state_of_all_trajectory, verbose=0)
        #
        # # Assemble next_state_of_all_trajectory + next_state_q_values_of_all_trajectory
        # # predict expects: [array(x, 18),array(x, 9)]=[array of arrays of master input, array of arrays of agent input]
        # master_inputs_of_next_state_of_all_trajectory = np.array([np.concatenate(next_state[0]) for next_state in next_states])
        # agent_inputs_of_next_state_of_all_trajectory = np.array([next_state[1] for next_state in next_states])
        # next_state_of_all_trajectory = [master_inputs_of_next_state_of_all_trajectory, agent_inputs_of_next_state_of_all_trajectory]
        # next_state_q_values_of_all_trajectory = self.network.predict(next_state_of_all_trajectory, verbose=0)
        #
        # # Update Q-values using the DQN update rule for each step in the trajectory
        # max_next_q_values = np.max(next_state_q_values_of_all_trajectory, axis=1)
        # targets = rewards + self.discount_factor * max_next_q_values
        # for i, action in enumerate(actions):
        #     current_state_q_values_of_all_trajectory[i][action] += LEARNING_RATE * (targets[i] - current_state_q_values_of_all_trajectory[i][action])
        #
        #
        # # TODO: organize this code - maybe move to another function? create one option with saving gradient and one without?
        # # TODO: make sure that the gradients are applied and the weights are changing
        # with tf.GradientTape() as tape:
        #     predictions = self.network(current_state_of_all_trajectory, training=True)
        #     loss = tf.keras.losses.mean_squared_error(current_state_q_values_of_all_trajectory, predictions)
        # gradients = tape.gradient(loss, self.network.trainable_variables)
        #
        # # print gradients
        # # for grad, var in zip(gradients, self.network.trainable_variables):
        # #     print(f"Gradient for {var.name}: {grad}")
        #
        # # Apply gradients
        # self.network.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        #
        # return loss.numpy().mean()

    @staticmethod
    def process_trajectory(trajectory):
        """ Convert the trajectory into separate lists for each component. """
        states, actions, next_states, rewards = zip(*trajectory)
        return states, actions, next_states, rewards

    @staticmethod
    def prepare_state_inputs(states):
        """ Assemble the master and agent inputs from states. """
        master_inputs = np.array([np.concatenate(state[0]) for state in states])
        agent_inputs = np.array([state[1] for state in states])
        return [master_inputs, agent_inputs]

    def predict_q_values(self, states):
        """ Predict Q-values for the given states. """
        inputs = self.prepare_state_inputs(states)
        return self.network.predict(inputs, verbose=0)

    def update_q_values(self, actions, rewards, current_q_values, next_q_values):
        """ Update Q-values using the DQN update rule for each step in the trajectory. """
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + self.discount_factor * max_next_q_values
        for i, action in enumerate(actions):
            current_q_values[i][action] += LEARNING_RATE * (targets[i] - current_q_values[i][action])
        return current_q_values  # these are the updated q-values

    def apply_gradients(self, tape, updated_q_values, current_state):
        """ Calculate and apply gradients to the network. """
        current_state_q_values = self.network(current_state, training=True)
        loss = tf.keras.losses.mean_squared_error(updated_q_values, current_state_q_values)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss, gradients

    def sample_action(self, car1_state, car2_state):

        if np.random.binomial(1, p=self.epsilon):  # epsilon greedy

            if LOG_ACTIONS_SELECTED:
                self.logger.log_actions_selected_random()

            car1_random_action = np.random.randint(2, size=(1, 1))[0][0]
            car2_random_action = np.random.randint(2, size=(1, 1))[0][0]
            return car1_random_action, car2_random_action
        else:
            master_input = np.concatenate((car1_state, car2_state), axis=0).reshape(1, -1)
            car1_state = np.reshape(car1_state, (1, -1))
            car2_state = np.reshape(car2_state, (1, -1))

            car1_action = self.network.predict([master_input, car1_state], verbose=0).argmax()
            car2_action = self.network.predict([master_input, car2_state], verbose=0).argmax()

            if LOG_ACTIONS_SELECTED:
                self.logger.log_actions_selected(self.network, car1_state, car2_state, car1_action, car2_action)

            return car1_action, car2_action

    def set_controls_according_to_sampled_action(self, car_name, sampled_action):
        current_controls = self.airsim.get_car_controls(car_name)
        updated_controls = self.action_to_controls(current_controls, sampled_action)
        self.airsim.set_car_controls(updated_controls, car_name)

    @staticmethod
    def calculate_reward(car1_state, collision_occurred, reached_target):

        # avoid starvation
        reward = STARVATION_REWARD

        # too close
        # TODO: change car1_state[-1] to be more generic
        if car1_state[-1] < SAFETY_DISTANCE_FOR_PUNISH:
            reward = NOT_KEEPING_SAFETY_DISTANCE_REWARD

        # keeping safety distance
        # TODO: change car1_state[-1] to be more generic
        if car1_state[-1] > SAFETY_DISTANCE_FOR_BONUS:
            reward = KEEPING_SAFETY_DISTANCE_REWARD

        # reached target
        if reached_target:
            reward = REACHED_TARGET_REWARD

        # collision occurred
        if collision_occurred:
            reward = COLLISION_REWARD

        return reward

    @staticmethod
    def action_to_controls(current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls



