import time
import numpy as np
import tensorflow as tf

from RL.config import LEARNING_RATE, CAR1_NAME, CAR2_NAME, COLLISION_REWARD, REACHED_TARGET_REWARD, STARVATION_REWARD, \
    NOT_KEEPING_SAFETY_DISTANCE_REWARD, KEEPING_SAFETY_DISTANCE_REWARD, SAFETY_DISTANCE_FOR_PUNISH, \
    SAFETY_DISTANCE_FOR_BONUS, BATCH_SIZE
from RL.utils.NN_utils import *


class RL:

    def __init__(self, logger, airsim):
        self.logger = logger
        self.airsim = airsim
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        self.discount_factor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.network = init_network(self.optimizer)
        # self.batch_training_memory = []
        self.batch_size = BATCH_SIZE

    def step(self):
        """
        Main Idea: get current state, sample action, get next state, get reward, detect collision_occurred or reached_target
        returns: state, action, next_state, collision_occurred, reached_target, reward
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
        time.sleep(0.4)

        # get next state
        car1_next_state = self.airsim.get_car1_state()
        car2_next_state = self.airsim.get_car2_state()

        # calculate reward
        collision_occurred = self.airsim.collision_occurred()
        reached_target = self.airsim.has_reached_target(car1_next_state)
        reward = self.calculate_reward(car1_next_state, collision_occurred, reached_target)

        # TODO: organize this shit
        # turn dictionary state into array state (for training)
        master_input_as_array = self.get_master_input_as_array(car1_state, car2_state)
        car1_state_as_array = self.get_agent_input_as_array(car1_state)
        car2_state_as_array = self.get_agent_input_as_array(car2_state)

        master_input_of_next_state_as_array = self.get_master_input_as_array(car1_next_state, car2_next_state)
        car1_next_state_as_array = self.get_agent_input_as_array(car1_next_state)
        car2_next_state_as_array = self.get_agent_input_as_array(car2_next_state)

        # TODO: organize this shit
        return [[master_input_as_array, car1_state_as_array], [master_input_as_array, car2_state_as_array]], \
            [car1_action, car2_action], \
            [[master_input_of_next_state_as_array, car1_next_state_as_array], [master_input_of_next_state_as_array, car2_next_state_as_array]], \
            collision_occurred, reached_target, reward

    # self.local_network_memory_buffer.append((np.array([list(cars_current_state_car1_perspective.values())]),
    #                                          action_car1, reward,
    #                                          np.array([list(cars_next_state_car1_perspective.values())])))
    # self.local_network_memory_buffer.append((np.array([list(cars_current_state_car2_perspective.values())]),
    #                                          action_car2, reward,
    #                                          np.array([list(cars_next_state_car2_perspective.values())])))
    # # Batch training every buffer_limit steps
    # if len(self.local_network_memory_buffer) > self.local_network_buffer_limit:
    #     loss_local = self.local_network_batch_train()
    #     self.logger.log('loss', loss_local.history["loss"][-1], steps_counter)
    #     self.local_network_memory_buffer.clear()  # Clear the buffer after training

    def train_batch(self, trajectory, train_step_or_trajectory):
        """
        train_step_or_trajectory = 'step' -> train on the last 2 items (these are the items that describe the last step)
        train_step_or_trajectory = 'trajectory' -> train on whole trajectory
        """

        # TODO: add training for step / training for trajectory - use: train_step_or_trajectory
        # TODO: batch size for trajectories

        # Convert the experiences in the buffer into separate lists for each component
        states, actions, next_states, rewards = zip(*trajectory)

        print("train_batch after zip:")
        print(states)
        print(actions)
        print(rewards)
        print(next_states)
        print()

        # Reshape the array
        # reshaped_states = states_array.reshape(-1, *states_array.shape[2:])
        states = np.array(states).reshape(-1, *states[0].shape[2:])
        actions = np.array(actions)
        next_states = np.array(next_states).reshape(-1, *next_states[0].shape[1:])
        rewards = np.array(rewards)

        # Predict Q-values for current and next states
        current_q_values = self.network.predict(states, verbose=0)
        next_q_values = self.network.predict(next_states, verbose=0)

        # Update Q-values using the DQN update rule for each experience in the batch
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + self.discount_factor * max_next_q_values
        for i, action in enumerate(actions):
            current_q_values[i][action] += LEARNING_RATE * (targets[i] - current_q_values[i][action])

        # Batch update the network
        # fit for batch training expects np.array of np.arrays in each of the inputs.
        loss = self.network.fit(states, current_q_values, batch_size=len(states), verbose=0, epochs=1)

        print(loss)

        return loss

    def sample_action(self, car1_state, car2_state):

        if np.random.binomial(1, p=self.epsilon):  # epsilon greedy
            car1_random_action = np.random.randint(2, size=(1, 1))[0][0]
            car2_random_action = np.random.randint(2, size=(1, 1))[0][0]
            return car1_random_action, car2_random_action
        else:
            # TODO: change that the states are always np arrays, and if I need specific value I use
            # TODO: a function that given a wanted value, uses a dict, that maps value to index in the array
            # TODO: x_c1: 0, y_c1: 1
            # TODO: and then pull the value from the np array and return it.
            master_input_as_array = self.get_master_input_as_array(car1_state, car2_state)
            car1_state_as_array = self.get_agent_input_as_array(car1_state)
            car2_state_as_array = self.get_agent_input_as_array(car2_state)
            car1_action = self.network.predict([master_input_as_array, car1_state_as_array], verbose=0).argmax()
            car2_action = self.network.predict([master_input_as_array, car2_state_as_array], verbose=0).argmax()
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
        if car1_state["distance_car1_car2"] < SAFETY_DISTANCE_FOR_PUNISH:
            reward = NOT_KEEPING_SAFETY_DISTANCE_REWARD

        # keeping safety distance
        if car1_state["distance_car1_car2"] > SAFETY_DISTANCE_FOR_BONUS:
            reward = KEEPING_SAFETY_DISTANCE_REWARD

        # reached target
        if reached_target:
            reward = REACHED_TARGET_REWARD

        # collision occurred
        if collision_occurred:
            reward = COLLISION_REWARD

        return reward

    @staticmethod
    def get_agent_input_as_array(car_state):
        return np.array([list(car_state.values())])

    @staticmethod
    def get_master_input_as_array(car1_state, car2_state):
        master_input = list(car1_state.values()) + list(car2_state.values())
        master_input = np.array([master_input])
        return master_input

    @staticmethod
    def action_to_controls(current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls
