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
        self.batch_training_memory = []
        self.batch_size = BATCH_SIZE

    def step(self, steps_counter):
        """
        Main Idea: get current state, sample action, get next state, get reward, train the network
        returns: collision_occurred, reached_target, reward
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

        #


        self.batch_training_memory.append(
            self.get_agent_input_as_array()
        )

        # TODO: organize batch train (if needed...) - divide to functions not in step function
        # batch training (update weights according to DQN formula)
        # store step (in replay buffer) for batch training
        self.local_network_memory_buffer.append((np.array([list(cars_current_state_car1_perspective.values())]),
                                                 action_car1, reward,
                                                 np.array([list(cars_next_state_car1_perspective.values())])))
        self.local_network_memory_buffer.append((np.array([list(cars_current_state_car2_perspective.values())]),
                                                 action_car2, reward,
                                                 np.array([list(cars_next_state_car2_perspective.values())])))
        # Batch training every buffer_limit steps
        if len(self.local_network_memory_buffer) > self.local_network_buffer_limit:
            loss_local = self.local_network_batch_train()
            self.logger.log('loss', loss_local.history["loss"][-1], steps_counter)
            self.local_network_memory_buffer.clear()  # Clear the buffer after training

        # Epsilon decay
        if (steps_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        return collision_occurred, reached_target, reward

    def local_network_batch_train(self):
        if not self.local_network_memory_buffer:
            return

        # Convert the experiences in the buffer into separate lists for each component
        states, actions, rewards, next_states = zip(*self.local_network_memory_buffer)

        # Convert lists to numpy arrays for batch processing
        states = np.array(states).reshape(-1, *states[0].shape[1:])
        next_states = np.array(next_states).reshape(-1, *next_states[0].shape[1:])
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Predict Q-values for current and next states
        current_q_values = self.local_network_car1.predict(states, verbose=0)
        next_q_values = self.local_network_car1.predict(next_states, verbose=0)

        # Update Q-values using the DQN update rule for each experience in the batch
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + self.discount_factor * max_next_q_values
        for i, action in enumerate(actions):
            current_q_values[i][action] += LEARNING_RATE * (targets[i] - current_q_values[i][action])

        # Batch update the network
        # fit for batch training expects np.array of np.arrays in each of the inputs.
        loss_local = self.local_network_car1.fit(states, current_q_values, batch_size=len(states), verbose=0, epochs=1)

        return loss_local

    def global_network_batch_train(self):
        input_for_global_network, expected_rewards = zip(*self.global_network_memory_buffer)
        input_for_global_network = np.array(input_for_global_network).reshape(-1, *input_for_global_network[0].shape[1:])
        expected_rewards = np.array(expected_rewards)
        # fit for batch training expects np.array of np.arrays in each of the inputs.
        print(f"predicted values: {self.expected_reward_network.predict(input_for_global_network)}")
        print(f"expected values: {expected_rewards}")
        global_loss = self.expected_reward_network.fit(input_for_global_network, expected_rewards,
                                                       batch_size=len(input_for_global_network), verbose=0, epochs=1)
        print(f"loss is: {global_loss}")
        print()
        return global_loss

    def sample_action(self, car1_state, car2_state):

        if np.random.binomial(1, p=self.epsilon):  # epsilon greedy
            car1_random_action = np.random.randint(2, size=(1, 1))[0][0]
            car2_random_action = np.random.randint(2, size=(1, 1))[0][0]
            return car1_random_action, car2_random_action
        else:
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
