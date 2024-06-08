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
        # TODO: maybe delete batch_training_memory
        # self.batch_training_memory = []
        # TODO: what it is used for?
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

        # Put together master input
        master_input_as_array = [car1_state, car2_state]
        master_input_of_next_state = [car1_next_state, car2_next_state]

        # TODO: organize this
        return [[master_input_as_array, car1_state], [master_input_as_array, car2_state]], \
            [car1_action, car2_action], \
            [[master_input_of_next_state, car1_next_state], [master_input_of_next_state, car2_next_state]], \
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

        # Convert the trajectory into separate lists for each component
        states, actions, next_states, rewards = zip(*trajectory)

        # print("train_batch after zip:")
        # print(states)
        # print(actions)
        # print(rewards)
        # print(next_states)
        # print()

        master_inputs_of_all_trajectory = np.array([np.concatenate(state[0]) for state in states])
        agent_inputs_of_all_trajectory = np.array([state[1] for state in states])

        print(master_inputs_of_all_trajectory[0].shape)
        print(agent_inputs_of_all_trajectory[0].shape)

        current_state_full_input_of_all_trajectory = [master_inputs_of_all_trajectory, agent_inputs_of_all_trajectory]
        # TODO: add documantation of what predict function expects
        current_state_q_values = self.network.predict(current_state_full_input_of_all_trajectory, verbose=0)

        master_inputs_of_next_state_of_all_trajectory = np.array([np.concatenate(next_state[0]) for next_state in next_states])
        agent_inputs_of_next_state_of_all_trajectory = np.array([next_state[1] for next_state in next_states])
        next_state_full_input_of_all_trajectory = [master_inputs_of_next_state_of_all_trajectory, agent_inputs_of_next_state_of_all_trajectory]
        # TODO: add documantation of what predict function expects
        next_state_q_values = self.network.predict(next_state_full_input_of_all_trajectory, verbose=0)

        # Reshape the array
        # states = np.array(states).reshape(-1, *states[0].shape[2:])
        # actions = np.array(actions)
        # next_states = np.array(next_states).reshape(-1, *next_states[0].shape[1:])
        # rewards = np.array(rewards)

        # Predict Q-values for current and next states
        # current_q_values = self.network.predict(states, verbose=0)
        # next_q_values = self.network.predict(next_states, verbose=0)

        # Update Q-values using the DQN update rule for each step in the trajectory
        max_next_q_values = np.max(next_state_q_values, axis=1)
        targets = rewards + self.discount_factor * max_next_q_values
        for i, action in enumerate(actions):
            current_state_q_values[i][action] += LEARNING_RATE * (targets[i] - current_state_q_values[i][action])

        # Batch update the network
        # TODO: add documantation of what fit function expects
        # fit for batch training expects np.array of np.arrays in each of the inputs. ????
        # TODO: check that fit runs on whole batch, check what batch_size=len(states), and how it connects with batch size of trajectories
        loss = self.network.fit(current_state_full_input_of_all_trajectory, current_state_q_values, batch_size=len(states), verbose=0, epochs=1)

        print(loss)

        # TODO: should it return loss or somthing else?
        return loss

    def sample_action(self, car1_state, car2_state):

        if np.random.binomial(1, p=self.epsilon):  # epsilon greedy
            car1_random_action = np.random.randint(2, size=(1, 1))[0][0]
            car2_random_action = np.random.randint(2, size=(1, 1))[0][0]
            return car1_random_action, car2_random_action
        else:

            print(car1_state)
            master_input = np.concatenate((car1_state, car2_state))
            print(master_input)

            print(master_input.shape)
            print(car1_state.shape)

            print([master_input, car1_state])

            # TODO: check which one was useful and why (I think that reshaping master_input does nothing)
            master_input = np.reshape(master_input, (1, -1))
            car1_state = np.reshape(car1_state, (1, -1))
            car2_state = np.reshape(car2_state, (1, -1))

            # TODO: add documantation of what predict function expects
            car1_action = self.network.predict([master_input, car1_state], verbose=0).argmax()
            car2_action = self.network.predict([master_input, car2_state], verbose=0).argmax()
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
