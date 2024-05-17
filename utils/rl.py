import time

import numpy as np
import tensorflow as tf
from RL.utils.environment_utils import *
from RL.utils.NN_utils import *
from RL.utils.replay_buffer_utils import *
from RL.config import LEARNING_RATE, CAR1_DESIRED_POSITION, CAR1_NAME, CAR2_NAME


class RLAgent:

    def __init__(self, logger):
        self.logger = logger
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        self.cars_state = None
        self.discount_factor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99

        self.network = init_network(self.opt)

        self.local_network_car1 = init_local_network(self.opt)
        self.local_network_car2 = copy_network(self.local_network_car1)
        self.local_network_memory_buffer = []  # Initialize an empty list to store experiences
        self.local_network_buffer_limit = 10  # Set the buffer size limit for batch training
        self.global_network, self.expected_reward_network = init_global_network(self.opt)
        self.global_network_memory_buffer = []
        self.global_network_buffer_limit = 10

    def step(self, airsim_client_handler, steps_counter):
        """
            1. Detect + Handle collision
            2. Get Input from environment (state of both cars)
            3. Sample action (for both cars)
            4.
        """

        # Detect Collision and handle consequences
        collision, collision_reward = airsim_client_handler.detect_and_handle_collision()
        if collision:
            return collision, None, None, None, collision_reward

        # get current state
        cars_current_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_current_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        # Sample actions and update targets
        action_car1, action_car2 = self.sample_action(airsim_client_handler.airsim_client)

        # delay code to let next state take effect
        time.sleep(0.4)

        # get next state
        cars_next_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_next_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        reward, reached_target = self.calc_reward(collision,
                                                  cars_next_state_car1_perspective)  # sent car1_perspective for distance between cars

        # store step (of both car perspective) in replay buffer for batch training
        self.local_network_memory_buffer.append((np.array([list(cars_current_state_car1_perspective.values())]),
                                                 action_car1, reward,
                                                 np.array([list(cars_next_state_car1_perspective.values())])))
        self.local_network_memory_buffer.append((np.array([list(cars_current_state_car2_perspective.values())]),
                                                 action_car2, reward,
                                                 np.array([list(cars_next_state_car2_perspective.values())])))

        # Epsilon decay for exploration-exploitation balance
        if (steps_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        # Translate actions to car controls
        updated_controls_car1 = self.get_updated_controls_according_to_action_selected(airsim_client_handler, CAR1_NAME, action_car1)
        updated_controls_car2 = self.get_updated_controls_according_to_action_selected(airsim_client_handler, CAR2_NAME, action_car2)

        # Batch training every buffer_limit steps
        if len(self.local_network_memory_buffer) > self.local_network_buffer_limit:
            loss_local = self.local_network_batch_train()
            if not reached_target:
                self.logger.log('loss', loss_local.history["loss"][-1], steps_counter)

            self.local_network_memory_buffer.clear()  # Clear the buffer after training

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

    def step_local_2_cars(self, airsim_client_handler, steps_counter):
        """
        This function describes one step in the RL algorithm:
            current_state -> action -> (small delay in code for car movement) -> next_state -> reward
            -> enter to batch training memory
        """

        # Detect Collision and handle consequences
        collision, collision_reward = airsim_client_handler.detect_and_handle_collision()
        if collision:
            return collision, None, None, None, collision_reward

        # get current state
        cars_current_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_current_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        # Sample actions and update targets
        action_car1, action_car2 = self.sample_action(airsim_client_handler.airsim_client)

        # delay code to let next state take effect
        time.sleep(0.4)

        # get next state
        cars_next_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_next_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        reward, reached_target = self.calc_reward(collision, cars_next_state_car1_perspective)  # sent car1_perspective for distance between cars

        # store step (of both car perspective) in replay buffer for batch training
        self.local_network_memory_buffer.append((np.array([list(cars_current_state_car1_perspective.values())]), action_car1, reward, np.array([list(cars_next_state_car1_perspective.values())])))
        self.local_network_memory_buffer.append((np.array([list(cars_current_state_car2_perspective.values())]), action_car2, reward, np.array([list(cars_next_state_car2_perspective.values())])))

        # Epsilon decay for exploration-exploitation balance
        if (steps_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        # Translate actions to car controls
        updated_controls_car1 = self.get_updated_controls_according_to_action_selected(airsim_client_handler, CAR1_NAME, action_car1)
        updated_controls_car2 = self.get_updated_controls_according_to_action_selected(airsim_client_handler, CAR2_NAME, action_car2)

        # Batch training every buffer_limit steps
        if len(self.local_network_memory_buffer) > self.local_network_buffer_limit:
            loss_local = self.local_network_batch_train()
            if not reached_target:
                self.logger.log('loss', loss_local.history["loss"][-1], steps_counter)

            self.local_network_memory_buffer.clear()  # Clear the buffer after training

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

    def step_global_2_cars(self, airsim_client_handler, steps_counter):
        """
        This function describes one step in the RL algorithm:
            current_state -> action -> (small delay in code for car movement) -> next_state -> reward
            -> enter to batch training memory
        """

        # Detect Collision and handle consequences
        collision, collision_reward = airsim_client_handler.detect_and_handle_collision()
        if collision:
            return collision, None, None, None, collision_reward

        # get current state
        cars_current_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_current_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        # get proto-plans from global network
        # we only pass cars_current_state_car1_perspective because it holds the state of car1 and car2.
        proto_plan_1, proto_plan_2, input_for_global_network = self.get_proto_plans_and_reward_from_global_network(cars_current_state_car1_perspective)

        # Sample actions and update targets
        action_car1, action_car2 = self.sample_action(airsim_client_handler.airsim_client, proto_plan_1, proto_plan_2)

        # delay code to let next state take effect
        time.sleep(0.4)

        # get next state
        cars_next_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_next_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        reward, reached_target = self.calc_reward(collision, cars_next_state_car1_perspective)  # sent car1_perspective for distance between cars

        # batch train the global network (via reward and expected reward)
        self.global_network_memory_buffer.append((np.array([input_for_global_network]), np.array([reward])))
        if len(self.global_network_memory_buffer) > self.global_network_buffer_limit:
            loss_global = self.global_network_batch_train()
            if not reached_target:
                self.logger.log('global_network loss', loss_global.history["loss"][-1], steps_counter)

            self.global_network_memory_buffer.clear()  # Clear the buffer after training

        # add proto-action to current state + next state.
        full_current_state_car1_perspective = self.combine_cars_current_state_and_proto_action(cars_current_state_car1_perspective, proto_plan_1)
        full_current_state_car2_perspective = self.combine_cars_current_state_and_proto_action(cars_current_state_car2_perspective, proto_plan_2)
        full_next_state_car1_perspective = self.combine_cars_current_state_and_proto_action(cars_next_state_car1_perspective, proto_plan_1)
        full_next_state_car2_perspective = self.combine_cars_current_state_and_proto_action(cars_next_state_car2_perspective, proto_plan_2)

        # store step (of both car perspective) in replay buffer for batch training
        self.local_network_memory_buffer.append((full_current_state_car1_perspective, action_car1, reward, full_next_state_car1_perspective))
        self.local_network_memory_buffer.append((full_current_state_car2_perspective, action_car2, reward, full_next_state_car2_perspective))

        # Epsilon decay for exploration-exploitation balance
        if (steps_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        # Translate actions to car controls
        current_controls_car1 = airsim_client_handler.airsim_client.getCarControls(CAR1_NAME)
        updated_controls_car1 = self.action_to_controls(current_controls_car1, action_car1)

        current_controls_car2 = airsim_client_handler.airsim_client.getCarControls(CAR2_NAME)
        updated_controls_car2 = self.action_to_controls(current_controls_car2, action_car2)

        # Batch training every buffer_limit steps
        if len(self.local_network_memory_buffer) > self.local_network_buffer_limit:
            loss_local = self.local_network_batch_train()
            if not reached_target:
                self.logger.log('local_network loss', loss_local.history["loss"][-1], steps_counter)

            self.local_network_memory_buffer.clear()  # Clear the buffer after training

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

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

    def sample_action(self, airsim_client, proto_plan_1, proto_plan_2):

        # epsilon greedy
        if np.random.binomial(1, p=self.epsilon):
            rand_action = np.random.randint(2, size=(1, 1))[0][0]
            return rand_action, rand_action
        else:
            # both networks receive the same input (but with different perspectives (meaning-> different order))
            # Another difference is, that there are 2 versions of the network.
            cars_state_car1_perspective = get_local_input_car1_perspective(airsim_client)
            full_input_for_local_network_car_1_perspective = self.combine_cars_current_state_and_proto_action(cars_state_car1_perspective, proto_plan_1)
            action_selected_car1 = self.local_network_car1.predict(full_input_for_local_network_car_1_perspective, verbose=0).argmax()

            cars_state_car2_perspective = get_local_input_car2_perspective(airsim_client)
            full_input_for_local_network_car_2_perspective = self.combine_cars_current_state_and_proto_action(cars_state_car2_perspective, proto_plan_2)
            action_selected_car2 = self.local_network_car2.predict(full_input_for_local_network_car_2_perspective, verbose=0).argmax()
            return action_selected_car1, action_selected_car2

    @staticmethod
    def calc_reward(self, collision, cars_next_state_car1_perspective):

        # avoid starvation
        reward = -0.1

        # collision
        if collision:
            reward -= 1000

        # too close
        if cars_next_state_car1_perspective["dist_c1_c2"] < 70:
            reward -= 150

        # keeping safety distance
        if cars_next_state_car1_perspective["dist_c1_c2"] > 100:
            reward += 60

        reached_target = False

        # reached target
        if cars_next_state_car1_perspective['x_c1'] > CAR1_DESIRED_POSITION[0]:
            reward += 1000
            reached_target = True

        return reward, reached_target

    def get_proto_plans_and_reward_from_global_network(self, cars_current_state_car1_perspective) -> (np.array((1, 5)), np.array((1, 5)), float):
        input_for_global_network_before_proto_plan1 = np.array([cars_current_state_car1_perspective["x_c1"],
                                                                cars_current_state_car1_perspective["y_c1"],
                                                                cars_current_state_car1_perspective["Vx_c1"],
                                                                cars_current_state_car1_perspective["Vy_c1"],
                                                                -1, -1, -1, -1, -1,
                                                                cars_current_state_car1_perspective["x_c2"],
                                                                cars_current_state_car1_perspective["y_c2"],
                                                                cars_current_state_car1_perspective["Vx_c2"],
                                                                cars_current_state_car1_perspective["Vy_c2"],
                                                                -1, -1, -1, -1, -1,
                                                                cars_current_state_car1_perspective["dist_c1_c2"]])
        proto_plan_1 = self.global_network.predict(input_for_global_network_before_proto_plan1, verbose=0)

        input_for_global_network_with_proto_plan1 = np.array([cars_current_state_car1_perspective["x_c1"],
                                                             cars_current_state_car1_perspective["y_c1"],
                                                             cars_current_state_car1_perspective["Vx_c1"],
                                                             cars_current_state_car1_perspective["Vy_c1"],
                                                             proto_plan_1[0][0], proto_plan_1[0][1], proto_plan_1[0][2],
                                                             proto_plan_1[0][3], proto_plan_1[0][4],
                                                             cars_current_state_car1_perspective["x_c2"],
                                                             cars_current_state_car1_perspective["y_c2"],
                                                             cars_current_state_car1_perspective["Vx_c2"],
                                                             cars_current_state_car1_perspective["Vy_c2"],
                                                             -1, -1, -1, -1, -1,
                                                             cars_current_state_car1_perspective["dist_c1_c2"]])

        proto_plan_2 = self.global_network.predict(input_for_global_network_with_proto_plan1, verbose=0)

        # returning input_for_global_network_with_proto_plan1 for training the global network.
        return proto_plan_1, proto_plan_2, input_for_global_network_with_proto_plan1

    @ staticmethod
    def combine_cars_current_state_and_proto_action(current_state, proto_action) -> np.array((1, 14)):
        full_current_state = list(current_state.values()) + list(proto_action[0])
        full_current_state = np.array([full_current_state])
        return full_current_state

    @ staticmethod
    def action_to_controls(current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls

    def get_updated_controls_according_to_action_selected(self, airsim_client_handler, car_name, action_selected):
        current_controls = airsim_client_handler.airsim_client.getCarControls(car_name)
        updated_controls = self.action_to_controls(current_controls, action_selected)
        return updated_controls
