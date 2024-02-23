import tensorflow as tf
from RL.utils.environment_utils import *
from RL.utils.NN_utils import *
from RL.utils.airsim_utils import *
from RL.utils.replay_buffer_utils import *


class RLAgent:

    def __init__(self, learning_rate, verbose, tensorboard):
        self.gil = "gil"
        self.step_counter = 0
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        self.cars_state = None
        self.c1_desire = np.array([10, 0])
        self.discount_factor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.tensorboard = tensorboard
        self.local_network_car1 = init_local_network(self.opt)
        self.local_network_car2 = copy_network(self.local_network_car1)
        self.memory_buffer = []  # Initialize an empty list to store experiences
        self.buffer_limit = 10  # Set the buffer size limit for batch training

    def step_local_2_cars(self, airsim_client, steps_counter):

        self.step_counter += 1

        self.cars_state = get_cars_state(airsim_client)

        # Detect Collision and handle consequences
        collision, collision_reward = detect_and_handle_collision(airsim_client)
        if collision:
            return collision, None, None, None, collision_reward

        # Sample actions and update targets
        action_car1, action_car2 = self.sample_action()
        reward, reached_target = self.store_step_in_replay_buffer(airsim_client, action_car1, collision)

        # Epsilon decay for exploration-exploitation balance
        if (self.step_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        # Translate actions to car controls
        current_controls_car1 = airsim_client.getCarControls("Car1")
        updated_controls_car1 = self.action_to_controls(current_controls_car1, action_car1)

        current_controls_car2 = airsim_client.getCarControls("Car2")
        updated_controls_car2 = self.action_to_controls(current_controls_car2, action_car2)

        # Batch training every buffer_limit steps
        if len(self.memory_buffer) > self.buffer_limit:
            loss_local = self.batch_train()
            print(self.step_counter)
            if not reached_target:
                with self.tensorboard.as_default():
                    tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

            self.memory_buffer.clear()  # Clear the buffer after training

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

    def store_step_in_replay_buffer(self, airsim_client, action_car1, collision):

        """
        pay attention that only action_car1 is the input.
        """

        # Store the current state
        current_state = np.array([list(self.cars_state.values())])

        # Compute the immediate reward and check if the target is reached
        reward, reached_target = self.calc_reward(collision)

        # Get next state
        self.cars_state = get_cars_state(airsim_client)
        next_state = np.array([list(self.cars_state.values())])

        # Store experience in the memory buffer
        self.memory_buffer.append((current_state, action_car1, reward, next_state))

        return reward, reached_target

    def batch_train(self):
        if not self.memory_buffer:
            return

        # Convert the experiences in the buffer into separate lists for each component
        states, actions, rewards, next_states = zip(*self.memory_buffer)

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
            current_q_values[i][action] += self.learning_rate * (targets[i] - current_q_values[i][action])

        # Batch update the network
        loss_local = self.local_network_car1.fit(states, current_q_values, batch_size=len(states), verbose=0, epochs=1)

        return loss_local

    def sample_action(self):

        # epsilon greedy
        if np.random.binomial(1, p=self.epsilon):
            rand_action = np.random.randint(2, size=(1, 1))[0][0]
            return rand_action, rand_action
        else:
            # both networks receive the same input, the difference is that they are 2 versions of the network.
            cars_state = np.array([list(self.cars_state.values())])
            action_selected_car1 = self.local_network_car1.predict(cars_state, verbose=self.verbose).argmax()
            action_selected_car2 = self.local_network_car2.predict(cars_state, verbose=self.verbose).argmax()
            return action_selected_car1, action_selected_car2

    def action_to_controls(self, current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls

    def calc_reward(self, collision):

        # avoid starvation
        reward = -0.1

        # collision
        if collision:
            reward -= 1000

        # too close
        if self.cars_state["dist_c1_c2"] < 70:
            reward -= 150

        # keeping safety distance
        if self.cars_state["dist_c1_c2"] > 100:
            reward += 60

        reached_target = False

        # reached target
        if self.cars_state['x_c1'] > self.c1_desire[0]:
            reward += 1000
            reached_target = True

        return reward, reached_target
