import time
import tensorflow as tf
from RL.utils.environment_utils import *
from RL.utils.NN_utils import *
from RL.utils.replay_buffer_utils import *
from RL.config import LEARNING_RATE, CAR1_DESIRED_POSITION, CAR1_NAME, CAR2_NAME


class RLAgent:

    def __init__(self, tensorboard):
        self.learning_rate = LEARNING_RATE
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        self.cars_state = None
        self.discount_factor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.tensorboard = tensorboard
        self.local_network_car1 = init_local_network(self.opt)
        self.local_network_car2 = copy_network(self.local_network_car1)
        self.memory_buffer = []  # Initialize an empty list to store experiences
        self.buffer_limit = 10  # Set the buffer size limit for batch training

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
        time.sleep(0.5)

        # get next state
        cars_next_state_car1_perspective = get_local_input_car1_perspective(airsim_client_handler.airsim_client)
        cars_next_state_car2_perspective = get_local_input_car2_perspective(airsim_client_handler.airsim_client)

        reward, reached_target = self.calc_reward(collision, cars_next_state_car1_perspective)  # sent car1_perspective for distance between cars

        # store step (of both car perspective) in replay buffer for batch training
        self.memory_buffer.append((np.array([list(cars_current_state_car1_perspective.values())]), action_car1, reward, np.array([list(cars_next_state_car1_perspective.values())])))
        self.memory_buffer.append((np.array([list(cars_current_state_car2_perspective.values())]), action_car2, reward, np.array([list(cars_next_state_car2_perspective.values())])))

        # Epsilon decay for exploration-exploitation balance
        if (steps_counter % 5) == 0:
            self.epsilon *= self.epsilon_decay

        # Translate actions to car controls
        current_controls_car1 = airsim_client_handler.airsim_client.getCarControls(CAR1_NAME)
        updated_controls_car1 = self.action_to_controls(current_controls_car1, action_car1)

        current_controls_car2 = airsim_client_handler.airsim_client.getCarControls(CAR2_NAME)
        updated_controls_car2 = self.action_to_controls(current_controls_car2, action_car2)

        # Batch training every buffer_limit steps
        if len(self.memory_buffer) > self.buffer_limit:
            loss_local = self.batch_train()
            if not reached_target:
                with self.tensorboard.as_default():
                    tf.summary.scalar('loss', loss_local.history["loss"][-1], step=steps_counter)

            self.memory_buffer.clear()  # Clear the buffer after training

        return collision, reached_target, updated_controls_car1, updated_controls_car2, reward

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

    def sample_action(self, airsim_client):

        # epsilon greedy
        if np.random.binomial(1, p=self.epsilon):
            rand_action = np.random.randint(2, size=(1, 1))[0][0]
            return rand_action, rand_action
        else:
            # both networks receive the same input (but with different perspectives (meaning-> different order))
            # Another difference is, that there are 2 versions of the network.
            cars_state_car1_perspective = np.array([list(get_local_input_car1_perspective(airsim_client).values())])
            action_selected_car1 = self.local_network_car1.predict(cars_state_car1_perspective, verbose=0).argmax()

            cars_state_car2_perspective = np.array([list(get_local_input_car2_perspective(airsim_client).values())])
            action_selected_car2 = self.local_network_car2.predict(cars_state_car2_perspective, verbose=0).argmax()
            return action_selected_car1, action_selected_car2

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

    @ staticmethod
    def action_to_controls(current_controls, action):
        # translate index of action to controls in car:
        if action == 0:
            current_controls.throttle = 0.75
        elif action == 1:
            current_controls.throttle = 0.4
        return current_controls  # called current_controls - but it is updated controls
