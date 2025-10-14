import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from src import project_globals
from src.model.agent_handler import Driver
from src.project_globals import rollout_buffers
from src.training.general_utils import ensure_tensor, get_agent_values_from_observation, get_scaler_action_and_action_array
from src.training.rollout_buffer_utils import reset_all_buffers


def reshape_drivers_states(all_drivers_states):
    all_drivers_states = all_drivers_states.reshape(-1) if isinstance(all_drivers_states, np.ndarray) and len(all_drivers_states.shape) == 2 else all_drivers_states
    return all_drivers_states


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    all_rewards, actions_per_episode = [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    crashed = False

    car_observations, _ = env.reset()
    done, truncated = False, False

    while not done and not truncated:
        steps_counter += 1

        # Get all drivers states (without master embedding) from environment
        all_drivers_states = env.env.current_state

        # Compute master embedding & (for later optionally use) its value/log_prob
        embedding, value, log_prob = master_model.get_proto_action(ensure_tensor(all_drivers_states))

        # Choose agent action and compute its metrics
        actions = Driver.get_action(agent_model, car_observations, total_steps, experiment.EXPLORATION_EXPLOITATION_THRESHOLD)

        cars_scalar_action, cars_action_arrays = [], []
        for action in actions:
            car_scalar_action, car_action_array = get_scaler_action_and_action_array(action)
            cars_scalar_action.append(car_scalar_action)
            cars_action_arrays.append(car_action_array)

        #print(f"Actions: [{car1_scalar_action}, {car2_scalar_action}]")

        # Get agent values
        cars_values, cars_log_probas = [], []
        if train_both or not training_master:
            for car_index in range(len(car_observations)):
                car_values, car_log_prob = get_agent_values_from_observation(car_observations[car_index],
                                                                             cars_action_arrays[car_index],
                                                                             agent_model)
                cars_values.append(car_values)
                cars_log_probas.append(car_log_prob)
        env.render()

        action_tuple = tuple(cars_scalar_action)
        cars_next_obs, reward, done, truncated, info = env.step(action_tuple)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        # actions_per_episode.append(car1_scalar_action)

        if done and info.get("crashed", False):
            crashed = True

        episode_start = (steps_counter == 1)
        all_drivers_states = reshape_drivers_states(all_drivers_states)

        if train_both:  # added flag for arrived that turns True after one time of is_arrived, and add to buffer only if True (for car1 and car2)
            for car_index in range(len(project_globals.after_is_arrived_flags)):
                if not project_globals.after_is_arrived_flags[car_index]:
                    rollout_buffers[car_index].add(car_observations[car_index], cars_action_arrays[car_index], reward,
                                                   episode_start, cars_values[car_index], cars_log_probas[car_index])
            master_model.rollout_buffer.add(all_drivers_states, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            master_model.rollout_buffer.add(all_drivers_states, embedding, reward, episode_start, value, log_prob)
        else:
            for car_index in range(len(project_globals.after_is_arrived_flags)):
                if not project_globals.after_is_arrived_flags[car_index]:
                    rollout_buffers[car_index].add(car_observations[car_index], cars_action_arrays[car_index], reward,
                                                   episode_start, cars_values[car_index], cars_log_probas[car_index])

        car_observations = cars_next_obs

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


def process_episode(episode_idx, total_steps, env, master_model, agent_model, experiment, train_both, training_master):
    """
    Run an episode and log results.
    Returns: (reward, actions, steps, crashed)
    """
    master_model.rollout_buffer.reset()
    reset_all_buffers()  # reset all buffers (for each Driver)

    # TODO: Create an assert here to see that they are reset propely

    reward, actions, steps, crashed = run_episode(experiment, total_steps, env, master_model, agent_model,
                                                  train_both=train_both, training_master=training_master)
    status = "Collision" if crashed else "Success"

    print(f"Episode {episode_idx} | Result: {status} | Reward: {reward} | Steps: {steps}")

    return reward, actions, steps, crashed
