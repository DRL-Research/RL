import logging

import numpy as np

from src.model.agent_handler import Driver
from src.training.general_utils import ensure_tensor, get_agent_values_from_observation, get_scaler_action_and_action_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reshape_drivers_states(all_drivers_states):
    all_drivers_states = all_drivers_states.reshape(-1) if isinstance(all_drivers_states, np.ndarray) and len(all_drivers_states.shape) == 2 else all_drivers_states
    return all_drivers_states


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    all_rewards, actions_per_episode = [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    crashed = False

    car1_observation, car2_observation, _ = env.reset()
    done, truncated = False, False

    while not done and not truncated:
        steps_counter += 1

        # Get all drivers states (without master embedding) from environment
        all_drivers_states = env.env.current_state

        # Compute master embedding & (for later optionally use) its value/log_prob
        embedding, value, log_prob = master_model.get_proto_action(ensure_tensor(all_drivers_states))

        # Choose agent action and compute its metrics
        car1_action, car2_action = Driver.get_action(agent_model, car1_observation, car2_observation, total_steps,
                                                     experiment.EXPLORATION_EXPLOITATION_THRESHOLD)

        car1_scalar_action, car1_action_array = get_scaler_action_and_action_array(car1_action)
        car2_scalar_action, car2_action_array = get_scaler_action_and_action_array(car2_action)

        #print(f"Actions: [{car1_scalar_action}, {car2_scalar_action}]")

        # Get agent values
        if train_both or not training_master:
            car1_values, car1_log_prob = get_agent_values_from_observation(car1_observation, car1_action_array, agent_model)
            car2_values, car2_log_prob = get_agent_values_from_observation(car2_observation, car2_action_array, agent_model)

        env.render()

        car1_next_obs, car2_next_obs, reward, done, truncated, info = env.step((car1_scalar_action, car2_scalar_action))
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        actions_per_episode.append(car1_scalar_action)

        if done and info.get("crashed", False):
            crashed = True

        episode_start = (steps_counter == 1)
        all_drivers_states = reshape_drivers_states(all_drivers_states)

        if train_both:
            agent_model.rollout_buffer.add(car1_observation, car1_action_array, reward, episode_start, car1_values, car1_log_prob)
            master_model.rollout_buffer.add(all_drivers_states, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            master_model.rollout_buffer.add(all_drivers_states, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(car1_observation, car1_action_array, reward, episode_start, car1_values, car1_log_prob)

        car1_observation = car1_next_obs
        car2_observation = car2_next_obs


    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


def process_episode(episode_idx, total_steps, env, master_model, agent_model, experiment, train_both, training_master):
    """
    Run an episode and log results.
    Returns: (reward, actions, steps, crashed)
    """
    logger.info("Episode %d", episode_idx)
    master_model.rollout_buffer.reset()
    agent_model.rollout_buffer.reset()

    reward, actions, steps, crashed = run_episode(experiment, total_steps, env, master_model, agent_model,
                                                  train_both=train_both, training_master=training_master)
    status = "Collision" if crashed else "Success"
    if crashed:
        logger.warning("Episode %d ended with %s", episode_idx, status)
    else:
        logger.info("Episode %d ended with %s", episode_idx, status)

    logger.info(
        "Result: %s | Reward: %.2f | Steps: %d", status, reward, steps)
    return reward, actions, steps, crashed
