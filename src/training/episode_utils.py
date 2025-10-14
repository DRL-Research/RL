import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

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

        # Get all drivers states (without master control) from environment
        all_drivers_states = env.env.current_state
        master_controls = env.env.current_controls
        master_value = env.env.current_value
        master_log_prob = env.env.current_log_prob

        env.render()

        cars_next_obs, reward, done, truncated, info = env.step(None)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)

        applied_controls = env.env.last_applied_controls
        discrete_actions = env.env.get_discrete_master_actions()
        actions_per_episode.append(list(discrete_actions))

        if done and info.get("crashed", False):
            crashed = True

        episode_start = (steps_counter == 1)
        all_drivers_states = reshape_drivers_states(all_drivers_states)

        if master_controls is None:
            control_count = len(discrete_actions)
            master_controls = np.zeros(control_count, dtype=np.float32)

        if applied_controls is None:
            applied_controls = master_controls

        if train_both or training_master:
            master_model.rollout_buffer.add(all_drivers_states, applied_controls, reward, episode_start,
                                            master_value, master_log_prob)

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
