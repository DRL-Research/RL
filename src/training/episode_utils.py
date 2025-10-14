import logging
from typing import List, Tuple

import numpy as np

from src.training.general_utils import ensure_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _reshape_state(state: np.ndarray) -> np.ndarray:
    if isinstance(state, np.ndarray) and state.ndim == 2:
        return state.reshape(-1)
    return np.array(state).reshape(-1)


def run_episode(experiment, env, master_model) -> Tuple[float, List[List[int]], int, bool, np.ndarray]:
    """Execute a full episode using only the master policy."""
    all_rewards: List[float] = []
    actions_per_episode: List[List[int]] = []
    steps_counter = 0
    episode_sum_of_rewards = 0.0
    crashed = False

    current_state, _ = env.reset()
    current_state = np.array(current_state)
    done = truncated = False

    while not done and not truncated:
        steps_counter += 1

        master_input = ensure_tensor(current_state)
        raw_actions, value, log_prob = master_model.get_proto_action(master_input)

        num_controlled = current_state.shape[0] if isinstance(current_state, np.ndarray) and current_state.ndim == 2 else experiment.CARS_AMOUNT
        num_controlled = min(num_controlled, len(raw_actions))
        discrete_actions = [1 if raw_actions[i] >= 0 else 0 for i in range(num_controlled)]
        print(f"Step {steps_counter}: Master actions {discrete_actions}")

        next_state, reward, done, truncated, info = env.step(tuple(discrete_actions))
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        actions_per_episode.append(discrete_actions)

        if done and info.get("crashed", False):
            crashed = True

        episode_start = steps_counter == 1
        flattened_state = _reshape_state(current_state)
        master_model.rollout_buffer.add(flattened_state, raw_actions, reward, episode_start, value, log_prob)

        current_state = np.array(next_state)

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed, current_state


def process_episode(episode_idx, env, master_model, experiment):
    """Run an episode and log results."""
    master_model.rollout_buffer.reset()

    reward, actions, steps, crashed, final_state = run_episode(experiment, env, master_model)
    status = "Collision" if crashed else "Success"

    print(f"Episode {episode_idx} | Result: {status} | Reward: {reward} | Steps: {steps}")

    return reward, actions, steps, crashed, final_state
