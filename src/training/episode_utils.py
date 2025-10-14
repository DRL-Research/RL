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
        # Query the master policy for the raw (continuous) acceleration proposals for every
        # vehicle that it believes it should control, alongside the state-value estimate and
        # log probability that are required for PPO training updates.
        raw_actions, value, log_prob = master_model.get_proto_action(master_input)

        # Determine the number of master-controlled vehicles directly from the
        # environment so that we never override the behaviour of static cars that
        # the scenario builder injects. Fallback to the configured fleet size only
        # when the attribute is missing (e.g. during tests) and clamp to the amount
        # of actions proposed by the master network to avoid index errors.
        env_controlled = getattr(env, "controlled_vehicles", None)
        expected_controlled = len(env_controlled) if env_controlled is not None else experiment.CARS_AMOUNT
        available_actions = len(raw_actions)

        if available_actions < expected_controlled:
            missing = expected_controlled - available_actions
            logger.warning(
                "Master proposed %s actions for %s controlled vehicles; reusing the last proposal to pad the remaining %s.",
                available_actions,
                expected_controlled,
                missing,
            )

        # Convert each raw acceleration (positive -> accelerate, negative -> brake) to a
        # discrete binary action that the environment understands and record it for logging.
        discrete_actions = []
        for idx in range(expected_controlled):
            source_idx = min(idx, max(available_actions - 1, 0))
            raw_action = raw_actions[source_idx] if available_actions > 0 else 0.0
            discrete_actions.append(1 if raw_action >= 0 else 0)

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
