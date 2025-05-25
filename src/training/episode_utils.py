import logging
from typing import Tuple, Any

import numpy as np
import torch

from src.model.agent_handler import Agent
from src.training.general_utils import (
    ensure_tensor, combine_agent_obs, get_action_array, get_obs_of_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    all_rewards, actions_per_episode = [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    crashed = False

    car1_observation, car2_observation, _ = env.reset()
    done, truncated = False, False

    while not done and not truncated:
        steps_counter += 1

        # Get full, flattened observation from environment
        full_obs = env.env.current_state

        # Compute master embedding & (for later optionally use) its value/log_prob
        master_input = ensure_tensor(full_obs)
        embedding, value, log_prob = master_model.get_proto_action(master_input)

        # Prepare agent observation vector
        car1_state = get_obs_of_agent(all_agents_states=full_obs, car_index=0)
        # car2_state = get_obs_of_agent(all_agents_states=full_obs, car_index=1)

        agent_obs = combine_agent_obs(car1_state, embedding, agent_model.policy.observation_space.shape[0])
        # agent_obs_car2 = combine_agent_obs(car1_state, embedding, agent_model.policy.observation_space.shape[0])

        # Choose agent action and compute its metrics
        agent_action = Agent.get_action(agent_model, car1_observation, total_steps, experiment.EXPLORATION_EXPLOITATION_THRESHOLD)
        scalar_action = agent_action[0] if (hasattr(agent_action, 'shape') and len(agent_action.shape) > 0) else \
            (agent_action[0] if isinstance(agent_action, list) and len(agent_action) > 0 else agent_action)
        action_array = get_action_array(agent_action)

        # Get agent values
        if train_both or not training_master:
            with torch.no_grad():
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                agent_value = agent_model.policy.predict_values(obs_tensor)
                dist = agent_model.policy.get_distribution(obs_tensor)
                log_prob_agent = dist.log_prob(torch.tensor(action_array, dtype=torch.long))

        env.render()
        next_obs, reward, done, truncated, info = env.step(scalar_action)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)
        actions_per_episode.append(scalar_action)

        if done and info.get("crashed", False):
            crashed = True

        episode_start = (steps_counter == 1)
        if train_both:
            full_obs_flat = full_obs.reshape(-1) if isinstance(full_obs, np.ndarray) and len(full_obs.shape) == 2 else full_obs
            agent_model.rollout_buffer.add(agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent)
            master_model.rollout_buffer.add(full_obs_flat, embedding, reward, episode_start, value, log_prob)
        elif training_master:
            full_obs_flat = full_obs.reshape(-1) if isinstance(full_obs, np.ndarray) and len(full_obs.shape) == 2 else full_obs
            master_model.rollout_buffer.add(full_obs_flat, embedding, reward, episode_start, value, log_prob)
        else:
            agent_model.rollout_buffer.add(agent_obs, action_array, reward, episode_start, agent_value, log_prob_agent)

        car1_observation = next_obs  # TODO: make is generic for more than car1

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


def process_episode(episode_idx, total_steps, env, master_model, agent_model, experiment, train_both, training_master) \
        -> Tuple[float, Any, int, bool]:
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
