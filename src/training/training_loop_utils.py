import logging
import traceback
from typing import Tuple, List, Dict, Any

import numpy as np
import torch

from src.training.general_utils import ensure_tensor, flatten_obs, combine_agent_obs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_training_results() -> Dict[str, List[Any]]:
    """
    Initialize the structure for tracking training results.
    """
    return {
        "episode_rewards": [],
        "master_policy_losses": [],
        "master_value_losses": [],
        "master_total_losses": [],
        "agent_policy_losses": [],
        "agent_value_losses": [],
        "agent_total_losses": [],
        "all_actions": []
    }


def prepare_models_for_cycle(
    cycle_num: int,
    total_cycles: int,
    master_model,
    agent_model
) -> Tuple[bool, bool, bool]:
    """
    Determine which networks to train this cycle and set freeze/unfreeze states.
    Returns flags: (train_both, training_master, training_agent)
    """
    logger.info("Cycle %d/%d", cycle_num, total_cycles)
    train_both = cycle_num == 1
    training_master = not train_both and (cycle_num % 2 == 1)
    training_agent = not train_both and (cycle_num % 2 == 0)

    # Freeze/unfreeze master and set agent training mode
    if train_both or training_master:
        master_model.unfreeze()
    else:
        master_model.freeze()
    agent_model.policy.set_training_mode(train_both or training_agent)

    return train_both, training_master, training_agent


def record_losses(
    losses: Tuple[float, float, float] or None,
    policy_list: List[Any],
    value_list: List[Any],
    total_list: List[Any]
) -> None:
    """
    Append losses or None to tracking lists.
    """
    policy_loss = value_loss = total_loss = None
    if losses:
        policy_loss, value_loss, total_loss = losses
    policy_list.append(policy_loss)
    value_list.append(value_loss)
    total_list.append(total_loss)




def train_master_and_reset_buffer(master_model, full_obs):
    """Trains the master model and resets its buffer - Returns loss values"""
    policy_loss_val = value_loss_val = total_loss_val = None
    try:
        with torch.no_grad():
            last_master_tensor = ensure_tensor(full_obs)
            last_value = master_model.model.policy.predict_values(last_master_tensor)
        if master_model.rollout_buffer.pos == 0:
            print("Master model: No data to train on")
            return None
        master_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
        orig_get = master_model.rollout_buffer.get

        def modified_get(batch_size):
            if not master_model.rollout_buffer.full:
                orig_full = master_model.rollout_buffer.full
                master_model.rollout_buffer.full = True
                try:
                    indices = np.arange(master_model.rollout_buffer.pos)
                    if len(indices) > 0:
                        return master_model.rollout_buffer._get_samples(indices)
                    else:
                        return None
                finally:
                    master_model.rollout_buffer.full = orig_full
            else:
                return next(orig_get(batch_size))

        try:
            master_model.rollout_buffer.get = modified_get
            rollout_data = master_model.rollout_buffer.get(batch_size=None)
            if rollout_data is not None:
                steps_trained = len(rollout_data.observations)
                print(f"Master model trained on {steps_trained} steps")
                observations_tensor = torch.FloatTensor(rollout_data.observations)
                actions_tensor = torch.FloatTensor(rollout_data.actions)
                policy = master_model.model.policy
                optimizer = policy.optimizer
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()
                policy_loss = -log_probs.mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()
                print(
                    f"Master loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
            else:
                print("Master model: No valid data for training")
        finally:
            master_model.rollout_buffer.get = orig_get
    except Exception as e:
        print(f"Error during master training: {str(e)}")
        traceback.print_exc()
    master_model.rollout_buffer.reset()
    print("Master buffer reset")
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


def train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor):
    """Trains the agent model and resets its buffer - Returns loss values"""
    policy_loss_val = value_loss_val = total_loss_val = None
    try:
        with torch.no_grad():
            shaped_tensor = ensure_tensor(last_master_tensor)
            embedding = master_model.get_proto_action(shaped_tensor)
            car_state = flatten_obs(last_master_tensor)
            expected_dim = agent_model.policy.observation_space.shape[0]
            agent_obs = combine_agent_obs(car_state, embedding, expected_dim)
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
            last_value = agent_model.policy.predict_values(agent_obs_tensor)
        if agent_model.rollout_buffer.pos == 0:
            print("Agent model: No data to train on")
            return None
        agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
        orig_get = agent_model.rollout_buffer.get

        def modified_get(batch_size):
            if not agent_model.rollout_buffer.full:
                orig_full = agent_model.rollout_buffer.full
                agent_model.rollout_buffer.full = True
                try:
                    indices = np.arange(agent_model.rollout_buffer.pos)
                    if len(indices) > 0:
                        return agent_model.rollout_buffer._get_samples(indices)
                    else:
                        return None
                finally:
                    agent_model.rollout_buffer.full = orig_full
            else:
                return next(orig_get(batch_size))

        try:
            agent_model.rollout_buffer.get = modified_get
            rollout_data = agent_model.rollout_buffer.get(batch_size=None)
            if rollout_data is not None:
                steps_trained = len(rollout_data.observations)
                print(f"Agent model trained on {steps_trained} steps")
                observations_tensor = torch.FloatTensor(rollout_data.observations)
                actions_tensor = torch.FloatTensor(rollout_data.actions)
                policy = agent_model.policy
                optimizer = policy.optimizer
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()
                advantages_tensor = torch.FloatTensor(rollout_data.advantages)
                policy_loss = -(log_probs * advantages_tensor).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()
                print(
                    f"Agent loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
            else:
                print("Agent model: No valid data for training")
        finally:
            agent_model.rollout_buffer.get = orig_get
    except Exception as e:
        print(f"Error during agent training: {str(e)}")
        traceback.print_exc()
    agent_model.rollout_buffer.reset()
    print("Agent buffer reset")
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None




def perform_training_phase(
    train_both: bool,
    training_master: bool,
    training_agent: bool,
    master_model,
    agent_model,
    full_state: Any,
    state_tensor: torch.Tensor,
    results: Dict[str, List[Any]]) -> None:
    """
    Execute training for master and/or agent and update results.
    """
    master_losses = None
    agent_losses = None
    if train_both or training_master:
        master_losses = train_master_and_reset_buffer(master_model, full_state)
    if train_both or training_agent:
        agent_losses = train_agent_and_reset_buffer(master_model, agent_model, state_tensor)

    record_losses(
        master_losses,
        results["master_policy_losses"],
        results["master_value_losses"],
        results["master_total_losses"]
    )
    record_losses(
        agent_losses,
        results["agent_policy_losses"],
        results["agent_value_losses"],
        results["agent_total_losses"]
    )

