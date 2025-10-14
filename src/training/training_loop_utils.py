import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.training.general_utils import ensure_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_training_results() -> Dict[str, List[Any]]:
    """Initialize the structure for tracking training results."""
    return {
        "episode_rewards": [],
        "master_policy_losses": [],
        "master_value_losses": [],
        "master_total_losses": [],
        "all_actions": [],
    }


def prepare_master_for_cycle(cycle_num: int, total_cycles: int, master_model) -> None:
    """Log cycle progress and ensure the master is trainable."""
    print(f"Cycle {cycle_num}/{total_cycles}")
    master_model.unfreeze()


def record_losses(losses: Optional[List[float]], results: Dict[str, List[Any]]) -> None:
    """Append loss values (or None) to the tracking structure."""
    policy_loss = value_loss = total_loss = None
    if losses:
        policy_loss, value_loss, total_loss = losses

    results["master_policy_losses"].append(policy_loss)
    results["master_value_losses"].append(value_loss)
    results["master_total_losses"].append(total_loss)


def train_master_and_reset_buffer(master_model, full_obs) -> Optional[List[float]]:
    """Train the master model using the accumulated rollout buffer."""
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
            else:
                print("Master model: No valid data for training")
        finally:
            master_model.rollout_buffer.get = orig_get
    except Exception as e:
        print(f"Error during master training: {str(e)}")
        import traceback
        traceback.print_exc()
    master_model.rollout_buffer.reset()
    print("Master buffer reset")
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


def perform_training_phase(master_model, full_state, results: Dict[str, List[Any]]) -> None:
    """Execute master training and log the resulting losses."""
    master_losses = train_master_and_reset_buffer(master_model, full_state)
    record_losses(master_losses, results)
