import logging
import traceback
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from src.training.general_utils import ensure_tensor, flatten_obs, combine_agent_obs, sanitize_float_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _TrainingValueNormalizer:
    """Hook for optional running value statistics. Unified training re-inits these each run."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        pass


agent_value_normalizer = _TrainingValueNormalizer()
master_value_normalizer = _TrainingValueNormalizer()


def init_training_results() -> Dict[str, List[Any]]:
    """
    Initialize the structure for tracking training results.
    """
    return {
        "episode_rewards": [],
        "arrival_rates": [],
        "collision_rates": [],
        "master_policy_losses": [],
        "master_value_losses": [],
        "master_total_losses": [],
        "agent_policy_losses": [],
        "agent_value_losses": [],
        "agent_total_losses": [],
        "all_actions": [],
        "ppo_training_log": [],  # detailed dicts appended in perform_training_phase
    }


def prepare_models_for_cycle(
    cycle_num: int,
    total_cycles: int,
    master_model,
    agent_model,
    *,
    cotrain_cycles: bool = False,
    full_joint: bool = False,
) -> Tuple[bool, bool, bool, bool]:
    """
    Returns (train_both, training_local_master, training_agent, training_global_master).

    ``full_joint`` + ``cotrain_cycles`` → train all heads every cycle.
    Otherwise mirrors the legacy alternating cycle rule, with GM following local masters.
    """
    logger.info("Cycle %d/%d", cycle_num, total_cycles)

    if cotrain_cycles and full_joint:
        train_both = True
        training_local_master = True
        training_agent = True
        training_global_master = True
    elif cotrain_cycles:
        train_both = cycle_num == 1
        training_local_master = train_both or (cycle_num % 2 == 0)
        training_agent = train_both or (cycle_num % 2 == 1)
        training_global_master = training_local_master
    else:
        train_both = cycle_num == 1
        training_local_master = train_both or (cycle_num % 2 == 0)
        training_agent = train_both or (cycle_num % 2 == 1)
        training_global_master = training_local_master

    if train_both or training_local_master or training_global_master:
        master_model.unfreeze()
    else:
        master_model.freeze()
    agent_model.policy.set_training_mode(train_both or training_agent)

    return train_both, training_local_master, training_agent, training_global_master


def _sb3_policy_flat_l2(policy) -> float:
    """Concatenated-parameter Euclidean norm ‖θ‖₂ (SB3 ActorCriticPolicies)."""
    chunks: List[torch.Tensor] = []
    for p in policy.parameters():
        chunks.append(p.detach().float().reshape(-1))
    if not chunks:
        return 0.0
    return float(torch.linalg.norm(torch.cat(chunks)).cpu())


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



def _compute_returns_partial(
    rollout_buffer: RolloutBuffer,
    last_values: torch.Tensor,
    dones: np.ndarray,
) -> None:
    """
    GAE(lambda) only on ``[0, pos)``. SB3's ``compute_returns_and_advantage``
    scans the entire ``buffer_size`` and corrupts advantages when the buffer is
    only partially filled (short episodes << N_STEPS).
    """
    p = int(rollout_buffer.pos)
    if p <= 0:
        rollout_buffer.advantages.fill(0)
        rollout_buffer.returns.fill(0)
        return

    lv = np.asarray(last_values.detach().cpu().numpy().flatten(), dtype=np.float32)
    dn = np.asarray(dones, dtype=np.float32).flatten()
    last_gae_lam = np.zeros_like(rollout_buffer.rewards[0], dtype=np.float32)

    for step in reversed(range(p)):
        if step == p - 1:
            next_non_terminal = np.ones_like(rollout_buffer.rewards[step], dtype=np.float32) - dn
            next_values = lv
        else:
            next_non_terminal = 1.0 - rollout_buffer.episode_starts[step + 1].astype(np.float32)
            next_values = rollout_buffer.values[step + 1].astype(np.float32)

        rew = rollout_buffer.rewards[step].astype(np.float32)
        val = rollout_buffer.values[step].astype(np.float32)
        delta = rew + rollout_buffer.gamma * next_values * next_non_terminal - val
        last_gae_lam = delta + rollout_buffer.gamma * rollout_buffer.gae_lambda * next_non_terminal * last_gae_lam
        rollout_buffer.advantages[step] = last_gae_lam

    rollout_buffer.returns[:p] = rollout_buffer.advantages[:p] + rollout_buffer.values[:p]
    rollout_buffer.advantages[p:].fill(0)
    rollout_buffer.returns[p:].fill(0)


def _squeeze_rollout_obs_act(
    rollout_data,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(T,n_env,dim)→(T,dim); policy expects batched vectors with no env singleton."""
    obs = torch.as_tensor(rollout_data.observations, dtype=torch.float32)
    act = torch.as_tensor(rollout_data.actions, dtype=torch.float32)
    if obs.dim() == 3 and obs.size(1) == 1:
        obs = obs.squeeze(1)
    if act.dim() == 3 and act.size(1) == 1:
        act = act.squeeze(1)
    return obs, act


def _normalize_advantages_ppo(advantages_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Center divide-by-std matching this file's legacy PPO update.

    ``Tensor.std()`` defaults to Bessel/unbiased variance; batch size 1 yields NaN
    and destroys shared policy weights across cars.
    """
    advantages_tensor = torch.nan_to_num(
        advantages_tensor, nan=0.0, posinf=0.0, neginf=0.0
    )
    centered = advantages_tensor - advantages_tensor.mean()
    raw_std = advantages_tensor.std(unbiased=False)
    if not torch.isfinite(raw_std):
        raw_std = advantages_tensor.new_zeros(())
    return centered / (raw_std + eps)




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
        _compute_returns_partial(master_model.rollout_buffer, last_value, np.array([True], dtype=np.float32))
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
                observations_tensor, actions_tensor = _squeeze_rollout_obs_act(rollout_data)
                observations_tensor = torch.nan_to_num(observations_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                actions_tensor = torch.nan_to_num(actions_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                steps_trained = int(observations_tensor.shape[0])
                print(f"Master model trained on {steps_trained} steps")
                policy = master_model.model.policy
                optimizer = policy.optimizer
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                n_batch = int(observations_tensor.shape[0])
                returns_tensor = torch.as_tensor(rollout_data.returns, dtype=torch.float32).reshape(-1)[:n_batch]
                advantages_tensor = torch.nan_to_num(
                    torch.as_tensor(rollout_data.advantages, dtype=torch.float32).reshape(-1)[:n_batch],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                values_flat = values.reshape(-1)[:n_batch]
                log_probs_flat = log_probs.reshape(-1)[:n_batch]
                value_loss = ((values_flat - returns_tensor) ** 2).mean()
                advantages_tensor = _normalize_advantages_ppo(advantages_tensor)
                policy_loss = -(log_probs_flat * advantages_tensor).mean()
                entropy_loss = (
                    -entropy.mean()
                    if entropy is not None
                    else observations_tensor.new_zeros(())
                )
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                if torch.isfinite(loss).all():
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


def train_shared_master_on_buffer(
    master_model,
    full_obs_np: np.ndarray,
    rollout_buffer: RolloutBuffer,
    *,
    last_done: np.ndarray | None = None,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
):
    """PPO-style update on an arbitrary master rollout buffer (LM1 / LM2 / GM)."""
    policy_loss_val = value_loss_val = total_loss_val = None
    ld = last_done if last_done is not None else np.array([1.0], dtype=np.float32)
    try:
        with torch.no_grad():
            last_master_tensor = ensure_tensor(full_obs_np)
            last_value = master_model.model.policy.predict_values(last_master_tensor)
        if rollout_buffer.pos == 0:
            return None
        _compute_returns_partial(rollout_buffer, last_value, np.asarray(ld, dtype=np.float32))
        orig_get = rollout_buffer.get

        def modified_get(batch_size):
            if not rollout_buffer.full:
                orig_full = rollout_buffer.full
                rollout_buffer.full = True
                try:
                    indices = np.arange(rollout_buffer.pos)
                    if len(indices) > 0:
                        return rollout_buffer._get_samples(indices)
                    return None
                finally:
                    rollout_buffer.full = orig_full
            return next(orig_get(batch_size))

        try:
            rollout_buffer.get = modified_get
            rollout_data = rollout_buffer.get(batch_size=None)
            if rollout_data is None:
                return None
            observations_tensor, actions_tensor = _squeeze_rollout_obs_act(rollout_data)
            observations_tensor = torch.nan_to_num(observations_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            actions_tensor = torch.nan_to_num(actions_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            steps_trained = int(observations_tensor.shape[0])
            print(f"  [Master aux buffer] trained on {steps_trained} steps")
            policy = master_model.model.policy
            optimizer = policy.optimizer
            values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
            n_batch = int(observations_tensor.shape[0])
            returns_tensor = torch.as_tensor(rollout_data.returns, dtype=torch.float32).reshape(-1)[:n_batch]
            advantages_tensor = torch.nan_to_num(
                torch.as_tensor(rollout_data.advantages, dtype=torch.float32).reshape(-1)[:n_batch],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            values_flat = values.reshape(-1)[:n_batch]
            log_probs_flat = log_probs.reshape(-1)[:n_batch]
            value_loss = ((values_flat - returns_tensor) ** 2).mean()
            advantages_tensor = _normalize_advantages_ppo(advantages_tensor)
            policy_loss = -(log_probs_flat * advantages_tensor).mean()
            entropy_loss = (
                -entropy.mean()
                if entropy is not None
                else observations_tensor.new_tensor(0.0)
            )
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            optimizer.zero_grad()
            if torch.isfinite(loss).all():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                total_loss_val = loss.item()
        finally:
            rollout_buffer.get = orig_get
    except Exception as e:
        print(f"Error during auxiliary master training: {e}")
        traceback.print_exc()
    rollout_buffer.reset()
    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None


def train_agents_with_bootstrap(
    agent_model,
    last_agent_obs_list: list[np.ndarray],
    *,
    last_done: np.ndarray | None = None,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
):
    """Train each agent buffer using its own bootstrap observation."""
    from src.project_globals import rollout_buffers

    ld = np.asarray(last_done if last_done is not None else np.array([1.0]), dtype=np.float32)
    agg_policy = agg_value = agg_total = []
    for i, current_rollout_buffer in enumerate(rollout_buffers):
        policy_loss_val = value_loss_val = total_loss_val = None
        try:
            obs_space = getattr(agent_model.policy, "observation_space", None)
            odim = int(obs_space.shape[0]) if obs_space is not None and hasattr(obs_space, "shape") else None
            obs_np = sanitize_float_vector(last_agent_obs_list[i], dim=odim)
            with torch.no_grad():
                agent_obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
                last_value = agent_model.policy.predict_values(agent_obs_tensor)

            if current_rollout_buffer.pos == 0:
                continue

            _compute_returns_partial(current_rollout_buffer, last_value, ld)
            orig_get = current_rollout_buffer.get

            def modified_get(batch_size):
                if not current_rollout_buffer.full:
                    orig_full = current_rollout_buffer.full
                    current_rollout_buffer.full = True
                    try:
                        indices = np.arange(current_rollout_buffer.pos)
                        if len(indices) > 0:
                            return current_rollout_buffer._get_samples(indices)
                        return None
                    finally:
                        current_rollout_buffer.full = orig_full
                return next(orig_get(batch_size))

            try:
                current_rollout_buffer.get = modified_get
                rollout_data = current_rollout_buffer.get(batch_size=None)
                if rollout_data is None:
                    continue
                observations_tensor, actions_tensor = _squeeze_rollout_obs_act(rollout_data)
                observations_tensor = torch.nan_to_num(
                    observations_tensor, nan=0.0, posinf=0.0, neginf=0.0
                )
                actions_tensor = torch.nan_to_num(
                    actions_tensor,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                policy = agent_model.policy
                optimizer = policy.optimizer
                values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                n_batch = int(observations_tensor.shape[0])
                returns_tensor = torch.as_tensor(rollout_data.returns, dtype=torch.float32).reshape(-1)[:n_batch]
                advantages_tensor = torch.nan_to_num(
                    torch.as_tensor(rollout_data.advantages, dtype=torch.float32).reshape(-1)[:n_batch],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                values_flat = values.reshape(-1)[:n_batch]
                log_probs_flat = log_probs.reshape(-1)[:n_batch]
                value_loss = ((values_flat - returns_tensor) ** 2).mean()
                advantages_tensor = _normalize_advantages_ppo(advantages_tensor)
                policy_loss = -(log_probs_flat * advantages_tensor).mean()
                entropy_loss = (
                    -entropy.mean()
                    if entropy is not None
                    else observations_tensor.new_zeros(())
                )
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                optimizer.zero_grad()
                if torch.isfinite(loss).all():
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()
                    total_loss_val = loss.item()
            finally:
                current_rollout_buffer.get = orig_get
        except Exception as e:
            print(f"Error during agent training car {i}: {e}")
            traceback.print_exc()
        finally:
            current_rollout_buffer.reset()

        if policy_loss_val is not None:
            agg_policy.append(policy_loss_val)
            agg_value.append(value_loss_val)
            agg_total.append(total_loss_val)

    if not agg_policy:
        return None
    return [
        float(np.mean(agg_policy)),
        float(np.mean(agg_value)),
        float(np.mean(agg_total)),
    ]



def train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor):
    """Trains the agent model and resets its buffer - Returns loss values"""
    from src.project_globals import rollout_buffers

    for current_rollout_buffer in rollout_buffers:

        policy_loss_val = value_loss_val = total_loss_val = None
        try:
            with torch.no_grad():
                shaped_tensor = ensure_tensor(last_master_tensor)
                embedding, _, _ = master_model.get_proto_action(shaped_tensor)
                car_state = flatten_obs(last_master_tensor)
                expected_dim = agent_model.policy.observation_space.shape[0]
                agent_obs = combine_agent_obs(car_state, embedding, expected_dim)
                agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                last_value = agent_model.policy.predict_values(agent_obs_tensor)

            if current_rollout_buffer.pos == 0:
                print("Agent model: No data to train on")
                return None

            _compute_returns_partial(current_rollout_buffer, last_value, np.array([1.0], dtype=np.float32))
            orig_get = current_rollout_buffer.get

            def modified_get(batch_size):
                if not current_rollout_buffer.full:
                    orig_full = current_rollout_buffer.full
                    current_rollout_buffer.full = True
                    try:
                        indices = np.arange(current_rollout_buffer.pos)
                        if len(indices) > 0:
                            return current_rollout_buffer._get_samples(indices)
                        else:
                            return None
                    finally:
                        current_rollout_buffer.full = orig_full
                else:
                    return next(orig_get(batch_size))

            try:
                current_rollout_buffer.get = modified_get
                rollout_data = current_rollout_buffer.get(batch_size=None)
                if rollout_data is not None:
                    observations_tensor, actions_tensor = _squeeze_rollout_obs_act(rollout_data)
                    observations_tensor = torch.nan_to_num(
                        observations_tensor, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    actions_tensor = torch.nan_to_num(actions_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                    steps_trained = int(observations_tensor.shape[0])
                    print(f"Agent model trained on {steps_trained} steps")
                    policy = agent_model.policy
                    optimizer = policy.optimizer
                    values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
                    n_batch = int(observations_tensor.shape[0])
                    returns_tensor = torch.as_tensor(
                        rollout_data.returns, dtype=torch.float32
                    ).reshape(-1)[:n_batch]
                    advantages_tensor = torch.nan_to_num(
                        torch.as_tensor(rollout_data.advantages, dtype=torch.float32).reshape(-1)[:n_batch],
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    values_flat = values.reshape(-1)[:n_batch]
                    log_probs_flat = log_probs.reshape(-1)[:n_batch]
                    value_loss = ((values_flat - returns_tensor) ** 2).mean()
                    advantages_tensor = _normalize_advantages_ppo(advantages_tensor)
                    policy_loss = -(log_probs_flat * advantages_tensor).mean()
                    entropy_loss = (
                        -entropy.mean()
                        if entropy is not None
                        else observations_tensor.new_zeros(())
                    )
                    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                    optimizer.zero_grad()
                    if torch.isfinite(loss).all():
                        loss.backward()  # TODO: log the losses better
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
                current_rollout_buffer.get = orig_get
        except Exception as e:
            print(f"Error during agent training: {str(e)}")
            traceback.print_exc()
        current_rollout_buffer.reset()
        print("Agent buffer reset")


    if policy_loss_val is not None:
        return [policy_loss_val, value_loss_val, total_loss_val]
    return None

#
# def train_agent_and_reset_buffer(master_model, agent_model, last_master_tensor):
#     """Trains the agent model and resets its buffer - Returns loss values"""
#     policy_loss_val = value_loss_val = total_loss_val = None
#     try:
#         with torch.no_grad():
#             shaped_tensor = ensure_tensor(last_master_tensor)
#             embedding, _, _ = master_model.get_proto_action(shaped_tensor)
#             car_state = flatten_obs(last_master_tensor)
#             expected_dim = agent_model.policy.observation_space.shape[0]
#             agent_obs = combine_agent_obs(car_state, embedding, expected_dim)
#             agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
#             last_value = agent_model.policy.predict_values(agent_obs_tensor)
#         if agent_model.rollout_buffer.pos == 0:
#             print("Agent model: No data to train on")
#             return None
#         agent_model.rollout_buffer.compute_returns_and_advantage(last_value, np.array([True]))
#         orig_get = agent_model.rollout_buffer.get
#
#         def modified_get(batch_size):
#             if not agent_model.rollout_buffer.full:
#                 orig_full = agent_model.rollout_buffer.full
#                 agent_model.rollout_buffer.full = True
#                 try:
#                     indices = np.arange(agent_model.rollout_buffer.pos)
#                     if len(indices) > 0:
#                         return agent_model.rollout_buffer._get_samples(indices)
#                     else:
#                         return None
#                 finally:
#                     agent_model.rollout_buffer.full = orig_full
#             else:
#                 return next(orig_get(batch_size))
#
#         try:
#             agent_model.rollout_buffer.get = modified_get
#             rollout_data = agent_model.rollout_buffer.get(batch_size=None)
#             if rollout_data is not None:
#                 steps_trained = len(rollout_data.observations)
#                 print(f"Agent model trained on {steps_trained} steps")
#                 observations_tensor = torch.FloatTensor(rollout_data.observations)
#                 actions_tensor = torch.FloatTensor(rollout_data.actions)
#                 policy = agent_model.policy
#                 optimizer = policy.optimizer
#                 values, log_probs, entropy = policy.evaluate_actions(observations_tensor, actions_tensor)
#                 value_loss = ((values - torch.FloatTensor(rollout_data.returns)) ** 2).mean()
#                 advantages_tensor = torch.FloatTensor(rollout_data.advantages)
#                 policy_loss = -(log_probs * advantages_tensor).mean()
#                 entropy_loss = -entropy.mean()
#                 loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
#                 optimizer.step()
#                 policy_loss_val = policy_loss.item()
#                 value_loss_val = value_loss.item()
#                 total_loss_val = loss.item()
#                 #print(
#                 #    f"Agent loss - Policy: {policy_loss_val:.4f}, Value: {value_loss_val:.4f}, Total: {total_loss_val:.4f}")
#             else:
#                 print("Agent model: No valid data for training")
#         finally:
#             agent_model.rollout_buffer.get = orig_get
#     except Exception as e:
#         print(f"Error during agent training: {str(e)}")
#         traceback.print_exc()
#     agent_model.rollout_buffer.reset()
#     print("Agent buffer reset")
#     if policy_loss_val is not None:
#         return [policy_loss_val, value_loss_val, total_loss_val]
#     return None




def perform_training_phase(
    *,
    train_both: bool = False,
    training_local_master: bool = False,
    training_agent: bool = False,
    training_global_master: bool = False,
    master_model=None,
    agent_model=None,
    last_agent_obs_8d: list | None = None,
    last_lm_obs_25d: np.ndarray | None = None,
    last_lm2_obs_25d: np.ndarray | None = None,
    last_gm_obs_25d: np.ndarray | None = None,
    results: Dict[str, List[Any]] | None = None,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    n_ppo_epochs: int = 1,
    n_value_epochs: int = 0,
    last_done: bool = False,
    training_episode: int = -1,
) -> Dict[str, Any]:
    """
    Hierarchical training step used by ``run_unified`` and ``training_handler``.
    Ignores ``clip_range`` / ``n_ppo_epochs`` / ``n_value_epochs`` for now (single epoch manual PPO).
    """
    from src import project_globals

    results = results or init_training_results()
    ld = np.array([1.0 if last_done else 0.0], dtype=np.float32)

    mp_pre = mp_post = ag_pre = ag_post = float("nan")
    try:
        mp_pre = _sb3_policy_flat_l2(master_model.model.policy)
        ag_pre = _sb3_policy_flat_l2(agent_model.policy)
    except Exception:
        pass

    master_losses = None
    ml_per_buf: dict[str, list[float] | None] = {"LM1": None, "LM2": None, "GM": None}
    ml_losses: list[list[float]] = []
    if train_both or training_local_master or training_global_master:
        # Train shared master sequentially on LM1 → LM2 → GM buffers.
        if (train_both or training_local_master) and len(project_globals.local_master_rollout_buffers) >= 2:
            if last_lm_obs_25d is not None:
                r = train_shared_master_on_buffer(
                    master_model, last_lm_obs_25d,
                    project_globals.local_master_rollout_buffers[0],
                    last_done=ld, ent_coef=ent_coef, vf_coef=vf_coef,
                )
                if r:
                    ml_losses.append(r)
                    ml_per_buf["LM1"] = [float(r[0]), float(r[1]), float(r[2])]
            if last_lm2_obs_25d is not None:
                r = train_shared_master_on_buffer(
                    master_model, last_lm2_obs_25d,
                    project_globals.local_master_rollout_buffers[1],
                    last_done=ld, ent_coef=ent_coef, vf_coef=vf_coef,
                )
                if r:
                    ml_losses.append(r)
                    ml_per_buf["LM2"] = [float(r[0]), float(r[1]), float(r[2])]
        if (train_both or training_global_master) and project_globals.global_master_rollout_buffer is not None:
            if last_gm_obs_25d is not None:
                r = train_shared_master_on_buffer(
                    master_model, last_gm_obs_25d,
                    project_globals.global_master_rollout_buffer,
                    last_done=ld, ent_coef=ent_coef, vf_coef=vf_coef,
                )
                if r:
                    ml_losses.append(r)
                    ml_per_buf["GM"] = [float(r[0]), float(r[1]), float(r[2])]
        if ml_losses:
            master_losses = [
                float(np.mean([x[0] for x in ml_losses])),
                float(np.mean([x[1] for x in ml_losses])),
                float(np.mean([x[2] for x in ml_losses])),
            ]

    agent_losses = None
    if (train_both or training_agent) and last_agent_obs_8d is not None:
        agent_losses = train_agents_with_bootstrap(
            agent_model, last_agent_obs_8d, last_done=ld, ent_coef=ent_coef, vf_coef=vf_coef,
        )

    try:
        mp_post = _sb3_policy_flat_l2(master_model.model.policy)
        ag_post = _sb3_policy_flat_l2(agent_model.policy)
    except Exception:
        pass

    record_losses(
        master_losses,
        results["master_policy_losses"],
        results["master_value_losses"],
        results["master_total_losses"],
    )
    record_losses(
        agent_losses,
        results["agent_policy_losses"],
        results["agent_value_losses"],
        results["agent_total_losses"],
    )

    summary: Dict[str, Any] = {
        "training_episode": int(training_episode),
        "train_both": bool(train_both),
        "training_local_master": bool(training_local_master),
        "training_agent": bool(training_agent),
        "training_global_master": bool(training_global_master),
        "last_done": float(ld.flatten()[0]) if ld.size else 0.0,
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "master_avg_policy_loss": master_losses[0] if master_losses else None,
        "master_avg_value_loss": master_losses[1] if master_losses else None,
        "master_avg_total_loss": master_losses[2] if master_losses else None,
        "lm1_policy_loss": ml_per_buf["LM1"][0] if ml_per_buf["LM1"] else None,
        "lm1_value_loss": ml_per_buf["LM1"][1] if ml_per_buf["LM1"] else None,
        "lm1_total_loss": ml_per_buf["LM1"][2] if ml_per_buf["LM1"] else None,
        "lm2_policy_loss": ml_per_buf["LM2"][0] if ml_per_buf["LM2"] else None,
        "lm2_value_loss": ml_per_buf["LM2"][1] if ml_per_buf["LM2"] else None,
        "lm2_total_loss": ml_per_buf["LM2"][2] if ml_per_buf["LM2"] else None,
        "gm_policy_loss": ml_per_buf["GM"][0] if ml_per_buf["GM"] else None,
        "gm_value_loss": ml_per_buf["GM"][1] if ml_per_buf["GM"] else None,
        "gm_total_loss": ml_per_buf["GM"][2] if ml_per_buf["GM"] else None,
        "agent_policy_loss": agent_losses[0] if agent_losses else None,
        "agent_value_loss": agent_losses[1] if agent_losses else None,
        "agent_total_loss": agent_losses[2] if agent_losses else None,
        "master_param_l2norm_pre": mp_pre,
        "master_param_l2norm_post": mp_post,
        "agent_param_l2norm_pre": ag_pre,
        "agent_param_l2norm_post": ag_post,
        "delta_master_param_l2norm": mp_post - mp_pre if mp_pre == mp_pre and mp_post == mp_post else float("nan"),
        "delta_agent_param_l2norm": ag_post - ag_pre if ag_pre == ag_pre and ag_post == ag_post else float("nan"),
    }
    results.setdefault("ppo_training_log", []).append(summary)
    return summary

