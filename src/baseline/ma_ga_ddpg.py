import copy
import csv
import logging
import os
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baseline.vn_maddpg import JointReplayBuffer, OUNoise, hard_update, one_hot_from_logits, soft_update


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def canonicalize_ma_ga_algorithm_name(algorithm_name: str | None) -> str:    
    """Normalize supported aliases to the canonical MA-GA DDPG algorithm name."""

    normalized_name = (algorithm_name or "ma_ga_ddpg").lower()
    if normalized_name in {"attention-maddpg", "a_maddpg"}:
        return "attention_maddpg"
    if normalized_name in {"ma-ga-ddpg", "maga_ddpg", "maga"}:
        return "ma_ga_ddpg"
    if normalized_name in {"attention_maddpg", "ma_ga_ddpg"}:
        return normalized_name
    raise ValueError(
        f"Unsupported algorithm '{algorithm_name}'. "
        "Expected one of: attention_maddpg, ma_ga_ddpg."
    )


class AttentionActorNetwork(nn.Module):
    """Actor network that attends over nearby vehicle representations."""

    def __init__(self, vehicle_state_dim: int, action_dim: int, hidden_dim: int, attention_heads: int) -> None:
        """Build the encoder, attention block, and action decoder."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vehicle_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        observation_matrix: torch.Tensor,
        ego_index: int,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict action logits for one controlled agent from the joint observation."""

        encoded_matrix = self.encoder(observation_matrix)
        ego_encoding = encoded_matrix[:, ego_index : ego_index + 1, :]

        valid_mask = observation_matrix.abs().sum(dim=-1) > 1e-6
        key_padding_mask = ~valid_mask
        key_padding_mask[:, ego_index] = True

        fallback_mask = key_padding_mask.all(dim=1)
        safe_key_padding_mask = key_padding_mask.clone()
        if torch.any(fallback_mask):
            safe_key_padding_mask[fallback_mask, ego_index] = False

        attention_output, attention_weights = self.attention(
            query=ego_encoding,
            key=encoded_matrix,
            value=encoded_matrix,
            key_padding_mask=safe_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        attention_output = attention_output.squeeze(1)
        attention_weights = attention_weights.squeeze(1)

        if torch.any(fallback_mask):
            attention_output[fallback_mask] = 0.0
            attention_weights[fallback_mask] = 0.0

        attention_weights[:, ego_index] = 0.0
        normalizer = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = torch.where(
            normalizer > 0.0,
            attention_weights / normalizer.clamp(min=1e-8),
            torch.zeros_like(attention_weights),
        )

        decoder_input = torch.cat([encoded_matrix[:, ego_index, :], attention_output], dim=-1)
        action_logits = self.decoder(decoder_input)
        if return_attention:
            return action_logits, attention_weights
        return action_logits, None


class CriticNetwork(nn.Module):
    """Centralized critic that scores a state and joint action."""

    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dim: int) -> None:
        """Initialize the critic multilayer perceptron."""

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        """Estimate the value of the provided state and joint action."""

        critic_input = torch.cat([state, joint_action], dim=-1)
        return self.network(critic_input)


class MAGADDPGTrainer:
    """Train or evaluate the attention MADDPG and MA-GA DDPG baselines."""

    def __init__(self, experiment_config, env_config: dict[str, Any]) -> None:
        """Create the environment, networks, optimizers, and logging state."""

        self.experiment_config = experiment_config
        self.algorithm = canonicalize_ma_ga_algorithm_name(experiment_config.ALGORITHM)
        self.use_safety_inspector = self.algorithm == "ma_ga_ddpg"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make("RELintersection-v0", render_mode=experiment_config.RENDER_MODE, config=env_config)
        self.env_config = env_config
        self.num_agents = len(env_config["controlled_cars"])
        self.action_dim = int(experiment_config.ACTION_SPACE_SIZE)
        self.vehicle_state_dim = int(experiment_config.AGENT_STATE_SIZE)
        self.max_vehicle_count = int(experiment_config.CARS_AMOUNT)
        self.state_dim = self.max_vehicle_count * self.vehicle_state_dim
        self.total_episodes = int(experiment_config.EPISODES_PER_CYCLE * experiment_config.CYCLES)

        hidden_dim = int(experiment_config.MA_GA_HIDDEN_DIM)
        attention_heads = int(experiment_config.MA_GA_ATTENTION_HEADS)
        joint_action_dim = self.num_agents * self.action_dim

        self.actors = [
            AttentionActorNetwork(
                vehicle_state_dim=self.vehicle_state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                attention_heads=attention_heads,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_actors = [copy.deepcopy(actor).to(self.device) for actor in self.actors]
        self.critics = [
            CriticNetwork(self.state_dim, joint_action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_critics = [copy.deepcopy(critic).to(self.device) for critic in self.critics]

        for actor, target_actor in zip(self.actors, self.target_actors):
            hard_update(actor, target_actor)
        for critic, target_critic in zip(self.critics, self.target_critics):
            hard_update(critic, target_critic)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=float(experiment_config.MA_GA_ACTOR_LR))
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=float(experiment_config.MA_GA_CRITIC_LR))
            for critic in self.critics
        ]

        self.replay_buffer = JointReplayBuffer(
            capacity=int(experiment_config.MA_GA_BUFFER_SIZE),
            prioritized=False,
            alpha=0.0,
        )

        self.noise_processes = [
            OUNoise(self.action_dim, sigma=float(experiment_config.MA_GA_NOISE_SIGMA))
            for _ in range(self.num_agents)
        ]
        self.total_steps = 0
        self.update_steps = 0

        self.history = {
            "episode_rewards": [],
            "actor_losses": [],
            "critic_losses": [],
            "noise_scales": [],
            "success_flags": [],
            "collision_flags": [],
            "episode_lengths": [],
        }

    def close(self) -> None:
        """Release the underlying Gym environment."""

        self.env.close()

    def _prepare_observation(self, observation: Any) -> np.ndarray:
        """Pad and sanitize an observation into the fixed vehicle matrix shape."""

        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.ndim == 1:
            observation_array = observation_array.reshape(-1, self.vehicle_state_dim)

        prepared_observation = np.zeros((self.max_vehicle_count, self.vehicle_state_dim), dtype=np.float32)
        rows_to_copy = min(self.max_vehicle_count, observation_array.shape[0])
        prepared_observation[:rows_to_copy] = observation_array[:rows_to_copy]

        controlled_vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        for agent_index in range(min(self.num_agents, len(controlled_vehicles), prepared_observation.shape[0])):
            if getattr(controlled_vehicles[agent_index], "is_arrived", False):
                prepared_observation[agent_index] = 0.0

        return prepared_observation

    def _flatten_state(self, prepared_observation: np.ndarray) -> np.ndarray:
        """Flatten a prepared observation matrix into the critic state vector."""

        flattened_state = prepared_observation.reshape(-1).astype(np.float32)
        if flattened_state.shape[0] != self.state_dim:
            padded_state = np.zeros(self.state_dim, dtype=np.float32)
            copy_count = min(self.state_dim, flattened_state.shape[0])
            padded_state[:copy_count] = flattened_state[:copy_count]
            return padded_state
        return flattened_state

    def _get_noise_scale(self, episode_index: int) -> float:
        """Interpolate the exploration noise scale for the current episode."""

        initial_noise = float(self.experiment_config.MA_GA_INITIAL_NOISE)
        final_noise = float(self.experiment_config.MA_GA_FINAL_NOISE)
        if self.total_episodes <= 1:
            return final_noise
        progress_ratio = min(1.0, max(0.0, (episode_index - 1) / max(self.total_episodes - 1, 1)))
        return initial_noise + (final_noise - initial_noise) * progress_ratio

    def _extract_rewards(self, info: dict[str, Any], shared_reward: float, active_mask: np.ndarray) -> np.ndarray:
        """Build a per-agent reward vector from environment info and activity flags."""

        reward_values = info.get("agents_rewards")
        if reward_values is None:
            rewards = np.full(self.num_agents, shared_reward, dtype=np.float32)
        else:
            rewards = np.asarray(reward_values[: self.num_agents], dtype=np.float32)
            if rewards.shape[0] < self.num_agents:
                padded_rewards = np.full(self.num_agents, shared_reward, dtype=np.float32)
                padded_rewards[: rewards.shape[0]] = rewards
                rewards = padded_rewards
        return rewards * active_mask

    def _extract_done_flags(
        self,
        info: dict[str, Any],
        episode_finished: bool,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Return per-agent termination flags, forcing inactive agents to done."""

        if episode_finished:
            done_flags = np.ones(self.num_agents, dtype=np.float32)
        else:
            info_done_flags = info.get("agents_terminated")
            if info_done_flags is None:
                done_flags = np.zeros(self.num_agents, dtype=np.float32)
            else:
                done_flags = np.asarray(info_done_flags[: self.num_agents], dtype=np.float32)
                if done_flags.shape[0] < self.num_agents:
                    padded_done_flags = np.zeros(self.num_agents, dtype=np.float32)
                    padded_done_flags[: done_flags.shape[0]] = done_flags
                    done_flags = padded_done_flags
        done_flags = np.where(active_mask > 0.0, done_flags, 1.0)
        return done_flags.astype(np.float32)

    def _has_any_controlled_collision(self, info: dict[str, Any]) -> bool:
        """Check whether any controlled vehicle has collided in the current step."""

        if bool(info.get("crashed", False)):
            return True
        controlled_vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        return any(getattr(vehicle, "crashed", False) for vehicle in controlled_vehicles[: self.num_agents])

    def _actions_to_one_hot_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert discrete per-agent actions into a flattened joint one-hot tensor."""

        return F.one_hot(actions.long(), num_classes=self.action_dim).float().reshape(actions.shape[0], -1)

    def _is_present_vehicle(self, vehicle_state: np.ndarray) -> bool:
        """Return whether a padded vehicle slot contains a real vehicle state."""

        return bool(np.any(np.abs(vehicle_state) > 1e-6))

    def _select_interaction_objects(
        self,
        ego_index: int,
        observation_matrix: np.ndarray,
        attention_matrix: np.ndarray,
    ) -> list[int]:
        """Choose the most relevant neighboring vehicles for safety inspection."""

        ego_state = observation_matrix[ego_index]
        if not self._is_present_vehicle(ego_state):
            return []

        interaction_distance = float(self.experiment_config.MA_GA_INTERACTION_DISTANCE)
        attention_threshold = float(self.experiment_config.MA_GA_ATTENTION_THRESHOLD)
        interaction_limit = int(self.experiment_config.MA_GA_MAX_INTERACTION_OBJECTS)

        candidate_indices: list[int] = []
        fallback_candidates: list[tuple[float, float, int]] = []
        ego_position = ego_state[:2]

        for vehicle_index, vehicle_state in enumerate(observation_matrix):
            if vehicle_index == ego_index or not self._is_present_vehicle(vehicle_state):
                continue

            distance = float(np.linalg.norm(vehicle_state[:2] - ego_position))
            if distance > interaction_distance:
                continue

            attention_weight = float(attention_matrix[ego_index, vehicle_index])
            fallback_candidates.append((distance, -attention_weight, vehicle_index))
            if attention_weight >= attention_threshold:
                candidate_indices.append(vehicle_index)

        candidate_indices.sort(
            key=lambda vehicle_index: (
                -float(attention_matrix[ego_index, vehicle_index]),
                float(np.linalg.norm(observation_matrix[vehicle_index, :2] - ego_position)),
            )
        )

        selected_indices = list(candidate_indices[:interaction_limit])
        if len(selected_indices) < interaction_limit:
            fallback_candidates.sort()
            for _, _, vehicle_index in fallback_candidates:
                if vehicle_index in selected_indices:
                    continue
                selected_indices.append(vehicle_index)
                if len(selected_indices) >= interaction_limit:
                    break

        return selected_indices

    def _get_controlled_priority_order(
        self,
        observation_matrix: np.ndarray,
        attention_matrix: np.ndarray,
    ) -> list[int]:
        """Rank controlled agents by attention and proximity for conflict resolution."""

        global_attention = np.sum(attention_matrix[: self.num_agents], axis=0)
        center_distances = np.linalg.norm(observation_matrix[: self.num_agents, :2], axis=1)
        present_agents = [
            agent_index
            for agent_index in range(self.num_agents)
            if self._is_present_vehicle(observation_matrix[agent_index])
        ]
        absent_agents = [
            agent_index
            for agent_index in range(self.num_agents)
            if agent_index not in present_agents
        ]
        return sorted(
            present_agents,
            key=lambda agent_index: (
                float(global_attention[agent_index]),
                -float(center_distances[agent_index]),
            ),
            reverse=True,
        ) + absent_agents

    def _predict_vehicle_trajectory(
        self,
        observation_matrix: np.ndarray,
        vehicle_index: int,
        candidate_action: int | None,
    ) -> np.ndarray:
        """Roll out a short constant-velocity trajectory under an optional speed change."""

        vehicle_state = observation_matrix[vehicle_index]
        horizon = int(self.experiment_config.MA_GA_PREDICTION_STEPS)
        trajectory = np.zeros((horizon, 2), dtype=np.float32)
        if not self._is_present_vehicle(vehicle_state):
            return trajectory

        current_position = vehicle_state[:2].astype(np.float32).copy()
        current_velocity = vehicle_state[2:4].astype(np.float32).copy()
        speed_norm = float(np.linalg.norm(current_velocity))

        if candidate_action is not None and speed_norm > 1e-6:
            if candidate_action == 0:
                target_scale = float(self.experiment_config.MA_GA_SLOWER_SCALE)
            else:
                target_scale = float(self.experiment_config.MA_GA_FASTER_SCALE)
            adjusted_speed = min(
                float(self.experiment_config.MA_GA_MAX_PREDICTED_SPEED),
                speed_norm * target_scale,
            )
            current_velocity = (current_velocity / speed_norm) * adjusted_speed

        prediction_delta = float(self.experiment_config.MA_GA_PREDICTION_DELTA)
        for step_index in range(horizon):
            current_position = current_position + current_velocity * prediction_delta
            trajectory[step_index] = current_position

        return trajectory

    def _count_focus_conflicts(
        self,
        observation_matrix: np.ndarray,
        focus_agent_index: int,
        candidate_action: int,
        action_plan: list[int],
        monitored_indices: list[int],
    ) -> int:
        """Count predicted close-approach conflicts for one agent action choice."""

        if not monitored_indices:
            return 0

        focus_trajectory = self._predict_vehicle_trajectory(observation_matrix, focus_agent_index, candidate_action)
        conflict_distance = float(self.experiment_config.MA_GA_CONFLICT_DISTANCE)
        conflict_count = 0

        for other_index in monitored_indices:
            if other_index == focus_agent_index:
                continue

            other_action = action_plan[other_index] if other_index < self.num_agents else None
            other_trajectory = self._predict_vehicle_trajectory(observation_matrix, other_index, other_action)
            distances = np.linalg.norm(focus_trajectory - other_trajectory, axis=1)
            conflict_count += int(np.sum(distances < conflict_distance))

        return conflict_count

    def _apply_safety_inspector(
        self,
        observation_matrix: np.ndarray,
        proposed_actions: tuple[int, ...],
        attention_matrix: np.ndarray,
        policy_logits: np.ndarray,
    ) -> tuple[tuple[int, ...], int]:
        """Adjust proposed actions to reduce forecast conflicts among agents."""

        if not self.use_safety_inspector:
            return proposed_actions, 0

        selected_interactions = {
            agent_index: self._select_interaction_objects(agent_index, observation_matrix, attention_matrix)
            for agent_index in range(self.num_agents)
        }
        priority_order = self._get_controlled_priority_order(observation_matrix, attention_matrix)

        corrected_actions = list(proposed_actions)
        processed_agents: list[int] = []
        override_count = 0

        for agent_index in priority_order:
            monitored_indices = list(selected_interactions[agent_index])
            monitored_indices.extend(processed_agents)
            monitored_indices = list(dict.fromkeys(index for index in monitored_indices if index != agent_index))
            if not monitored_indices:
                processed_agents.append(agent_index)
                continue

            current_action = corrected_actions[agent_index]
            best_action = current_action
            best_conflict_count = self._count_focus_conflicts(
                observation_matrix=observation_matrix,
                focus_agent_index=agent_index,
                candidate_action=current_action,
                action_plan=corrected_actions,
                monitored_indices=monitored_indices,
            )
            best_logit = float(policy_logits[agent_index, best_action])

            for candidate_action in range(self.action_dim):
                candidate_conflict_count = self._count_focus_conflicts(
                    observation_matrix=observation_matrix,
                    focus_agent_index=agent_index,
                    candidate_action=candidate_action,
                    action_plan=corrected_actions,
                    monitored_indices=monitored_indices,
                )
                candidate_logit = float(policy_logits[agent_index, candidate_action])

                if candidate_conflict_count < best_conflict_count:
                    best_action = candidate_action
                    best_conflict_count = candidate_conflict_count
                    best_logit = candidate_logit
                    continue

                if candidate_conflict_count == best_conflict_count:
                    if best_action != current_action and candidate_action == current_action:
                        best_action = candidate_action
                        best_logit = candidate_logit
                        continue
                    if candidate_action == best_action:
                        continue
                    if best_action == current_action:
                        continue
                    if candidate_logit > best_logit:
                        best_action = candidate_action
                        best_logit = candidate_logit

            if best_action != current_action:
                corrected_actions[agent_index] = best_action
                override_count += 1

            processed_agents.append(agent_index)

        return tuple(corrected_actions), override_count

    def _select_actions(
        self,
        observation_matrix: np.ndarray,
        episode_index: int,
        agent_finished: np.ndarray,
        deterministic: bool,
    ) -> tuple[tuple[int, ...], float, int]:
        """Sample or greedily choose actions, then optionally apply safety overrides."""

        noise_scale = 0.0 if deterministic else self._get_noise_scale(episode_index)
        proposed_actions: list[int] = []
        attention_vectors: list[np.ndarray] = []
        policy_logits: list[np.ndarray] = []

        observation_tensor = torch.tensor(
            observation_matrix,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        for agent_index in range(self.num_agents):
            if agent_finished[agent_index]:
                proposed_actions.append(0)
                attention_vectors.append(np.zeros(self.max_vehicle_count, dtype=np.float32))
                policy_logits.append(np.zeros(self.action_dim, dtype=np.float32))
                continue

            with torch.no_grad():
                action_logits_tensor, attention_weights_tensor = self.actors[agent_index](
                    observation_tensor,
                    ego_index=agent_index,
                    return_attention=True,
                )
            action_logits = action_logits_tensor.squeeze(0).cpu().numpy()
            attention_weights = attention_weights_tensor.squeeze(0).cpu().numpy()

            policy_logits.append(action_logits.astype(np.float32))
            attention_vectors.append(attention_weights.astype(np.float32))

            if deterministic:
                proposed_actions.append(int(np.argmax(action_logits)))
                continue

            noisy_logits = action_logits + self.noise_processes[agent_index].sample(noise_scale)
            proposed_actions.append(int(np.argmax(noisy_logits)))

        proposed_action_tuple = tuple(proposed_actions)
        attention_matrix = np.stack(attention_vectors).astype(np.float32)
        policy_logit_matrix = np.stack(policy_logits).astype(np.float32)
        corrected_actions, override_count = self._apply_safety_inspector(
            observation_matrix=observation_matrix,
            proposed_actions=proposed_action_tuple,
            attention_matrix=attention_matrix,
            policy_logits=policy_logit_matrix,
        )
        return corrected_actions, noise_scale, override_count

    def _update_networks(self) -> tuple[float | None, float | None]:
        """Run one or more replay updates and return mean actor and critic losses."""

        minimum_buffer_size = max(
            int(self.experiment_config.MA_GA_BATCH_SIZE),
            int(self.experiment_config.MA_GA_WARMUP_STEPS),
        )
        if len(self.replay_buffer) < minimum_buffer_size:
            return None, None
        if self.total_steps % int(self.experiment_config.MA_GA_TRAIN_EVERY) != 0:
            return None, None

        actor_loss_values: list[float] = []
        critic_loss_values: list[float] = []

        for _ in range(int(self.experiment_config.MA_GA_UPDATES_PER_STEP)):
            batch = self.replay_buffer.sample(int(self.experiment_config.MA_GA_BATCH_SIZE), beta=1.0)

            state_batch = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
            obs_batch = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
            reward_batch = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
            next_state_batch = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
            next_obs_batch = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
            done_batch = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)
            active_mask_batch = torch.tensor(batch["active_mask"], dtype=torch.float32, device=self.device)

            joint_action_batch = self._actions_to_one_hot_tensor(action_batch)

            with torch.no_grad():
                target_next_actions = []
                for agent_index in range(self.num_agents):
                    target_action_logits, _ = self.target_actors[agent_index](
                        next_obs_batch,
                        ego_index=agent_index,
                        return_attention=False,
                    )
                    target_next_actions.append(one_hot_from_logits(target_action_logits))
                target_joint_action_batch = torch.cat(target_next_actions, dim=-1)

            for agent_index in range(self.num_agents):
                active_mask = active_mask_batch[:, agent_index : agent_index + 1]

                current_q_values = self.critics[agent_index](state_batch, joint_action_batch)
                with torch.no_grad():
                    target_q_values = reward_batch[:, agent_index : agent_index + 1] + (
                        float(self.experiment_config.MA_GA_GAMMA)
                        * (1.0 - done_batch[:, agent_index : agent_index + 1])
                        * self.target_critics[agent_index](next_state_batch, target_joint_action_batch)
                    )

                td_error = target_q_values - current_q_values
                critic_denominator = torch.clamp(active_mask.sum(), min=1.0)
                critic_loss = (active_mask * td_error.pow(2)).sum() / critic_denominator
                self.critic_optimizers[agent_index].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critics[agent_index].parameters(),
                    float(self.experiment_config.MA_GA_MAX_GRAD_NORM),
                )
                self.critic_optimizers[agent_index].step()
                critic_loss_values.append(float(critic_loss.item()))

                for critic_param in self.critics[agent_index].parameters():
                    critic_param.requires_grad = False

                policy_actions = []
                for other_agent_index in range(self.num_agents):
                    other_action_logits, _ = self.actors[other_agent_index](
                        obs_batch,
                        ego_index=other_agent_index,
                        return_attention=False,
                    )
                    if other_agent_index == agent_index:
                        policy_action = F.gumbel_softmax(
                            other_action_logits,
                            tau=float(self.experiment_config.MA_GA_GUMBEL_TAU),
                            hard=True,
                        )
                    else:
                        with torch.no_grad():
                            policy_action = one_hot_from_logits(other_action_logits)
                    policy_actions.append(policy_action)

                policy_joint_action_batch = torch.cat(policy_actions, dim=-1)
                actor_denominator = torch.clamp(active_mask.sum(), min=1.0)
                actor_loss = -(
                    self.critics[agent_index](state_batch, policy_joint_action_batch) * active_mask
                ).sum() / actor_denominator

                self.actor_optimizers[agent_index].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actors[agent_index].parameters(),
                    float(self.experiment_config.MA_GA_MAX_GRAD_NORM),
                )
                self.actor_optimizers[agent_index].step()
                actor_loss_values.append(float(actor_loss.item()))

                for critic_param in self.critics[agent_index].parameters():
                    critic_param.requires_grad = True

            self.update_steps += 1
            if self.update_steps % int(self.experiment_config.MA_GA_TARGET_UPDATE_INTERVAL) == 0:
                for actor, target_actor in zip(self.actors, self.target_actors):
                    soft_update(actor, target_actor, float(self.experiment_config.MA_GA_TAU))
                for critic, target_critic in zip(self.critics, self.target_critics):
                    soft_update(critic, target_critic, float(self.experiment_config.MA_GA_TAU))

        average_actor_loss = float(np.mean(actor_loss_values)) if actor_loss_values else None
        average_critic_loss = float(np.mean(critic_loss_values)) if critic_loss_values else None
        return average_actor_loss, average_critic_loss

    def _run_episode(self, episode_index: int, training: bool) -> dict[str, Any]:
        """Execute a full episode and collect training or evaluation metrics."""

        raw_observation, _ = self.env.reset()
        prepared_observation = self._prepare_observation(raw_observation)
        agent_finished = np.zeros(self.num_agents, dtype=bool)

        for noise_process in self.noise_processes:
            noise_process.reset()

        episode_reward = 0.0
        episode_length = 0
        collision_occurred = False
        actor_loss_values = []
        critic_loss_values = []
        last_noise_scale = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action_tuple, last_noise_scale, _ = self._select_actions(
                observation_matrix=prepared_observation,
                episode_index=episode_index,
                agent_finished=agent_finished,
                deterministic=not training,
            )

            if self.experiment_config.RENDER_MODE is not None:
                self.env.render()

            next_raw_observation, reward, terminated, truncated, info = self.env.step(action_tuple)
            next_prepared_observation = self._prepare_observation(next_raw_observation)

            current_state = self._flatten_state(prepared_observation)
            next_state = self._flatten_state(next_prepared_observation)
            active_mask = (~agent_finished).astype(np.float32)
            done_flags = self._extract_done_flags(
                info=info,
                episode_finished=bool(terminated or truncated),
                active_mask=active_mask,
            )
            rewards = self._extract_rewards(info=info, shared_reward=float(reward), active_mask=active_mask)

            if training:
                transition = {
                    "state": current_state,
                    "obs": np.asarray(prepared_observation, dtype=np.float32),
                    "actions": np.asarray(action_tuple, dtype=np.int64),
                    "rewards": rewards.astype(np.float32),
                    "next_state": next_state,
                    "next_obs": np.asarray(next_prepared_observation, dtype=np.float32),
                    "dones": done_flags.astype(np.float32),
                    "active_mask": active_mask.astype(np.float32),
                }
                self.replay_buffer.add(transition)
                average_actor_loss, average_critic_loss = self._update_networks()
                if average_actor_loss is not None:
                    actor_loss_values.append(average_actor_loss)
                if average_critic_loss is not None:
                    critic_loss_values.append(average_critic_loss)

            self.total_steps += 1
            episode_reward += float(reward)
            episode_length += 1
            collision_occurred = collision_occurred or self._has_any_controlled_collision(info)

            agent_finished = np.logical_or(agent_finished, done_flags.astype(bool))
            prepared_observation = next_prepared_observation

        episode_success = bool(terminated and not truncated and not collision_occurred)
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "collision": collision_occurred,
            "success": episode_success,
            "noise_scale": last_noise_scale,
            "actor_loss": float(np.mean(actor_loss_values)) if actor_loss_values else None,
            "critic_loss": float(np.mean(critic_loss_values)) if critic_loss_values else None,
        }

    def _checkpoint_path_candidates(self, path_prefix: str) -> list[str]:
        """Generate checkpoint filename variants compatible with older naming schemes."""

        if not path_prefix:
            return []
        base_prefix = path_prefix[:-4] if path_prefix.endswith(".zip") else path_prefix
        return [
            path_prefix,
            base_prefix,
            f"{base_prefix}_{self.algorithm}.pt",
            f"{base_prefix}_baseline_{self.algorithm}.pt",
        ]

    def _save_checkpoint(self) -> str:
        """Persist the current actor and critic weights to disk."""

        checkpoint_path = f"{self.experiment_config.SAVE_MODEL_DIRECTORY}_{self.algorithm}.pt"
        checkpoint = {
            "algorithm": self.algorithm,
            "num_agents": self.num_agents,
            "actors": [actor.state_dict() for actor in self.actors],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved baseline checkpoint to %s", checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self) -> bool:
        """Load the first compatible checkpoint found from the configured path candidates."""

        for checkpoint_candidate in self._checkpoint_path_candidates(self.experiment_config.LOAD_MODEL_DIRECTORY):
            if not checkpoint_candidate or not os.path.exists(checkpoint_candidate) or os.path.isdir(checkpoint_candidate):
                continue

            try:
                checkpoint = torch.load(checkpoint_candidate, map_location=self.device)
            except Exception as checkpoint_error:
                logger.warning("Failed to load checkpoint %s: %s", checkpoint_candidate, checkpoint_error)
                continue

            checkpoint_num_agents = checkpoint.get("num_agents")
            if checkpoint_num_agents != self.num_agents:
                logger.warning(
                    "Skipping checkpoint %s because it was saved for %s agents, while the current environment has %s.",
                    checkpoint_candidate,
                    checkpoint_num_agents,
                    self.num_agents,
                )
                continue

            for actor, actor_state in zip(self.actors, checkpoint["actors"]):
                actor.load_state_dict(actor_state)
            for target_actor, actor_state in zip(self.target_actors, checkpoint["target_actors"]):
                target_actor.load_state_dict(actor_state)
            for critic, critic_state in zip(self.critics, checkpoint["critics"]):
                critic.load_state_dict(critic_state)
            for target_critic, critic_state in zip(self.target_critics, checkpoint["target_critics"]):
                target_critic.load_state_dict(critic_state)

            logger.info("Loaded baseline checkpoint from %s", checkpoint_candidate)
            return True

        logger.warning("No compatible baseline checkpoint was found for %s.", self.algorithm)
        return False

    def _write_progress_csv(self) -> str:
        """Write per-episode training metrics to the baseline progress CSV."""

        baseline_log_dir = os.path.join(self.experiment_config.EXPERIMENT_PATH, "baseline_logs")
        os.makedirs(baseline_log_dir, exist_ok=True)
        csv_path = os.path.join(baseline_log_dir, "progress.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "episode",
                    "reward",
                    "actor_loss",
                    "critic_loss",
                    "noise_scale",
                    "success",
                    "collision",
                    "episode_length",
                ]
            )
            for episode_index in range(len(self.history["episode_rewards"])):
                writer.writerow(
                    [
                        episode_index + 1,
                        self.history["episode_rewards"][episode_index],
                        self.history["actor_losses"][episode_index],
                        self.history["critic_losses"][episode_index],
                        self.history["noise_scales"][episode_index],
                        self.history["success_flags"][episode_index],
                        self.history["collision_flags"][episode_index],
                        self.history["episode_lengths"][episode_index],
                    ]
                )

        return csv_path

    def _plot_training_curves(self) -> None:
        """Render reward, loss, and running-rate plots for the current history."""

        if not self.history["episode_rewards"]:
            return

        plots_dir = os.path.join(self.experiment_config.EXPERIMENT_PATH, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        episodes = np.arange(1, len(self.history["episode_rewards"]) + 1)
        plt.style.use("ggplot")

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.history["episode_rewards"], color="#1f77b4", linewidth=2)
        plt.title(f"{self.algorithm} Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.algorithm}_episode_rewards.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        valid_actor_losses = [loss if loss is not None else np.nan for loss in self.history["actor_losses"]]
        valid_critic_losses = [loss if loss is not None else np.nan for loss in self.history["critic_losses"]]
        plt.plot(episodes, valid_actor_losses, label="Actor Loss", linewidth=2)
        plt.plot(episodes, valid_critic_losses, label="Critic Loss", linewidth=2)
        plt.title(f"{self.algorithm} Losses")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.algorithm}_losses.png"))
        plt.close()

        success_rate = np.cumsum(np.asarray(self.history["success_flags"], dtype=np.float32)) / episodes
        collision_rate = np.cumsum(np.asarray(self.history["collision_flags"], dtype=np.float32)) / episodes

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, success_rate, label="Success Rate", linewidth=2)
        plt.plot(episodes, collision_rate, label="Collision Rate", linewidth=2)
        plt.title(f"{self.algorithm} Running Rates")
        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.algorithm}_rates.png"))
        plt.close()

    def train(self) -> tuple[int, dict[str, list[Any]]]:
        """Train for all configured episodes and return collisions plus metric history."""

        collision_count = 0

        for episode_index in range(1, self.total_episodes + 1):
            episode_result = self._run_episode(episode_index=episode_index, training=True)
            collision_count += int(episode_result["collision"])

            self.history["episode_rewards"].append(episode_result["episode_reward"])
            self.history["actor_losses"].append(episode_result["actor_loss"])
            self.history["critic_losses"].append(episode_result["critic_loss"])
            self.history["noise_scales"].append(episode_result["noise_scale"])
            self.history["success_flags"].append(int(episode_result["success"]))
            self.history["collision_flags"].append(int(episode_result["collision"]))
            self.history["episode_lengths"].append(episode_result["episode_length"])

            logger.info(
                "[%s] Episode %d/%d | reward=%.2f | success=%s | collision=%s | noise=%.4f",
                self.algorithm,
                episode_index,
                self.total_episodes,
                episode_result["episode_reward"],
                episode_result["success"],
                episode_result["collision"],
                episode_result["noise_scale"],
            )

        self._save_checkpoint()
        self._write_progress_csv()
        self._plot_training_curves()
        return collision_count, self.history

    def evaluate(self) -> tuple[list[float], list[bool]]:
        """Run inference episodes and return rewards alongside success indicators."""

        evaluation_rewards = []
        evaluation_successes = []
        evaluation_episodes = int(getattr(self.experiment_config, "MA_GA_EVAL_EPISODES", 5))

        for episode_index in range(1, evaluation_episodes + 1):
            episode_result = self._run_episode(episode_index=episode_index, training=False)
            evaluation_rewards.append(episode_result["episode_reward"])
            evaluation_successes.append(bool(episode_result["success"]))
            logger.info(
                "[%s] Evaluation episode %d/%d | reward=%.2f | success=%s",
                self.algorithm,
                episode_index,
                evaluation_episodes,
                episode_result["episode_reward"],
                episode_result["success"],
            )

        return evaluation_rewards, evaluation_successes


def run_ma_ga_ddpg_experiment(experiment_config, env_config: dict[str, Any]):
    """Create a trainer, optionally load weights, and run training or evaluation."""

    trainer = MAGADDPGTrainer(experiment_config, env_config)

    try:
        should_try_loading = bool(experiment_config.LOAD_PREVIOUS_WEIGHT and experiment_config.LOAD_MODEL_DIRECTORY)
        if should_try_loading:
            trainer._load_checkpoint()

        if experiment_config.ONLY_INFERENCE:
            logger.info("Running %s in inference-only mode", trainer.algorithm)
            evaluation_rewards, evaluation_successes = trainer.evaluate()
            logger.info(
                "[%s] Average evaluation reward: %.2f | success rate: %.2f",
                trainer.algorithm,
                float(np.mean(evaluation_rewards)) if evaluation_rewards else 0.0,
                float(np.mean(evaluation_successes)) if evaluation_successes else 0.0,
            )
            return trainer, None, int(np.sum(np.logical_not(evaluation_successes)))

        logger.info("Running %s in training mode", trainer.algorithm)
        collision_count, history = trainer.train()
        logger.info("%s training completed. Total collisions: %s", trainer.algorithm, collision_count)
        return trainer, history, collision_count
    finally:
        trainer.close()
