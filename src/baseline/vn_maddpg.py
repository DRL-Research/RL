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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def canonicalize_algorithm_name(algorithm_name: str | None) -> str:
    normalized_name = (algorithm_name or "experiment").lower()
    if normalized_name == "baseline":
        return "vn_maddpg"
    if normalized_name in {"experiment", "maddpg", "vn_maddpg"}:
        return normalized_name
    raise ValueError(
        f"Unsupported algorithm '{algorithm_name}'. "
        "Expected one of: experiment, baseline, maddpg, vn_maddpg."
    )


def soft_update(source_network: nn.Module, target_network: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(source_network: nn.Module, target_network: nn.Module) -> None:
    target_network.load_state_dict(source_network.state_dict())


def one_hot_from_logits(logits: torch.Tensor) -> torch.Tensor:
    action_indices = torch.argmax(logits, dim=-1)
    return F.one_hot(action_indices, num_classes=logits.shape[-1]).float()


class OUNoise:
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2) -> None:
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size, dtype=np.float32) * self.mu

    def reset(self) -> None:
        self.state = np.ones(self.size, dtype=np.float32) * self.mu

    def sample(self, scale: float) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state = self.state + dx
        return self.state * scale


class ActorNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([state, joint_action], dim=-1)
        return self.network(critic_input)


class JointReplayBuffer:
    def __init__(self, capacity: int, prioritized: bool, alpha: float, epsilon: float = 1e-6) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.epsilon = epsilon
        self.storage: list[dict[str, np.ndarray]] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: dict[str, np.ndarray]) -> None:
        if len(self.storage) == 0:
            priority = 1.0
        else:
            priority = float(np.max(self.priorities[: len(self.storage)]))
            priority = max(priority, 1.0)

        if len(self.storage) < self.capacity:
            self.storage.append(transition)
            insert_index = len(self.storage) - 1
        else:
            if self.prioritized:
                insert_index = int(np.argmin(self.priorities[: len(self.storage)]))
            else:
                insert_index = self.position
                self.position = (self.position + 1) % self.capacity
            self.storage[insert_index] = transition

        self.priorities[insert_index] = max(priority, self.epsilon)

    def sample(self, batch_size: int, beta: float) -> dict[str, np.ndarray]:
        current_size = len(self.storage)
        effective_batch_size = min(batch_size, current_size)
        replace = current_size < effective_batch_size

        if self.prioritized:
            scaled_priorities = self.priorities[:current_size] ** self.alpha
            if float(np.sum(scaled_priorities)) <= 0.0:
                sample_probabilities = np.ones(current_size, dtype=np.float32) / current_size
            else:
                sample_probabilities = scaled_priorities / np.sum(scaled_priorities)
            indices = np.random.choice(current_size, size=effective_batch_size, replace=replace, p=sample_probabilities)
            importance_weights = (current_size * sample_probabilities[indices]) ** (-beta)
            importance_weights = importance_weights / np.max(importance_weights)
        else:
            indices = np.random.choice(current_size, size=effective_batch_size, replace=replace)
            importance_weights = np.ones(effective_batch_size, dtype=np.float32)

        transitions = [self.storage[index] for index in indices]
        batch = {
            "indices": indices.astype(np.int64),
            "weights": importance_weights.astype(np.float32),
            "states": np.stack([transition["state"] for transition in transitions]).astype(np.float32),
            "obs": np.stack([transition["obs"] for transition in transitions]).astype(np.float32),
            "actions": np.stack([transition["actions"] for transition in transitions]).astype(np.int64),
            "rewards": np.stack([transition["rewards"] for transition in transitions]).astype(np.float32),
            "next_states": np.stack([transition["next_state"] for transition in transitions]).astype(np.float32),
            "next_obs": np.stack([transition["next_obs"] for transition in transitions]).astype(np.float32),
            "dones": np.stack([transition["dones"] for transition in transitions]).astype(np.float32),
            "active_mask": np.stack([transition["active_mask"] for transition in transitions]).astype(np.float32),
        }
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if not self.prioritized:
            return
        for buffer_index, priority in zip(indices, priorities):
            self.priorities[int(buffer_index)] = max(float(priority), self.epsilon)


class BaselineTrainer:
    def __init__(self, experiment_config, env_config: dict[str, Any]) -> None:
        self.experiment_config = experiment_config
        self.algorithm = canonicalize_algorithm_name(experiment_config.ALGORITHM)
        self.use_variable_noise = self.algorithm == "vn_maddpg"
        self.use_prioritized_replay = self.algorithm == "vn_maddpg"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make("RELintersection-v0", render_mode=experiment_config.RENDER_MODE, config=env_config)
        self.env_config = env_config
        self.num_agents = len(env_config["controlled_cars"])
        self.action_dim = int(experiment_config.ACTION_SPACE_SIZE)
        self.observation_dim = int(experiment_config.AGENT_STATE_SIZE)
        self.state_dim = int(experiment_config.CARS_AMOUNT * experiment_config.AGENT_STATE_SIZE)
        self.total_episodes = int(experiment_config.EPISODES_PER_CYCLE * experiment_config.CYCLES)

        joint_action_dim = self.num_agents * self.action_dim
        hidden_dim = int(experiment_config.BASELINE_HIDDEN_DIM)

        self.actors = [
            ActorNetwork(self.observation_dim, self.action_dim, hidden_dim).to(self.device)
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
            torch.optim.Adam(actor.parameters(), lr=experiment_config.BASELINE_ACTOR_LR)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=experiment_config.BASELINE_CRITIC_LR)
            for critic in self.critics
        ]

        self.replay_buffer = JointReplayBuffer(
            capacity=int(experiment_config.BASELINE_BUFFER_SIZE),
            prioritized=self.use_prioritized_replay,
            alpha=float(experiment_config.BASELINE_PRIORITY_ALPHA),
        )

        self.noise_processes = [OUNoise(self.action_dim) for _ in range(self.num_agents)]
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
        self.env.close()

    def _prepare_observation(self, observation: Any) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.ndim == 1:
            observation_array = observation_array.reshape(-1, self.observation_dim)

        target_rows = int(self.experiment_config.CARS_AMOUNT)
        prepared_observation = np.zeros((target_rows, self.observation_dim), dtype=np.float32)
        rows_to_copy = min(target_rows, observation_array.shape[0])
        prepared_observation[:rows_to_copy] = observation_array[:rows_to_copy]

        unwrapped_env = self.env.unwrapped
        controlled_vehicles = getattr(unwrapped_env, "controlled_vehicles", [])
        for agent_index in range(min(self.num_agents, len(controlled_vehicles), prepared_observation.shape[0])):
            if getattr(controlled_vehicles[agent_index], "is_arrived", False):
                prepared_observation[agent_index] = 0.0

        return prepared_observation

    def _extract_agent_observations(self, prepared_observation: np.ndarray) -> np.ndarray:
        return np.asarray(prepared_observation[: self.num_agents], dtype=np.float32)

    def _flatten_state(self, prepared_observation: np.ndarray) -> np.ndarray:
        flattened_state = prepared_observation.reshape(-1).astype(np.float32)
        if flattened_state.shape[0] != self.state_dim:
            padded_state = np.zeros(self.state_dim, dtype=np.float32)
            copy_count = min(self.state_dim, flattened_state.shape[0])
            padded_state[:copy_count] = flattened_state[:copy_count]
            return padded_state
        return flattened_state

    def _get_noise_scale(self, episode_index: int) -> float:
        initial_noise = float(self.experiment_config.BASELINE_INITIAL_NOISE)
        final_noise = float(self.experiment_config.BASELINE_FINAL_NOISE)
        if not self.use_variable_noise:
            return initial_noise
        remaining_ratio = max(0.0, (self.total_episodes - episode_index + 1) / max(self.total_episodes, 1))
        return final_noise + (initial_noise - final_noise) * remaining_ratio

    def _get_beta(self, episode_index: int) -> float:
        if not self.use_prioritized_replay:
            return 1.0
        beta_start = float(self.experiment_config.BASELINE_PRIORITY_BETA_START)
        if self.total_episodes <= 1:
            progress_ratio = 1.0
        else:
            progress_ratio = min(1.0, (episode_index - 1) / (self.total_episodes - 1))
        return beta_start + (1.0 - beta_start) * progress_ratio

    def _select_actions(
        self,
        agent_observations: np.ndarray,
        episode_index: int,
        agent_finished: np.ndarray,
        deterministic: bool,
    ) -> tuple[tuple[int, ...], float]:
        noise_scale = 0.0 if deterministic else self._get_noise_scale(episode_index)
        selected_actions: list[int] = []

        for agent_index in range(self.num_agents):
            if agent_finished[agent_index]:
                selected_actions.append(0)
                continue

            if (not deterministic) and self.total_steps < int(self.experiment_config.BASELINE_WARMUP_STEPS):
                selected_actions.append(int(np.random.randint(self.action_dim)))
                continue

            observation_tensor = torch.tensor(
                agent_observations[agent_index],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            with torch.no_grad():
                action_logits = self.actors[agent_index](observation_tensor).squeeze(0).cpu().numpy()

            if deterministic:
                selected_actions.append(int(np.argmax(action_logits)))
                continue

            noisy_logits = action_logits + self.noise_processes[agent_index].sample(noise_scale)
            selected_actions.append(int(np.argmax(noisy_logits)))

        return tuple(selected_actions), noise_scale

    def _extract_rewards(self, info: dict[str, Any], shared_reward: float, active_mask: np.ndarray) -> np.ndarray:
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
        if bool(info.get("crashed", False)):
            return True

        controlled_vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        return any(getattr(vehicle, "crashed", False) for vehicle in controlled_vehicles[: self.num_agents])

    def _actions_to_one_hot_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_dim).float().reshape(actions.shape[0], -1)

    def _update_networks(self, episode_index: int) -> tuple[float | None, float | None]:
        minimum_buffer_size = max(
            int(self.experiment_config.BASELINE_BATCH_SIZE),
            int(self.experiment_config.BASELINE_WARMUP_STEPS),
        )

        if len(self.replay_buffer) < minimum_buffer_size:
            return None, None
        if self.total_steps % int(self.experiment_config.BASELINE_TRAIN_EVERY) != 0:
            return None, None

        actor_loss_values: list[float] = []
        critic_loss_values: list[float] = []
        beta = self._get_beta(episode_index)

        for _ in range(int(self.experiment_config.BASELINE_UPDATES_PER_STEP)):
            batch = self.replay_buffer.sample(int(self.experiment_config.BASELINE_BATCH_SIZE), beta)

            state_batch = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
            obs_batch = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
            reward_batch = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
            next_state_batch = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
            next_obs_batch = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
            done_batch = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)
            active_mask_batch = torch.tensor(batch["active_mask"], dtype=torch.float32, device=self.device)
            weight_batch = torch.tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(-1)

            joint_action_batch = self._actions_to_one_hot_tensor(action_batch)

            with torch.no_grad():
                target_next_actions = [
                    one_hot_from_logits(self.target_actors[agent_index](next_obs_batch[:, agent_index, :]))
                    for agent_index in range(self.num_agents)
                ]
                target_joint_action_batch = torch.cat(target_next_actions, dim=-1)

            td_error_values = []

            for agent_index in range(self.num_agents):
                active_mask = active_mask_batch[:, agent_index : agent_index + 1]

                current_q_values = self.critics[agent_index](state_batch, joint_action_batch)
                with torch.no_grad():
                    target_q_values = reward_batch[:, agent_index : agent_index + 1] + (
                        float(self.experiment_config.BASELINE_GAMMA)
                        * (1.0 - done_batch[:, agent_index : agent_index + 1])
                        * self.target_critics[agent_index](next_state_batch, target_joint_action_batch)
                    )

                td_error = target_q_values - current_q_values
                td_error_values.append(torch.abs(td_error.detach()))

                critic_denominator = torch.clamp(active_mask.sum(), min=1.0)
                critic_loss = ((weight_batch * active_mask * td_error.pow(2)).sum()) / critic_denominator
                self.critic_optimizers[agent_index].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critics[agent_index].parameters(),
                    float(self.experiment_config.BASELINE_MAX_GRAD_NORM),
                )
                self.critic_optimizers[agent_index].step()
                critic_loss_values.append(float(critic_loss.item()))

                for critic_param in self.critics[agent_index].parameters():
                    critic_param.requires_grad = False

                policy_actions = []
                for other_agent_index in range(self.num_agents):
                    other_agent_logits = self.actors[other_agent_index](obs_batch[:, other_agent_index, :])
                    if other_agent_index == agent_index:
                        policy_action = F.gumbel_softmax(
                            other_agent_logits,
                            tau=float(self.experiment_config.BASELINE_GUMBEL_TAU),
                            hard=True,
                        )
                    else:
                        with torch.no_grad():
                            policy_action = one_hot_from_logits(other_agent_logits)
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
                    float(self.experiment_config.BASELINE_MAX_GRAD_NORM),
                )
                self.actor_optimizers[agent_index].step()
                actor_loss_values.append(float(actor_loss.item()))

                for critic_param in self.critics[agent_index].parameters():
                    critic_param.requires_grad = True

            if self.use_prioritized_replay:
                td_error_tensor = torch.cat(td_error_values, dim=1)
                averaged_priorities = (
                    (td_error_tensor * active_mask_batch).sum(dim=1)
                    / torch.clamp(active_mask_batch.sum(dim=1), min=1.0)
                )
                new_priorities = averaged_priorities.detach().cpu().numpy() + 1e-6
                self.replay_buffer.update_priorities(batch["indices"], new_priorities)

            self.update_steps += 1
            if self.update_steps % int(self.experiment_config.BASELINE_TARGET_UPDATE_INTERVAL) == 0:
                for actor, target_actor in zip(self.actors, self.target_actors):
                    soft_update(actor, target_actor, float(self.experiment_config.BASELINE_TAU))
                for critic, target_critic in zip(self.critics, self.target_critics):
                    soft_update(critic, target_critic, float(self.experiment_config.BASELINE_TAU))

        average_actor_loss = float(np.mean(actor_loss_values)) if actor_loss_values else None
        average_critic_loss = float(np.mean(critic_loss_values)) if critic_loss_values else None
        return average_actor_loss, average_critic_loss

    def _run_episode(self, episode_index: int, training: bool) -> dict[str, Any]:
        raw_observation, _ = self.env.reset()
        prepared_observation = self._prepare_observation(raw_observation)
        agent_observations = self._extract_agent_observations(prepared_observation)
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
            action_tuple, last_noise_scale = self._select_actions(
                agent_observations=agent_observations,
                episode_index=episode_index,
                agent_finished=agent_finished,
                deterministic=not training,
            )

            if self.experiment_config.RENDER_MODE is not None:
                self.env.render()

            next_raw_observation, reward, terminated, truncated, info = self.env.step(action_tuple)
            next_prepared_observation = self._prepare_observation(next_raw_observation)
            next_agent_observations = self._extract_agent_observations(next_prepared_observation)

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
                    "obs": np.asarray(agent_observations, dtype=np.float32),
                    "actions": np.asarray(action_tuple, dtype=np.int64),
                    "rewards": rewards.astype(np.float32),
                    "next_state": next_state,
                    "next_obs": np.asarray(next_agent_observations, dtype=np.float32),
                    "dones": done_flags.astype(np.float32),
                    "active_mask": active_mask.astype(np.float32),
                }
                self.replay_buffer.add(transition)
                average_actor_loss, average_critic_loss = self._update_networks(episode_index)
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
            agent_observations = next_agent_observations

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
        for checkpoint_candidate in self._checkpoint_path_candidates(self.experiment_config.LOAD_MODEL_DIRECTORY):
            if not checkpoint_candidate:
                continue
            if not os.path.exists(checkpoint_candidate):
                continue
            if os.path.isdir(checkpoint_candidate):
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
        evaluation_rewards = []
        evaluation_successes = []
        evaluation_episodes = int(getattr(self.experiment_config, "BASELINE_EVAL_EPISODES", 5))

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


def run_baseline_experiment(experiment_config, env_config: dict[str, Any]):
    trainer = BaselineTrainer(experiment_config, env_config)

    try:
        should_try_loading = bool(experiment_config.LOAD_PREVIOUS_WEIGHT and experiment_config.LOAD_MODEL_DIRECTORY)
        if should_try_loading:
            trainer._load_checkpoint()

        if experiment_config.ONLY_INFERENCE:
            logger.info("Running baseline in inference-only mode")
            evaluation_rewards, evaluation_successes = trainer.evaluate()
            logger.info(
                "[%s] Average evaluation reward: %.2f | success rate: %.2f",
                trainer.algorithm,
                float(np.mean(evaluation_rewards)) if evaluation_rewards else 0.0,
                float(np.mean(evaluation_successes)) if evaluation_successes else 0.0,
            )
            return trainer, None, int(np.sum(np.logical_not(evaluation_successes)))

        logger.info("Running baseline in training mode")
        collision_count, history = trainer.train()
        logger.info("Baseline training completed. Total collisions: %s", collision_count)
        return trainer, history, collision_count
    finally:
        trainer.close()
