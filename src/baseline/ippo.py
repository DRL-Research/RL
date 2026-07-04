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

from torch.distributions import Categorical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def forward(self, observation: torch.Tensor) -> Categorical:
        logits = self.network(observation)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def get(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )


class IPPOTrainer:
    def __init__(self, experiment_config, env_config: dict[str, Any]) -> None:
        self.experiment_config = experiment_config
        self.algorithm = "ippo"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(experiment_config.ENV_ID, render_mode=experiment_config.RENDER_MODE, config=env_config)
        self.env_config = env_config
        self.num_agents = len(env_config["controlled_cars"])
        self.action_dim = int(experiment_config.ACTION_SPACE_SIZE)
        self.observation_dim = int(experiment_config.AGENT_STATE_SIZE)
        self.total_episodes = int(experiment_config.EPISODES_PER_CYCLE * experiment_config.CYCLES)

        hidden_dim = int(getattr(experiment_config, "IPPO_HIDDEN_DIM", 64))
        self.lr_actor = float(getattr(experiment_config, "IPPO_ACTOR_LR", 3e-4))
        self.lr_critic = float(getattr(experiment_config, "IPPO_CRITIC_LR", 1e-3))
        self.gamma = float(getattr(experiment_config, "IPPO_GAMMA", 0.99))
        self.gae_lambda = float(getattr(experiment_config, "IPPO_GAE_LAMBDA", 0.95))
        self.clip_epsilon = float(getattr(experiment_config, "IPPO_CLIP_EPSILON", 0.2))
        self.entropy_coef = float(getattr(experiment_config, "IPPO_ENTROPY_COEF", 0.01))
        self.ppo_epochs = int(getattr(experiment_config, "IPPO_EPOCHS", 10))
        self.batch_size = int(getattr(experiment_config, "IPPO_BATCH_SIZE", 64))
        self.rollout_steps = int(getattr(experiment_config, "IPPO_ROLLOUT_STEPS", 2048))

        self.actors = [
            ActorNetwork(self.observation_dim, self.action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.critics = [
            CriticNetwork(self.observation_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics]

        self.buffers = [RolloutBuffer() for _ in range(self.num_agents)]
        
        self.history = {
            "episode_rewards": [],
            "actor_losses": [],
            "critic_losses": [],
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

    def _has_any_controlled_collision(self, info: dict[str, Any]) -> bool:
        if bool(info.get("crashed", False)):
            return True
        controlled_vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        return any(getattr(vehicle, "crashed", False) for vehicle in controlled_vehicles[: self.num_agents])

    def _update_agent(self, agent_index: int):
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.buffers[agent_index].get()
        
        if len(states) == 0:
            return None, None
            
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(old_values, dtype=torch.float32, device=self.device).squeeze(-1)

        with torch.no_grad():
            next_values = self.critics[agent_index](next_states_t).squeeze(-1)
            
            advantages = torch.zeros_like(rewards_t, device=self.device)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones_t[t]
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones_t[t]
                    nextvalues = old_values_t[t+1]
                delta = rewards_t[t] + self.gamma * nextvalues * nextnonterminal - old_values_t[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + old_values_t
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                if len(batch_idx) == 0: continue
                
                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                
                dist = self.actors[agent_index](b_states)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                values = self.critics[agent_index](b_states).squeeze(-1)
                critic_loss = F.mse_loss(values, b_returns)
                
                self.actor_optimizers[agent_index].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent_index].parameters(), 0.5)
                self.actor_optimizers[agent_index].step()
                
                self.critic_optimizers[agent_index].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critics[agent_index].parameters(), 0.5)
                self.critic_optimizers[agent_index].step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.buffers[agent_index].clear()
        
        return np.mean(actor_losses) if actor_losses else None, np.mean(critic_losses) if critic_losses else None

    def _run_episode(self, episode_index: int, training: bool) -> dict[str, Any]:
        raw_observation, _ = self.env.reset()
        prepared_observation = self._prepare_observation(raw_observation)
        agent_observations = self._extract_agent_observations(prepared_observation)
        agent_finished = np.zeros(self.num_agents, dtype=bool)

        episode_reward = 0.0
        episode_length = 0
        collision_occurred = False
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = []
            log_probs = []
            values = []

            for agent_index in range(self.num_agents):
                if agent_finished[agent_index]:
                    actions.append(0)
                    log_probs.append(0.0)
                    values.append(0.0)
                    continue

                obs_t = torch.tensor(agent_observations[agent_index], dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    dist = self.actors[agent_index](obs_t)
                    value = self.critics[agent_index](obs_t)
                    
                    if not training:
                        action = torch.argmax(dist.logits, dim=-1).item()
                        log_prob = dist.log_prob(torch.tensor([action], device=self.device)).item()
                    else:
                        action_t = dist.sample()
                        action = action_t.item()
                        log_prob = dist.log_prob(action_t).item()
                
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value.item())

            if self.experiment_config.RENDER_MODE is not None:
                self.env.render()

            next_raw_observation, reward, terminated, truncated, info = self.env.step(tuple(actions))
            next_prepared_observation = self._prepare_observation(next_raw_observation)
            next_agent_observations = self._extract_agent_observations(next_prepared_observation)

            active_mask = (~agent_finished).astype(np.float32)
            
            episode_finished = bool(terminated or truncated)
            if episode_finished:
                done_flags = np.ones(self.num_agents, dtype=np.float32)
            else:
                info_done_flags = info.get("agents_terminated")
                if info_done_flags is None:
                    done_flags = np.zeros(self.num_agents, dtype=np.float32)
                else:
                    done_flags = np.asarray(info_done_flags[: self.num_agents], dtype=np.float32)
            done_flags = np.where(active_mask > 0.0, done_flags, 1.0)
            
            reward_values = info.get("agents_rewards")
            if reward_values is None:
                rewards = np.full(self.num_agents, float(reward), dtype=np.float32)
            else:
                rewards = np.asarray(reward_values[: self.num_agents], dtype=np.float32)
            rewards = rewards * active_mask

            if training:
                for agent_index in range(self.num_agents):
                    if active_mask[agent_index] > 0:
                        self.buffers[agent_index].add(
                            agent_observations[agent_index],
                            actions[agent_index],
                            rewards[agent_index],
                            next_agent_observations[agent_index],
                            done_flags[agent_index],
                            log_probs[agent_index],
                            values[agent_index]
                        )

            episode_reward += float(reward)
            episode_length += 1
            collision_occurred = collision_occurred or self._has_any_controlled_collision(info)

            agent_finished = np.logical_or(agent_finished, done_flags.astype(bool))
            agent_observations = next_agent_observations

        episode_success = bool(terminated and not truncated and not collision_occurred)

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "collision": collision_occurred,
            "success": episode_success,
        }

    def train(self) -> tuple[int, dict[str, list[Any]]]:
        collision_count = 0
        total_steps = 0
        
        for episode_index in range(1, self.total_episodes + 1):
            episode_result = self._run_episode(episode_index=episode_index, training=True)
            collision_count += int(episode_result["collision"])
            total_steps += episode_result["episode_length"]

            self.history["episode_rewards"].append(episode_result["episode_reward"])
            self.history["success_flags"].append(int(episode_result["success"]))
            self.history["collision_flags"].append(int(episode_result["collision"]))
            self.history["episode_lengths"].append(episode_result["episode_length"])

            if total_steps >= self.rollout_steps:
                actor_losses = []
                critic_losses = []
                for agent_index in range(self.num_agents):
                    a_loss, c_loss = self._update_agent(agent_index)
                    if a_loss is not None:
                        actor_losses.append(a_loss)
                    if c_loss is not None:
                        critic_losses.append(c_loss)
                
                avg_a_loss = np.mean(actor_losses) if actor_losses else 0.0
                avg_c_loss = np.mean(critic_losses) if critic_losses else 0.0
                
                # Append to history for the last 'rollout_steps' episodes
                rem = len(self.history["episode_rewards"]) - len(self.history["actor_losses"])
                self.history["actor_losses"].extend([avg_a_loss] * rem)
                self.history["critic_losses"].extend([avg_c_loss] * rem)
                total_steps = 0

            logger.info(
                "[%s] Episode %d/%d | reward=%.2f | success=%s | collision=%s",
                self.algorithm,
                episode_index,
                self.total_episodes,
                episode_result["episode_reward"],
                episode_result["success"],
                episode_result["collision"],
            )
            
        # Pad losses if remaining episodes didn't trigger an update
        rem = len(self.history["episode_rewards"]) - len(self.history["actor_losses"])
        if rem > 0:
            self.history["actor_losses"].extend([0.0] * rem)
            self.history["critic_losses"].extend([0.0] * rem)

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

    def _save_checkpoint(self) -> str:
        checkpoint_path = f"{self.experiment_config.SAVE_MODEL_DIRECTORY}_{self.algorithm}.pt"
        checkpoint = {
            "algorithm": self.algorithm,
            "num_agents": self.num_agents,
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved baseline checkpoint to %s", checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self) -> bool:
        checkpoint_candidate = f"{self.experiment_config.LOAD_MODEL_DIRECTORY}_{self.algorithm}.pt"
        if not os.path.exists(checkpoint_candidate):
            logger.warning("No compatible baseline checkpoint was found for %s.", self.algorithm)
            return False

        try:
            checkpoint = torch.load(checkpoint_candidate, map_location=self.device)
        except Exception as checkpoint_error:
            logger.warning("Failed to load checkpoint %s: %s", checkpoint_candidate, checkpoint_error)
            return False

        for actor, actor_state in zip(self.actors, checkpoint["actors"]):
            actor.load_state_dict(actor_state)
        for critic, critic_state in zip(self.critics, checkpoint["critics"]):
            critic.load_state_dict(critic_state)

        logger.info("Loaded baseline checkpoint from %s", checkpoint_candidate)
        return True

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

def run_ippo_experiment(experiment_config, env_config: dict[str, Any]):
    trainer = IPPOTrainer(experiment_config, env_config)

    try:
        should_try_loading = bool(experiment_config.LOAD_PREVIOUS_WEIGHT and experiment_config.LOAD_MODEL_DIRECTORY)
        if should_try_loading:
            trainer._load_checkpoint()

        if getattr(experiment_config, "ONLY_INFERENCE", False):
            logger.info("Running IPPO in inference-only mode")
            evaluation_rewards, evaluation_successes = trainer.evaluate()
            logger.info(
                "[%s] Average evaluation reward: %.2f | success rate: %.2f",
                trainer.algorithm,
                float(np.mean(evaluation_rewards)) if evaluation_rewards else 0.0,
                float(np.mean(evaluation_successes)) if evaluation_successes else 0.0,
            )
            return trainer, None, int(np.sum(np.logical_not(evaluation_successes)))

        logger.info("Running IPPO in training mode")
        collision_count, history = trainer.train()
        logger.info("IPPO training completed. Total collisions: %s", collision_count)
        return trainer, history, collision_count
    finally:
        trainer.close()
