import copy
import csv
import logging
import os
import random
from collections import deque
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
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

class VDNReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
        
    def __len__(self):
        return len(self.buffer)

class VDNTrainer:
    def __init__(self, experiment_config, env_config: dict[str, Any]) -> None:
        self.experiment_config = experiment_config
        self.algorithm = "vdn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(experiment_config.ENV_ID, render_mode=experiment_config.RENDER_MODE, config=env_config)
        self.env_config = env_config
        self.num_agents = len(env_config["controlled_cars"])
        self.action_dim = int(experiment_config.ACTION_SPACE_SIZE)
        self.observation_dim = int(experiment_config.AGENT_STATE_SIZE)
        self.total_episodes = int(experiment_config.EPISODES_PER_CYCLE * experiment_config.CYCLES)

        hidden_dim = int(getattr(experiment_config, "VDN_HIDDEN_DIM", 64))
        self.lr = float(getattr(experiment_config, "VDN_LR", 1e-3))
        self.gamma = float(getattr(experiment_config, "VDN_GAMMA", 0.99))
        self.batch_size = int(getattr(experiment_config, "VDN_BATCH_SIZE", 64))
        buffer_size = int(getattr(experiment_config, "VDN_BUFFER_SIZE", 100000))
        self.target_update_interval = getattr(experiment_config, "VDN_TARGET_UPDATE_INTERVAL", 100)
        
        self.epsilon = float(getattr(experiment_config, "VDN_EPSILON_START", 1.0))
        self.epsilon_min = float(getattr(experiment_config, "VDN_EPSILON_MIN", 0.05))
        self.epsilon_decay = float(getattr(experiment_config, "VDN_EPSILON_DECAY", 0.995))

        self.q_networks = nn.ModuleList([
            QNetwork(self.observation_dim, self.action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ])
        
        self.target_q_networks = copy.deepcopy(self.q_networks)
        self.train_step = 0
        
        self.optimizer = torch.optim.Adam(self.q_networks.parameters(), lr=self.lr)

        self.buffer = VDNReplayBuffer(buffer_size)
        
        self.history = {
            "episode_rewards": [],
            "q_losses": [],
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

    def _update_network(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        q_tot = 0
        target_q_tot = 0
        
        for i in range(self.num_agents):
            q_i = self.q_networks[i](states_t[:, i])
            q_i_taken = q_i.gather(1, actions_t[:, i].unsqueeze(1)).squeeze(1)
            q_tot = q_tot + q_i_taken
            
            with torch.no_grad():
                target_q_i = self.target_q_networks[i](next_states_t[:, i])
                target_q_i_max = target_q_i.max(dim=1)[0]
                target_q_tot = target_q_tot + target_q_i_max
                
        # VDN targets
        y = rewards_t + self.gamma * (1 - dones_t) * target_q_tot
        
        loss = F.mse_loss(q_tot, y.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_networks.parameters(), 0.5)
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            for q_net, t_q_net in zip(self.q_networks, self.target_q_networks):
                t_q_net.load_state_dict(q_net.state_dict())
                
        return loss.item()

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
        
        losses = []

        while not terminated and not truncated:
            actions = []

            for agent_index in range(self.num_agents):
                if agent_finished[agent_index]:
                    actions.append(0)
                    continue
                
                if training and random.random() < self.epsilon:
                    action = random.randint(0, self.action_dim - 1)
                else:
                    obs_t = torch.tensor(agent_observations[agent_index], dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        q_vals = self.q_networks[agent_index](obs_t)
                        action = torch.argmax(q_vals, dim=-1).item()
                
                actions.append(action)

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

            joint_reward = float(np.sum(rewards))
            
            # Use max to get joint done flag (if any agent is active and env is done, or if all are done)
            joint_done = float(episode_finished)

            if training:
                # Store joint transition
                self.buffer.add(
                    agent_observations,
                    actions,
                    joint_reward,
                    next_agent_observations,
                    joint_done
                )
                
                loss = self._update_network()
                if loss is not None:
                    losses.append(loss)

            episode_reward += joint_reward
            episode_length += 1
            collision_occurred = collision_occurred or self._has_any_controlled_collision(info)

            agent_finished = np.logical_or(agent_finished, done_flags.astype(bool))
            agent_observations = next_agent_observations

        episode_success = bool(terminated and not truncated and not collision_occurred)
        
        if training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "collision": collision_occurred,
            "success": episode_success,
            "loss": np.mean(losses) if losses else 0.0
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
            self.history["q_losses"].append(episode_result["loss"])

            logger.info(
                "[%s] Episode %d/%d | reward=%.2f | loss=%.4f | epsilon=%.3f | success=%s | collision=%s",
                self.algorithm,
                episode_index,
                self.total_episodes,
                episode_result["episode_reward"],
                episode_result["loss"],
                self.epsilon,
                episode_result["success"],
                episode_result["collision"],
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

    def _save_checkpoint(self) -> str:
        checkpoint_path = f"{self.experiment_config.SAVE_MODEL_DIRECTORY}_{self.algorithm}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            "algorithm": self.algorithm,
            "num_agents": self.num_agents,
            "q_networks": [q.state_dict() for q in self.q_networks],
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

        for q_net, q_state in zip(self.q_networks, checkpoint["q_networks"]):
            q_net.load_state_dict(q_state)

        logger.info("Loaded baseline checkpoint from %s", checkpoint_candidate)
        return True

    def _write_progress_csv(self) -> None:
        baseline_log_dir = os.path.join(self.experiment_config.EXPERIMENT_PATH, "baseline_logs")
        os.makedirs(baseline_log_dir, exist_ok=True)
        csv_path = os.path.join(baseline_log_dir, "progress.csv")
        try:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Episode", "Reward", "QLoss", "Success", "Collision", "Length"
                ])
                for idx in range(len(self.history["episode_rewards"])):
                    writer.writerow([
                        idx + 1,
                        self.history["episode_rewards"][idx],
                        self.history["q_losses"][idx],
                        self.history["success_flags"][idx],
                        self.history["collision_flags"][idx],
                        self.history["episode_lengths"][idx],
                    ])
        except Exception as e:
            logger.error("Failed to write baseline progress CSV: %s", e)

    def _plot_training_curves(self) -> None:
        baseline_log_dir = os.path.join(self.experiment_config.EXPERIMENT_PATH, "baseline_logs")
        os.makedirs(baseline_log_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"[{self.algorithm.upper()}] Training Progress", fontsize=16)

        episodes = np.arange(1, len(self.history["episode_rewards"]) + 1)

        ax = axes[0, 0]
        ax.plot(episodes, self.history["episode_rewards"], label="Episode Reward", color="blue", alpha=0.6)
        ax.set_title("Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(episodes, self.history["q_losses"], label="Q Loss", color="red", alpha=0.6)
        ax.set_title("Losses")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend()

        ax = axes[1, 0]
        smoothed_success = np.convolve(self.history["success_flags"], np.ones(10) / 10, mode="valid")
        smoothed_collision = np.convolve(self.history["collision_flags"], np.ones(10) / 10, mode="valid")
        valid_episodes = np.arange(len(smoothed_success)) + 10
        ax.plot(valid_episodes, smoothed_success, label="Success (10-ep MA)", color="green")
        ax.plot(valid_episodes, smoothed_collision, label="Collision (10-ep MA)", color="red")
        ax.set_title("Success & Collision Rates")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(episodes, self.history["episode_lengths"], label="Episode Length", color="purple", alpha=0.6)
        ax.set_title("Episode Lengths")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(baseline_log_dir, f"training_curves_{self.algorithm}.png")
        plt.savefig(plot_path)
        plt.close()

def run_vdn_experiment(experiment_config, env_config: dict[str, Any]):
    logger.info("Initializing %s experiment", experiment_config.ALGORITHM)
    trainer = VDNTrainer(experiment_config, env_config)
    try:
        if getattr(experiment_config, "ONLY_INFERENCE", False):
            logger.info("Running baseline in inference-only mode")
            success = trainer._load_checkpoint()
            if not success:
                logger.warning("Proceeding with untrained models since checkpoint failed to load.")
            trainer.evaluate()
        else:
            logger.info("Running baseline in training mode")
            if getattr(experiment_config, "LOAD_PREVIOUS_WEIGHT", False):
                trainer._load_checkpoint()
            trainer.train()
    finally:
        trainer.close()
    return None, trainer.history, None
