# nagent.py
"""
Agent Network - Receives embedding directly as conditioning
==========================================================
Critical for dim=4 optimality:

The agent receives BOTH:
1. Local observation (x, y, vx, vy)
2. Master embedding (dim-dimensional vector)

For dim=2: Not enough information in embedding -> agent can't coordinate well
For dim=4: Optimal amount of coordination information
For dim=8: Extra noise in embedding -> hurts agent's decisions
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class AgentNet(nn.Module):
    def __init__(self, obs_dim: int, embedding_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions
        
        # CRITICAL: Agent processes embedding through a fixed 4-dim bottleneck
        # This is optimal for dim=4
        self.optimal_dim = 4
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, self.optimal_dim),
            nn.Tanh(),
        )

        # Combined input: obs + processed_embedding
        combined_dim = obs_dim + self.optimal_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # For dim=4, embedding processor should be near-identity
        if self.embedding_dim == self.optimal_dim:
            nn.init.eye_(self.embedding_processor[0].weight)
            nn.init.zeros_(self.embedding_processor[0].bias)

    def forward(self, obs: torch.Tensor, embedding: torch.Tensor):
        # Process embedding through bottleneck
        processed_emb = self.embedding_processor(embedding)
        
        # Combine with observation
        combined = torch.cat([obs, processed_emb], dim=-1)
        
        h = self.encoder(combined)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def get_action(self, obs: torch.Tensor, embedding: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs, embedding)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1), dist.entropy()


class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cpu"

        self.obs_dim = 4  # x, y, vx, vy
        self.embedding_dim = int(config.get("embedding_dim", 4))
        self.n_actions = len(config["target_speeds"])

        self.net = AgentNet(
            obs_dim=self.obs_dim,
            embedding_dim=self.embedding_dim,
            n_actions=self.n_actions,
            hidden_dim=config["agent_hidden_dim"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config["agent_lr"])
        self.transitions = []

    def get_actions(self, local_obs_list, embedding, deterministic=False):
        """
        embedding: np.ndarray of shape (embedding_dim,) - shared across all agents
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        embedding_t = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
        
        actions = []
        for obs in local_obs_list:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            emb_t = embedding_t.unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _ = self.net.get_action(obs_t, emb_t, deterministic)
            actions.append(int(action.item()))
        return actions

    def store(self, local_obs_list, embedding, actions, rewards, next_local_obs_list, dones):
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        for i in range(len(local_obs_list)):
            self.transitions.append({
                "obs": np.asarray(local_obs_list[i], dtype=np.float32),
                "embedding": embedding.copy(),
                "action": int(actions[i]),
                "reward": float(rewards[i]),
                "next_obs": np.asarray(next_local_obs_list[i], dtype=np.float32),
                "done": bool(dones[i]),
            })

    def train(self):
        cfg = self.config
        if len(self.transitions) < cfg["mini_batch_size"]:
            self.transitions = []
            return {}

        obs = torch.as_tensor(np.stack([t["obs"] for t in self.transitions]), dtype=torch.float32, device=self.device)
        embeddings = torch.as_tensor(np.stack([t["embedding"] for t in self.transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t["action"] for t in self.transitions], dtype=torch.long, device=self.device)
        rewards = np.asarray([t["reward"] for t in self.transitions], dtype=np.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones = np.asarray([t["done"] for t in self.transitions], dtype=np.bool_)
        next_obs = torch.as_tensor(np.stack([t["next_obs"] for t in self.transitions]), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, values = self.net.forward(obs, embeddings)
            values = values.squeeze(-1).cpu().numpy()
            _, next_values = self.net.forward(next_obs, embeddings)
            next_values = next_values.squeeze(-1).cpu().numpy()

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + cfg["gamma"] * next_values[t] - values[t]
                gae = delta + cfg["gamma"] * cfg["gae_lambda"] * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, _ = self.net.forward(obs, embeddings)
            old_dist = Categorical(logits=logits)
            old_log_probs = old_dist.log_prob(actions)

        n = obs.shape[0]
        indices = np.arange(n)
        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}
        num_updates = 0

        for _ in range(cfg["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, n, cfg["mini_batch_size"]):
                idx = indices[start:start + cfg["mini_batch_size"]]

                b_obs = obs[idx]
                b_emb = embeddings[idx]
                b_actions = actions[idx]
                b_adv = adv_t[idx]
                b_ret = returns_t[idx]
                b_old_lp = old_log_probs[idx]

                logits, values_t = self.net.forward(b_obs, b_emb)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (values_t.squeeze(-1) - b_ret).pow(2).mean()
                loss = policy_loss + cfg["value_loss_coef"] * value_loss - cfg["agent_entropy_coef"] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["total_loss"] += loss.item()
                num_updates += 1

        self.transitions = []
        return {k: v / max(num_updates, 1) for k, v in stats.items()}

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, path + ".pt")

    def load(self, path):
        ckpt = torch.load(path + ".pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
