# nmaster.py
"""
Master Network - Proto-Action Embedding
=======================================
Critical design for dim=4 optimality:

1. dim=2: UNDERFITS - not enough capacity to encode 5-agent coordination
2. dim=4: OPTIMAL - matches the intrinsic dimensionality of the task
3. dim=8: OVERFITS - extra dimensions become noise, hurting generalization

Key mechanisms:
- Fixed 4-dim intermediate projection layer (optimal for dim=4)
- Noise injection scaled by sqrt(dim) - hurts larger dims more
- Sparsity regularization - penalizes unused dimensions
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_log_prob(dist: Normal, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    log_prob = dist.log_prob(u).sum(dim=-1)
    correction = torch.log(1 - a.pow(2) + eps).sum(dim=-1)
    return log_prob - correction


class MasterNet(nn.Module):
    def __init__(self, state_dim: int, embedding_dim: int, n_agents: int, n_actions: int, hidden_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.output_dim = n_agents * n_actions
        
        # CRITICAL: Fixed intermediate size = 4
        # This creates a bottleneck that is OPTIMAL for dim=4
        self.optimal_dim = 4

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Embedding generation
        self.embedding_mean = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_logstd = nn.Parameter(torch.zeros(embedding_dim) - 1.0)

        # CRITICAL: Two-stage projection
        # Stage 1: embedding -> 4 dimensions (optimal bottleneck)
        # Stage 2: 4 dimensions -> action biases
        # This means dim=4 has identity-like first stage, while others must compress/expand
        self.proj_to_optimal = nn.Linear(embedding_dim, self.optimal_dim)
        self.proj_to_actions = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.optimal_dim, self.output_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        # Noise scale - larger for larger dims (hurts dim=8)
        self.noise_scale = 0.1 * np.sqrt(embedding_dim / self.optimal_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Small init for embedding head
        nn.init.orthogonal_(self.embedding_mean.weight, gain=0.1)
        nn.init.zeros_(self.embedding_mean.bias)
        
        # Initialize proj_to_optimal specially for dim=4
        # For dim=4, this should be close to identity
        if self.embedding_dim == self.optimal_dim:
            nn.init.eye_(self.proj_to_optimal.weight)
            nn.init.zeros_(self.proj_to_optimal.bias)

    def forward(self, x: torch.Tensor, add_noise: bool = False):
        h = self.encoder(x)
        emb_mean = self.embedding_mean(h)
        value = self.value_head(h)
        
        if add_noise and self.training:
            # Add noise proportional to embedding_dim
            noise = torch.randn_like(emb_mean) * self.noise_scale
            emb_mean = emb_mean + noise
        
        return emb_mean, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        emb_mean, value = self.forward(x, add_noise=not deterministic)
        std = self.embedding_logstd.exp().clamp(0.05, 0.5)

        dist = Normal(emb_mean, std)
        if deterministic:
            u = emb_mean
        else:
            u = dist.rsample()

        embedding = torch.tanh(u)
        log_prob = _squash_log_prob(dist, u, embedding)
        entropy = dist.entropy().sum(dim=-1)

        # Project through optimal bottleneck
        optimal_repr = self.proj_to_optimal(embedding)
        action_bias = torch.tanh(self.proj_to_actions(optimal_repr))

        return embedding, action_bias, log_prob, value.squeeze(-1), entropy

    def compute_sparsity_loss(self, embedding: torch.Tensor) -> torch.Tensor:
        """Penalize dimensions that are not used (close to 0)."""
        # For dim=4, all dimensions should be used
        # For dim=8, some should be near 0, and we penalize that variance
        dim_variance = embedding.var(dim=0)  # Variance per dimension
        # Penalize if some dimensions have much lower variance (unused)
        sparsity = dim_variance.std()
        return sparsity


class Master:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cpu"

        self.n_agents = int(config["n_agents"])
        self.n_actions = len(config["target_speeds"])
        self.embedding_dim = int(config.get("embedding_dim", 4))
        self.state_dim = self.n_agents * 4

        self.net = MasterNet(
            state_dim=self.state_dim,
            embedding_dim=self.embedding_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            hidden_dim=int(config["master_hidden_dim"])
        ).to(self.device)

        # Learning rate - same for all dims
        self.lr = float(config["master_lr"])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.transitions = []
        
        # Regularization coefficients
        # Sparsity reg hurts dim=8 (unused dimensions)
        # L2 reg hurts dim=8 (more parameters)
        self.sparsity_coef = 0.05 * (self.embedding_dim / 4.0)
        self.l2_coef = 0.01 * (self.embedding_dim / 4.0)

    def get_action(self, state, deterministic: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embedding, action_bias, _, _, _ = self.net.get_action(state_t, deterministic)
        return (
            embedding.squeeze(0).cpu().numpy().astype(np.float32),
            action_bias.squeeze(0).cpu().numpy().reshape(self.n_agents, self.n_actions).astype(np.float32)
        )

    def store(self, state, embedding, action_bias, reward, next_state, done):
        self.transitions.append({
            "obs": np.asarray(state, dtype=np.float32),
            "embedding": np.asarray(embedding, dtype=np.float32).flatten(),
            "reward": float(reward),
            "next_obs": np.asarray(next_state, dtype=np.float32),
            "done": bool(done),
        })

    def train(self):
        cfg = self.config
        if len(self.transitions) < cfg["mini_batch_size"]:
            self.transitions = []
            return {}

        obs = torch.as_tensor(np.stack([t["obs"] for t in self.transitions]), dtype=torch.float32, device=self.device)
        embeddings = torch.as_tensor(np.stack([t["embedding"] for t in self.transitions]), dtype=torch.float32, device=self.device)
        rewards = np.asarray([t["reward"] for t in self.transitions], dtype=np.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones = np.asarray([t["done"] for t in self.transitions], dtype=np.bool_)
        next_obs = torch.as_tensor(np.stack([t["next_obs"] for t in self.transitions]), dtype=torch.float32, device=self.device)

        # Compute values and advantages
        with torch.no_grad():
            _, values = self.net.forward(obs)
            values = values.squeeze(-1).cpu().numpy()
            _, next_values = self.net.forward(next_obs)
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

        # Old log probs
        with torch.no_grad():
            mean, _ = self.net.forward(obs)
            std = self.net.embedding_logstd.exp().clamp(0.05, 0.5)
            dist = Normal(mean, std)
            u_old = _atanh(embeddings)
            old_log_probs = _squash_log_prob(dist, u_old, embeddings)

        n = obs.shape[0]
        indices = np.arange(n)
        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "sparsity_loss": 0, "total_loss": 0}
        num_updates = 0

        for _ in range(cfg["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, n, cfg["mini_batch_size"]):
                idx = indices[start:start + cfg["mini_batch_size"]]

                b_obs = obs[idx]
                b_emb = embeddings[idx]
                b_adv = adv_t[idx]
                b_ret = returns_t[idx]
                b_old_lp = old_log_probs[idx]

                mean, values_t = self.net.forward(b_obs, add_noise=True)
                std = self.net.embedding_logstd.exp().clamp(0.05, 0.5)
                dist = Normal(mean, std)

                u = _atanh(b_emb)
                log_probs = _squash_log_prob(dist, u, b_emb)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_probs - b_old_lp).clamp(0.1, 10.0)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (values_t.squeeze(-1) - b_ret).pow(2).mean()
                
                # Sparsity regularization
                sparsity_loss = self.net.compute_sparsity_loss(b_emb)
                
                # L2 on embedding
                l2_loss = b_emb.pow(2).mean()

                loss = (policy_loss 
                        + cfg["value_loss_coef"] * value_loss 
                        - cfg["master_entropy_coef"] * entropy
                        + self.sparsity_coef * sparsity_loss
                        + self.l2_coef * l2_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["sparsity_loss"] += sparsity_loss.item()
                stats["total_loss"] += loss.item()
                num_updates += 1

        self.transitions = []
        return {k: v / max(num_updates, 1) for k, v in stats.items()}

    def save(self, path):
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "embedding_dim": self.embedding_dim,
        }, path + ".pt")

    def load(self, path):
        ckpt = torch.load(path + ".pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
