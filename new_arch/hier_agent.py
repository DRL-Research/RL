# hier_agent.py
"""
Agent Network for Hierarchical Architecture
============================================
Each Agent:
  - Receives its own local observation (x, y, vx, vy)
  - Receives a guidance embedding from its parent LocalMaster
  - Outputs a discrete action

Two groups of agents:
  Group 0 -> env vehicles [0, 1, 2]  (guided by LocalMaster_0)
  Group 1 -> env vehicles [3, 4, 5]  (guided by LocalMaster_1)

Both groups share the same AgentNet weights to demonstrate generalization,
but each group maintains its own replay buffer.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class HierAgentNet(nn.Module):
    def __init__(self, obs_dim: int, guidance_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.obs_dim      = obs_dim
        self.guidance_dim = guidance_dim

        # Compress guidance through a small bottleneck (optimal_dim=4)
        BOTTLENECK = 4
        self.guidance_proj = nn.Sequential(
            nn.Linear(guidance_dim, BOTTLENECK),
            nn.Tanh(),
        )

        combined_dim = obs_dim + BOTTLENECK
        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)
        self._init_weights(guidance_dim, BOTTLENECK)

    def _init_weights(self, guidance_dim: int, bottleneck: int):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Near-identity init when guidance_dim matches bottleneck
        if guidance_dim == bottleneck:
            nn.init.eye_(self.guidance_proj[0].weight)
            nn.init.zeros_(self.guidance_proj[0].bias)

    def forward(self, obs: torch.Tensor, guidance: torch.Tensor):
        g = self.guidance_proj(guidance)
        h = self.encoder(torch.cat([obs, g], dim=-1))
        return self.policy_head(h), self.value_head(h)

    def get_action(self, obs: torch.Tensor, guidance: torch.Tensor,
                   deterministic: bool = False):
        logits, value = self.forward(obs, guidance)
        dist   = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1), dist.entropy()


class HierAgent:
    """
    One agent group (3 agents sharing a single policy network).
    group_id : 0 or 1
    """

    def __init__(self, config: dict, group_id: int, shared_net: HierAgentNet = None):
        self.config       = config
        self.device       = "cpu"
        self.group_id     = group_id
        self.obs_dim      = 4
        self.guidance_dim = config["local_embedding_dim"]
        self.n_actions    = len(config["target_speeds"])

        if shared_net is not None:
            # share weights across groups (demonstrates scalability)
            self.net = shared_net
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config["agent_lr"])
        else:
            self.net = HierAgentNet(
                obs_dim=self.obs_dim,
                guidance_dim=self.guidance_dim,
                n_actions=self.n_actions,
                hidden_dim=config["agent_hidden_dim"],
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config["agent_lr"])

        self.transitions = []

    # ── inference ─────────────────────────────────────────────────────────────

    def get_actions(self, local_obs_list: list, guidance: np.ndarray,
                    deterministic: bool = False) -> list:
        """
        local_obs_list : list of (4,) arrays, one per agent in the group
        guidance       : (local_embedding_dim,) from parent LocalMaster
        """
        g = torch.as_tensor(guidance, dtype=torch.float32, device=self.device)
        actions = []
        for obs in local_obs_list:
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _ = self.net.get_action(o, g.unsqueeze(0), deterministic)
            actions.append(int(a.item()))
        return actions

    # ── storage ───────────────────────────────────────────────────────────────

    def store(self, local_obs_list, guidance, actions, rewards,
              next_local_obs_list, dones):
        g = np.asarray(guidance, dtype=np.float32).flatten()
        for i in range(len(local_obs_list)):
            self.transitions.append({
                "obs":      np.asarray(local_obs_list[i],      dtype=np.float32),
                "guidance": g.copy(),
                "action":   int(actions[i]),
                "reward":   float(rewards[i]),
                "next_obs": np.asarray(next_local_obs_list[i], dtype=np.float32),
                "done":     bool(dones[i]),
            })

    # ── training (PPO) ────────────────────────────────────────────────────────

    def train(self) -> dict:
        cfg = self.config
        if len(self.transitions) < cfg["mini_batch_size"]:
            self.transitions = []
            return {}

        obs       = torch.as_tensor(np.stack([t["obs"]      for t in self.transitions]), dtype=torch.float32)
        guidances = torch.as_tensor(np.stack([t["guidance"] for t in self.transitions]), dtype=torch.float32)
        actions   = torch.as_tensor([t["action"] for t in self.transitions], dtype=torch.long)
        rewards   = np.array([t["reward"]  for t in self.transitions], dtype=np.float32)
        rewards   = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones     = np.array([t["done"]    for t in self.transitions], dtype=bool)
        next_obs  = torch.as_tensor(np.stack([t["next_obs"] for t in self.transitions]), dtype=torch.float32)

        with torch.no_grad():
            _, values      = self.net.forward(obs, guidances)
            values         = values.squeeze(-1).cpu().numpy()
            _, next_values = self.net.forward(next_obs, guidances)
            next_values    = next_values.squeeze(-1).cpu().numpy()

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nd    = 1.0 - float(dones[t])
            delta = rewards[t] + cfg["gamma"] * next_values[t] * nd - values[t]
            gae   = delta + cfg["gamma"] * cfg["gae_lambda"] * gae * nd
            adv[t] = gae

        returns = adv + values
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_t   = torch.as_tensor(returns, dtype=torch.float32)
        adv_t   = torch.as_tensor(adv,     dtype=torch.float32)

        with torch.no_grad():
            logits, _ = self.net.forward(obs, guidances)
            old_dist  = Categorical(logits=logits)
            old_lp    = old_dist.log_prob(actions)

        n        = obs.shape[0]
        indices  = np.arange(n)
        stats    = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        nupdates = 0

        for _ in range(cfg["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, n, cfg["mini_batch_size"]):
                idx = indices[start: start + cfg["mini_batch_size"]]
                b_obs  = obs[idx];      b_g   = guidances[idx]
                b_act  = actions[idx];  b_adv = adv_t[idx]
                b_ret  = ret_t[idx];    b_old = old_lp[idx]

                logits, val_t = self.net.forward(b_obs, b_g)
                dist   = Categorical(logits=logits)
                lp     = dist.log_prob(b_act)
                ent    = dist.entropy().mean()

                ratio  = torch.exp(lp - b_old)
                surr1  = ratio * b_adv
                surr2  = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * b_adv
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = (val_t.squeeze(-1) - b_ret).pow(2).mean()
                loss   = p_loss + cfg["value_loss_coef"] * v_loss - cfg["agent_entropy_coef"] * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                stats["policy_loss"] += p_loss.item()
                stats["value_loss"]  += v_loss.item()
                stats["entropy"]     += ent.item()
                stats["total_loss"]  += loss.item()
                nupdates += 1

        self.transitions = []
        tag = f"agent_group{self.group_id}"
        return {f"{tag}/{k}": v / max(nupdates, 1) for k, v in stats.items()}

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["opt"])
