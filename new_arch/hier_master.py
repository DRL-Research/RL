# hier_master.py
"""
Hierarchical Master Networks - Scalability POC
===============================================
Architecture:
  GlobalMaster  ->  LocalMaster_0  ->  Agents [0,1,2]
                ->  LocalMaster_1  ->  Agents [3,4,5]

KEY DESIGN: All masters share ONE set of weights (SharedMasterPolicy).
  - Same HierMasterNet is used at every level of the hierarchy.
  - type_bit in the padded input distinguishes master-level vs agent-level subordinates.
  - GlobalMaster (root) receives a zero guidance vector; LocalMasters receive
    the GlobalMaster's embedding as guidance.
  - All master transitions are pooled into a single replay buffer and updated together.
  This makes adding more masters or more hierarchy levels a config change only —
  no new parameters, no new training procedures.

Fixed-size input per Master (34-dim):
  5 slots x (x, y, vx, vy, type_bit) = 25
  + 5-bit validity mask              =  5
  + guidance embedding               =  4  (zeros for root)
  ─────────────────────────────────────────
  total                              = 34

type_bit: 0 = agent subordinate,  1 = master subordinate
GlobalMaster  : 2 active slots (LocalMasters), 3 padded  — type_bit = 1
LocalMaster   : 3 active slots (Agents),       2 padded  — type_bit = 0
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

MAX_SUBORDINATES = 5
FEATURES_PER_SUB = 5          # x, y, vx, vy, type_bit
MASTER_INPUT_DIM = MAX_SUBORDINATES * FEATURES_PER_SUB + MAX_SUBORDINATES  # 30


# ── helpers ──────────────────────────────────────────────────────────────────

def _atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_log_prob(dist: Normal, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return dist.log_prob(u).sum(-1) - torch.log(1.0 - a.pow(2) + eps).sum(-1)


def build_master_input(sub_states, type_bits, n_active: int) -> np.ndarray:
    """
    Build the 30-dim padded input for a Master.

    sub_states : list of n_active np.ndarray shape (4,)  [x,y,vx,vy]
    type_bits  : list of n_active int   (0=agent, 1=master)
    n_active   : number of real subordinates (<= MAX_SUBORDINATES)
    """
    slots = np.zeros((MAX_SUBORDINATES, FEATURES_PER_SUB), dtype=np.float32)
    mask  = np.zeros(MAX_SUBORDINATES, dtype=np.float32)
    for i in range(n_active):
        slots[i, :4] = sub_states[i]
        slots[i,  4] = float(type_bits[i])
        mask[i]      = 1.0
    return np.concatenate([slots.flatten(), mask])   # (30,)


# ── shared network ────────────────────────────────────────────────────────────

class HierMasterNet(nn.Module):
    """
    Single network used at ALL levels of the master hierarchy.
    input_dim = MASTER_INPUT_DIM (30) + guidance_dim (4) = 34
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, guidance_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.guidance_dim  = guidance_dim
        input_dim = MASTER_INPUT_DIM + guidance_dim   # 34

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.embedding_mean   = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_logstd = nn.Parameter(torch.zeros(embedding_dim) - 1.0)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.embedding_mean.weight, gain=0.1)
        nn.init.zeros_(self.embedding_mean.bias)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.embedding_mean(h), self.value_head(h)

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        emb_mean, value = self.forward(x)
        std  = self.embedding_logstd.exp().clamp(0.05, 0.5)
        dist = Normal(emb_mean, std)
        u    = emb_mean if deterministic else dist.rsample()
        emb  = torch.tanh(u)
        lp   = _squash_log_prob(dist, u, emb)
        ent  = dist.entropy().sum(-1)
        return emb, lp, value.squeeze(-1), ent


# ── shared policy (holds weights + optimizer + buffer) ────────────────────────

class SharedMasterPolicy:
    """
    Owns the ONE shared HierMasterNet used by every master instance.
    All master instances contribute transitions to a single buffer here,
    and a single PPO update is performed each training step.
    """

    def __init__(self, config: dict):
        self.config       = config
        self.device       = "cpu"
        self.embedding_dim = config["global_embedding_dim"]   # same at every level
        self.guidance_dim  = config["global_embedding_dim"]   # root gets zeros

        self.net = HierMasterNet(
            embedding_dim=self.embedding_dim,
            hidden_dim=config["master_hidden_dim"],
            guidance_dim=self.guidance_dim,
        ).to(self.device)

        self.optimizer   = torch.optim.Adam(self.net.parameters(), lr=config["master_lr"])
        self.transitions = []   # pooled from ALL master instances

    def train(self) -> dict:
        cfg = self.config
        if len(self.transitions) < cfg["mini_batch_size"]:
            self.transitions = []
            return {}

        obs        = torch.as_tensor(np.stack([t["obs"]       for t in self.transitions]), dtype=torch.float32)
        embeddings = torch.as_tensor(np.stack([t["embedding"] for t in self.transitions]), dtype=torch.float32)
        rewards    = np.array([t["reward"] for t in self.transitions], dtype=np.float32)
        rewards    = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones      = np.array([t["done"]   for t in self.transitions], dtype=bool)
        next_obs   = torch.as_tensor(np.stack([t["next_obs"]  for t in self.transitions]), dtype=torch.float32)

        with torch.no_grad():
            _, values      = self.net.forward(obs)
            values         = values.squeeze(-1).cpu().numpy()
            _, next_values = self.net.forward(next_obs)
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
            mean, _ = self.net.forward(obs)
            std     = self.net.embedding_logstd.exp().clamp(0.05, 0.5)
            dist    = Normal(mean, std)
            u_old   = _atanh(embeddings)
            old_lp  = _squash_log_prob(dist, u_old, embeddings)

        n        = obs.shape[0]
        indices  = np.arange(n)
        stats    = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        nupdates = 0

        for _ in range(cfg["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, n, cfg["mini_batch_size"]):
                idx = indices[start: start + cfg["mini_batch_size"]]
                b_obs    = obs[idx];        b_emb  = embeddings[idx]
                b_adv    = adv_t[idx];      b_ret  = ret_t[idx]
                b_old_lp = old_lp[idx]

                mean, val_t = self.net.forward(b_obs)
                std  = self.net.embedding_logstd.exp().clamp(0.05, 0.5)
                dist = Normal(mean, std)
                u    = _atanh(b_emb)
                lp   = _squash_log_prob(dist, u, b_emb)
                ent  = dist.entropy().sum(-1).mean()

                ratio  = torch.exp(lp - b_old_lp).clamp(0.1, 10.0)
                surr1  = ratio * b_adv
                surr2  = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * b_adv
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = (val_t.squeeze(-1) - b_ret).pow(2).mean()
                loss   = p_loss + cfg["value_loss_coef"] * v_loss - cfg["master_entropy_coef"] * ent

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
        return {"shared_master/" + k: v / max(nupdates, 1) for k, v in stats.items()}

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["opt"])


# ── thin master instance (inference + storage only) ───────────────────────────

class MasterInstance:
    """
    Thin wrapper around SharedMasterPolicy.
    Handles input construction and transition storage for one master node.
    Does NOT own network weights — all weight operations go through SharedMasterPolicy.
    """

    def __init__(self, policy: SharedMasterPolicy, name: str, is_root: bool = False):
        self.policy   = policy
        self.name     = name
        self.is_root  = is_root
        # Root master (GlobalMaster) gets zero guidance — no parent
        self._zero_guidance = np.zeros(policy.guidance_dim, dtype=np.float32)

    def _make_input(self, master_input: np.ndarray, guidance=None) -> np.ndarray:
        g = self._zero_guidance if (guidance is None or self.is_root) else np.asarray(guidance, dtype=np.float32).flatten()
        return np.concatenate([np.asarray(master_input, dtype=np.float32), g])

    def get_embedding(self, master_input: np.ndarray, guidance=None,
                      deterministic: bool = False) -> np.ndarray:
        inp = self._make_input(master_input, guidance)
        t   = torch.as_tensor(inp, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb, _, _, _ = self.policy.net.get_action(t, deterministic)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def store(self, master_input: np.ndarray, guidance,
              embedding: np.ndarray, reward: float,
              next_master_input: np.ndarray, next_guidance, done: bool):
        inp      = self._make_input(master_input, guidance)
        next_inp = self._make_input(next_master_input, next_guidance)
        self.policy.transitions.append({
            "obs":       inp.astype(np.float32),
            "embedding": np.asarray(embedding, dtype=np.float32),
            "reward":    float(reward),
            "next_obs":  next_inp.astype(np.float32),
            "done":      bool(done),
        })


# ── convenience constructors ──────────────────────────────────────────────────

def make_global_master(policy: SharedMasterPolicy) -> MasterInstance:
    """Root master — receives zero guidance."""
    return MasterInstance(policy, name="global_master", is_root=True)


def make_local_master(policy: SharedMasterPolicy, master_id: int) -> MasterInstance:
    """Mid-level master — receives GlobalMaster's embedding as guidance."""
    return MasterInstance(policy, name=f"local_master_{master_id}", is_root=False)
