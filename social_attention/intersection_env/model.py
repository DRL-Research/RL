"""
Ego-Attention DQN — model architecture for intersection-v0 environment, matching rl-agents reference implementation.
Architecture (ego_attention_2h.json):
  embedding_layer        : MLP [64, 64], in=7   — encodes ego vehicle
  others_embedding_layer : MLP [64, 64], in=7   — encodes neighbours (shared across the 14)
  attention_layer        : EgoAttention(feature_size=64, heads=2)
                           separate bias-free linears query_ego / key_all / value_all
                           + attention_combine; result = (combine(ctx) + ego) / 2
  output_layer           : MLP [64, 64] -> n_actions
Activations: ReLU (rl-agents default). Masking: -1e9. Init: Xavier-uniform (BaseModule.reset).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (must match train_model.py) ────────────────────────────────────
N_FEATURES = 7    # presence, x, y, vx, vy, cos_h, sin_h
EMBED_DIM  = 64
N_HEADS    = 2
N_VEHICLES = 15   # 1 ego + 14 neighbors
N_ACTIONS  = 3    # SLOWER, IDLE, FASTER


# ── Helper: Multi-Layer Perceptron ───────────────────────────────────────────

class MLP(nn.Module):
    """
    Linear/ReLU stack — mirrors rl-agents MultiLayerPerceptron
    (default activation = RELU; reshape disabled, we keep the [B, N, F] shape).

    MLP(in, [64, 64])         -> Linear(in->64)->ReLU->Linear(64->64)->ReLU
    MLP(in, [64, 64], out=3)  -> ... ->Linear(64->3)
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = None):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Core attention block: faithful port of rl-agents EgoAttention ─────────────

class EgoAttention(nn.Module):
    """
    Q = query_ego(ego)        [B, 1, d]      (ego only)
    K = key_all(input_all)    [B, N, d]
    V = value_all(input_all)  [B, N, d]      (LEARNED value projection)
    attn = softmax(QK^T / sqrt(d_head))  with absent keys masked to -1e9
    ctx  = attn @ V
    out  = (attention_combine(ctx) + ego) / 2
    """
    def __init__(self, feature_size=EMBED_DIM, heads=N_HEADS, dropout=0.0):
        super().__init__()
        assert feature_size % heads == 0
        self.feature_size = feature_size
        self.heads = heads
        self.features_per_head = feature_size // heads
        self.value_all         = nn.Linear(feature_size, feature_size, bias=False)
        self.key_all           = nn.Linear(feature_size, feature_size, bias=False)
        self.query_ego         = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_combine = nn.Linear(feature_size, feature_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ego, others, mask=None):
        '''Compute ego-attention output and attention weights. Mask should have True for absent vehicles.'''
        # ego [B,1,d]; others [B,N-1,d]; mask [B,N] with True == absent
        B = others.shape[0]
        n = others.shape[1] + 1
        H, Hd = self.heads, self.features_per_head
        input_all = torch.cat([ego, others], dim=1)                             # [B,N,d]

        key   = self.key_all(input_all).view(B, n, H, Hd).permute(0, 2, 1, 3)   # [B,H,N,Hd]
        value = self.value_all(input_all).view(B, n, H, Hd).permute(0, 2, 1, 3) # [B,H,N,Hd]
        query = self.query_ego(ego).view(B, 1, H, Hd).permute(0, 2, 1, 3)       # [B,H,1,Hd]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(Hd)     # [B,H,1,N]
        if mask is not None:
            scores = scores.masked_fill(mask.view(B, 1, 1, n), -1e9)
        attn = F.softmax(scores, dim=-1)                                        # [B,H,1,N]
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, value)                                         # [B,H,1,Hd]
        ctx = ctx.reshape(B, self.feature_size)                                 # [B,d]
        result = (self.attention_combine(ctx) + ego.squeeze(1)) / 2             # [B,d]
        return result, attn


# ── Core: Ego-Attention Network ──────────────────────────────────────────────

class EgoAttentionNetwork(nn.Module):
    """
    forward(obs):
      ego    = embedding_layer(obs[:, 0:1, :])          [B, 1,   d]
      others = others_embedding_layer(obs[:, 1:, :])    [B, N-1, d]
      mask   = obs[:, :, 0] < 0.5                        [B, N]
      attended, attn = attention_layer(ego, others, mask)
      q      = output_layer(attended)                    [B, n_actions]
    """

    def __init__(
        self,
        n_features = N_FEATURES,
        embed_dim  = EMBED_DIM,
        n_heads    = N_HEADS,
        n_actions  = N_ACTIONS,
        n_vehicles = N_VEHICLES,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim  = embed_dim
        self.n_heads    = n_heads
        self.head_dim   = embed_dim // n_heads   # 32 with d=64, H=2
        self.n_vehicles = n_vehicles
        self.n_features = n_features

        self.embedding_layer        = MLP(n_features, [embed_dim, embed_dim])
        self.others_embedding_layer = MLP(n_features, [embed_dim, embed_dim])
        self.attention_layer        = EgoAttention(embed_dim, n_heads)
        self.output_layer           = MLP(embed_dim, [embed_dim, embed_dim],
                                          output_dim=n_actions)

        # rl-agents BaseModule: Xavier-uniform weights, zero biases (applied via reset()).
        self.reset()

    def reset(self):
        """Xavier-uniform init of all Linear weights, zero biases (matches rl-agents)."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        self.apply(_init)

    def _reshape_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Accept [N,F], [B,N,F], [B,N*F], or [N*F] and return [B,N,F]."""
        if obs.dim() == 1:
            obs = obs.reshape(1, self.n_vehicles, self.n_features)
        elif obs.dim() == 2 and obs.shape[-1] == self.n_vehicles * self.n_features:
            obs = obs.reshape(obs.shape[0], self.n_vehicles, self.n_features)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0)
        return obs  # obs.dim() == 3: already [B, N, F]

    def forward(self, obs: torch.Tensor, return_attention: bool = False):
        """
        Returns:
            q_values     : [B, n_actions]
            attn_weights : [B, N]  (only when return_attention=True; mean over heads)
        """
        obs = self._reshape_obs(obs)                       # [B, N, F]

        ego_in    = obs[:, 0:1, :]                         # [B, 1,   F]
        others_in = obs[:, 1:, :]                          # [B, N-1, F]
        mask = obs[:, :, 0] < 0.5                          # [B, N]  True == absent

        ego    = self.embedding_layer(ego_in)              # [B, 1,   d]
        others = self.others_embedding_layer(others_in)    # [B, N-1, d]

        attended, attn = self.attention_layer(ego, others, mask)  # [B, d], [B, H, 1, N]
        q_values = self.output_layer(attended)             # [B, n_actions]

        if return_attention:
            attn_weights = attn.mean(dim=1).squeeze(1)      # [B, N]
            return q_values, attn_weights
        return q_values


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EgoAttentionNetwork()

    obs = torch.zeros(2, N_VEHICLES, N_FEATURES)
    obs[:, 0, 0] = 1.0   # ego present
    obs[:, 1, 0] = 1.0   # neighbor 1 present
    obs[:, 2, 0] = 1.0   # neighbor 2 present

    q, attn = model(obs, return_attention=True)
    print(f"obs shape  : {obs.shape}")
    print(f"q_values   : {q.shape}")     # [2, 3]
    print(f"attn shape : {attn.shape}")  # [2, 15]
    print(f"attn sum   : {attn[0].sum().item():.4f}")  # 1.0

    total = 0
    print("\nParameter breakdown:")
    for name, m in model.named_children():
        n = sum(p.numel() for p in m.parameters())
        total += n
        print(f"  {name:<28} {n:>6,}")
    print(f"  {'TOTAL':<28} {total:>6,}")