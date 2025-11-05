"""Configuration dataclasses for SafeR-ADAIM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SafeRADAIMConfig:
    """Hyper-parameters and environment settings for SafeR-ADAIM.

    These defaults are derived from the paper and adapted to the
    intersection benchmark contained in this repository. The defaults keep
    the same hidden layer sizes as the paper (two 128-unit hidden layers
    for policy/value networks and a deeper cost network) while exposing
    the reward and cost shaping parameters that appear in Section V-A of
    the paper.
    """

    # rollout / training
    epochs: int = 512
    steps_per_epoch: int = 2048
    gamma: float = 0.99
    cost_gamma: float = 0.99
    gae_lambda: float = 0.95
    cost_gae_lambda: float = 0.95
    line_search_coeff: float = 0.8
    max_line_search_steps: int = 10
    damping_coeff: float = 0.1
    cg_iters: int = 15
    max_kl: float = 0.01

    # safety constraint (per epoch average cost limit)
    cost_limit: float = 0.01

    # learning rates for value networks
    value_lr: float = 3e-4
    cost_value_lr: float = 3e-4

    # neural network widths
    policy_hidden_sizes: Sequence[int] = field(default_factory=lambda: (128, 128))
    value_hidden_sizes: Sequence[int] = field(default_factory=lambda: (128, 128))
    cost_value_hidden_sizes: Sequence[int] = field(default_factory=lambda: (128, 128, 128, 128))

    # action bounds for expected velocity outputs (m/s)
    min_velocity: float = 0.0
    max_velocity: float = 15.0

    # reward shaping parameters (Section V-A of the paper)
    epsilon_v: float = 0.1
    epsilon_t: float = 0.05
    epsilon_pass_single: float = 1.0
    epsilon_pass_all: float = 3.0
    epsilon_a: float = -0.05
    epsilon_j: float = -0.02

    # dense / sparse cost increments
    epsilon_distance_cost: float = 1.0
    epsilon_collision_cost: float = 10.0

    safe_distance: float = 5.0

    # training control
    device: str = "cpu"

    # logging
    log_interval: int = 10


@dataclass
class SafeRADAIMTrainSettings:
    """High level toggles for the SafeR-ADAIM trainer."""

    total_epochs: int = 128
    evaluate_every: int = 32
    evaluation_episodes: int = 5

