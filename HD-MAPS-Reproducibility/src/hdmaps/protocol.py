"""Locked protocol for HD-MAPS Table 6.4 (Chapter 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

SCALE_CONFIGS: tuple[tuple[int, int], ...] = (
    (1, 3),
    (2, 6),
    (4, 12),
    (6, 18),
    (8, 24),
    (12, 36),
    (16, 48),
)

AGENTS_PER_INTERSECTION = 3
LMS_PER_INTERSECTION = 1
FANOUT = 4
NUM_MASTER_SLOTS = 5
EMBEDDING_DIM = 4
SLOT_VEC_DIM = 4
MASTER_OBS_DIM = NUM_MASTER_SLOTS * (SLOT_VEC_DIM + 1)
AGENT_OBS_DIM = 4 + EMBEDDING_DIM
TARGET_SPEEDS = (5, 10)

SIM_CONNECTOR_M = 150
SIM_DURATION = 200
SIM_MAX_STEPS = 180
SIM_VARIED_SEED = 2026
SIM_UNIFORM_SEED = 4242
SIM_EPISODES_PER_SUITE = 100
SIM_JITTER_M = 12.0

STAG_EPISODES = 100
STAG_SEED = 2026
STAG_COORD_RADIUS_M = 80.0
STAG_CONNECTOR_MIN_M = 80
STAG_CONNECTOR_MAX_M = 200
STAG_HORIZON_MIN = 120
STAG_HORIZON_MAX = 210
STAG_ENTRY_SPACING_M = 50.0

COLLISION_REWARD = -50.0
ARRIVED_REWARD = 50.0
STARVATION_REWARD = 0.0
HIGH_SPEED_REWARD = 5.0

TRAIN_EPISODES = 11500
TRAIN_MAX_AGENTS = 6
TRAIN_N_INT = 2
TRAIN_SEED = 42
TRAIN_AGENT_LR = 3e-3
TRAIN_MASTER_LR = 3e-4
TRAIN_GAMMA = 0.90
TRAIN_CLIP = 0.2
TRAIN_ENT_COEF = 0.005
VALIDATION_SEED = 7
VALIDATION_EPISODES = 20

DETERMINISTIC_AGENT = True
DETERMINISTIC_MASTER = False


@dataclass(frozen=True)
class PublishedRow:
    fleet: int
    n_int: int
    hierarchy: str
    sim_crash_pct: float
    sim_crash_lo: float
    sim_crash_hi: float
    sim_arrival_pct: float
    sim_arrival_sem: float
    stag_crash_pct: float
    stag_crash_lo: float
    stag_crash_hi: float
    stag_zero_crash_pct: float


PUBLISHED_TABLE: tuple[PublishedRow, ...] = (
    PublishedRow(3, 1, "1 / 0 / 0", 0.0, 0.0, 1.9, 100.0, 0.0, 2.0, 0.6, 7.0, 7.0),
    PublishedRow(6, 2, "2 / 0 / 1", 4.0, 2.0, 7.7, 96.9, 1.1, 2.0, 0.6, 7.0, 78.0),
    PublishedRow(12, 4, "4 / 0 / 1", 6.5, 3.8, 10.8, 95.0, 1.5, 3.0, 1.0, 8.5, 77.0),
    PublishedRow(18, 6, "6 / 2 / 1", 9.5, 6.2, 14.4, 91.9, 1.9, 7.0, 3.4, 13.7, 85.0),
    PublishedRow(24, 8, "8 / 2 / 1", 12.0, 8.2, 17.2, 90.0, 2.0, 3.0, 1.0, 8.5, 92.0),
    PublishedRow(36, 12, "12 / 3 / 1", 10.5, 7.0, 15.5, 91.0, 2.0, 2.0, 0.6, 7.0, 91.0),
    PublishedRow(48, 16, "16 / 4 / 1", 16.5, 12.0, 22.3, 85.5, 2.4, 7.0, 3.4, 13.7, 97.0),
)

SCALE_DIR_NAMES = {
    3: "1int_3agents",
    6: "2int_6agents",
    12: "4int_12agents",
    18: "6int_18agents",
    24: "8int_24agents",
    36: "12int_36agents",
    48: "16int_48agents",
}
