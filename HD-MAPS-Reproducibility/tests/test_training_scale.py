from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hdmaps.protocol import (
    SIM_CONNECTOR_M,
    TRAIN_MAX_AGENTS,
    TRAIN_MAX_AGENTS_PER_MASTER,
    TRAIN_N_INT,
)
from hdmaps.training.hdmaps import archive_overscale_stage, sample_stage1, sample_stage2
from hdmaps.training.rollout import train_lm_groups


def _origin_int(agent) -> int:
    return int(agent[0][0].split("_")[0][1:])


def test_stage1_stays_on_one_intersection():
    rng = np.random.default_rng(0)
    for _ in range(200):
        spec = sample_stage1(rng)
        assert spec["n_int"] == 1
        assert 1 <= spec["n_agents"] <= TRAIN_MAX_AGENTS_PER_MASTER
        assert spec["connector"] == SIM_CONNECTOR_M
        assert len(spec["scenario"]["agents"]) == spec["n_agents"]


def test_stage2_is_two_masters_at_most_six_agents():
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        spec = sample_stage2(rng)
        assert spec["n_int"] == TRAIN_N_INT == 2
        assert spec["n_agents"] <= TRAIN_MAX_AGENTS
        assert spec["n_agents"] in (4, 6)
        assert spec["connector"] == SIM_CONNECTOR_M
        counts: dict[int, int] = {}
        for agent in spec["scenario"]["agents"]:
            counts[_origin_int(agent)] = counts.get(_origin_int(agent), 0) + 1
        assert set(counts) <= {0, 1}
        assert all(c <= TRAIN_MAX_AGENTS_PER_MASTER for c in counts.values())
        assert all(c <= 3 for c in counts.values())
        seen.add(spec["n_agents"])
    assert 6 in seen


def test_overscale_stage_is_archived(tmp_path: Path):
    dest = tmp_path / "stage2"
    dest.mkdir()
    (dest / "history.jsonl").write_text(json.dumps({"n_int": 15, "n_agents": 45}) + "\n", encoding="utf-8")
    (dest / "master.pth").write_bytes(b"x")
    bak = archive_overscale_stage(dest)
    assert bak is not None
    assert bak.name.endswith("_overscale")
    assert bak.is_dir()
    assert not dest.exists()
    assert archive_overscale_stage(dest) is None


def test_paper_scale_stage_is_not_archived(tmp_path: Path):
    dest = tmp_path / "stage2"
    dest.mkdir()
    (dest / "history.jsonl").write_text(json.dumps({"n_int": 2, "n_agents": 6}) + "\n", encoding="utf-8")
    assert archive_overscale_stage(dest) is None
    assert dest.is_dir()


class _Zones:
    def __init__(self, zones):
        self._zones = zones

    def agent_zones(self):
        return list(self._zones)


def test_train_lm_groups_one_master_per_intersection():
    groups = train_lm_groups(_Zones([0, 0, 0, 1, 1, 1]))
    assert groups == [[0, 1, 2], [3, 4, 5]]
    groups = train_lm_groups(_Zones([0, 0, 0, 0]))
    assert groups == [[0, 1, 2, 3]]


def test_train_lm_groups_splits_when_a_zone_exceeds_four():
    groups = train_lm_groups(_Zones([0, 0, 0, 0, 0]))
    assert groups == [[0, 1, 2, 3], [4]]
    assert all(len(g) <= TRAIN_MAX_AGENTS_PER_MASTER for g in groups)
