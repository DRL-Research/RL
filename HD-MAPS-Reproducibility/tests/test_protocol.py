from __future__ import annotations

import json

from hdmaps.paths import artifacts_root, simultaneous_dir
from hdmaps.protocol import SIM_UNIFORM_SEED, SIM_VARIED_SEED, TRAIN_SEED, VALIDATION_SEED
from hdmaps.scenarios.regional import generate_suite


def test_eval_seeds_are_disjoint_from_train_and_validation():
    assert SIM_VARIED_SEED != TRAIN_SEED
    assert SIM_UNIFORM_SEED != TRAIN_SEED
    assert SIM_VARIED_SEED != VALIDATION_SEED
    assert SIM_UNIFORM_SEED != VALIDATION_SEED
    assert TRAIN_SEED != VALIDATION_SEED


def test_generated_suites_are_seed_deterministic():
    import numpy as np

    a = generate_suite(2, 6, 3, np.random.default_rng(2026), varied=True)
    b = generate_suite(2, 6, 3, np.random.default_rng(2026), varied=True)
    assert a == b


def test_manifest_lists_checkpoints():
    manifest = json.loads((artifacts_root() / "manifest.json").read_text(encoding="utf-8"))
    files = {row["path"] for row in manifest["files"]}
    assert "checkpoints/agent.pth" in files
    assert "checkpoints/master_stage2_regional.pth" in files
    assert (simultaneous_dir() / "varied" / "16int_48agents" / "results.json").is_file()
    assert (simultaneous_dir() / "uniform" / "16int_48agents" / "results.json").is_file()
