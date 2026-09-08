from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return package_root().parents[1]


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "published" / "v1"


def checkpoint_dir() -> Path:
    return artifacts_root() / "checkpoints"


def simultaneous_dir() -> Path:
    return artifacts_root() / "results" / "simultaneous"


def agent_checkpoint() -> Path:
    return checkpoint_dir() / "agent.pth"


def master_checkpoint() -> Path:
    return checkpoint_dir() / "master_stage2_regional.pth"
