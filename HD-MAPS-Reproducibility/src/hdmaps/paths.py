from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return package_root().parents[1]


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "published" / "v1"


def retrain_root() -> Path:
    return repo_root() / "artifacts" / "retrain" / "v1"


def checkpoint_dir() -> Path:
    return artifacts_root() / "checkpoints"


def simultaneous_dir() -> Path:
    return artifacts_root() / "results" / "simultaneous"


def agent_checkpoint() -> Path:
    return checkpoint_dir() / "agent.pth"


def master_checkpoint() -> Path:
    return checkpoint_dir() / "master_stage2_regional.pth"


def stage1_master_checkpoint() -> Path:
    return checkpoint_dir() / "master_stage1.pth"


def base_master_checkpoint() -> Path:
    local = checkpoint_dir() / "master_base.pth"
    if local.is_file():
        return local
    parent = repo_root().parent / "models" / "master" / "master.pth"
    if parent.is_file():
        return parent
    return local


def retrain_agent_checkpoint() -> Path:
    return retrain_root() / "checkpoints" / "agent.pth"


def retrain_master_checkpoint() -> Path:
    return retrain_root() / "checkpoints" / "master.pth"


def init_master_checkpoint(init: str) -> Path:
    if init == "published":
        return master_checkpoint()
    if init == "stage1":
        path = stage1_master_checkpoint()
        if path.is_file():
            return path
    return base_master_checkpoint()
