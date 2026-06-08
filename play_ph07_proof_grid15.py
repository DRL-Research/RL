"""
IDE entry point for the 15-config PH07 proof grid (no CLI needed).

Edit PLAY_MODE or SINGLE_LABEL, then Run / Play.
Modes: full | single | smoke | small_batch
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))


PLAY_MODE = "full"  # full | single | smoke | small_batch
_SINGLE_LABEL = "G01_PH07_proof_baseline"

# When PLAY_MODE == small_batch, run these first (edit freely).
_SMALL_BATCH = (
    "G01_PH07_proof_baseline",
    "G06_proof_master_low_std",
    "G15_proof_composite_ckpt",
)


def _argv() -> list[str]:
    exe = sys.argv[0] if sys.argv else "python"
    argv = [exe]
    mode = PLAY_MODE
    if mode == "full":
        return argv
    if mode == "single":
        return argv + ["--only-config", _SINGLE_LABEL]
    if mode == "smoke":
        return argv + ["--smoke", "--only-config", _SINGLE_LABEL]
    if mode == "small_batch":
        return argv + ["--configs", *_SMALL_BATCH]
    raise ValueError(f"Unknown PLAY_MODE={mode!r}; use full | single | smoke | small_batch")


def main() -> None:
    os.chdir(_REPO)
    sys.argv[:] = _argv()
    import run_ph07_master_proof_grid15 as sweep

    sweep.main()


if __name__ == "__main__":
    main()
