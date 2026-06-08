"""
IDE entry point: Run Python File (Play) / F5 — no CLI args needed.

Change PLAY_MODE or the label lists below, then Play.
Uses the same dated folder layout as ``run_pl75_p01_p03_hybrid_sweep.py``.
"""
from __future__ import annotations

import os
import sys


# --- IDE-only switches (edit here) ---
# single   = PH07 recommended re-run only
# top3     = PH07 + PH04 + PH05 in one dated run folder (rankings include all three)
# full     = all 8 HYBRID configs
# smoke    = 3 train / 2 test episodes on PH07 (pipeline sanity check)
PLAY_MODE = "single"

_SINGLE_LABEL = "PH07_blend_035_s1650"

_TOP3_LABELS = (
    "PH07_blend_035_s1650",
    "PH04_blend_004_s1800",
    "PH05_blend_025_s1200_slow",
)


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _compose_argv(play_mode: str) -> list[str]:
    exe = sys.argv[0] if sys.argv else "python"
    argv = [exe]
    if play_mode == "single":
        argv += ["--only-config", _SINGLE_LABEL]
    elif play_mode == "top3":
        argv += ["--configs", *_TOP3_LABELS]
    elif play_mode == "full":
        pass
    elif play_mode == "smoke":
        argv += ["--smoke", "--only-config", _SINGLE_LABEL]
    else:
        raise ValueError(f"Unknown PLAY_MODE={play_mode!r}; use single | top3 | full | smoke")
    return argv


def main() -> None:
    repo = _repo_root()
    os.chdir(repo)
    sys.argv[:] = _compose_argv(PLAY_MODE)

    import run_pl75_p01_p03_hybrid_sweep as sweep

    sweep.main()


if __name__ == "__main__":
    main()
