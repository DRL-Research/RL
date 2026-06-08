"""
IDE entry point for the G03 five-run episode sweep (no CLI flags).

Open this file in Cursor and press Run / Debug, or choose the matching launch preset.

Artifacts: experiment_runs/G03_CROSSING_QUAD_<timestamp>/ with run_00…run_04.

Each run trains longer (default 2000, 3000, 4000, 5000, 6000 episodes); `--train-episodes`
on the CLI entry is not exposed here unless you extend _argv().
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))

# --- Edit these ---
# "full" = real bundle (episode count taken from merged G03 / run_00 config JSON, usually 2000–3000).
# "smoke" = sanity check only (2 train episodes, tiny eval set).
PLAY_MODE = "full"  # "full" | "smoke"

# Leave "" to generate a new crossing-heavy test pool on each run.
# Set to an existing bundle folder to reuse its scenario pickle (and run_00/config when present).
REPRODUCE_FROM = ""


def _argv() -> list[str]:
    exe = sys.argv[0] if sys.argv else "python"
    argv = [exe]

    mode = PLAY_MODE.lower().strip()
    if mode == "full":
        pass
    elif mode == "smoke":
        argv.append("--smoke")
    else:
        raise ValueError(f'PLAY_MODE must be "full" or "smoke", got {PLAY_MODE!r}')

    if (REPRODUCE_FROM or "").strip():
        abs_src = REPRODUCE_FROM if os.path.isabs(REPRODUCE_FROM) else os.path.join(_REPO, REPRODUCE_FROM)
        argv += ["--reproduce-from", os.path.normpath(abs_src)]

    return argv


def main() -> None:
    os.chdir(_REPO)
    sys.argv[:] = _argv()

    import run_g03_crossing_quad_bundle as bundle

    bundle.main()


if __name__ == "__main__":
    main()