"""
Smoke-verify the 50 appended base scenarios per layout (see extra_solvable_scenarios.py).

Runs short all-SLOW rollouts with forced scenario indices. Fails on early collision.
Usage: py -3.11 verify_extra_scenarios.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

with contextlib.redirect_stdout(io.StringIO()):
    from highwayenv.utils import (
        patch_intersection_env,
        register_intersection_env,
        register_roundabout_env,
        register_double_intersection_env,
    )

    patch_intersection_env()
    register_intersection_env()
    register_roundabout_env()
    register_double_intersection_env()

import gymnasium as gym

from src.experiment import (
    extra_solvable_scenarios as ex,
    scenarios as sc,
)
from src.experiment.scenarios_config import make_env_config_exp7
from src.experiment.new_envs_config import (
    make_roundabout_env_config,
    make_double_intersection_env_config,
)


def _rollout_slow_no_crash(env_id: str, cfg: dict, horizon: int = 80) -> None:
    env = gym.make(env_id, render_mode=None, config=cfg)
    obs, info = env.reset()
    n_agents = len(env.unwrapped.controlled_vehicles)
    action = tuple([0] * n_agents)
    for step in range(horizon):
        _, _, term, trunc, _ = env.step(action)
        if any(v.crashed for v in env.unwrapped.controlled_vehicles):
            env.close()
            raise RuntimeError(f"collision @{step} env={env_id}")
        if term or trunc:
            break
    env.close()


def main() -> None:
    kw = dict(
        collision_reward=-50,
        arrived_reward=50,
        starvation_reward=0,
        high_speed_reward=5,
    )

    n_ix = len(sc.base_complete_scenarios_6_cars) - ex.EXTRA_INTERSECTION_6CAR_COUNT
    n_rb = len(sc.roundabout_base_scenarios) - ex.EXTRA_ROUNDABOUT_COUNT
    n_dbl = len(sc.double_intersection_base_scenarios) - ex.EXTRA_DOUBLE_INTERSECTION_COUNT

    # Intersection / roundabout: one index per new base (orientation 0 slot in expanded list).
    # Intersection: standard EXP-7 car template (lanes o0-o3).
    for bi in range(n_ix, len(sc.base_complete_scenarios_6_cars)):
        idx = bi * 4
        cfg = make_env_config_exp7(**kw)
        cfg["force_scenario_index"] = idx
        _rollout_slow_no_crash("RELintersection-v0", cfg)

    # Roundabout: same lane IDs as intersection; dedicated config avoids INIT issues.
    for bi in range(n_rb, len(sc.roundabout_base_scenarios)):
        idx = bi * 4
        cfg = make_roundabout_env_config(**kw)
        cfg["force_scenario_index"] = idx
        _rollout_slow_no_crash("RELroundabout-v0", cfg)

    # Double intersection: A_/B_ lane prefixes required in controlled_cars.
    for bi in range(n_dbl, len(sc.double_intersection_base_scenarios)):
        cfg = make_double_intersection_env_config(**kw)
        cfg["force_scenario_index"] = bi
        _rollout_slow_no_crash("RELdouble-intersection-v0", cfg)

    print(
        "OK - slow-rollout verify passed for "
        f"{ex.EXTRA_INTERSECTION_6CAR_COUNT} intersection bases, "
        f"{ex.EXTRA_ROUNDABOUT_COUNT} roundabout bases, "
        f"{ex.EXTRA_DOUBLE_INTERSECTION_COUNT} double-intersection bases "
        f"(first orientation only for rotated envs)."
    )


if __name__ == "__main__":
    main()
