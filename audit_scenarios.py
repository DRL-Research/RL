"""
Test each of the 100 scenarios with RANDOM actions to find truly unavoidable collisions.
Suppresses noisy gym/env print output for speed.

Usage:
    C:\\Users\\glebb\\AppData\\Local\\Programs\\Python\\Python311\\python.exe audit_scenarios.py
"""
import os, sys, random as pyrandom, io, contextlib
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

# Suppress init prints
with contextlib.redirect_stdout(io.StringIO()):
    from highwayenv.utils import patch_intersection_env, register_intersection_env
    patch_intersection_env()
    register_intersection_env()

import gymnasium as gym
from src.experiment.scenarios_config import make_env_config_exp7

# Monkey-patch noisy prints inside the env
import builtins
_real_print = builtins.print
_suppress_patterns = ("Base scenario", "Placed ", "Adding missing", "Target speed",
                      "IntersectionEnv", "Successfully patched")

def _quiet_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if any(p in msg for p in _suppress_patterns):
        return
    _real_print(*args, **kwargs)

builtins.print = _quiet_print

TRIALS_PER_SCENARIO = 15
N_SCENARIOS = 100

crash_counts = {}
min_crash_step = {}

for scenario_idx in range(N_SCENARIOS):
    crashes = 0
    first_crash_steps = []

    for trial in range(TRIALS_PER_SCENARIO):
        env_cfg = make_env_config_exp7(
            collision_reward=-50,
            arrived_reward=50,
            starvation_reward=0,
            high_speed_reward=5,
        )
        env_cfg["force_scenario_index"] = scenario_idx

        env = gym.make("RELintersection-v0", render_mode=None, config=env_cfg)
        obs, info = env.reset()
        n_agents = len(env.unwrapped.controlled_vehicles)

        done = False
        crashed_this_ep = False
        step = 0
        while not done and step < 60:
            # Each trial uses a different random seed pattern
            # Some trials: all slow (0), some: all fast (1), most: random mix
            if trial < 3:
                action = tuple([0] * n_agents)        # all SLOW
            elif trial < 6:
                action = tuple([1] * n_agents)        # all FAST
            else:
                action = tuple(pyrandom.randint(0, 1) for _ in range(n_agents))  # random
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            if any(v.crashed for v in env.unwrapped.controlled_vehicles):
                crashed_this_ep = True
                first_crash_steps.append(step)
                done = True
        if crashed_this_ep:
            crashes += 1
        env.close()

    crash_rate = crashes / TRIALS_PER_SCENARIO
    avg_crash_step = sum(first_crash_steps) / len(first_crash_steps) if first_crash_steps else None
    crash_counts[scenario_idx] = crash_rate

    base = scenario_idx // 4
    rot  = scenario_idx % 4
    step_info = f" (avg crash @ step {avg_crash_step:.1f})" if avg_crash_step else ""
    flag = "  *** UNAVOIDABLE ***" if crash_rate >= 0.93 else (
           "  !! very hard !!"    if crash_rate >= 0.70 else (
           "  ~ hard ~"           if crash_rate >= 0.40 else ""))
    _real_print(f"Scenario {scenario_idx:3d} (base={base:2d} rot={rot}) crash={crash_rate:.0%}{step_info}{flag}")
    sys.stdout.flush()

# ── Summary ─────────────────────────────────────────────────────────────────
unavoidable = sorted(i for i, r in crash_counts.items() if r >= 0.93)
very_hard    = sorted(i for i, r in crash_counts.items() if 0.70 <= r < 0.93)
hard         = sorted(i for i, r in crash_counts.items() if 0.40 <= r < 0.70)
safe         = sorted(i for i, r in crash_counts.items() if r < 0.40)

_real_print(f"\n{'='*70}")
_real_print(f"TRULY UNAVOIDABLE (crash >=93% with mixed policy): {len(unavoidable)}")
_real_print(f"  indices: {unavoidable}")
_real_print(f"  base scenarios: {sorted(set(i // 4 for i in unavoidable))}")
_real_print(f"Very hard (70-93%): {len(very_hard)} — {very_hard}")
_real_print(f"Hard (40-70%): {len(hard)} — {hard}")
_real_print(f"Safe (<40%): {len(safe)} — {safe}")
_real_print(f"\nBase scenarios to DELETE: {sorted(set(i // 4 for i in unavoidable))}")
