"""
Run the G03 custom-crossing test rollout with a visible pygame window.

Put RENDER_MODE = "human" (below, after imports) or None for headless.
Pass best_model path on the CLI or use PASTE_BEST_MODEL_PATH in eval_pretrained_on_custom_crossing.py.

Example — one scenario:
  py -3.11 watch_custom_crossing_tests.py "<path/to/best_model>" --conditions normal --n-scenarios 1 --scenario-start 0

All scenarios in pickle (often 100):
  py -3.11 watch_custom_crossing_tests.py "<path>" --conditions normal --all-scenarios
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

from eval_pretrained_on_custom_crossing import (  # noqa: E402
    DEFAULT_SCENARIO_PKL,
    PASTE_BEST_MODEL_PATH,
    experiment_home_from_input,
    resolve_checkpoint_prefix,
)
import run_proto_action_sweep as rps  # noqa: E402
import run_g03_crossing_quad_bundle as bundle  # noqa: E402

# Same idea as Experiment.RENDER_MODE / Driver — passed to gym.make(..., render_mode=).
RENDER_MODE = "human"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("best_model_or_experiment", nargs="?", default=PASTE_BEST_MODEL_PATH.strip())
    parser.add_argument("--scenarios-pickle", type=str, default=DEFAULT_SCENARIO_PKL)
    parser.add_argument("--g03-config", type=str, default=bundle.G03_DEFAULT_CONFIG)
    parser.add_argument("--eval-run-id", type=int, default=0)
    parser.add_argument("--scenario-start", type=int, default=0)
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=1,
        metavar="N",
        help="How many scenarios to play starting at --scenario-start (default 1). Ignored if --all-scenarios.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Play every scenario in the pickle from --scenario-start to the last index.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(bundle.CUSTOM_TEST_CONDITIONS),
        help="Comma-separated; any token in RUN_EPISODE_SUPPORTED_CONDITIONS (see run_g03_crossing_quad_bundle.py).",
    )
    parser.add_argument("--step-sleep", type=float, default=0.02)
    parser.add_argument("--pause-between", action="store_true")
    args = parser.parse_args()

    user_path = (args.best_model_or_experiment or "").strip()
    if not user_path:
        print("Need a path to best_model or experiment folder.", file=sys.stderr)
        sys.exit(2)

    conds = list(bundle.parse_conditions_csv(args.conditions))

    scenarios = bundle.load_scenarios_pickle(
        os.path.normpath(os.path.abspath(os.path.expanduser(args.scenarios_pickle)))
    )
    n_pool = len(scenarios)
    start = max(0, int(args.scenario_start))
    if args.all_scenarios:
        end = n_pool
    else:
        end = min(n_pool, start + max(1, int(args.n_scenarios)))
    if start >= n_pool:
        sys.exit(f"--scenario-start {start} out of range (pickle has {n_pool})")

    cfg_path = os.path.normpath(os.path.abspath(os.path.expanduser(args.g03_config)))
    cfg = bundle.load_json(cfg_path)
    cfg.setdefault("master_broadcast_const_test", 9999.0)

    ckpt_prefix = resolve_checkpoint_prefix(user_path)
    experiment_home = experiment_home_from_input(user_path)

    rps.set_all_seeds(123)
    out_dir = os.path.join(experiment_home, "custom_crossing_eval_watch")
    os.makedirs(out_dir, exist_ok=True)
    proto_exp = rps.ProtoExperiment(cfg, out_dir)
    master_model, agent_model = rps.make_models(proto_exp)
    if not rps.load_models(agent_model, master_model, ckpt_prefix):
        sys.exit(f"Failed to load checkpoints from {ckpt_prefix}")

    env_extra = {
        "custom_regular_scenarios": scenarios,
        "custom_regular_only": True,
        "conflict_ratio": 0.0,
        "use_conflict_scenarios_only": False,
        "use_held_out_scenarios": False,
    }
    test_env = rps.ProtoHighwayWrapper(
        proto_exp,
        conflict_ratio=0.0,
        conflict_only=False,
        env_extra=env_extra,
        render_mode=RENDER_MODE,
        step_sleep_seconds=float(args.step_sleep),
    )
    eval_pm_backup = bool(proto_exp.cfg.get("eval_policy_mean", False))
    proto_exp.cfg["eval_policy_mean"] = True
    run_id = int(args.eval_run_id)
    try:
        for ep_ix in range(start, end):
            base_seed = 8_000_000 + run_id * 100_000 + ep_ix
            for condition in conds:
                print(f"\n=== scenario {ep_ix} | {condition} | seed={base_seed} ===", flush=True)
                if args.pause_between:
                    input("Enter for next episode…")
                summary, _, _ = rps.run_episode(
                    proto_exp=proto_exp,
                    env=test_env,
                    master_model=master_model,
                    agent_model=agent_model,
                    episode=ep_ix + 1,
                    train=False,
                    condition=condition,
                    replay_seed=base_seed,
                    custom_scenario_index=ep_ix,
                )
                print(f"arrival_pct={summary.get('arrival_pct')} crashed={summary.get('crashed')}", flush=True)
    finally:
        test_env.close()
        proto_exp.cfg["eval_policy_mean"] = eval_pm_backup


if __name__ == "__main__":
    main()
