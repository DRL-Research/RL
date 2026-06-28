import argparse

from src.experiment.experiment_config import Experiment
from src.experiment.scalability_suite import run_requested_suite_comparison


def _parse_csv_integers(csv_text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in csv_text.split(",") if part.strip())


def _parse_csv_strings(csv_text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in csv_text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MAPS, VN-MA-DDPG, and MA-GA-DDPG on the requested new scenario suite."
    )
    parser.add_argument("--experiment-id", default="new_scenarios_suite")
    parser.add_argument("--episodes-per-cycle", type=int, default=300)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--algorithms", default="experiment,vn_maddpg,ma_ga_ddpg")
    parser.add_argument("--moving-avg-window", type=int, default=50)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument(
        "--include-extra-regular",
        action="store_true",
        help="Also include the extra procedural regular scenarios appended in the branch file.",
    )
    args = parser.parse_args()

    base_experiment = Experiment(
        ALGORITHM="experiment",
        ENV_ID="RELintersection-v0",
        CARS_AMOUNT=6,
        RENDER_MODE=None,
        EXPERIMENT_ID=args.experiment_id,
        EPISODES_PER_CYCLE=args.episodes_per_cycle,
        CYCLES=args.cycles,
        SHOW_PLOTS=args.show_plot,
    )

    summary = run_requested_suite_comparison(
        base_experiment=base_experiment,
        seeds=_parse_csv_integers(args.seeds),
        algorithms=_parse_csv_strings(args.algorithms),
        moving_avg_window=args.moving_avg_window,
        show_plot=args.show_plot,
        include_extra_regular=args.include_extra_regular,
    )

    print(f"Overall comparison plot: {summary['overall_plot_path']}")
    for env_key, env_summary in summary["environments"].items():
        print(
            f"{env_summary['label']}: {env_summary['plot_path']} "
            f"({env_summary['scenario_count']} scenarios, env={env_key})"
        )


if __name__ == "__main__":
    main()
