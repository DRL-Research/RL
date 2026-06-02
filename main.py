import logging

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src.experiment.comparison_runner import run_multi_seed_comparison
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import run_experiment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    patch_intersection_env()
    register_intersection_env()
    run_mode = "single"  # single | comparison
    algorithm_to_run = "experiment"  # experiment | baseline | maddpg | vn_maddpg | attention_maddpg | ma_ga_ddpg

    experiment5_config = Experiment(
        ALGORITHM=algorithm_to_run,
        RENDER_MODE='human',  # None = do not render. if not defined, default is to render
        EXPERIMENT_ID='Experiment5',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=1,
        CYCLES=3,
        SHOW_PLOTS=True,
    )

    # dictionary were the keys are EXPERIMENT_ID (experiment name) and the values are environment configurations defined in scenarios_config.py
    custom_env_configs = {
        experiment5_config.EXPERIMENT_ID: sc.full_env_config_exp5
    }

    if run_mode == "comparison":
        comparison_summary = run_multi_seed_comparison(
            base_experiment=experiment5_config,
            env_config=custom_env_configs[experiment5_config.EXPERIMENT_ID],
            seeds=(11, 22, 33),
            algorithms=("experiment", "vn_maddpg", "ma_ga_ddpg"),
            moving_avg_window=50,
            show_plot=experiment5_config.SHOW_PLOTS,
        )
        print(f"Comparison completed. Plot saved to: {comparison_summary['plot_path']}")
    else:
        experiments = [experiment5_config]
        for experiment_config in experiments:
            print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
            run_experiment(experiment_config, custom_env_configs[experiment_config.EXPERIMENT_ID])
            print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
