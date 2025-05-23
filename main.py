from highwayenv.utils import patch_intersection_env, register_intersection_env
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training_loop.training_loop import run_experiment

if __name__ == "__main__":

    patch_intersection_env()
    register_intersection_env()

    experiment1_config = Experiment(
        EXPERIMENT_ID='Experiment1',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,)

    experiment2_config = Experiment(
        EXPERIMENT_ID='Experiment2',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4
    )

    experiment3_config = Experiment(
        EXPERIMENT_ID='Experiment3',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4)

    experiment4_config = Experiment(
        EXPERIMENT_ID='Experiment4',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=2)

    experiment5_config = Experiment(
        EXPERIMENT_ID='Experiment5',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=10)

    # dictionary were the keys are EXPERIMENT_ID (experiment name) and the values are environment configurations defined in scenarios_config.py
    custom_env_configs = {
        experiment1_config.EXPERIMENT_ID: sc.full_env_config_exp1,
        experiment2_config.EXPERIMENT_ID: sc.full_env_config_exp2,
        experiment3_config.EXPERIMENT_ID: sc.full_env_config_exp3,
        experiment4_config.EXPERIMENT_ID: sc.full_env_config_exp4,
        experiment5_config.EXPERIMENT_ID: sc.full_env_config_exp5
    }

    experiments = [experiment5_config]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config, custom_env_configs[experiment_config.EXPERIMENT_ID])
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
