from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import full_env_config_exp1, full_env_config_exp2, full_env_config_exp3, \
    full_env_config_exp4, full_env_config_exp5
from utils.training_loop import run_experiment
from gymnasium.envs.registration import registry, register


if __name__ == "__main__":
    if "RELintersection-v0" not in registry:
        register(
            id="RELintersection-v0",
            entry_point="highwayenv.intersection_class:IntersectionEnv",
        )

    """
    experiment2 details:
    Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
    Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
    Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
    """
    experiment1_config = Experiment(
        EXPERIMENT_ID='Experiment1',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,
        LOAD_PREVIOUS_WEIGHT=True,
        BYPASS_RANDOM_INITIALIZATION=False)

    experiment2_config = Experiment(
        EXPERIMENT_ID='Experiment2',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,
        LOAD_PREVIOUS_WEIGHT=True,
        BYPASS_RANDOM_INITIALIZATION=False
    )

    experiment3_config = Experiment(
        EXPERIMENT_ID='Experiment3',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,
        LOAD_PREVIOUS_WEIGHT=True,
        BYPASS_RANDOM_INITIALIZATION=False)

    experiment4_config = Experiment(
        EXPERIMENT_ID='Experiment4',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=2,
        LOAD_PREVIOUS_WEIGHT=True,
        BYPASS_RANDOM_INITIALIZATION=False)

    experiment5_config = Experiment(
        EXPERIMENT_ID='Experiment5',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=2,
        LOAD_PREVIOUS_WEIGHT=True,
        BYPASS_RANDOM_INITIALIZATION=False)

    # dictionary were the keys are EXPERIMENT_ID (experiment name) and the values are environment configurations defined in experiment_constants.py
    custom_env_configs ={
        experiment1_config.EXPERIMENT_ID : full_env_config_exp1,
        experiment2_config.EXPERIMENT_ID : full_env_config_exp2,
        experiment3_config.EXPERIMENT_ID : full_env_config_exp3,
        experiment4_config.EXPERIMENT_ID : full_env_config_exp4,
        experiment5_config.EXPERIMENT_ID : full_env_config_exp5
    }

    experiments = [experiment4_config] #, experiment1]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config, custom_env_configs[experiment_config.EXPERIMENT_ID])
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
