from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import full_config_exp1, full_config_exp2, full_config_exp3
from utils.training_loop import run_experiment
from gymnasium.envs.registration import registry, register
from utils.model.model_handler import Model
from utils.model.model_constants import ModelType

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
    experiment2 = Experiment(
        EXPERIMENT_ID='Experiment2',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,
        # ROLE=Role.CAR1,
        # EXPLORATION_EXPLOTATION_THRESHOLD=1,
        #LOAD_PREVIOUS_WEIGHT=True,
        #BYPASS_RANDOM_INITIALIZATION=False
    )

    experiment1 = Experiment(
        EXPERIMENT_ID='Experiment1',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,
        # ROLE=Role.CAR1,
        # EXPLORATION_EXPLOTATION_THRESHOLD=1,
        #LOAD_PREVIOUS_WEIGHT=True,
        #BYPASS_RANDOM_INITIALIZATION=False
        )
    experiment3 = Experiment(
        EXPERIMENT_ID='Experiment3',
        LOAD_MODEL_DIRECTORY='experiments/08_12_2024-13_56_13_Experiment1/trained_model.zip',
        EPOCHS=4,
        # ROLE=Role.CAR1,
        # EXPLORATION_EXPLOTATION_THRESHOLD=1,
        #LOAD_PREVIOUS_WEIGHT=True,
        #BYPASS_RANDOM_INITIALIZATION=False
    )
    custom_configs ={
        experiment1.EXPERIMENT_ID : full_config_exp1,
        experiment2.EXPERIMENT_ID : full_config_exp2,
        experiment3.EXPERIMENT_ID : full_config_exp3,
    }
    # experiments = [experiment1]
    # for experiment_config in experiments:
    #     print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
    #     run_experiment(experiment_config, CONFIG_EXP1)
    #     print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")


    # experiments = [experiment2]
    # for experiment_config in experiments:
    #     print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
    #     run_experiment(experiment_config, CONFIG_EXP2)
    #     print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")

    experiments = [experiment1, experiment2]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config, custom_configs[experiment_config.EXPERIMENT_ID])
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")

    # experiment1 = Experiment(
    #     EXPERIMENT_ID='Experiment1',
    #     # experiment details:
    #     # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
    #     # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
    #     # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
    #     ONLY_INFERENCE=False,
    #     EPOCHS=50,
    #     ROLE=Role.CAR1,
    #     EXPLORATION_EXPLOTATION_THRESHOLD=200,
    #     LEARNING_RATE=1e-4,
    #     MODEL_TYPE=ModelType.PPO,
    #     LOAD_PREVIOUS_WEIGHT=False,
    #     BYPASS_RANDOM_INITIALIZATION=True
    #     # LOAD_WEIGHT_DIRECTORY=get_latest_model("experiments")
    # )
    # path='experiments/'