from turns.training_loop_turns import run_experiment_turns
from utils.experiment.experiment_turns_config import ExperimentTurns
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType


if __name__ == "__main__":

    experiment1 = ExperimentTurns(
        EXPERIMENT_ID='Experiment1',
        # experiment details:
        # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        EPOCHS=3,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        # INFERENCE Mode
        ONLY_INFERENCE=False,
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )

    # experiments = [experiment1, experiment2]
    experiments = [experiment1]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment_turns(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
