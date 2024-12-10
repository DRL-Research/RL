from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.logger.neptune_logger import NeptuneLogger
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment

if __name__ == "__main__":

    experiment1 = Experiment(
        EXPERIMENT_ID='Experiment1',
        # experiment details:
        # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        ONLY_INFERENCE=False,
        EPOCHS=10,
        ROLE=Role.CAR1,
        EXPLORATION_EXPLOTATION_THRESHOLD=50,
        LEARNING_RATE=0.0001,
        MODEL_TYPE=ModelType.PPO,
        # LOAD_WEIGHT_DIRECTORY=get_latest_model("experiments")
    )

    # experiment2 = Experiment(
    #     EXPERIMENT_ID='Experiment2',
    #     # experiment details:
    #     # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
    #     # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
    #     # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
    #     ONLY_INFERENCE=False,
    #     EPOCHS=2,
    #     ROLE=Role.CAR1,
    #     EXPLORATION_EXPLOTATION_THRESHOLD=50,
    #     LEARNING_RATE=200,
    #     MODEL_TYPE=ModelType.DQN,
    #     # LOAD_WEIGHT_DIRECTORY=get_latest_model("experiments")
    # )

    experiments = [experiment1]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        logger = NeptuneLogger(
            project_name="katiusha8642/DRL",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjJkY2Q5Zi03YWRkLTQxMjMtYmUwYi1iYzM5ZGNmNDkxMGEifQ==",
            run_name=experiment_config.EXPERIMENT_ID,
            tags=["experiment", "training"]
        )

        # Log experiment hyperparameters to Neptune
        logger.log_hyperparameters({
            "experiment_id": experiment_config.EXPERIMENT_ID,
            "only_inference": experiment_config.ONLY_INFERENCE,
            "epochs": experiment_config.EPOCHS,
            "role": experiment_config.ROLE,
            "exploration_exploitation_threshold": experiment_config.EXPLORATION_EXPLOTATION_THRESHOLD,
            "time_between_steps": experiment_config.TIME_BETWEEN_STEPS,
            "learning_rate": experiment_config.LEARNING_RATE,
            "n_steps": experiment_config.N_STEPS,
            "batch_size": experiment_config.BATCH_SIZE,
        })

        # Run the experiment
        run_experiment(experiment_config, logger)
        logger.stop()
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
