from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment

if __name__ == "__main__":
    experiment5_inference=Experiment(
        EXPERIMENT_ID='Experiment5_in',
        EPOCHS= 100,
        ROLE=Role.BOTH,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=200,
        ONLY_INFERENCE=True,
        INIT_SERIAL=True,
    )

    experiments = [experiment5_inference]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
