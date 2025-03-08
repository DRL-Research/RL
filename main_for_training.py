from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment

if __name__ == "__main__":
    experiment5 = Experiment(
        EXPERIMENT_ID='Experiment5',
        EPOCHS= 100,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=350,
        ONLY_INFERENCE=False,
    )

    experiments = [experiment5]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")

#Last Experiment - [MasterModel] Saved model to experiments/22_02_2025-21_28_27_Experiment5/trained_model_master.pth