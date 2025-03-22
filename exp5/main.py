from src.constants import Role, ModelType
from src.experiment_config import Experiment
from training_loop import run_experiment

if __name__ == "__main__":
    print('mainnnn')
    experiment5 = Experiment(
        EXPERIMENT_ID='Experiment5',
        EPOCHS=1,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=50,
        ONLY_INFERENCE=False,
        INIT_SERIAL=True,
    )

    experiments = [experiment5]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")

# Last Experiment - [MasterModel] Saved model to experiments/22_02_2025-21_28_27_Experiment5/trained_model_master.pth
