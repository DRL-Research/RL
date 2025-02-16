import numpy as np

from model.model_handler import get_latest_model
from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment
from utils.model.model_handler import get_model_path_from_experiment_name

if __name__ == "__main__":

    experiment5 = Experiment(
        EXPERIMENT_ID='Experiment5',
        EPOCHS= 100,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=200,
        ONLY_INFERENCE=False,
        INIT_SERIAL=True
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )

    experiment5_inference=Experiment(
        EXPERIMENT_ID='Experiment5_in',
        EPOCHS= 100,
        ROLE=Role.BOTH,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=200,
        ONLY_INFERENCE=True,
        INIT_SERIAL=True,
        #LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("experiments/15_02_2025-21_15_42_Experiment5")
    )

    # experiments = [experiment1, experiment2]
    experiments = [experiment5_inference]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
