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
        EPOCHS= Experiment.CYCLES* Experiment.EPISODES_PER_CYCLE,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.5,
        EXPLORATION_EXPLOTATION_THRESHOLD=500,
        ONLY_INFERENCE=False,
        THROTTLE_SLOW=0.4,
        THROTTLE_FAST=0.6,
        FIXED_THROTTLE=np.random.uniform(0.4, 0.6, 1),
        LEARNING_RATE=0.001,
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )


    # experiments = [experiment1, experiment2]
    experiments = [experiment5]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
