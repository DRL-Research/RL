import numpy as np

from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.model.model_handler import get_latest_model, get_model_path_from_experiment_name
from utils.training_loop import run_experiment

if __name__ == "__main__":

    experiment4 = Experiment(
        EXPERIMENT_ID='Experiment4',
        EPOCHS= 1,
        MODEL_TYPE=ModelType.CustomPPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=25,
        ONLY_INFERENCE=False,
        THROTTLE_SLOW=0.6,
        THROTTLE_FAST=1.0,
        FIXED_THROTTLE=np.random.uniform(0.6, 1.0, 1),
        LEARNING_RATE=0.01,
        SELF_PLAY_MODE=True,
        LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("27_12_2024-13_24_21_Experiment1")
    )

    # experiment4_inference = Experiment(
    #     EXPERIMENT_ID='Experiment4_inference',
    #     EPOCHS=50,
    #     ROLE=Role.BOTH,
    #     MODEL_TYPE=ModelType.CustomPPO,
    #     TIME_BETWEEN_STEPS=0.75,
    #     EXPLORATION_EXPLOTATION_THRESHOLD=1,
    #     ONLY_INFERENCE=True,
    #     THROTTLE_SLOW=0.6,
    #     THROTTLE_FAST=1.0,
    #     FIXED_THROTTLE=np.random.uniform(0.6, 1.0),
    #     LEARNING_RATE=0.01,
    #     LOAD_MODEL_DIRECTORY=get_latest_model('experiments')
    # )

    experiments = [experiment4]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
