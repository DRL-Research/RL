from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment
from utils.model.model_handler import get_model_path_from_experiment_name

if __name__ == "__main__":

  experiment2 = Experiment(
        EXPERIMENT_ID='Experiment2',
        # experiment details:
        # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        EPOCHS=20,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        # INFERENCE Mode
        ONLY_INFERENCE=False,
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
        LEARNING_RATE=1e-2,
        TIME_BETWEEN_STEPS=0.75,
        THROTTLE_FAST=25,
        THROTTLE_SLOW=10,
        FIXED_SPEED_CAR2=15,
        FIXED_SPEED_CAR3=3,
    )


    # experiments = [experiment1, experiment2]
    experiments = [experiment2]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
