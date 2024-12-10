from gymnasium.envs.registration import register
from model.model_handler import Model
from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment
from utils.model.model_handler import Model
import highway_env
import gymnasium as gym
print(gym.envs.registry.keys())


if __name__ == "__main__":

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
    experiment2 = Experiment(
        EXPERIMENT_ID='Experiment2',
        # experiment details:
        # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        ONLY_INFERENCE=False,
        EPOCHS=5,
        ROLE=Role.CAR1,
        LEARNING_RATE=1e-4,
        MODEL_TYPE=ModelType.PPO,
        LOAD_PREVIOUS_WEIGHT=False,
        BYPASS_RANDOM_INITIALIZATION=False

    )

    experiments = [experiment2]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")

