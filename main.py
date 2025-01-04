import numpy as np

from model.model_handler import get_latest_model
from utils.experiment.experiment_config import Experiment
from utils.experiment.experiment_constants import Role
from utils.model.model_constants import ModelType
from utils.training_loop import run_experiment
from utils.model.model_handler import get_model_path_from_experiment_name

if __name__ == "__main__":

    experiment1 = Experiment(
        EXPERIMENT_ID='Experiment1',
        # experiment details:
        # Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        # Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        # Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        EPOCHS=100,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        # INFERENCE Mode
        ONLY_INFERENCE=False,
        EXPLORATION_EXPLOTATION_THRESHOLD=250,
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )

    experiment2 = Experiment(
        EXPERIMENT_ID='Experiment2',
        # experiment details:
        # Car2 (going right/left randomly) fixed high speed - 100 episodes learning.
        # expecting to see that car1 will always slow down to avoid crashes.
        EPOCHS=50,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=200,
        # INFERENCE Mode
        ONLY_INFERENCE=False,
        THROTTLE_SLOW=0.6,
        THROTTLE_FAST=1.0,
        FIXED_THROTTLE=0.8,
        LEARNING_RATE=0.01,
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )

    experiment3 = Experiment(
        EXPERIMENT_ID='Experiment3',
        # experiment details:
        # Car2 (going right/left randomly) fixed high speed - 100 episodes learning.
        # expecting to see that car1 will always slow down to avoid crashes.
        EPOCHS=100,
        ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=400,
        # INFERENCE Mode
        ONLY_INFERENCE=False,
        THROTTLE_SLOW=0.6,
        THROTTLE_FAST=1.0,
        FIXED_THROTTLE=np.random.uniform(0.6, 1.0),
        LEARNING_RATE=0.01,
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )

    experiment4 = Experiment(
        EXPERIMENT_ID='Experiment4',
        # experiment details:
        # Car2 (going right/left randomly) fixed high speed - 100 episodes learning.
        # expecting to see that car1 will always slow down to avoid crashes.
        EPOCHS= Experiment.CYCLES* Experiment.EPISODES_PER_CYCLE,
        #ROLE=Role.CAR1,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=25,
        # INFERENCE Mode
        ONLY_INFERENCE=False,
        THROTTLE_SLOW=0.6,
        THROTTLE_FAST=1.0,
        FIXED_THROTTLE=np.random.uniform(0.6, 1.0, 1),
        LEARNING_RATE=0.01,
        SELF_PLAY_MODE=True
        # LOAD_MODEL_DIRECTORY=get_model_path_from_experiment_name("15_12_2024-20_08_51_Experiment1")
    )

    experimenti_infernce = Experiment(
        EXPERIMENT_ID='Experiment3_infernce',
        # experiment details:
        # Car2 (going right/left randomly) fixed high speed - 100 episodes learning.
        # expecting to see that car1 will always slow down to avoid crashes.
        EPOCHS=50,
        ROLE=Role.BOTH,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=500,
        # INFERENCE Mode
        ONLY_INFERENCE=True,
        THROTTLE_SLOW=0.6,
        THROTTLE_FAST=1.0,
        FIXED_THROTTLE=np.random.uniform(0.6, 1.0),
        LEARNING_RATE=0.01,
        LOAD_MODEL_DIRECTORY=get_latest_model('experiments')
    )

    experiment4_infernce = Experiment(
        EXPERIMENT_ID='Experiment4_infernce',
        # experiment details:
        # Car2 (going right/left randomly) fixed high speed - 100 episodes learning.
        # expecting to see that car1 will always slow down to avoid crashes.
        EPOCHS=50,
        ROLE=Role.BOTH,
        MODEL_TYPE=ModelType.PPO,
        TIME_BETWEEN_STEPS=0.75,
        EXPLORATION_EXPLOTATION_THRESHOLD=1,
        # INFERENCE Mode
        ONLY_INFERENCE=True,
        THROTTLE_SLOW=0.6,
        THROTTLE_FAST=1.0,
        FIXED_THROTTLE=np.random.uniform(0.6, 1.0),
        LEARNING_RATE=0.01,
        LOAD_MODEL_DIRECTORY=get_latest_model('experiments')
    )

    # experiments = [experiment1, experiment2]
    experiments = [experiment4_infernce]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
