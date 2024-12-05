from experiment.experiment_config import Experiment
from experiment.experiment_constants import Role
from model.model_constants import ModelType
from utils.training_loop import run_experiment

if __name__ == "__main__":

    experiment1 = Experiment(
        EXPERIMENT_ID='Experiment1',
        #experiment details:
        #Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        #Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        #Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        ONLY_INFERENCE=False,
        EPOCHS=2,
        ROLE=Role.CAR1,
        EXPLORATION_EXPLOTATION_THRESHOLD=50,
        LEARNING_RATE=200,
        MODEL_TYPE=ModelType.PPO,
        #LOAD_WEIGHT_DIRECTORY=get_latest_model("experiments")
        )

    experiment2 = Experiment(
        EXPERIMENT_ID='Experiment1',
        #experiment details:
        #Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        #Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        #Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        ONLY_INFERENCE=False,
        EPOCHS=2,
        ROLE=Role.CAR1,
        EXPLORATION_EXPLOTATION_THRESHOLD=50,
        LEARNING_RATE=200,
        MODEL_TYPE=ModelType.DQN,
        #LOAD_WEIGHT_DIRECTORY=get_latest_model("experiments")
        )


    experiments = [experiment1, experiment2]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config)
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")