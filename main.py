import random
from os_file_handler import get_latest_model
from utils.experiment import Experiment
from utils.training_loop import run_experiment

if __name__ == "__main__":

    # experiment1 = Experiment(
    #     EXPERIMENT_ID='Experiment1',
    #     #experiment details:
    #     #Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
    #     #Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
    #     #Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
    #     ONLY_INFERENCE=False,
    #     EPOCHS=50,
    #     ROLE='Car1',
    #     EXPLORATION_EXPLOTATION_THRESHOLD=500,
    #     TIME_BETWEEN_STEPS=0.05,
    #     LEARNING_RATE=1,
    #     N_STEPS=160,
    #     BATCH_SIZE=160,
    #     FIXED_THROTTLE=0.4,
    #     THROTTLE_FAST=0.5,
    #     THROTTLE_SLOW=0.3
    #     )

    #Debugging
    experiment1 = Experiment(
        EXPERIMENT_ID='Experiment1',
        #experiment details:
        #Car2 (going right/left randomly) fixed speed (0.4) - 100 episodes learning. expecting to see that car1 will always go fast to avoid crashes.
        #Rewards and loss (during training): Success reward: +10 Collision reward: -20 Starvation reward: -0.1
        #Hence, negative cumulative reward even if success due to (starvation reward*steps) + success reward.
        ONLY_INFERENCE=False,
        EPOCHS=50,
        ROLE='Car1',
        EXPLORATION_EXPLOTATION_THRESHOLD=2500,
        TIME_BETWEEN_STEPS=0.05,
        LEARNING_RATE=0.0000001,
        N_STEPS=200,
        BATCH_SIZE=200,
        #LOAD_WEIGHT_DIRECTORY=get_latest_model("experiments")
        )

    # experiment2 = Experiment(
    #     EXPERIMENT_ID='Experiment2',
    #     ONLY_INFERENCE=False,
    #     EPOCHS=50,
    #     ROLE='Car1',
    #     EXPLORATION_EXPLOTATION_THRESHOLD=1000,
    #     TIME_BETWEEN_STEPS=0.05,
    #     LEARNING_RATE=1,
    #     N_STEPS=190,
    #     BATCH_SIZE=190,
    #     FIXED_THROTTLE=0.75,
    #     THROTTLE_FAST=0.75,
    #     THROTTLE_SLOW=0.5
    # )

    # experiment2_INFERENCE_ONLY = Experiment(
    #     EXPERIMENT_ID='Experiment2_INFERENCE_ONLY',
    #     ONLY_INFERENCE=True,
    #     EPOCHS=5,
    #     ROLE='Both',
    #     EXPLORATION_EXPLOTATION_THRESHOLD=0,
    #     TIME_BETWEEN_STEPS=0.05,
    #     LEARNING_RATE=1,
    #     N_STEPS=160,
    #     BATCH_SIZE=160,
    #     FIXED_THROTTLE=0.75,
    #     THROTTLE_FAST=1,
    #     THROTTLE_SLOW=0.5,
    #     LOAD_WEIGHT_DIRECTORY= get_latest_model("experiments")
    # )

    # experiment3 = Experiment(
    #     EXPERIMENT_ID='Experiment3',
    #     ONLY_INFERENCE=False,
    #     EPOCHS=50,
    #     ROLE='Both',
    #     EXPLORATION_EXPLOTATION_THRESHOLD=500,
    #     TIME_BETWEEN_STEPS=0.05,
    #     LEARNING_RATE=1,
    #     N_STEPS=160,
    #     BATCH_SIZE=160,
    #     FIXED_THROTTLE=random.uniform(0.4,0.75),
    #     THROTTLE_FAST=0.75,
    #     THROTTLE_SLOW=0.4
    # )

    experiments = [experiment1]
    for experiment in experiments:
        print(f"Starting experiment: {experiment.EXPERIMENT_ID}")
        run_experiment(experiment)
        print(f"Experiment {experiment.EXPERIMENT_ID} completed.")