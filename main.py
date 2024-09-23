from utils.experiment import Experiment
from utils.training_loop import run_experiment

if __name__ == "__main__":

    # TODO: fix requirements
    # TODO: what is the purpose of src folder? if not relevant put in .gitignore

    experiment1 = Experiment(
        EXPERIMENT_ID='Car_2_high_speed',
        CAR1_INITIAL_POSITION=[-30, 0],
        CAR2_INITIAL_POSITION=[0, 30],
        CAR1_INITIAL_YAW=0,
        CAR2_INITIAL_YAW=270,
        ONLY_INFERENCE=False,
        MAX_EPISODES=100,
        # TODO: naming of LOAD_WEIGHT_DIRECTORY should be more generic + if saving only the model, than should be named trained_models (not experiments)
        LOAD_WEIGHT_DIRECTORY="experiments/24_08_2024-19_45_30Car_2_Random_Side/model.zip",
        ROLE='Car1',
        EXPLORATION_EXPLOTATION_THRESHOLD=2500,
        TIME_BETWEEN_STEPS=0.05,
        LEARNING_RATE=1,
        N_STEPS=160,
        BATCH_SIZE=160,
        FIXED_THROTTLE=1)

    experiments = [experiment1]
    for experiment in experiments:
        print(f"Starting experiment: {experiment.EXPERIMENT_ID}")
        run_experiment(experiment)
        print(f"Experiment {experiment.EXPERIMENT_ID} completed.")
