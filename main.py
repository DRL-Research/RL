from utils.experiment import Experiment
from utils.training_loop import run_experiment

if __name__ == "__main__":

    # TODO: create new interpreter with req file.

    # TODO: create all (start with first 2) experiments configs, run them one after the other.

    experiment1 = Experiment(
        EXPERIMENT_ID='Car_2_high_speed',
        # CAR1_INITIAL_POSITION=[-30, 0],
        # CAR2_INITIAL_POSITION=[0, 30],
        # CAR1_INITIAL_YAW=0,
        # CAR2_INITIAL_YAW=270,
        ONLY_INFERENCE=False,
        EPOCHS=2,
        LOAD_WEIGHT_DIRECTORY="experiments/Car_2_Car2_high_speed/trained_model/model.zip",
        ROLE='Car1',
        EXPLORATION_EXPLOTATION_THRESHOLD=200,
        TIME_BETWEEN_STEPS=0.05, # TODO: same as default value no?
        LEARNING_RATE=1, # TODO: same as default value no?
        N_STEPS=160, # TODO: same as default value no?
        BATCH_SIZE=160, # TODO: same as default value no?
        FIXED_THROTTLE=1, # TODO: same as default value no?
        )

    experiments = [experiment1]
    for experiment in experiments:
        print(f"Starting experiment: {experiment.EXPERIMENT_ID}")
        run_experiment(experiment)
        print(f"Experiment {experiment.EXPERIMENT_ID} completed.")
