import config
from utils.experiment import Experiment
from utils.training_loop import *
from training_loop import model_training

if __name__ == "__main__":
    config_exp1 = {
        'EXPERIMENT_ID': 'Car_2_high_speed',
        'AGENT_ONLY': True,
        'CAR1_INITIAL_POSITION': [-30, 0],
        'CAR2_INITIAL_POSITION': [0, 30],
        'CAR1_INITIAL_YAW': 0,
        'CAR2_INITIAL_YAW': 270,
        'ONLY_INFERENCE': False,
        'MAX_EPISODES': 50,
        'LOAD_WEIGHT_DIRECTORY': 'experiments/12_09_2024-13_41_21_Car_2_Random_Side/model.zip',
        'ROLE': 'Car1',
        'EXPLORATION_EXPLOTATION_THRESHOLD': 500,
        'TIME_BETWEEN_STEPS': 0.05,
        'LEARNING_RATE': 0.01,
        'N_STEPS': 2400,
        'BATCH_SIZE': 2400,
        'FIXED_THROTTLE': 0.75,
        ####################
    }

    print("Setting up experiment...")
    experiment1 = Experiment(config_exp1)
    experiments = [experiment1]
    for experiment in experiments:
        print("Starting experiment")
        config = config.Config()
        config.set_config(experiment.config_dict)
        print(config.SAVE_WEIGHT_DIRECTORY)
        print("Config set up complete")
        model_training(config, config.SAVE_WEIGHT_DIRECTORY)
        print("Experiment completed.\n")
