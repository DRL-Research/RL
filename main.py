import config
from utils.experiment import Experiment
from utils.training_loop import *
from training_loop import model_training

if __name__ == "__main__":
    config_exp1 = {
        'EXPERIMENT_ID': 'Car_2_Random_Side',
        'AGENT_ONLY': True,
        'CAR1_INITIAL_POSITION': [-30, 0],
        'CAR2_INITIAL_POSITION': [0, 30],
        'CAR1_INITIAL_YAW': 0,
        'CAR2_INITIAL_YAW': 270,
        'ONLY_INFERENCE': True,
        'MAX_EPISODES': 100,
        'LOAD_WEIGHT_DIRECTORY': "experiments/24_08_2024-19_45_30Car_2_Random_Side/model.zip",
        'ROLE': 'Both',
        'EXPLORATION_EXPLOTATION_THRESHOLD': 2500,
        'TIME_BETWEEN_STEPS': 0.05,
        'LEARNING_RATE': 0.001,
        'N_STEPS': 160,
        'BATCH_SIZE': 160,
    }

    print("Starting script...")  # Debugging script start

    print("Setting up experiment...")
    experiment1 = Experiment(config_exp1)
    experiments = [experiment1]

    for experiment in experiments:
        print("Starting experiment")
        config = config.Config()
        config.set_config(experiment.config_dict)
        print("Config set up complete")
        model_training(config, config.SAVE_WEIGHT_DIRECTORY)
        print("Experiment completed.\n")
