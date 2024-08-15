import config
from utils.experiment import Experiment
from utils.training_loop import *


print("Starting script...")  # Debugging script start
if __name__ == "__main__":
    config_exp1 = {
        'EXPERIMENT_ID': 'EXP9-EPOCHS-1-MAX_EPISODES-3',
        'AGENT_ONLY': True,
        'CAR1_INITIAL_POSITION': [-30, 0],
        'CAR2_INITIAL_POSITION': [0, -30],
        'LOG_Q_VALUES': True,
        'LOG_CAR_STATES': True,
        'ONLY_INFERENCE': False,
        'EPOCHS': 1,
        'MAX_EPISODES': 3,
        'LOAD_WEIGHT_DIRECTORY': "experiments/EXP9-EPOCHS-1-MAX_EPISODES-3/weights/epochs_0_100.zip",
        #experiments/EXP6-EPOCHS-5-MAX_EPISODES-100/weights/epochs_0_100.h5
        'TRAIN_OPTION': 'trajectory',
    }
    # Debuggingg.....
    print("Setting up experiment...")
    experiment1 = Experiment(config_exp1)
    experiments = [experiment1]
    for experiment in experiments:
        print("Starting experiment")
        config = config.Config()
        config.set_config(experiment.config_dict)
        print("Config set up complete")
        training_loop_ariel_step(config)
        print("Experiment completed.\n")

