import config
from utils.experiment import Experiment
from utils.training_loop import *
from training_loop import training_loop_ariel_step
from datetime import datetime
exp_id='Car_2_Random_Side'
path = 'experiments/' + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") +exp_id
print("Starting script...")  # Debugging script start
if __name__ == "__main__":
    config_exp1 = {
        'EXPERIMENT_ID': 'Car_2_Random_Side',
        'AGENT_ONLY': True,
        'CAR1_INITIAL_POSITION': [-30, 0],
        'CAR2_INITIAL_POSITION': [0, 30],
        'CAR1_INITIAL_YAW': 0,
        'CAR2_INITIAL_YAW': 270,
        'LOG_Q_VALUES': True,
        'LOG_CAR_STATES': True,
        'ONLY_INFERENCE': True,
        'EPOCHS': 1,
        'MAX_EPISODES': 100,
        'LOAD_WEIGHT_DIRECTORY': "experiments/24_08_2024-19_45_30Car_2_Random_Side/model.zip",
        'SAVE_WEIGHT_DIRECTORY': path,
        #experiments/EXP6-EPOCHS-5-MAX_EPISODES-100/weights/epochs_0_100.h5
        'TRAIN_OPTION': 'trajectory',
        'ROLE': 'Car1', #who using the DRL model? options are: Car1 or Car2 or Both
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
        training_loop_ariel_step(config,path)
        print("Experiment completed.\n")

