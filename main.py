import config
from utils.experiment import Experiment
from utils.training_loop import *



if __name__ == "__main__":

    config_exp1 = {'EXPERIMENT_ID': 'EXP6-EPOCHS-5-MAX_EPISODES-100',
                   'AGENT_ONLY': True,
                   'CAR1_INITIAL_POSITION': [-30, 0],
                   'CAR2_INITIAL_POSITION': [0, -30],
                   'LOG_Q_VALUES': True,
                   'LOG_CAR_STATES': True,
                   ######################################
                   'ONLY_INFERENCE': True,
                   'EPOCHS': 5,
                   'MAX_EPISODES': 100,
                   'LOAD_WEIGHT_DIRECTORY': "experiments/EXP6-EPOCHS-5-MAX_EPISODES-100/weights/epochs_0_100.h5",
                   }
    # Define experiments
    experiment1 = Experiment(config_exp1)

    # Define experiments list
    experiments = [experiment1]

    # Run all experiments
    for experiment in experiments:
        config = config.Config()
        config.set_config(experiment.config_dict)

        if config.AGENT_ONLY:
            # training_loop_agent_only(config)
            training_loop_ido(config)
        else:
            print("")
            # training_loop(config)

        # TODO: reset cars to initial position
