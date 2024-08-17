import config
from utils.experiment import Experiment
from utils.training_loop import *


if __name__ == "__main__":

    # Define experiments
    config_exp1 = {'EXPERIMENT_ID': 'EXP9',
                   'AGENT_ONLY': True,
                   'CAR1_INITIAL_POSITION': [-30, 0],
                   'CAR2_INITIAL_POSITION': [0, -30],
                   'LOG_Q_VALUES': True,
                   'LOG_CAR_STATES': True,
                   # 'EPSILON_DECAY': 0.995,

                   # 'LEARNING_RATE': 0.001,  # 0.001 / 0.0001 / 0.00001
                   # 'EPSILON': 0.9,  # 0.9 / 0.1 (for inference only)
                   'MAX_EPISODES': 1,
                   'ONLY_INFERENCE': True,
                   # 'ONLY_INFERENCE': False,
                   'LOAD_WEIGHT_DIRECTORY': "experiments/EXP8/weights/epochs_0_100.h5"
                   }
    experiment1 = Experiment(config_exp1)

    # Define experiments list
    experiments = [experiment1, experiment1]

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
