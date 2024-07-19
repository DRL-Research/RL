import config
from utils.experiment import Experiment
from utils.training_loop import *


if __name__ == "__main__":

    # Define experiments
    config_exp1 = {'EXPERIMENT_ID': 'EXP1',
                   'AGENT_ONLY': True}
    experiment1 = Experiment(config_exp1)

    # Define experiments list
    experiments = [experiment1]

    # Run all experiments
    for experiment in experiments:
        config = config.Config()
        config.set_config(experiment.config_dict)

        if config.AGENT_ONLY:
            training_loop_agent_only(config)
        else:
            training_loop(config)

        # TODO: reset cars to initial position
