import config
from utils.experiment import Experiment
from utils.training_loop import training_loop


if __name__ == "__main__":

    # Define experiments
    config_exp1 = {'EXPERIMENT_ID': 'EXP26'}
    experiment1 = Experiment(config_exp1)

    # Define experiments list
    experiments = [experiment1]

    # Run all experiments
    for experiment in experiments:
        config = config.Config()
        config.set_config(experiment.config_dict)
        training_loop(config)

