import logging

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import run_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    patch_intersection_env()
    register_intersection_env()

    # ── Experiment 7: 6 controlled vehicles, hierarchical 2-LM + 1-GM ─────────
    exp7_config = Experiment(
        RENDER_MODE=None,            # set to 'human' to watch the simulation
        EXPERIMENT_ID='Experiment7_6cars_hierarchical',
        LOAD_MODEL_DIRECTORY='',     # leave empty to start from scratch
        EPOCHS=1,
        CYCLES=4,                    # 1=all, 2=local masters, 3=agents, 4=global master
    )

    custom_env_configs = {
        exp7_config.EXPERIMENT_ID: sc.full_env_config_exp7,
    }

    experiments = [exp7_config]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
        run_experiment(experiment_config, custom_env_configs[experiment_config.EXPERIMENT_ID])
        print(f"Experiment {experiment_config.EXPERIMENT_ID} completed.")
