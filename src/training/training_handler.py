import logging
from typing import Tuple

from src.plotting_utils.plotting_utils import plot_training_results
from src.training.episode_utils import process_episode
from src.training.general_utils import (
    setup_experiment_dirs,
    initialize_models,
    setup_loggers,
    close_everything,
)
from src.training.training_loop_utils import (
    init_training_results,
    prepare_master_for_cycle,
    perform_training_phase,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##########################################
# Training Loop
##########################################


def training_loop(experiment, env, master_model):
    """Main training loop that orchestrates cycles and episodes."""

    collision_counter = 0
    episode_counter = 0

    results = init_training_results()

    for cycle_num in range(1, experiment.CYCLES + 1):
        print('Cycle', cycle_num, 'out of ', experiment.CYCLES)
        prepare_master_for_cycle(cycle_num, experiment.CYCLES, master_model)

        for _ in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            print()
            print("*" * 40)
            print('Episode ', episode_counter, '/', experiment.EPISODES_PER_CYCLE * experiment.CYCLES, 'episodes')
            print("*" * 40)
            print()

            episode_reward, actions, steps, crashed, final_state = process_episode(
                episode_counter,
                env,
                master_model,
                experiment,
            )

            if crashed:
                collision_counter += 1

            results["episode_rewards"].append(episode_reward)
            results["all_actions"].append(actions)

            if episode_counter % experiment.EPISODE_AMOUNT_FOR_TRAIN == 0:
                perform_training_phase(master_model, final_state, results)

    print("Training completed.")
    return master_model, collision_counter, results


##########################################
# Main experiment entrypoint
##########################################


def run_experiment(experiment_config, env_config) -> Tuple[object, int, dict]:
    print(
        f"Environment configuration: {len(env_config['controlled_cars'])} controlled cars, {len(env_config['static_cars'])} static cars"
    )
    setup_experiment_dirs(experiment_config.EXPERIMENT_PATH)
    master_model, env = initialize_models(experiment_config, env_config)
    master_logger = setup_loggers(experiment_config.EXPERIMENT_PATH)
    master_model.set_logger(master_logger)

    print("Running in training mode")
    master_model, collision_counter, training_results = training_loop(
        experiment=experiment_config,
        env=env,
        master_model=master_model,
    )
    plot_training_results(experiment_config, training_results, show_plots=True)
    print("Training completed.")
    print("Total collisions:", collision_counter)
    close_everything(env, master_logger)
    return master_model, collision_counter, training_results
