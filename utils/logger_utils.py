from stable_baselines3.common.logger import Logger
import os


def log_metrics(logger: Logger, metrics: dict, step: int):
    for metric_name, value in metrics.items():
        logger.record(metric_name, value)


def save_model(model, save_directory: str):
    os.makedirs(save_directory, exist_ok=True)
    model.save(save_directory)


def log_hyperparameters(logger: Logger, experiment_config):
    hyperparams = {
        "experiment_id": experiment_config.EXPERIMENT_ID,
        "model_type": experiment_config.MODEL_TYPE,
        "epochs": experiment_config.EPOCHS,
        "learning_rate": experiment_config.LEARNING_RATE,
        "n_steps": experiment_config.N_STEPS,
        "batch_size": experiment_config.BATCH_SIZE,
        "seed": experiment_config.SEED,
        "policy_network": str(experiment_config.PPO_NETWORK_ARCHITECTURE['pi']),
        "value_network": str(experiment_config.PPO_NETWORK_ARCHITECTURE['vf']),
        "loss_function": experiment_config.LOSS_FUNCTION,
        "random_init": experiment_config.RANDOM_INIT,
        "time_between_steps": experiment_config.TIME_BETWEEN_STEPS,
        "exploration_threshold": experiment_config.EXPLORATION_EXPLOTATION_THRESHOLD,
        "success_reward": experiment_config.REACHED_TARGET_REWARD,
        "starvation_reward": experiment_config.STARVATION_REWARD,
        "collision_reward": experiment_config.COLLISION_REWARD,
        "car1_initial_position_options": str([
            experiment_config.CAR1_INITIAL_POSITION_OPTION_1,
            experiment_config.CAR1_INITIAL_POSITION_OPTION_2,
        ]),
        "car2_initial_position_options": str([
            experiment_config.CAR2_INITIAL_POSITION_OPTION_1,
            experiment_config.CAR2_INITIAL_POSITION_OPTION_2,
        ]),
        "car1_action_fast": experiment_config.THROTTLE_FAST,
        "car1_action_slow": experiment_config.THROTTLE_SLOW,
        "car2_action": (experiment_config.THROTTLE_FAST + experiment_config.THROTTLE_SLOW) / 2,
        "car1_state": "[x1, y1, Vx1, Vy1, x2, y2, Vx2, Vy2]",
        "car2_state": "[x2, y2, Vx2, Vy2, x1, y1, Vx1, Vy1]",
    }

    for param_name, value in hyperparams.items():
        logger.record(param_name, value)
