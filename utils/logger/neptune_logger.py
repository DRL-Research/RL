import os
import neptune
import numpy as np
from neptune.types import File
import pandas as pd


class NeptuneLogger:
    def __init__(self, project_name, api_token, run_name=None, tags=None):
        self.run = neptune.init_run(
            project=project_name,
            api_token=api_token,
            name=run_name,
            tags=tags
        )

    def log_metric(self, name, value, step=None):
        if step is not None:
            self.run[name].append(value, step=step)
        else:
            self.run[name].append(value)


    def log_artifact(self, artifact_path, artifact_name):
        if os.path.exists(artifact_path):
            self.run[f"artifacts/{artifact_name}"].upload(File(artifact_path))
        else:
            print(f"Artifact file not found: {artifact_path}")

    def log_text(self, name, text):
        self.run[name] = text

    def log_image(self, name, image_path):
        if os.path.exists(image_path):
            self.run[name].upload(File(image_path))
        else:
            print(f"Image file not found: {image_path}")

    def log_from_csv(self, path, column_name, metric_name):
        try:
            data = pd.read_csv(path)

            # Extract values from the specified column
            values = data[column_name].tolist()

            # Log each value to Neptune
            for step, value in enumerate(values):
                self.log_metric(metric_name, value, step=step)

            print(f"Successfully logged '{metric_name}' from '{column_name}' in '{path}' to Neptune.")
        except FileNotFoundError:
            print(f"CSV file not found: {path}")
        except KeyError:
            print(f"Column '{column_name}' not found in CSV file.")
        except Exception as e:
            print(f"Error logging data from CSV: {e}")

    def stop(self):
        self.run.stop()

    def log_metrics(self, metrics: dict, step=None):
        for metric_name, value in metrics.items():

            if step is not None:
                self.run[metric_name].append(value, step=step)
            else:
                self.run[metric_name].append(value)

    def log_actions_per_episode(self, actions_per_episode: list, car_name: str):
        actions_str = [str(action) for action in actions_per_episode]
        actions_str_combined = ', '.join(actions_str)
        self.run[f"{car_name}_actions_per_episode"].log(actions_str_combined)

    def log_state_to_neptune(self, car_state: np.array, car_name: str):  # TODO implement this function
        self.run[f"{car_name}_state"].log(car_state)

    def log_model(self, model_path, model_name="trained_model"):
        model_path = f"{model_path}.zip"
        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
        elif not os.path.isfile(model_path):
            print(f"Model path is not a file: {model_path}")
        else:
            self.run[f"model/{model_name}"].upload(model_path)
            print(f"Successfully logged {model_name} from: {model_path} to Neptune")


    def log_hyperparameters(self, experiment_config):
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

        self.run["hyperparameters"] = hyperparams
