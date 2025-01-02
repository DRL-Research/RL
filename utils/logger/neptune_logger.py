import os
import neptune
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


    def log_hyperparameters(self, params):
        self.run["hyperparameters"] = params


    def log_model(self, model_path, model_name="trained_model"):
        if os.path.exists(model_path):
            self.run[f"models/{model_name}"].upload(File(model_path))
        else:
            print(f"Model file not found: {model_path}")


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

            print(f"Successfully logged '{metric_name}' from '{column_name}' in '{path}'.")
        except FileNotFoundError:
            print(f"CSV file not found: {path}")
        except KeyError:
            print(f"Column '{column_name}' not found in CSV file.")
        except Exception as e:
            print(f"Error logging data from CSV: {e}")

    def stop(self):
        self.run.stop()

# Example usage:
# logger = NeptuneLogger("katiusha8642/DRL", "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjJkY2Q5Zi03YWRkLTQxMjMtYmUwYi1iYzM5ZGNmNDkxMGEifQ==", run_name="Experiment 1", tags=["experiment", "test"])
# logger.log_hyperparameters({"learning_rate": 0.001, "batch_size": 32})
# logger.log_metric("accuracy", 0.95, step=1)
# logger.log_model("path/to/trained_model.pth")
# logger.log_text("experiment_notes", "This experiment tests Neptune integration.")
# logger.stop()