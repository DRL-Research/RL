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


    def log_chart_from_csv(self, chart_name, csv_path, x_column=None, y_column=None, x_label="X-Axis", y_label="Y-Axis", title="Chart"):
        """
        Logs a chart to Neptune directly from a CSV file.

        Args:
            chart_name (str): Name of the chart in Neptune.
            csv_path (str): Path to the CSV file.
            x_column (str): Column name for the X-axis. Defaults to the index if None.
            y_column (str): Column name for the Y-axis. Required.
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis.
            title (str): Title of the chart.
        """
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        try:
            # Load the CSV
            data = pd.read_csv(csv_path)

            # Validate the y_column
            if y_column not in data.columns:
                print(f"Column '{y_column}' not found in CSV file.")
                return

            # Use index if x_column is not specified
            x_data = data.index if x_column is None else data[x_column]
            y_data = data[y_column]

            # Log the chart
            self.run[chart_name].log_chart(
                title=title,
                x=x_data.tolist(),
                y=y_data.tolist(),
                xaxis=x_label,
                yaxis=y_label,
            )
            print(f"Chart '{chart_name}' logged successfully from CSV.")

        except Exception as e:
            print(f"Error logging chart from CSV: {e}")

    def stop(self):
        self.run.stop()

# Example usage:
# logger = NeptuneLogger("katiusha8642/DRL", "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjJkY2Q5Zi03YWRkLTQxMjMtYmUwYi1iYzM5ZGNmNDkxMGEifQ==", run_name="Experiment 1", tags=["experiment", "test"])
# logger.log_hyperparameters({"learning_rate": 0.001, "batch_size": 32})
# logger.log_metric("accuracy", 0.95, step=1)
# logger.log_model("path/to/trained_model.pth")
# logger.log_text("experiment_notes", "This experiment tests Neptune integration.")
# logger.stop()
