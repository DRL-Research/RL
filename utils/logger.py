import tensorflow as tf


class Logger:

    def __init__(self, experiment_id, experiment_date_time):
        # Create a unique experiment directory path by combining experiment_id and currentDateTime
        self.experiment_logging_dir = f"experiments/{experiment_id}/rewards_and_losses/{experiment_date_time}"
        # Create a TensorBoard writer for the specified experiment directory
        self.logger = tf.summary.create_file_writer(self.experiment_logging_dir)

    def log(self, log_name, log_x_value, log_y_value):
        with self.logger.as_default():
            tf.summary.scalar(log_name, log_x_value, step=log_y_value)
