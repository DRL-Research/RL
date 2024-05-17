import tensorflow as tf


class TensorBoard:

    def __init__(self, experiment_id, experiment_date_time):

        # Create a unique experiment directory path by combining experiment_id and currentDateTime
        experiment_logging_dir = f"experiments/{experiment_id}/rewards_and_losses/{experiment_date_time}"

        # Create a TensorBoard writer for the specified experiment directory
        self.tensorboard = tf.summary.create_file_writer(experiment_logging_dir)
