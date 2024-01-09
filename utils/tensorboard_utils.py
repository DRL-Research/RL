import tensorflow as tf


def init_tensorboard(experiment_id, current_datetime):
    # Create a unique experiment directory path by combining experiment_id and currentDateTime
    experiment_logging_dir = f"experiments/{experiment_id}/rewards_and_losses/{current_datetime}"

    # Create a TensorBoard writer for the specified experiment directory
    tensorboard = tf.summary.create_file_writer(experiment_logging_dir)

    # Return the TensorBoard writer for logging data
    return tensorboard

