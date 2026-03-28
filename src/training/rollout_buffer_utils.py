from src.project_globals import rollout_buffers


def reset_all_buffers():
    for rollout_buffer in rollout_buffers:
        rollout_buffer.reset()