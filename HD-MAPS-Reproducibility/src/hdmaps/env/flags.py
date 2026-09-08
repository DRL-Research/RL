after_is_arrived_flags: list[bool] = []
rollout_buffers: list = []
local_master_rollout_buffers: list = []
global_master_rollout_buffer = None
episode_count = 0


def reset_globals() -> None:
    global global_master_rollout_buffer, episode_count
    after_is_arrived_flags.clear()
    rollout_buffers.clear()
    local_master_rollout_buffers.clear()
    global_master_rollout_buffer = None
    episode_count = 0
