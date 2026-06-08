after_is_arrived_flags = []

# One RolloutBuffer per agent (6 in EXP-7)
rollout_buffers = []

# Two RolloutBuffers for Local Master 1 and Local Master 2
# (both train the SAME shared MasterModel)
local_master_rollout_buffers = []

# One RolloutBuffer for the Global Master
# (also trains the same shared MasterModel, but with identifier_bit=1 inputs)
global_master_rollout_buffer = None

episode_count = 0


def reset_globals():
    """Clear all module-level state so a fresh experiment can start cleanly.

    IMPORTANT: use .clear() on lists — never `rollout_buffers = []`, or any module
    that did `from project_globals import rollout_buffers` keeps pointing at the
    OLD list and training writes to the wrong buffers.
    """
    global global_master_rollout_buffer, episode_count
    after_is_arrived_flags.clear()
    rollout_buffers.clear()
    local_master_rollout_buffers.clear()
    global_master_rollout_buffer = None
    episode_count = 0
