from __future__ import annotations

import pytest

from hdmaps.paths import agent_checkpoint, master_checkpoint


@pytest.mark.slow
def test_checkpoints_load():
    from hdmaps.policies.master import load_models

    master, agent = load_models(str(agent_checkpoint()), str(master_checkpoint()))
    assert master.embedding_size == 4
    assert agent.observation_space.shape[0] == 8
