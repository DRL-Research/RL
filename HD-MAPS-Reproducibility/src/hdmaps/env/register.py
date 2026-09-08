from gymnasium.envs.registration import register, registry


def register_envs() -> None:
    if "RELintersection-v0" not in registry:
        register(
            id="RELintersection-v0",
            entry_point="hdmaps.env.highway.intersection_class:MultiAgentIntersectionEnv",
        )
    if "RELchain-intersection-v0" not in registry:
        register(
            id="RELchain-intersection-v0",
            entry_point="hdmaps.env.highway.chain_intersection_class:MultiAgentChainIntersectionEnv",
        )
