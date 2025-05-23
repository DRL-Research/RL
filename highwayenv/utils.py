from gymnasium.envs.registration import registry, register

from highwayenv.intersection_class import IntersectionEnv


def patch_intersection_env():
    """
    Monkey-patch the IntersectionEnv class to handle missing reward_speed_range.
    This function modifies the _agent_rewards method at runtime to add a default value
    for reward_speed_range if it's missing from the configuration.
    """

    # Store the original method
    original_agent_rewards = IntersectionEnv._agent_rewards

    # Define the patched method
    def patched_agent_rewards(self, vehicle):
        # Add reward_speed_range if missing
        if "reward_speed_range" not in self.config:
            print("Adding missing reward_speed_range parameter to environment config")
            self.config["reward_speed_range"] = [7.0, 9.0]
        return original_agent_rewards(self, vehicle)

    # Replace the original method with our patched version
    IntersectionEnv._agent_rewards = patched_agent_rewards
    print("Successfully patched IntersectionEnv._agent_rewards method")




def register_intersection_env():
    if "RELintersection-v0" not in registry:
        register(
            id="RELintersection-v0",
            entry_point="highwayenv.intersection_class:IntersectionEnv",
        )
    if "TwoIntersection-v0" not in registry:
        register(
            id="TwoIntersection-v0",
            entry_point="highwayenv.intersection_class:TwoIntersectionEnv",
        )
