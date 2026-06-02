from gymnasium.envs.registration import registry, register

def patch_intersection_env():
    from highwayenv.intersection_class import MultiAgentIntersectionEnv
    original_agent_rewards = MultiAgentIntersectionEnv._agent_rewards
    def patched_agent_rewards(self, vehicle):
        if 'reward_speed_range' not in self.config:
            self.config['reward_speed_range'] = [7.0, 9.0]
        return original_agent_rewards(self, vehicle)
    MultiAgentIntersectionEnv._agent_rewards = patched_agent_rewards

def register_intersection_env():
    register(id='RELintersection-v0', entry_point='highwayenv.intersection_class:MultiAgentIntersectionEnv')

def register_roundabout_env():
    from highwayenv.roundabout_class import MultiAgentRoundaboutEnv
    register(id='RELroundabout-v0', entry_point='highwayenv.roundabout_class:MultiAgentRoundaboutEnv')

def register_double_intersection_env():
    from highwayenv.double_intersection_class import MultiAgentDoubleIntersectionEnv
    register(id='RELdouble-intersection-v0', entry_point='highwayenv.double_intersection_class:MultiAgentDoubleIntersectionEnv')

def register_composable_env():
    from highwayenv.composable_road_class import MultiAgentComposableRoadEnv
    register(id='RELcomposable-layout-v0', entry_point='highwayenv.composable_road_class:MultiAgentComposableRoadEnv')
