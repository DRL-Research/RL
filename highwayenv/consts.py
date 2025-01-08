CONFIG_EXP2 = (
        {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False,
            },
            "action": {
                "type": "CustomDiscreteAction",
                "target_speeds": [5, 10],
            },
            "duration": 13,  # [s]
            "destination": "o2",
            "controlled_vehicles": 2,
            "initial_vehicle_count": 0,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": -20,
            "high_speed_reward": 2,
            "arrived_reward": 10,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False,
        })

