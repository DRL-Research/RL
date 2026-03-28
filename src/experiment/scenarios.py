





base_complete_scenarios_3_cars = [
    # Scenario 0: Heavy north-south flow
    # {
    #     "agents": [
    #         (('o0', 'ir0', 0), "o2", 0),
    #         (('o1', 'ir1', 0), "o3", 0),
    #         (('o0', 'ir0', 0), "o2", -50)  # Moved from static
    #     ],
    #     "static": [
    #         (('o2', 'ir2', 0), "o0", -35),
    #         (('o0', 'ir0', 0), "o1", -70)
    #     ]
    # },
    # Scenario 1: All cars turning right
    {
        "agents": [
            (('o0', 'ir0', 0), "o1", 0),
            (('o1', 'ir1', 0), "o2", 0),
            (('o2', 'ir2', 0), "o3", -25)
        ],
        "static": [
            (('o3', 'ir3', 0), "o0", -35),
            (('o0', 'ir0', 0), "o1", -15)
        ]
    },
    # Scenario 2: Crossing patterns
    # {
    #     "agents": [
    #         (('o0', 'ir0', 0), "o3", 0),
    #         (('o2', 'ir2', 0), "o1", 0),
    #         (('o1', 'ir1', 0), "o0", -20)
    #     ],
    #     "static": [
    #         (('o3', 'ir3', 0), "o2", -30),
    #         (('o0', 'ir0', 0), "o2", -40)
    #     ]
    # },
    # Scenario 3: East-west corridor
    {
        "agents": [
            (('o1', 'ir1', 0), "o3", 0),
            (('o3', 'ir3', 0), "o1", 0),
            (('o1', 'ir1', 0), "o3", -30)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -25),
            (('o2', 'ir2', 0), "o1", -35)
        ]
    },
    # Scenario 4: Complex multi-direction
    {
        "agents": [
            (('o2', 'ir2', 0), "o0", 0),
            (('o1', 'ir1', 0), "o2", 0),
            (('o0', 'ir0', 0), "o3", -45)
        ],
        "static": [
            (('o1', 'ir1', 0), "o2", -30),
            (('o2', 'ir2', 0), "o0", -20)
        ]
    },
    # Scenario 5: Same origin dispersal
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o0', 'ir0', 0), "o1", 20),
            (('o0', 'ir0', 0), "o2", -50)
        ],
        "static": [
            (('o1', 'ir1', 0), "o3", -30),
            (('o2', 'ir2', 0), "o0", -25)
        ]
    },
    # Scenario 6: Convergence to same destination
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o1', 'ir1', 0), "o2", 0),
            (('o3', 'ir3', 0), "o2", -40)
        ],
        "static": [
            (('o2', 'ir2', 0), "o1", -30),
            (('o0', 'ir0', 0), "o3", -50)
        ]
    },
    # Scenario 7: Minimum conflict scenario
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o2', 'ir2', 0), "o0", 0),
            (('o1', 'ir1', 0), "o3", -40)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -35),
            (('o0', 'ir0', 0), "o1", -60)
        ]
    },
    # Scenario 8: Rush hour challenge
    {
        "agents": [
            (('o3', 'ir3', 0), "o1", 0),
            (('o0', 'ir0', 0), "o2", 0),
            (('o1', 'ir1', 0), "o3", -20)
        ],
        "static": [
            (('o2', 'ir2', 0), "o0", -15),
            (('o3', 'ir3', 0), "o2", -45)
        ]
    },
    # Scenario 13: Complex intersection
    {
        "agents": [
            (('o0', 'ir0', 0), "o1", 0),
            (('o3', 'ir3', 0), "o0", 0),
            (('o1', 'ir1', 0), "o2", -25)
        ],
        "static": [
            (('o2', 'ir2', 0), "o3", -30),
            (('o0', 'ir0', 0), "o3", -60)
        ]
    },
    # Scenario 14: Parallel lanes
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o0', 'ir0', 0), "o2", -30),
            (('o2', 'ir2', 0), "o0", -60)
        ],
        "static": [
            (('o1', 'ir1', 0), "o3", -55),
            (('o3', 'ir3', 0), "o1", -65)
        ]
    },
    # Scenario 15: Cross traffic
    {
        "agents": [
            (('o1', 'ir1', 0), "o3", 0),
            (('o2', 'ir2', 0), "o0", 0),
            (('o0', 'ir0', 0), "o2", -30)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -25),
            (('o1', 'ir1', 0), "o0", -50)
        ]
    },
    # Scenario 16: Left turn conflict
    {
        "agents": [
            (('o0', 'ir0', 0), "o3", 0),
            (('o1', 'ir1', 0), "o0", 0),
            (('o2', 'ir2', 0), "o1", -35)
        ],
        "static": [
            (('o3', 'ir3', 0), "o2", -40),
            (('o0', 'ir0', 0), "o1", -55)
        ]
    },
    # Scenario 17: Right turn priority
    {
        "agents": [
            (('o0', 'ir0', 0), "o1", 0),
            (('o2', 'ir2', 0), "o3", 0),
            (('o1', 'ir1', 0), "o2", -50)
        ],
        "static": [
            (('o3', 'ir3', 0), "o0", -55),
            (('o0', 'ir0', 0), "o2", -55)
        ]
    },
    # Scenario 18: Staggered timing
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o1', 'ir1', 0), "o3", -40),
            (('o2', 'ir2', 0), "o0", -50)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -35),
            (('o0', 'ir0', 0), "o1", -20)
        ]
    },
    # Scenario 20: Opposite directions
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o2', 'ir2', 0), "o0", 0),
            (('o1', 'ir1', 0), "o3", -30)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -35),
            (('o0', 'ir0', 0), "o1", -10)
        ]
    },
    # Scenario 21: Diagonal crossing
    {
        "agents": [
            (('o0', 'ir0', 0), "o3", 0),
            (('o1', 'ir1', 0), "o2", 0),
            (('o2', 'ir2', 0), "o1", -25)
        ],
        "static": [
            (('o3', 'ir3', 0), "o0", -30),
            (('o0', 'ir0', 0), "o2", -35)
        ]
    },
    # Scenario 22: Sequential turns
    {
        "agents": [
            (('o0', 'ir0', 0), "o1", 0),
            (('o1', 'ir1', 0), "o2", 0),
            (('o2', 'ir2', 0), "o3", -50)
        ],
        "static": [
            (('o3', 'ir3', 0), "o0", -55),
            (('o0', 'ir0', 0), "o3", -65)
        ]
    },
    # Scenario 23: Wide spacing
    {
        "agents": [
            (('o0', 'ir0', 0), "o2", 0),
            (('o1', 'ir1', 0), "o3", 50),
            (('o2', 'ir2', 0), "o0", -60)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -45),
            (('o0', 'ir0', 0), "o1", -75)
        ]
    },
    # Scenario 24: Mixed patterns
    {
        "agents": [
            (('o2', 'ir2', 0), "o3", 0),
            (('o3', 'ir3', 0), "o2", 15),
            (('o0', 'ir0', 0), "o1", -30)
        ],
        "static": [
            (('o1', 'ir1', 0), "o0", -35),
            (('o2', 'ir2', 0), "o0", -50)
        ]
    }
]




base_complete_scenarios_2_cars = [
            # Scenario 0: Heavy north-south flow
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o3", 0)  # Agent 2: East to West
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -50),  # Static: North to South (behind)
                    (('o2', 'ir2', 0), "o0", -35),  # Static: South to North
                    (('o0', 'ir0', 0), "o1", -70)  # Static: North to East
                ]
            },
            # Scenario 1: All cars turning right
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o3", -25),  # Static: South to West (right)
                    (('o3', 'ir3', 0), "o0", -35),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o1", -15)  # Static: North to East (right)
                ]
            },
            # Scenario 2: Crossing patterns
            {
                "agents": [
                    (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left)
                    (('o2', 'ir2', 0), "o1", 0)  # Agent 2: South to East (left)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o0", -20),  # Static: East to North (left)
                    (('o3', 'ir3', 0), "o2", -30),  # Static: West to South (left)
                    (('o0', 'ir0', 0), "o2", -40)  # Static: North to South (straight)
                ]
            },
            # Scenario 3: East-west corridor
            {
                "agents": [
                    (('o1', 'ir1', 0), "o3", 0),  # Agent 1: East to West
                    (('o3', 'ir3', 0), "o1", 0)  # Agent 2: West to East
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -30),  # Static: East to West (behind)
                    (('o3', 'ir3', 0), "o1", -25),  # Static: West to East (behind)
                    (('o2', 'ir2', 0), "o1", -35)  # Static: South to East (left)
                ]
            },
            # Scenario 4: Complex multi-direction
            {
                "agents": [
                    (('o2', 'ir2', 0), "o0", 0),  # Agent 1: South to North
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o0', 'ir0', 0), "o3", -45),  # Static: North to West (left, far behind)
                    (('o1', 'ir1', 0), "o2", -30),  # Static: East to South (right)
                    (('o2', 'ir2', 0), "o0", -20)  # Static: South to North (straight)
                ]
            },
            # Scenario 5: Same origin dispersal
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o0', 'ir0', 0), "o1", 20)  # Agent 2: North to East (offset forward)
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -50),  # Static: North to South (behind)
                    (('o1', 'ir1', 0), "o3", -30),  # Static: East to West
                    (('o2', 'ir2', 0), "o0", -25)  # Static: South to North
                ]
            },
            # Scenario 6: Convergence to same destination
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o3', 'ir3', 0), "o2", -40),  # Static: West to South (converge)
                    (('o2', 'ir2', 0), "o1", -30),  # Static: South to East
                    (('o0', 'ir0', 0), "o3", -50)  # Static: North to West
                ]
            },
            # Scenario 7: Minimum conflict scenario
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South (straight)
                    (('o2', 'ir2', 0), "o0", 0)  # Agent 2: South to North (straight)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -40),  # Static: East to West (parallel)
                    (('o3', 'ir3', 0), "o1", -35),  # Static: West to East (parallel)
                    (('o0', 'ir0', 0), "o1", -60)  # Static: North to East (turn)
                ]
            },
            # Scenario 8: Rush hour challenge
            {
                "agents": [
                    (('o3', 'ir3', 0), "o1", 0),  # Agent 1: West to East
                    (('o0', 'ir0', 0), "o2", 0)  # Agent 2: North to South
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -20),  # Static: East to West (opposite)
                    (('o2', 'ir2', 0), "o0", -15),  # Static: South to North (opposite)
                    (('o3', 'ir3', 0), "o2", -45)  # Static: West to South (turn)
                ]
            },
            # # Scenario 9: Mixed speed challenge
            # {
            #     "agents": [
            #         (('o1', 'ir1', 0), "o0", 0),  # Agent 1: East to North (left)
            #         (('o2', 'ir2', 0), "o3", 0)  # Agent 2: South to West (left)
            #     ],
            #     "static": [
            #         (('o0', 'ir0', 0), "o1", -25),  # Static: North to East (right)
            #         (('o3', 'ir3', 0), "o2", -30),  # Static: West to South (right)
            #         (('o1', 'ir1', 0), "o2", -50)  # Static: East to South (right)
            #     ]
            # },
            # Scenario 10: Emergency scenario
            # {
            #     "agents": [
            #         (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left turn)
            #         (('o1', 'ir1', 0), "o0", 30)  # Agent 2: East to North (ahead)
            #     ],
            #     "static": [
            #         (('o2', 'ir2', 0), "o1", -35),  # Static: South to East
            #         (('o3', 'ir3', 0), "o0", -40),  # Static: West to North
            #         (('o0', 'ir0', 0), "o2", -55)  # Static: North to South
            #     ]
            # },
            # Scenario 11: T-junction behavior
            # {
            #     "agents": [
            #         (('o2', 'ir2', 0), "o1", 0),  # Agent 1: South to East (left)
            #         (('o2', 'ir2', 0), "o3", 20)  # Agent 2: South to West (right, ahead)
            #     ],
            #     "static": [
            #         (('o1', 'ir1', 0), "o0", -30),  # Static: East to North
            #         (('o3', 'ir3', 0), "o2", -25),  # Static: West to South
            #         (('o0', 'ir0', 0), "o1", -45)  # Static: North to East
            #     ]
            # },
            # Scenario 12: Highway merge simulation
            # {
            #     "agents": [
            #         (('o3', 'ir3', 0), "o2", 0),  # Agent 1: West to South (left)
            #         (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
            #     ],
            #     "static": [
            #         (('o0', 'ir0', 0), "o2", -20),  # Static: North to South (merge point)
            #         (('o2', 'ir2', 0), "o0", -35),  # Static: South to North
            #         (('o3', 'ir3', 0), "o1", -50)  # Static: West to East
            #     ]
            # },
            # Scenario 13: Complex intersection
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o3', 'ir3', 0), "o0", 0)  # Agent 2: West to North (right)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o2", -25),  # Static: East to South (right)
                    (('o2', 'ir2', 0), "o3", -30),  # Static: South to West (right)
                    (('o0', 'ir0', 0), "o3", -60)  # Static: North to West (left)
                ]
            },
            # Scenario 14: Parallel lanes
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o0', 'ir0', 0), "o2", -30)  # Agent 2: North to South (behind)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o0", -60),  # Static: South to North (parallel)
                    (('o1', 'ir1', 0), "o3", -55),  # Static: East to West (crossing)
                    (('o3', 'ir3', 0), "o1", -65)  # Static: West to East (crossing)
                ]
            },
            # Scenario 15: Cross traffic
            {
                "agents": [
                    (('o1', 'ir1', 0), "o3", 0),  # Agent 1: East to West
                    (('o2', 'ir2', 0), "o0", 0)  # Agent 2: South to North
                ],
                "static": [
                    (('o0', 'ir0', 0), "o2", -30),  # Static: North to South (crossing)
                    (('o3', 'ir3', 0), "o1", -25),  # Static: West to East (opposite)
                    (('o1', 'ir1', 0), "o0", -50)  # Static: East to North (turn)
                ]
            },
            # Scenario 16: Left turn conflict
            {
                "agents": [
                    (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left)
                    (('o1', 'ir1', 0), "o0", 0)  # Agent 2: East to North (left)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o1", -35),  # Static: South to East (left)
                    (('o3', 'ir3', 0), "o2", -40),  # Static: West to South (left)
                    (('o0', 'ir0', 0), "o1", -55)  # Static: North to East (right)
                ]
            },
            # Scenario 17: Right turn priority
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o2', 'ir2', 0), "o3", 0)  # Agent 2: South to West (right)
                ],
                "static": [
                    (('o1', 'ir1', 0), "o2", -50),  # Static: East to South (right)
                    (('o3', 'ir3', 0), "o0", -55),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o2", -55)  # Static: North to South (straight)
                ]
            },
            # Scenario 18: Staggered timing
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o3", -40)  # Agent 2: East to West (behind)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o0", -50),  # Static: South to North (behind)
                    (('o3', 'ir3', 0), "o1", -35),  # Static: West to East (behind)
                    (('o0', 'ir0', 0), "o1", -20)  # Static: North to East (behind)
                ]
            },
            # # Scenario 19: Dense traffic
            # {
            #     "agents": [
            #         (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
            #         (('o1', 'ir1', 0), "o2", -10)  # Agent 2: East to South (close behind)
            #     ],
            #     "static": [
            #         (('o3', 'ir3', 0), "o2", -20),  # Static: West to South (converge)
            #         (('o2', 'ir2', 0), "o0", -25),  # Static: South to North (opposite)
            #         (('o0', 'ir0', 0), "o3", -30)  # Static: North to West (turn)
            #     ]
            # },
            # # Scenario 20: Opposite directions
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o2', 'ir2', 0), "o0", 0)  # Agent 2: South to North
                ],
                "static": [
                    (('o1', 'ir1', 0), "o3", -30),  # Static: East to West (parallel)
                    (('o3', 'ir3', 0), "o1", -35),  # Static: West to East (parallel)
                    (('o0', 'ir0', 0), "o1", -10)  # Static: North to East (turn)
                ]
            },
            # Scenario 21: Diagonal crossing
            {
                "agents": [
                    (('o0', 'ir0', 0), "o3", 0),  # Agent 1: North to West (left)
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o1", -25),  # Static: South to East (left)
                    (('o3', 'ir3', 0), "o0", -30),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o2", -35)  # Static: North to South (straight)
                ]
            },
            # Scenario 22: Sequential turns
            {
                "agents": [
                    (('o0', 'ir0', 0), "o1", 0),  # Agent 1: North to East (right)
                    (('o1', 'ir1', 0), "o2", 0)  # Agent 2: East to South (right)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o3", -50),  # Static: South to West (right)
                    (('o3', 'ir3', 0), "o0", -55),  # Static: West to North (right)
                    (('o0', 'ir0', 0), "o3", -65)  # Static: North to West (left)
                ]
            },
            # Scenario 23: Wide spacing
            {
                "agents": [
                    (('o0', 'ir0', 0), "o2", 0),  # Agent 1: North to South
                    (('o1', 'ir1', 0), "o3", 50)  # Agent 2: East to West (far ahead)
                ],
                "static": [
                    (('o2', 'ir2', 0), "o0", -60),  # Static: South to North (far behind)
                    (('o3', 'ir3', 0), "o1", -45),  # Static: West to East (behind)
                    (('o0', 'ir0', 0), "o1", -75)  # Static: North to East (far behind)
                ]
            },
            # Scenario 24: Mixed patterns
            {
                "agents": [
                    (('o2', 'ir2', 0), "o3", 0),  # Agent 1: South to West (right)
                    (('o3', 'ir3', 0), "o2", 15)  # Agent 2: West to South (left, ahead)
                ],
                "static": [
                    (('o0', 'ir0', 0), "o1", -30),  # Static: North to East (right)
                    (('o1', 'ir1', 0), "o0", -35),  # Static: East to North (left)
                    (('o2', 'ir2', 0), "o0", -50)  # Static: South to North (straight)
                ]
            }
        ]
