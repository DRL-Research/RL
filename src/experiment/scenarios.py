



# base_complete_scenarios_4_cars = [
#
#     # Scenario 1: All cars turning right
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o1", 0),
#             (('o1', 'ir1', 0), "o2", 0),
#             (('o2', 'ir2', 0), "o3", -25),
#             (('o3', 'ir3', 0), "o0", -35)
#         ],
#         "static": [
#             (('o0', 'ir0', 0), "o1", -15)
#         ]
#     },
#
#     # Scenario 3: East-west corridor
#     {
#         "agents": [
#             (('o1', 'ir1', 0), "o3", 0),
#             (('o3', 'ir3', 0), "o1", 0),
#             (('o1', 'ir1', 0), "o3", -30),
#             (('o3', 'ir3', 0), "o1", -25),
#         ],
#         "static": [
#             (('o2', 'ir2', 0), "o1", -35)
#         ]
#     },
#     # Scenario 4: Complex multi-direction
#     {
#         "agents": [
#             (('o2', 'ir2', 0), "o0", 0),
#             (('o1', 'ir1', 0), "o2", 0),
#             (('o0', 'ir0', 0), "o3", -45),
#             (('o1', 'ir1', 0), "o2", -30),
#         ],
#         "static": [
#
#             (('o2', 'ir2', 0), "o0", -20)
#         ]
#     },
#     # Scenario 5: Same origin dispersal
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o0', 'ir0', 0), "o1", 20),
#             (('o0', 'ir0', 0), "o2", -50),
#             (('o1', 'ir1', 0), "o3", -30),
#         ],
#         "static": [
#
#             (('o2', 'ir2', 0), "o0", -25)
#         ]
#     },
#     # Scenario 6: Convergence to same destination
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o1', 'ir1', 0), "o2", 0),
#             (('o3', 'ir3', 0), "o2", -40),
#             (('o2', 'ir2', 0), "o1", -30),
#         ],
#         "static": [
#
#             (('o0', 'ir0', 0), "o3", -50)
#         ]
#     },
#     # Scenario 7: Minimum conflict scenario
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o2', 'ir2', 0), "o0", 0),
#             (('o1', 'ir1', 0), "o3", -40),
#             (('o3', 'ir3', 0), "o1", -35),
#         ],
#         "static": [
#             (('o0', 'ir0', 0), "o1", -60)
#         ]
#     },
#     # Scenario 8: Rush hour challenge
#     {
#         "agents": [
#             (('o3', 'ir3', 0), "o1", 0),
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o1', 'ir1', 0), "o3", -20),
#             (('o2', 'ir2', 0), "o0", -15),
#         ],
#         "static": [
#             (('o3', 'ir3', 0), "o2", -45)
#         ]
#     },
#     # Scenario 13: Complex intersection
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o1", 0),
#             (('o3', 'ir3', 0), "o0", 0),
#             (('o1', 'ir1', 0), "o2", -25),
#             (('o2', 'ir2', 0), "o3", -30),
#         ],
#         "static": [
#             (('o0', 'ir0', 0), "o3", -60)
#         ]
#     },
#     # Scenario 14: Parallel lanes
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o0', 'ir0', 0), "o2", -30),
#             (('o2', 'ir2', 0), "o0", -60),
#             (('o1', 'ir1', 0), "o3", -55),
#         ],
#         "static": [
#             (('o3', 'ir3', 0), "o1", -65)
#         ]
#     },
#     # Scenario 15: Cross traffic
#     {
#         "agents": [
#             (('o1', 'ir1', 0), "o3", 0),
#             (('o2', 'ir2', 0), "o0", 0),
#             (('o0', 'ir0', 0), "o2", -30),
#             (('o3', 'ir3', 0), "o1", -25),
#         ],
#         "static": [
#             (('o1', 'ir1', 0), "o0", -50)
#         ]
#     },
#     # Scenario 16: Left turn conflict
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o3", 0),
#             (('o1', 'ir1', 0), "o0", 0),
#             (('o2', 'ir2', 0), "o1", -35),
#             (('o3', 'ir3', 0), "o2", -40),
#         ],
#         "static": [
#             (('o0', 'ir0', 0), "o1", -55)
#         ]
#     },
#     # Scenario 17: Right turn priority
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o1", 0),
#             (('o2', 'ir2', 0), "o3", 0),
#             (('o1', 'ir1', 0), "o2", -50),
#             (('o3', 'ir3', 0), "o0", -55),
#         ],
#         "static": [
#             (('o0', 'ir0', 0), "o2", -55)
#         ]
#     },
#     # Scenario 18: Staggered timing
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o1', 'ir1', 0), "o3", -40),
#             (('o2', 'ir2', 0), "o0", -50),
#             (('o3', 'ir3', 0), "o1", -35),
#         ],
#         "static": [
#
#             (('o0', 'ir0', 0), "o1", -20)
#         ]
#     },
#     # Scenario 20: Opposite directions
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o2', 'ir2', 0), "o0", 0),
#             (('o1', 'ir1', 0), "o3", -30),
#             (('o3', 'ir3', 0), "o1", -35),
#         ],
#         "static": [
#
#             (('o0', 'ir0', 0), "o1", -10)
#         ]
#     },
#     # Scenario 21: Diagonal crossing
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o3", 0),
#             (('o1', 'ir1', 0), "o2", 0),
#             (('o2', 'ir2', 0), "o1", -25),
#             (('o3', 'ir3', 0), "o0", -30),
#         ],
#         "static": [
#
#             (('o0', 'ir0', 0), "o2", -35)
#         ]
#     },
#     # Scenario 22: Sequential turns
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o1", 0),
#             (('o1', 'ir1', 0), "o2", 0),
#             (('o2', 'ir2', 0), "o3", -50),
#             (('o3', 'ir3', 0), "o0", -55),
#         ],
#         "static": [
#
#             (('o0', 'ir0', 0), "o3", -65)
#         ]
#     },
#     # Scenario 23: Wide spacing
#     {
#         "agents": [
#             (('o0', 'ir0', 0), "o2", 0),
#             (('o1', 'ir1', 0), "o3", 50),
#             (('o2', 'ir2', 0), "o0", -60),
#             (('o3', 'ir3', 0), "o1", -45),
#         ],
#         "static": [
#
#             (('o0', 'ir0', 0), "o1", -75)
#         ]
#     },
#     # Scenario 24: Mixed patterns
#     {
#         "agents": [
#             (('o2', 'ir2', 0), "o3", 0),
#             (('o3', 'ir3', 0), "o2", 15),
#             (('o0', 'ir0', 0), "o1", -30),
#             (('o1', 'ir1', 0), "o0", -35),
#         ],
#         "static": [
#             (('o2', 'ir2', 0), "o0", -50)
#         ]
#     }
# ]

base_complete_scenarios_3_cars = [

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
    # Scenario 2: Complex multi-direction
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
    # Scenario 3: Complex multi-direction
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
    # Scenario 4: Same origin dispersal
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
    # Scenario 5: Convergence to same destination
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
    # Scenario 6: Minimum conflict scenario
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
    # Scenario 7: Rush hour challenge
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
    # Scenario 8: Complex intersection
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
    # Scenario 9: Parallel lanes
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
    # Scenario 10: Cross traffic
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
    # Scenario 11: Left turn conflict
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
    # Scenario 12: Right turn priority
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
    # Scenario 13: Staggered timing
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
    # Scenario 14: Opposite directions
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
    # Scenario 15: Diagonal crossing
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
    # Scenario 16: Sequential turns
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
    # Scenario 17: Wide spacing
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
    # Scenario 18: Mixed patterns
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
