





# ──────────────────────────────────────────────────────────────────────────────
# 6-car scenarios (no static vehicles).
# Agents 0-2 → Local Master 1 | Agents 3-5 → Local Master 2
#
# Lane convention: o0=south, o1=west, o2=north, o3=east (approaches)
# offset: positive → closer to intersection (s = BASE_LONG(40) + offset)
# Same-lane pairs always have ≥25 unit gap to guarantee safe separation.
# ──────────────────────────────────────────────────────────────────────────────

# Scenario indices (0-99) that are structurally near-impossible to solve:
# even with a diverse mixed policy (some SLOW, some FAST, some RANDOM),
# these scenarios crashed ≥70% of the time — meaning the timing window is
# too tight for RL to reliably learn coordination.
# Empirically determined by audit_scenarios.py (April 2026).
#
# 1 truly unavoidable (≥93%): [6]
# 20 very hard (70-93%): [1,2,3,5,7,13,14,15,16,18,21,25,27,37,67,71,73,74,75,81]
EXCLUDED_SCENARIO_INDICES = frozenset([
    6,                                   # truly unavoidable (93%+)
    1, 2, 3, 5, 7,                       # base 0 (rots 1-3) & base 1 (rots 1,3)
    13, 14, 15,                          # base 3 (rots 1-3)
    16, 18,                              # base 4 (rots 0,2)
    21,                                  # base 5 (rot 1)
    25, 27,                              # base 6 (rots 1,3)
    37,                                  # base 9 (rot 1)
    67,                                  # base 16 (rot 3)
    71,                                  # base 17 (rot 3)
    73, 74, 75,                          # base 18 (rots 1-3)
    81,                                  # base 20 (rot 1)
])
# After exclusion: 79 scenarios remain for training.

# ── Held-out test scenarios (never seen during training) ──────────────────────
# 5 scenarios spread across different base configurations, all verified safe
# (not in EXCLUDED_SCENARIO_INDICES). Used for final evaluation after training.
HELD_OUT_SCENARIO_INDICES = frozenset([
    4,    # base 1,  rot 0
    11,   # base 2,  rot 3
    24,   # base 6,  rot 0
    50,   # base 12, rot 2
    88,   # base 22, rot 0
])

base_complete_scenarios_6_cars = [
    # S1: All-four cross + 2 right turns (staggered, safe)
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o0', 'ir0', 0), "o1", -30), (('o1', 'ir1', 0), "o2", -30),
    ], "static": []},
    # S2: All-four cross + 2 right turns (different pair)
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o2', 'ir2', 0), "o3", -30), (('o3', 'ir3', 0), "o0", -30),
    ], "static": []},
    # S3: All left turns + 2 straight (staggered)
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),   (('o2', 'ir2', 0), "o1", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o0', 'ir0', 0), "o2", -30), (('o2', 'ir2', 0), "o0", -30),
    ], "static": []},
    # S4: All right turns + 2 extra straight
    {"agents": [
        (('o0', 'ir0', 0), "o1", 0),   (('o1', 'ir1', 0), "o2", 0),   (('o2', 'ir2', 0), "o3", 0),
        (('o3', 'ir3', 0), "o0", 0),   (('o0', 'ir0', 0), "o2", -30), (('o3', 'ir3', 0), "o2", -30),
    ], "static": []},
    # S5: Mixed turns
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o0", 0),   (('o2', 'ir2', 0), "o3", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o1', 'ir1', 0), "o3", -30), (('o3', 'ir3', 0), "o2", -30),
    ], "static": []},
    # S6: Mixed left + straight extra
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o2', 'ir2', 0), "o1", 0),   (('o1', 'ir1', 0), "o3", 0),
        (('o3', 'ir3', 0), "o0", 0),   (('o0', 'ir0', 0), "o1", -30), (('o2', 'ir2', 0), "o0", -30),
    ], "static": []},
    # S7: Two cars per lane (front+back) on two lanes + two on separate lanes
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o0', 'ir0', 0), "o1", 25),  (('o2', 'ir2', 0), "o0", 0),
        (('o2', 'ir2', 0), "o3", 25),  (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
    ], "static": []},
    # S8: All-four cross + N/E stagger
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o1', 'ir1', 0), "o0", -25), (('o3', 'ir3', 0), "o2", -25),
    ], "static": []},
    # S9: Mixed straight+right + stagger
    {"agents": [
        (('o0', 'ir0', 0), "o1", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o3", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o0', 'ir0', 0), "o2", -25), (('o2', 'ir2', 0), "o0", -25),
    ], "static": []},
    # S10: Left+straight combos
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o1', 'ir1', 0), "o0", -30), (('o2', 'ir2', 0), "o1", -30),
    ], "static": []},
    # S11: Platooning pattern (two groups in convoy)
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o0', 'ir0', 0), "o3", -30), (('o2', 'ir2', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", -30), (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
    ], "static": []},
    # S12: E-W corridor + N-S corridor + extras
    {"agents": [
        (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),   (('o0', 'ir0', 0), "o2", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o1', 'ir1', 0), "o2", -25), (('o3', 'ir3', 0), "o0", -25),
    ], "static": []},
    # S13: Two convoys on perpendicular lanes
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o0', 'ir0', 0), "o3", 25),  (('o2', 'ir2', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", 25),  (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
    ], "static": []},
    # S14: Back-pairs on two approaches
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o0', 'ir0', 0), "o3", -25), (('o1', 'ir1', 0), "o3", 0),
        (('o1', 'ir1', 0), "o0", -25), (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o1", 0),
    ], "static": []},
    # S15: Back-pairs on N and E
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", -25), (('o3', 'ir3', 0), "o1", 0),   (('o3', 'ir3', 0), "o2", -25),
    ], "static": []},
    # S16: Left-turn focus
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o0', 'ir0', 0), "o2", -30), (('o1', 'ir1', 0), "o3", -30),
    ], "static": []},
    # S17: Right-turn focus + left stagger
    {"agents": [
        (('o0', 'ir0', 0), "o1", 0),   (('o1', 'ir1', 0), "o2", 0),   (('o2', 'ir2', 0), "o3", 0),
        (('o3', 'ir3', 0), "o0", 0),   (('o0', 'ir0', 0), "o3", -25), (('o2', 'ir2', 0), "o1", -25),
    ], "static": []},
    # S18: Three converging to north
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o2", 0),   (('o3', 'ir3', 0), "o2", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o2', 'ir2', 0), "o3", -30), (('o3', 'ir3', 0), "o1", -30),
    ], "static": []},
    # S19: Convoy on S+W, singles elsewhere
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o0', 'ir0', 0), "o1", -30),
        (('o1', 'ir1', 0), "o2", -30), (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o1", 0),
    ], "static": []},
    # S20: Criss-cross left+right
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),   (('o1', 'ir1', 0), "o0", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", -25), (('o3', 'ir3', 0), "o1", -25),
    ], "static": []},
    # S21: Mixed all directions + straight stagger
    {"agents": [
        (('o0', 'ir0', 0), "o1", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o1", 0),
        (('o3', 'ir3', 0), "o0", 0),   (('o0', 'ir0', 0), "o2", -30), (('o2', 'ir2', 0), "o0", -30),
    ], "static": []},
    # S22: All left turns + stagger
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o2", 0),   (('o2', 'ir2', 0), "o1", 0),
        (('o3', 'ir3', 0), "o0", 0),   (('o0', 'ir0', 0), "o1", -25), (('o2', 'ir2', 0), "o3", -25),
    ], "static": []},
    # S23: Close-front convoy (offset +25) + opposites
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o0', 'ir0', 0), "o2", 25),  (('o3', 'ir3', 0), "o1", 25),
    ], "static": []},
    # S24: Left+right mix + stagger on N+E
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),   (('o2', 'ir2', 0), "o1", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o0', 'ir0', 0), "o2", -30), (('o2', 'ir2', 0), "o0", -30),
    ], "static": []},
    # S25: Diverse mixed destinations
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o0", 0),   (('o2', 'ir2', 0), "o3", 0),
        (('o3', 'ir3', 0), "o1", 0),   (('o0', 'ir0', 0), "o1", -30), (('o2', 'ir2', 0), "o3", -30),
    ], "static": []},
]

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


# ── Forced-conflict scenarios (coordination-critical) ─────────────────────────
# These scenarios are DESIGNED to require master coordination:
#   - Multiple vehicles arrive at the intersection center SIMULTANEOUSLY (offset=0)
#   - Paths CROSS in the center (not parallel)
#   - Without one vehicle yielding (slowing down), collision is guaranteed
#   - Only a master that communicates "you go, you wait" can solve these reliably
#
# 20 base scenarios × 4 rotations = 80 total.
# Used alongside regular scenarios in unified training.
#
# Conflict types:
#   TYPE A: 4-way simultaneous cross (4 vehicles, offset=0, all crossing)
#   TYPE B: 3-way merge to same destination (3 vehicles targeting same exit)
#   TYPE C: Head-on + perpendicular (2 pairs of crossing paths, all offset=0)
#   TYPE D: 6-vehicle full conflict (all 6 at offset=0 with crossing paths)

conflict_base_scenarios = [
    # ── TYPE D: Full 6-vehicle conflict (hardest) ─────────────────────────────
    # D1: 4 cross + 2 left turns, ALL at offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),
    ], "static": []},
    # D2: All left turns, ALL at offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", 0),   (('o3', 'ir3', 0), "o2", 0),
        (('o0', 'ir0', 0), "o1", 0),   (('o2', 'ir2', 0), "o3", 0),
    ], "static": []},
    # D3: Mixed cross + left, ALL at offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o0", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o2", 0),
    ], "static": []},
    # D4: 3 straight + 3 left, ALL at offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o0', 'ir0', 0), "o3", 0),
        (('o1', 'ir1', 0), "o0", 0),   (('o2', 'ir2', 0), "o1", 0),
    ], "static": []},
    # D5: All crossing to opposite + 2 perpendicular, ALL at offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o0', 'ir0', 0), "o1", 0),   (('o2', 'ir2', 0), "o3", 0),
    ], "static": []},

    # ── TYPE A: 4-way simultaneous + 2 staggered ─────────────────────────────
    # A1: 4-way cross at 0, 2 close behind at -10 (almost simultaneous)
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o0', 'ir0', 0), "o3", -10),  (('o2', 'ir2', 0), "o1", -10),
    ], "static": []},
    # A2: 4-way cross at 0, 2 close at -15
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o1', 'ir1', 0), "o0", -10),  (('o3', 'ir3', 0), "o2", -10),
    ], "static": []},
    # A3: 4 left turns at 0, 2 straight close behind
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", 0),   (('o3', 'ir3', 0), "o2", 0),
        (('o0', 'ir0', 0), "o2", -10),  (('o2', 'ir2', 0), "o0", -10),
    ], "static": []},

    # ── TYPE B: 3-way merge (3 vehicles → same destination) ──────────────────
    # B1: 3 vehicles all heading to o2 (north), + 3 heading to o0 (south)
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o2", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o1', 'ir1', 0), "o0", 0),   (('o3', 'ir3', 0), "o0", 0),
    ], "static": []},
    # B2: 3 vehicles → o3 (east), 3 → o1 (west)
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o3", 0),
        (('o2', 'ir2', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o0', 'ir0', 0), "o1", 0),   (('o2', 'ir2', 0), "o1", 0),
    ], "static": []},
    # B3: 4 vehicles → o2, 2 → o0
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o2", 0),
        (('o3', 'ir3', 0), "o2", 0),   (('o2', 'ir2', 0), "o2", 0),
        (('o2', 'ir2', 0), "o0", 0),   (('o3', 'ir3', 0), "o0", 0),
    ], "static": []},

    # ── TYPE C: Head-on pairs + perpendicular ─────────────────────────────────
    # C1: N↔S head-on + E↔W head-on, all at offset=0, + 2 left turns
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o0', 'ir0', 0), "o3", -5),   (('o1', 'ir1', 0), "o0", -5),
    ], "static": []},
    # C2: Same but left turns at offset=0 too
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o1', 'ir1', 0), "o3", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),
    ], "static": []},
    # C3: Two perpendicular pairs + 2 converging left
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o1', 'ir1', 0), "o0", 0),   (('o3', 'ir3', 0), "o2", 0),
        (('o0', 'ir0', 0), "o1", -5),   (('o2', 'ir2', 0), "o3", -5),
    ], "static": []},

    # ── TYPE D continued: Maximum conflict density ────────────────────────────
    # D6: All right turns at 0 + 2 straight at 0
    {"agents": [
        (('o0', 'ir0', 0), "o1", 0),   (('o1', 'ir1', 0), "o2", 0),
        (('o2', 'ir2', 0), "o3", 0),   (('o3', 'ir3', 0), "o0", 0),
        (('o0', 'ir0', 0), "o2", 0),   (('o2', 'ir2', 0), "o0", 0),
    ], "static": []},
    # D7: 3 from south + 3 from north, all at 0, crossing paths
    {"agents": [
        (('o0', 'ir0', 0), "o2", 0),   (('o0', 'ir0', 0), "o1", 0),
        (('o0', 'ir0', 0), "o3", 0),   (('o2', 'ir2', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", 0),   (('o2', 'ir2', 0), "o3", 0),
    ], "static": []},
    # D8: 3 from east + 3 from west, all at 0, crossing paths
    {"agents": [
        (('o1', 'ir1', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),
        (('o1', 'ir1', 0), "o2", 0),   (('o3', 'ir3', 0), "o1", 0),
        (('o3', 'ir3', 0), "o0", 0),   (('o3', 'ir3', 0), "o2", 0),
    ], "static": []},
    # D9: All 6 from 4 directions targeting perpendicular exits
    {"agents": [
        (('o0', 'ir0', 0), "o1", 0),   (('o1', 'ir1', 0), "o2", 0),
        (('o2', 'ir2', 0), "o3", 0),   (('o3', 'ir3', 0), "o0", 0),
        (('o0', 'ir0', 0), "o3", 0),   (('o2', 'ir2', 0), "o1", 0),
    ], "static": []},
    # D10: Maximum chaos — 6 vehicles, all different crossing paths
    {"agents": [
        (('o0', 'ir0', 0), "o3", 0),   (('o1', 'ir1', 0), "o0", 0),
        (('o2', 'ir2', 0), "o1", 0),   (('o3', 'ir3', 0), "o2", 0),
        (('o0', 'ir0', 0), "o2", 0),   (('o1', 'ir1', 0), "o3", 0),
    ], "static": []},
]

# Held-out conflict scenarios (5 out of 80 total = 20 base × 4 rot)
CONFLICT_HELD_OUT_INDICES = frozenset([3, 22, 41, 58, 75])

# ── Roundabout held-out (5 scenarios never seen during training) ──────────────
# 60 total (15 base × 4 rotations).  Spread: one per 12-scenario band.
ROUNDABOUT_HELD_OUT_INDICES = frozenset([2, 17, 32, 47, 55])

# ── Double-intersection held-out (5 scenarios never seen during training) ─────
# 20 total scenarios.
DOUBLE_INTERSECTION_HELD_OUT_INDICES = frozenset([1, 6, 11, 16, 19])

# ── Double-intersection excluded (too hard / connector deadlock risk) ──────────
# S10 (index 9): full cross-swap — all 6 vehicles cross the A↔B connector
# simultaneously.  Creates systematic connector deadlock before the model can
# learn basic intersection navigation.
DOUBLE_INTERSECTION_EXCLUDED_INDICES = frozenset([9])

# ── Roundabout scenarios ───────────────────────────────────────────────────────
# Roundabout ring topology (CCW flow):
#   ir0 → ir3 → ir2 → ir1 → ir0   (each arc = 1/4 ring)
# Exit connectors: ir{i} → il{i} → o{i}
# Approach spurs:  o{i} → ir{i}   (100 m, vehicles start at longitudinal 40 = 60 m from ring)
#
# Shortest paths from o{i}:
#   o0→o3 = 1 arc   o0→o2 = 2 arcs   o0→o1 = 3 arcs
#   o1→o0 = 1 arc   o1→o3 = 2 arcs   o1→o2 = 3 arcs
#   o2→o1 = 1 arc   o2→o0 = 2 arcs   o2→o3 = 3 arcs
#   o3→o2 = 1 arc   o3→o1 = 2 arcs   o3→o0 = 3 arcs
#
# Design rules for safe roundabout scenarios:
#   1. At most 2 vehicles with off >= 0 (reach ring simultaneously)
#   2. Others at off = -25 / -50 / -75 to stagger arrivals
#   3. No U-turns (start and destination must differ)
#   4. Two-car same-approach convoys need >= 25 m gap
#
# 15 handcrafted base scenarios (+ 50 appended conservative bases in extra_solvable_scenarios)
roundabout_base_scenarios = [
    # S1: 2 active on opposite approaches, 4 staggered
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc
        (('o2', 'ir2', 0), "o1",   0),   # LM1: 1 arc  (opposite, non-conflicting)
        (('o1', 'ir1', 0), "o0",  -30),  # LM1: 1 arc
        (('o3', 'ir3', 0), "o2",  -30),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o2",  -60),  # LM2: 2 arcs
        (('o2', 'ir2', 0), "o0",  -60),  # LM2: 2 arcs
    ], "static": []},

    # S2: 1 active, 5 staggered — minimal ring pressure
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # LM1: 2 arcs — solo active
        (('o1', 'ir1', 0), "o3",  -20),  # LM1: 2 arcs
        (('o2', 'ir2', 0), "o0",  -40),  # LM1: 2 arcs
        (('o3', 'ir3', 0), "o2",  -20),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o3",  -50),  # LM2: 1 arc
        (('o1', 'ir1', 0), "o0",  -70),  # LM2: 1 arc
    ], "static": []},

    # S3: Convoy on o0, singles staggered on other approaches
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # LM1: front of convoy
        (('o0', 'ir0', 0), "o3",  -25),  # LM1: back of convoy  (25 m gap)
        (('o2', 'ir2', 0), "o1",  -30),  # LM1: single
        (('o3', 'ir3', 0), "o2",  -30),  # LM2: single
        (('o1', 'ir1', 0), "o3",  -55),  # LM2: single
        (('o2', 'ir2', 0), "o0",  -60),  # LM2: single
    ], "static": []},

    # S4: All 4 approaches, purely sequential (15 m stagger each)
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc — first
        (('o1', 'ir1', 0), "o0",  -15),  # LM1: 1 arc
        (('o2', 'ir2', 0), "o1",  -30),  # LM1: 1 arc
        (('o3', 'ir3', 0), "o2",  -45),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o2",  -60),  # LM2: 2 arcs
        (('o2', 'ir2', 0), "o0",  -75),  # LM2: 2 arcs
    ], "static": []},

    # S5: Long ring paths (2-3 arcs) with good staggering
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # LM1: 2 arcs
        (('o1', 'ir1', 0), "o3",   0),   # LM1: 2 arcs  (non-conflicting pair)
        (('o2', 'ir2', 0), "o0",  -35),  # LM1: 2 arcs
        (('o3', 'ir3', 0), "o1",  -35),  # LM2: 2 arcs
        (('o0', 'ir0', 0), "o1",  -65),  # LM2: 3 arcs
        (('o2', 'ir2', 0), "o3",  -65),  # LM2: 3 arcs
    ], "static": []},

    # S6: Two convoys from two opposite approaches
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc
        (('o0', 'ir0', 0), "o2",  -25),  # LM1: 2 arcs
        (('o2', 'ir2', 0), "o1",   0),   # LM1: 1 arc  (opposite — no ring conflict)
        (('o2', 'ir2', 0), "o0",  -25),  # LM2: 2 arcs
        (('o1', 'ir1', 0), "o0",  -55),  # LM2: 1 arc
        (('o3', 'ir3', 0), "o2",  -55),  # LM2: 1 arc
    ], "static": []},

    # S7: Staggered from 3 approaches, cross-flow
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # LM1: 2 arcs
        (('o1', 'ir1', 0), "o3",  -20),  # LM1: 2 arcs
        (('o3', 'ir3', 0), "o0",  -40),  # LM1: 3 arcs
        (('o2', 'ir2', 0), "o0",  -15),  # LM2: 2 arcs
        (('o0', 'ir0', 0), "o3",  -45),  # LM2: 1 arc
        (('o1', 'ir1', 0), "o0",  -65),  # LM2: 1 arc
    ], "static": []},

    # S8: All short paths (1 arc), spread over ~75 m
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc
        (('o1', 'ir1', 0), "o0",  -25),  # LM1: 1 arc
        (('o2', 'ir2', 0), "o1",  -50),  # LM1: 1 arc
        (('o3', 'ir3', 0), "o2",  -75),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o3",  -35),  # LM2: 1 arc  (same approach, gap 35)
        (('o2', 'ir2', 0), "o1",  -70),  # LM2: 1 arc
    ], "static": []},

    # S9: Mixed — short and medium, natural flow pattern
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc
        (('o3', 'ir3', 0), "o2",   0),   # LM1: 1 arc  (non-conflicting)
        (('o1', 'ir1', 0), "o3",  -30),  # LM1: 2 arcs
        (('o2', 'ir2', 0), "o0",  -20),  # LM2: 2 arcs
        (('o0', 'ir0', 0), "o2",  -55),  # LM2: 2 arcs
        (('o3', 'ir3', 0), "o1",  -55),  # LM2: 2 arcs
    ], "static": []},

    # S10: Two heavily-loaded approaches, two light
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # LM1: 2 arcs
        (('o0', 'ir0', 0), "o3",  -30),  # LM1: 1 arc  (convoy, 30 m gap)
        (('o1', 'ir1', 0), "o0",   0),   # LM1: 1 arc  (non-conflicting with o0)
        (('o2', 'ir2', 0), "o0",  -40),  # LM2: 2 arcs
        (('o2', 'ir2', 0), "o1",  -65),  # LM2: 1 arc  (convoy, 25 m gap)
        (('o3', 'ir3', 0), "o1",  -40),  # LM2: 2 arcs
    ], "static": []},

    # S11: Single active + 5 evenly spaced
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc
        (('o1', 'ir1', 0), "o0",  -20),  # LM1: 1 arc
        (('o2', 'ir2', 0), "o1",  -40),  # LM1: 1 arc
        (('o3', 'ir3', 0), "o2",  -30),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o2",  -55),  # LM2: 2 arcs
        (('o1', 'ir1', 0), "o3",  -70),  # LM2: 2 arcs
    ], "static": []},

    # S12: Cross-flow (long paths), well staggered
    {"agents": [
        (('o0', 'ir0', 0), "o1",   0),   # LM1: 3 arcs
        (('o2', 'ir2', 0), "o3",   0),   # LM1: 3 arcs  (non-conflicting)
        (('o1', 'ir1', 0), "o2",  -35),  # LM1: 3 arcs
        (('o3', 'ir3', 0), "o0",  -35),  # LM2: 3 arcs
        (('o0', 'ir0', 0), "o2",  -65),  # LM2: 2 arcs
        (('o2', 'ir2', 0), "o0",  -65),  # LM2: 2 arcs
    ], "static": []},

    # S13: 3 approaches (not 4), staggered
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # LM1: 2 arcs
        (('o1', 'ir1', 0), "o0",   0),   # LM1: 1 arc
        (('o0', 'ir0', 0), "o3",  -35),  # LM1: 1 arc  (same approach, 35 m gap)
        (('o3', 'ir3', 0), "o1",  -20),  # LM2: 2 arcs
        (('o1', 'ir1', 0), "o3",  -45),  # LM2: 2 arcs
        (('o3', 'ir3', 0), "o2",  -65),  # LM2: 1 arc
    ], "static": []},

    # S14: Mixed ring-path lengths, progressive stagger
    {"agents": [
        (('o1', 'ir1', 0), "o2",   0),   # LM1: 3 arcs
        (('o0', 'ir0', 0), "o3",  -10),  # LM1: 1 arc
        (('o2', 'ir2', 0), "o0",  -25),  # LM1: 2 arcs
        (('o3', 'ir3', 0), "o2",  -15),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o2",  -50),  # LM2: 2 arcs
        (('o2', 'ir2', 0), "o1",  -55),  # LM2: 1 arc
    ], "static": []},

    # S15: Natural roundabout flow — sequential single-file
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # LM1: 1 arc
        (('o3', 'ir3', 0), "o2",  -15),  # LM1: 1 arc
        (('o2', 'ir2', 0), "o1",  -30),  # LM1: 1 arc
        (('o1', 'ir1', 0), "o0",  -45),  # LM2: 1 arc
        (('o0', 'ir0', 0), "o3",  -60),  # LM2: 1 arc
        (('o3', 'ir3', 0), "o2",  -75),  # LM2: 1 arc
    ], "static": []},
]


# ── Roundabout CONFLICT scenarios (coordination-critical) ────────────────────
# All vehicles enter the ring simultaneously (offset=0 or near-0).
# Without yielding, merging vehicles collide on the shared ring arc.
# The master must coordinate who enters first and who waits.
#
# 10 base × 4 rotations = 40 conflict scenarios for roundabout.
roundabout_conflict_base_scenarios = [
    # RC1: All 6 from 4 approaches at offset=0, long paths → ring jam
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o3",   0),   # 2 arcs
        (('o2', 'ir2', 0), "o0",   0),   # 2 arcs
        (('o3', 'ir3', 0), "o1",   0),   # 2 arcs
        (('o0', 'ir0', 0), "o1",   0),   # 3 arcs
        (('o2', 'ir2', 0), "o3",   0),   # 3 arcs
    ], "static": []},
    # RC2: All 6 from 4 approaches at offset=0, short paths → merge conflict
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # 1 arc
        (('o1', 'ir1', 0), "o0",   0),   # 1 arc
        (('o2', 'ir2', 0), "o1",   0),   # 1 arc
        (('o3', 'ir3', 0), "o2",   0),   # 1 arc
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o3",   0),   # 2 arcs
    ], "static": []},
    # RC3: 3 vehicles from adjacent approaches, all offset=0 → ring merge crash
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o3",   0),   # 2 arcs
        (('o2', 'ir2', 0), "o0",   0),   # 2 arcs
        (('o0', 'ir0', 0), "o3",   0),   # 1 arc
        (('o1', 'ir1', 0), "o0",   0),   # 1 arc
        (('o3', 'ir3', 0), "o0",   0),   # 3 arcs
    ], "static": []},
    # RC4: Convoy from same approach + simultaneous from others
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o0', 'ir0', 0), "o3",  -5),   # 1 arc (near-simultaneous convoy)
        (('o1', 'ir1', 0), "o3",   0),   # 2 arcs
        (('o2', 'ir2', 0), "o0",   0),   # 2 arcs
        (('o3', 'ir3', 0), "o1",   0),   # 2 arcs
        (('o3', 'ir3', 0), "o2",  -5),   # 1 arc
    ], "static": []},
    # RC5: All targeting same exit (o2) → funnel crash
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o2",   0),   # 3 arcs
        (('o3', 'ir3', 0), "o2",   0),   # 1 arc
        (('o0', 'ir0', 0), "o2",  -5),   # 2 arcs
        (('o2', 'ir2', 0), "o0",   0),   # 2 arcs (counter-flow)
        (('o2', 'ir2', 0), "o1",  -5),   # 1 arc
    ], "static": []},
    # RC6: Pairs from adjacent entries, crossing ring arcs
    {"agents": [
        (('o0', 'ir0', 0), "o1",   0),   # 3 arcs
        (('o1', 'ir1', 0), "o2",   0),   # 3 arcs
        (('o2', 'ir2', 0), "o3",   0),   # 3 arcs
        (('o3', 'ir3', 0), "o0",   0),   # 3 arcs
        (('o0', 'ir0', 0), "o3",   0),   # 1 arc
        (('o2', 'ir2', 0), "o1",   0),   # 1 arc
    ], "static": []},
    # RC7: 3 from o0, 3 from o2 — head-on ring fill
    {"agents": [
        (('o0', 'ir0', 0), "o3",   0),   # 1 arc
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o0', 'ir0', 0), "o1",   0),   # 3 arcs
        (('o2', 'ir2', 0), "o1",   0),   # 1 arc
        (('o2', 'ir2', 0), "o0",   0),   # 2 arcs
        (('o2', 'ir2', 0), "o3",   0),   # 3 arcs
    ], "static": []},
    # RC8: 3 from o1, 3 from o3 — perpendicular flood
    {"agents": [
        (('o1', 'ir1', 0), "o0",   0),   # 1 arc
        (('o1', 'ir1', 0), "o3",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o2",   0),   # 3 arcs
        (('o3', 'ir3', 0), "o2",   0),   # 1 arc
        (('o3', 'ir3', 0), "o1",   0),   # 2 arcs
        (('o3', 'ir3', 0), "o0",   0),   # 3 arcs
    ], "static": []},
    # RC9: Mixed paths, all 4 approaches, offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o0",   0),   # 1 arc
        (('o2', 'ir2', 0), "o3",   0),   # 3 arcs
        (('o3', 'ir3', 0), "o1",   0),   # 2 arcs
        (('o0', 'ir0', 0), "o3",  -5),   # 1 arc
        (('o2', 'ir2', 0), "o1",  -5),   # 1 arc
    ], "static": []},
    # RC10: Maximum ring occupancy — all long paths at offset=0
    {"agents": [
        (('o0', 'ir0', 0), "o1",   0),   # 3 arcs
        (('o1', 'ir1', 0), "o2",   0),   # 3 arcs
        (('o2', 'ir2', 0), "o3",   0),   # 3 arcs
        (('o3', 'ir3', 0), "o0",   0),   # 3 arcs
        (('o0', 'ir0', 0), "o2",   0),   # 2 arcs
        (('o1', 'ir1', 0), "o3",   0),   # 2 arcs
    ], "static": []},
]

ROUNDABOUT_CONFLICT_HELD_OUT_INDICES = frozenset([5, 18, 27, 34])


# ── Double-intersection CONFLICT scenarios (coordination-critical) ────────────
# Both intersections get simultaneous traffic that must use the A↔B connector,
# creating forced merge conflicts.  Without master coordination, connector
# deadlock or intersection collision is guaranteed.
#
# 10 base scenarios (no rotations — topology is fixed).
double_intersection_conflict_base_scenarios = [
    # DC1: All 6 cross to opposite intersection, offset=0
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o0",  0),
        (('B_o0', 'B_ir0', 0), "A_o1",  0),
        (('B_o2', 'B_ir2', 0), "A_o0",  0),
        (('B_o3', 'B_ir3', 0), "A_o2",  0),
    ], "static": []},
    # DC2: 3 A→B, 3 B within (A floods the connector)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o1', 'A_ir1', 0), "B_o3",  0),
        (('A_o2', 'A_ir2', 0), "B_o2",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
        (('B_o3', 'B_ir3', 0), "B_o0",  0),
    ], "static": []},
    # DC3: 3 B→A, 3 A within (B floods the connector)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o1",  0),
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "A_o2",  0),
    ], "static": []},
    # DC4: Bidirectional connector flood (2 A→B + 2 B→A + 2 local)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "A_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ], "static": []},
    # DC5: All A-cars cross, all B-cars to same B exit
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o2",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o2",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ], "static": []},
    # DC6: A and B simultaneously cross, meeting at connector
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ], "static": []},
    # DC7: Max conflict — 4 vehicles on connector + 2 local
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "A_o2",  0),
        (('B_o3', 'B_ir3', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
    ], "static": []},
    # DC8: Both intersections full, mixed local + cross
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "A_o0",  0),
        (('B_o3', 'B_ir3', 0), "A_o1",  0),
    ], "static": []},
    # DC9: A convoy to B + B convoy to A (near-simultaneous)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o0', 'A_ir0', 0), "B_o3", -5),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "A_o1", -5),
        (('B_o3', 'B_ir3', 0), "A_o2",  0),
    ], "static": []},
    # DC10: Simultaneous left turns + cross at both intersections
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
    ], "static": []},
]

DOUBLE_INTERSECTION_CONFLICT_HELD_OUT_INDICES = frozenset([2, 7])


# ── Double-intersection scenarios ─────────────────────────────────────────────
# Two intersections: A (left, arms A_o0=south, A_o1=west, A_o2=north) and
# B (right, arms B_o0=south, B_o2=north, B_o3=east).
# A_o3 and B_o1 are replaced by the A↔B connector (not outer exits).
#
# Agents 0-2 (LM1): start at intersection A  |  Agents 3-5 (LM2): start at B
#
# Valid within-A routes:
#   A_o0→A_o1 (left), A_o0→A_o2 (straight)
#   A_o1→A_o0 (right), A_o1→A_o2 (left)
#   A_o2→A_o0 (straight), A_o2→A_o1 (right)
# Valid within-B routes:
#   B_o0→B_o2 (straight), B_o0→B_o3 (right)
#   B_o2→B_o0 (straight), B_o2→B_o3 (left)
#   B_o3→B_o0 (left), B_o3→B_o2 (right)
# Cross A→B (via right/straight/left in A):
#   A_o0→B_*, A_o1→B_*, A_o2→B_*  (B_* ∈ {B_o0, B_o2, B_o3})
# Cross B→A (via left/right/straight in B):
#   B_o0→A_*, B_o2→A_*, B_o3→A_*  (A_* ∈ {A_o0, A_o1, A_o2})
double_intersection_base_scenarios = [
    # S1: All within own intersection
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2", 0),
        (('A_o1', 'A_ir1', 0), "A_o0", 0),
        (('A_o2', 'A_ir2', 0), "A_o1", 0),
        (('B_o0', 'B_ir0', 0), "B_o2", 0),
        (('B_o2', 'B_ir2', 0), "B_o0", 0),
        (('B_o3', 'B_ir3', 0), "B_o2", 0),
    ], "static": []},
    # S2: All within, left-turn focus
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1", 0),
        (('A_o1', 'A_ir1', 0), "A_o2", 0),
        (('A_o2', 'A_ir2', 0), "A_o1", 0),
        (('B_o0', 'B_ir0', 0), "B_o3", 0),
        (('B_o2', 'B_ir2', 0), "B_o3", 0),
        (('B_o3', 'B_ir3', 0), "B_o0", 0),
    ], "static": []},
    # S3: A crosses to B, B stays within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3", 0),
        (('A_o1', 'A_ir1', 0), "B_o0", 0),
        (('A_o2', 'A_ir2', 0), "B_o2", 0),
        (('B_o0', 'B_ir0', 0), "B_o2", 0),
        (('B_o2', 'B_ir2', 0), "B_o0", 0),
        (('B_o3', 'B_ir3', 0), "B_o2", 0),
    ], "static": []},
    # S4: B crosses to A, A stays within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2", 0),
        (('A_o1', 'A_ir1', 0), "A_o0", 0),
        (('A_o2', 'A_ir2', 0), "A_o1", 0),
        (('B_o0', 'B_ir0', 0), "A_o2", 0),
        (('B_o2', 'B_ir2', 0), "A_o0", 0),
        (('B_o3', 'B_ir3', 0), "A_o1", 0),
    ], "static": []},
    # S5: Bidirectional crossing (A→B and B→A mixed)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0", 0),
        (('A_o1', 'A_ir1', 0), "A_o2", 0),
        (('A_o2', 'A_ir2', 0), "A_o0", 0),
        (('B_o0', 'B_ir0', 0), "A_o1", 0),
        (('B_o2', 'B_ir2', 0), "B_o3", 0),
        (('B_o3', 'B_ir3', 0), "B_o0", 0),
    ], "static": []},
    # S6: Staggered on A_o0 and B_o3
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o0', 'A_ir0', 0), "A_o1", -25),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
        (('B_o3', 'B_ir3', 0), "B_o0", -25),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
    ], "static": []},
    # S7: Staggered on A_o1 and B_o2
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o0", -25),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
        (('B_o2', 'B_ir2', 0), "B_o3", -25),
    ], "static": []},
    # S8: Two A-cars cross to B, rest in own intersection
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
        (('B_o3', 'B_ir3', 0), "B_o0",  0),
    ], "static": []},
    # S9: Two B-cars cross to A, rest in own intersection
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o2",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ], "static": []},
    # S10: Full cross swap — all A to B, all B to A
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
        (('B_o0', 'B_ir0', 0), "A_o1",  0),
        (('B_o2', 'B_ir2', 0), "A_o2",  0),
        (('B_o3', 'B_ir3', 0), "A_o0",  0),
    ], "static": []},
    # S11: Staggered convoy from A to B
    {"agents": [
        (('A_o1', 'A_ir1', 0), "B_o3",   0),
        (('A_o1', 'A_ir1', 0), "B_o2",  -25),
        (('A_o2', 'A_ir2', 0), "A_o0",   0),
        (('B_o0', 'B_ir0', 0), "B_o2",   0),
        (('B_o2', 'B_ir2', 0), "B_o0",   0),
        (('B_o3', 'B_ir3', 0), "B_o2",   0),
    ], "static": []},
    # S12: Mixed left/right A + 1 cross + B within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ], "static": []},
    # S13: Staggered on A_o2 and B_o0
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o3", -25),
    ], "static": []},
    # S14: Cross with staggering
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o0', 'A_ir0', 0), "A_o2",  25),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o3', 'B_ir3', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  25),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
    ], "static": []},
    # S15: All B crossing to A, A within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o1",  0),
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "A_o2",  0),
    ], "static": []},
    # S16: Diagonal: A_o0→B_o3, B_o3→A_o0
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('B_o3', 'B_ir3', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ], "static": []},
    # S17: Pressure on connector (A_o0 + A_o1 crossing + B_o3 crossing)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o2",  0),
        (('A_o1', 'A_ir1', 0), "B_o3",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ], "static": []},
    # S18: Spread across all starting lanes, all within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ], "static": []},
    # S19: Mixed destinations across both intersections
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "A_o2",  0),
        (('B_o3', 'B_ir3', 0), "B_o0",  0),
    ], "static": []},
    # S20: Staggered on A_o2 and B_o3
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o1", -25),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o3', 'B_ir3', 0), "B_o0",  0),
        (('B_o3', 'B_ir3', 0), "B_o2", -25),
    ], "static": []},
]

# Appended procedural pools (50 conservative bases per layout — see extra_solvable_scenarios.py).
from src.experiment import extra_solvable_scenarios as _extra_sol  # noqa: E402

base_complete_scenarios_6_cars = (
    base_complete_scenarios_6_cars + list(_extra_sol.EXTRA_INTERSECTION_6CAR_50)
)
roundabout_base_scenarios = (
    roundabout_base_scenarios + list(_extra_sol.EXTRA_ROUNDABOUT_50)
)
double_intersection_base_scenarios = (
    double_intersection_base_scenarios + list(_extra_sol.EXTRA_DOUBLE_INTERSECTION_50)
)
