

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

roundabout_base_scenarios = base_complete_scenarios_3_cars

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

roundabout_conflict_base_scenarios = base_complete_scenarios_3_cars

ROUNDABOUT_CONFLICT_HELD_OUT_INDICES = frozenset([5, 18, 27, 34])


# ── Double-intersection CONFLICT scenarios (coordination-critical) ────────────
# Both intersections get simultaneous traffic that must use the A↔B connector,
# creating forced merge conflicts.  Without master coordination, connector
# deadlock or intersection collision is guaranteed.
#
# 10 base scenarios (no rotations — topology is fixed).
double_intersection_conflict_base_scenarios = [
    # DC1: All 5 cross to opposite intersection, offset=0
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o1",  0),
        (('B_o2', 'B_ir2', 0), "A_o0",  0),
    ]},
    # DC2: 3 A→B, 2 B within (A floods the connector)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o1', 'A_ir1', 0), "B_o3",  0),
        (('A_o2', 'A_ir2', 0), "B_o2",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
    ]},
    # DC3: 3 B→A, 2 A within (B floods the connector)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o1",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
    ]},
    # DC4: Bidirectional connector flood (2 A→B + 2 B→A + 1 local)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "A_o2",  0),
    ]},
    # DC5: All A-cars cross, all B-cars to same B exit
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o2",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o2",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ]},
    # DC6: A and B simultaneously cross, meeting at connector
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
    ]},
    # DC7: Max conflict — 4 vehicles on connector + 1 local
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o2",  0),
        (('B_o3', 'B_ir3', 0), "A_o0",  0),
    ]},
    # DC8: Both intersections full, mixed local + cross
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "A_o0",  0),
    ]},
    # DC9: A convoy to B + B convoy to A (near-simultaneous)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o0', 'A_ir0', 0), "B_o3", -5),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "A_o1", -5),
    ]},
    # DC10: Simultaneous left turns + cross at both intersections
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
    ]},
]

DOUBLE_INTERSECTION_CONFLICT_HELD_OUT_INDICES = frozenset([2, 7])


# ── Double-intersection scenarios ─────────────────────────────────────────────
double_intersection_base_scenarios = [
    # S1: All within own intersection
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2", 0),
        (('A_o1', 'A_ir1', 0), "A_o0", 0),
        (('A_o2', 'A_ir2', 0), "A_o1", 0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2", 0),
        (('B_o2', 'B_ir2', 0), "B_o0", 0),
    ]},
    # S2: All within, left-turn focus
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1", 0),
        (('A_o1', 'A_ir1', 0), "A_o2", 0),
        (('A_o2', 'A_ir2', 0), "A_o1", 0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o3", 0),
        (('B_o2', 'B_ir2', 0), "B_o3", 0),
    ]},
    # S3: A crosses to B, B stays within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3", 0),
        (('A_o1', 'A_ir1', 0), "B_o0", 0),
        (('A_o2', 'A_ir2', 0), "B_o2", 0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2", 0),
        (('B_o2', 'B_ir2', 0), "B_o0", 0),
    ]},
    # S4: B crosses to A, A stays within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2", 0),
        (('A_o1', 'A_ir1', 0), "A_o0", 0),
        (('A_o2', 'A_ir2', 0), "A_o1", 0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o2", 0),
        (('B_o2', 'B_ir2', 0), "A_o0", 0),
    ]},
    # S5: Bidirectional crossing (A→B and B→A mixed)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0", 0),
        (('A_o1', 'A_ir1', 0), "A_o2", 0),
        (('A_o2', 'A_ir2', 0), "A_o0", 0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o1", 0),
        (('B_o2', 'B_ir2', 0), "B_o3", 0),
    ]},
    # S6: Staggered on A_o0 and B_o3
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o0', 'A_ir0', 0), "A_o1", -25),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o3', 'B_ir3', 0), "B_o2",  0),
        (('B_o3', 'B_ir3', 0), "B_o0", -25),
    ]},
    # S7: Staggered on A_o1 and B_o2
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o0", -25),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ]},
    # S8: Two A-cars cross to B, rest in own intersection
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ]},
    # S9: Two B-cars cross to A, rest in own intersection
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o2",  0),
    ]},
    # S10: Full cross swap — all A to B, all B to A
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o0",  0),
        (('A_o1', 'A_ir1', 0), "B_o2",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o1",  0),
        (('B_o2', 'B_ir2', 0), "A_o2",  0),
    ]},
    # S11: Staggered convoy from A to B
    {"agents": [
        (('A_o1', 'A_ir1', 0), "B_o3",   0),
        (('A_o1', 'A_ir1', 0), "B_o2",  -25),
        (('A_o2', 'A_ir2', 0), "A_o0",   0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",   0),
        (('B_o2', 'B_ir2', 0), "B_o0",   0),
    ]},
    # S12: Mixed left/right A + 1 cross + B within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "B_o3",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
    ]},
    # S13: Staggered on A_o2 and B_o0
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ]},
    # S14: Cross with staggering
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o0', 'A_ir0', 0), "A_o2",  25),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o3', 'B_ir3', 0), "A_o1",  0),
        (('B_o3', 'B_ir3', 0), "B_o2",  25),
    ]},
    # S15: All B crossing to A, A within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o1",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "A_o0",  0),
        (('B_o2', 'B_ir2', 0), "A_o1",  0),
    ]},
    # S16: Diagonal: A_o0→B_o3, B_o3→A_o0
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o3",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
    ], "static": [
        (('B_o3', 'B_ir3', 0), "A_o0",  0),
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
    ]},
    # S17: Pressure on connector (A_o0 + A_o1 crossing + B_o3 crossing)
    {"agents": [
        (('A_o0', 'A_ir0', 0), "B_o2",  0),
        (('A_o1', 'A_ir1', 0), "B_o3",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "B_o0",  0),
    ]},
    # S18: Spread across all starting lanes, all within
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o1', 'A_ir1', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o2', 'B_ir2', 0), "B_o3",  0),
    ]},
    # S19: Mixed destinations across both intersections
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o1",  0),
        (('A_o1', 'A_ir1', 0), "B_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o3",  0),
        (('B_o2', 'B_ir2', 0), "A_o2",  0),
    ]},
    # S20: Staggered on A_o2 and B_o3
    {"agents": [
        (('A_o0', 'A_ir0', 0), "A_o2",  0),
        (('A_o2', 'A_ir2', 0), "A_o0",  0),
        (('A_o2', 'A_ir2', 0), "A_o1", -25),
    ], "static": [
        (('B_o0', 'B_ir0', 0), "B_o2",  0),
        (('B_o3', 'B_ir3', 0), "B_o0",  0),
    ]},
]
