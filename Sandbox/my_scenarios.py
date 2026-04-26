# =============================================================================
# MY SCENARIOS — Sandbox Playground
# =============================================================================
# Edit SANDBOX_SCENARIOS and SANDBOX_ENV_CONFIG, then run run_sandbox.py to
# test visually. When happy, copy your scenario dicts into:
#   DRL_Research_scale/src/experiment/scenarios.py  →  base_complete_scenarios_3_cars
# =============================================================================

# ---------------------------------------------------------------------------
# LANE NOTATION REFERENCE
# ---------------------------------------------------------------------------
# Outer (approach) lanes  — where cars start / end:
#   o0  =  SOUTH approach   (car comes from south, enters from south)
#   o1  =  WEST approach
#   o2  =  NORTH approach
#   o3  =  EAST approach
#
# Inner right-turn lanes (entry arc):
#   ir0, ir1, ir2, ir3  — match the outer lane index
#
# Start lane tuple format:  (('oX', 'irX', 0), destination, longitudinal_offset_m)
#   destination           — where the car exits, e.g. "o2"
#   longitudinal_offset   — distance FROM the intersection (negative = further away)
#                           0 = at the entry; -50 = 50 m back; +20 = slightly ahead
#
# Common routes:
#   SOUTH → NORTH  straight:   start=('o0','ir0',0)  dest="o2"
#   SOUTH → EAST   right turn: start=('o0','ir0',0)  dest="o1"
#   SOUTH → WEST   left turn:  start=('o0','ir0',0)  dest="o3"
#   WEST  → EAST   straight:   start=('o1','ir1',0)  dest="o3"
#   NORTH → SOUTH  straight:   start=('o2','ir2',0)  dest="o0"
#   EAST  → WEST   straight:   start=('o3','ir3',0)  dest="o1"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# YOUR SCENARIOS
# Each scenario needs:
#   "agents"  — list of 3 tuples (start_lane, destination, offset)  → controlled cars
#   "static"  — list of 2 tuples (start_lane, destination, offset)  → non-controlled cars
# ---------------------------------------------------------------------------

SANDBOX_SCENARIOS = [

    # ------------------------------------------------------------------
    # Scenario A: Simple straight-through (all go straight)
    # ------------------------------------------------------------------
    {
        "agents": [
            (('o0', 'ir0', 0), "o2",  0),    # SOUTH → NORTH
            (('o1', 'ir1', 0), "o3",  0),    # WEST  → EAST
            (('o2', 'ir2', 0), "o0", -25),   # NORTH → SOUTH, 25 m back
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -35),   # EAST → WEST, 35 m back
            (('o0', 'ir0', 0), "o1", -15),   # SOUTH → EAST, 15 m back
        ]
    },

    # ------------------------------------------------------------------
    # Scenario B: Crossing conflict — two agents cross each other
    # ------------------------------------------------------------------
    {
        "agents": [
            (('o0', 'ir0', 0), "o3",   0),   # SOUTH → WEST  (left turn)
            (('o1', 'ir1', 0), "o2",   0),   # WEST  → NORTH (left turn)
            (('o2', 'ir2', 0), "o0", -40),   # NORTH → SOUTH (straight, behind)
        ],
        "static": [
            (('o3', 'ir3', 0), "o2", -30),   # EAST  → NORTH
            (('o0', 'ir0', 0), "o2", -55),   # SOUTH → NORTH (far back)
        ]
    },


    # ------------------------------------------------------------------
    # Scenario C: Crossing conflict — three agents cross each other
    # -----------------------------------------------------------------
    
    {
        "agents": [
            (('o0', 'ir0', 0), "o3",   0),   # SOUTH → WEST (left turn)
            (('o1', 'ir1', 0), "o2", -19),   # WEST  → NORTH (left turn)
            (('o2', 'ir2', 0), "o1", -40),   # NORTH → WEST (right turn)
        ],
        "static": [ 
            (('o3', 'ir3', 0), "o1", -30),   # EAST  → WEST
            (('o0', 'ir0', 0), "o1", -55),   # SOUTH → EAST (far back)
        ]
    },

# ------------------------------------------------------------------
    # Scenario D: The Blocked Intersection (Congestion Test)
    # ------------------------------------------------------------------
    {
        "agents": [
            (('o1', 'ir1', 0), "o3",   0),   # WEST  → EAST  (straight, arrives first)
            (('o0', 'ir0', 0), "o2", -20),   # SOUTH → NORTH (straight, arrives second)
            (('o3', 'ir3', 0), "o0", -45),   # EAST  → SOUTH (left turn)
        ],
        "static": [ 
            (('o2', 'ir2', 0), "o0",  -5),   # NO
            (('o3', 'ir3', 0), "o1", -10),   # EAST  → WEST  
        ]
    },

    # _____________________________________________________________________________
    {
        "agents": [
            (('o0', 'ir0', 0), "o3",   0),   # SOUTH → WEST  (left turn)
            (('o1', 'ir1', 0), "o2", -19),   # WEST  → NORTH (left turn)
            (('o2', 'ir2', 0), "o1", -40),   # NORTH → WEST  (right turn)
        ],
        "static": [ 
            (('o3', 'ir3', 0), "o1", -30),   # EAST  → WEST
            (('o0', 'ir0', 0), "o1", -55),   # SOUTH → EAST (far back)
        ]
    }


    

    # ------------------------------------------------------------------
    # ADD YOUR OWN SCENARIOS BELOW
    # ------------------------------------------------------------------
    # {
    #     "agents": [
    #         (('o0', 'ir0', 0), "o2",  0),
    #         (('o1', 'ir1', 0), "o3",  0),
    #         (('o3', 'ir3', 0), "o1", -20),
    #     ],
    #     "static": [
    #         (('o2', 'ir2', 0), "o0", -30),
    #         (('o0', 'ir0', 0), "o1", -50),
    #     ]
    # },

]


# =============================================================================
# ROUNDABOUT SCENARIOS
# =============================================================================
# Lane notation for the roundabout:
#   Approach spurs : o0 (South), o1 (West), o2 (North), o3 (East)
#   Ring junctions : ir0, ir1, ir2, ir3 (matching approach index)
#   Exit labels    : il0→o0, il1→o1, il2→o2, il3→o3
#
# Start lane format: (('oX', 'irX', 0), destination, longitudinal_offset)
#   - same tuple structure as the intersection
#   - destination = "oY" where Y is the exit the car should leave from
#
# Common routes:
#   South→North:  start=('o0','ir0',0)  dest='o2'   (half-circle)
#   South→West:   start=('o0','ir0',0)  dest='o1'   (quarter-circle, first exit)
#   South→East:   start=('o0','ir0',0)  dest='o3'   (three-quarter circle)
# =============================================================================

ROUNDABOUT_SCENARIOS = [

    # ------------------------------------------------------------------
    # Roundabout A: Three cars, each crosses half the ring
    # ------------------------------------------------------------------
    {
        "agents": [
            (('o0', 'ir0', 0), "o2",   0),   # South → North (half ring)
            (('o1', 'ir1', 0), "o3",   0),   # West  → East  (half ring)
            (('o2', 'ir2', 0), "o0", -25),   # North → South (half ring)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -35),   # East → West
            (('o0', 'ir0', 0), "o1", -15),   # South → West (first exit)
        ]
    },

    # ------------------------------------------------------------------
    # Roundabout B: First-exit vs long-route conflict
    # ------------------------------------------------------------------
    {
        "agents": [
            (('o0', 'ir0', 0), "o1",   0),   # South → West (first exit, quick)
            (('o1', 'ir1', 0), "o0", -10),   # West  → South (three-quarter)
            (('o3', 'ir3', 0), "o2", -30),   # East  → North (first exit)
        ],
        "static": [
            (('o2', 'ir2', 0), "o0", -40),   # North → South
            (('o0', 'ir0', 0), "o3", -55),   # South → East (three-quarter)
        ]
    },

    # ------------------------------------------------------------------
    # Roundabout C: Three distinct destinations, staggered entries
    # ------------------------------------------------------------------
    {
        "agents": [
            (('o0', 'ir0', 0), "o2",   0),   # South → North (half ring)
            (('o1', 'ir1', 0), "o0", -15),   # West  → South (three-quarter ring)
            (('o2', 'ir2', 0), "o3", -35),   # North → East  (first exit)
        ],
        "static": [
            (('o3', 'ir3', 0), "o1", -30),   # East → West
            (('o1', 'ir1', 0), "o3", -50),   # West → East (first exit, far back)
        ]
    },
]


# ---------------------------------------------------------------------------
# ENVIRONMENT CONFIGURATION  (Intersection)
# Tweak rewards, screen size, speeds, duration, etc.
# ---------------------------------------------------------------------------

SANDBOX_ENV_CONFIG = {
    # ---- rewards ----
    "collision_reward":  -300,
    "arrived_reward":     50,
    "starvation_reward":  -5.0,
    "normalize_reward":   False,

    # ---- episode timing ----
    "duration": 50,              # max episode length in seconds

    # ---- vehicles ----
    "initial_vehicle_count": 5,  # 3 agents + 2 static
    "spawn_probability": 0.0,    # no random spawning

    # ---- display ----
    "screen_width":  900,
    "screen_height": 800,
    "scaling":       3.9,
    "centering_position": [0.5, 0.6],

    # ---- observation ----
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "features_range": {
            "x":  [-100, 100],
            "y":  [-100, 100],
            "vx": [-20,   20],
            "vy": [-20,   20],
        },
        "absolute": True,
        "flatten":  False,
        "observe_intentions": False,
    },

    # ---- action ----
    "action": {
        "type": "CustomMultiAgentAction",
        "action_config": {"type": "CustomDiscreteAction"},
        "target_speeds": [5, 10],   # m/s — SLOWER / FASTER targets
    },

    # ---- vehicle slots (_make_vehicles uses these to create objects;
    #       _reset overrides positions from the chosen scenario) ----
    "controlled_cars": {
        "agent_0": {
            "start_lane": ("o0", "ir0", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (255, 100, 100),   # red-ish
            "destination": "o2",
        },
        "agent_1": {
            "start_lane": ("o1", "ir1", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (100, 255, 100),   # green-ish
            "destination": "o3",
        },
        "agent_2": {
            "start_lane": ("o2", "ir2", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (100, 100, 255),   # blue-ish
            "destination": "o0",
        },
    },
    "static_cars": {
        "static_0": {
            "start_lane": ("o3", "ir3", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
        },
        "static_1": {
            "start_lane": ("o0", "ir0", 0),
            "init_location": {"longitudinal": 15, "lateral": 0},
            "speed": 8,
        },
    },
}


# ---------------------------------------------------------------------------
# ROUNDABOUT ENVIRONMENT CONFIGURATION
# Same structure as SANDBOX_ENV_CONFIG — uses the same lane key names.
# ---------------------------------------------------------------------------

ROUNDABOUT_ENV_CONFIG = {
    # ---- rewards ----
    "collision_reward":  -300,
    "arrived_reward":     50,
    "starvation_reward":  -5.0,
    "normalize_reward":   False,

    # ---- episode timing ----
    "duration": 50,

    # ---- vehicles ----
    "initial_vehicle_count": 5,  # 3 agents + 2 static
    "spawn_probability": 0.0,

    # ---- display ----
    "screen_width":  900,
    "screen_height": 800,
    "scaling":       3.9,
    "centering_position": [0.5, 0.6],

    # ---- observation ----
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "features_range": {
            "x":  [-100, 100],
            "y":  [-100, 100],
            "vx": [-20,   20],
            "vy": [-20,   20],
        },
        "absolute": True,
        "flatten":  False,
        "observe_intentions": False,
    },

    # ---- action ----
    "action": {
        "type": "CustomMultiAgentAction",
        "action_config": {"type": "CustomDiscreteAction"},
        "target_speeds": [5, 10],
    },

    # ---- vehicle slots ----
    "controlled_cars": {
        "agent_0": {
            "start_lane": ("o0", "ir0", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (255, 100, 100),
            "destination": "o2",
        },
        "agent_1": {
            "start_lane": ("o1", "ir1", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (100, 255, 100),
            "destination": "o3",
        },
        "agent_2": {
            "start_lane": ("o2", "ir2", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (100, 100, 255),
            "destination": "o0",
        },
    },
    "static_cars": {
        "static_0": {
            "start_lane": ("o3", "ir3", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
        },
        "static_1": {
            "start_lane": ("o0", "ir0", 0),
            "init_location": {"longitudinal": 15, "lateral": 0},
            "speed": 8,
        },
    },
}


# =============================================================================
# DOUBLE INTERSECTION SCENARIOS
# =============================================================================
# Two intersections side by side: A (left) and B (right), connected
# via A_east ↔ B_west bidirectional road.
#
# Lane names use A_ / B_ prefix:
#   A_o0 = A south, A_o1 = A west, A_o2 = A north, A_o3 = A east (connector)
#   B_o0 = B south, B_o1 = B west (connector), B_o2 = B north, B_o3 = B east
#
# Through-routes (cross both intersections):
#   A west → B east:  start ('A_o1','A_ir1',0)  dest "B_o3"
#   B east → A west:  start ('B_o3','B_ir3',0)  dest "A_o1"
# =============================================================================

DOUBLE_INTERSECTION_SCENARIOS = [

    # ------------------------------------------------------------------
    # DI-6A: Full cross-through — all 6 agents cross to the other node
    #   A agents (0-2) head toward B exits
    #   B agents (3-5) head toward A exits
    # ------------------------------------------------------------------
    {
        "agents": [
            # ── Intersection A agents ──
            (('A_o0', 'A_ir0', 0), "B_o0",    0),   # A south → through → B south
            (('A_o1', 'A_ir1', 0), "B_o3",  -15),   # A west  → through → B east
            (('A_o2', 'A_ir2', 0), "B_o2",  -30),   # A north → through → B north
            # ── Intersection B agents ──
            (('B_o0', 'B_ir0', 0), "A_o2",    0),   # B south → through → A north
            (('B_o3', 'B_ir3', 0), "A_o1",  -15),   # B east  → through → A west
            (('B_o2', 'B_ir2', 0), "A_o0",  -30),   # B north → through → A south
        ],
        "static": [
            (('A_o0', 'A_ir0', 0), "A_o1",  -60),   # A south → A west (far back)
            (('B_o0', 'B_ir0', 0), "B_o2",  -60),   # B south → B north (far back)
        ]
    },

    # ------------------------------------------------------------------
    # DI-6B: Heavy connector + local conflict
    #   Most agents cross through, one per node stays local
    # ------------------------------------------------------------------
    {
        "agents": [
            # ── Intersection A agents ──
            (('A_o0', 'A_ir0', 0), "B_o2",    0),   # A south → through → B north
            (('A_o1', 'A_ir1', 0), "B_o0",  -20),   # A west  → through → B south
            (('A_o2', 'A_ir2', 0), "A_o0",  -40),   # A north → A south (local straight)
            # ── Intersection B agents ──
            (('B_o0', 'B_ir0', 0), "A_o1",    0),   # B south → through → A west
            (('B_o3', 'B_ir3', 0), "A_o2",  -20),   # B east  → through → A north
            (('B_o2', 'B_ir2', 0), "B_o0",  -40),   # B north → B south (local straight)
        ],
        "static": [
            (('A_o2', 'A_ir2', 0), "A_o1",  -70),   # A north → A west (far back)
            (('B_o0', 'B_ir0', 0), "B_o2",  -70),   # B south → B north (far back)
        ]
    },

    # ------------------------------------------------------------------
    # DI-6C: Mixed local + through traffic
    #   One agent per node stays local, two cross through
    # ------------------------------------------------------------------
    {
        "agents": [
            # ── Intersection A agents ──
            (('A_o0', 'A_ir0', 0), "A_o2",    0),   # A south → A north (local straight)
            (('A_o1', 'A_ir1', 0), "B_o3",  -15),   # A west  → through → B east
            (('A_o2', 'A_ir2', 0), "B_o0",  -35),   # A north → through → B south
            # ── Intersection B agents ──
            (('B_o0', 'B_ir0', 0), "B_o2",    0),   # B south → B north (local straight)
            (('B_o3', 'B_ir3', 0), "A_o1",  -15),   # B east  → through → A west
            (('B_o2', 'B_ir2', 0), "A_o0",  -35),   # B north → through → A south
        ],
        "static": [
            (('B_o3', 'B_ir3', 0), "B_o2",  -55),   # B east → B north (far back)
            (('A_o0', 'A_ir0', 0), "A_o1",  -55),   # A south → A west (far back)
        ]
    },

    # ------------------------------------------------------------------
    # DI-7C: Mixed local + through traffic
    #   One agent per node stays local, two cross through
    # ------------------------------------------------------------------

    {
        "agents": [
            # ── Intersection A agents ──
            (('A_o0', 'A_ir0', 0), "A_o2",    100),   # A south → A north (local straight)
            (('A_o1', 'A_ir1', 0), "B_o3",  -15),   # A west  → through → B east
            (('A_o2', 'A_ir2', 0), "B_o0",  -35),   # A north → through → B south
            # ── Intersection B agents ──
            (('B_o0', 'B_ir0', 0), "B_o2",    0),   # B south → B north (local straight)
            (('B_o3', 'B_ir3', 0), "A_o1",  -15),   # B east  → through → A west
            (('B_o2', 'B_ir2', 0), "A_o0",  -35),   # B north → through → A south
        ],
        "static": [
            (('B_o3', 'B_ir3', 0), "B_o2",  -55),   # B east → B north (far back)
            (('A_o0', 'A_ir0', 0), "A_o1",  -55),   # A south → A west (far back)
        ]
    },

    
]


# ---------------------------------------------------------------------------
# DOUBLE INTERSECTION ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------------------------

DOUBLE_INTERSECTION_ENV_CONFIG = {
    # ---- rewards ----
    "collision_reward":  -300,
    "arrived_reward":     50,
    "starvation_reward":  -5.0,
    "normalize_reward":   False,

    # ---- episode timing ----
    "duration": 50,

    # ---- vehicles ----
    "initial_vehicle_count": 8,  # 6 agents + 2 static
    "spawn_probability": 0.0,

    # ---- display (wider for two intersections side by side) ----
    "screen_width":  1400,
    "screen_height":  800,
    "scaling":        2.5,
    "centering_position": [0.5, 0.5],

    # ---- connector ----
    "connector_length": 80,  # metres between the two intersections

    # ---- observation ----
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "features_range": {
            "x":  [-200, 200],
            "y":  [-100, 100],
            "vx": [-20,   20],
            "vy": [-20,   20],
        },
        "absolute": True,
        "flatten":  False,
        "observe_intentions": False,
    },

    # ---- action ----
    "action": {
        "type": "CustomMultiAgentAction",
        "action_config": {"type": "CustomDiscreteAction"},
        "target_speeds": [5, 10],
    },

    # ---- vehicle slots (6 agents: 3 at intersection A, 3 at intersection B) ----
    "controlled_cars": {
        "agent_0": {
            "start_lane": ("A_o0", "A_ir0", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (255, 100, 100),
            "destination": "A_o2",
        },
        "agent_1": {
            "start_lane": ("A_o1", "A_ir1", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (100, 255, 100),
            "destination": "B_o3",
        },
        "agent_2": {
            "start_lane": ("A_o2", "A_ir2", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (100, 100, 255),
            "destination": "B_o0",
        },
        "agent_3": {
            "start_lane": ("B_o0", "B_ir0", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (255, 165, 0),
            "destination": "A_o2",
        },
        "agent_4": {
            "start_lane": ("B_o2", "B_ir2", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (0, 255, 255),
            "destination": "A_o0",
        },
        "agent_5": {
            "start_lane": ("B_o3", "B_ir3", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
            "color": (255, 0, 255),
            "destination": "A_o1",
        },
    },
    "static_cars": {
        "static_0": {
            "start_lane": ("B_o3", "B_ir3", 0),
            "init_location": {"longitudinal": 40, "lateral": 0},
            "speed": 8,
        },
        "static_1": {
            "start_lane": ("A_o2", "A_ir2", 0),
            "init_location": {"longitudinal": 15, "lateral": 0},
            "speed": 8,
        },
    },
}
