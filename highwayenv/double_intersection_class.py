from __future__ import annotations

import random
import functools

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork

from highwayenv.intersection_class import (
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)
from src.experiment.scenarios import (
    double_intersection_base_scenarios,
    double_intersection_conflict_base_scenarios,
)
from src import project_globals


class DoubleIntersectionEnv(IntersectionEnv):
    """
    Two 4-way intersections placed side-by-side and connected by a
    bidirectional road between A's east arm and B's west arm.
    """

    OUTER_EXIT_TARGETS = {
        "A_o0", "A_o1", "A_o2",
        "B_o0", "B_o2", "B_o3",
    }

    def _make_road(self) -> None:
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 100

        connector_length = self.config.get("connector_length", 80)

        half_sep = outer_distance + connector_length / 2
        center_a = np.array([-half_sep, 0.0])
        center_b = np.array([+half_sep, 0.0])

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        def build_intersection(prefix, center, skip_exit=None, skip_approach=None):
            for corner in range(4):
                angle = np.radians(90 * corner)
                is_horizontal = corner % 2
                priority = 3 if is_horizontal else 1
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ])

                o = prefix + "o" + str(corner)
                ir = prefix + "ir" + str(corner)
                il_right = prefix + "il" + str((corner - 1) % 4)
                il_left = prefix + "il" + str((corner + 1) % 4)
                il_str = prefix + "il" + str((corner + 2) % 4)

                if corner != skip_approach:
                    start = center + rotation @ np.array([lane_width / 2, access_length + outer_distance])
                    end = center + rotation @ np.array([lane_width / 2, outer_distance])
                    net.add_lane(o, ir, StraightLane(start, end, line_types=[s, c],
                                                     priority=priority, speed_limit=10))

                r_center = center + rotation @ np.array([outer_distance, outer_distance])
                net.add_lane(ir, il_right, CircularLane(
                    r_center, right_turn_radius,
                    angle + np.radians(180), angle + np.radians(270),
                    line_types=[n, c], priority=priority, speed_limit=10))

                l_center = center + rotation @ np.array(
                    [-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2])
                net.add_lane(ir, il_left, CircularLane(
                    l_center, left_turn_radius,
                    angle + np.radians(0), angle + np.radians(-90),
                    clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))

                start = center + rotation @ np.array([lane_width / 2, outer_distance])
                end = center + rotation @ np.array([lane_width / 2, -outer_distance])
                net.add_lane(ir, il_str, StraightLane(start, end, line_types=[s, n],
                                                      priority=priority, speed_limit=10))

                exit_corner = (corner - 1) % 4
                if exit_corner != skip_exit:
                    il_exit = prefix + "il" + str(exit_corner)
                    o_exit = prefix + "o" + str(exit_corner)
                    start = center + rotation @ np.flip(
                        [lane_width / 2, access_length + outer_distance], axis=0)
                    end = center + rotation @ np.flip(
                        [lane_width / 2, outer_distance], axis=0)
                    net.add_lane(il_exit, o_exit,
                                 StraightLane(end, start, line_types=[n, c],
                                              priority=priority, speed_limit=10))

        build_intersection("A_", center_a, skip_exit=3, skip_approach=3)
        build_intersection("B_", center_b, skip_exit=1, skip_approach=1)

        # ── Connector lanes (A→B and B→A) ────────────────────────────────────
        a_il3_pos = center_a + np.array([outer_distance, lane_width / 2])
        angle_west = np.radians(90)
        rot_west = np.array([
            [np.cos(angle_west), -np.sin(angle_west)],
            [np.sin(angle_west), np.cos(angle_west)],
        ])
        b_ir1_pos = center_b + rot_west @ np.array([lane_width / 2, outer_distance])
        net.add_lane("A_il3", "B_ir1",
                     StraightLane(a_il3_pos, b_ir1_pos, line_types=[s, c], priority=2, speed_limit=10))

        b_il1_pos = center_b + np.array([-outer_distance, -lane_width / 2])
        angle_east = np.radians(270)
        rot_east = np.array([
            [np.cos(angle_east), -np.sin(angle_east)],
            [np.sin(angle_east), np.cos(angle_east)],
        ])
        a_ir3_pos = center_a + rot_east @ np.array([lane_width / 2, outer_distance])
        net.add_lane("B_il1", "A_ir3",
                     StraightLane(b_il1_pos, a_ir3_pos, line_types=[s, c], priority=2, speed_limit=10))

        self.road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def has_arrived(self, vehicle, exit_distance: float = 25) -> bool:
        src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
        if dst not in self.OUTER_EXIT_TARGETS:
            return False
        if "_il" not in src:
            return False
        return vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _make_vehicles(self, n_vehicles: int = 1) -> None:
        """
        Override parent's _make_vehicles to prevent KeyErrors.
        The parent class attempts to spawn vehicles on single-intersection
        lanes (like 'o0') which no longer exist in our double-intersection map.
        We initialize the objects on a valid lane here, and _reset() will
        immediately move them to their correct scenario positions.
        """
        import functools  # Ensure this is imported
        from highway_env.vehicle.kinematics import Vehicle

        self.controlled_vehicles = []

        # 1. Spawn controlled agents using the environment's action_type
        # to ensure IPPO/RL action bindings work correctly.
        agent_count = 6  # Force 6 for double intersection as scenarios require it
        
        if hasattr(self, "action_type") and hasattr(self.action_type, "vehicle_class"):
            v_class = self.action_type.vehicle_class
            # UNWRAP THE PARTIAL HERE
            if isinstance(v_class, functools.partial):
                v_class = v_class.func
        else:
            v_class = Vehicle

        # Place agents temporarily at A_o0
        for _ in range(agent_count):
            lane_index = ("A_o0", "A_ir0", 0)
            # Pass lane_index instead of the lane object, and add longitudinal=0
            vehicle = v_class.make_on_lane(self.road, lane_index, longitudinal=0, speed=0)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        # 2. Spawn buffer static vehicles for the scenarios to utilize.
        # We spawn a generous buffer so _reset() has enough objects to pull from.
        for _ in range(n_vehicles + 10):
            lane_index = ("B_o0", "B_ir0", 0)
            # Same fix here for the buffer vehicles
            vehicle = Vehicle.make_on_lane(self.road, lane_index, longitudinal=0, speed=0)
            self.road.vehicles.append(vehicle)

    def _clear_vehicles(self) -> None:
        def is_leaving(vehicle):
            src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
            if dst not in self.OUTER_EXIT_TARGETS:
                return False
            if "_il" not in src:
                return False
            return (vehicle.lane.local_coordinates(vehicle.position)[0]
                    >= vehicle.lane.length - 4 * vehicle.LENGTH)

        self.road.vehicles = [
            v for v in self.road.vehicles
            if v in self.controlled_vehicles or not is_leaving(v)
        ]

    def _adapt_scenario(self, scenario: dict) -> dict:
        """
        Translates single-intersection scenarios (o0, o1, etc.) into the
        Double Intersection network layout (A_o0, B_o3, etc.).
        """
        node_map = {
            'o0': 'A_o0',  # South defaults to Intersection A
            'ir0': 'A_ir0',
            'o1': 'A_o1',  # West must be Intersection A
            'ir1': 'A_ir1',
            'o2': 'A_o2',  # North defaults to Intersection A
            'ir2': 'A_ir2',
            'o3': 'B_o3',  # East must be Intersection B
            'ir3': 'B_ir3'
        }

        def map_lane(lane_key):
            if isinstance(lane_key, tuple) and len(lane_key) == 3:
                u, v, idx = lane_key
                return (node_map.get(u, u), node_map.get(v, v), idx)
            return lane_key

        adapted = {"agents": [], "static": []}

        for lane_key, dest, off in scenario.get("agents", []):
            adapted["agents"].append((map_lane(lane_key), node_map.get(dest, dest), off))

        for lane_key, dest, off in scenario.get("static", []):
            adapted["static"].append((map_lane(lane_key), node_map.get(dest, dest), off))

        return adapted

    def _reset(self) -> None:
        project_globals.after_is_arrived_flags = [False] * len(self.controlled_vehicles)

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40

        all_regular = list(double_intersection_base_scenarios)
        all_conflict = list(double_intersection_conflict_base_scenarios)

        if not all_regular and not all_conflict:
            print("[DoubleIntersectionEnv._reset] WARNING: no scenarios loaded!")
            return

        use_conflict_only = self.config.get("use_conflict_scenarios_only", False)
        conflict_ratio = self.config.get("conflict_ratio", 0.0)

        chosen_scenario = None
        force_idx = self.config.get("force_scenario_index", None)

        if force_idx is not None:
            chosen_scenario = all_regular[int(force_idx) % len(all_regular)]
        elif use_conflict_only and all_conflict:
            chosen_scenario = random.choice(all_conflict)
        elif conflict_ratio > 0.0 and all_conflict and random.random() < conflict_ratio:
            chosen_scenario = random.choice(all_conflict)
        elif all_regular:
            chosen_scenario = random.choice(all_regular)
        else:
            chosen_scenario = random.choice(all_conflict)

        assert chosen_scenario is not None

        # Translate single-intersection nodes to the double-intersection map
        adapted_scenario = self._adapt_scenario(chosen_scenario)

        all_scenarios = all_regular + all_conflict
        try:
            self.last_scenario_index = all_scenarios.index(chosen_scenario)
        except ValueError:
            self.last_scenario_index = -1

        for i, (lane_key, destination, off) in enumerate(adapted_scenario["agents"]):
            vehicle = self.controlled_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(BASE_LONG + off, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, 'plan_route_to'):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]

        all_vehicles = self.road.vehicles
        controlled_count = len(self.controlled_vehicles)

        safe_static_scenario = []
        for lane_key, destination, off in adapted_scenario["static"]:
            lane = self.road.network.get_lane(lane_key)
            position = np.array(lane.position(BASE_LONG + off, 0))

            too_close = False
            for cv in self.controlled_vehicles:
                if np.linalg.norm(cv.position - position) < 10:
                    too_close = True
                    break
            for existing_key, _, existing_off in safe_static_scenario:
                existing_lane = self.road.network.get_lane(existing_key)
                if np.linalg.norm(np.array(existing_lane.position(BASE_LONG + existing_off, 0)) - position) < 10:
                    too_close = True
                    break

            if not too_close:
                safe_static_scenario.append((lane_key, destination, off))

        for i in range(controlled_count, min(len(all_vehicles), controlled_count + len(safe_static_scenario))):
            static_index = i - controlled_count
            if static_index >= len(safe_static_scenario):
                break
            lane_key, destination, off = safe_static_scenario[static_index]
            vehicle = all_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(BASE_LONG + off, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, 'plan_route_to'):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]


class MultiAgentDoubleIntersectionEnv(DoubleIntersectionEnv, MultiAgentIntersectionEnv):
    """Multi-agent wrapper — inherits MultiAgent defaults from
    MultiAgentIntersectionEnv and road geometry from DoubleIntersectionEnv."""
    pass