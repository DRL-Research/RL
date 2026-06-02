from __future__ import annotations

import random

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from highwayenv.intersection_class import (
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)
from src.experiment.scenarios import double_intersection_base_scenarios
from src import project_globals


class DoubleIntersectionEnv(IntersectionEnv):
    """
    Two 4-way intersections placed side-by-side (horizontally) and connected
    by a bidirectional road between A's east arm and B's west arm.

    Layout
    ======
                 A_o2              B_o2
                  |                 |
        A_o1 -- [A] -- connector -- [B] -- B_o3
                  |                 |
                 A_o0              B_o0

    Node naming: every original node gets an "A_" or "B_" prefix.
      Intersection A: A_o0, A_ir0, A_il0, …, A_o3, A_ir3, A_il3
      Intersection B: B_o0, B_ir0, B_il0, …, B_o3, B_ir3, B_il3

    Connector (bidirectional):
      A→B:  A_il3 → B_ir1   (A east exit  → B west entry)
      B→A:  B_il1 → A_ir3   (B west exit  → A east entry)

    Outer exit spurs (where has_arrived triggers):
      A: A_il3→A_o3 is REMOVED (replaced by connector), so A has south/west/north exits
         A_il0→A_o0 (south), A_il1→A_o1 (west), A_il2→A_o2 (north)
      B: B_il1→B_o1 is REMOVED (replaced by connector), so B has south/north/east exits
         B_il0→B_o0 (south), B_il2→B_o2 (north), B_il3→B_o3 (east)
    """

    # The set of outer exit node pairs where has_arrived / _clear_vehicles apply.
    # These are the 6 remaining exit spurs (il→o) that are NOT replaced by connectors.
    OUTER_EXIT_TARGETS = {
        "A_o0", "A_o1", "A_o2",
        "B_o0", "B_o2", "B_o3",
    }

    def _make_road(self) -> None:
        """Build two intersections connected horizontally."""
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # 100 m approach/exit spurs

        connector_length = self.config.get("connector_length", 80)

        # Horizontal offset: each intersection center is half the gap apart
        # Total center-to-center = 2*outer_distance + connector_length
        half_sep = outer_distance + connector_length / 2
        center_a = np.array([-half_sep, 0.0])
        center_b = np.array([+half_sep, 0.0])

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        # --- Helper: build one intersection into the network ---
        def build_intersection(prefix: str, center: np.ndarray,
                               skip_exit: int | None = None,
                               skip_approach: int | None = None):
            """
            Build a 4-way intersection with prefixed node names.

            skip_exit: corner index whose exit spur (il→o) should NOT be built
                       (will be replaced by the connector).
            skip_approach: corner index whose approach spur (o→ir) should NOT
                          be built (will be replaced by the connector).
            """
            for corner in range(4):
                angle = np.radians(90 * corner)
                is_horizontal = corner % 2
                priority = 3 if is_horizontal else 1
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]]
                )

                # Node names with prefix
                o  = prefix + "o" + str(corner)
                ir = prefix + "ir" + str(corner)
                il_right = prefix + "il" + str((corner - 1) % 4)
                il_left  = prefix + "il" + str((corner + 1) % 4)
                il_str   = prefix + "il" + str((corner + 2) % 4)

                # ── Approach spur: o{i} → ir{i} ──────────────────────
                if corner != skip_approach:
                    start = center + rotation @ np.array(
                        [lane_width / 2, access_length + outer_distance]
                    )
                    end = center + rotation @ np.array(
                        [lane_width / 2, outer_distance]
                    )
                    net.add_lane(
                        o, ir,
                        StraightLane(start, end, line_types=[s, c],
                                     priority=priority, speed_limit=10),
                    )

                # ── Right turn: ir{i} → il{(i-1)%4} ─────────────────
                r_center = center + rotation @ np.array(
                    [outer_distance, outer_distance]
                )
                net.add_lane(
                    ir, il_right,
                    CircularLane(
                        r_center, right_turn_radius,
                        angle + np.radians(180),
                        angle + np.radians(270),
                        line_types=[n, c], priority=priority, speed_limit=10,
                    ),
                )

                # ── Left turn: ir{i} → il{(i+1)%4} ──────────────────
                l_center = center + rotation @ np.array(
                    [-left_turn_radius + lane_width / 2,
                     left_turn_radius - lane_width / 2]
                )
                net.add_lane(
                    ir, il_left,
                    CircularLane(
                        l_center, left_turn_radius,
                        angle + np.radians(0),
                        angle + np.radians(-90),
                        clockwise=False,
                        line_types=[n, n], priority=priority - 1, speed_limit=10,
                    ),
                )

                # ── Straight: ir{i} → il{(i+2)%4} ───────────────────
                start = center + rotation @ np.array(
                    [lane_width / 2, outer_distance]
                )
                end = center + rotation @ np.array(
                    [lane_width / 2, -outer_distance]
                )
                net.add_lane(
                    ir, il_str,
                    StraightLane(start, end, line_types=[s, n],
                                 priority=priority, speed_limit=10),
                )

                # ── Exit spur: il{(i-1)%4} → o{(i-1)%4} ────────────
                exit_corner = (corner - 1) % 4
                if exit_corner != skip_exit:
                    il_exit = prefix + "il" + str(exit_corner)
                    o_exit  = prefix + "o" + str(exit_corner)
                    start = center + rotation @ np.flip(
                        [lane_width / 2, access_length + outer_distance], axis=0
                    )
                    end = center + rotation @ np.flip(
                        [lane_width / 2, outer_distance], axis=0
                    )
                    net.add_lane(
                        il_exit, o_exit,
                        StraightLane(end, start, line_types=[n, c],
                                     priority=priority, speed_limit=10),
                    )

        # --- Build the two intersections ---
        # A: skip east exit (corner 3 exit → il3→o3) and keep east approach
        # B: skip west exit (corner 1 exit → il1→o1) and keep west approach
        build_intersection("A_", center_a, skip_exit=3, skip_approach=None)
        build_intersection("B_", center_b, skip_exit=1, skip_approach=None)

        # --- Remove A's east approach spur (A_o3→A_ir3) and
        #     B's west approach spur (B_o1→B_ir1) since the connector replaces them ---
        # Actually we keep the approach spurs! The connector goes from
        # A's exit internal node (A_il3) to B's entry internal node (B_ir1),
        # and from B's exit internal node (B_il1) to A's entry internal node (A_ir3).
        # But we do NOT need the external approach spurs on the connected sides,
        # because traffic enters from the connector instead.
        # So let's rebuild without those approach spurs:
        net = RoadNetwork()  # start fresh
        build_intersection("A_", center_a, skip_exit=3, skip_approach=3)
        build_intersection("B_", center_b, skip_exit=1, skip_approach=1)

        # --- Connector lanes (bidirectional) ---
        # A_il3 is the internal exit node on A's east side.
        # B_ir1 is the internal entry node on B's west side.
        # We need the physical positions of these nodes.

        # A's east exit internal position (il3): same geometry as corner=0's exit
        # but we compute it from the intersection structure.
        # Corner 3 = East: angle = 270°
        # The il3 node sits at the east inner edge of intersection A.
        # Corner 0 (South) builds exit for il3→o3.
        # The exit lane starts at: rotation(0°) @ flip([lw/2, access+od]) = [access+od, lw/2]
        # The exit lane ends at:   rotation(0°) @ flip([lw/2, od]) = [od, lw/2]
        # The lane goes from "end" to "start" (StraightLane(end, start)).
        # So the "end" position (near intersection) for il3 is at:
        #   center + [od, lw/2] for the original intersection.
        # But actually, for the connector we need to connect from where the
        # exit spur WOULD have started (at the il node) to where the approach
        # spur WOULD have ended (at the ir node on the other side).

        # A's il3 position (east inner edge):
        # The exit lane for corner 3 is built by corner=0 (exit_corner=(0-1)%4=3).
        # corner=0, angle=0:
        #   start = center + rotation(0) @ flip([lw/2, access+od]) = center + [access+od, lw/2]
        #   end   = center + rotation(0) @ flip([lw/2, od])        = center + [od, lw/2]
        #   StraightLane(end, start) → lane goes from [od, lw/2] outward
        # So il3 position (lane start, near intersection) = center + [od, lw/2]

        # B's ir1 position (west inner edge):
        # The approach for corner 1 (West): angle=90°
        #   rotation(90°) @ [lw/2, od] = [-od, lw/2]
        #   So ir1 entry = center + [-od, lw/2]...
        # Actually let's just compute the endpoint positions directly.

        # Corner 3 (East) for intersection A:
        angle_east = np.radians(90 * 3)  # 270°
        rot_east = np.array([
            [np.cos(angle_east), -np.sin(angle_east)],
            [np.sin(angle_east),  np.cos(angle_east)]
        ])

        # A's il3 exit position (where the exit spur would start from, near intersection):
        # From the exit code: corner=0 builds il3→o3.
        # The geometry uses: end = rotation(0°) @ flip([lw/2, od]) = [od, lw/2]
        # This is the START of the StraightLane (near intersection side).
        a_il3_pos = center_a + np.array([outer_distance, lane_width / 2])

        # B's ir1 approach end position (where approach spur ends, at intersection):
        # Corner 1 (West): angle=90°
        angle_west = np.radians(90 * 1)
        rot_west = np.array([
            [np.cos(angle_west), -np.sin(angle_west)],
            [np.sin(angle_west),  np.cos(angle_west)]
        ])
        # Approach end = center + rotation @ [lw/2, od]
        b_ir1_pos = center_b + rot_west @ np.array([lane_width / 2, outer_distance])

        # A→B connector: A_il3 → B_ir1
        net.add_lane(
            "A_il3", "B_ir1",
            StraightLane(a_il3_pos, b_ir1_pos,
                         line_types=[s, c], priority=2, speed_limit=10),
        )

        # B's il1 exit position (where B's west exit spur would start):
        # Corner 1 builds exit for il1 via corner=2 (exit_corner=(2-1)%4=1).
        # corner=2, angle=180°:
        #   start = center + rotation(180°) @ flip([lw/2, access+od])
        #   end   = center + rotation(180°) @ flip([lw/2, od])
        #   rotation(180°) @ flip([lw/2, od]) = rotation(180°) @ [od, lw/2]
        #       = [-od, -lw/2]
        # So B's il1 pos = center_b + [-od, -lw/2]
        b_il1_pos = center_b + np.array([-outer_distance, -lane_width / 2])

        # A's ir3 approach end position:
        # Corner 3 (East): angle=270°
        # Approach end = center + rotation(270°) @ [lw/2, od]
        #   rotation(270°) @ [lw/2, od] = [od, -lw/2]
        a_ir3_pos = center_a + rot_east @ np.array([lane_width / 2, outer_distance])

        # B→A connector: B_il1 → A_ir3
        net.add_lane(
            "B_il1", "A_ir3",
            StraightLane(b_il1_pos, a_ir3_pos,
                         line_types=[s, c], priority=2, speed_limit=10),
        )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    # ------------------------------------------------------------------
    # has_arrived — only on the 6 outer exit spurs, NOT on connectors
    # ------------------------------------------------------------------
    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
        if dst not in self.OUTER_EXIT_TARGETS:
            return False
        if "_il" not in src:
            return False
        return vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    # ------------------------------------------------------------------
    # _clear_vehicles — only remove non-controlled vehicles on outer exits
    # ------------------------------------------------------------------
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
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles or not is_leaving(vehicle)
        ]

    # ------------------------------------------------------------------
    # _reset — reads double_intersection_base_scenarios, no rotation
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        for i, vehicle in enumerate(self.controlled_vehicles):
            project_globals.after_is_arrived_flags[i] = False

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40

        all_scenarios = list(double_intersection_base_scenarios)

        if not all_scenarios:
            print("[DoubleIntersectionEnv._reset] WARNING: no scenarios loaded!")
            return

        chosen_scenario = random.choice(all_scenarios)

        # Place agents
        for i, (lane_key, destination, off) in enumerate(chosen_scenario["agents"]):
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

        # Place static vehicles with safety check (minimum distance = 10m)
        all_vehicles = self.road.vehicles
        controlled_count = len(self.controlled_vehicles)

        safe_static_scenario = []
        for i, (lane_key, destination, off) in enumerate(chosen_scenario["static"]):
            lane = self.road.network.get_lane(lane_key)
            position = np.array(lane.position(BASE_LONG + off, 0))

            too_close_to_agent = False
            for controlled_vehicle in self.controlled_vehicles:
                if np.linalg.norm(controlled_vehicle.position - position) < 10:
                    too_close_to_agent = True
                    break

            too_close_to_static = False
            for existing_lane_key, existing_dest, existing_off in safe_static_scenario:
                existing_lane = self.road.network.get_lane(existing_lane_key)
                existing_position = np.array(existing_lane.position(BASE_LONG + existing_off, 0))
                if np.linalg.norm(existing_position - position) < 10:
                    too_close_to_static = True
                    break

            if not too_close_to_agent and not too_close_to_static:
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

        print(f"[DoubleIntersectionEnv._reset] Scenario loaded")
        print(f"  Placed {len(safe_static_scenario)}/{len(chosen_scenario['static'])} static vehicles safely")


class MultiAgentDoubleIntersectionEnv(DoubleIntersectionEnv, MultiAgentIntersectionEnv):
    """Multi-agent wrapper — inherits MultiAgent defaults from
    MultiAgentIntersectionEnv and road geometry from DoubleIntersectionEnv."""
    pass
