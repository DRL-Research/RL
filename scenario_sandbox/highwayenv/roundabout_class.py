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
    rotate_scenario_clockwise,
)
from src.experiment.scenarios import roundabout_base_scenarios
from src import project_globals


class RoundaboutEnv(IntersectionEnv):
    """
    A traffic-circle (roundabout) environment that reuses all of
    IntersectionEnv's reward / termination / vehicle logic but builds
    a circular ring road instead of a 4-way intersection.

    Road topology
    =============
    4 approach spurs   :  o0→ir0, o1→ir1, o2→ir2, o3→ir3   (StraightLane, 100 m)
    4 ring arcs (CCW)  :  ir0→ir1, ir1→ir2, ir2→ir3, ir3→ir0  (CircularLane)
    4 exit spurs       :  ir0→o3, ir1→o0, ir2→o1, ir3→o2   (StraightLane, 100 m)

    The node naming mirrors the intersection so that:
      - rotate_lane_id / rotate_scenario_clockwise work unchanged
      - scenario tuple format is identical: (('o0','ir0',0), 'o2', offset)
      - has_arrived checks exit spurs the same way as the intersection
    """

    def _make_road(self) -> None:
        """
        Build a 4-arm roundabout.

        Topology per arm i (0=South, 1=West, 2=North, 3=East):
          o{i}  →(StraightLane 100 m)→  ir{i}  approach spur
          ir{i} →(CircularLane  90°)→   ir{j}  ring arc  (j = next arm, CCW on screen)
          ir{i} →(StraightLane   5 m)→  il{i}  exit connector  (creates the il node for has_arrived)
          il{i} →(StraightLane 100 m)→  o{i}   exit spur

        Lateral offsets keep approach and exit physically separated so vehicles do not collide:
          Approach:  offset = +tangent * lw/2  (right of inward direction = towards ring CW)
          Exit:      offset = -tangent * lw/2  (right of outward direction = away from ring CW)
        where tangent = rotation @ [1, 0]  (CCW perpendicular to outward).

        Ring arc direction:
          clockwise=True → direction=+1 → phi increases → traces CCW on screen (y-down).
          Maps:  South(90°) → West(180°) → North(270°) → East(360°) → South(90°+360°)
          This is the correct direction for right-hand-traffic roundabouts.
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        radius = 20.0
        access_length = 100.0
        connector_length = 5.0   # must be < has_arrived threshold (25 m)
        speed_limit = 10

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        center = np.array([0.0, 0.0])

        for corner in range(4):
            rot_angle = np.radians(90 * corner)
            rotation = np.array([
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle),  np.cos(rot_angle)],
            ])

            outward = rotation @ np.array([0.0, 1.0])   # from centre toward arm
            tangent  = rotation @ np.array([1.0, 0.0])  # CCW perpendicular to outward

            junction = center + outward * radius

            # Lateral offsets: lw/2 each side so spurs never overlap physically
            approach_shift = tangent * (lane_width / 2)
            exit_shift     = -tangent * (lane_width / 2)

            # Angle of this junction from the +x axis (for CircularLane phase parameters)
            # corner 0 → 90°, corner 1 → 180°, corner 2 → 270°, corner 3 → 0° / 360°
            ang_junction = np.radians((90 * (corner + 1)) % 360)

            # ── Approach spur: o{i} → ir{i} ─────────────────────────────────────
            approach_outer = junction + outward * access_length + approach_shift
            approach_inner = junction + approach_shift
            net.add_lane(
                "o" + str(corner), "ir" + str(corner),
                StraightLane(approach_outer, approach_inner,
                             line_types=[s, c], priority=1, speed_limit=speed_limit),
            )

            # ── Ring arc: ir{i} → ir{prev} ──────────────────────────────────
            # CCW roundabout (anti-clockwise from above, correct for right-hand traffic):
            #   S(ir0) → E(ir3) → N(ir2) → W(ir1) → S(ir0)
            # corresponds to corner → (corner-1)%4 in the arm numbering.
            #
            # clockwise=False (direction=-1): phi decreases each arc by pi/2.
            # Junction i angle (from +x): arctan2(outward[1], outward[0])
            #   corner0(S)=90°, corner1(W)=180°, corner2(N)=-90°, corner3(E)=0°
            # Arc S→E: 90°→0° (decrease 90°) ✓
            # Arc E→N: 0°→-90° ✓   Arc N→W: -90°→-180° ✓   Arc W→S: 180°→90° ✓
            prev_corner = (corner - 1) % 4
            ang_start = np.arctan2(outward[1], outward[0])   # angle of this junction
            ang_end   = ang_start - np.pi / 2                 # 90° CCW step (decreasing phi)

            net.add_lane(
                "ir" + str(corner), "ir" + str(prev_corner),
                CircularLane(
                    center, radius,
                    ang_start, ang_end,
                    clockwise=False,         # direction=-1 → decreasing phi → CCW on screen
                    line_types=[n, c], priority=2, speed_limit=speed_limit,
                ),
            )

            # ── Exit connector: ir{i} → il{i} ───────────────────────────────────
            exit_conn_start = junction + exit_shift
            exit_conn_end   = junction + outward * connector_length + exit_shift
            net.add_lane(
                "ir" + str(corner), "il" + str(corner),
                StraightLane(exit_conn_start, exit_conn_end,
                             line_types=[n, n], priority=1, speed_limit=speed_limit),
            )

            # ── Exit spur: il{i} → o{i} ─────────────────────────────────────────
            # has_arrived() triggers at ≥ 25 m along this 100 m lane.
            exit_outer = exit_conn_end + outward * access_length
            net.add_lane(
                "il" + str(corner), "o" + str(corner),
                StraightLane(exit_conn_end, exit_outer,
                             line_types=[n, c], priority=0, speed_limit=speed_limit),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    # ------------------------------------------------------------------
    # _reset — identical logic to IntersectionEnv but reads from
    #          roundabout_base_scenarios instead of base_complete_scenarios_3_cars
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        # reset after_is_arrived_flags
        for i, vehicle in enumerate(self.controlled_vehicles):
            project_globals.after_is_arrived_flags[i] = False

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40

        base_complete_scenarios = roundabout_base_scenarios

        # Generate rotations for complete scenarios
        def rotate_complete_scenario(scenario, rotation):
            rotated_agents = [rotate_scenario_clockwise([agent], rotation)[0] for agent in scenario["agents"]]
            rotated_static = [rotate_scenario_clockwise([static], rotation)[0] for static in scenario["static"]]
            return {
                "agents": rotated_agents,
                "static": rotated_static
            }

        all_scenarios = []
        for base_scenario in base_complete_scenarios:
            all_scenarios.append(base_scenario)
            for rotation in [1, 2, 3]:
                rotated_scenario = rotate_complete_scenario(base_scenario, rotation)
                all_scenarios.append(rotated_scenario)

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

        # Place static vehicles with safety check
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

        scenario_index = all_scenarios.index(chosen_scenario)
        base_scenario_num = scenario_index // 4
        rotation_num = scenario_index % 4
        rotation_names = ["Original", "90° CW", "180° CW", "270° CW"]

        print(f"[RoundaboutEnv._reset] Using scenario {scenario_index}/{len(all_scenarios)}")
        print(f"  Base scenario {base_scenario_num} ({rotation_names[rotation_num]})")
        print(f"  Placed {len(safe_static_scenario)}/{len(chosen_scenario['static'])} static vehicles safely")

    # ------------------------------------------------------------------
    # has_arrived / _clear_vehicles — reuse the same il→o check from intersection
    # The roundabout uses the same "il{i}" → "o{i}" exit spur naming.
    # ------------------------------------------------------------------
    # (inherited from IntersectionEnv — no override needed)


class MultiAgentRoundaboutEnv(RoundaboutEnv, MultiAgentIntersectionEnv):
    """Multi-agent wrapper for the roundabout — inherits MultiAgent defaults
    from MultiAgentIntersectionEnv and road geometry from RoundaboutEnv."""
    pass
