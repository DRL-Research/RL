from __future__ import annotations

import random

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from highwayenv.intersection_class import IntersectionEnv, MultiAgentIntersectionEnv
from src import project_globals
from src.composable_layout import (
    lane_name,
    normalize_layout_config,
    visible_approaches,
)
from src.experiment.scenarios import composable_base_scenarios


def _intersection_outer_distance() -> float:
    lane_width = AbstractLane.DEFAULT_WIDTH
    right_turn_radius = lane_width + 5
    return right_turn_radius + lane_width / 2


def _build_intersection_module(
    net: RoadNetwork,
    slot: str,
    center: np.ndarray,
    *,
    skip_exit: int | None = None,
    skip_approach: int | None = None,
) -> None:
    lane_width = AbstractLane.DEFAULT_WIDTH
    right_turn_radius = lane_width + 5
    left_turn_radius = right_turn_radius + lane_width
    outer_distance = right_turn_radius + lane_width / 2
    access_length = 100.0

    n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

    for corner in range(4):
        angle = np.radians(90 * corner)
        is_horizontal = corner % 2
        priority = 3 if is_horizontal else 1
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        o = lane_name(slot, "o", corner)
        ir = lane_name(slot, "ir", corner)
        il_right = lane_name(slot, "il", (corner - 1) % 4)
        il_left = lane_name(slot, "il", (corner + 1) % 4)
        il_straight = lane_name(slot, "il", (corner + 2) % 4)

        if corner != skip_approach:
            start = center + rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = center + rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                o,
                ir,
                StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10),
            )

        r_center = center + rotation @ np.array([outer_distance, outer_distance])
        net.add_lane(
            ir,
            il_right,
            CircularLane(
                r_center,
                right_turn_radius,
                angle + np.radians(180),
                angle + np.radians(270),
                line_types=[n, c],
                priority=priority,
                speed_limit=10,
            ),
        )

        l_center = center + rotation @ np.array(
            [-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]
        )
        net.add_lane(
            ir,
            il_left,
            CircularLane(
                l_center,
                left_turn_radius,
                angle + np.radians(0),
                angle + np.radians(-90),
                clockwise=False,
                line_types=[n, n],
                priority=priority - 1,
                speed_limit=10,
            ),
        )

        start = center + rotation @ np.array([lane_width / 2, outer_distance])
        end = center + rotation @ np.array([lane_width / 2, -outer_distance])
        net.add_lane(
            ir,
            il_straight,
            StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10),
        )

        exit_corner = (corner - 1) % 4
        if exit_corner != skip_exit:
            il_exit = lane_name(slot, "il", exit_corner)
            o_exit = lane_name(slot, "o", exit_corner)
            start = center + rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = center + rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                il_exit,
                o_exit,
                StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10),
            )


def _build_roundabout_module(
    net: RoadNetwork,
    slot: str,
    center: np.ndarray,
    *,
    skip_exit: int | None = None,
    skip_approach: int | None = None,
) -> None:
    lane_width = AbstractLane.DEFAULT_WIDTH
    radius = 20.0
    access_length = 100.0
    exit_connector_length = 5.0
    speed_limit = 10

    n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

    for corner in range(4):
        rot_angle = np.radians(90 * corner)
        rotation = np.array(
            [[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]]
        )

        outward = rotation @ np.array([0.0, 1.0])
        tangent = rotation @ np.array([1.0, 0.0])
        junction = center + outward * radius

        approach_shift = tangent * (lane_width / 2)
        exit_shift = -tangent * (lane_width / 2)

        if corner != skip_approach:
            approach_outer = junction + outward * access_length + approach_shift
            approach_inner = junction + approach_shift
            net.add_lane(
                lane_name(slot, "o", corner),
                lane_name(slot, "ir", corner),
                StraightLane(
                    approach_outer,
                    approach_inner,
                    line_types=[s, c],
                    priority=1,
                    speed_limit=speed_limit,
                ),
            )

        prev_corner = (corner - 1) % 4
        ang_start = np.arctan2(outward[1], outward[0])
        ang_end = ang_start - np.pi / 2
        net.add_lane(
            lane_name(slot, "ir", corner),
            lane_name(slot, "ir", prev_corner),
            CircularLane(
                center,
                radius,
                ang_start,
                ang_end,
                clockwise=False,
                line_types=[n, c],
                priority=2,
                speed_limit=speed_limit,
            ),
        )

        exit_conn_start = junction + exit_shift
        exit_conn_end = junction + outward * exit_connector_length + exit_shift
        net.add_lane(
            lane_name(slot, "ir", corner),
            lane_name(slot, "il", corner),
            StraightLane(
                exit_conn_start,
                exit_conn_end,
                line_types=[n, n],
                priority=1,
                speed_limit=speed_limit,
            ),
        )

        if corner != skip_exit:
            exit_outer = exit_conn_end + outward * access_length
            net.add_lane(
                lane_name(slot, "il", corner),
                lane_name(slot, "o", corner),
                StraightLane(
                    exit_conn_end,
                    exit_outer,
                    line_types=[n, c],
                    priority=0,
                    speed_limit=speed_limit,
                ),
            )


def _module_entry_position(module_type: str, center: np.ndarray, side: str) -> np.ndarray:
    lane_width = AbstractLane.DEFAULT_WIDTH
    if module_type == "intersection":
        outer_distance = _intersection_outer_distance()
        if side == "east":
            return center + np.array([outer_distance, -lane_width / 2])
        return center + np.array([-outer_distance, lane_width / 2])

    radius = 20.0
    if side == "east":
        return center + np.array([radius, -lane_width / 2])
    return center + np.array([-radius, lane_width / 2])


def _module_exit_position(module_type: str, center: np.ndarray, side: str) -> np.ndarray:
    lane_width = AbstractLane.DEFAULT_WIDTH
    if module_type == "intersection":
        outer_distance = _intersection_outer_distance()
        if side == "east":
            return center + np.array([outer_distance, lane_width / 2])
        return center + np.array([-outer_distance, -lane_width / 2])

    radius = 20.0
    exit_connector_length = 5.0
    if side == "east":
        return center + np.array([radius + exit_connector_length, lane_width / 2])
    return center + np.array([-radius - exit_connector_length, -lane_width / 2])


def _module_center_positions(layout_config: dict, connector_length: float, single_slot_offset: float) -> dict[str, np.ndarray]:
    layout = normalize_layout_config(layout_config)

    if layout["left"] != "empty" and layout["right"] != "empty":
        left_exit = _module_exit_position(layout["left"], np.array([0.0, 0.0]), "east")[0]
        right_entry = _module_entry_position(layout["right"], np.array([0.0, 0.0]), "west")[0]
        half_sep = (connector_length - right_entry + left_exit) / 2
        return {
            "left": np.array([-half_sep, 0.0]),
            "right": np.array([half_sep, 0.0]),
        }

    if layout["left"] != "empty":
        return {"left": np.array([-single_slot_offset, 0.0])}
    return {"right": np.array([single_slot_offset, 0.0])}


class ComposableRoadEnv(IntersectionEnv):
    def _layout(self) -> dict[str, str]:
        return normalize_layout_config(self.config.get("layout_config"))

    def _outer_exit_targets(self) -> set[str]:
        return set(visible_approaches(self._layout()))

    def _make_road(self) -> None:
        layout = self._layout()
        connector_length = self.config.get("connector_length", 80.0)
        single_slot_offset = self.config.get("single_slot_offset", 65.0)

        centers = _module_center_positions(layout, connector_length, single_slot_offset)
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        for slot in ("left", "right"):
            module_type = layout[slot]
            if module_type == "empty":
                continue

            connected = layout["left"] != "empty" and layout["right"] != "empty"
            connected_corner = 3 if slot == "left" else 1
            skip_exit = connected_corner if connected else None
            skip_approach = connected_corner if connected else None

            if module_type == "intersection":
                _build_intersection_module(
                    net,
                    slot,
                    centers[slot],
                    skip_exit=skip_exit,
                    skip_approach=skip_approach,
                )
            else:
                _build_roundabout_module(
                    net,
                    slot,
                    centers[slot],
                    skip_exit=skip_exit,
                    skip_approach=skip_approach,
                )

        if layout["left"] != "empty" and layout["right"] != "empty":
            left_exit_pos = _module_exit_position(layout["left"], centers["left"], "east")
            right_entry_pos = _module_entry_position(layout["right"], centers["right"], "west")
            net.add_lane(
                lane_name("left", "il", 3),
                lane_name("right", "ir", 1),
                StraightLane(
                    left_exit_pos,
                    right_entry_pos,
                    line_types=[s, c],
                    priority=2,
                    speed_limit=10,
                ),
            )

            right_exit_pos = _module_exit_position(layout["right"], centers["right"], "west")
            left_entry_pos = _module_entry_position(layout["left"], centers["left"], "east")
            net.add_lane(
                lane_name("right", "il", 1),
                lane_name("left", "ir", 3),
                StraightLane(
                    right_exit_pos,
                    left_entry_pos,
                    line_types=[s, c],
                    priority=2,
                    speed_limit=10,
                ),
            )

        self.road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
        if dst not in self._outer_exit_targets():
            return False
        if "_il" not in src:
            return False
        return vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _clear_vehicles(self) -> None:
        outer_exit_targets = self._outer_exit_targets()

        def is_leaving(vehicle):
            src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
            if dst not in outer_exit_targets:
                return False
            if "_il" not in src:
                return False
            return vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - 4 * vehicle.LENGTH

        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles or not is_leaving(vehicle)
        ]

    def _reset(self) -> None:
        for i, vehicle in enumerate(self.controlled_vehicles):
            project_globals.after_is_arrived_flags[i] = False

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, "arrived_vehicles"):
            self.arrived_vehicles.clear()

        base_long = 40
        all_scenarios = list(composable_base_scenarios)

        if not all_scenarios:
            print("[ComposableRoadEnv._reset] WARNING: no scenarios loaded!")
            return

        chosen_scenario = random.choice(all_scenarios)

        for i, (lane_key, destination, offset) in enumerate(chosen_scenario["agents"]):
            vehicle = self.controlled_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(base_long + offset, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, "plan_route_to"):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]

        all_vehicles = self.road.vehicles
        controlled_count = len(self.controlled_vehicles)

        safe_static_scenario = []
        for lane_key, destination, offset in chosen_scenario["static"]:
            lane = self.road.network.get_lane(lane_key)
            position = np.array(lane.position(base_long + offset, 0))

            too_close_to_agent = False
            for controlled_vehicle in self.controlled_vehicles:
                if np.linalg.norm(controlled_vehicle.position - position) < 10:
                    too_close_to_agent = True
                    break

            too_close_to_static = False
            for existing_lane_key, _existing_dest, existing_offset in safe_static_scenario:
                existing_lane = self.road.network.get_lane(existing_lane_key)
                existing_position = np.array(existing_lane.position(base_long + existing_offset, 0))
                if np.linalg.norm(existing_position - position) < 10:
                    too_close_to_static = True
                    break

            if not too_close_to_agent and not too_close_to_static:
                safe_static_scenario.append((lane_key, destination, offset))

        for i in range(controlled_count, min(len(all_vehicles), controlled_count + len(safe_static_scenario))):
            static_index = i - controlled_count
            if static_index >= len(safe_static_scenario):
                break

            lane_key, destination, offset = safe_static_scenario[static_index]
            vehicle = all_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(base_long + offset, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, "plan_route_to"):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]

        layout = self._layout()
        print(f"[ComposableRoadEnv._reset] Scenario loaded")
        print(f"  Layout: left={layout['left']}  right={layout['right']}")
        print(f"  Placed {len(safe_static_scenario)}/{len(chosen_scenario['static'])} static vehicles safely")


class MultiAgentComposableRoadEnv(ComposableRoadEnv, MultiAgentIntersectionEnv):
    pass
