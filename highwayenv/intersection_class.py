from __future__ import annotations

import random

import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from src.experiment.experiment_config import Experiment
from src.experiment.scenarios import base_complete_scenarios_3_cars
from highwayenv.CustomControlledVehicle import CustomControlledVehicle
from highwayenv.custom_action import action_factory
from src import project_globals


def rotate_lane_id(lane_id, rotation):
    """
    Rotate a lane ID clockwise by rotation steps (1-3)
    Examples: 'o0' -> 'o1' -> 'o2' -> 'o3'
    """
    if len(lane_id) < 2 or not lane_id[-1].isdigit():
        return lane_id

    prefix = lane_id[:-1]  # 'o', 'ir', 'il'
    direction = int(lane_id[-1])  # 0, 1, 2, 3

    new_direction = (direction + rotation) % 4
    return prefix + str(new_direction)


def rotate_scenario_clockwise(scenario, rotation):
    """
    Rotate an entire scenario clockwise by rotation steps
    """
    rotated_scenario = []
    for lane_key, destination, offset in scenario:
        # Rotate lane_key (can be tuple like ('o0', 'ir0', 0))
        if isinstance(lane_key, tuple):
            rotated_lane = tuple(
                rotate_lane_id(part, rotation) if isinstance(part, str) else part
                for part in lane_key
            )
        else:
            rotated_lane = rotate_lane_id(lane_key, rotation)

        # Rotate destination
        rotated_dest = rotate_lane_id(destination, rotation)

        rotated_scenario.append((rotated_lane, rotated_dest, offset))

    return rotated_scenario


class IntersectionEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        This function overrides the define_spaces function of the base class AbstractEnv, in order to allow calling our
        own "action_factory" function from custom_action.py instead of the built-in action.py.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    # def _reward(self, action: int) -> float:
    #     """Aggregated reward, for cooperative agents."""
    #     return sum(
    #         self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
    #     ) / len(self.controlled_vehicles)

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        mean_reward_type = False

        # Work with vehicles that haven't arrived yet
        # TODO: do we want to filter only not arrived? or do we need to filter not crashed also?
        active_vehicles = [
            vehicle for vehicle, flag in zip(self.controlled_vehicles, project_globals.after_is_arrived_flags)
            if not flag
        ]
        print(f"Active vehicles: {len(active_vehicles)}")

        if not active_vehicles:
            print("No active vehicles left. Reward: 0.0")
            return 0.0  # or some default value when all vehicles have arrived

        # TODO: do we always want to return one reward shared between all (non crashed) vehicles?
        if mean_reward_type:
            mean_reward = sum(
                self._agent_reward(vehicle) for vehicle in active_vehicles
            ) / len(active_vehicles)
            print("Reward:", mean_reward)
            return mean_reward
        else:
            min_reward = min(
                self._agent_reward(vehicle) for vehicle in active_vehicles
            )
            print("Min Reward:", min_reward)
            return min_reward


    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
                  / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, vehicle: Vehicle) -> float:
        """Enhanced reward with movement incentive."""
        rewards = self._agent_rewards(vehicle)
        if self.has_arrived(vehicle):
            reward = self.config["arrived_reward"]
        elif vehicle.crashed:
            reward = self.config["collision_reward"]
            print(f"Vehicle {vehicle} crashed! Reward: {reward}")
        else:
            if vehicle.speed > Experiment.FIXED_THROTTLE:
                reward = Experiment.HIGH_SPEED_REWARD
            else:
                reward = Experiment.STARVATION_REWARD
        return reward

    def _agent_rewards(self, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": 1,
            "arrived_reward": self.has_arrived(vehicle),
            "starvation_reward": True
        }

    def update_after_is_arrived_flags(self):
        for i, vehicle in enumerate(self.controlled_vehicles):
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                project_globals.after_is_arrived_flags[i] = True

    def _is_terminated(self) -> bool:

        self.update_after_is_arrived_flags()

        # Initialize arrived_vehicles set if not exists
        if not hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles = set()

        # Check for new arrivals
        for vehicle in self.controlled_vehicles:
            if self.has_arrived(vehicle) and vehicle not in self.arrived_vehicles:
                # New arrival
                vehicle.position = np.array(
                    [0.0 + len(self.arrived_vehicles) * 20.0, 1000 + len(self.arrived_vehicles) * 20.0],
                    dtype=np.float64)
                vehicle.target_speed = 0.0
                vehicle.MAX_SPEED = 0
                vehicle.is_arrived = True
                self.arrived_vehicles.add(vehicle)

        arrived_count = len(self.arrived_vehicles)

        # Immediate termination when all arrive
        if arrived_count == len(self.controlled_vehicles):
            return True

        # TODO: do we want to continue even if there was a crash?
        return any((vehicle.crashed for vehicle in self.controlled_vehicles) or all(
            vehicle.is_arrived for vehicle in self.controlled_vehicles))

    def get_observation(self):

        self.update_after_is_arrived_flags()

        obs = []

        for vehicle in self.controlled_vehicles:
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                obs.append([0.0, 0.0, 0.0, 0.0])
            else:
                obs.append([
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.velocity[0],
                    vehicle.velocity[1]
                ])

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:

        # reset after_is_arrived_flags
        for i, vehicle in enumerate(self.controlled_vehicles):
            project_globals.after_is_arrived_flags[i] = False

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40

        base_complete_scenarios = base_complete_scenarios_3_cars

        # Generate rotations for complete scenarios
        def rotate_complete_scenario(scenario, rotation):
            rotated_agents = [rotate_scenario_clockwise([agent], rotation)[0] for agent in scenario["agents"]]
            rotated_static = [rotate_scenario_clockwise([static], rotation)[0] for static in scenario["static"]]
            return {
                "agents": rotated_agents,
                "static": rotated_static
            }

        # Generate all 100 scenarios (25 × 4 rotations)
        all_scenarios = []
        for base_scenario in base_complete_scenarios:
            # Add original (0° rotation)
            all_scenarios.append(base_scenario)

            # Add 3 rotations (90°, 180°, 270°)
            for rotation in [1, 2, 3]:
                rotated_scenario = rotate_complete_scenario(base_scenario, rotation)
                all_scenarios.append(rotated_scenario)

        # Choose scenario (3 options: specific, random, serial)

        # 1. For random
        chosen_scenario = random.choice(all_scenarios)

        # 2. For specific scenarios (uncomment to use)
        # desired_scenario = 11  # Choose scenario 0-99
        # chosen_scenario = all_scenarios[desired_scenario]

        # 3. For serial cycling (uncomment to use)
        # if not hasattr(self, 'scenario_counter'):
        #     self.scenario_counter = 0
        # chosen_scenario = all_scenarios[self.scenario_counter % len(all_scenarios)]
        # self.scenario_counter += 1

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
                vehicle.route = [lane_key, (destination, 'ir' + destination[1:], 0)]

        # Place static vehicles with safety check (minimum distance = 10m)
        all_vehicles = self.road.vehicles
        controlled_count = len(self.controlled_vehicles)

        # Safety check: Filter out conflicting static vehicles
        safe_static_scenario = []
        for i, (lane_key, destination, off) in enumerate(chosen_scenario["static"]):
            lane = self.road.network.get_lane(lane_key)
            position = np.array(lane.position(BASE_LONG + off, 0))

            # Check distance from controlled vehicles
            too_close_to_agent = False
            for controlled_vehicle in self.controlled_vehicles:
                if np.linalg.norm(controlled_vehicle.position - position) < 10:  # Changed to 10m
                    too_close_to_agent = True
                    break

            # Check distance from other static vehicles
            too_close_to_static = False
            for existing_lane_key, existing_dest, existing_off in safe_static_scenario:
                existing_lane = self.road.network.get_lane(existing_lane_key)
                existing_position = np.array(existing_lane.position(BASE_LONG + existing_off, 0))

                if np.linalg.norm(existing_position - position) < 10:  # Changed to 10m minimum distance
                    too_close_to_static = True
                    break

            # Only add if safe
            if not too_close_to_agent and not too_close_to_static:
                safe_static_scenario.append((lane_key, destination, off))

        # Place the safe static vehicles
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
                vehicle.route = [lane_key, (destination, 'ir' + destination[1:], 0)]

        # Calculate scenario info for display
        scenario_index = all_scenarios.index(chosen_scenario)
        base_scenario_num = scenario_index // 4
        rotation_num = scenario_index % 4
        rotation_names = ["Original", "90° CW", "180° CW", "270° CW"]

        print(f"[IntersectionEnv._reset] Using scenario {scenario_index}/100")
        print(f"  Base scenario {base_scenario_num} ({rotation_names[rotation_num]})")
        print(f"  Placed {len(safe_static_scenario)}/{len(chosen_scenario['static'])} static vehicles safely")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
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
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
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
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10
                ),
            )
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        self.controlled_vehicles = []
        controlled_conf = self.config["controlled_cars"]
        other_conf = self.config["static_cars"]

        # Controlled vehicles
        for other_car in controlled_conf.values():
            lane = self.road.network.get_lane(other_car["start_lane"])
            controlled_vehicle = CustomControlledVehicle(
                self.road,
                lane.position(other_car["init_location"]["longitudinal"], other_car["init_location"]["lateral"]),
                speed=other_car["speed"],
                heading=lane.heading_at(other_car["init_location"]["longitudinal"]),
            )
            controlled_vehicle.color = other_car["color"]
            controlled_vehicle.plan_route_to(other_car["destination"])
            self.road.vehicles.append(controlled_vehicle)
            self.controlled_vehicles.append(controlled_vehicle)

        # Regular Vehicle: Constant speed
        for other_car in other_conf.values():
            lane = self.road.network.get_lane(other_car["start_lane"])
            other_vehicle = Vehicle(
                self.road,
                lane.position(other_car["init_location"]["longitudinal"], other_car["init_location"]["lateral"]),
                speed=other_car["speed"],
                heading=lane.heading_at(other_car["init_location"]["longitudinal"]),
            )
            self.road.vehicles.append(other_vehicle)

        for v in self.road.vehicles:
            if (
                    v is not controlled_vehicle
                    and np.linalg.norm(v.position - controlled_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)

    def _spawn_vehicle(
            self,
            longitudinal: float = 0,
            position_deviation: float = 1.0,
            speed_deviation: float = 1.0,
            spawn_probability: float = 0.6,
            go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                    longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
                            and "o" in vehicle.lane_index[1]
                            and vehicle.lane.local_coordinates(vehicle.position)[0]
                            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
               or not (is_leaving(vehicle))
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
                "il" in vehicle.lane_index[0]
                and "o" in vehicle.lane_index[1]
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config


class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
            }
        )
        #return
        return config