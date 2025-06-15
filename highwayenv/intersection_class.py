from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
import gym

from highwayenv.CustomControlledVehicle import CustomControlledVehicle
from highwayenv.custom_action import action_factory
from src import project_globals


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


    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        mean_reward_type = True

        # Filter vehicles that haven't arrived yet
        active_vehicles = [
            vehicle for vehicle, flag in zip(self.controlled_vehicles, project_globals.after_is_arrived_flags)
            if not flag
        ]
        print(f"Active vehicles: {len(active_vehicles)}")


        if not active_vehicles:
            print("No active vehicles left. Reward: 0.0")
            return 0.0  # or some default value when all vehicles have arrived

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

        # mean_reward_type = True
        #
        # if mean_reward_type:
        #     mean_reward = sum(
        #         self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
        #     ) / len(self.controlled_vehicles)
        #     print("Reward:", mean_reward)
        #     return mean_reward
        # else:
        #     min_reward = min(
        #         self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
        #     )
        #     print("Min Reward:", min_reward)
        #     return min_reward

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
        # rewards = self._agent_rewards(vehicle)
        # reward = sum(
        #    self.config.get(name, 0) * reward for name, reward in rewards.items()
        # )
        if self.has_arrived(vehicle):
            reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else 0
        elif vehicle.crashed:
            reward = self.config["collision_reward"]
        else: # Movement incentives - only for vehicles that haven't arrived
            if vehicle.speed > 1.0:  # Moving at least 1 m/s
                reward = self.config["high_speed_reward"]  # Small bonus for moving
            else:
                reward = self.config["starvation_reward"] # Penalty for being stationary
        return reward

    def _agent_rewards(self, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        # scaled_speed = utils.lmap(
        #     vehicle.speed, self.config["reward_speed_range"], [0, 1]
        # )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": 1,
            "arrived_reward": self.has_arrived(vehicle),
            "starvation_reward": True
            # "on_road_reward": vehicle.on_road,
            # "high_speed_reward": np.clip(scaled_speed, 0, 1),
        }

    def update_after_is_arrived_flags(self):
        for i, vehicle in enumerate(self.controlled_vehicles):
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                project_globals.after_is_arrived_flags[i] = True

    def _is_terminated(self) -> bool:

        self.update_after_is_arrived_flags()
        # print(self.after_is_arrived_flags)

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
                #print(f"Vehicle moved to parking area")

        arrived_count = len(self.arrived_vehicles)
        #print(f"Arrived: {arrived_count}/{len(self.controlled_vehicles)}")

        # Immediate termination when all arrive
        if arrived_count == len(self.controlled_vehicles):
            #print(f"SUCCESS: All {len(self.controlled_vehicles)} vehicles completed!")
            return True

        return any((vehicle.crashed for vehicle in self.controlled_vehicles) or all(vehicle.is_arrived for vehicle in self.controlled_vehicles))


    def get_observation(self): # TODO: never reaching this code

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
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        # reset after_is_arrived_flags
        for i, vehicle in enumerate(self.controlled_vehicles):
            project_globals.after_is_arrived_flags[i] = False
        # print(f"Reset the after_is_arrived_flags: {self.after_is_arrived_flags}")


    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

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

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        # for t in range(n_vehicles - 1):
        #     self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
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
            controlled_vehicle.color = other_car["color"]   # Set color to green (RGB format)
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

        for v in self.road.vehicles:  # Prevent early collisions
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
               or not (is_leaving(vehicle))  #or vehicle.route is None)
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
        return config