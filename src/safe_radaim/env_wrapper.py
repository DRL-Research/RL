"""Environment wrapper exposing the state/action spaces expected by SafeR-ADAIM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from highwayenv.custom_action import ExpectedVelocityMultiAgentAction


@dataclass
class RewardBreakdown:
    efficiency: float
    comfort: float
    safety: float


class SafeIntersectionEnv(gym.Env):
    """Wrap ``IntersectionEnv`` to expose SafeR-ADAIM signals.

    The wrapper augments the original environment with:

    * a continuous action space representing desired velocities for all
      controlled vehicles;
    * a global observation vector with distances-to-exit, speeds and turn
      intentions encoded for every controlled vehicle;
    * dense and sparse cost terms derived from vehicle spacing and
      collisions, matching the definitions from the paper.
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        base_env: gym.Env,
        *,
        min_velocity: float,
        max_velocity: float,
        epsilon_distance_cost: float,
        epsilon_collision_cost: float,
        epsilon_v: float,
        epsilon_t: float,
        epsilon_pass_single: float,
        epsilon_pass_all: float,
        epsilon_a: float,
        epsilon_j: float,
        safe_distance: float,
    ) -> None:
        super().__init__()
        self.base_env = base_env
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.epsilon_distance_cost = epsilon_distance_cost
        self.epsilon_collision_cost = epsilon_collision_cost
        self.epsilon_v = epsilon_v
        self.epsilon_t = epsilon_t
        self.epsilon_pass_single = epsilon_pass_single
        self.epsilon_pass_all = epsilon_pass_all
        self.epsilon_a = epsilon_a
        self.epsilon_j = epsilon_j
        self.safe_distance = safe_distance
        policy_freq = float(self.base_env.config.get("policy_frequency", 5))
        self._dt = 1.0 / policy_freq if policy_freq > 0 else 0.1

        if not hasattr(self.base_env, "config"):
            raise TypeError("Expected a highway-env style environment with 'config'")

        # swap-in the expected velocity action model
        action_config = {
            "type": "ExpectedVelocityMultiAgentAction",
            "min_velocity": min_velocity,
            "max_velocity": max_velocity,
        }
        self._action_config = action_config
        self.base_env.config = dict(self.base_env.config)
        self.base_env.config["action"] = action_config
        self.base_env.action_type = ExpectedVelocityMultiAgentAction(self.base_env, action_config)

        self._controlled: List = list(self.base_env.controlled_vehicles)
        self._num_controlled = len(self._controlled)

        self.action_space = spaces.Box(
            low=np.full(self._num_controlled, min_velocity, dtype=np.float32),
            high=np.full(self._num_controlled, max_velocity, dtype=np.float32),
            dtype=np.float32,
        )

        # per-vehicle features: distance, speed, one-hot turn intention (3)
        self._features_per_vehicle = 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._num_controlled * self._features_per_vehicle,),
            dtype=np.float32,
        )

        self._previous_speeds = np.zeros(self._num_controlled, dtype=np.float32)
        self._previous_accels = np.zeros(self._num_controlled, dtype=np.float32)

    # ------------------------------------------------------------------
    # gym.Env API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.base_env.action_type = ExpectedVelocityMultiAgentAction(
            self.base_env, self._action_config
        )
        self._controlled = list(self.base_env.controlled_vehicles)
        self._num_controlled = len(self._controlled)
        self._previous_speeds = self._extract_speeds()
        self._previous_accels = np.zeros_like(self._previous_speeds)
        return self._build_observation(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self._num_controlled,), "Action must match number of controlled vehicles"

        obs, base_reward, terminated, truncated, info = self.base_env.step(action)
        current_speeds = self._extract_speeds()
        accelerations = (current_speeds - self._previous_speeds) / self._dt
        jerk = (accelerations - self._previous_accels) / self._dt

        dense_cost = self._compute_risk_cost()
        collision_cost = self._compute_collision_cost(info)
        total_cost = dense_cost + collision_cost

        reward_breakdown = self._compute_reward_components(
            current_speeds=current_speeds,
            accelerations=accelerations,
            jerk=jerk,
            dense_cost=dense_cost + collision_cost,
        )
        reward = reward_breakdown.efficiency + reward_breakdown.comfort + reward_breakdown.safety

        self._previous_speeds = current_speeds
        self._previous_accels = accelerations

        info = dict(info or {})
        info.update(
            {
                "reward_breakdown": reward_breakdown,
                "risk_cost": float(total_cost),
                "dense_cost": float(dense_cost),
                "collision_cost": float(collision_cost),
                "base_reward": base_reward,
            }
        )

        return self._build_observation(), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()

    # ------------------------------------------------------------------
    # feature engineering
    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        features: List[float] = []
        for vehicle in self._controlled:
            distance = self._distance_to_exit(vehicle)
            speed = float(np.linalg.norm(vehicle.velocity))
            intention = self._turn_intention(vehicle)
            features.extend([distance, speed, *intention])
        return np.asarray(features, dtype=np.float32)

    def _extract_speeds(self) -> np.ndarray:
        return np.array([np.linalg.norm(vehicle.velocity) for vehicle in self._controlled], dtype=np.float32)

    def _distance_to_exit(self, vehicle) -> float:
        lane = self.base_env.road.network.get_lane(vehicle.lane_index)
        local_s, _ = lane.local_coordinates(vehicle.position)
        remaining = max(lane.length - local_s, 0.0)
        if getattr(vehicle, "route", None):
            for step in vehicle.route[1:]:
                lane_key = step if isinstance(step, tuple) else step
                try:
                    next_lane = self.base_env.road.network.get_lane(lane_key)
                except KeyError:
                    continue
                remaining += getattr(next_lane, "length", 0.0)
        return float(remaining)

    def _turn_intention(self, vehicle) -> Tuple[float, float, float]:
        route = getattr(vehicle, "route", None)
        if not route or len(route) < 2:
            return (0.0, 1.0, 0.0)  # assume straight
        current_lane = route[0]
        next_lane = route[1]
        current_dir = self._lane_direction(current_lane)
        next_dir = self._lane_direction(next_lane)
        if current_dir is None or next_dir is None:
            return (0.0, 1.0, 0.0)
        diff = (next_dir - current_dir) % 4
        if diff == 0:
            return (0.0, 1.0, 0.0)
        if diff == 1:
            return (1.0, 0.0, 0.0)  # left
        if diff == 3:
            return (0.0, 0.0, 1.0)  # right
        return (0.0, 1.0, 0.0)

    @staticmethod
    def _lane_direction(lane_key) -> int | None:
        if isinstance(lane_key, tuple):
            lane_key = lane_key[0]
        if isinstance(lane_key, str) and lane_key[-1].isdigit():
            return int(lane_key[-1])
        return None

    # ------------------------------------------------------------------
    # reward/cost computation helpers
    # ------------------------------------------------------------------
    def _compute_risk_cost(self) -> float:
        cost = 0.0
        for i in range(self._num_controlled):
            veh_i = self._controlled[i]
            pos_i = np.asarray(veh_i.position)
            for j in range(i + 1, self._num_controlled):
                veh_j = self._controlled[j]
                pos_j = np.asarray(veh_j.position)
                if np.linalg.norm(pos_i - pos_j) < self.safe_distance:
                    cost += self.epsilon_distance_cost
        return cost

    def _compute_collision_cost(self, info: Dict) -> float:
        if info.get("crashed", False):
            return self.epsilon_collision_cost
        for vehicle in self._controlled:
            if getattr(vehicle, "crashed", False):
                return self.epsilon_collision_cost
        return 0.0

    def _compute_reward_components(
        self,
        *,
        current_speeds: np.ndarray,
        accelerations: np.ndarray,
        jerk: np.ndarray,
        dense_cost: float,
    ) -> RewardBreakdown:
        step_efficiency = float(np.sum(self.epsilon_v * current_speeds) - self.epsilon_t)

        single_pass_bonus = 0.0
        passed_flags = [getattr(vehicle, "is_arrived", False) for vehicle in self._controlled]
        if any(passed_flags):
            single_pass_bonus = self.epsilon_pass_single * sum(passed_flags)
        all_pass_bonus = self.epsilon_pass_all if all(passed_flags) else 0.0

        efficiency = step_efficiency + single_pass_bonus + all_pass_bonus
        comfort = float(np.sum(self.epsilon_a * accelerations + self.epsilon_j * jerk))
        safety = -dense_cost
        return RewardBreakdown(efficiency=efficiency, comfort=comfort, safety=safety)

    # ------------------------------------------------------------------
    # convenience API
    # ------------------------------------------------------------------
    @property
    def num_controlled(self) -> int:
        return self._num_controlled

    def controlled_vehicle_ids(self) -> Iterable[int]:
        return range(self._num_controlled)

    def unwrap(self) -> gym.Env:
        return self.base_env

