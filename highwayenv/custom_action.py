from __future__ import annotations

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import (
    ActionType,
    ContinuousAction,
    DiscreteAction,
    DiscreteMetaAction,
    MultiAgentAction,
)

import functools
import itertools
from typing import TYPE_CHECKING, Callable, Iterable, Union

import numpy as np
from gymnasium import spaces

from highway_env import utils
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle


class CustomDiscreteAction(ActionType):
    def __init__(
        self,
        env: AbstractEnv,
        actions=None,
        target_speeds: Vector | None = None,
        **kwargs,
    ) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param target_speeds: the list of speeds the vehicle is able to track
        """
        super().__init__(env)
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else MDPVehicle.DEFAULT_TARGET_SPEEDS
        )
        if actions is None:
            actions = {0: "SLOWER", 1: "FASTER"} #, 2: "SUPER SLOWER", 3: "SUPER FASTER"}
        self.actions = actions
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: int | np.ndarray) -> None:
        self.controlled_vehicle.act(self.actions[int(action)])

    def get_available_actions(self) -> list[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        actions = []
        if self.controlled_vehicle.speed_index < self.controlled_vehicle.target_speeds.size - 1:
            actions.append(self.actions_indexes["FASTER"])
        if self.controlled_vehicle.speed_index > 0:
            actions.append(self.actions_indexes["SLOWER"])
        return actions


def action_factory(env: AbstractEnv, config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    if config["type"] == "DiscreteAction":
        return DiscreteAction(env, **config)
    if config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    if config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    if config["type"] == "CustomDiscreteAction":
        return CustomDiscreteAction(env, **config)
    if config["type"] == "CustomMultiAgentAction":
        return CustomMultiAgentAction(env, **config)
    if config["type"] == "ExpectedVelocityMultiAgentAction":
        return ExpectedVelocityMultiAgentAction(env, **config)
    raise ValueError("Unknown action type")


class CustomMultiAgentAction(ActionType):
    def __init__(self, env: AbstractEnv, action_config: dict, **kwargs) -> None:
        """
        Create a multi-agent version of CustomDiscreteAction.

        :param env: the environment
        :param action_config: the configuration for each agent's action
        """
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []

        # Instantiate a CustomDiscreteAction for each controlled vehicle
        for vehicle in self.env.controlled_vehicles:
            action_type = CustomDiscreteAction(env, **self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        """
        Return a tuple space where each element is the action space of a specific agent.
        """
        return spaces.Tuple(
            [action_type.space() for action_type in self.agents_action_types]
        )

    @property
    def vehicle_class(self) -> Callable:
        """
        Return the vehicle class for the agents, assuming all share the same config.
        """
        return CustomDiscreteAction(self.env, **self.action_config).vehicle_class

    def act(self, action: tuple[int]) -> None:
        """
        Apply the action tuple to the corresponding controlled vehicles.
        """
        assert isinstance(action, tuple), "Action must be a tuple for multi-agent setup"
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

    # def get_available_actions(self):
    #     """
    #     Return the Cartesian product of available actions for all agents.
    #     """
    #     return itertools.product(
    #         *[action_type.get_available_actions() for action_type in self.agents_action_types]
    #     )


class ExpectedVelocityAction(ActionType):
    """Control a single vehicle using desired velocity commands."""

    def __init__(self, env: AbstractEnv, *, min_velocity: float, max_velocity: float) -> None:
        super().__init__(env)
        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.array([self.min_velocity], dtype=np.float32),
            high=np.array([self.max_velocity], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def vehicle_class(self) -> Callable:
        return CustomDiscreteAction(self.env, target_speeds=[self.max_velocity]).vehicle_class

    def act(self, action: np.ndarray | float) -> None:
        desired = np.clip(float(np.asarray(action).squeeze()), self.min_velocity, self.max_velocity)
        vehicle = self.controlled_vehicle
        vehicle.target_speed = desired
        vehicle.max_speed = max(vehicle.max_speed, desired)
        vehicle.act(None)


class ExpectedVelocityMultiAgentAction(ActionType):
    """Apply expected velocity control to all controlled vehicles."""

    def __init__(self, env: AbstractEnv, config: dict | None = None, **kwargs) -> None:
        super().__init__(env)
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.min_velocity = float(cfg.get("min_velocity", 0.0))
        self.max_velocity = float(cfg.get("max_velocity", 15.0))
        self.actions: list[ExpectedVelocityAction] = []
        for vehicle in self.env.controlled_vehicles:
            action = ExpectedVelocityAction(
                env,
                min_velocity=self.min_velocity,
                max_velocity=self.max_velocity,
            )
            action.controlled_vehicle = vehicle
            self.actions.append(action)

    def space(self) -> spaces.Space:
        lows = np.full(len(self.actions), self.min_velocity, dtype=np.float32)
        highs = np.full(len(self.actions), self.max_velocity, dtype=np.float32)
        return spaces.Box(low=lows, high=highs, dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return ExpectedVelocityAction(
            self.env,
            min_velocity=self.min_velocity,
            max_velocity=self.max_velocity,
        ).vehicle_class

    def act(self, action: np.ndarray | Iterable[float]) -> None:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        assert values.shape[0] == len(self.actions), "Expected velocity per controlled vehicle"
        for value, action_type in zip(values, self.actions):
            action_type.act(value)
