from __future__ import annotations

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import ActionType, ContinuousAction, DiscreteAction, DiscreteMetaAction, \
    MultiAgentAction

import functools
import itertools
from typing import TYPE_CHECKING, Callable, Union

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
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
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
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    elif config["type"] == "CustomDiscreteAction":
        return CustomDiscreteAction(env, **config)
    else:
        raise ValueError("Unknown action type")