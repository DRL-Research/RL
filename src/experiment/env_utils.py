from __future__ import annotations

from typing import Any, Mapping


INTERSECTION_ENV_ID = "RELintersection-v0"
ROUNDABOUT_ENV_ID = "RELroundabout-v0"
DOUBLE_INTERSECTION_ENV_ID = "RELdouble-intersection-v0"

_ENVIRONMENTS_REGISTERED = False


def resolve_env_id(experiment_config: Any, env_config: Mapping[str, Any] | None = None) -> str:
    """Resolve the Gym environment id for the current experiment."""

    configured_env_id = getattr(experiment_config, "ENV_ID", None)
    if isinstance(configured_env_id, str) and configured_env_id.strip():
        return configured_env_id.strip()

    config = env_config or getattr(experiment_config, "CONFIG", None) or {}
    controlled_cars = config.get("controlled_cars") or {}
    first_controlled_car = next(iter(controlled_cars.values()), None)
    start_lane = (first_controlled_car or {}).get("start_lane")

    if isinstance(start_lane, tuple) and start_lane and isinstance(start_lane[0], str):
        if start_lane[0].startswith(("A_", "B_")):
            return DOUBLE_INTERSECTION_ENV_ID

    return INTERSECTION_ENV_ID


def register_all_supported_envs() -> None:
    """Register every environment used by the requested comparison suite."""

    global _ENVIRONMENTS_REGISTERED
    if _ENVIRONMENTS_REGISTERED:
        return

    from highwayenv.utils import (
        patch_intersection_env,
        register_double_intersection_env,
        register_intersection_env,
        register_roundabout_env,
    )

    patch_intersection_env()
    register_intersection_env()
    register_roundabout_env()
    register_double_intersection_env()
    _ENVIRONMENTS_REGISTERED = True
