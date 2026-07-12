"""Offline-only Habitat shortest-path action labels for Stage 1 warm-up."""

from __future__ import annotations

from typing import Any


_ACTION_NAMES = {
    "STOP": "STOP",
    "MOVE_FORWARD": "MOVE_FORWARD",
    "TURN_LEFT": "TURN_LEFT",
    "TURN_RIGHT": "TURN_RIGHT",
}
_HABITAT_ACTION_ALIASES = {
    "STOP": ("stop", "STOP"),
    "MOVE_FORWARD": ("move_forward", "MOVE_FORWARD"),
    "TURN_LEFT": ("turn_left", "TURN_LEFT"),
    "TURN_RIGHT": ("turn_right", "TURN_RIGHT"),
}


class OracleActionError(ValueError):
    """Raised when Habitat's shortest-path action cannot be used by CFRP."""


def cfrp_action_from_habitat_oracle(action: Any) -> str:
    """Map a Habitat ``ShortestPathFollower`` result to a CFRP primitive.

    This helper is intentionally for offline label collection only.  It must
    never be imported by online model prompting or inference code.
    """

    if action is None:
        raise OracleActionError("shortest-path follower returned no action")
    name = getattr(action, "name", None)
    if name is None:
        try:
            from habitat.sims.habitat_simulator.actions import HabitatSimActions

            return _cfrp_action_from_registry_id(action, HabitatSimActions)
        except (ImportError, TypeError, ValueError, OracleActionError) as exc:
            raise OracleActionError(f"unsupported Habitat oracle action: {action!r}") from exc
    action_name = _ACTION_NAMES.get(str(name).upper())
    if action_name is None:
        raise OracleActionError(f"oracle action is not a CFRP primitive: {name}")
    return action_name


def _cfrp_action_from_registry_id(action: Any, registry: Any) -> str:
    """Resolve an ID by comparing named entries in Habitat's singleton registry."""

    action_id = int(action)
    for cfrp_action, aliases in _HABITAT_ACTION_ALIASES.items():
        for alias in aliases:
            value = getattr(registry, alias, None)
            if value is not None and int(value) == action_id:
                return cfrp_action
    raise OracleActionError(f"oracle action is not a CFRP primitive: {action_id}")
