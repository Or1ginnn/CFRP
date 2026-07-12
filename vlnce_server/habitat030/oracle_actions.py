"""Offline-only Habitat shortest-path action labels for Stage 1 warm-up."""

from __future__ import annotations

from typing import Any


_ACTION_NAMES = {
    "STOP": "STOP",
    "MOVE_FORWARD": "MOVE_FORWARD",
    "TURN_LEFT": "TURN_LEFT",
    "TURN_RIGHT": "TURN_RIGHT",
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

            name = HabitatSimActions(int(action)).name
        except (ImportError, TypeError, ValueError) as exc:
            raise OracleActionError(f"unsupported Habitat oracle action: {action!r}") from exc
    action_name = _ACTION_NAMES.get(str(name))
    if action_name is None:
        raise OracleActionError(f"oracle action is not a CFRP primitive: {name}")
    return action_name
