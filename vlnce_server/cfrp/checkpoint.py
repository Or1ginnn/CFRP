"""Minimal static-scene checkpoints for CFRP counterfactual branches.

This module intentionally snapshots only agent pose and CFRP control memory.
It is sufficient for static VLN scenes; dynamic-world state and Habitat task
measure rollback belong to a later environment-specific layer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from .controller import CFRPController
from .protocol import PlanState


class CFRPCheckpointError(ValueError):
    """Raised when a checkpoint cannot safely be restored."""


@dataclass(frozen=True)
class CFRPCheckpoint:
    """A branch point for a static Habitat navigation episode."""

    agent_position: Any
    agent_rotation: Any
    current_plan: PlanState | None
    controller_action_history: tuple[str, ...]
    recent_observation_history: tuple[Any, ...]
    recent_action_history: tuple[str, ...]
    turn_index: int
    cooldown_steps: int = 0
    episode_id: str | None = None


@dataclass(frozen=True)
class RestoredCFRPState:
    recent_observation_history: tuple[Any, ...]
    recent_action_history: tuple[str, ...]
    turn_index: int
    cooldown_steps: int


def capture_cfrp_checkpoint(
    simulator: Any,
    controller: CFRPController,
    *,
    recent_observation_history: Sequence[Any],
    recent_action_history: Sequence[str],
    turn_index: int,
    cooldown_steps: int = 0,
    episode_id: str | None = None,
    agent_id: int = 0,
) -> CFRPCheckpoint:
    """Capture pose plus the CFRP control memory required by a branch rollout."""

    if turn_index < 0 or cooldown_steps < 0:
        raise CFRPCheckpointError("turn_index and cooldown_steps must be non-negative")
    state = _get_agent_state(simulator, agent_id)
    return CFRPCheckpoint(
        agent_position=deepcopy(state.position),
        agent_rotation=deepcopy(state.rotation),
        current_plan=controller.current_plan,
        controller_action_history=tuple(controller.action_history),
        recent_observation_history=tuple(deepcopy(recent_observation_history)),
        recent_action_history=tuple(recent_action_history),
        turn_index=turn_index,
        cooldown_steps=cooldown_steps,
        episode_id=episode_id,
    )


def restore_cfrp_checkpoint(
    simulator: Any,
    controller: CFRPController,
    checkpoint: CFRPCheckpoint,
    *,
    current_episode_id: str | None = None,
    agent_id: int = 0,
) -> RestoredCFRPState:
    """Restore a static-scene branch point without resetting rendered sensors."""

    if checkpoint.episode_id is not None and current_episode_id != checkpoint.episode_id:
        raise CFRPCheckpointError(
            f"checkpoint episode mismatch: expected {checkpoint.episode_id}, got {current_episode_id}"
        )
    _set_agent_state(
        simulator,
        deepcopy(checkpoint.agent_position),
        deepcopy(checkpoint.agent_rotation),
        agent_id,
    )
    controller.current_plan = checkpoint.current_plan
    controller.action_history = list(checkpoint.controller_action_history)
    return RestoredCFRPState(
        recent_observation_history=tuple(deepcopy(checkpoint.recent_observation_history)),
        recent_action_history=checkpoint.recent_action_history,
        turn_index=checkpoint.turn_index,
        cooldown_steps=checkpoint.cooldown_steps,
    )


def _get_agent_state(simulator: Any, agent_id: int) -> Any:
    """Support both Habitat-Lab's wrapper and raw habitat_sim.Simulator."""

    get_agent_state = getattr(simulator, "get_agent_state", None)
    if callable(get_agent_state):
        return get_agent_state(agent_id)
    get_agent = getattr(simulator, "get_agent", None)
    if callable(get_agent):
        return get_agent(agent_id).get_state()
    raise CFRPCheckpointError("simulator does not expose an agent state API")


def _set_agent_state(simulator: Any, position: Any, rotation: Any, agent_id: int) -> None:
    """Restore a pose while keeping sensor extrinsics attached to the agent body."""

    set_agent_state = getattr(simulator, "set_agent_state", None)
    if callable(set_agent_state):
        set_agent_state(position, rotation, agent_id=agent_id, reset_sensors=False)
        return
    get_agent = getattr(simulator, "get_agent", None)
    if callable(get_agent):
        agent = get_agent(agent_id)
        state = agent.get_state()
        state.position = position
        state.rotation = rotation
        agent.set_state(state, reset_sensors=False)
        return
    raise CFRPCheckpointError("simulator does not expose an agent restore API")
