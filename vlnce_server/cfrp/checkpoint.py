"""Minimal static-scene checkpoints for CFRP counterfactual branches.

This module intentionally snapshots only agent pose and CFRP control memory.
It is sufficient for static VLN scenes; dynamic-world state and Habitat task
measure rollback belong to a later environment-specific layer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .controller import CFRPController
from .protocol import PlanState


class CFRPCheckpointError(ValueError):
    """Raised when a checkpoint cannot safely be restored."""


class AgentStateSimulator(Protocol):
    def get_agent_state(self, agent_id: int = 0) -> Any: ...

    def set_agent_state(
        self,
        position: Any,
        rotation: Any,
        agent_id: int = 0,
        reset_sensors: bool = False,
    ) -> Any: ...


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
    episode_id: str | None = None


@dataclass(frozen=True)
class RestoredCFRPState:
    recent_observation_history: tuple[Any, ...]
    recent_action_history: tuple[str, ...]
    turn_index: int


def capture_cfrp_checkpoint(
    simulator: AgentStateSimulator,
    controller: CFRPController,
    *,
    recent_observation_history: Sequence[Any],
    recent_action_history: Sequence[str],
    turn_index: int,
    episode_id: str | None = None,
    agent_id: int = 0,
) -> CFRPCheckpoint:
    """Capture pose plus the CFRP control memory required by a branch rollout."""

    if turn_index < 0:
        raise CFRPCheckpointError("turn_index must be non-negative")
    state = simulator.get_agent_state(agent_id)
    return CFRPCheckpoint(
        agent_position=deepcopy(state.position),
        agent_rotation=deepcopy(state.rotation),
        current_plan=controller.current_plan,
        controller_action_history=tuple(controller.action_history),
        recent_observation_history=tuple(deepcopy(recent_observation_history)),
        recent_action_history=tuple(recent_action_history),
        turn_index=turn_index,
        episode_id=episode_id,
    )


def restore_cfrp_checkpoint(
    simulator: AgentStateSimulator,
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
    simulator.set_agent_state(
        deepcopy(checkpoint.agent_position),
        deepcopy(checkpoint.agent_rotation),
        agent_id=agent_id,
        reset_sensors=False,
    )
    controller.current_plan = checkpoint.current_plan
    controller.action_history = list(checkpoint.controller_action_history)
    return RestoredCFRPState(
        recent_observation_history=tuple(deepcopy(checkpoint.recent_observation_history)),
        recent_action_history=checkpoint.recent_action_history,
        turn_index=checkpoint.turn_index,
    )
