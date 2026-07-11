"""Typed records exposed by the Habitat 0.3 navigation wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class NavigationObservation:
    """Model-visible navigation inputs only."""

    episode_id: str
    instruction: str
    rgb: Any
    allowed_actions: Tuple[str, ...]


@dataclass(frozen=True)
class NavigationMetrics:
    """Task metrics kept outside model observations."""

    distance_to_goal: Optional[float]
    success: Optional[float]
    spl: Optional[float]
    path_length: Optional[float]
    extra: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class NavigationStep:
    """Result of executing one CFRP primitive through Habitat-Lab."""

    observation: NavigationObservation
    metrics: NavigationMetrics
    episode_over: bool
    action: str
    habitat_action: str


@dataclass(frozen=True)
class PrivilegedNavigationState:
    """Training/logging/checkpoint-only state that must not enter prompts."""

    episode_id: str
    agent_position: Tuple[float, ...]
    agent_rotation: Tuple[float, ...]
    goal_positions: Tuple[Tuple[float, ...], ...]
    expert_path: Tuple[Tuple[float, ...], ...]
