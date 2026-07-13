"""Public-API Habitat 0.3 navigation environment wrapper."""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from vlnce_server.cfrp import HabitatActionAdapter

from .records import (
    NavigationMetrics,
    NavigationObservation,
    NavigationStep,
    PrivilegedNavigationState,
)


ALLOWED_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
_STANDARD_METRICS = ("distance_to_goal", "success", "spl", "path_length")


class Habitat030NavigationEnvironment:
    """Small CFRP-facing wrapper around a constructed Habitat-Lab 0.3 Env.

    Habitat is intentionally injected instead of imported here so unit tests can
    exercise the contract with fake environments and without Habitat installed.
    """

    def __init__(
        self,
        env: Any,
        action_adapter: Optional[HabitatActionAdapter] = None,
        fallback_instruction: Optional[str] = None,
    ) -> None:
        self.env = env
        self.action_adapter = action_adapter or HabitatActionAdapter()
        self.fallback_instruction = fallback_instruction
        self._last_raw_observation: Optional[Mapping[str, Any]] = None
        self._path_length = 0.0
        self._last_position: Optional[Tuple[float, ...]] = None

    def reset(self) -> NavigationObservation:
        raw_observation = self.env.reset()
        self._last_raw_observation = _as_mapping(raw_observation)
        self._path_length = 0.0
        self._last_position = self._current_position()
        return self._navigation_observation(self._last_raw_observation)

    def step(self, cfrp_action: str) -> NavigationStep:
        command = self.action_adapter.for_task(cfrp_action)
        if command.habitat_action is None:
            raise ValueError(f"CFRP action has no Habitat task action: {cfrp_action}")
        raw_observation = self.env.step(command.habitat_action)
        self._last_raw_observation = _as_mapping(raw_observation)
        self._accumulate_path_length()
        return NavigationStep(
            observation=self._navigation_observation(self._last_raw_observation),
            metrics=self.metrics(),
            episode_over=self._episode_over(command.terminate_episode),
            action=cfrp_action,
            habitat_action=command.habitat_action,
        )

    def metrics(self) -> NavigationMetrics:
        raw_metrics = self.env.get_metrics() or {}
        metrics = _as_mapping(raw_metrics)
        extra = []
        for key, value in metrics.items():
            if key in _STANDARD_METRICS:
                continue
            number = _to_optional_float(value)
            if number is not None:
                extra.append((str(key), number))
        extra.sort(key=lambda item: item[0])
        native_path_length = _to_optional_float(metrics.get("path_length"))
        return NavigationMetrics(
            distance_to_goal=_to_optional_float(metrics.get("distance_to_goal")),
            success=_to_optional_float(metrics.get("success")),
            spl=_to_optional_float(metrics.get("spl")),
            path_length=native_path_length if native_path_length is not None else self._path_length,
            extra=tuple(extra),
        )

    def raw_metrics(self) -> Mapping[str, Any]:
        """Return logging-only Habitat measurements; never use these as model input."""
        return _as_mapping(self.env.get_metrics() or {})

    def agent_pose(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        state = self.env.sim.get_agent_state()
        return (_to_float_tuple(state.position), _rotation_to_tuple(state.rotation))

    def privileged_state(self) -> PrivilegedNavigationState:
        episode = self.env.current_episode
        position, rotation = self.agent_pose()
        return PrivilegedNavigationState(
            episode_id=str(episode.episode_id),
            agent_position=position,
            agent_rotation=rotation,
            goal_positions=_goal_positions(episode),
            expert_path=_expert_path(episode),
        )

    def close(self) -> None:
        self.env.close()

    def _navigation_observation(self, raw_observation: Mapping[str, Any]) -> NavigationObservation:
        episode = self.env.current_episode
        return NavigationObservation(
            episode_id=str(episode.episode_id),
            instruction=self._instruction(raw_observation, episode),
            rgb=raw_observation.get("rgb"),
            allowed_actions=ALLOWED_ACTIONS,
        )

    def _instruction(self, raw_observation: Mapping[str, Any], episode: Any) -> str:
        instruction = raw_observation.get("instruction")
        text = _instruction_text(instruction)
        if text:
            return text
        episode_instruction = getattr(episode, "instruction", None)
        text = _instruction_text(episode_instruction)
        if text:
            return text
        if self.fallback_instruction is not None:
            return self.fallback_instruction
        return ""

    def _episode_over(self, fallback_when_stop: bool) -> bool:
        value = getattr(self.env, "episode_over", None)
        if value is None:
            return bool(fallback_when_stop)
        return bool(value)

    def _current_position(self) -> Tuple[float, ...]:
        state = self.env.sim.get_agent_state()
        return _to_float_tuple(state.position)

    def _accumulate_path_length(self) -> None:
        current_position = self._current_position()
        if self._last_position is not None:
            self._path_length += _euclidean_distance(self._last_position, current_position)
        self._last_position = current_position


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return dict(value)


def _instruction_text(instruction: Any) -> str:
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, Mapping):
        for key in ("text", "instruction_text"):
            text = instruction.get(key)
            if text:
                return str(text)
        return ""
    for attr_name in ("text", "instruction_text"):
        text = getattr(instruction, attr_name, None)
        if text:
            return str(text)
    return ""


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Number):
        return float(value)
    return None


def _to_float_tuple(value: Any) -> Tuple[float, ...]:
    if value is None:
        return tuple()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(float(item) for item in value)


def _rotation_to_tuple(rotation: Any) -> Tuple[float, ...]:
    if rotation is None:
        return tuple()
    if hasattr(rotation, "real") and hasattr(rotation, "imag"):
        imag = rotation.imag
        if hasattr(imag, "tolist"):
            imag_values = imag.tolist()
        elif isinstance(imag, Iterable):
            imag_values = list(imag)
        else:
            imag_values = [getattr(imag, axis) for axis in ("x", "y", "z")]
        return tuple(float(value) for value in [rotation.real] + list(imag_values))
    return _to_float_tuple(rotation)


def _goal_positions(episode: Any) -> Tuple[Tuple[float, ...], ...]:
    goals = getattr(episode, "goals", None) or []
    positions = []
    for goal in goals:
        position = getattr(goal, "position", None)
        if position is not None:
            positions.append(_to_float_tuple(position))
    return tuple(positions)


def _expert_path(episode: Any) -> Tuple[Tuple[float, ...], ...]:
    for attr_name in ("expert_path", "reference_path"):
        path = getattr(episode, attr_name, None)
        if path:
            return _path_to_tuple(path)
    shortest_paths = getattr(episode, "shortest_paths", None) or []
    if shortest_paths:
        first_path = shortest_paths[0]
        return _path_to_tuple(first_path)
    return tuple()


def _path_to_tuple(path: Iterable[Any]) -> Tuple[Tuple[float, ...], ...]:
    points = []
    for point in path:
        position = getattr(point, "position", point)
        points.append(_to_float_tuple(position))
    return tuple(points)


def _euclidean_distance(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    if len(left) != len(right):
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
