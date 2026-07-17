"""One-action-per-observation Habitat loop for the Phase 0 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlnce_server.qwen3vl.action_policy import ActionModelRequest, parse_action_xml

from .records import NavigationMetrics, NavigationObservation


@dataclass(frozen=True)
class ActionTrajectoryStep:
    step_index: int
    raw_xml: str
    action: str
    habitat_action: str
    episode_over: bool
    metrics: NavigationMetrics


class ActionOnlyEpisodeRunner:
    """Execute one model action, observe, then require a fresh model decision."""

    def __init__(self, env_wrapper: Any) -> None:
        self.env_wrapper = env_wrapper
        self.observations: list[NavigationObservation] = []
        self.trajectory: list[ActionTrajectoryStep] = []

    def reset(self) -> NavigationObservation:
        observation = self.env_wrapper.reset()
        self.observations = [observation]
        self.trajectory = []
        return observation

    def model_request(self) -> ActionModelRequest:
        if not self.observations:
            self.reset()
        current = self.observations[-1]
        return ActionModelRequest.from_episode_history(
            current.instruction,
            tuple(observation.rgb for observation in self.observations),
            current.allowed_actions,
        )

    def step(self, raw_xml: str) -> ActionTrajectoryStep:
        if not self.observations:
            self.reset()
        action = parse_action_xml(raw_xml, self.observations[-1].allowed_actions)
        result = self.env_wrapper.step(action)
        self.observations.append(result.observation)
        step = ActionTrajectoryStep(
            step_index=len(self.trajectory),
            raw_xml=raw_xml.strip(),
            action=action,
            habitat_action=result.habitat_action,
            episode_over=result.episode_over,
            metrics=result.metrics,
        )
        self.trajectory.append(step)
        return step
