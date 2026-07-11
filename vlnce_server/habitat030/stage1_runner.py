"""Scripted Stage 1 loop runner for Habitat 0.3 navigation smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from vlnce_server.cfrp import (
    CFRPController,
    PlanState,
    parse_cfrp_output,
)

from .records import NavigationMetrics, NavigationObservation


@dataclass(frozen=True)
class FixedHistoryBuffer:
    """Model-visible fixed-window observation/action history."""

    visual_history: Tuple[NavigationObservation, ...] = tuple()
    action_history: Tuple[str, ...] = tuple()
    max_visual: int = 6
    max_action: int = 8

    @classmethod
    def create(cls, max_visual: int = 6, max_action: int = 8) -> "FixedHistoryBuffer":
        if max_visual < 1:
            raise ValueError("max_visual must be at least 1")
        if max_action < 1:
            raise ValueError("max_action must be at least 1")
        return cls(max_visual=max_visual, max_action=max_action)

    def reset(self, observation: NavigationObservation) -> "FixedHistoryBuffer":
        _assert_oracle_free_observation(observation)
        return FixedHistoryBuffer(
            visual_history=(observation,),
            action_history=tuple(),
            max_visual=self.max_visual,
            max_action=self.max_action,
        )

    def append(
        self,
        observation: NavigationObservation,
        action: str,
    ) -> "FixedHistoryBuffer":
        _assert_oracle_free_observation(observation)
        return FixedHistoryBuffer(
            visual_history=(self.visual_history + (observation,))[-self.max_visual :],
            action_history=(self.action_history + (action,))[-self.max_action :],
            max_visual=self.max_visual,
            max_action=self.max_action,
        )


@dataclass(frozen=True)
class Stage1TrajectoryStep:
    turn_index: int
    raw_xml: str
    progress: str
    subgoal: str
    action: str
    habitat_action: str
    episode_over: bool
    plan_xml: str
    history_visual_count: int
    history_action_count: int
    metrics: NavigationMetrics


class Stage1EpisodeRunner:
    """Run scripted Stage 1 CFRP XML outputs through a navigation wrapper."""

    def __init__(
        self,
        env_wrapper: object,
        initial_plan: PlanState,
        history: Optional[FixedHistoryBuffer] = None,
    ) -> None:
        self.env_wrapper = env_wrapper
        self.controller = CFRPController(
            allowed_actions=set(getattr(env_wrapper, "allowed_actions", ()))
            or {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"},
            current_plan=initial_plan,
            mode="stage1",
        )
        self.history = history or FixedHistoryBuffer.create()
        self.trajectory: list[Stage1TrajectoryStep] = []
        self.initial_observation: Optional[NavigationObservation] = None

    def reset(self) -> NavigationObservation:
        observation = self.env_wrapper.reset()
        self.initial_observation = observation
        self.history = self.history.reset(observation)
        self.trajectory = []
        return observation

    def step(self, raw_xml: str, turn_index: Optional[int] = None) -> Stage1TrajectoryStep:
        if self.initial_observation is None:
            self.reset()

        output = parse_cfrp_output(raw_xml)
        controller_result = self.controller.step(output)
        env_step = self.env_wrapper.step(controller_result.action)
        self.history = self.history.append(env_step.observation, controller_result.action)
        current_plan = controller_result.current_plan
        assert current_plan is not None

        trajectory_step = Stage1TrajectoryStep(
            turn_index=len(self.trajectory) if turn_index is None else turn_index,
            raw_xml=output.raw_xml,
            progress=controller_result.progress or "",
            subgoal=controller_result.subgoal,
            action=controller_result.action,
            habitat_action=env_step.habitat_action,
            episode_over=env_step.episode_over,
            plan_xml=current_plan.to_xml(),
            history_visual_count=len(self.history.visual_history),
            history_action_count=len(self.history.action_history),
            metrics=env_step.metrics,
        )
        self.trajectory.append(trajectory_step)
        return trajectory_step

    def run(self, raw_xml_outputs: Iterable[str]) -> Tuple[Stage1TrajectoryStep, ...]:
        self.reset()
        for turn_index, raw_xml in enumerate(raw_xml_outputs):
            trajectory_step = self.step(raw_xml, turn_index=turn_index)
            if trajectory_step.episode_over or trajectory_step.action == "STOP":
                break
        return tuple(self.trajectory)


def _assert_oracle_free_observation(observation: NavigationObservation) -> None:
    forbidden = (
        "pose",
        "agent_position",
        "agent_rotation",
        "goal_positions",
        "distance_to_goal",
        "reference_path",
        "expert_path",
    )
    leaked = [name for name in forbidden if hasattr(observation, name)]
    if leaked:
        raise ValueError(f"Stage 1 history observation leaked privileged fields: {leaked}")
