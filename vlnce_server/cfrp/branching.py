"""Structured records for CFRP counterfactual branch collection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from .checkpoint import CFRPCheckpoint
from .protocol import CFRPProtocolError, parse_cfrp_output


Vector = tuple[float, ...]
BranchTool = Literal["continue", "replan"]
InfoScalar = str | int | float | bool | None


class CFRPBranchingError(ValueError):
    """Raised when counterfactual branch data is inconsistent."""


def _vector(values: Sequence[float], name: str) -> Vector:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise CFRPBranchingError(f"{name} must not be empty")
    return vector


@dataclass(frozen=True)
class NavigationPose:
    """Agent body pose used to define an embodied branch state."""

    position: Vector
    rotation: Vector

    def __post_init__(self) -> None:
        if not self.position or not self.rotation:
            raise CFRPBranchingError("navigation pose requires position and rotation")


def make_navigation_pose(
    position: Sequence[float], rotation: Sequence[float]
) -> NavigationPose:
    return NavigationPose(
        position=_vector(position, "position"),
        rotation=_vector(rotation, "rotation"),
    )


@dataclass(frozen=True)
class EpisodeReference:
    """Privileged immutable task metadata shared by both training branches."""

    episode_id: str
    scene_id: str
    instruction: str
    start_pose: NavigationPose
    goal_description: str
    goal_positions: tuple[Vector, ...]
    allowed_actions: tuple[str, ...]
    success_distance: float
    success_condition: str
    expert_path: tuple[NavigationPose, ...]

    def __post_init__(self) -> None:
        required = (
            self.episode_id,
            self.scene_id,
            self.instruction,
            self.goal_description,
            self.success_condition,
        )
        if not all(required):
            raise CFRPBranchingError("episode reference fields must not be empty")
        if self.success_distance <= 0:
            raise CFRPBranchingError("success_distance must be positive")
        if not self.goal_positions or not self.allowed_actions or not self.expert_path:
            raise CFRPBranchingError(
                "goal_positions, allowed_actions, and expert_path must not be empty"
            )


@dataclass(frozen=True)
class MetricSnapshot:
    """Numeric Habitat metrics captured at one navigation state."""

    values: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "MetricSnapshot":
        return cls(
            tuple(sorted((str(name), float(value)) for name, value in values.items()))
        )

    def get(self, name: str) -> float | None:
        return next((value for metric_name, value in self.values if metric_name == name), None)


@dataclass(frozen=True)
class EnvironmentInfo:
    """Small scalar environment fields retained for debugging and scoring."""

    values: tuple[tuple[str, InfoScalar], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, InfoScalar]) -> "EnvironmentInfo":
        return cls(tuple(sorted((str(name), deepcopy(value)) for name, value in values.items())))

    def get(self, name: str) -> InfoScalar:
        return next((value for info_name, value in self.values if info_name == name), None)


@dataclass(frozen=True)
class TrajectoryPrefix:
    """One immutable lightweight pose/action history shared by both branches."""

    poses: tuple[NavigationPose, ...]
    actions: tuple[str, ...]
    path_length: float
    collisions: int
    elapsed_steps: int
    metrics: MetricSnapshot

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.actions) + 1:
            raise CFRPBranchingError("prefix poses must equal actions plus one")
        if self.path_length < 0 or self.collisions < 0 or self.elapsed_steps < 0:
            raise CFRPBranchingError("prefix counters must be non-negative")
        if self.elapsed_steps < len(self.actions):
            raise CFRPBranchingError("elapsed_steps cannot be less than recorded actions")


@dataclass(frozen=True)
class CriticalStateBaseline:
    distance_to_goal: float
    distance_to_expert: float
    expert_progress_index: int

    def __post_init__(self) -> None:
        if self.distance_to_goal < 0 or self.distance_to_expert < 0:
            raise CFRPBranchingError("critical distances must be non-negative")
        if self.expert_progress_index < 0:
            raise CFRPBranchingError("expert_progress_index must be non-negative")


@dataclass(frozen=True)
class BranchContext:
    """Shared normal-prompt state for forced continue and replan rollouts."""

    checkpoint: CFRPCheckpoint
    episode: EpisodeReference
    prefix: TrajectoryPrefix
    baseline: CriticalStateBaseline
    normal_prompt: str
    critical_step: int
    trigger_reason: str

    def __post_init__(self) -> None:
        if self.checkpoint.episode_id not in {None, self.episode.episode_id}:
            raise CFRPBranchingError(
                "checkpoint and episode reference must have the same episode_id"
            )
        if not self.normal_prompt.strip() or not self.trigger_reason.strip():
            raise CFRPBranchingError("normal_prompt and trigger_reason must not be empty")
        if self.critical_step < 0:
            raise CFRPBranchingError("critical_step must be non-negative")
        if self.baseline.expert_progress_index >= len(self.episode.expert_path):
            raise CFRPBranchingError("expert_progress_index exceeds expert_path")
        if self.prefix.poses[-1].position != tuple(self.checkpoint.agent_position):
            raise CFRPBranchingError("prefix endpoint must match checkpoint position")


@dataclass(frozen=True)
class BranchStep:
    """One parsed XML decision and its resulting environment state."""

    raw_xml: str
    tool: str
    subgoal: str
    action: str
    valid: bool
    pose: NavigationPose
    collided: bool = False
    metrics: MetricSnapshot = field(default_factory=lambda: MetricSnapshot(()))
    environment_info: EnvironmentInfo = field(
        default_factory=lambda: EnvironmentInfo(())
    )

    def __post_init__(self) -> None:
        if not self.raw_xml.strip() or not self.action:
            raise CFRPBranchingError("branch raw_xml and action must not be empty")
        if self.valid and (self.tool not in {"continue", "replan"} or not self.subgoal):
            raise CFRPBranchingError("valid branch step requires tool and subgoal")


@dataclass(frozen=True)
class BranchTrace:
    """The suffix after one forced tool intervention at a critical state."""

    forced_tool: BranchTool
    first_output_xml: str
    first_output_valid: bool
    start_pose: NavigationPose
    steps: tuple[BranchStep, ...]
    terminal_reason: str | None = None
    final_metrics: MetricSnapshot = field(default_factory=lambda: MetricSnapshot(()))

    def __post_init__(self) -> None:
        if self.forced_tool not in {"continue", "replan"}:
            raise CFRPBranchingError(f"invalid forced tool: {self.forced_tool}")
        if not self.first_output_xml.strip():
            raise CFRPBranchingError("branch first_output_xml must not be empty")
        if self.first_output_valid:
            try:
                parsed = parse_cfrp_output(self.first_output_xml)
            except CFRPProtocolError as exc:
                raise CFRPBranchingError(f"valid first output is not parseable: {exc}") from exc
            if parsed.tool != self.forced_tool:
                raise CFRPBranchingError("first output tool does not match forced tool")
            if not self.steps:
                raise CFRPBranchingError("valid branch trace must contain at least one step")
            first_step = self.steps[0]
            if first_step.tool != parsed.tool or first_step.action != parsed.action:
                raise CFRPBranchingError("first branch step does not match first_output_xml")

    @property
    def poses(self) -> tuple[NavigationPose, ...]:
        return (self.start_pose,) + tuple(step.pose for step in self.steps)

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.steps)

    @property
    def collisions(self) -> int:
        return sum(step.collided for step in self.steps)


class BranchTraceRecorder:
    def __init__(
        self,
        *,
        forced_tool: BranchTool,
        first_output_xml: str,
        first_output_valid: bool,
        start_pose: NavigationPose,
    ) -> None:
        self._forced_tool = forced_tool
        self._first_output_xml = first_output_xml
        self._first_output_valid = first_output_valid
        self._start_pose = start_pose
        self._steps: list[BranchStep] = []

    def record_step(
        self,
        *,
        raw_xml: str,
        tool: str,
        subgoal: str,
        action: str,
        valid: bool,
        pose: NavigationPose,
        collided: bool = False,
        metrics: Mapping[str, float] | None = None,
        environment_info: Mapping[str, InfoScalar] | None = None,
    ) -> None:
        self._steps.append(
            BranchStep(
                raw_xml=raw_xml,
                tool=tool,
                subgoal=subgoal,
                action=action,
                valid=valid,
                pose=pose,
                collided=collided,
                metrics=MetricSnapshot.from_mapping(metrics or {}),
                environment_info=EnvironmentInfo.from_mapping(environment_info or {}),
            )
        )

    def finish(
        self,
        *,
        terminal_reason: str | None = None,
        final_metrics: Mapping[str, float] | None = None,
    ) -> BranchTrace:
        return BranchTrace(
            forced_tool=self._forced_tool,
            first_output_xml=self._first_output_xml,
            first_output_valid=self._first_output_valid,
            start_pose=self._start_pose,
            steps=tuple(self._steps),
            terminal_reason=terminal_reason,
            final_metrics=MetricSnapshot.from_mapping(final_metrics or {}),
        )


@dataclass(frozen=True)
class CounterfactualGroup:
    context: BranchContext
    continue_trace: BranchTrace
    replan_trace: BranchTrace

    def __post_init__(self) -> None:
        if self.continue_trace.forced_tool != "continue":
            raise CFRPBranchingError("continue_trace must force continue")
        if self.replan_trace.forced_tool != "replan":
            raise CFRPBranchingError("replan_trace must force replan")
        expected = self.context.prefix.poses[-1]
        if self.continue_trace.start_pose != expected or self.replan_trace.start_pose != expected:
            raise CFRPBranchingError("both branch traces must start at the prefix endpoint")


def make_trajectory_prefix(
    *,
    poses: Iterable[NavigationPose],
    actions: Iterable[str],
    path_length: float,
    collisions: int,
    elapsed_steps: int,
    metrics: Mapping[str, float],
) -> TrajectoryPrefix:
    return TrajectoryPrefix(
        poses=tuple(poses),
        actions=tuple(actions),
        path_length=float(path_length),
        collisions=int(collisions),
        elapsed_steps=int(elapsed_steps),
        metrics=MetricSnapshot.from_mapping(metrics),
    )
