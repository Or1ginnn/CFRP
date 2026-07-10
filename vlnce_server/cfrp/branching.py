"""Structured data for CFRP counterfactual branch collection.

The simulator checkpoint restores a physical branch point. This module records
the shared episode/prefix context and the two branch-local suffixes used later
for scoring and group-relative optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

from .checkpoint import CFRPCheckpoint


Pose = tuple[float, ...]
BranchTool = Literal["continue", "replan"]


class CFRPBranchingError(ValueError):
    """Raised when counterfactual branch data is inconsistent."""


def _pose(values: Sequence[float]) -> Pose:
    pose = tuple(float(value) for value in values)
    if not pose:
        raise CFRPBranchingError("pose must not be empty")
    return pose


@dataclass(frozen=True)
class EpisodeReference:
    """Immutable task metadata shared by every branch of an episode."""

    episode_id: str
    instruction: str
    goal_description: str
    success_distance: float
    expert_path: tuple[Pose, ...]

    def __post_init__(self) -> None:
        if not self.episode_id or not self.instruction or not self.goal_description:
            raise CFRPBranchingError("episode reference fields must not be empty")
        if self.success_distance <= 0:
            raise CFRPBranchingError("success_distance must be positive")
        if not self.expert_path:
            raise CFRPBranchingError("expert_path must not be empty")


@dataclass(frozen=True)
class MetricSnapshot:
    """Numeric Habitat task metrics captured at one navigation state."""

    values: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "MetricSnapshot":
        return cls(tuple(sorted((str(name), float(value)) for name, value in values.items())))

    def get(self, name: str) -> float | None:
        for metric_name, value in self.values:
            if metric_name == name:
                return value
        return None


@dataclass(frozen=True)
class TrajectoryPrefix:
    """The fixed trajectory before a counterfactual critical state."""

    poses: tuple[Pose, ...]
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
    """Reference values used to measure local recovery after branching."""

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
    """All data shared by forced continue and forced replan rollouts."""

    checkpoint: CFRPCheckpoint
    episode: EpisodeReference
    prefix: TrajectoryPrefix
    baseline: CriticalStateBaseline

    def __post_init__(self) -> None:
        if self.checkpoint.episode_id is not None and self.checkpoint.episode_id != self.episode.episode_id:
            raise CFRPBranchingError("checkpoint and episode reference must have the same episode_id")
        if self.baseline.expert_progress_index >= len(self.episode.expert_path):
            raise CFRPBranchingError("expert_progress_index exceeds expert_path")


@dataclass(frozen=True)
class BranchStep:
    """One action and its resulting pose in a branch-local suffix."""

    action: str
    pose: Pose
    collided: bool = False

    def __post_init__(self) -> None:
        if not self.action:
            raise CFRPBranchingError("branch action must not be empty")


@dataclass(frozen=True)
class BranchTrace:
    """The suffix generated after a forced tool decision at a critical state."""

    forced_tool: BranchTool
    first_output_xml: str
    start_pose: Pose
    steps: tuple[BranchStep, ...]
    terminal_reason: str | None = None
    final_metrics: MetricSnapshot = field(default_factory=lambda: MetricSnapshot(()))

    def __post_init__(self) -> None:
        if self.forced_tool not in {"continue", "replan"}:
            raise CFRPBranchingError(f"invalid forced tool: {self.forced_tool}")
        if not self.first_output_xml.strip():
            raise CFRPBranchingError("branch first_output_xml must not be empty")
        if not self.steps:
            raise CFRPBranchingError("branch trace must contain at least one step")

    @property
    def poses(self) -> tuple[Pose, ...]:
        return (self.start_pose,) + tuple(step.pose for step in self.steps)

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.steps)

    @property
    def collisions(self) -> int:
        return sum(step.collided for step in self.steps)


class BranchTraceRecorder:
    """Mutable collector that produces one validated immutable branch trace."""

    def __init__(self, *, forced_tool: BranchTool, first_output_xml: str, start_pose: Sequence[float]) -> None:
        self._forced_tool = forced_tool
        self._first_output_xml = first_output_xml
        self._start_pose = _pose(start_pose)
        self._steps: list[BranchStep] = []

    def record_step(self, *, action: str, pose: Sequence[float], collided: bool = False) -> None:
        self._steps.append(BranchStep(action=action, pose=_pose(pose), collided=collided))

    def finish(
        self,
        *,
        terminal_reason: str | None = None,
        final_metrics: Mapping[str, float] | None = None,
    ) -> BranchTrace:
        return BranchTrace(
            forced_tool=self._forced_tool,
            first_output_xml=self._first_output_xml,
            start_pose=self._start_pose,
            steps=tuple(self._steps),
            terminal_reason=terminal_reason,
            final_metrics=MetricSnapshot.from_mapping(final_metrics or {}),
        )


@dataclass(frozen=True)
class CounterfactualGroup:
    """A same-context pair ready for later reward comparison."""

    context: BranchContext
    continue_trace: BranchTrace
    replan_trace: BranchTrace

    def __post_init__(self) -> None:
        if self.continue_trace.forced_tool != "continue":
            raise CFRPBranchingError("continue_trace must force continue")
        if self.replan_trace.forced_tool != "replan":
            raise CFRPBranchingError("replan_trace must force replan")
        expected_pose = self.prefix_end_pose
        if self.continue_trace.start_pose != expected_pose or self.replan_trace.start_pose != expected_pose:
            raise CFRPBranchingError("both branch traces must start at the prefix endpoint")

    @property
    def prefix_end_pose(self) -> Pose:
        return self.context.prefix.poses[-1]


def make_trajectory_prefix(
    *,
    poses: Iterable[Sequence[float]],
    actions: Iterable[str],
    path_length: float,
    collisions: int,
    elapsed_steps: int,
    metrics: Mapping[str, float],
) -> TrajectoryPrefix:
    """Normalize recorded navigation data into an immutable prefix."""

    return TrajectoryPrefix(
        poses=tuple(_pose(pose) for pose in poses),
        actions=tuple(actions),
        path_length=float(path_length),
        collisions=int(collisions),
        elapsed_steps=int(elapsed_steps),
        metrics=MetricSnapshot.from_mapping(metrics),
    )
