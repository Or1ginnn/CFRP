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
from .temporal_history import (
    DEFAULT_HISTORY_ANCHOR_COUNT,
    DEFAULT_MODEL_VISUAL_FRAME_COUNT,
    DEFAULT_RECENT_CONTIGUOUS_COUNT,
    DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL,
    DEFAULT_VISUAL_CONTEXT_WINDOW,
    SlowFastVisualHistory,
)


DEFAULT_MAX_VISUAL_HISTORY = DEFAULT_MODEL_VISUAL_FRAME_COUNT
DEFAULT_MAX_ACTION_HISTORY = 8


@dataclass(frozen=True)
class FixedHistoryBuffer:
    """Slow-fast visual state plus a bounded executed-action history."""

    visual_state: SlowFastVisualHistory[NavigationObservation] = SlowFastVisualHistory()
    action_history: Tuple[str, ...] = tuple()
    max_visual: int = DEFAULT_MAX_VISUAL_HISTORY
    max_action: int = DEFAULT_MAX_ACTION_HISTORY
    visual_context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW
    history_anchor_count: int = DEFAULT_HISTORY_ANCHOR_COUNT
    recent_contiguous_count: int = DEFAULT_RECENT_CONTIGUOUS_COUNT
    slow_memory_update_interval: int = DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL

    @classmethod
    def create(
        cls,
        max_visual: int = DEFAULT_MAX_VISUAL_HISTORY,
        max_action: int = DEFAULT_MAX_ACTION_HISTORY,
        *,
        visual_context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW,
        history_anchor_count: Optional[int] = None,
        recent_contiguous_count: Optional[int] = None,
        slow_memory_update_interval: int = DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL,
    ) -> "FixedHistoryBuffer":
        if max_visual < 1:
            raise ValueError("max_visual must be at least 1")
        if max_action < 1:
            raise ValueError("max_action must be at least 1")
        # Explicitly requested legacy budgets retain the old contiguous-tail
        # behavior for narrow smoke tests.  The Stage 1 default is always 6+3.
        if history_anchor_count is None and recent_contiguous_count is None and max_visual != DEFAULT_MAX_VISUAL_HISTORY:
            history_anchor_count, recent_contiguous_count = 0, max_visual
        history_anchor_count = (
            DEFAULT_HISTORY_ANCHOR_COUNT if history_anchor_count is None else history_anchor_count
        )
        recent_contiguous_count = (
            DEFAULT_RECENT_CONTIGUOUS_COUNT
            if recent_contiguous_count is None
            else recent_contiguous_count
        )
        if history_anchor_count < 0 or recent_contiguous_count < 1:
            raise ValueError("invalid visual history composition")
        if max_visual != history_anchor_count + recent_contiguous_count:
            raise ValueError("max_visual must equal history anchors plus recent frames")
        visual_state = SlowFastVisualHistory[NavigationObservation].create(
            context_window=visual_context_window,
            history_anchor_count=history_anchor_count,
            recent_contiguous_count=recent_contiguous_count,
            slow_memory_update_interval=slow_memory_update_interval,
        )
        return cls(
            max_visual=max_visual,
            max_action=max_action,
            visual_state=visual_state,
            visual_context_window=visual_state.context_window,
            history_anchor_count=visual_state.history_anchor_count,
            recent_contiguous_count=visual_state.recent_contiguous_count,
            slow_memory_update_interval=visual_state.slow_memory_update_interval,
        )

    @property
    def visual_history(self) -> Tuple[NavigationObservation, ...]:
        return self.visual_state.visible

    @property
    def visual_context(self) -> Tuple[NavigationObservation, ...]:
        return self.visual_state.context

    def reset(self, observation: NavigationObservation) -> "FixedHistoryBuffer":
        _assert_oracle_free_observation(observation)
        return FixedHistoryBuffer(
            visual_state=self.visual_state.reset(observation),
            action_history=tuple(),
            max_visual=self.max_visual,
            max_action=self.max_action,
            visual_context_window=self.visual_context_window,
            history_anchor_count=self.history_anchor_count,
            recent_contiguous_count=self.recent_contiguous_count,
            slow_memory_update_interval=self.slow_memory_update_interval,
        )

    def append(
        self,
        observation: NavigationObservation,
        action: str,
    ) -> "FixedHistoryBuffer":
        _assert_oracle_free_observation(observation)
        return FixedHistoryBuffer(
            visual_state=self.visual_state.append(observation),
            action_history=(self.action_history + (action,))[-self.max_action :],
            max_visual=self.max_visual,
            max_action=self.max_action,
            visual_context_window=self.visual_context_window,
            history_anchor_count=self.history_anchor_count,
            recent_contiguous_count=self.recent_contiguous_count,
            slow_memory_update_interval=self.slow_memory_update_interval,
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

    def model_request(self):
        """Return the current oracle-free state to a Stage 1 model policy."""

        if self.initial_observation is None:
            self.reset()
        current_plan = self.controller.current_plan
        if current_plan is None:
            raise RuntimeError("Stage 1 runner has no controller-owned plan")

        from vlnce_server.qwen3vl import Stage1ModelRequest

        latest_observation = self.history.visual_history[-1]
        # The observation is the authoritative per-turn action contract.  Some
        # Habitat wrappers expose no class-level ``allowed_actions`` attribute.
        allowed_actions = tuple(latest_observation.allowed_actions) or tuple(
            getattr(self.env_wrapper, "allowed_actions", ())
        )
        return Stage1ModelRequest(
            instruction=latest_observation.instruction,
            current_plan=current_plan,
            visual_history=tuple(observation.rgb for observation in self.history.visual_history),
            action_history=self.history.action_history,
            allowed_actions=allowed_actions,
        )

    def step_with_policy(self, policy: object, turn_index: Optional[int] = None) -> Stage1TrajectoryStep:
        """Generate one decision, then reuse the normal XML/controller/action path."""

        generate_xml = getattr(policy, "generate_xml", None)
        if not callable(generate_xml):
            raise TypeError("Stage 1 policy must provide generate_xml(request)")
        return self.step(generate_xml(self.model_request()), turn_index=turn_index)

    def run(self, raw_xml_outputs: Iterable[str]) -> Tuple[Stage1TrajectoryStep, ...]:
        self.reset()
        for turn_index, raw_xml in enumerate(raw_xml_outputs):
            trajectory_step = self.step(raw_xml, turn_index=turn_index)
            if trajectory_step.episode_over or trajectory_step.action == "STOP":
                break
        return tuple(self.trajectory)

    def run_with_policy(self, policy: object, max_steps: int) -> Tuple[Stage1TrajectoryStep, ...]:
        """Run a policy for at most ``max_steps`` turns or until task STOP."""

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.reset()
        for turn_index in range(max_steps):
            trajectory_step = self.step_with_policy(policy, turn_index=turn_index)
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
