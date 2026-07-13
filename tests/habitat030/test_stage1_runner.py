from __future__ import annotations

from dataclasses import dataclass

import pytest

from vlnce_server.cfrp import CFRPProtocolError, PlanPoint, PlanState
from vlnce_server.habitat030.records import NavigationMetrics, NavigationObservation, NavigationStep
from vlnce_server.habitat030.stage1_runner import FixedHistoryBuffer, Stage1EpisodeRunner


@dataclass
class FakeWrapper:
    episode_over_on_stop: bool = True

    def __post_init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = []
        self.allowed_actions = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")

    def reset(self) -> NavigationObservation:
        self.reset_calls += 1
        return self._observation("reset")

    def step(self, action: str) -> NavigationStep:
        self.step_calls.append(action)
        episode_over = action == "STOP" and self.episode_over_on_stop
        return NavigationStep(
            observation=self._observation(f"after-{len(self.step_calls)}"),
            metrics=NavigationMetrics(
                distance_to_goal=10.0 - len(self.step_calls),
                success=1.0 if episode_over else 0.0,
                spl=0.5 if episode_over else 0.0,
                path_length=float(len(self.step_calls)),
                extra=tuple(),
            ),
            episode_over=episode_over,
            action=action,
            habitat_action={
                "MOVE_FORWARD": "move_forward",
                "TURN_LEFT": "turn_left",
                "TURN_RIGHT": "turn_right",
                "STOP": "stop",
            }[action],
        )

    def _observation(self, suffix: str) -> NavigationObservation:
        return NavigationObservation(
            episode_id="episode-1",
            instruction=f"instruction-{suffix}",
            rgb=f"rgb-{suffix}",
            allowed_actions=self.allowed_actions,
        )


def plan() -> PlanState:
    return PlanState(
        global_goal="reach the destination",
        points=(
            PlanPoint(id="p1", status="current", text="first"),
            PlanPoint(id="p2", status="todo", text="second"),
            PlanPoint(id="p3", status="todo", text="third"),
        ),
    )


def stage1_xml(progress: str, action: str = "MOVE_FORWARD") -> str:
    return (
        f"<progress>{progress}</progress>"
        "<subgoal>scripted subgoal</subgoal>"
        f"<action>{action}</action>"
    )


def test_stage1_hold_and_advance_run_on_fake_wrapper():
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())

    trajectory = runner.run((stage1_xml("hold", "TURN_LEFT"), stage1_xml("advance", "STOP")))

    assert [step.progress for step in trajectory] == ["hold", "advance"]
    assert [step.action for step in trajectory] == ["TURN_LEFT", "STOP"]
    assert runner.controller.current_plan is not None
    assert runner.controller.current_plan.current_points()[0].id == "p2"


def test_advance_marks_current_done_and_next_todo_current():
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())

    runner.run((stage1_xml("advance", "MOVE_FORWARD"),))

    assert runner.controller.current_plan is not None
    by_id = {point.id: point.status for point in runner.controller.current_plan.points}
    assert by_id["p1"] == "done"
    assert by_id["p2"] == "current"
    assert by_id["p3"] == "todo"


def test_hold_does_not_change_plan():
    initial_plan = plan()
    runner = Stage1EpisodeRunner(FakeWrapper(), initial_plan)

    runner.run((stage1_xml("hold", "MOVE_FORWARD"),))

    assert runner.controller.current_plan == initial_plan


class FakeStage1Policy:
    def __init__(self, outputs: tuple[str, ...]) -> None:
        self.outputs = iter(outputs)
        self.requests = []

    def generate_xml(self, request):
        self.requests.append(request)
        return next(self.outputs)


def test_policy_runner_reuses_history_prompt_protocol_and_action_path():
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())
    policy = FakeStage1Policy(
        (
            stage1_xml("advance", "MOVE_FORWARD"),
            stage1_xml("hold", "STOP"),
        )
    )

    trajectory = runner.run_with_policy(policy, max_steps=3)

    assert [step.action for step in trajectory] == ["MOVE_FORWARD", "STOP"]
    assert runner.env_wrapper.step_calls == ["MOVE_FORWARD", "STOP"]
    assert policy.requests[0].instruction == "instruction-reset"
    assert policy.requests[0].visual_history == ("rgb-reset",)
    assert policy.requests[1].visual_history == ("rgb-reset", "rgb-after-1")
    assert policy.requests[1].action_history == ("MOVE_FORWARD",)
    assert policy.requests[1].current_plan.current_points()[0].id == "p2"


def test_model_request_uses_observation_actions_when_wrapper_does_not_expose_them():
    wrapper = FakeWrapper()
    runner = Stage1EpisodeRunner(wrapper, plan())
    runner.reset()
    del wrapper.allowed_actions

    assert runner.model_request().allowed_actions == ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


@pytest.mark.parametrize(
    "raw_xml",
    [
        "<tool>continue</tool><progress>hold</progress><subgoal>x</subgoal><action>STOP</action>",
        (
            "<progress>hold</progress><subgoal>x</subgoal><action>STOP</action>"
            "<plan_update><abandon_id>p1</abandon_id><current>x</current><future>y</future></plan_update>"
        ),
        (
            "<progress>hold</progress><subgoal>x</subgoal><action>STOP</action>"
            "<plan><global>g</global><local><p id=\"p1\" status=\"current\">x</p></local></plan>"
        ),
    ],
)
def test_stage1_rejects_tool_plan_and_plan_update(raw_xml: str):
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())
    runner.reset()

    with pytest.raises(CFRPProtocolError):
        runner.step(raw_xml)


def test_history_windows_are_capped():
    runner = Stage1EpisodeRunner(
        FakeWrapper(episode_over_on_stop=False),
        plan(),
        history=FixedHistoryBuffer.create(max_visual=6, max_action=8),
    )

    runner.run(stage1_xml("hold", "TURN_LEFT") for _ in range(12))

    assert len(runner.history.visual_history) == 6
    assert len(runner.history.action_history) == 8
    assert max(step.history_visual_count for step in runner.trajectory) == 6
    assert max(step.history_action_count for step in runner.trajectory) == 8


def test_default_history_window_matches_stage1_plan():
    history = FixedHistoryBuffer.create()

    assert history.max_visual == 6
    assert history.max_action == 8


def test_history_contains_no_privileged_fields():
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())

    runner.run((stage1_xml("hold", "MOVE_FORWARD"),))

    for observation in runner.history.visual_history:
        assert not hasattr(observation, "goal_positions")
        assert not hasattr(observation, "distance_to_goal")
        assert not hasattr(observation, "reference_path")
        assert not hasattr(observation, "expert_path")


def test_stop_calls_wrapper_task_stop_and_ends_loop():
    wrapper = FakeWrapper()
    runner = Stage1EpisodeRunner(wrapper, plan())

    trajectory = runner.run((stage1_xml("hold", "STOP"), stage1_xml("hold", "MOVE_FORWARD")))

    assert wrapper.step_calls == ["STOP"]
    assert len(trajectory) == 1
    assert trajectory[0].episode_over is True
    assert trajectory[0].habitat_action == "stop"


def test_trajectory_step_saves_protocol_action_habitat_action_and_plan_xml():
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())

    trajectory = runner.run((stage1_xml("hold", "TURN_RIGHT"),))

    step = trajectory[0]
    assert step.progress == "hold"
    assert step.action == "TURN_RIGHT"
    assert step.habitat_action == "turn_right"
    assert '<p id="p1" status="current">first</p>' in step.plan_xml
    assert step.raw_xml == stage1_xml("hold", "TURN_RIGHT")


def test_non_stop_does_not_end_loop_early():
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())

    trajectory = runner.run(
        (
            stage1_xml("hold", "MOVE_FORWARD"),
            stage1_xml("hold", "TURN_LEFT"),
            stage1_xml("hold", "STOP"),
        )
    )

    assert [step.action for step in trajectory] == ["MOVE_FORWARD", "TURN_LEFT", "STOP"]


@pytest.mark.parametrize(
    "raw_xml",
    [
        "<progress>hold<progress><subgoal>x</subgoal><action>STOP</action>",
        "<progress>hold</progress><subgoal>x</subgoal><action>JUMP</action>",
    ],
)
def test_invalid_xml_or_action_raises_protocol_error(raw_xml: str):
    runner = Stage1EpisodeRunner(FakeWrapper(), plan())
    runner.reset()

    with pytest.raises(CFRPProtocolError):
        runner.step(raw_xml)
