from vlnce_server.habitat030.action_runner import ActionOnlyEpisodeRunner
from vlnce_server.habitat030.records import NavigationMetrics, NavigationObservation, NavigationStep


class _FakeEnvironment:
    def __init__(self) -> None:
        self.actions = []

    def reset(self):
        return self._observation(0)

    def step(self, action):
        self.actions.append(action)
        index = len(self.actions)
        return NavigationStep(
            observation=self._observation(index),
            metrics=NavigationMetrics(
                distance_to_goal=None,
                success=1.0 if action == "STOP" else 0.0,
                spl=0.0,
                path_length=float(index),
                extra=tuple(),
            ),
            episode_over=action == "STOP",
            action=action,
            habitat_action=action.lower(),
        )

    @staticmethod
    def _observation(index):
        return NavigationObservation(
            episode_id="1",
            instruction="Walk to the doorway.",
            rgb=f"rgb-{index}",
            allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
        )


def test_action_runner_observes_after_each_single_action() -> None:
    env = _FakeEnvironment()
    runner = ActionOnlyEpisodeRunner(env)
    runner.reset()
    first = runner.step("<action>MOVE_FORWARD</action>")
    assert first.action == "MOVE_FORWARD"
    assert runner.model_request().visual_history[-1] == "rgb-1"
    second = runner.step("<action>STOP</action>")
    assert second.episode_over is True
    assert env.actions == ["MOVE_FORWARD", "STOP"]


def test_action_runner_uses_at_most_nine_uniform_frames() -> None:
    runner = ActionOnlyEpisodeRunner(_FakeEnvironment())
    runner.reset()
    for _ in range(17):
        runner.step("<action>MOVE_FORWARD</action>")
    request = runner.model_request()
    assert len(request.visual_history) == 9
    assert request.visual_history[0] == "rgb-0"
    assert request.visual_history[-1] == "rgb-17"
