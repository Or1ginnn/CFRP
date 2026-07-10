from dataclasses import dataclass

import pytest

from vlnce_server.cfrp import (
    CFRPCheckpointError,
    CFRPController,
    capture_cfrp_checkpoint,
    parse_cfrp_output,
    restore_cfrp_checkpoint,
)


@dataclass
class FakeAgentState:
    position: list[float]
    rotation: list[float]


class FakeSimulator:
    def __init__(self):
        self.state = FakeAgentState(position=[1.0, 2.0, 3.0], rotation=[1.0, 0.0, 0.0, 0.0])
        self.restore_calls = []

    def get_agent_state(self, agent_id=0):
        assert agent_id == 0
        return self.state

    def set_agent_state(self, position, rotation, agent_id=0, reset_sensors=False):
        self.restore_calls.append((position, rotation, agent_id, reset_sensors))
        self.state = FakeAgentState(position=position, rotation=rotation)


def _controller():
    controller = CFRPController(allowed_actions={"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"})
    controller.step(
        parse_cfrp_output(
            """
            <plan><global>reach target</global><local>
              <p id="p1" status="current">move forward</p>
            </local></plan>
            <tool>continue</tool><subgoal>move forward</subgoal><action>MOVE_FORWARD</action>
            """
        )
    )
    return controller


def test_checkpoint_restores_pose_and_cfrp_control_memory():
    simulator = FakeSimulator()
    controller = _controller()
    observations = [{"frame": [1]}]
    checkpoint = capture_cfrp_checkpoint(
        simulator,
        controller,
        recent_observation_history=observations,
        recent_action_history=["MOVE_FORWARD"],
        turn_index=3,
        episode_id="episode-1",
    )

    simulator.state.position[0] = 99.0
    controller.current_plan = None
    controller.action_history.append("TURN_LEFT")
    observations[0]["frame"].append(2)

    restored = restore_cfrp_checkpoint(
        simulator, controller, checkpoint, current_episode_id="episode-1"
    )

    assert simulator.state.position == [1.0, 2.0, 3.0]
    assert simulator.restore_calls[-1][3] is False
    assert controller.current_plan == checkpoint.current_plan
    assert controller.action_history == ["MOVE_FORWARD"]
    assert restored.recent_observation_history == ({"frame": [1]},)
    assert restored.recent_action_history == ("MOVE_FORWARD",)
    assert restored.turn_index == 3


def test_checkpoint_rejects_a_different_episode():
    checkpoint = capture_cfrp_checkpoint(
        FakeSimulator(),
        _controller(),
        recent_observation_history=(),
        recent_action_history=(),
        turn_index=0,
        episode_id="episode-1",
    )

    with pytest.raises(CFRPCheckpointError, match="episode mismatch"):
        restore_cfrp_checkpoint(
            FakeSimulator(), _controller(), checkpoint, current_episode_id="episode-2"
        )


def test_checkpoint_requires_non_negative_turn_index():
    with pytest.raises(CFRPCheckpointError, match="non-negative"):
        capture_cfrp_checkpoint(
            FakeSimulator(),
            _controller(),
            recent_observation_history=(),
            recent_action_history=(),
            turn_index=-1,
        )
