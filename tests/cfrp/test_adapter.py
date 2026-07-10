import pytest

from vlnce_server.cfrp import (
    CFRPActionAdapterError,
    CFRPController,
    HabitatActionAdapter,
    parse_cfrp_output,
)


def test_simulator_maps_motion_actions():
    adapter = HabitatActionAdapter()

    assert adapter.for_simulator("MOVE_FORWARD").habitat_action == "move_forward"
    assert adapter.for_simulator("TURN_LEFT").habitat_action == "turn_left"
    assert adapter.for_simulator("TURN_RIGHT").habitat_action == "turn_right"
    assert adapter.for_simulator("TURN_LEFT").terminate_episode is False


def test_simulator_stop_has_no_physical_action():
    command = HabitatActionAdapter().for_simulator("STOP")

    assert command.habitat_action is None
    assert command.target == "simulator"
    assert command.terminate_episode is True


def test_task_stop_uses_habitat_stop_action():
    command = HabitatActionAdapter().for_task("STOP")

    assert command.habitat_action == "stop"
    assert command.target == "task"
    assert command.terminate_episode is True


def test_adapter_rejects_unavailable_target_action():
    adapter = HabitatActionAdapter(simulator_actions=("move_forward",))

    with pytest.raises(CFRPActionAdapterError, match="action unavailable: turn_left"):
        adapter.for_simulator("TURN_LEFT")


def test_adapter_rejects_unknown_primitive_action():
    with pytest.raises(CFRPActionAdapterError, match="unsupported CFRP"):
        HabitatActionAdapter().for_task("JUMP")


def test_adapter_accepts_controller_result():
    controller = CFRPController(
        allowed_actions={"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
    )
    result = controller.step(
        parse_cfrp_output(
            """
            <plan>
              <global>reach target</global>
              <local><p id="p1" status="current">move ahead</p></local>
            </plan>
            <tool>continue</tool>
            <subgoal>move ahead</subgoal>
            <action>MOVE_FORWARD</action>
            """
        )
    )

    command = HabitatActionAdapter().controller_result_for_task(result)

    assert command.habitat_action == "move_forward"
    assert command.terminate_episode is False
