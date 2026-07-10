import pytest

from vlnce_server.cfrp import CFRPController, CFRPProtocolError, parse_cfrp_output, validate_output
from vlnce_server.cfrp.prompts import build_step_prompt


ALLOWED_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}


INIT_XML = """
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen sink</global>
  <local>
    <p id="p1" status="current">exit the bedroom</p>
    <p id="p2" status="todo">follow the hallway toward the stairs</p>
    <p id="p3" status="todo">stop near the kitchen sink</p>
  </local>
</plan>
<tool>continue</tool>
<subgoal>exit the bedroom through the doorway</subgoal>
<action>MOVE_FORWARD</action>
"""


CONTINUE_XML = """
<tool>continue</tool>
<subgoal>continue along the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
"""


RECOVERY_XML = """
<plan>
  <global>bedroom -> hallway -> stairs -> kitchen sink</global>
  <local>
    <p id="p1" status="done">exit the bedroom</p>
    <p id="p2" status="abandoned">follow the hallway toward the stairs</p>
    <p id="r1" status="current">turn around, leave the side room, and return to the hallway</p>
    <p id="p3" status="todo">continue along the hallway toward the stairs</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>turn around, leave the side room through the doorway, and return to the hallway</subgoal>
<action>TURN_LEFT</action>
"""


STOP_ACTION_XML = """
<tool>continue</tool>
<subgoal>stop near the target location</subgoal>
<action>STOP</action>
"""


PLAN_UPDATE_XML = """
<tool>replan</tool>
<plan_update>
  <abandon>p1</abandon>
  <current>return to the hallway entrance</current>
  <future>continue through the hallway toward the sink</future>
</plan_update>
<subgoal>leave the side room and return to the hallway</subgoal>
<action>TURN_LEFT</action>
"""


def test_parse_and_validate_initial_continue_with_plan():
    output = parse_cfrp_output(INIT_XML)

    validate_output(output, ALLOWED_ACTIONS)

    assert output.tool == "continue"
    assert output.plan is not None
    assert output.plan.current_points()[0].id == "p1"


def test_controller_runs_init_then_continue():
    controller = CFRPController(allowed_actions=ALLOWED_ACTIONS)

    first = controller.step(parse_cfrp_output(INIT_XML))
    second = controller.step(parse_cfrp_output(CONTINUE_XML))

    assert first.action == "MOVE_FORWARD"
    assert second.action == "MOVE_FORWARD"
    assert second.current_plan == first.current_plan


def test_replan_updates_plan_after_initialization():
    initial = parse_cfrp_output(INIT_XML)
    output = parse_cfrp_output(RECOVERY_XML)

    validate_output(output, ALLOWED_ACTIONS, previous_plan=initial.plan)

    assert output.tool == "replan"
    assert output.plan is not None
    assert output.plan.current_points()[0].id == "r1"


def test_stop_is_a_continue_action_not_a_tool():
    output = parse_cfrp_output(STOP_ACTION_XML)

    validate_output(output, ALLOWED_ACTIONS)

    assert output.tool == "continue"
    assert output.action == "STOP"


def test_stop_tool_is_rejected():
    output = parse_cfrp_output(STOP_ACTION_XML.replace("continue", "stop", 1))

    with pytest.raises(CFRPProtocolError, match="invalid tool"):
        validate_output(output, ALLOWED_ACTIONS)


def test_invalid_action_is_rejected():
    output = parse_cfrp_output(
        """
        <tool>continue</tool>
        <subgoal>walk left</subgoal>
        <action>GO_LEFT</action>
        """
    )

    with pytest.raises(CFRPProtocolError, match="invalid action"):
        validate_output(output, ALLOWED_ACTIONS)


def test_replan_requires_plan():
    output = parse_cfrp_output(
        """
        <tool>replan</tool>
        <subgoal>return to the hallway</subgoal>
        <action>TURN_LEFT</action>
        """
    )

    with pytest.raises(CFRPProtocolError, match="replan must output"):
        validate_output(output, ALLOWED_ACTIONS)


def test_controller_applies_compact_plan_update():
    controller = CFRPController(allowed_actions=ALLOWED_ACTIONS)
    controller.step(parse_cfrp_output(INIT_XML))

    result = controller.step(parse_cfrp_output(PLAN_UPDATE_XML))

    assert result.current_plan is not None
    assert result.current_plan.current_points()[0].text == "return to the hallway entrance"
    assert next(point for point in result.current_plan.points if point.id == "p1").status == "abandoned"


def test_plan_update_requires_the_current_point():
    initial = parse_cfrp_output(INIT_XML).plan
    assert initial is not None
    output = parse_cfrp_output(PLAN_UPDATE_XML.replace("<abandon>p1</abandon>", "<abandon>p2</abandon>"))

    with pytest.raises(CFRPProtocolError, match="current point"):
        validate_output(output, ALLOWED_ACTIONS, previous_plan=initial)


def test_continue_rejects_plan_after_initialization():
    initial = parse_cfrp_output(INIT_XML)
    output = parse_cfrp_output(INIT_XML)

    with pytest.raises(CFRPProtocolError, match="continue must not output"):
        validate_output(output, ALLOWED_ACTIONS, previous_plan=initial.plan)


def test_plan_rejects_multiple_current_points():
    with pytest.raises(CFRPProtocolError, match="exactly one current"):
        parse_cfrp_output(
        """
        <plan>
          <global>go to the kitchen</global>
          <local>
            <p id="p1" status="current">exit bedroom</p>
            <p id="p2" status="current">follow hallway</p>
          </local>
        </plan>
        <tool>continue</tool>
        <subgoal>exit bedroom</subgoal>
        <action>MOVE_FORWARD</action>
        """
        )


def test_done_points_are_immutable_on_replan():
    previous = parse_cfrp_output(RECOVERY_XML).plan
    assert previous is not None
    changed_done = RECOVERY_XML.replace("exit the bedroom", "leave the bedroom")
    output = parse_cfrp_output(changed_done)

    with pytest.raises(CFRPProtocolError, match="done point text changed"):
        validate_output(output, ALLOWED_ACTIONS, previous_plan=previous)


def test_step_prompt_includes_plan_and_allowed_actions():
    plan = parse_cfrp_output(INIT_XML).plan
    prompt = build_step_prompt(
        full_instruction="Exit the bedroom and stop near the sink.",
        allowed_actions=["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
        current_observation="[image]",
        recent_visual_history=["t-1: bedroom doorway"],
        recent_actions=["MOVE_FORWARD"],
        current_plan=plan,
        active_instruction_excerpt="Exit the bedroom",
    )

    assert "Full instruction:" in prompt
    assert "MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP" in prompt
    assert "<plan>" in prompt
