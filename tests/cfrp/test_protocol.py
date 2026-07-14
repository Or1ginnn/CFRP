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


STAGE1_HOLD_XML = """
<progress>hold</progress>
<subgoal>exit the bedroom through the doorway</subgoal>
<action>MOVE_FORWARD</action>
"""


STAGE1_ADVANCE_XML = """
<progress>advance</progress>
<subgoal>follow the hallway toward the stairs</subgoal>
<action>MOVE_FORWARD</action>
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


def test_stage1_controller_holds_then_advances_plan_cursor():
    plan = parse_cfrp_output(INIT_XML).plan
    assert plan is not None
    controller = CFRPController(
        allowed_actions=ALLOWED_ACTIONS,
        current_plan=plan,
        mode="stage1",
    )

    held = controller.step(parse_cfrp_output(STAGE1_HOLD_XML))
    advanced = controller.step(parse_cfrp_output(STAGE1_ADVANCE_XML))

    assert held.progress == "hold"
    assert held.current_plan == plan
    assert advanced.progress == "advance"
    assert advanced.current_plan is not None
    assert advanced.current_plan.current_index == 1
    assert advanced.current_plan.points[0].status == "done"
    assert advanced.current_plan.points[0].text == plan.points[0].text
    assert advanced.current_plan.points[1].status == "current"


def test_stage1_rejects_tool_and_plan_updates():
    plan = parse_cfrp_output(INIT_XML).plan
    assert plan is not None

    with pytest.raises(CFRPProtocolError, match="must not contain <tool>"):
        validate_output(
            parse_cfrp_output(CONTINUE_XML),
            ALLOWED_ACTIONS,
            previous_plan=plan,
            mode="stage1",
        )

    with pytest.raises(CFRPProtocolError, match="must not contain plan updates"):
        validate_output(
            parse_cfrp_output(
                STAGE1_HOLD_XML
                + "<plan_update><abandon>p1</abandon><current>x</current><future>y</future></plan_update>"
            ),
            ALLOWED_ACTIONS,
            previous_plan=plan,
            mode="stage1",
        )


def test_stage1_requires_valid_progress_and_controller_plan():
    output = parse_cfrp_output(STAGE1_HOLD_XML.replace("hold", "done", 1))
    plan = parse_cfrp_output(INIT_XML).plan
    assert plan is not None

    with pytest.raises(CFRPProtocolError, match="invalid progress"):
        validate_output(output, ALLOWED_ACTIONS, previous_plan=plan, mode="stage1")
    with pytest.raises(CFRPProtocolError, match="controller-owned"):
        validate_output(
            parse_cfrp_output(STAGE1_HOLD_XML), ALLOWED_ACTIONS, mode="stage1"
        )


def test_plan_cursor_cannot_advance_past_final_point():
    plan = parse_cfrp_output(
        """
        <plan><global>reach target</global><local>
          <p id="p1" status="current">stop at target</p>
        </local></plan>
        <tool>continue</tool><subgoal>stop at target</subgoal><action>STOP</action>
        """
    ).plan
    assert plan is not None

    with pytest.raises(CFRPProtocolError, match="following todo"):
        plan.advance_current()


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


def test_stage1_accepts_a_bounded_action_chunk_but_stage2_remains_atomic():
    plan = parse_cfrp_output(INIT_XML).plan
    assert plan is not None
    output = parse_cfrp_output(
        "<progress>hold</progress><subgoal>cross the room</subgoal>"
        "<actions><action>MOVE_FORWARD</action><action>MOVE_FORWARD</action>"
        "<action>TURN_LEFT</action></actions>"
    )

    validate_output(output, ALLOWED_ACTIONS, previous_plan=plan, mode="stage1")
    assert output.actions == ("MOVE_FORWARD", "MOVE_FORWARD", "TURN_LEFT")
    assert output.action == "MOVE_FORWARD"
    with pytest.raises(CFRPProtocolError, match="exactly one action"):
        validate_output(output, ALLOWED_ACTIONS, previous_plan=plan, mode="stage2")


def test_stage1_rejects_stop_mixed_with_other_chunk_actions():
    plan = parse_cfrp_output(INIT_XML).plan
    assert plan is not None
    output = parse_cfrp_output(
        "<progress>hold</progress><subgoal>stop</subgoal>"
        "<actions><action>MOVE_FORWARD</action><action>STOP</action></actions>"
    )

    with pytest.raises(CFRPProtocolError, match="STOP must be the only"):
        validate_output(output, ALLOWED_ACTIONS, previous_plan=plan, mode="stage1")


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
