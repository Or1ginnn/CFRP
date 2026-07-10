import pytest

from vlnce_server.cfrp import CFRPProtocolError, run_scripted_cfrp_loop


ALLOWED_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def test_scripted_loop_runs_until_stop():
    observations = (
        "t0: bedroom doorway ahead",
        "t1: hallway visible",
        "t2: side room recovery needed",
        "t3: target visible",
    )
    model_outputs = (
        """
        <plan>
          <global>bedroom -> hallway -> target</global>
          <local>
            <p id="p1" status="current">exit bedroom</p>
            <p id="p2" status="todo">follow hallway</p>
          </local>
        </plan>
        <tool>continue</tool>
        <subgoal>exit bedroom</subgoal>
        <action>MOVE_FORWARD</action>
        """,
        """
        <tool>continue</tool>
        <subgoal>follow hallway</subgoal>
        <action>MOVE_FORWARD</action>
        """,
        """
        <plan>
          <global>bedroom -> hallway -> target</global>
          <local>
            <p id="p1" status="done">exit bedroom</p>
            <p id="p2" status="abandoned">follow hallway</p>
            <p id="r1" status="current">recover to hallway</p>
          </local>
        </plan>
        <tool>replan</tool>
        <subgoal>recover to hallway</subgoal>
        <action>TURN_LEFT</action>
        """,
        """
        <tool>continue</tool>
        <subgoal>stop near target</subgoal>
        <action>STOP</action>
        """,
    )

    turns = run_scripted_cfrp_loop(
        full_instruction="Exit the bedroom and stop near the target.",
        observations=observations,
        model_outputs=model_outputs,
        allowed_actions=ALLOWED_ACTIONS,
    )

    assert [turn.tool for turn in turns] == ["continue", "continue", "replan", "continue"]
    assert [turn.action for turn in turns] == ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_LEFT", "STOP"]
    assert turns[-1].is_stop is True
    assert turns[1].prompt.count("<plan>") == 1


def test_scripted_loop_rejects_invalid_model_action():
    with pytest.raises(CFRPProtocolError, match="invalid action"):
        run_scripted_cfrp_loop(
            full_instruction="Exit the room.",
            observations=("t0",),
            model_outputs=(
                """
                <tool>continue</tool>
                <subgoal>exit</subgoal>
                <action>JUMP</action>
                """,
            ),
            allowed_actions=ALLOWED_ACTIONS,
        )


def test_scripted_loop_requires_enough_model_outputs():
    with pytest.raises(ValueError, match="model_outputs"):
        run_scripted_cfrp_loop(
            full_instruction="Exit the room.",
            observations=("t0", "t1"),
            model_outputs=("",),
            allowed_actions=ALLOWED_ACTIONS,
        )
