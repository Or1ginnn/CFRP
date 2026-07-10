import pytest

from vlnce_server.cfrp import (
    BranchContext,
    BranchTraceRecorder,
    CFRPBranchingError,
    CFRPCheckpoint,
    CounterfactualGroup,
    CriticalStateBaseline,
    EpisodeReference,
    make_navigation_pose,
    make_trajectory_prefix,
)


ROTATION = (1.0, 0.0, 0.0, 0.0)
CONTINUE_XML = """
<tool>continue</tool><subgoal>move ahead</subgoal><action>MOVE_FORWARD</action>
"""
REPLAN_XML = """
<plan><global>reach kitchen</global><local>
  <p id="r1" status="current">return to hallway</p>
</local></plan>
<tool>replan</tool><subgoal>return to hallway</subgoal><action>TURN_LEFT</action>
"""


def _pose(x, z):
    return make_navigation_pose((x, 0.0, z), ROTATION)


def _context():
    endpoint = _pose(1.0, 1.0)
    checkpoint = CFRPCheckpoint(
        agent_position=endpoint.position,
        agent_rotation=endpoint.rotation,
        current_plan=None,
        controller_action_history=("MOVE_FORWARD",),
        recent_observation_history=("t0",),
        recent_action_history=("MOVE_FORWARD",),
        turn_index=1,
        cooldown_steps=0,
        episode_id="episode-1",
    )
    episode = EpisodeReference(
        episode_id="episode-1",
        scene_id="scene-1",
        instruction="Leave the bedroom and reach the kitchen.",
        start_pose=_pose(0.0, 0.0),
        goal_description="kitchen",
        goal_positions=((2.0, 0.0, 2.0),),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
        success_distance=3.0,
        success_condition="STOP within success_distance",
        expert_path=(_pose(0.0, 0.0), endpoint, _pose(2.0, 2.0)),
    )
    prefix = make_trajectory_prefix(
        poses=(_pose(0.0, 0.0), endpoint),
        actions=("MOVE_FORWARD",),
        path_length=1.4,
        collisions=0,
        elapsed_steps=1,
        metrics={"distance_to_goal": 5.0, "spl": 0.0},
    )
    return BranchContext(
        checkpoint=checkpoint,
        episode=episode,
        prefix=prefix,
        baseline=CriticalStateBaseline(5.0, 0.2, 1),
        normal_prompt="Current observation: side room. Choose continue or replan.",
        critical_step=1,
        trigger_reason="distance_to_expert exceeded threshold",
    )


def _trace(tool, xml, action, end_pose):
    recorder = BranchTraceRecorder(
        forced_tool=tool,
        first_output_xml=xml,
        first_output_valid=True,
        start_pose=_pose(1.0, 1.0),
    )
    subgoal = "move ahead" if tool == "continue" else "return to hallway"
    recorder.record_step(
        raw_xml=xml,
        tool=tool,
        subgoal=subgoal,
        action=action,
        valid=True,
        pose=end_pose,
        metrics={"distance_to_goal": 4.5},
        environment_info={"episode_over": False},
    )
    return recorder.finish(final_metrics={"distance_to_goal": 4.5})


def test_context_keeps_normal_prompt_reference_and_shared_prefix():
    context = _context()
    assert context.normal_prompt.startswith("Current observation")
    assert context.episode.scene_id == "scene-1"
    assert context.prefix.metrics.get("distance_to_goal") == 5.0
    assert context.checkpoint.cooldown_steps == 0


def test_trace_records_full_xml_decision_and_environment_result():
    trace = _trace("replan", REPLAN_XML, "TURN_LEFT", _pose(1.0, 1.0))
    step = trace.steps[0]
    assert step.tool == "replan"
    assert step.subgoal == "return to hallway"
    assert step.valid is True
    assert step.environment_info.get("episode_over") is False
    assert step.metrics.get("distance_to_goal") == 4.5


def test_invalid_first_output_can_end_without_executing_a_step():
    trace = BranchTraceRecorder(
        forced_tool="replan",
        first_output_xml="not xml",
        first_output_valid=False,
        start_pose=_pose(1.0, 1.0),
    ).finish(terminal_reason="invalid_first_output")
    assert trace.steps == ()


def test_valid_first_output_must_match_forced_tool():
    recorder = BranchTraceRecorder(
        forced_tool="replan",
        first_output_xml=CONTINUE_XML,
        first_output_valid=True,
        start_pose=_pose(1.0, 1.0),
    )
    recorder.record_step(
        raw_xml=CONTINUE_XML,
        tool="continue",
        subgoal="move ahead",
        action="MOVE_FORWARD",
        valid=True,
        pose=_pose(1.5, 1.5),
    )
    with pytest.raises(CFRPBranchingError, match="does not match forced tool"):
        recorder.finish()


def test_counterfactual_group_shares_one_context_and_prefix():
    context = _context()
    group = CounterfactualGroup(
        context=context,
        continue_trace=_trace("continue", CONTINUE_XML, "MOVE_FORWARD", _pose(1.5, 1.5)),
        replan_trace=_trace("replan", REPLAN_XML, "TURN_LEFT", _pose(1.0, 1.0)),
    )
    assert group.context is context
    assert group.continue_trace.start_pose == context.prefix.poses[-1]
    assert group.replan_trace.start_pose == context.prefix.poses[-1]


def test_prefix_requires_one_more_pose_than_actions():
    with pytest.raises(CFRPBranchingError, match="poses must equal"):
        make_trajectory_prefix(
            poses=(_pose(0.0, 0.0),),
            actions=("MOVE_FORWARD",),
            path_length=0.0,
            collisions=0,
            elapsed_steps=1,
            metrics={},
        )
