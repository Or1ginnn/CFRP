import pytest

from vlnce_server.cfrp import (
    BranchContext,
    BranchTraceRecorder,
    CFRPBranchingError,
    CFRPCheckpoint,
    CounterfactualGroup,
    CriticalStateBaseline,
    EpisodeReference,
    make_trajectory_prefix,
)


def _context():
    checkpoint = CFRPCheckpoint(
        agent_position=(1.0, 0.0, 1.0),
        agent_rotation=(1.0, 0.0, 0.0, 0.0),
        current_plan=None,
        controller_action_history=("MOVE_FORWARD",),
        recent_observation_history=("t0",),
        recent_action_history=("MOVE_FORWARD",),
        turn_index=1,
        episode_id="episode-1",
    )
    episode = EpisodeReference(
        episode_id="episode-1",
        instruction="Leave the bedroom and reach the kitchen.",
        goal_description="kitchen",
        success_distance=3.0,
        expert_path=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)),
    )
    prefix = make_trajectory_prefix(
        poses=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
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
        baseline=CriticalStateBaseline(
            distance_to_goal=5.0, distance_to_expert=0.2, expert_progress_index=1
        ),
    )


def test_branch_context_separates_shared_reference_and_prefix():
    context = _context()

    assert context.episode.instruction.startswith("Leave")
    assert context.prefix.metrics.get("distance_to_goal") == 5.0
    assert context.prefix.poses[-1] == (1.0, 0.0, 1.0)
    assert context.baseline.expert_progress_index == 1


def test_trace_recorder_records_only_branch_suffix():
    recorder = BranchTraceRecorder(
        forced_tool="replan",
        first_output_xml="<tool>replan</tool><action>TURN_LEFT</action>",
        start_pose=(1.0, 0.0, 1.0),
    )
    recorder.record_step(action="TURN_LEFT", pose=(1.0, 0.0, 1.0))
    recorder.record_step(action="MOVE_FORWARD", pose=(1.2, 0.0, 1.4), collided=True)
    trace = recorder.finish(
        terminal_reason="horizon", final_metrics={"distance_to_goal": 4.1}
    )

    assert trace.actions == ("TURN_LEFT", "MOVE_FORWARD")
    assert trace.poses[0] == (1.0, 0.0, 1.0)
    assert trace.collisions == 1
    assert trace.final_metrics.get("distance_to_goal") == 4.1


def test_counterfactual_group_requires_same_prefix_endpoint():
    context = _context()
    continue_trace = BranchTraceRecorder(
        forced_tool="continue",
        first_output_xml="<tool>continue</tool><action>MOVE_FORWARD</action>",
        start_pose=context.prefix.poses[-1],
    )
    continue_trace.record_step(action="MOVE_FORWARD", pose=(1.5, 0.0, 1.5))
    replan_trace = BranchTraceRecorder(
        forced_tool="replan",
        first_output_xml="<tool>replan</tool><action>TURN_LEFT</action>",
        start_pose=context.prefix.poses[-1],
    )
    replan_trace.record_step(action="TURN_LEFT", pose=(1.0, 0.0, 1.0))

    group = CounterfactualGroup(context, continue_trace.finish(), replan_trace.finish())

    assert group.prefix_end_pose == (1.0, 0.0, 1.0)


def test_prefix_requires_one_more_pose_than_actions():
    with pytest.raises(CFRPBranchingError, match="poses must equal"):
        make_trajectory_prefix(
            poses=((0.0, 0.0, 0.0),),
            actions=("MOVE_FORWARD",),
            path_length=0.0,
            collisions=0,
            elapsed_steps=1,
            metrics={},
        )


def test_context_requires_matching_episode_id():
    context = _context()
    mismatched_checkpoint = CFRPCheckpoint(
        **{**context.checkpoint.__dict__, "episode_id": "another-episode"}
    )

    with pytest.raises(CFRPBranchingError, match="same episode_id"):
        BranchContext(
            checkpoint=mismatched_checkpoint,
            episode=context.episode,
            prefix=context.prefix,
            baseline=context.baseline,
        )
