#!/usr/bin/env python
"""Show one shared-prefix continue/replan counterfactual group."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlnce_server.cfrp import (
    BranchContext,
    BranchTraceRecorder,
    CFRPCheckpoint,
    CounterfactualGroup,
    CriticalStateBaseline,
    EpisodeReference,
    make_navigation_pose,
    make_trajectory_prefix,
)


ROTATION = (1.0, 0.0, 0.0, 0.0)


def pose(x, z):
    return make_navigation_pose((x, 0.0, z), ROTATION)


def trace(tool, xml, action, subgoal, start_pose, end_pose):
    recorder = BranchTraceRecorder(
        forced_tool=tool,
        first_output_xml=xml,
        first_output_valid=True,
        start_pose=start_pose,
    )
    recorder.record_step(
        raw_xml=xml,
        tool=tool,
        subgoal=subgoal,
        action=action,
        valid=True,
        pose=end_pose,
        environment_info={"episode_over": False},
    )
    return recorder.finish()


def main() -> None:
    start, critical = pose(0.0, 0.0), pose(1.0, 1.0)
    prefix = make_trajectory_prefix(
        poses=(start, critical),
        actions=("MOVE_FORWARD",),
        path_length=1.4,
        collisions=0,
        elapsed_steps=1,
        metrics={"distance_to_goal": 5.0},
    )
    context = BranchContext(
        checkpoint=CFRPCheckpoint(
            critical.position,
            critical.rotation,
            None,
            ("MOVE_FORWARD",),
            ("t0",),
            ("MOVE_FORWARD",),
            1,
            0,
            "demo-episode",
        ),
        episode=EpisodeReference(
            "demo-episode",
            "scene-1",
            "Reach the kitchen.",
            start,
            "kitchen",
            ((2.0, 0.0, 2.0),),
            ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
            3.0,
            "STOP within success_distance",
            (start, critical, pose(2.0, 2.0)),
        ),
        prefix=prefix,
        baseline=CriticalStateBaseline(5.0, 0.2, 1),
        normal_prompt="Observation: side room. Choose continue or replan.",
        critical_step=1,
        trigger_reason="distance_to_expert exceeded threshold",
    )
    continue_xml = "<tool>continue</tool><subgoal>move ahead</subgoal><action>MOVE_FORWARD</action>"
    replan_xml = """<plan><global>kitchen</global><local><p id="r1" status="current">return</p></local></plan>
<tool>replan</tool><subgoal>return</subgoal><action>TURN_LEFT</action>"""
    group = CounterfactualGroup(
        context,
        trace("continue", continue_xml, "MOVE_FORWARD", "move ahead", critical, pose(1.5, 1.5)),
        trace("replan", replan_xml, "TURN_LEFT", "return", critical, critical),
    )
    print(f"episode={group.context.episode.episode_id} shared_prefix={group.context.prefix.actions}")
    print(f"normal_prompt_saved={bool(group.context.normal_prompt)} cooldown={group.context.checkpoint.cooldown_steps}")
    print(f"continue={group.continue_trace.actions} replan={group.replan_trace.actions}")
    print("cfrp_branch_context: OK")


if __name__ == "__main__":
    main()
