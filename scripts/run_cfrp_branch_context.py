#!/usr/bin/env python
"""Show the structure of one same-state continue/replan counterfactual group."""

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
    make_trajectory_prefix,
)


def main() -> None:
    prefix = make_trajectory_prefix(
        poses=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        actions=("MOVE_FORWARD",),
        path_length=1.4,
        collisions=0,
        elapsed_steps=1,
        metrics={"distance_to_goal": 5.0},
    )
    context = BranchContext(
        checkpoint=CFRPCheckpoint(
            agent_position=(1.0, 0.0, 1.0),
            agent_rotation=(1.0, 0.0, 0.0, 0.0),
            current_plan=None,
            controller_action_history=("MOVE_FORWARD",),
            recent_observation_history=("t0",),
            recent_action_history=("MOVE_FORWARD",),
            turn_index=1,
            episode_id="demo-episode",
        ),
        episode=EpisodeReference(
            episode_id="demo-episode",
            instruction="Reach the kitchen.",
            goal_description="kitchen",
            success_distance=3.0,
            expert_path=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)),
        ),
        prefix=prefix,
        baseline=CriticalStateBaseline(5.0, 0.2, 1),
    )
    continue_trace = BranchTraceRecorder(
        forced_tool="continue",
        first_output_xml="<tool>continue</tool><action>MOVE_FORWARD</action>",
        start_pose=prefix.poses[-1],
    )
    continue_trace.record_step(action="MOVE_FORWARD", pose=(1.5, 0.0, 1.5))
    replan_trace = BranchTraceRecorder(
        forced_tool="replan",
        first_output_xml="<tool>replan</tool><action>TURN_LEFT</action>",
        start_pose=prefix.poses[-1],
    )
    replan_trace.record_step(action="TURN_LEFT", pose=(1.0, 0.0, 1.0))
    group = CounterfactualGroup(context, continue_trace.finish(), replan_trace.finish())
    print(f"episode={group.context.episode.episode_id} prefix_actions={group.context.prefix.actions}")
    print(f"continue_actions={group.continue_trace.actions}")
    print(f"replan_actions={group.replan_trace.actions}")
    print("cfrp_branch_context: OK")


if __name__ == "__main__":
    main()
