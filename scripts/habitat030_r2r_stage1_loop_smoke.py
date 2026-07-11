from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp import PlanPoint, PlanState
from vlnce_server.habitat030 import Habitat030NavigationEnvironment
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env
from vlnce_server.habitat030.stage1_runner import FixedHistoryBuffer, Stage1EpisodeRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted Stage 1 loop on real R2R-CE Habitat 0.3.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--max-visual-history", required=True, type=int)
    parser.add_argument("--max-action-history", required=True, type=int)
    return parser.parse_args()


def scripted_outputs() -> tuple[str, ...]:
    return (
        "<progress>hold</progress>"
        "<subgoal>inspect the current route</subgoal>"
        "<action>TURN_LEFT</action>",
        "<progress>advance</progress>"
        "<subgoal>continue through the route</subgoal>"
        "<action>MOVE_FORWARD</action>",
        "<progress>hold</progress>"
        "<subgoal>stop for scripted smoke completion</subgoal>"
        "<action>STOP</action>",
    )


def initial_plan() -> PlanState:
    return PlanState(
        global_goal="follow the R2R instruction",
        points=(
            PlanPoint(id="p1", status="current", text="inspect the current route"),
            PlanPoint(id="p2", status="todo", text="continue through the route"),
            PlanPoint(id="p3", status="todo", text="stop for scripted smoke completion"),
        ),
    )


def rgb_shape(rgb: Any) -> Any:
    return getattr(rgb, "shape", None)


def assert_history_oracle_free(observations: Iterable[object]) -> None:
    forbidden = (
        "pose",
        "agent_position",
        "agent_rotation",
        "goal_positions",
        "distance_to_goal",
        "reference_path",
        "expert_path",
    )
    for observation in observations:
        leaked = [name for name in forbidden if hasattr(observation, name)]
        if leaked:
            raise RuntimeError(f"Stage 1 history leaked privileged fields: {leaked}")


def main() -> int:
    args = parse_args()
    env, record = create_r2r_habitat_env(
        config_path=args.config,
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
        split=args.split,
        episode_id=args.episode_id,
    )
    wrapper = Habitat030NavigationEnvironment(env)
    runner = Stage1EpisodeRunner(
        wrapper,
        initial_plan(),
        history=FixedHistoryBuffer.create(
            max_visual=args.max_visual_history,
            max_action=args.max_action_history,
        ),
    )
    try:
        trajectory = runner.run(scripted_outputs())
        if len(trajectory) != 3:
            raise RuntimeError(f"expected 3 trajectory steps, got {len(trajectory)}")
        final_plan = runner.controller.current_plan
        if final_plan is None:
            raise RuntimeError("runner lost controller plan")
        current_id = final_plan.current_points()[0].id
        if current_id != "p2":
            raise RuntimeError(f"expected p2 current after advance, got {current_id}")
        assert_history_oracle_free(runner.history.visual_history)

        print(f"r2r_episode_id={record.episode_id}")
        print("stage1_mode=OK")
        print(f"r2r_scene_path={record.scene_path}")
        if runner.initial_observation is not None:
            print(f"r2r_instruction_present={bool(runner.initial_observation.instruction)}")
            print(f"rgb_shape={rgb_shape(runner.initial_observation.rgb)}")
        for step in trajectory:
            print(
                f"turn={step.turn_index} progress={step.progress} "
                f"action={step.action} habitat_action={step.habitat_action} "
                f"episode_over={step.episode_over}"
            )
        print(f"plan_cursor_after_advance={current_id}")
        print(f"history_visual_max_observed={max(step.history_visual_count for step in trajectory)}")
        print(f"history_action_max_observed={max(step.history_action_count for step in trajectory)}")
        print(f"trajectory_steps={len(trajectory)}")
        print("stage1_history_oracle_free=OK")
        print("habitat030_r2r_stage1_loop_smoke: OK")
        return 0
    finally:
        wrapper.close()


if __name__ == "__main__":
    raise SystemExit(main())
