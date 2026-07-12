"""Collect offline shortest-path Stage 1 XML warm-up examples from R2R-CE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp import PlanPoint, PlanState, Stage1RolloutRequest
from vlnce_server.habitat030 import (
    Habitat030NavigationEnvironment,
    OracleActionError,
    cfrp_action_from_habitat_oracle,
)
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env
from vlnce_server.habitat030.stage1_runner import FixedHistoryBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-ids", required=True, help="Comma-separated R2R IDs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-visual-history", type=int, default=4)
    parser.add_argument("--max-action-history", type=int, default=3)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument(
        "--oracle-goal-radius",
        type=float,
        default=None,
        help="Optional stricter oracle STOP radius; defaults to the configured R2R success distance.",
    )
    return parser.parse_args()


def initial_plan(instruction: str) -> PlanState:
    return PlanState(
        global_goal=instruction,
        points=(
            PlanPoint(id="p1", status="current", text="follow the instruction from the current view"),
            PlanPoint(id="p2", status="todo", text="continue toward the described destination"),
        ),
    )


def target_xml(action: str) -> str:
    subgoal = "stop at the destination" if action == "STOP" else "follow the instruction from the current view"
    return f"<progress>hold</progress><subgoal>{subgoal}</subgoal><action>{action}</action>"


def main() -> int:
    args = parse_args()
    episode_ids = tuple(item.strip() for item in args.episode_ids.split(",") if item.strip())
    if not episode_ids:
        raise ValueError("--episode-ids must contain at least one ID")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "stage1_warmup.jsonl"
    manifest = {
        "schema": "cfrp.stage1.warmup.v1",
        "split": args.split,
        "episode_ids": list(episode_ids),
        "seed": args.seed,
        "max_visual_history": args.max_visual_history,
        "max_action_history": args.max_action_history,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    written = 0
    complete_episodes = 0
    with records_path.open("w", encoding="utf-8") as handle:
        for episode_id in episode_ids:
            records, complete = collect_episode(args, episode_id, output_dir)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            complete_episodes += int(complete)

    print(f"records={written}")
    print(f"complete_episodes={complete_episodes}/{len(episode_ids)}")
    print(f"records_path={records_path}")
    print("habitat030_collect_stage1_warmup: OK")
    return 0


def collect_episode(args: argparse.Namespace, episode_id: str, output_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    env, record = create_r2r_habitat_env(
        config_path=args.config,
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
        split=args.split,
        episode_id=episode_id,
        seed=args.seed,
        success_distance=args.success_distance,
    )
    wrapper = Habitat030NavigationEnvironment(env)
    episode_dir = output_dir / f"episode-{episode_id}"
    frames_dir = episode_dir / "frames"
    try:
        observation = wrapper.reset()
        history = FixedHistoryBuffer.create(args.max_visual_history, args.max_action_history).reset(observation)
        frame_paths = [_save_frame(observation.rgb, frames_dir, 0)]
        task_success_distance = _task_success_distance(env, args.success_distance)
        follower_goal_radius = (
            task_success_distance
            if args.oracle_goal_radius is None
            else min(task_success_distance, args.oracle_goal_radius)
        )
        follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=follower_goal_radius,
            return_one_hot=False,
        )
        records = []
        for turn_index in range(args.max_steps):
            raw_action = follower.get_next_action(env.current_episode.goals[0].position)
            action = cfrp_action_from_habitat_oracle(raw_action)
            request = Stage1RolloutRequest(
                episode_id=episode_id,
                request_id=turn_index,
                turn_index=turn_index,
                instruction=observation.instruction,
                current_plan=initial_plan(record.instruction_text),
                visual_history_paths=tuple(frame_paths),
                action_history=history.action_history,
                allowed_actions=observation.allowed_actions,
            )
            oracle_state = wrapper.privileged_state()
            records.append(
                {
                    "model_input": request.to_dict(),
                    "target_xml": target_xml(action),
                    "oracle_only": {
                        "oracle_action": action,
                        "task_success_distance": task_success_distance,
                        "follower_goal_radius": follower_goal_radius,
                        "agent_position": list(oracle_state.agent_position),
                        "goal_positions": [list(position) for position in oracle_state.goal_positions],
                        "expert_path": [list(position) for position in oracle_state.expert_path],
                    },
                }
            )
            step = wrapper.step(action)
            observation = step.observation
            history = history.append(observation, action)
            frame_paths = _append_capped(
                frame_paths,
                _save_frame(observation.rgb, frames_dir, turn_index + 1),
                args.max_visual_history,
            )
            if step.episode_over or action == "STOP":
                success = bool(step.metrics.success and step.metrics.success >= 1.0)
                if not success:
                    raise RuntimeError(
                        f"oracle STOP was not task-successful for episode {episode_id}; "
                        f"task_success_distance={task_success_distance} "
                        f"follower_goal_radius={follower_goal_radius} "
                        f"distance_to_goal={step.metrics.distance_to_goal}"
                    )
                return records, True
        raise RuntimeError(f"oracle did not terminate episode {episode_id} within {args.max_steps} steps")
    except OracleActionError as exc:
        raise RuntimeError(f"failed to collect oracle label for episode {episode_id}: {exc}") from exc
    finally:
        wrapper.close()


def _save_frame(rgb: Any, frames_dir: Path, frame_index: int) -> str:
    import numpy as np

    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"frame-{frame_index:04d}.npy"
    np.save(path, rgb)
    return str(path)


def _append_capped(values: Sequence[str], value: str, limit: int) -> list[str]:
    return list((tuple(values) + (value,))[-limit:])


def _task_success_distance(env: Any, fallback: float) -> float:
    """Use the task's actual success threshold for oracle STOP labels."""

    value: Any = getattr(env, "config", None)
    for key in ("habitat", "task", "measurements", "success", "success_distance"):
        if value is None:
            break
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    if value is None:
        return fallback
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
