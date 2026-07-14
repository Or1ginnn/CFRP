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

from vlnce_server.cfrp import (
    Stage1RolloutRequest,
    advance_turn_indices,
    initialize_plan_from_instruction,
)
from vlnce_server.habitat030 import (
    Habitat030NavigationEnvironment,
    OracleActionError,
    cfrp_action_from_habitat_oracle,
)
from vlnce_server.habitat030.r2r_dataset import load_r2r_dataset
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env
from vlnce_server.habitat030.stage1_runner import (
    DEFAULT_MAX_ACTION_HISTORY,
    DEFAULT_MAX_VISUAL_HISTORY,
    FixedHistoryBuffer,
)
from vlnce_server.habitat030.temporal_history import (
    DEFAULT_MODEL_VISUAL_FRAME_COUNT,
    DEFAULT_VISUAL_CONTEXT_WINDOW,
    SlowFastVisualHistory,
    temporal_history_spec,
)
from vlnce_server.qwen3vl.vision import (
    HABITAT_RGB_HEIGHT,
    HABITAT_RGB_WIDTH,
    qwen3vl_image_size,
    qwen3vl_processor_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode-ids", help="Comma-separated R2R IDs")
    selection.add_argument(
        "--episode-count",
        type=int,
        help="Collect a deterministic contiguous shard from the split; use with --episode-offset.",
    )
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--max-visual-history", type=int, default=DEFAULT_MAX_VISUAL_HISTORY)
    parser.add_argument("--visual-context-window", type=int, default=DEFAULT_VISUAL_CONTEXT_WINDOW)
    parser.add_argument("--max-action-history", type=int, default=DEFAULT_MAX_ACTION_HISTORY)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument(
        "--oracle-goal-radius",
        type=float,
        default=None,
        help="Optional stricter oracle STOP radius; defaults to the configured R2R success distance.",
    )
    return parser.parse_args()


def target_xml(progress: str, subgoal: str, actions: Sequence[str]) -> str:
    if len(actions) == 1:
        action_xml = f"<action>{actions[0]}</action>"
    else:
        action_xml = "<actions>" + "".join(f"<action>{action}</action>" for action in actions) + "</actions>"
    return f"<progress>{progress}</progress><subgoal>{subgoal}</subgoal>{action_xml}"


def main() -> int:
    args = parse_args()
    episode_ids = _select_episode_ids(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "stage1_warmup.jsonl"
    written = 0
    complete_episodes = 0
    completed_episode_ids: list[str] = []
    with records_path.open("w", encoding="utf-8") as handle:
        try:
            for episode_id in episode_ids:
                records, complete = collect_episode(args, episode_id, output_dir)
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                complete_episodes += int(complete)
                if complete:
                    completed_episode_ids.append(episode_id)
        except Exception as exc:
            _write_collection_status(
                output_dir,
                args,
                episode_ids,
                completed_episode_ids,
                status="failed",
                error=str(exc),
            )
            raise

    _write_collection_status(
        output_dir,
        args,
        episode_ids,
        completed_episode_ids,
        status="complete",
    )

    print(f"records={written}")
    print(f"complete_episodes={complete_episodes}/{len(episode_ids)}")
    print(f"records_path={records_path}")
    print("habitat030_collect_stage1_warmup: OK")
    return 0


def _write_collection_status(
    output_dir: Path,
    args: argparse.Namespace,
    requested_episode_ids: Sequence[str],
    completed_episode_ids: Sequence[str],
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Write a machine-readable status without mislabeling partial records."""

    payload = {
        "schema": "cfrp.stage1.warmup.v1",
        "status": status,
        "split": args.split,
        "requested_episode_ids": list(requested_episode_ids),
        "completed_episode_ids": list(completed_episode_ids),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_visual_history": args.max_visual_history,
        "max_action_history": args.max_action_history,
        "visual_contract": {
            "habitat_rgb_size": [HABITAT_RGB_WIDTH, HABITAT_RGB_HEIGHT],
            "model_image_size": list(qwen3vl_image_size()),
            "processor_kwargs": qwen3vl_processor_kwargs(),
        },
    }
    if args.max_visual_history == DEFAULT_MODEL_VISUAL_FRAME_COUNT:
        payload["temporal_visual_history"] = temporal_history_spec()
    if error is not None:
        payload["error"] = error
        destination = output_dir / "collection_status.json"
    else:
        destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _select_episode_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.episode_ids is not None:
        episode_ids = tuple(item.strip() for item in args.episode_ids.split(",") if item.strip())
        if not episode_ids:
            raise ValueError("--episode-ids must contain at least one ID")
        return episode_ids
    if args.episode_count is None or args.episode_count < 1:
        raise ValueError("--episode-count must be at least one")
    if args.episode_offset < 0:
        raise ValueError("--episode-offset must not be negative")
    # Load the real split order once so repeated collection jobs form stable,
    # disjoint shards without hard-coding thousands of episode IDs in shell.
    records = load_r2r_dataset(
        dataset_root=args.dataset_root,
        split=args.split,
        scenes_dir=args.scenes_dir,
    )
    selected = records[args.episode_offset : args.episode_offset + args.episode_count]
    if len(selected) != args.episode_count:
        raise ValueError(
            f"requested shard offset={args.episode_offset} count={args.episode_count}, "
            f"but split has {len(records)} episodes"
        )
    return tuple(record.episode_id for record in selected)


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
        history = FixedHistoryBuffer.create(
            args.max_visual_history,
            args.max_action_history,
            visual_context_window=args.visual_context_window,
        ).reset(observation)
        path_history = SlowFastVisualHistory[str].create(
            context_window=args.visual_context_window,
            history_anchor_count=history.history_anchor_count,
            recent_contiguous_count=history.recent_contiguous_count,
            slow_memory_update_interval=history.slow_memory_update_interval,
        ).reset(_save_frame(observation.rgb, frames_dir, 0))
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
        raw_steps = []
        for turn_index in range(args.max_steps):
            raw_action = follower.get_next_action(env.current_episode.goals[0].position)
            action = cfrp_action_from_habitat_oracle(raw_action)
            oracle_state = wrapper.privileged_state()
            raw_steps.append(
                {
                    "turn_index": turn_index,
                    "instruction": observation.instruction,
                    "visual_history_paths": path_history.visible,
                    "action_history": history.action_history,
                    "allowed_actions": observation.allowed_actions,
                    "action": action,
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
            path_history = path_history.append(
                _save_frame(observation.rgb, frames_dir, turn_index + 1)
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
                return _label_trajectory(record, raw_steps), True
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


def _label_trajectory(record: Any, raw_steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = initialize_plan_from_instruction(record.instruction_text)
    non_stop_actions = sum(1 for step in raw_steps if step["action"] != "STOP")
    advance_indices = set(advance_turn_indices(non_stop_actions, plan))
    records = []
    cursor = 0
    decision_index = 0
    while cursor < len(raw_steps):
        raw_step = raw_steps[cursor]
        turn_index = raw_step["turn_index"]
        progress = "advance" if turn_index in advance_indices else "hold"
        chunk = [raw_step]
        if progress == "hold" and raw_step["action"] != "STOP":
            while len(chunk) < 4 and cursor + len(chunk) < len(raw_steps):
                candidate = raw_steps[cursor + len(chunk)]
                if candidate["turn_index"] in advance_indices or candidate["action"] == "STOP":
                    break
                chunk.append(candidate)
        current_point = plan.current_points()[0]
        request = Stage1RolloutRequest(
            episode_id=record.episode_id,
            request_id=decision_index,
            turn_index=turn_index,
            instruction=raw_step["instruction"],
            current_plan=plan,
            visual_history_paths=raw_step["visual_history_paths"],
            action_history=raw_step["action_history"],
            allowed_actions=raw_step["allowed_actions"],
        )
        records.append(
            {
                "model_input": request.to_dict(),
                "target_xml": target_xml(progress, current_point.text, tuple(item["action"] for item in chunk)),
                "oracle_only": {
                    **raw_step["oracle_only"],
                    "oracle_actions": [item["action"] for item in chunk],
                },
            }
        )
        if progress == "advance":
            plan = plan.advance_current()
        cursor += len(chunk)
        decision_index += 1
    return records


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
