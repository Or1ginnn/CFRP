"""Collect JanusVLN-style force-expert R2R action SFT with Habitat 0.3."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030 import (
    Habitat030NavigationEnvironment,
    OracleActionError,
    cfrp_action_from_habitat_oracle,
)
from vlnce_server.habitat030.r2r_dataset import load_r2r_dataset
from vlnce_server.habitat030.r2r_environment import (
    JANUS_FINAL_WAYPOINT_RADIUS,
    JANUS_INTERMEDIATE_WAYPOINT_RADIUS,
    R2R_MAX_EPISODE_STEPS,
    R2R_SUCCESS_DISTANCE,
    create_r2r_habitat_env,
    janus_r2r_oracle_contract,
    janus_r2r_simulator_contract,
)
from vlnce_server.qwen3vl.action_sft import (
    ACTION_SFT_MAX_FRAMES,
    ACTION_SFT_SCHEMA,
    make_action_sft_example,
)
from vlnce_server.qwen3vl.vision import (
    HABITAT_RGB_HEIGHT,
    HABITAT_RGB_WIDTH,
    prepare_qwen3vl_image,
    qwen3vl_image_size,
    qwen3vl_processor_kwargs,
)


JPEG_QUALITY = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode-ids", help="Comma-separated R2R episode IDs")
    selection.add_argument("--episode-count", type=int)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=R2R_MAX_EPISODE_STEPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.max_steps > R2R_MAX_EPISODE_STEPS:
        raise ValueError(f"max-steps must be in [1, {R2R_MAX_EPISODE_STEPS}]")
    episode_ids = select_episode_ids(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "action_sft.jsonl"
    completed: list[str] = []
    action_counts: Counter[str] = Counter()
    examples = 0

    with output_path.open("w", encoding="utf-8") as destination:
        try:
            for episode_id in episode_ids:
                rows = collect_episode(args, episode_id, output_dir)
                for row in rows:
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    action_counts[row["targets"][0]["action"]] += 1
                    examples += 1
                completed.append(episode_id)
        except Exception as exc:
            write_status(
                output_dir,
                args,
                episode_ids,
                completed,
                examples,
                action_counts,
                status="failed",
                error=str(exc),
            )
            raise

    write_status(
        output_dir,
        args,
        episode_ids,
        completed,
        examples,
        action_counts,
        status="complete",
    )
    print(f"episodes={len(completed)}")
    print(f"examples={examples}")
    print(f"action_sft_jsonl={output_path}")
    print("habitat030_collect_janus_action_sft: OK")
    return 0


def select_episode_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.episode_ids is not None:
        values = tuple(value.strip() for value in args.episode_ids.split(",") if value.strip())
        if not values:
            raise ValueError("--episode-ids must contain at least one ID")
        return values
    if args.episode_count is None or args.episode_count < 1:
        raise ValueError("--episode-count must be positive")
    if args.episode_offset < 0:
        raise ValueError("--episode-offset must not be negative")
    records = load_r2r_dataset(
        dataset_root=args.dataset_root,
        split=args.split,
        scenes_dir=args.scenes_dir,
    )
    selected = records[args.episode_offset : args.episode_offset + args.episode_count]
    if len(selected) != args.episode_count:
        raise ValueError("requested episode shard exceeds the selected R2R split")
    return tuple(record.episode_id for record in selected)


def collect_episode(
    args: argparse.Namespace,
    episode_id: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    env, record = create_r2r_habitat_env(
        config_path=args.config,
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
        split=args.split,
        episode_id=episode_id,
        seed=args.seed,
        success_distance=R2R_SUCCESS_DISTANCE,
    )
    wrapper = Habitat030NavigationEnvironment(env)
    frames_dir = output_dir / f"episode-{episode_id}" / "frames"
    rows: list[dict[str, Any]] = []
    frame_uris: list[str] = []
    try:
        observation = wrapper.reset()
        waypoints = reference_waypoints(record.reference_path)
        waypoint_index = 0
        follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=waypoint_radius(waypoint_index, len(waypoints)),
            return_one_hot=False,
            stop_on_error=False,
        )
        for step_index in range(args.max_steps):
            frame_uris.append(save_model_frame(observation.rgb, frames_dir, step_index))
            while True:
                raw_action = follower.get_next_action(waypoints[waypoint_index])
                action = cfrp_action_from_habitat_oracle(raw_action)
                if action != "STOP" or waypoint_index == len(waypoints) - 1:
                    break
                waypoint_index += 1
                follower = ShortestPathFollower(
                    sim=env.sim,
                    goal_radius=waypoint_radius(waypoint_index, len(waypoints)),
                    return_one_hot=False,
                    stop_on_error=False,
                )

            rows.append(
                make_action_sft_example(
                    episode_id=record.episode_id,
                    step_index=step_index,
                    instruction=observation.instruction,
                    frame_uris=frame_uris,
                    expert_action=action,
                )
            )
            step = wrapper.step(action)
            observation = step.observation
            if action == "STOP" or step.episode_over:
                success = bool(step.metrics.success and step.metrics.success >= 1.0)
                if action != "STOP" or not success:
                    raise RuntimeError(
                        f"Janus expert did not finish episode {episode_id} successfully; "
                        f"action={action} distance_to_goal={step.metrics.distance_to_goal}"
                    )
                return rows
        raise RuntimeError(
            f"Janus expert did not terminate episode {episode_id} within {args.max_steps} steps"
        )
    except OracleActionError as exc:
        raise RuntimeError(f"failed to collect Janus expert action for episode {episode_id}: {exc}") from exc
    finally:
        wrapper.close()


def waypoint_radius(waypoint_index: int, waypoint_count: int) -> float:
    if waypoint_count < 1 or not 0 <= waypoint_index < waypoint_count:
        raise ValueError("invalid reference waypoint index")
    if waypoint_index == waypoint_count - 1:
        return JANUS_FINAL_WAYPOINT_RADIUS
    return JANUS_INTERMEDIATE_WAYPOINT_RADIUS


def reference_waypoints(
    reference_path: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    points = tuple(tuple(float(value) for value in point) for point in reference_path)
    if len(points) < 2:
        raise ValueError("R2R reference_path must contain a start and at least one waypoint")
    return points[1:]


def save_model_frame(rgb: Any, frames_dir: Path, frame_index: int) -> str:
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = (frames_dir / f"frame-{frame_index:04d}.jpg").resolve()
    prepare_qwen3vl_image(rgb).save(path, format="JPEG", quality=JPEG_QUALITY)
    return path.as_uri()


def write_status(
    output_dir: Path,
    args: argparse.Namespace,
    requested_episode_ids: Sequence[str],
    completed_episode_ids: Sequence[str],
    examples: int,
    action_counts: Counter[str],
    *,
    status: str,
    error: str | None = None,
) -> None:
    payload = {
        "schema": "cfrp.qwen3vl.janus_action_sft_collection.v1",
        "example_schema": ACTION_SFT_SCHEMA,
        "status": status,
        "split": args.split,
        "requested_episode_ids": list(requested_episode_ids),
        "completed_episode_ids": list(completed_episode_ids),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "examples": examples,
        "action_counts": dict(sorted(action_counts.items())),
        "simulator_contract": janus_r2r_simulator_contract(),
        "oracle_policy": janus_r2r_oracle_contract(),
        "visual_contract": {
            "habitat_rgb_size": [HABITAT_RGB_WIDTH, HABITAT_RGB_HEIGHT],
            "stored_model_image_size": list(qwen3vl_image_size()),
            "storage": "jpeg",
            "jpeg_quality": JPEG_QUALITY,
            "processor_kwargs": qwen3vl_processor_kwargs(),
        },
        "temporal_visual_contract": {
            "sampling": "janus_uniform_episode_prefix",
            "max_frames": ACTION_SFT_MAX_FRAMES,
            "current_frame_last": True,
        },
    }
    if error is not None:
        payload["error"] = error
    name = "manifest.json" if status == "complete" else "collection_status.json"
    (output_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
