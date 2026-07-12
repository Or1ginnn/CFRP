"""Evaluate a fixed R2R subset through split Habitat 0.3 and Qwen3-VL processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp import (
    CFRPProtocolError,
    Stage1RolloutRequest,
    initialize_plan_from_instruction,
    request_path,
    response_path,
    wait_for_response,
    write_request,
)
from vlnce_server.habitat030 import Habitat030NavigationEnvironment
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env
from vlnce_server.habitat030.stage1_runner import (
    DEFAULT_MAX_ACTION_HISTORY,
    DEFAULT_MAX_VISUAL_HISTORY,
    FixedHistoryBuffer,
    Stage1EpisodeRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-ids", required=True, help="Comma-separated frozen R2R episode IDs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-python", required=True, help="Python executable in cfrp-qwen3vl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-visual-history", type=int, default=DEFAULT_MAX_VISUAL_HISTORY)
    parser.add_argument("--max-action-history", type=int, default=DEFAULT_MAX_ACTION_HISTORY)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--response-timeout", type=float, default=600.0)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_ids = tuple(item.strip() for item in args.episode_ids.split(",") if item.strip())
    if not episode_ids:
        raise ValueError("--episode-ids must contain at least one ID")
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 for the repeatability gate")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")

    run_dir = Path(args.output_dir) / f"qwen-baseline-{int(time.time())}"
    exchange_dir = run_dir / "exchange"
    run_dir.mkdir(parents=True, exist_ok=False)
    worker = _start_worker(args, exchange_dir, run_dir)
    try:
        repetitions = []
        request_id = 0
        for repeat_index in range(args.repeat):
            episodes = []
            for episode_id in episode_ids:
                episode, request_id = run_episode(
                    args=args,
                    episode_id=episode_id,
                    repeat_index=repeat_index,
                    run_dir=run_dir,
                    exchange_dir=exchange_dir,
                    request_id=request_id,
                )
                episodes.append(episode)
            repetitions.append({"repeat_index": repeat_index, "episodes": episodes, "summary": summarize(episodes)})

        report = {
            "schema": "cfrp.qwen_stage1_baseline.v1",
            "episode_ids": list(episode_ids),
            "seed": args.seed,
            "repeat_count": args.repeat,
            "config": {
                "max_steps": args.max_steps,
                "max_visual_history": args.max_visual_history,
                "max_action_history": args.max_action_history,
                "success_distance": args.success_distance,
                "model": args.model,
                "adapter": args.adapter,
            },
            "repetitions": repetitions,
            "repeatability": compare_repetitions(repetitions),
        }
        (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"run_dir={run_dir}")
        print(json.dumps(repetitions[0]["summary"], sort_keys=True))
        print(f"repeatable={report['repeatability']['repeatable']}")
        print("habitat030_r2r_qwen_baseline: OK")
        return 0
    finally:
        _stop_worker(worker, exchange_dir)


def run_episode(
    *,
    args: argparse.Namespace,
    episode_id: str,
    repeat_index: int,
    run_dir: Path,
    exchange_dir: Path,
    request_id: int,
) -> tuple[dict[str, Any], int]:
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
    episode_dir = run_dir / f"repeat-{repeat_index:02d}" / f"episode-{episode_id}"
    frames_dir = episode_dir / "frames"
    try:
        runner = Stage1EpisodeRunner(
            wrapper,
            initialize_plan_from_instruction(record.instruction_text),
            history=FixedHistoryBuffer.create(args.max_visual_history, args.max_action_history),
        )
        runner.reset()
        frame_paths = [_save_current_frame(runner.history.visual_history[-1].rgb, frames_dir, 0)]
        steps = []
        minimum_distance = _distance(wrapper.metrics())
        end_reason = "max_steps"

        for turn_index in range(args.max_steps):
            current_plan = runner.controller.current_plan
            if current_plan is None:
                raise RuntimeError("Stage 1 controller lost its current plan")
            latest = runner.history.visual_history[-1]
            request = Stage1RolloutRequest(
                episode_id=episode_id,
                request_id=request_id,
                turn_index=turn_index,
                instruction=latest.instruction,
                current_plan=current_plan,
                visual_history_paths=tuple(frame_paths),
                action_history=runner.history.action_history,
                allowed_actions=latest.allowed_actions,
            )
            write_request(request_path(exchange_dir, request_id), request)
            response = wait_for_response(
                response_path(exchange_dir, request_id),
                timeout_seconds=args.response_timeout,
            )
            request_id += 1
            if (
                response.episode_id != episode_id
                or response.request_id != request.request_id
                or response.turn_index != turn_index
            ):
                raise RuntimeError("model response does not match the pending rollout request")
            if response.error:
                end_reason = "model_error"
                steps.append({"turn_index": turn_index, "raw_xml": response.raw_xml, "model_error": response.error})
                break
            try:
                step = runner.step(response.raw_xml, turn_index=turn_index)
            except CFRPProtocolError as exc:
                end_reason = "invalid_xml_or_action"
                steps.append(
                    {
                        "turn_index": turn_index,
                        "raw_xml": response.raw_xml,
                        "protocol_error": str(exc),
                    }
                )
                break

            minimum_distance = _minimum(minimum_distance, _distance(step.metrics))
            oracle_state = wrapper.privileged_state()
            steps.append(
                {
                    "turn_index": turn_index,
                    "raw_xml": step.raw_xml,
                    "progress": step.progress,
                    "subgoal": step.subgoal,
                    "action": step.action,
                    "habitat_action": step.habitat_action,
                    "plan_xml": step.plan_xml,
                    "history": {
                        "visual_count": step.history_visual_count,
                        "action_count": step.history_action_count,
                        "rgb_paths": list(frame_paths),
                    },
                    "metrics": _metrics_to_dict(step.metrics),
                    "oracle_only": _oracle_to_dict(oracle_state),
                }
            )
            frame_paths = _append_capped(
                frame_paths,
                _save_current_frame(runner.history.visual_history[-1].rgb, frames_dir, turn_index + 1),
                args.max_visual_history,
            )
            if step.episode_over or step.action == "STOP":
                end_reason = "stop"
                break

        final_metrics = wrapper.metrics()
        final_success = final_metrics.success or 0.0
        result = {
            "episode_id": episode_id,
            "scene_id": record.scene_id,
            "instruction": record.instruction_text,
            "end_reason": end_reason,
            "steps": steps,
            "final_metrics": _metrics_to_dict(final_metrics),
            "navigation_error": final_metrics.distance_to_goal,
            "oracle_success": bool(minimum_distance is not None and minimum_distance <= args.success_distance),
            "stop_correct": bool(end_reason == "stop" and final_success >= 1.0),
            "invalid_output": end_reason == "invalid_xml_or_action",
        }
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "trajectory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result, request_id
    finally:
        wrapper.close()


def summarize(episodes: Sequence[dict[str, Any]]) -> dict[str, float]:
    count = len(episodes)
    if not count:
        raise ValueError("cannot summarize zero episodes")
    return {
        "episodes": float(count),
        "sr": _mean(float(item["final_metrics"].get("success") or 0.0) for item in episodes),
        "spl": _mean(float(item["final_metrics"].get("spl") or 0.0) for item in episodes),
        "navigation_error": _mean_optional(item.get("navigation_error") for item in episodes),
        "oracle_success": _mean(float(bool(item["oracle_success"])) for item in episodes),
        "invalid_output_rate": _mean(float(bool(item["invalid_output"])) for item in episodes),
        "invalid_output_step_rate": sum(
            float(bool(item["invalid_output"])) for item in episodes
        )
        / sum(len(item["steps"]) for item in episodes),
        "average_steps": _mean(float(len(item["steps"])) for item in episodes),
        "stop_correct_rate": _mean(float(bool(item["stop_correct"])) for item in episodes),
    }


def compare_repetitions(repetitions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reference = repetitions[0]["episodes"]
    mismatches = []
    for repetition in repetitions[1:]:
        for expected, observed in zip(reference, repetition["episodes"]):
            expected_actions = [step.get("action") for step in expected["steps"]]
            observed_actions = [step.get("action") for step in observed["steps"]]
            if expected["end_reason"] != observed["end_reason"] or expected_actions != observed_actions:
                mismatches.append({"episode_id": expected["episode_id"], "repeat_index": repetition["repeat_index"]})
    return {"repeatable": not mismatches, "action_or_termination_mismatches": mismatches}


def _start_worker(args: argparse.Namespace, exchange_dir: Path, run_dir: Path) -> subprocess.Popen:
    worker_script = ROOT / "scripts" / "qwen3vl_stage1_file_worker.py"
    command = _worker_command(args, exchange_dir, worker_script)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    stdout = (run_dir / "qwen_worker.stdout.log").open("w", encoding="utf-8")
    stderr = (run_dir / "qwen_worker.stderr.log").open("w", encoding="utf-8")
    try:
        return subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=stdout, stderr=stderr)
    finally:
        stdout.close()
        stderr.close()


def _worker_command(args: argparse.Namespace, exchange_dir: Path, worker_script: Path) -> list[str]:
    command = [
        args.model_python,
        str(worker_script),
        "--exchange-dir",
        str(exchange_dir),
        "--model",
        args.model,
    ]
    if args.adapter is not None:
        command.extend(("--adapter", args.adapter))
    command.extend(("--max-new-tokens", str(args.max_new_tokens)))
    return command


def _stop_worker(worker: subprocess.Popen, exchange_dir: Path) -> None:
    (exchange_dir / "worker.stop").touch()
    try:
        worker.wait(timeout=30)
    except subprocess.TimeoutExpired:
        worker.terminate()
        worker.wait(timeout=30)


def _save_current_frame(rgb: Any, frames_dir: Path, frame_index: int) -> str:
    import numpy as np

    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"frame-{frame_index:04d}.npy"
    np.save(path, rgb)
    return str(path)


def _append_capped(values: Sequence[str], value: str, limit: int) -> list[str]:
    return list((tuple(values) + (value,))[-limit:])


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    return {
        "distance_to_goal": metrics.distance_to_goal,
        "success": metrics.success,
        "spl": metrics.spl,
        "path_length": metrics.path_length,
        "extra": dict(metrics.extra),
    }


def _oracle_to_dict(state: Any) -> dict[str, Any]:
    return {
        "agent_position": list(state.agent_position),
        "agent_rotation": list(state.agent_rotation),
        "goal_positions": [list(position) for position in state.goal_positions],
        "expert_path": [list(position) for position in state.expert_path],
    }


def _distance(metrics: Any) -> float | None:
    return metrics.distance_to_goal


def _minimum(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    present = tuple(float(value) for value in values if value is not None)
    return None if not present else _mean(present)


if __name__ == "__main__":
    raise SystemExit(main())
