"""Evaluate CFRP Stage 1 with concurrent Habitat workers and a vLLM server.

vLLM performs continuous batching across concurrent OpenAI-compatible requests.
Each worker owns one Habitat episode at a time; when it finishes, the pool
immediately starts the next episode, mirroring ActiveVLN's active-environment
rollout scheduling without importing its Habitat 0.1.7 implementation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.habitat030_r2r_qwen_baseline import (
    _append_capped,
    _distance,
    _metrics_to_dict,
    _minimum,
    _oracle_to_dict,
    _save_current_frame,
    compare_repetitions,
    summarize,
)
from vlnce_server.cfrp import CFRPProtocolError, initialize_plan_from_instruction
from vlnce_server.habitat030 import Habitat030NavigationEnvironment
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env
from vlnce_server.habitat030.stage1_runner import (
    DEFAULT_MAX_ACTION_HISTORY,
    DEFAULT_MAX_VISUAL_HISTORY,
    FixedHistoryBuffer,
    Stage1EpisodeRunner,
)
from vlnce_server.qwen3vl import VLLMStage1Client


@dataclass(frozen=True)
class EvaluationJob:
    episode_id: str
    repeat_index: int
    run_dir: str
    dataset_root: str
    scenes_dir: str
    config: str
    split: str
    seed: int
    max_steps: int
    max_visual_history: int
    max_action_history: int
    success_distance: float
    vllm_base_url: str
    vllm_model: str
    max_new_tokens: int
    response_timeout: float
    save_frames: bool
    save_video: bool
    save_oracle_trace: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-ids", required=True, help="Comma-separated frozen R2R episode IDs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-model", default="cfrp-stage1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--max-visual-history", type=int, default=DEFAULT_MAX_VISUAL_HISTORY)
    parser.add_argument("--max-action-history", type=int, default=DEFAULT_MAX_ACTION_HISTORY)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--response-timeout", type=float, default=600.0)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--save-frames", action="store_true", help="Persist RGB frames for replay")
    parser.add_argument("--save-video", action="store_true", help="Persist RGB/top-down-map composite frames for MP4 rendering")
    parser.add_argument("--save-oracle-trace", action="store_true", help="Persist privileged debug state in trajectories")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_ids = tuple(item.strip() for item in args.episode_ids.split(",") if item.strip())
    if not episode_ids:
        raise ValueError("--episode-ids must contain at least one ID")
    if args.workers < 1 or args.repeat < 1 or args.max_steps < 1:
        raise ValueError("workers, repeat, and max-steps must be positive")

    run_dir = Path(args.output_dir) / "vllm-stage1-{}".format(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=False)
    jobs = [
        _make_job(args, run_dir, episode_id, repeat_index)
        for repeat_index in range(args.repeat)
        for episode_id in episode_ids
    ]
    results = _run_jobs(jobs, args.workers)
    repetitions = []
    for repeat_index in range(args.repeat):
        episodes = [results[(repeat_index, episode_id)] for episode_id in episode_ids]
        repetitions.append({"repeat_index": repeat_index, "episodes": episodes, "summary": summarize(episodes)})
    report = {
        "schema": "cfrp.qwen_stage1_vllm_eval.v1",
        "episode_ids": list(episode_ids),
        "seed": args.seed,
        "repeat_count": args.repeat,
        "config": {
            "workers": args.workers,
            "max_steps": args.max_steps,
            "max_visual_history": args.max_visual_history,
            "max_action_history": args.max_action_history,
            "success_distance": args.success_distance,
            "vllm_base_url": args.vllm_base_url,
            "vllm_model": args.vllm_model,
            "save_frames": args.save_frames,
            "save_video": args.save_video,
            "save_oracle_trace": args.save_oracle_trace,
        },
        "repetitions": repetitions,
        "repeatability": compare_repetitions(repetitions) if args.repeat > 1 else {"repeatable": None},
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("run_dir={}".format(run_dir))
    print(json.dumps(repetitions[0]["summary"], sort_keys=True))
    print("repeatable={}".format(report["repeatability"]["repeatable"]))
    print("habitat030_r2r_vllm_eval: OK")
    return 0


def _make_job(args: argparse.Namespace, run_dir: Path, episode_id: str, repeat_index: int) -> EvaluationJob:
    return EvaluationJob(
        episode_id=episode_id,
        repeat_index=repeat_index,
        run_dir=str(run_dir),
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
        config=args.config,
        split=args.split,
        seed=args.seed,
        max_steps=args.max_steps,
        max_visual_history=args.max_visual_history,
        max_action_history=args.max_action_history,
        success_distance=args.success_distance,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        max_new_tokens=args.max_new_tokens,
        response_timeout=args.response_timeout,
        save_frames=args.save_frames,
        save_video=args.save_video,
        save_oracle_trace=args.save_oracle_trace,
    )


def _run_jobs(jobs: Sequence[EvaluationJob], workers: int) -> Dict[Tuple[int, str], Dict[str, Any]]:
    context = mp.get_context("spawn")
    results: Dict[Tuple[int, str], Dict[str, Any]] = {}
    with context.Pool(processes=workers) as pool:
        for repeat_index, episode_id, episode in pool.imap_unordered(_run_job, jobs):
            results[(repeat_index, episode_id)] = episode
            print("completed repeat={} episode={}".format(repeat_index, episode_id), flush=True)
    return results


def _run_job(job: EvaluationJob) -> Tuple[int, str, Dict[str, Any]]:
    env, record = create_r2r_habitat_env(
        config_path=job.config,
        dataset_root=job.dataset_root,
        scenes_dir=job.scenes_dir,
        split=job.split,
        episode_id=job.episode_id,
        seed=job.seed,
        success_distance=job.success_distance,
        include_top_down_map=job.save_video,
    )
    wrapper = Habitat030NavigationEnvironment(env)
    client = VLLMStage1Client(
        job.vllm_base_url,
        job.vllm_model,
        job.max_new_tokens,
        job.response_timeout,
        job.seed,
    )
    episode_dir = Path(job.run_dir) / "repeat-{:02d}".format(job.repeat_index) / "episode-{}".format(job.episode_id)
    frames_dir = episode_dir / "frames"
    try:
        runner = Stage1EpisodeRunner(
            wrapper,
            initialize_plan_from_instruction(record.instruction_text),
            history=FixedHistoryBuffer.create(job.max_visual_history, job.max_action_history),
        )
        runner.reset()
        frame_paths = _initial_frame_paths(runner, frames_dir, job.save_frames or job.save_video, wrapper, job.save_video)
        steps: List[Dict[str, Any]] = []
        minimum_distance = _distance(wrapper.metrics())
        end_reason = "max_steps"
        for turn_index in range(job.max_steps):
            try:
                step = runner.step(client.generate_xml(runner.model_request()), turn_index=turn_index)
            except CFRPProtocolError as exc:
                end_reason = "invalid_xml_or_action"
                steps.append({"turn_index": turn_index, "protocol_error": str(exc)})
                break
            except Exception as exc:
                end_reason = "model_error"
                steps.append({"turn_index": turn_index, "model_error": "{}: {}".format(type(exc).__name__, exc)})
                break
            minimum_distance = _minimum(minimum_distance, _distance(step.metrics))
            step_record = {
                "turn_index": turn_index,
                "raw_xml": step.raw_xml,
                "progress": step.progress,
                "subgoal": step.subgoal,
                "action": step.action,
                "habitat_action": step.habitat_action,
                "plan_xml": step.plan_xml,
                "history": {"visual_count": step.history_visual_count, "action_count": step.history_action_count, "rgb_paths": list(frame_paths)},
                "metrics": _metrics_to_dict(step.metrics),
                "agent_pose": _pose_to_dict(wrapper.agent_pose()),
            }
            if job.save_oracle_trace:
                step_record["oracle_only"] = _oracle_to_dict(wrapper.privileged_state())
            steps.append(step_record)
            if job.save_frames or job.save_video:
                frame_paths = _append_capped(
                    frame_paths,
                    _save_visual_frame(runner.history.visual_history[-1].rgb, frames_dir, turn_index + 1, wrapper, job.save_video),
                    job.max_visual_history,
                )
            if step.episode_over or step.action == "STOP":
                end_reason = "stop"
                break
        final_metrics = wrapper.metrics()
        result = {
            "episode_id": job.episode_id,
            "scene_id": record.scene_id,
            "instruction": record.instruction_text,
            "end_reason": end_reason,
            "steps": steps,
            "final_metrics": _metrics_to_dict(final_metrics),
            "navigation_error": final_metrics.distance_to_goal,
            "oracle_success": bool(minimum_distance is not None and minimum_distance <= job.success_distance),
            "stop_correct": bool(end_reason == "stop" and (final_metrics.success or 0.0) >= 1.0),
            "invalid_output": end_reason == "invalid_xml_or_action",
        }
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "trajectory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_activevln_style_logs(Path(job.run_dir), result, os.getpid())
        return job.repeat_index, job.episode_id, result
    finally:
        wrapper.close()


def _initial_frame_paths(runner: Stage1EpisodeRunner, frames_dir: Path, save_frames: bool, wrapper: Habitat030NavigationEnvironment, composite: bool) -> List[str]:
    if not save_frames:
        return []
    return [_save_visual_frame(runner.history.visual_history[-1].rgb, frames_dir, 0, wrapper, composite)]


def _save_visual_frame(rgb: Any, frames_dir: Path, index: int, wrapper: Habitat030NavigationEnvironment, composite: bool) -> str:
    if not composite:
        return _save_current_frame(rgb, frames_dir, index)
    import numpy as np
    from habitat.utils.visualizations import maps

    top_down = wrapper.raw_metrics().get("top_down_map")
    if top_down is None:
        raise RuntimeError("top_down_map measurement is unavailable")
    map_image = maps.colorize_draw_agent_and_fit_to_height(top_down, rgb.shape[0])
    path = frames_dir / "frame-{:04d}.npy".format(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.concatenate((rgb, map_image), axis=1))
    return str(path)


def _pose_to_dict(pose: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> Dict[str, List[float]]:
    return {"position": list(pose[0]), "rotation": list(pose[1])}


def _write_activevln_style_logs(run_dir: Path, result: Dict[str, Any], worker_id: int) -> None:
    extra_dir, log_dir = run_dir / "extra_info", run_dir / "log"
    extra_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    episode_id = result["episode_id"]
    extra = {"instruction": result["instruction"], "conversations": [{"role": "assistant", "content": step.get("raw_xml", "")} for step in result["steps"]]}
    (extra_dir / "info_{}_{}.json".format(episode_id, worker_id)).write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")
    trajectory = {"episode_id": episode_id, "scene_id": result["scene_id"], "instruction": result["instruction"], "final_metrics": result["final_metrics"], "end_reason": result["end_reason"], "steps": result["steps"]}
    (log_dir / "_traj_{}_{}.jsonl".format(episode_id, worker_id)).write_text(json.dumps(trajectory) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
