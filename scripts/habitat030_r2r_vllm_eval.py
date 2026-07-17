"""Evaluate CFRP Stage 1 with rank-local vLLM and Habitat workers.

Artifacts follow InternNav's evaluation lifecycle: each episode appends one
progress record, writes one rollout record, and immediately emits an RGB plus
top-down-map MP4 under ``vis_<rank>/<scene>/<episode>.mp4``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.habitat030_r2r_qwen_baseline import (
    _distance,
    _metrics_to_dict,
    _minimum,
    _oracle_to_dict,
    _save_current_frame,
    compare_repetitions,
    summarize,
)
from vlnce_server.cfrp import CFRPProtocolError
from vlnce_server.habitat030 import Habitat030NavigationEnvironment
from vlnce_server.habitat030.r2r_environment import (
    R2R_MAX_EPISODE_STEPS,
    create_r2r_habitat_env,
)
from vlnce_server.habitat030.stage1_runner import (
    DEFAULT_MAX_ACTION_HISTORY,
    DEFAULT_MAX_VISUAL_HISTORY,
    FixedHistoryBuffer,
    Stage1EpisodeRunner,
)
from vlnce_server.habitat030.temporal_history import (
    DEFAULT_VISUAL_CONTEXT_WINDOW,
    SlowFastVisualHistory,
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
    action_queue_mode: str
    max_actions_during_inference: int
    rank: int


@dataclass
class _InflightModelRequest:
    request_id: int
    observation_step: int
    future: Future[str]
    executed_actions: List[str]


@dataclass(frozen=True)
class _AcceptedModelResponse:
    request_id: int
    observation_step: int
    accepted_step: int
    response_lag_steps: int
    replaced_actions: Tuple[str, ...]
    model_actions: Tuple[str, ...]
    accepted_actions: Tuple[str, ...]
    reconciled_prefix: Tuple[str, ...]


class _RollingQueueRuntime:
    """Overlap one model request with a bounded executable action prefix."""

    def __init__(
        self,
        client: VLLMStage1Client,
        runner: Stage1EpisodeRunner,
        max_actions_during_inference: int,
    ) -> None:
        self.client = client
        self.runner = runner
        self.max_actions_during_inference = max_actions_during_inference
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cfrp-stage1")
        self.inflight: Optional[_InflightModelRequest] = None
        self.active_response: Optional[_AcceptedModelResponse] = None
        self.active_executed_actions: List[str] = []
        self.request_count = 0
        self.accepted_count = 0
        self.last_completed_raw_xml: Optional[str] = None

    def prepare_next_action(
        self, executed_steps: int
    ) -> Tuple[_AcceptedModelResponse, bool]:
        """Ensure one action is executable and report whether a response was refreshed."""

        accepted_now = False
        while True:
            if self.inflight is None:
                if self.runner.pending_actions:
                    assert self.active_response is not None
                    return self.active_response, accepted_now
                self._submit(executed_steps)

            assert self.inflight is not None
            must_accept = (
                self.inflight.future.done()
                or not self.runner.pending_actions
                or len(self.inflight.executed_actions) >= self.max_actions_during_inference
            )
            if not must_accept:
                assert self.active_response is not None
                return self.active_response, accepted_now

            inflight = self.inflight
            raw_xml = inflight.future.result()
            self.last_completed_raw_xml = raw_xml
            replacement = self.runner.replace_pending_actions(
                raw_xml,
                executed_since_request=tuple(inflight.executed_actions),
            )
            self.inflight = None
            self.accepted_count += 1
            self.active_response = _AcceptedModelResponse(
                request_id=inflight.request_id,
                observation_step=inflight.observation_step,
                accepted_step=executed_steps,
                response_lag_steps=executed_steps - inflight.observation_step,
                replaced_actions=replacement.replaced_actions,
                model_actions=replacement.model_actions,
                accepted_actions=replacement.accepted_actions,
                reconciled_prefix=replacement.reconciled_prefix,
            )
            self.active_executed_actions = list(replacement.reconciled_prefix)
            accepted_now = True
            if self.runner.pending_actions:
                return self.active_response, accepted_now
            # Every predicted action already ran while this response was in
            # flight. Query the factual latest state instead of replaying one.

    def after_action(self, action: str, executed_steps: int, *, submit_next: bool) -> Optional[int]:
        """Record actual execution and launch the next factual-state request."""

        self.active_executed_actions.append(action)
        if self.inflight is not None:
            self.inflight.executed_actions.append(action)
            return None
        if submit_next:
            return self._submit(executed_steps)
        return None

    def close(self) -> None:
        if self.inflight is not None:
            self.inflight.future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _submit(self, observation_step: int) -> int:
        if self.inflight is not None:
            raise RuntimeError("only one Stage 1 rolling request may be in flight")
        request_id = self.request_count
        request = self.runner.model_request()
        if self.active_response is not None:
            reconcile = getattr(self.client, "retain_last_assistant_executed_actions", None)
            if callable(reconcile):
                reconcile(tuple(self.active_executed_actions))
        self.inflight = _InflightModelRequest(
            request_id=request_id,
            observation_step=observation_step,
            future=self.executor.submit(self.client.generate_xml, request),
            executed_actions=[],
        )
        self.request_count += 1
        return request_id


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
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=R2R_MAX_EPISODE_STEPS,
        help="Maximum executed Habitat primitive actions per episode.",
    )
    parser.add_argument("--max-visual-history", type=int, default=DEFAULT_MAX_VISUAL_HISTORY)
    parser.add_argument("--max-action-history", type=int, default=DEFAULT_MAX_ACTION_HISTORY)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--response-timeout", type=float, default=600.0)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--save-frames", action="store_true", help="Keep raw RGB frames in addition to videos")
    parser.add_argument("--save-video", action="store_true", help="Write an MP4 for every episode")
    parser.add_argument("--save-oracle-trace", action="store_true")
    parser.add_argument(
        "--action-queue-mode",
        choices=("rolling", "drain"),
        default="rolling",
        help="Replace pending actions as refreshed responses arrive, or drain each chunk first.",
    )
    parser.add_argument(
        "--max-actions-during-inference",
        type=int,
        default=1,
        help="Maximum old queued actions executed while one rolling request is in flight.",
    )
    parser.add_argument(
        "--internnav-layout",
        action="store_true",
        help="Use output-dir itself as the results directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_ids = tuple(item.strip() for item in args.episode_ids.split(",") if item.strip())
    if not episode_ids:
        raise ValueError("--episode-ids must contain at least one ID")
    if args.workers < 1 or args.repeat < 1 or args.max_steps < 1 or args.rank < 0:
        raise ValueError("workers, repeat, max-steps, and rank must be valid")
    if args.max_actions_during_inference < 0:
        raise ValueError("--max-actions-during-inference must not be negative")

    run_dir = Path(args.output_dir)
    if not args.internnav_layout:
        run_dir = run_dir / "vllm-stage1-{}".format(int(time.time()))
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        _make_job(args, run_dir, episode_id, repeat_index)
        for repeat_index in range(args.repeat)
        for episode_id in episode_ids
    ]
    results = _run_jobs(jobs, args.workers, run_dir, args.rank)
    repetitions = []
    for repeat_index in range(args.repeat):
        episodes = [results[(repeat_index, episode_id)] for episode_id in episode_ids]
        repetitions.append(
            {"repeat_index": repeat_index, "episodes": episodes, "summary": summarize(episodes)}
        )
    report = {
        "schema": "cfrp.qwen_stage1_vllm_eval.v3",
        "episode_ids": list(episode_ids),
        "seed": args.seed,
        "repeat_count": args.repeat,
        "rank": args.rank,
        "config": {
            "workers": args.workers,
            "max_steps": args.max_steps,
            "step_unit": "habitat_primitive_action",
            "max_visual_history": args.max_visual_history,
            "max_action_history": args.max_action_history,
            "success_distance": args.success_distance,
            "vllm_base_url": args.vllm_base_url,
            "vllm_model": args.vllm_model,
            "save_frames": args.save_frames,
            "save_video": args.save_video,
            "save_oracle_trace": args.save_oracle_trace,
            "action_queue_mode": args.action_queue_mode,
            "max_actions_during_inference": args.max_actions_during_inference,
        },
        "repetitions": [
            {"repeat_index": item["repeat_index"], "summary": item["summary"]}
            for item in repetitions
        ],
        "repeatability": compare_repetitions(repetitions) if args.repeat > 1 else {"repeatable": None},
    }
    (run_dir / "summary_rank{}.json".format(args.rank)).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_rank_result(run_dir, args.rank, repetitions[0]["summary"])
    print("run_dir={}".format(run_dir))
    print(json.dumps(repetitions[0]["summary"], sort_keys=True))
    print("repeatable={}".format(report["repeatability"]["repeatable"]))
    print("habitat030_r2r_vllm_eval: OK")
    return 0


def _make_job(
    args: argparse.Namespace, run_dir: Path, episode_id: str, repeat_index: int
) -> EvaluationJob:
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
        action_queue_mode=args.action_queue_mode,
        max_actions_during_inference=args.max_actions_during_inference,
        rank=args.rank,
    )


def _run_jobs(
    jobs: Sequence[EvaluationJob], workers: int, run_dir: Path, rank: int
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    context = mp.get_context("spawn")
    results: Dict[Tuple[int, str], Dict[str, Any]] = {}
    with context.Pool(processes=workers) as pool:
        for repeat_index, episode_id, episode in pool.imap_unordered(_run_job, jobs):
            results[(repeat_index, episode_id)] = episode
            _append_rank_artifacts(run_dir, rank, episode)
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
    episode_dir = (
        Path(job.run_dir)
        / ".episodes"
        / "rank-{}".format(job.rank)
        / "repeat-{:02d}".format(job.repeat_index)
        / "episode-{}".format(job.episode_id)
    )
    frames_dir = episode_dir / "frames"
    try:
        runner = Stage1EpisodeRunner(
            wrapper,
            initial_plan=None,
            history=FixedHistoryBuffer.create(job.max_visual_history, job.max_action_history),
        )
        runner.reset()
        _write_sim_check_frame(
            Path(job.run_dir), job.rank, runner.history.visual_history[-1].rgb
        )
        frame_history = _initial_frame_history(runner, frames_dir, job.save_frames)
        video_frames = (
            [_visualization_frame(runner.history.visual_history[-1].rgb, wrapper)]
            if job.save_video
            else []
        )
        steps: List[Dict[str, Any]] = []
        minimum_distance = _distance(wrapper.metrics())
        end_reason = "max_environment_steps"
        model_decision_index = -1
        rolling_runtime = (
            _RollingQueueRuntime(client, runner, job.max_actions_during_inference)
            if job.action_queue_mode == "rolling"
            else None
        )
        for environment_step_index in range(job.max_steps):
            queried_model = False
            raw_xml = None
            accepted_response: Optional[_AcceptedModelResponse] = None
            launched_request_id: Optional[int] = None
            try:
                if rolling_runtime is not None:
                    accepted_response, queried_model = rolling_runtime.prepare_next_action(
                        environment_step_index
                    )
                    model_decision_index = accepted_response.request_id
                    step = runner.step_pending(turn_index=environment_step_index)
                else:
                    queried_model = runner.needs_model_decision
                    if queried_model:
                        model_decision_index += 1
                        raw_xml = client.generate_xml(runner.model_request())
                        step = runner.step(raw_xml, turn_index=environment_step_index)
                    else:
                        step = runner.step_pending(turn_index=environment_step_index)
            except CFRPProtocolError as exc:
                if rolling_runtime is not None:
                    raw_xml = rolling_runtime.last_completed_raw_xml
                end_reason = "invalid_xml_or_action"
                steps.append(
                    {
                        "turn_index": environment_step_index,
                        "environment_step_index": environment_step_index,
                        "model_decision_index": model_decision_index,
                        "queried_model": queried_model,
                        "raw_xml": raw_xml,
                        "protocol_error": str(exc),
                    }
                )
                break
            except Exception as exc:
                if rolling_runtime is not None:
                    raw_xml = rolling_runtime.last_completed_raw_xml
                end_reason = "model_error"
                steps.append(
                    {
                        "turn_index": environment_step_index,
                        "environment_step_index": environment_step_index,
                        "model_decision_index": model_decision_index,
                        "queried_model": queried_model,
                        "model_error": "{}: {}".format(
                            type(exc).__name__, exc
                        ),
                    }
                )
                break
            terminal_step = step.action == "STOP" or step.episode_over
            if rolling_runtime is not None:
                launched_request_id = rolling_runtime.after_action(
                    step.action,
                    environment_step_index + 1,
                    submit_next=(
                        not terminal_step and environment_step_index + 1 < job.max_steps
                    ),
                )
            minimum_distance = _minimum(minimum_distance, _distance(step.metrics))
            step_record = {
                # Kept for artifact compatibility; one turn is one executed
                # Habitat primitive in the current Stage 1 runner.
                "turn_index": environment_step_index,
                "environment_step_index": environment_step_index,
                "model_decision_index": model_decision_index,
                "queried_model": queried_model,
                "raw_xml": step.raw_xml,
                "progress": step.progress,
                "subgoal": step.subgoal,
                "action": step.action,
                "chunk_index": step.chunk_index,
                "chunk_size": step.chunk_size,
                "habitat_action": step.habitat_action,
                "plan_xml": step.plan_xml,
                "history": {
                    "visual_count": step.history_visual_count,
                    "action_count": step.history_action_count,
                    "rgb_paths": list(frame_history.visible) if frame_history is not None else [],
                },
                "metrics": _metrics_to_dict(step.metrics),
                "agent_pose": _pose_to_dict(wrapper.agent_pose()),
            }
            if accepted_response is not None:
                step_record["rolling_queue"] = {
                    "mode": job.action_queue_mode,
                    "request_id": accepted_response.request_id,
                    "request_observation_step": accepted_response.observation_step,
                    "response_accepted_step": accepted_response.accepted_step,
                    "response_lag_steps": accepted_response.response_lag_steps,
                    "response_accepted_before_action": queried_model,
                    "replaced_actions": list(accepted_response.replaced_actions),
                    "model_actions": list(accepted_response.model_actions),
                    "accepted_actions": list(accepted_response.accepted_actions),
                    "reconciled_prefix": list(accepted_response.reconciled_prefix),
                    "launched_request_after_action": launched_request_id,
                    "inflight_request_after_action": (
                        rolling_runtime.inflight.request_id
                        if rolling_runtime is not None and rolling_runtime.inflight is not None
                        else None
                    ),
                }
            if job.save_oracle_trace:
                step_record["oracle_only"] = _oracle_to_dict(wrapper.privileged_state())
            steps.append(step_record)
            if job.save_frames:
                frame_history = frame_history.append(
                    _save_current_frame(
                        runner.history.visual_history[-1].rgb,
                        frames_dir,
                        environment_step_index + 1,
                    )
                )
            if job.save_video:
                video_frames.append(
                    _visualization_frame(runner.history.visual_history[-1].rgb, wrapper)
                )
            termination_reason = _stage1_termination_reason(step)
            if termination_reason is not None:
                end_reason = termination_reason
                break

        final_metrics = wrapper.metrics()
        environment_steps = sum("action" in item for item in steps)
        result = {
            "episode_id": job.episode_id,
            "scene_id": record.scene_id,
            "instruction": record.instruction_text,
            "end_reason": end_reason,
            "environment_steps": environment_steps,
            "model_decisions": (
                rolling_runtime.accepted_count
                if rolling_runtime is not None
                else model_decision_index + 1
            ),
            "model_requests": (
                rolling_runtime.request_count
                if rolling_runtime is not None
                else model_decision_index + 1
            ),
            "action_queue_mode": job.action_queue_mode,
            "steps": steps,
            "final_metrics": _metrics_to_dict(final_metrics),
            "navigation_error": final_metrics.distance_to_goal,
            "oracle_success": bool(
                minimum_distance is not None and minimum_distance <= job.success_distance
            ),
            "stop_correct": bool(
                end_reason == "stop" and (final_metrics.success or 0.0) >= 1.0
            ),
            "invalid_output": end_reason == "invalid_xml_or_action",
        }
        if job.save_video:
            result["video_path"] = _write_internnav_video(
                Path(job.run_dir), job.rank, result, video_frames
            )
        if job.save_frames:
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "trajectory.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        else:
            shutil.rmtree(episode_dir, ignore_errors=True)
        return job.repeat_index, job.episode_id, result
    finally:
        if "rolling_runtime" in locals() and rolling_runtime is not None:
            rolling_runtime.close()
        wrapper.close()


def _initial_frame_history(
    runner: Stage1EpisodeRunner, frames_dir: Path, save_frames: bool
) -> Optional[SlowFastVisualHistory[str]]:
    if not save_frames:
        return None
    return SlowFastVisualHistory[str].create(
        context_window=DEFAULT_VISUAL_CONTEXT_WINDOW
    ).reset(_save_current_frame(runner.history.visual_history[-1].rgb, frames_dir, 0))


def _stage1_termination_reason(step: Any) -> Optional[str]:
    if step.action == "STOP":
        return "stop"
    if step.episode_over:
        return "environment_episode_over"
    return None


def _visualization_frame(rgb: Any, wrapper: Habitat030NavigationEnvironment) -> Any:
    import numpy as np
    from habitat.utils.visualizations.utils import observations_to_image
    from PIL import Image

    metrics = wrapper.raw_metrics()
    if metrics.get("top_down_map") is None:
        raise RuntimeError("top_down_map measurement is unavailable")
    rgb_array = np.asarray(rgb)
    internnav_frame = observations_to_image({"rgb": rgb_array}, metrics)
    map_panel = internnav_frame[:, rgb_array.shape[1] :]
    if map_panel.size == 0:
        raise RuntimeError("InternNav visualization did not produce a map panel")
    map_panel = np.asarray(
        Image.fromarray(map_panel).resize(
            (rgb_array.shape[1], rgb_array.shape[0]), resample=Image.Resampling.NEAREST
        )
    )
    return np.concatenate((rgb_array, map_panel), axis=1)


def _write_sim_check_frame(run_dir: Path, rank: int, rgb: Any) -> None:
    from PIL import Image

    path = run_dir / "check_sim_0" / "rgb_{}.jpg".format(rank)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _write_internnav_video(
    run_dir: Path, rank: int, result: Dict[str, Any], frames: Sequence[Any]
) -> str:
    from habitat.utils.visualizations.utils import images_to_video

    scene_id = Path(str(result["scene_id"])).parent.name
    video_dir = run_dir / "vis_{}".format(rank) / scene_id
    video_name = "{:04d}".format(int(result["episode_id"]))
    images_to_video(list(frames), str(video_dir), video_name, fps=6, quality=9)
    return str(video_dir / (video_name + ".mp4"))


def _append_rank_artifacts(run_dir: Path, rank: int, result: Dict[str, Any]) -> None:
    metrics = result["final_metrics"]
    progress = {
        "scene_id": Path(str(result["scene_id"])).parent.name,
        "episode_id": int(result["episode_id"]),
        "success": float(metrics.get("success") or 0.0),
        "spl": float(metrics.get("spl") or 0.0),
        "os": float(bool(result["oracle_success"])),
        "ne": float(result["navigation_error"] or 0.0),
        "steps": int(result["environment_steps"]),
        "model_decisions": int(result["model_decisions"]),
        "episode_instruction": result["instruction"],
        "end_reason": result["end_reason"],
        "video_path": result.get("video_path"),
    }
    with (run_dir / "progress_rank{}.json".format(rank)).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(progress) + "\n")
    with (run_dir / "trajectories_rank{}.jsonl".format(rank)).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")


def _write_rank_result(run_dir: Path, rank: int, summary: Dict[str, Any]) -> None:
    result = {
        "sucs_all": float(summary["sr"]),
        "spls_all": float(summary["spl"]),
        "oss_all": float(summary["oracle_success"]),
        "nes_all": float(summary["navigation_error"]),
        "length": int(summary["episodes"]),
    }
    (run_dir / "result_rank{}.json".format(rank)).write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )


def _pose_to_dict(
    pose: Tuple[Tuple[float, ...], Tuple[float, ...]]
) -> Dict[str, List[float]]:
    return {"position": list(pose[0]), "rotation": list(pose[1])}


if __name__ == "__main__":
    raise SystemExit(main())
