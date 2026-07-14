"""Launch a four-rank InternNav-style CFRP evaluation on one GPU host."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030.r2r_environment import R2R_MAX_EPISODE_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--habitat-python", required=True)
    parser.add_argument("--vllm-bin", required=True)
    parser.add_argument("--vllm-lib", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--splits", default="val_seen,val_unseen")
    parser.add_argument("--workers-per-rank", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=R2R_MAX_EPISODE_STEPS,
        help="Maximum executed Habitat primitive actions per episode.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="Optional prefix limit for smoke gates; omit for full evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    gpus = _csv(args.gpus)
    splits = _csv(args.splits)
    if not gpus or args.workers_per_rank < 1:
        raise ValueError("at least one GPU and one worker per rank are required")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("gpu-memory-utilization must be in [0.1, 0.95]")

    manifest = {
        "schema": "cfrp.phase0.internnav_eval.v1",
        "status": "starting",
        "gpus": gpus,
        "splits": splits,
        "workers_per_rank": args.workers_per_rank,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_episodes_per_split": args.max_episodes_per_split,
        "max_steps": args.max_steps,
        "step_unit": "habitat_primitive_action",
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "started_at": int(time.time()),
    }
    _write_json(output_root / "run_manifest.json", manifest)

    servers: List[subprocess.Popen] = []
    evaluators: List[subprocess.Popen] = []
    log_handles = []
    try:
        for rank, gpu in enumerate(gpus):
            log_handle = (logs_dir / "vllm_rank{}.log".format(rank)).open("w")
            log_handles.append(log_handle)
            server = subprocess.Popen(
                _vllm_command(args, rank),
                cwd=str(project_root),
                env=_vllm_environment(args, gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            servers.append(server)
        for rank in range(len(gpus)):
            _wait_for_health(args.base_port + rank, servers[rank], args.startup_timeout)
            print("vllm_rank_{}=READY".format(rank), flush=True)

        manifest["status"] = "evaluating"
        _write_json(output_root / "run_manifest.json", manifest)
        for split in splits:
            episode_ids = _load_episode_ids(Path(args.dataset_root), split)
            if args.max_episodes_per_split is not None:
                if args.max_episodes_per_split < 1:
                    raise ValueError("max-episodes-per-split must be positive")
                episode_ids = episode_ids[: args.max_episodes_per_split]
            shards = [episode_ids[rank :: len(gpus)] for rank in range(len(gpus))]
            results_dir = output_root / split / "results"
            results_dir.mkdir(parents=True)
            evaluators.clear()
            for rank, (gpu, shard) in enumerate(zip(gpus, shards)):
                log_handle = (logs_dir / "{}_rank{}.log".format(split, rank)).open("w")
                log_handles.append(log_handle)
                evaluator = subprocess.Popen(
                    _evaluator_command(args, split, rank, shard, results_dir),
                    cwd=str(project_root),
                    env=_habitat_environment(gpu),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                evaluators.append(evaluator)
            failures = []
            for rank, evaluator in enumerate(evaluators):
                return_code = evaluator.wait()
                if return_code != 0:
                    failures.append((rank, return_code))
            if failures:
                raise RuntimeError("{} evaluator failures: {}".format(split, failures))
            subprocess.run(
                [
                    args.habitat_python,
                    str(project_root / "scripts/merge_phase0_eval_ranks.py"),
                    "--results-dir",
                    str(results_dir),
                    "--expected-episodes",
                    str(len(episode_ids)),
                ],
                cwd=str(project_root),
                check=True,
            )
            print("split_{}=COMPLETE episodes={}".format(split, len(episode_ids)), flush=True)

        manifest["status"] = "complete"
        manifest["finished_at"] = int(time.time())
        _write_json(output_root / "run_manifest.json", manifest)
        print("output_root={}".format(output_root))
        print("phase0_4gpu_eval: OK")
        return 0
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at"] = int(time.time())
        _write_json(output_root / "run_manifest.json", manifest)
        raise
    finally:
        for process in evaluators:
            _terminate_group(process)
        for process in servers:
            _terminate_group(process)
        for handle in log_handles:
            handle.close()


def _vllm_command(args: argparse.Namespace, rank: int) -> List[str]:
    return [
        args.vllm_bin,
        "serve",
        str(Path(args.model).resolve()),
        "--served-model-name",
        "cfrp-stage1",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        "cfrp-stage1={}".format(Path(args.adapter).resolve()),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--limit-mm-per-prompt",
        '{"image": 9}',
        "--seed",
        str(args.seed),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.base_port + rank),
    ]


def _evaluator_command(
    args: argparse.Namespace,
    split: str,
    rank: int,
    episode_ids: Sequence[str],
    results_dir: Path,
) -> List[str]:
    return [
        args.habitat_python,
        str(Path(args.project_root) / "scripts/habitat030_r2r_vllm_eval.py"),
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--scenes-dir",
        str(Path(args.scenes_dir).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--split",
        split,
        "--episode-ids",
        ",".join(episode_ids),
        "--output-dir",
        str(results_dir),
        "--vllm-base-url",
        "http://127.0.0.1:{}".format(args.base_port + rank),
        "--vllm-model",
        "cfrp-stage1",
        "--workers",
        str(args.workers_per_rank),
        "--rank",
        str(rank),
        "--repeat",
        "1",
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--max-visual-history",
        "9",
        "--max-action-history",
        "8",
        "--save-video",
        "--internnav-layout",
    ]


def _vllm_environment(args: argparse.Namespace, gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_BATCH_INVARIANT": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "LD_LIBRARY_PATH": args.vllm_lib,
            "LD_PRELOAD": str(Path(args.vllm_lib) / "libstdc++.so.6"),
        }
    )
    return env


def _habitat_environment(gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "EGL_PLATFORM": "surfaceless"})
    return env


def _wait_for_health(port: int, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = "http://127.0.0.1:{}/health".format(port)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM on port {} exited with {}".format(port, process.returncode))
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("vLLM on port {} did not become healthy".format(port))


def _load_episode_ids(dataset_root: Path, split: str) -> List[str]:
    path = dataset_root / split / (split + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return [str(episode["episode_id"]) for episode in data["episodes"]]


def _terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
