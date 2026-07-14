"""Run a complete multi-GPU CFRP Stage 1 Habitat evaluation.

This is the single entrypoint for a trained Stage 1 adapter: it starts a local
vLLM server, waits for health, shards a frozen R2R split over simulator GPUs,
writes InternNav-style videos/JSONL artifacts, aggregates the final metrics,
and shuts the server down.  It deliberately defaults to local/offline model
paths so evaluation cannot silently download a different base model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed"
DEFAULT_SCENES_DIR = ROOT / "data/scene_datasets"
DEFAULT_CONFIG_CANDIDATES = (
    ROOT
    / "third_party/habitat-lab-v0.3.0/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml",
    ROOT
    / "third_party/habitat-lab-0.3.0/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter", required=True, help="Final or checkpoint LoRA adapter directory"
    )
    parser.add_argument("--output-dir", required=True, help="Must not already exist")
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--scenes-dir", default=str(DEFAULT_SCENES_DIR))
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--episode-ids", help="Optional comma-separated frozen episode IDs")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="First GPU serves vLLM; the rest run Habitat",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--habitat-python", default=sys.executable)
    parser.add_argument("--vllm-env", help="Conda environment root containing bin/vllm and lib/")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-model-name", default="cfrp-stage1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--hf-home", default=str(ROOT / "cache/huggingface"))
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--max-visual-history", type=int, default=9)
    parser.add_argument("--max-action-history", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--response-timeout", type=float, default=600.0)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-oracle-trace", action="store_true")
    parser.add_argument("--health-timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir()
    (output_dir / "commands").mkdir()
    vllm_process: Optional[subprocess.Popen] = None
    try:
        config = _resolve_config(args.config)
        episode_ids = _resolve_episode_ids(args, Path(args.dataset_root))
        gpus = _parse_gpus(args.gpus)
        shards = _partition_episode_ids(episode_ids, len(gpus) - 1)
        _write_launch_manifest(output_dir, args, config, episode_ids, gpus, shards)
        if args.dry_run:
            _update_launch_status(output_dir, "dry_run")
            print(f"output_dir={output_dir}")
            print(f"episodes={len(episode_ids)}")
            print(f"simulator_gpus={','.join(gpus[1:])}")
            print("evaluate_stage1_dry_run: OK")
            return 0

        vllm_process = _start_vllm(args, output_dir, gpus[0])
        _wait_for_vllm(args, vllm_process, output_dir)
        _update_launch_status(output_dir, "evaluating")
        evaluators = _start_evaluators(args, output_dir, config, shards, gpus[1:])
        _wait_for_evaluators(evaluators)
        summary = _merge_and_aggregate_results(output_dir, episode_ids)
        _write_json(output_dir / "summary.json", summary)
        _update_launch_status(output_dir, "complete")
        print(f"output_dir={output_dir}")
        print(json.dumps(summary, sort_keys=True))
        print("evaluate_stage1: OK")
    except BaseException as exc:
        _update_launch_status(output_dir, "failed", error=str(exc))
        raise
    finally:
        if vllm_process is not None:
            _terminate(vllm_process)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.episode_count < 1 or args.workers_per_gpu < 1:
        raise ValueError("episode-count and workers-per-gpu must be positive")
    if args.max_lora_rank < 1 or args.max_model_len < 1 or args.max_num_seqs < 1:
        raise ValueError("vLLM capacity arguments must be positive")
    if args.max_visual_history != 9:
        raise ValueError("Stage 1 evaluation requires the fixed 6+3 visual contract (9 frames)")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    for path in (args.adapter, args.model, args.dataset_root, args.scenes_dir):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if not Path(args.hf_home).exists():
        raise FileNotFoundError(args.hf_home)
    _validate_adapter_rank(Path(args.adapter), args.max_lora_rank)
    if Path(args.output_dir).exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {args.output_dir}")


def _validate_adapter_rank(adapter: Path, maximum: int) -> None:
    path = adapter / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rank = int(payload.get("r", 0))
    if rank < 1:
        raise ValueError(f"invalid LoRA rank in {path}: {rank}")
    if rank > maximum:
        raise ValueError(
            f"adapter rank {rank} exceeds --max-lora-rank {maximum}"
        )


def _resolve_config(value: Optional[str]) -> Path:
    if value is not None:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("no Habitat 0.3 PointNav config found; pass --config")


def _resolve_episode_ids(args: argparse.Namespace, dataset_root: Path) -> List[str]:
    if args.episode_ids:
        values = [item.strip() for item in args.episode_ids.split(",") if item.strip()]
        if not values:
            raise ValueError("--episode-ids was empty")
        return values
    split_path = dataset_root / args.split / f"{args.split}.json.gz"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    with gzip.open(split_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"dataset lacks episodes list: {split_path}")
    values = [str(item["episode_id"]) for item in episodes[: args.episode_count]]
    if len(values) < args.episode_count:
        raise ValueError(f"requested {args.episode_count} episodes, found only {len(values)}")
    return values


def _parse_gpus(value: str) -> List[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus needs one vLLM GPU plus at least one distinct Habitat GPU")
    return gpus


def _partition_episode_ids(episode_ids: Sequence[str], shards: int) -> List[List[str]]:
    return [list(episode_ids[index::shards]) for index in range(shards)]


def _write_launch_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    config: Path,
    episode_ids: Sequence[str],
    gpus: Sequence[str],
    shards: Sequence[Sequence[str]],
) -> None:
    for rank, shard in enumerate(shards, start=1):
        (output_dir / f"episode_ids_rank{rank}.txt").write_text(
            "\n".join(shard) + "\n", encoding="utf-8"
        )
    payload = {
        "schema": "cfrp.stage1.evaluation.v1",
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "scenes_dir": str(Path(args.scenes_dir).resolve()),
        "config": str(config),
        "split": args.split,
        "episode_ids": list(episode_ids),
        "gpus": list(gpus),
        "workers_per_gpu": args.workers_per_gpu,
        "repeat": 1,
        "max_steps": args.max_steps,
        "max_visual_history": args.max_visual_history,
        "max_action_history": args.max_action_history,
        "save_video": args.save_video,
        "save_frames": args.save_frames,
        "save_oracle_trace": args.save_oracle_trace,
        "max_lora_rank": args.max_lora_rank,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "status": "starting",
        "started_at": int(time.time()),
        "temporal_visual_contract": "6 slow-memory anchors + 3 recent contiguous frames",
    }
    (output_dir / "launch_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _vllm_command(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    environment = dict(os.environ)
    if args.vllm_env:
        root = Path(args.vllm_env).resolve()
        executable = root / "bin/vllm"
        if not executable.is_file():
            raise FileNotFoundError(executable)
        environment["PATH"] = str(root / "bin") + os.pathsep + environment.get("PATH", "")
        environment["LD_LIBRARY_PATH"] = str(root / "lib") + os.pathsep + environment.get(
            "LD_LIBRARY_PATH", ""
        )
        libstdcpp = root / "lib/libstdc++.so.6"
        if libstdcpp.is_file():
            environment["LD_PRELOAD"] = str(libstdcpp) + os.pathsep + environment.get(
                "LD_PRELOAD", ""
            ).strip(os.pathsep)
        command = [str(executable)]
    else:
        command = [args.vllm_bin]
    command.extend(
        [
            "serve",
            str(Path(args.model).resolve()),
            "--served-model-name",
            args.vllm_model_name,
            "--enable-lora",
            "--max-loras",
            "1",
            "--max-lora-rank",
            str(args.max_lora_rank),
            "--lora-modules",
            f"{args.vllm_model_name}={Path(args.adapter).resolve()}",
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--limit-mm-per-prompt",
            '{"image":9}',
            "--port",
            str(args.vllm_port),
            "--seed",
            str(args.seed),
            "--host",
            "127.0.0.1",
        ]
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": _parse_gpus(args.gpus)[0],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(Path(args.hf_home).resolve()),
            "VLLM_BATCH_INVARIANT": "1",
            # The server's legacy nvcc cannot compile FlashInfer for compute_89.
            # Greedy evaluation is numerically unaffected by this sampler fallback.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    return command, environment


def _start_vllm(args: argparse.Namespace, output_dir: Path, _gpu: str) -> subprocess.Popen:
    command, environment = _vllm_command(args)
    log_path = output_dir / "logs/vllm.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
    (output_dir / "commands/vllm.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    return process


def _wait_for_vllm(args: argparse.Namespace, process: subprocess.Popen, output_dir: Path) -> None:
    endpoint = f"http://127.0.0.1:{args.vllm_port}/health"
    deadline = time.monotonic() + args.health_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(_server_failure_message(output_dir, process.returncode))
        try:
            with urlopen(endpoint, timeout=2.0) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(2.0)
    raise TimeoutError(
        f"vLLM did not become healthy within {args.health_timeout}s; "
        f"see {output_dir / 'logs/vllm.log'}"
    )


def _server_failure_message(output_dir: Path, returncode: Optional[int]) -> str:
    log_path = output_dir / "logs/vllm.log"
    tail = ""
    if log_path.is_file():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    return f"vLLM exited with code {returncode}; see {log_path}\n{tail}"


def _start_evaluators(
    args: argparse.Namespace,
    output_dir: Path,
    config: Path,
    shards: Sequence[Sequence[str]],
    simulator_gpus: Sequence[str],
) -> List[subprocess.Popen]:
    processes: List[subprocess.Popen] = []
    try:
        for rank, (gpu, episode_ids) in enumerate(
            zip(simulator_gpus, shards), start=1
        ):
            command = [
                args.habitat_python,
                str(ROOT / "scripts/habitat030_r2r_vllm_eval.py"),
                "--dataset-root",
                str(Path(args.dataset_root).resolve()),
                "--scenes-dir",
                str(Path(args.scenes_dir).resolve()),
                "--config",
                str(config),
                "--episode-ids",
                ",".join(episode_ids),
                "--output-dir",
                str(output_dir),
                "--internnav-layout",
                "--split",
                args.split,
                "--repeat",
                "1",
                "--workers",
                str(args.workers_per_gpu),
                "--rank",
                str(rank),
                "--max-steps",
                str(args.max_steps),
                "--max-visual-history",
                str(args.max_visual_history),
                "--max-action-history",
                str(args.max_action_history),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--response-timeout",
                str(args.response_timeout),
                "--success-distance",
                str(args.success_distance),
                "--seed",
                str(args.seed),
            ]
            if args.save_video:
                command.append("--save-video")
            if args.save_frames:
                command.append("--save-frames")
            if args.save_oracle_trace:
                command.append("--save-oracle-trace")
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "EGL_PLATFORM": "surfaceless",
                    "PYTHONPATH": str(ROOT),
                }
            )
            log_path = output_dir / f"logs/eval_rank{rank}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
            (output_dir / f"commands/eval_rank{rank}.json").write_text(
                json.dumps(command, indent=2) + "\n", encoding="utf-8"
            )
            processes.append(process)
    except BaseException:
        for process in processes:
            _terminate(process)
        raise
    return processes


def _wait_for_evaluators(processes: Iterable[subprocess.Popen]) -> None:
    processes = list(processes)
    pending = set(processes)
    try:
        while pending:
            for process in tuple(pending):
                returncode = process.poll()
                if returncode is None:
                    continue
                pending.remove(process)
                if returncode != 0:
                    raise RuntimeError(
                        f"evaluation rank pid={process.pid} exited with code {returncode}"
                    )
            if pending:
                time.sleep(0.5)
    finally:
        for process in processes:
            if process.poll() is None:
                _terminate(process)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _merge_and_aggregate_results(
    output_dir: Path, expected_episode_ids: Sequence[str]
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("trajectories_rank*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if len(records) != len(expected_episode_ids):
        raise RuntimeError(
            f"expected {len(expected_episode_ids)} trajectories, found {len(records)}"
        )
    observed = Counter(str(record["episode_id"]) for record in records)
    if observed != Counter(expected_episode_ids):
        raise RuntimeError("trajectory episode IDs do not match the frozen evaluation set")
    records.sort(key=lambda item: (str(item["scene_id"]), int(item["episode_id"])))
    _write_jsonl(output_dir / "trajectories.jsonl", records)

    progress = []
    for path in sorted(output_dir.glob("progress_rank*.json")):
        with path.open(encoding="utf-8") as handle:
            progress.extend(json.loads(line) for line in handle if line.strip())
    if len(progress) != len(expected_episode_ids):
        raise RuntimeError(
            f"expected {len(expected_episode_ids)} progress records, found {len(progress)}"
        )
    progress.sort(key=lambda item: (str(item["scene_id"]), int(item["episode_id"])))
    _write_jsonl(output_dir / "progress.json", progress)

    summary = _summarize_records(records)
    _write_json(
        output_dir / "result.json",
        {
            "sucs_all": summary["sr"],
            "spls_all": summary["spl"],
            "oss_all": summary["oracle_success"],
            "nes_all": summary["navigation_error"],
            "length": int(summary["episodes"]),
        },
    )
    return summary


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _update_launch_status(output_dir: Path, status: str, error: Optional[str] = None) -> None:
    path = output_dir / "launch_manifest.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    if status in {"complete", "failed", "dry_run"}:
        payload["finished_at"] = int(time.time())
    if error:
        payload["error"] = error
    _write_json(path, payload)


def _summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    count = float(len(records))
    if not count:
        raise ValueError("cannot summarize no trajectories")
    successes = [float(record["final_metrics"].get("success") or 0.0) for record in records]
    spls = [float(record["final_metrics"].get("spl") or 0.0) for record in records]
    errors = [float(record.get("navigation_error") or 0.0) for record in records]
    return {
        "episodes": count,
        "sr": sum(successes) / count,
        "spl": sum(spls) / count,
        "navigation_error": sum(errors) / count,
        "oracle_success": sum(
            float(bool(record.get("oracle_success"))) for record in records
        )
        / count,
        "invalid_output_rate": sum(
            float(bool(record.get("invalid_output"))) for record in records
        )
        / count,
        "stop_correct_rate": sum(
            float(bool(record.get("stop_correct"))) for record in records
        )
        / count,
        "average_steps": sum(len(record.get("steps", ())) for record in records) / count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
