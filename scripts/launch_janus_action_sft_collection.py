"""Schedule resumable JanusVLN action-SFT shards across Habitat GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030.r2r_environment import R2R_MAX_EPISODE_STEPS
from vlnce_server.qwen3vl.action_sft import JANUS_ACTION_COLLECTION_SCHEMA


@dataclass(frozen=True)
class ShardSpec:
    index: int
    episode_offset: int
    episode_count: int

    @property
    def name(self) -> str:
        return f"shard-{self.index:04d}"


def build_shards(episode_count: int, shard_size: int) -> tuple[ShardSpec, ...]:
    if episode_count < 1 or shard_size < 1:
        raise ValueError("episode-count and shard-size must be positive")
    return tuple(
        ShardSpec(index, offset, min(shard_size, episode_count - offset))
        for index, offset in enumerate(range(0, episode_count, shard_size))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Habitat 0.3 Python executable")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-count", type=int, default=10819)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=R2R_MAX_EPISODE_STEPS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain distinct GPU IDs")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    for value in (args.python, args.dataset_root, args.scenes_dir, args.config):
        if not Path(value).exists():
            raise FileNotFoundError(value)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite collection: {output_dir}")
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shards = build_shards(args.episode_count, args.shard_size)
    write_json(
        output_dir / "launch_manifest.json",
        {
            "schema": "cfrp.qwen3vl.janus_action_sft_launch.v1",
            "status": "running",
            "split": args.split,
            "episode_count": args.episode_count,
            "shard_size": args.shard_size,
            "gpus": list(gpus),
            "workers_per_gpu": args.workers_per_gpu,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "shards": [asdict(shard) | {"name": shard.name} for shard in shards],
        },
    )

    pending: queue.Queue[ShardSpec] = queue.Queue()
    completed: list[str] = []
    skipped: list[str] = []
    failed: dict[str, int] = {}
    lock = threading.Lock()
    for shard in shards:
        shard_dir = raw_dir / shard.name
        if args.resume and is_complete_shard(shard_dir):
            skipped.append(shard.name)
        else:
            if shard_dir.exists():
                raise FileExistsError(f"remove incomplete shard before resume: {shard_dir}")
            pending.put(shard)

    def worker(gpu: str) -> None:
        while True:
            try:
                shard = pending.get_nowait()
            except queue.Empty:
                return
            environment = dict(os.environ)
            environment.update({"CUDA_VISIBLE_DEVICES": gpu, "EGL_PLATFORM": "surfaceless"})
            command = collector_command(args, raw_dir / shard.name, shard)
            with (logs_dir / f"{shard.name}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            with lock:
                if result.returncode == 0 and is_complete_shard(raw_dir / shard.name):
                    completed.append(shard.name)
                else:
                    failed[shard.name] = result.returncode
                write_run_status(output_dir, len(shards), completed, skipped, failed, pending.qsize())
            pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"janus-{gpu}-{index:02d}")
        for gpu in gpus
        for index in range(args.workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status = "complete" if not failed and len(completed) + len(skipped) == len(shards) else "failed"
    write_run_status(output_dir, len(shards), completed, skipped, failed, 0, status=status)
    print(f"status={status}")
    print(f"completed_shards={len(completed) + len(skipped)}/{len(shards)}")
    print(f"output_dir={output_dir}")
    return 0 if status == "complete" else 1


def collector_command(args: argparse.Namespace, output_dir: Path, shard: ShardSpec) -> list[str]:
    return [
        str(Path(args.python).resolve()),
        str(ROOT / "scripts/habitat030_collect_janus_action_sft.py"),
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--scenes-dir", str(Path(args.scenes_dir).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--split", args.split,
        "--episode-count", str(shard.episode_count),
        "--episode-offset", str(shard.episode_offset),
        "--output-dir", str(output_dir),
        "--seed", str(args.seed),
        "--max-steps", str(args.max_steps),
    ]


def is_complete_shard(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    data_path = path / "action_sft.jsonl"
    if not manifest_path.is_file() or not data_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("schema") == JANUS_ACTION_COLLECTION_SCHEMA and manifest.get("status") == "complete"


def write_run_status(
    output_dir: Path,
    total: int,
    completed: list[str],
    skipped: list[str],
    failed: dict[str, int],
    remaining: int,
    *,
    status: str = "running",
) -> None:
    write_json(
        output_dir / "status.json",
        {
            "schema": "cfrp.qwen3vl.janus_action_sft_launch_status.v1",
            "status": status,
            "total_shards": total,
            "completed_shards": sorted(completed),
            "skipped_shards": sorted(skipped),
            "failed_shards": dict(sorted(failed.items())),
            "remaining_shards": remaining,
        },
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
