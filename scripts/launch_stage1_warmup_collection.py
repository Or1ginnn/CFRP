"""Schedule resumable Stage 1 oracle collection shards across simulator GPUs."""

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030.r2r_environment import R2R_MAX_EPISODE_STEPS


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
        raise ValueError("episode_count and shard_size must be positive")
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
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent Habitat collector processes assigned to each GPU.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=R2R_MAX_EPISODE_STEPS,
        help="Maximum executed Habitat primitive actions per episode.",
    )
    parser.add_argument("--max-visual-history", type=int, default=9)
    parser.add_argument("--max-action-history", type=int, default=8)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain distinct GPU IDs")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    for path in (args.python, args.dataset_root, args.scenes_dir, args.config):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing collection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    shards = build_shards(args.episode_count, args.shard_size)
    _write_json(
        output_dir / "launch_manifest.json",
        {
            "schema": "cfrp.stage1.full_collection.v1",
            "status": "running",
            "split": args.split,
            "episode_count": args.episode_count,
            "shard_size": args.shard_size,
            "gpus": list(gpus),
            "workers_per_gpu": args.workers_per_gpu,
            "total_workers": len(gpus) * args.workers_per_gpu,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "step_unit": "habitat_primitive_action",
            "max_visual_history": args.max_visual_history,
            "max_action_history": args.max_action_history,
            "success_distance": args.success_distance,
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
        if args.resume and _is_complete_shard(shard_dir):
            skipped.append(shard.name)
        else:
            if shard_dir.exists():
                raise FileExistsError(
                    f"incomplete shard blocks resume; inspect and remove it explicitly: {shard_dir}"
                )
            pending.put(shard)

    def worker(gpu: str) -> None:
        while True:
            try:
                shard = pending.get_nowait()
            except queue.Empty:
                return
            command = _collector_command(args, raw_dir / shard.name, shard)
            environment = dict(os.environ)
            environment.update(
                {"CUDA_VISIBLE_DEVICES": gpu, "EGL_PLATFORM": "surfaceless"}
            )
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
                if result.returncode == 0 and _is_complete_shard(raw_dir / shard.name):
                    completed.append(shard.name)
                else:
                    failed[shard.name] = result.returncode
                _write_status(
                    output_dir,
                    total=len(shards),
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    remaining=pending.qsize(),
                )
            pending.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu,),
            name=f"collector-gpu-{gpu}-worker-{worker_index:02d}",
            daemon=False,
        )
        for gpu in gpus
        for worker_index in range(args.workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status = "complete" if not failed and len(completed) + len(skipped) == len(shards) else "failed"
    _write_status(
        output_dir,
        total=len(shards),
        completed=completed,
        skipped=skipped,
        failed=failed,
        remaining=0,
        status=status,
    )
    print(f"status={status}")
    print(f"completed_shards={len(completed) + len(skipped)}/{len(shards)}")
    print(f"output_dir={output_dir}")
    return 0 if status == "complete" else 1


def _collector_command(
    args: argparse.Namespace, output_dir: Path, shard: ShardSpec
) -> list[str]:
    return [
        str(Path(args.python).resolve()),
        str(ROOT / "scripts/habitat030_collect_stage1_warmup.py"),
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--scenes-dir",
        str(Path(args.scenes_dir).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--split",
        args.split,
        "--episode-count",
        str(shard.episode_count),
        "--episode-offset",
        str(shard.episode_offset),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--max-visual-history",
        str(args.max_visual_history),
        "--max-action-history",
        str(args.max_action_history),
        "--success-distance",
        str(args.success_distance),
    ]


def _is_complete_shard(path: Path) -> bool:
    manifest = path / "manifest.json"
    records = path / "stage1_warmup.jsonl"
    if not manifest.is_file() or not records.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def _write_status(
    output_dir: Path,
    *,
    total: int,
    completed: list[str],
    skipped: list[str],
    failed: dict[str, int],
    remaining: int,
    status: str = "running",
) -> None:
    _write_json(
        output_dir / "collection_status.json",
        {
            "schema": "cfrp.stage1.full_collection_status.v1",
            "status": status,
            "total_shards": total,
            "completed_shards": sorted(completed),
            "skipped_complete_shards": sorted(skipped),
            "failed_shards": dict(sorted(failed.items())),
            "remaining_shards": remaining,
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
