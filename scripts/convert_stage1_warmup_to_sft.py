"""Convert Habitat oracle JSONL records into portable Qwen3-VL Stage 1 SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.sft_data import (
    DEFAULT_CONVERSATION_TURNS,
    DEFAULT_STREAM_HISTORY_ANCHORS,
    make_stage1_sft_conversations,
)
from vlnce_server.qwen3vl.vision import qwen3vl_image_size, qwen3vl_processor_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-conversation-turns",
        type=int,
        default=DEFAULT_CONVERSATION_TURNS,
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=1,
        help="Parallel workers used to resize unique source frames before conversation export.",
    )
    parser.add_argument(
        "--image-storage",
        choices=("png", "source"),
        default="png",
        help="Materialize portable PNGs or reference the existing episode NPY frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    images_dir = output_dir / "images"
    if args.image_workers < 1:
        raise ValueError("image-workers must be at least one")
    if args.image_storage == "png":
        unique_images = _export_unique_images_parallel(
            input_path,
            images_dir,
            workers=args.image_workers,
        )
    else:
        unique_images = None
    output_path = output_dir / "stage1_sft.jsonl"
    windows = 0
    turns = 0
    episodes = 0
    image_cache: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        episode_records: list[dict] = []
        episode_images: list[list[str]] = []
        episode_id: str | None = None
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            record_episode_id = str(record["model_input"]["episode_id"])
            if episode_id is not None and record_episode_id != episode_id:
                created = make_stage1_sft_conversations(
                    episode_records,
                    episode_images,
                    max_turns=args.max_conversation_turns,
                )
                for example in created:
                    destination.write(json.dumps(example, ensure_ascii=False) + "\n")
                windows += len(created)
                turns += len(episode_records)
                episodes += 1
                episode_records = []
                episode_images = []
            episode_id = record_episode_id
            image_uris = _export_images(
                record,
                images_dir,
                image_cache,
                materialize_png=args.image_storage == "png",
            )
            episode_records.append(record)
            episode_images.append(image_uris)
        if episode_records:
            created = make_stage1_sft_conversations(
                episode_records,
                episode_images,
                max_turns=args.max_conversation_turns,
            )
            for example in created:
                destination.write(json.dumps(example, ensure_ascii=False) + "\n")
            windows += len(created)
            turns += len(episode_records)
            episodes += 1
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cfrp.qwen3vl.stage1_streaming_sft_manifest.v2",
                "episodes": episodes,
                "conversation_windows": windows,
                "supervised_turns": turns,
                "max_conversation_turns": args.max_conversation_turns,
                "streaming_visual_contract": {
                    "history_anchor_count": DEFAULT_STREAM_HISTORY_ANCHORS,
                    "new_observations_per_turn": 1,
                    "max_active_dialogue_turns": DEFAULT_CONVERSATION_TURNS,
                    "max_window_images": (
                        DEFAULT_STREAM_HISTORY_ANCHORS + DEFAULT_CONVERSATION_TURNS
                    ),
                },
                "unique_images": len(image_cache),
                "image_workers": args.image_workers,
                "image_storage": args.image_storage,
                "habitat_rgb_size": [640, 480],
                "model_image_size": list(qwen3vl_image_size()),
                "processor_kwargs": qwen3vl_processor_kwargs(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"episodes={episodes}")
    print(f"conversation_windows={windows}")
    print(f"supervised_turns={turns}")
    if unique_images is not None and unique_images != len(image_cache):
        raise RuntimeError("preconverted image count does not match exported image references")
    print(f"unique_images={len(image_cache)}")
    print(f"sft_jsonl={output_path}")
    print("convert_stage1_warmup_to_sft: OK")
    return 0


def _export_images(
    record: dict,
    images_dir: Path,
    image_cache: dict[str, str],
    *,
    materialize_png: bool = True,
) -> list[str]:
    request = record["model_input"]
    image_paths = request["visual_history_paths"]
    image_uris = []
    for image_path in image_paths:
        source = str(Path(image_path).resolve())
        image_uri = image_cache.get(source)
        if image_uri is None:
            if materialize_png:
                destination = _image_destination(source, images_dir)
                if not destination.is_file():
                    _convert_image((source, str(destination)))
                image_uri = destination.resolve().as_uri()
            else:
                if not Path(source).is_file():
                    raise ValueError(f"source RGB frame is missing: {source}")
                image_uri = Path(source).as_uri()
            image_cache[source] = image_uri
        image_uris.append(image_uri)
    return image_uris


def _export_unique_images_parallel(
    input_path: Path,
    images_dir: Path,
    *,
    workers: int,
) -> int:
    """Resize every referenced source frame once before ordered JSONL export."""

    images_dir.mkdir(parents=True, exist_ok=True)
    tasks = (
        (source, str(_image_destination(source, images_dir)))
        for source in _iter_unique_image_paths(input_path)
    )
    if workers == 1:
        return sum(1 for task in tasks if _convert_image(task))
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        return sum(
            1
            for _ in pool.imap_unordered(
                _convert_image,
                tasks,
                chunksize=16,
            )
        )


def _iter_unique_image_paths(input_path: Path) -> Iterator[str]:
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                paths = record["model_input"]["visual_history_paths"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"record {line_number} has no visual history paths") from exc
            for value in paths:
                path = str(Path(value).resolve())
                if path not in seen:
                    seen.add(path)
                    yield path


def _image_destination(source: str, images_dir: Path) -> Path:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    return images_dir / f"frame-{digest}.png"


def _convert_image(task: tuple[str, str]) -> bool:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("conversion requires numpy and Pillow") from exc

    source, destination_value = task
    destination = Path(destination_value)
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        array = np.load(source)
        Image.fromarray(array).convert("RGB").resize(
            qwen3vl_image_size(), resample=Image.Resampling.LANCZOS
        ).save(destination)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
