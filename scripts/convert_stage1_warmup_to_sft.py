"""Convert Habitat oracle JSONL records into portable Qwen3-VL Stage 1 SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.sft_data import (
    DEFAULT_CONVERSATION_TURNS,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    images_dir = output_dir / "images"
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
            image_uris = _export_images(record, images_dir, image_cache)
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
                "schema": "cfrp.qwen3vl.stage1_multiturn_sft_manifest.v1",
                "episodes": episodes,
                "conversation_windows": windows,
                "supervised_turns": turns,
                "max_conversation_turns": args.max_conversation_turns,
                "unique_images": len(image_cache),
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
    print(f"sft_jsonl={output_path}")
    print("convert_stage1_warmup_to_sft: OK")
    return 0


def _export_images(record: dict, images_dir: Path, image_cache: dict[str, str]) -> list[str]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("conversion requires numpy and Pillow") from exc

    request = record["model_input"]
    image_paths = request["visual_history_paths"]
    image_uris = []
    for image_path in image_paths:
        source = str(Path(image_path).resolve())
        image_uri = image_cache.get(source)
        if image_uri is None:
            array = np.load(source)
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
            destination = images_dir / f"frame-{digest}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).convert("RGB").resize(
                qwen3vl_image_size(), resample=Image.Resampling.LANCZOS
            ).save(destination)
            image_uri = destination.resolve().as_uri()
            image_cache[source] = image_uri
        image_uris.append(image_uri)
    return image_uris


if __name__ == "__main__":
    raise SystemExit(main())
