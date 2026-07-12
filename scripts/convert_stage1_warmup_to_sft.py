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

from vlnce_server.qwen3vl.sft_data import make_stage1_sft_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    images_dir = output_dir / "images"
    output_path = output_dir / "stage1_sft.jsonl"
    count = 0
    image_cache: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            image_uris = _export_images(record, images_dir, image_cache)
            example = make_stage1_sft_example(record, image_uris)
            destination.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cfrp.qwen3vl.stage1_sft_manifest.v1",
                "examples": count,
                "unique_images": len(image_cache),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"examples={count}")
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
            Image.fromarray(array).save(destination)
            image_uri = destination.resolve().as_uri()
            image_cache[source] = image_uri
        image_uris.append(image_uri)
    return image_uris


if __name__ == "__main__":
    raise SystemExit(main())
