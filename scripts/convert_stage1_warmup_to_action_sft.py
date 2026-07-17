"""Recover primitive expert decisions and export JanusVLN-style action SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.action_sft import ACTION_SFT_SCHEMA, make_action_sft_example
from vlnce_server.qwen3vl.vision import prepare_qwen3vl_image, qwen3vl_image_size, qwen3vl_processor_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-storage", choices=("png", "source"), default="source")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "action_sft.jsonl"
    image_cache: dict[str, str] = {}
    action_counts: Counter[str] = Counter()
    episodes = 0
    examples = 0

    with output_path.open("w", encoding="utf-8") as destination:
        for episode_id, instruction, actions, frame_paths in recover_expert_episodes(input_path):
            frame_uris = [
                _export_frame(path, output_dir / "images", image_cache, args.image_storage)
                for path in frame_paths
            ]
            for step_index, action in enumerate(actions):
                example = make_action_sft_example(
                    episode_id=episode_id,
                    step_index=step_index,
                    instruction=instruction,
                    frame_uris=frame_uris[: step_index + 1],
                    expert_action=action,
                )
                destination.write(json.dumps(example, ensure_ascii=False) + "\n")
                action_counts[action] += 1
                examples += 1
            episodes += 1

    manifest = {
        "schema": "cfrp.qwen3vl.action_sft_manifest.v1",
        "example_schema": ACTION_SFT_SCHEMA,
        "source": str(input_path.resolve()),
        "episodes": episodes,
        "examples": examples,
        "action_counts": dict(sorted(action_counts.items())),
        "image_storage": args.image_storage,
        "unique_source_frames": len(image_cache),
        "temporal_visual_contract": {
            "sampling": "janus_uniform_episode_prefix",
            "max_frames": 9,
            "current_frame_last": True,
        },
        "habitat_rgb_size": [640, 480],
        "model_image_size": list(qwen3vl_image_size()),
        "processor_kwargs": qwen3vl_processor_kwargs(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"episodes={episodes}")
    print(f"examples={examples}")
    print(f"actions={dict(sorted(action_counts.items()))}")
    print(f"action_sft_jsonl={output_path}")
    print("convert_stage1_warmup_to_action_sft: OK")
    return 0


def recover_expert_episodes(input_path: Path):
    """Yield complete primitive trajectories from chunked warmup records."""

    current_id: str | None = None
    instruction = ""
    actions: list[str] = []
    frames: list[Path] = []

    def finish():
        if current_id is None:
            return None
        if not actions or len(frames) != len(actions):
            raise ValueError(f"episode {current_id} has misaligned expert actions and frames")
        if actions[-1] != "STOP":
            raise ValueError(f"episode {current_id} is incomplete: final expert action is not STOP")
        return current_id, instruction, tuple(actions), tuple(frames)

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            request = record.get("model_input", {})
            oracle = record.get("oracle_only", {})
            episode_id = str(request.get("episode_id"))
            chunk = oracle.get("oracle_actions")
            history = request.get("visual_history_paths")
            turn_index = int(request.get("turn_index", -1))
            if not isinstance(chunk, list) or not chunk or not isinstance(history, list) or not history:
                raise ValueError(f"record {line_number} lacks oracle actions or frame history")
            if current_id is not None and episode_id != current_id:
                completed = finish()
                assert completed is not None
                yield completed
                actions, frames = [], []
            if current_id != episode_id:
                current_id = episode_id
                instruction = str(request.get("instruction", "")).strip()
            frame_dir = Path(history[-1]).resolve().parent
            if turn_index != len(actions):
                raise ValueError(f"episode {episode_id} has non-contiguous primitive steps")
            for offset, action in enumerate(chunk):
                frame = frame_dir / f"frame-{turn_index + offset:04d}.npy"
                if not frame.is_file():
                    raise ValueError(f"missing primitive frame: {frame}")
                actions.append(str(action))
                frames.append(frame)
    completed = finish()
    if completed is not None:
        yield completed


def _export_frame(source: Path, images_dir: Path, cache: dict[str, str], storage: str) -> str:
    key = str(source.resolve())
    cached = cache.get(key)
    if cached is not None:
        return cached
    if storage == "source":
        uri = source.resolve().as_uri()
    else:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        destination = images_dir / f"frame-{digest}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            import numpy as np

            prepare_qwen3vl_image(np.load(source)).save(destination)
        uri = destination.resolve().as_uri()
    cache[key] = uri
    return uri


if __name__ == "__main__":
    raise SystemExit(main())
