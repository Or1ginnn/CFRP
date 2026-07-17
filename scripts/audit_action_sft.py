"""Stream-audit a complete Phase 0 action-only SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.action_sft import validate_action_sft_example
from vlnce_server.qwen3vl.sft_manifest import local_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_action_sft(Path(args.input_jsonl), check_images=args.check_images)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    print("audit_action_sft: OK")
    return 0


def audit_action_sft(path: Path, *, check_images: bool = False) -> dict:
    actions: Counter[str] = Counter()
    examples = 0
    episodes = 0
    max_frames = 0
    current_episode: str | None = None
    next_step = 0
    last_action: str | None = None
    episode_current_frames: set[str] = set()

    def finish_episode() -> None:
        nonlocal episodes
        if current_episode is None:
            return
        if last_action != "STOP":
            raise ValueError(f"episode {current_episode} does not end with expert STOP")
        episodes += 1

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                validate_action_sft_example(example)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid action SFT example at {path}:{line_number}: {exc}") from exc
            episode_id = str(example["episode_id"])
            step_index = int(example["step_index"])
            if current_episode != episode_id:
                finish_episode()
                current_episode = episode_id
                next_step = 0
                last_action = None
                episode_current_frames = set()
            if step_index != next_step:
                raise ValueError(
                    f"episode {episode_id} expected step {next_step}, found {step_index}"
                )
            images = [str(item) for item in example["images"]]
            current_image = images[-1]
            if any(image not in episode_current_frames for image in images[:-1]):
                raise ValueError(f"episode {episode_id} step {step_index} references a future frame")
            if check_images and not local_image_path(current_image).is_file():
                raise ValueError(f"current image is missing: {local_image_path(current_image)}")
            episode_current_frames.add(current_image)
            last_action = str(example["targets"][0]["action"])
            actions[last_action] += 1
            examples += 1
            next_step += 1
            max_frames = max(max_frames, len(images))
    finish_episode()
    if not examples:
        raise ValueError(f"action SFT manifest is empty: {path}")
    if actions["STOP"] != episodes:
        raise ValueError("every episode must contain exactly one final STOP")
    return {
        "schema": "cfrp.qwen3vl.action_sft_audit.v1",
        "status": "passed",
        "examples": examples,
        "episodes": episodes,
        "action_counts": dict(sorted(actions.items())),
        "max_frames": max_frames,
        "current_frame_last": True,
        "primitive_steps_contiguous": True,
        "history_uses_observed_frames_only": True,
        "complete_episode_stop": True,
        "images_checked": check_images,
    }


if __name__ == "__main__":
    raise SystemExit(main())
