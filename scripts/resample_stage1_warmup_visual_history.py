"""Rebuild model-visible warm-up frame histories from retained raw RGB frames.

This is a data-only migration: it does not invoke Habitat or change oracle
actions/targets.  It upgrades old contiguous-history records to the canonical
six route anchors plus three recent consecutive frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030.temporal_history import (
    DEFAULT_MODEL_VISUAL_FRAME_COUNT,
    DEFAULT_VISUAL_CONTEXT_WINDOW,
    SlowFastVisualHistory,
    temporal_history_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Complete merged or shard warm-up directory")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resample_warmup_directory(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write a canonical slow-memory + recent-frame copy of complete warm-up data."""

    source_manifest = _load_manifest(input_dir / "manifest.json")
    source_records = input_dir / "stage1_warmup.jsonl"
    if not source_records.is_file():
        raise FileNotFoundError(f"warm-up records not found: {source_records}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True)
    destination = output_dir / "stage1_warmup.jsonl"
    count = 0
    with source_records.open("r", encoding="utf-8") as source, destination.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            migrated = resample_warmup_record(record, line_number=line_number)
            target.write(json.dumps(migrated, ensure_ascii=False) + "\n")
            count += 1

    manifest = dict(source_manifest)
    manifest["max_visual_history"] = DEFAULT_MODEL_VISUAL_FRAME_COUNT
    manifest["temporal_visual_history"] = temporal_history_spec()
    manifest["records"] = count
    manifest["source_warmup_dir"] = str(input_dir.resolve())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def resample_warmup_record(record: Mapping[str, Any], *, line_number: int = 0) -> dict[str, Any]:
    """Replace only visual_history_paths using raw episode frame files."""

    copied = dict(record)
    model_input = record.get("model_input")
    if not isinstance(model_input, Mapping):
        raise ValueError(_prefix(line_number) + "missing model_input")
    request = dict(model_input)
    paths = request.get("visual_history_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(_prefix(line_number) + "missing visual_history_paths")
    try:
        turn_index = int(request["turn_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(_prefix(line_number) + "invalid model_input.turn_index") from exc

    frames_dir = Path(str(paths[-1])).parent
    raw_paths = tuple(frames_dir / "frame-{:04d}.npy".format(index) for index in range(turn_index + 1))
    missing = [str(path) for path in raw_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(_prefix(line_number) + "missing retained raw frames: " + missing[0])
    path_history = SlowFastVisualHistory[Path].create(
        context_window=DEFAULT_VISUAL_CONTEXT_WINDOW
    )
    for path in raw_paths:
        path_history = path_history.reset(path) if not path_history.context else path_history.append(path)
    request["visual_history_paths"] = [str(path) for path in path_history.visible]
    copied["model_input"] = request
    return copied


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"complete warm-up manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cfrp.stage1.warmup.v1" or manifest.get("status") != "complete":
        raise ValueError(f"input warm-up is not complete: {path}")
    return manifest


def _prefix(line_number: int) -> str:
    return "line {}: ".format(line_number) if line_number else ""


def main() -> int:
    args = parse_args()
    manifest = resample_warmup_directory(Path(args.input_dir), Path(args.output_dir))
    print("records={}".format(manifest["records"]))
    print("output_dir={}".format(Path(args.output_dir).resolve()))
    print("resample_stage1_warmup_visual_history: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
