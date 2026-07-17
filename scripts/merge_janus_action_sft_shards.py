"""Merge complete, disjoint JanusVLN-compatible action SFT shards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.action_sft import (
    JANUS_ACTION_COLLECTION_SCHEMA,
    validate_action_sft_example,
    validate_janus_action_sft_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def merge_janus_action_sft_shards(
    shard_dirs: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    manifests = [validate_janus_action_sft_manifest(path / "action_sft.jsonl") for path in shard_dirs]
    reference = manifests[0]
    shared_keys = (
        "split",
        "seed",
        "max_steps",
        "simulator_contract",
        "oracle_policy",
        "visual_contract",
        "temporal_visual_contract",
    )
    for index, manifest in enumerate(manifests[1:], start=1):
        for key in shared_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"shard {index} has incompatible {key}")

    requested: list[str] = []
    completed: list[str] = []
    seen: set[str] = set()
    lines: list[str] = []
    actions: Counter[str] = Counter()
    source_shards = []
    for shard_dir, manifest in zip(shard_dirs, manifests):
        shard_ids = [str(value) for value in manifest["completed_episode_ids"]]
        overlap = seen.intersection(shard_ids)
        if overlap:
            raise ValueError(f"episode IDs overlap across shards: {sorted(overlap)}")
        shard_lines, observed, shard_actions = _read_shard(shard_dir / "action_sft.jsonl")
        if observed != shard_ids:
            raise ValueError(f"episode order/content differs from manifest in {shard_dir}")
        if len(shard_lines) != int(manifest["examples"]):
            raise ValueError(f"example count differs from manifest in {shard_dir}")
        lines.extend(shard_lines)
        requested.extend(str(value) for value in manifest["requested_episode_ids"])
        completed.extend(shard_ids)
        seen.update(shard_ids)
        actions.update(shard_actions)
        source_shards.append(
            {
                "path": str(shard_dir.resolve()),
                "episodes": len(shard_ids),
                "examples": len(shard_lines),
            }
        )

    manifest = {
        "schema": JANUS_ACTION_COLLECTION_SCHEMA,
        "example_schema": reference["example_schema"],
        "status": "complete",
        "split": reference["split"],
        "requested_episode_ids": requested,
        "completed_episode_ids": completed,
        "seed": reference["seed"],
        "max_steps": reference["max_steps"],
        "examples": len(lines),
        "action_counts": dict(sorted(actions.items())),
        "simulator_contract": reference["simulator_contract"],
        "oracle_policy": reference["oracle_policy"],
        "visual_contract": reference["visual_contract"],
        "temporal_visual_contract": reference["temporal_visual_contract"],
        "source_shards": source_shards,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "action_sft.jsonl").write_text("".join(lines), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _read_shard(path: Path) -> tuple[list[str], list[str], Counter[str]]:
    lines: list[str] = []
    episode_order: list[str] = []
    actions: Counter[str] = Counter()
    current_episode: str | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        example = json.loads(line)
        try:
            validate_action_sft_example(example)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid example at {path}:{line_number}: {exc}") from exc
        episode_id = str(example["episode_id"])
        if episode_id != current_episode:
            if episode_id in episode_order:
                raise ValueError(f"episode {episode_id} is not contiguous in {path}")
            episode_order.append(episode_id)
            current_episode = episode_id
        actions[str(example["targets"][0]["action"])] += 1
        lines.append(line + "\n")
    if not lines:
        raise ValueError(f"empty action SFT shard: {path}")
    return lines, episode_order, actions


def main() -> int:
    args = parse_args()
    manifest = merge_janus_action_sft_shards(
        [Path(value) for value in args.shard_dirs], Path(args.output_dir)
    )
    print(f"episodes={len(manifest['completed_episode_ids'])}")
    print(f"examples={manifest['examples']}")
    print(f"output_dir={Path(args.output_dir).resolve()}")
    print("merge_janus_action_sft_shards: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
