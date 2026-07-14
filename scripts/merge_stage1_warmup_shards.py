"""Merge complete, disjoint Stage 1 warm-up shards into one training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def merge_warmup_shards(shard_dirs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    manifests = [_load_complete_manifest(path) for path in shard_dirs]
    _validate_shared_contract(manifests)

    all_records: list[str] = []
    completed_episode_ids: list[str] = []
    seen_episode_ids: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []

    for shard_dir, manifest in zip(shard_dirs, manifests):
        declared_ids = tuple(str(item) for item in manifest["completed_episode_ids"])
        if len(set(declared_ids)) != len(declared_ids):
            raise ValueError(f"duplicate episode IDs declared by shard: {shard_dir}")
        overlap = seen_episode_ids.intersection(declared_ids)
        if overlap:
            raise ValueError(f"episode IDs overlap across shards: {sorted(overlap)}")

        lines, observed_ids = _load_record_lines(shard_dir / "stage1_warmup.jsonl")
        if observed_ids != set(declared_ids):
            raise ValueError(
                f"records do not match completed_episode_ids in {shard_dir}: "
                f"declared={sorted(declared_ids)} observed={sorted(observed_ids)}"
            )
        all_records.extend(lines)
        completed_episode_ids.extend(declared_ids)
        seen_episode_ids.update(declared_ids)
        shard_summaries.append(
            {
                "path": str(shard_dir.resolve()),
                "records": len(lines),
                "completed_episode_ids": list(declared_ids),
            }
        )

    reference = manifests[0]
    source_max_steps = sorted({int(manifest["max_steps"]) for manifest in manifests})
    merged_manifest = {
        "schema": "cfrp.stage1.warmup.v1",
        "status": "complete",
        "split": reference["split"],
        "requested_episode_ids": completed_episode_ids,
        "completed_episode_ids": completed_episode_ids,
        "seed": reference["seed"],
        # A completed trajectory is invariant to any larger collection cap.
        # This permits short 160-step shards and 500-step repair shards to be
        # merged without hiding which budgets produced the source artifacts.
        "max_steps": max(source_max_steps),
        "source_max_steps": source_max_steps,
        "step_unit": _step_unit(reference),
        "max_visual_history": reference["max_visual_history"],
        "max_action_history": reference["max_action_history"],
        "oracle_policy": reference.get("oracle_policy"),
        "visual_contract": reference["visual_contract"],
        "temporal_visual_history": reference.get("temporal_visual_history"),
        "source_shards": shard_summaries,
        "records": len(all_records),
    }

    output_dir.mkdir(parents=True)
    (output_dir / "stage1_warmup.jsonl").write_text("".join(all_records), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return merged_manifest


def _load_complete_manifest(shard_dir: Path) -> dict[str, Any]:
    path = shard_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"complete shard manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cfrp.stage1.warmup.v1":
        raise ValueError(f"unexpected warm-up schema in {path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"shard is not complete: {path}")
    if not manifest.get("completed_episode_ids"):
        raise ValueError(f"shard has no completed episodes: {path}")
    return manifest


def _validate_shared_contract(manifests: Sequence[dict[str, Any]]) -> None:
    reference = manifests[0]
    keys = (
        "split",
        "seed",
        "max_visual_history",
        "max_action_history",
        "oracle_policy",
        "visual_contract",
        "temporal_visual_history",
    )
    for index, manifest in enumerate(manifests[1:], start=1):
        for key in keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"shard {index} has incompatible {key}")
        if _step_unit(manifest) != _step_unit(reference):
            raise ValueError(f"shard {index} has incompatible step_unit")


def _step_unit(manifest: dict[str, Any]) -> str:
    # Warm-up manifests created before the explicit field was introduced also
    # counted one collector loop iteration per Habitat primitive action.
    return str(manifest.get("step_unit", "habitat_primitive_action"))


def _load_record_lines(path: Path) -> tuple[list[str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"warm-up records not found: {path}")
    lines: list[str] = []
    episode_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            episode_id = str(record["model_input"]["episode_id"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing model_input.episode_id at {path}:{line_number}") from exc
        episode_ids.add(episode_id)
        lines.append(line + "\n")
    if not lines:
        raise ValueError(f"warm-up shard has no records: {path}")
    return lines, episode_ids


def main() -> int:
    args = parse_args()
    manifest = merge_warmup_shards(
        [Path(value) for value in args.shard_dirs], Path(args.output_dir)
    )
    print(f"records={manifest['records']}")
    print(f"episodes={len(manifest['completed_episode_ids'])}")
    print(f"output_dir={Path(args.output_dir).resolve()}")
    print("merge_stage1_warmup_shards: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
