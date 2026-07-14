import json
from pathlib import Path

import pytest

from scripts.merge_stage1_warmup_shards import merge_warmup_shards


def _make_shard(
    root: Path,
    name: str,
    episode_ids: list[str],
    *,
    contract=None,
    temporal_history=None,
    max_steps=500,
) -> Path:
    shard = root / name
    shard.mkdir()
    contract = contract or {"habitat_rgb_size": [640, 480]}
    manifest = {
        "schema": "cfrp.stage1.warmup.v1",
        "status": "complete",
        "split": "train",
        "seed": 123,
        "max_steps": max_steps,
        "step_unit": "habitat_primitive_action",
        "max_visual_history": 6,
        "max_action_history": 8,
        "visual_contract": contract,
        "temporal_visual_history": temporal_history
        or {"history_anchor_count": 6, "recent_frame_count": 3},
        "completed_episode_ids": episode_ids,
    }
    (shard / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = [
        json.dumps({"model_input": {"episode_id": episode_id}, "target_xml": "<action>STOP</action>"})
        for episode_id in episode_ids
    ]
    (shard / "stage1_warmup.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    return shard


def test_merge_warmup_shards_writes_traceable_complete_manifest(tmp_path: Path):
    first = _make_shard(tmp_path, "shard-000", ["1", "2"])
    second = _make_shard(tmp_path, "shard-001", ["3"])

    result = merge_warmup_shards([first, second], tmp_path / "merged")

    assert result["records"] == 3
    assert result["completed_episode_ids"] == ["1", "2", "3"]
    assert result["temporal_visual_history"] == {
        "history_anchor_count": 6,
        "recent_frame_count": 3,
    }
    assert len(result["source_shards"]) == 2
    assert result["step_unit"] == "habitat_primitive_action"
    assert (tmp_path / "merged" / "stage1_warmup.jsonl").read_text().count("\n") == 3


def test_merge_warmup_shards_rejects_episode_overlap(tmp_path: Path):
    first = _make_shard(tmp_path, "shard-000", ["1"])
    second = _make_shard(tmp_path, "shard-001", ["1"])

    with pytest.raises(ValueError, match="overlap"):
        merge_warmup_shards([first, second], tmp_path / "merged")


def test_merge_warmup_shards_rejects_contract_mismatch(tmp_path: Path):
    first = _make_shard(tmp_path, "shard-000", ["1"])
    second = _make_shard(
        tmp_path, "shard-001", ["2"], contract={"habitat_rgb_size": [128, 128]}
    )

    with pytest.raises(ValueError, match="visual_contract"):
        merge_warmup_shards([first, second], tmp_path / "merged")


def test_merge_warmup_shards_rejects_temporal_history_mismatch(tmp_path: Path):
    first = _make_shard(tmp_path, "shard-000", ["1"])
    second = _make_shard(
        tmp_path,
        "shard-001",
        ["2"],
        temporal_history={"history_anchor_count": 4, "recent_frame_count": 3},
    )

    with pytest.raises(ValueError, match="temporal_visual_history"):
        merge_warmup_shards([first, second], tmp_path / "merged")


def test_merge_accepts_completed_shards_with_different_step_caps(tmp_path: Path):
    first = _make_shard(tmp_path, "shard-000", ["1"], max_steps=160)
    second = _make_shard(tmp_path, "shard-001", ["2"], max_steps=500)

    result = merge_warmup_shards([first, second], tmp_path / "merged")

    assert result["max_steps"] == 500
    assert result["source_max_steps"] == [160, 500]
