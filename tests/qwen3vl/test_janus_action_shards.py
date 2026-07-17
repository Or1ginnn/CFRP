import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

from scripts.habitat030_collect_janus_action_sft import write_status
from scripts.launch_janus_action_sft_collection import ShardSpec, build_shards, collector_command
from scripts.merge_janus_action_sft_shards import merge_janus_action_sft_shards
from vlnce_server.qwen3vl.action_sft import (
    load_action_sft_jsonl,
    make_action_sft_example,
)


def _make_shard(root: Path, index: int, episode_id: str) -> Path:
    shard = root / f"shard-{index:04d}"
    shard.mkdir()
    example = make_action_sft_example(
        episode_id=episode_id,
        step_index=0,
        instruction="Stop here.",
        frame_uris=((root / f"{episode_id}.jpg").resolve().as_uri(),),
        expert_action="STOP",
    )
    (shard / "action_sft.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
    write_status(
        shard,
        Namespace(split="train", seed=123, max_steps=500),
        (episode_id,),
        (episode_id,),
        1,
        Counter({"STOP": 1}),
        status="complete",
    )
    return shard


def test_merge_janus_action_shards_preserves_contract(tmp_path: Path):
    shards = [_make_shard(tmp_path, 0, "1"), _make_shard(tmp_path, 1, "2")]

    manifest = merge_janus_action_sft_shards(shards, tmp_path / "merged")
    examples = load_action_sft_jsonl(
        tmp_path / "merged" / "action_sft.jsonl",
        require_janus_contract=True,
    )

    assert manifest["completed_episode_ids"] == ["1", "2"]
    assert manifest["examples"] == len(examples) == 2


def test_janus_launcher_builds_disjoint_shards_and_fixed_collector_command(tmp_path: Path):
    assert build_shards(5, 2) == (
        ShardSpec(0, 0, 2),
        ShardSpec(1, 2, 2),
        ShardSpec(2, 4, 1),
    )
    args = Namespace(
        python=str(tmp_path / "python"),
        dataset_root=str(tmp_path / "dataset"),
        scenes_dir=str(tmp_path / "scenes"),
        config=str(tmp_path / "config.yaml"),
        split="train",
        seed=123,
        max_steps=500,
    )

    command = collector_command(args, tmp_path / "out", ShardSpec(1, 100, 50))

    assert command[1].endswith("scripts/habitat030_collect_janus_action_sft.py")
    assert command[command.index("--episode-offset") + 1] == "100"
    assert command[command.index("--episode-count") + 1] == "50"
    assert command[command.index("--max-steps") + 1] == "500"
