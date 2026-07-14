from types import SimpleNamespace

from scripts.train_qwen3vl_stage1_sft import (
    _equal_train_shard,
    _optimizer_step_count,
    _require_visual_tensors,
    _split_examples_by_episode,
)
import pytest


def _example(episode_id: str, turn_index: int) -> dict:
    return {"episode_id": episode_id, "turn_index": turn_index}


def test_validation_split_keeps_each_episode_whole():
    examples = [
        _example("episode-a", 0),
        _example("episode-a", 1),
        _example("episode-b", 0),
        _example("episode-b", 1),
        _example("episode-c", 0),
    ]

    train, validation = _split_examples_by_episode(examples, 0.2, 123)

    train_episodes = {item["episode_id"] for item in train}
    validation_episodes = {item["episode_id"] for item in validation}
    assert train_episodes
    assert validation_episodes
    assert train_episodes.isdisjoint(validation_episodes)


def test_ddp_train_shards_have_equal_micro_step_counts():
    examples = [_example(str(index), 0) for index in range(10)]

    shards = [_equal_train_shard(examples, rank, 4) for rank in range(4)]

    assert [len(shard) for shard in shards] == [3, 3, 3, 3]
    assert {item["episode_id"] for shard in shards for item in shard}.issuperset(
        {item["episode_id"] for item in examples}
    )


def test_optimizer_step_count_respects_epochs_and_global_micro_step_limit():
    args = SimpleNamespace(epochs=3, max_steps=None, gradient_accumulation=8)

    assert _optimizer_step_count(2203, args) == 828

    args.max_steps = 10
    assert _optimizer_step_count(2203, args) == 2


def test_visual_sft_requires_vit_inputs():
    _require_visual_tensors(
        {"pixel_values": object(), "image_grid_thw": object()}, "example"
    )

    with pytest.raises(RuntimeError, match="missing visual tensors"):
        _require_visual_tensors({"input_ids": object()}, "example")
