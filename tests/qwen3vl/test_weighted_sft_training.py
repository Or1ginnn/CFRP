from types import SimpleNamespace

from scripts.preflight_qwen3vl_stage1_sft import _processor_sample
from scripts.train_qwen3vl_stage1_sft import (
    _ResumeState,
    _Runtime,
    _action_accuracy_counts,
    _action_accuracy_metrics,
    _equal_train_shard,
    _fixed_validation_examples,
    _iter_batches,
    _milestone_steps,
    _next_resume_position,
    _optimizer_step_count,
    _require_visual_tensors,
    _split_examples_by_episode,
    _unwrap_distributed_model,
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


def test_distributed_model_is_unwrapped_for_uneven_validation_shards():
    inner = object()
    wrapper = SimpleNamespace(module=inner)

    assert _unwrap_distributed_model(wrapper) is inner
    assert _unwrap_distributed_model(inner) is inner


def test_optimizer_step_count_respects_epochs_and_global_micro_step_limit():
    args = SimpleNamespace(
        epochs=3,
        max_steps=None,
        per_device_batch_size=1,
        gradient_accumulation=8,
    )

    assert _optimizer_step_count(2203, args) == 828

    args.max_steps = 10
    assert _optimizer_step_count(2203, args) == 2


def test_optimizer_step_count_uses_real_per_device_batches():
    args = SimpleNamespace(
        epochs=3,
        max_steps=None,
        per_device_batch_size=4,
        gradient_accumulation=2,
    )

    assert _optimizer_step_count(2203, args) == 828

    args.max_steps = 10
    assert _optimizer_step_count(2203, args) == 2


def test_batch_iteration_and_resume_position_are_stable():
    examples = [_example(str(index), 0) for index in range(10)]

    batches = list(_iter_batches(examples, 4))

    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert _next_resume_position(2, 0, len(batches)) == (2, 1)
    assert _next_resume_position(2, 2, len(batches)) == (3, 0)
    assert _ResumeState(epoch=2, next_batch_index=1).epoch == 2


def test_milestones_match_formal_training_schedule():
    assert _milestone_steps(4660, 10) == (
        466,
        932,
        1398,
        1864,
        2330,
        2796,
        3262,
        3728,
        4194,
        4660,
    )
    assert _milestone_steps(4660, 5) == (932, 1864, 2796, 3728, 4660)


def test_fixed_validation_subset_is_order_independent():
    examples = [
        {"episode_id": str(index // 3), "window_index": index % 3}
        for index in range(30)
    ]

    first = _fixed_validation_examples(examples, 10, 123)
    second = _fixed_validation_examples(list(reversed(examples)), 10, 123)

    assert first == second
    assert len(first) == 10


def test_visual_sft_requires_vit_inputs():
    _require_visual_tensors(
        {"pixel_values": object(), "image_grid_thw": object()}, "example"
    )

    with pytest.raises(RuntimeError, match="missing visual tensors"):
        _require_visual_tensors({"input_ids": object()}, "example")


def test_action_accuracy_uses_only_expert_action_tokens():
    torch = pytest.importorskip("torch")
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, 4],
            [-100, 1, 2, 3, 4],
        ]
    )
    action_mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
        ]
    )
    logits = torch.zeros((2, 5, 8))
    for row in range(2):
        for target_position in range(1, 5):
            logits[row, target_position - 1, labels[row, target_position]] = 10
    logits[1, 1, 2] = 0
    logits[1, 1, 7] = 10

    counts = _action_accuracy_counts(logits, labels, action_mask)
    metrics = _action_accuracy_metrics(
        counts,
        _Runtime(rank=0, local_rank=0, world_size=1, distributed=False, device="cpu"),
    )

    assert counts.tolist() == [3.0, 4.0, 1.0, 2.0]
    assert metrics == {
        "action_token_accuracy": 0.75,
        "action_exact_match": 0.5,
    }


def test_processor_preflight_sample_is_stable_and_spans_episodes():
    examples = [
        {"episode_id": str(index // 2), "window_index": index % 2}
        for index in range(20)
    ]

    first = _processor_sample(examples, 5)
    second = _processor_sample(list(reversed(examples)), 5)

    assert first == second
    assert len(first) == 5
    assert len({item["episode_id"] for item in first}) > 1
