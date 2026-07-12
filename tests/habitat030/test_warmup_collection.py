from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.habitat030_collect_stage1_warmup import _select_episode_ids


def test_select_episode_ids_keeps_explicit_order():
    args = Namespace(episode_ids=" 7, 8 ,9 ", episode_count=None, episode_offset=0)

    assert _select_episode_ids(args) == ("7", "8", "9")


def test_select_episode_ids_uses_stable_contiguous_shard(monkeypatch):
    def fake_loader(**_kwargs):
        return tuple(SimpleNamespace(episode_id=str(index)) for index in range(10))

    monkeypatch.setattr("scripts.habitat030_collect_stage1_warmup.load_r2r_dataset", fake_loader)
    args = Namespace(
        episode_ids=None,
        episode_count=3,
        episode_offset=4,
        dataset_root="/dataset",
        split="train",
        scenes_dir="/scenes",
    )

    assert _select_episode_ids(args) == ("4", "5", "6")


def test_select_episode_ids_rejects_out_of_range_shard(monkeypatch):
    monkeypatch.setattr(
        "scripts.habitat030_collect_stage1_warmup.load_r2r_dataset",
        lambda **_kwargs: (SimpleNamespace(episode_id="1"),),
    )
    args = Namespace(
        episode_ids=None,
        episode_count=2,
        episode_offset=0,
        dataset_root="/dataset",
        split="train",
        scenes_dir="/scenes",
    )

    with pytest.raises(ValueError, match="requested shard"):
        _select_episode_ids(args)
