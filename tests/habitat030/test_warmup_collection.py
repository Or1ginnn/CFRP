from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.habitat030_collect_stage1_warmup import (
    _reference_waypoints,
    _select_episode_ids,
    _write_collection_status,
)


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


def test_failed_collection_writes_status_instead_of_complete_manifest(
    tmp_path: Path,
):
    args = Namespace(
        split="train",
        seed=123,
        max_steps=160,
        max_visual_history=4,
        max_action_history=3,
    )

    _write_collection_status(
        tmp_path,
        args,
        ("1", "2"),
        ("1",),
        status="failed",
        error="oracle did not terminate",
    )

    assert not (tmp_path / "manifest.json").exists()
    status = json.loads((tmp_path / "collection_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["completed_episode_ids"] == ["1"]
    assert status["oracle_policy"] == {
        "route": "r2r_reference_path",
        "waypoint_radius": 0.2,
    }


def test_reference_waypoints_preserve_expert_route_after_start():
    path = ((0, 0, 0), (1, 0, 2), (3, 0, 4))

    assert _reference_waypoints(path) == ((1.0, 0.0, 2.0), (3.0, 0.0, 4.0))


def test_reference_waypoints_require_a_route_after_start():
    with pytest.raises(ValueError, match="reference_path"):
        _reference_waypoints(((0, 0, 0),))
