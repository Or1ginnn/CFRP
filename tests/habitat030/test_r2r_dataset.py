from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from vlnce_server.habitat030.r2r_dataset import (
    R2RSceneNotFoundError,
    load_r2r_dataset,
    load_r2r_episode,
)


def write_split(root: Path, split: str, scene_id: str = "mp3d/house123/house123.glb") -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    payload = {
        "episodes": [
            {
                "episode_id": 7,
                "scene_id": scene_id,
                "start_position": [1, 2, 3],
                "start_rotation": [0, 0, 0, 1],
                "goals": [{"position": [4, 5, 6], "radius": 3.0}],
                "instruction": {"instruction_text": "walk to the target"},
                "reference_path": [[1, 2, 3], [4, 5, 6]],
            }
        ],
        "instruction_vocab": {},
    }
    with gzip.open(split_dir / f"{split}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def make_scene(root: Path, house_id: str = "house123") -> Path:
    scene = root / "mp3d" / house_id / f"{house_id}.glb"
    scene.parent.mkdir(parents=True)
    scene.write_text("glb", encoding="utf-8")
    return scene


def test_load_r2r_episode_maps_scene_and_reads_fields(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scenes_dir = tmp_path / "scene_datasets"
    scene = make_scene(scenes_dir)
    write_split(dataset_root, "val_seen")

    record = load_r2r_episode(str(dataset_root), split="val_seen", scenes_dir=str(scenes_dir))

    assert record.episode_id == "7"
    assert record.scene_id == "mp3d/house123/house123.glb"
    assert record.scene_path == str(scene)
    assert record.start_position == (1.0, 2.0, 3.0)
    assert record.start_rotation == (0.0, 0.0, 0.0, 1.0)
    assert record.goals == ((4.0, 5.0, 6.0),)
    assert record.goal_radii == (3.0,)
    assert record.instruction_text == "walk to the target"
    assert record.reference_path == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_load_r2r_dataset_filters_episode_ids_and_limit(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scenes_dir = tmp_path / "scene_datasets"
    make_scene(scenes_dir)
    write_split(dataset_root, "val_seen")

    records = load_r2r_dataset(
        str(dataset_root),
        split="val_seen",
        episode_ids=("7",),
        limit=1,
        scenes_dir=str(scenes_dir),
    )

    assert len(records) == 1
    assert records[0].episode_id == "7"


def test_missing_scene_raises_clear_error(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scenes_dir = tmp_path / "scene_datasets"
    write_split(dataset_root, "val_seen")

    with pytest.raises(R2RSceneNotFoundError, match="MP3D scene not found"):
        load_r2r_episode(str(dataset_root), split="val_seen", scenes_dir=str(scenes_dir))


def test_relative_data_scene_prefix_is_removed(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scenes_dir = tmp_path / "scene_datasets"
    scene = make_scene(scenes_dir)
    write_split(dataset_root, "val_seen", scene_id="data/scene_datasets/mp3d/house123/house123.glb")

    record = load_r2r_episode(str(dataset_root), split="val_seen", scenes_dir=str(scenes_dir))

    assert record.scene_path == str(scene)