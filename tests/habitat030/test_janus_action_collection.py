import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

from scripts.habitat030_collect_janus_action_sft import (
    reference_waypoints,
    save_model_frame,
    waypoint_radius,
    write_status,
)
from vlnce_server.qwen3vl.action_sft import validate_janus_action_sft_manifest


def test_janus_waypoint_radius_switches_only_for_final_goal():
    assert waypoint_radius(0, 3) == 1.8
    assert waypoint_radius(1, 3) == 1.8
    assert waypoint_radius(2, 3) == 0.25
    assert waypoint_radius(0, 1) == 0.25


def test_janus_reference_path_skips_start_pose():
    assert reference_waypoints(((0, 0, 0), (1, 0, 2), (3, 0, 4))) == (
        (1.0, 0.0, 2.0),
        (3.0, 0.0, 4.0),
    )


def test_janus_collector_saves_compact_jpeg(tmp_path: Path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    uri = save_model_frame(frame, tmp_path, 0)

    path = Path(uri.removeprefix("file://"))
    assert path.suffix == ".jpg"
    with Image.open(path) as image:
        assert image.size == (384, 288)


def test_complete_manifest_passes_exact_janus_gate(tmp_path: Path):
    args = Namespace(split="train", seed=123, max_steps=500)
    data = tmp_path / "action_sft.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    write_status(
        tmp_path,
        args,
        ("1",),
        ("1",),
        1,
        Counter({"STOP": 1}),
        status="complete",
    )

    manifest = validate_janus_action_sft_manifest(data)

    assert manifest["simulator_contract"]["turn_angle"] == 15
    assert manifest["oracle_policy"] == {
        "route": "r2r_reference_path",
        "intermediate_waypoint_radius": 1.8,
        "final_waypoint_radius": 0.25,
    }
    assert manifest["visual_contract"]["stored_model_image_size"] == [384, 288]
