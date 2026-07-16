import json
from pathlib import Path

from scripts.resample_stage1_warmup_visual_history import resample_warmup_directory


def test_resample_rebuilds_eight_route_anchors_plus_current_frame(tmp_path: Path):
    source = tmp_path / "source"
    frames = source / "episode-1" / "frames"
    frames.mkdir(parents=True)
    for index in range(51):
        (frames / "frame-{:04d}.npy".format(index)).write_bytes(b"frame")
    record = {
        "model_input": {
            "episode_id": "1",
            "request_id": 50,
            "turn_index": 50,
            "visual_history_paths": [str(frames / "frame-0050.npy")],
        },
        "target_xml": "<progress>hold</progress><subgoal>x</subgoal><action>MOVE_FORWARD</action>",
        "oracle_only": {},
    }
    manifest = {
        "schema": "cfrp.stage1.warmup.v1",
        "status": "complete",
        "max_visual_history": 6,
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "stage1_warmup.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    destination = tmp_path / "resampled"
    result = resample_warmup_directory(source, destination)

    migrated = json.loads((destination / "stage1_warmup.jsonl").read_text(encoding="utf-8"))
    assert [Path(path).name for path in migrated["model_input"]["visual_history_paths"]] == [
        "frame-0000.npy", "frame-0007.npy", "frame-0014.npy", "frame-0021.npy",
        "frame-0028.npy", "frame-0035.npy", "frame-0042.npy", "frame-0049.npy",
        "frame-0050.npy",
    ]
    assert result["max_visual_history"] == 9
    assert result["temporal_visual_history"]["history_anchor_count"] == 8
    assert result["temporal_visual_history"]["recent_contiguous_count"] == 1
