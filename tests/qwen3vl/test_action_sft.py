import json
import subprocess
import sys
from pathlib import Path

import pytest

from vlnce_server.qwen3vl.action_sft import (
    ACTION_SFT_SCHEMA,
    JANUS_ACTION_COLLECTION_SCHEMA,
    janus_frame_indices,
    load_action_sft_jsonl,
    make_action_sft_example,
    validate_action_sft_example,
)
from vlnce_server.habitat030.r2r_environment import (
    janus_r2r_oracle_contract,
    janus_r2r_simulator_contract,
)
from vlnce_server.qwen3vl.vision import qwen3vl_image_size
from vlnce_server.qwen3vl.vision import qwen3vl_processor_kwargs
from scripts.convert_stage1_warmup_to_action_sft import recover_expert_episodes
from scripts.preflight_qwen3vl_stage1_sft import (
    _model_forward_sample,
    _validate_action_images,
)
from scripts.habitat030_collect_janus_action_sft import save_model_frame


def test_janus_frame_indices_keep_current_frame_last() -> None:
    assert janus_frame_indices(0) == (0,)
    assert janus_frame_indices(8) == tuple(range(9))
    assert janus_frame_indices(17) == (0, 2, 4, 6, 8, 10, 12, 14, 17)


def test_action_example_contains_one_action_and_no_plan_tags() -> None:
    example = make_action_sft_example(
        episode_id="7",
        step_index=2,
        instruction="Walk through the doorway.",
        frame_uris=("file:///tmp/0.npy", "file:///tmp/1.npy", "file:///tmp/2.npy"),
        expert_action="TURN_LEFT",
    )
    assert example["schema"] == ACTION_SFT_SCHEMA
    assert example["messages"][-1]["content"] == "<action>TURN_LEFT</action>"
    assert example["images"][-1] == "file:///tmp/2.npy"
    assert "<plan" not in json.dumps(example)


def test_action_example_rejects_action_chunks() -> None:
    example = make_action_sft_example(
        episode_id="7",
        step_index=0,
        instruction="Move ahead.",
        frame_uris=("file:///tmp/0.npy",),
        expert_action="MOVE_FORWARD",
    )
    example["messages"][-1]["content"] = (
        "<action>MOVE_FORWARD</action><action>TURN_LEFT</action>"
    )
    example["targets"][0]["target_xml"] = example["messages"][-1]["content"]
    with pytest.raises(ValueError, match="exactly one primitive"):
        validate_action_sft_example(example)


def test_loader_checks_local_images_when_requested(tmp_path: Path) -> None:
    frame = tmp_path / "frame.npy"
    frame.write_bytes(b"frame")
    example = make_action_sft_example(
        episode_id="1",
        step_index=0,
        instruction="Stop here.",
        frame_uris=(frame.as_uri(),),
        expert_action="STOP",
    )
    source = tmp_path / "action.jsonl"
    source.write_text(json.dumps(example) + "\n", encoding="utf-8")
    assert load_action_sft_jsonl(source) == [example]
    validate_action_sft_example(example, check_images=True)


def test_action_preflight_decodes_compact_jpegs(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    uri = save_model_frame(np.zeros((480, 640, 3), dtype=np.uint8), tmp_path, 0)
    example = make_action_sft_example(
        episode_id="1",
        step_index=0,
        instruction="Stop here.",
        frame_uris=(uri,),
        expert_action="STOP",
    )

    assert _validate_action_images([example]) == {
        "unique_images_checked": 1,
        "width": 384,
        "height": 288,
    }


def test_model_forward_sample_prefers_nine_frame_action_context():
    one_frame = make_action_sft_example(
        episode_id="1",
        step_index=0,
        instruction="Move.",
        frame_uris=("file:///tmp/0.jpg",),
        expert_action="MOVE_FORWARD",
    )
    nine_frame = make_action_sft_example(
        episode_id="2",
        step_index=8,
        instruction="Move.",
        frame_uris=tuple(f"file:///tmp/{index}.jpg" for index in range(9)),
        expert_action="MOVE_FORWARD",
    )

    assert _model_forward_sample([one_frame, nine_frame], 1) == [nine_frame]


def test_recover_expert_episode_expands_chunks_to_primitive_steps(tmp_path: Path) -> None:
    frame_dir = tmp_path / "episode-1" / "frames"
    frame_dir.mkdir(parents=True)
    for index in range(3):
        (frame_dir / f"frame-{index:04d}.npy").write_bytes(b"frame")
    records = [
        {
            "model_input": {
                "episode_id": "1",
                "turn_index": 0,
                "instruction": "Walk forward, then stop.",
                "visual_history_paths": [str(frame_dir / "frame-0000.npy")],
            },
            "oracle_only": {"oracle_actions": ["MOVE_FORWARD", "TURN_RIGHT"]},
        },
        {
            "model_input": {
                "episode_id": "1",
                "turn_index": 2,
                "instruction": "Walk forward, then stop.",
                "visual_history_paths": [str(frame_dir / "frame-0002.npy")],
            },
            "oracle_only": {"oracle_actions": ["STOP"]},
        },
    ]
    source = tmp_path / "warmup.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    recovered = list(recover_expert_episodes(source))
    assert recovered[0][0:3] == (
        "1",
        "Walk forward, then stop.",
        ("MOVE_FORWARD", "TURN_RIGHT", "STOP"),
    )
    assert tuple(path.name for path in recovered[0][3]) == (
        "frame-0000.npy",
        "frame-0001.npy",
        "frame-0002.npy",
    )


def test_action_sft_trainer_dry_run_uses_action_contract(tmp_path: Path) -> None:
    example = make_action_sft_example(
        episode_id="1",
        step_index=0,
        instruction="Stop here.",
        frame_uris=("file:///tmp/frame.npy",),
        expert_action="STOP",
    )
    source = tmp_path / "action.jsonl"
    source.write_text(json.dumps(example) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": JANUS_ACTION_COLLECTION_SCHEMA,
                "status": "complete",
                "requested_episode_ids": ["1"],
                "completed_episode_ids": ["1"],
                "max_steps": 500,
                "examples": 1,
                "simulator_contract": janus_r2r_simulator_contract(),
                "oracle_policy": janus_r2r_oracle_contract(),
                "visual_contract": {
                    "habitat_rgb_size": [640, 480],
                    "stored_model_image_size": list(qwen3vl_image_size()),
                    "storage": "jpeg",
                    "processor_kwargs": qwen3vl_processor_kwargs(),
                },
                "temporal_visual_contract": {
                    "sampling": "janus_uniform_episode_prefix",
                    "max_frames": 9,
                    "current_frame_last": True,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dry-run"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen3vl_stage1_sft.py",
            "--contract",
            "action-only",
            "--train-jsonl",
            str(source),
            "--output-dir",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "qwen3vl_action_only_sft_dry_run: OK" in completed.stdout
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"] == "action-only"
    assert manifest["objective"] == "assistant_only_causal_cross_entropy"
    assert manifest["loss_weights"] == {
        "action": 1.0,
        "stop_action": 1.0,
        "xml": 1.0,
    }


def test_action_trainer_rejects_legacy_collection_manifest(tmp_path: Path) -> None:
    example = make_action_sft_example(
        episode_id="1",
        step_index=0,
        instruction="Stop here.",
        frame_uris=("file:///tmp/frame.jpg",),
        expert_action="STOP",
    )
    source = tmp_path / "action.jsonl"
    source.write_text(json.dumps(example) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "cfrp.qwen3vl.action_sft_manifest.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not produced"):
        load_action_sft_jsonl(source, require_janus_contract=True)
