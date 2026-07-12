import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.train_qwen3vl_stage1_sft import _messages_with_processor_image_paths
from vlnce_server.qwen3vl.llamafactory_data import make_llamafactory_stage1_example
from vlnce_server.qwen3vl.sft_data import SFT_SCHEMA
from vlnce_server.qwen3vl.sft_manifest import (
    load_stage1_sft_jsonl,
    local_file_uri,
    local_image_path,
    validate_stage1_sft_example,
)


def example():
    target = "<progress>hold</progress><subgoal>leave the room</subgoal><action>MOVE_FORWARD</action>"
    return {
        "schema": SFT_SCHEMA,
        "episode_id": "1",
        "turn_index": 0,
        "images": ["file:///tmp/frame.png"],
        "target_xml": target,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "image", "image": "file:///tmp/frame.png"}]},
            {"role": "assistant", "content": target},
        ],
    }


def test_validate_manifest_example():
    validate_stage1_sft_example(example())


def test_manifest_rejects_non_local_image_uri():
    payload = example()
    payload["images"] = ["https://example.com/frame.png"]
    payload["messages"][1]["content"][0]["image"] = "https://example.com/frame.png"
    with pytest.raises(ValueError, match="local path or file URI"):
        validate_stage1_sft_example(payload, check_images=True)


def test_load_manifest_reports_line_number(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(example()) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="samples.jsonl:2"):
        load_stage1_sft_jsonl(path)


def test_local_file_uri_decodes_paths():
    assert str(local_file_uri("file:///tmp/a%20frame.png")) == "/tmp/a frame.png"


def test_local_image_path_accepts_path_or_file_uri():
    assert local_image_path("/tmp/frame.png") == Path("/tmp/frame.png")
    assert local_image_path("file:///tmp/frame.png") == Path("/tmp/frame.png")


def test_training_messages_normalize_file_uri_without_mutating_manifest():
    messages = example()["messages"]

    normalized = _messages_with_processor_image_paths(messages)

    assert normalized[1]["content"][0]["image"] == "/tmp/frame.png"
    assert messages[1]["content"][0]["image"] == "file:///tmp/frame.png"


def test_llamafactory_export_preserves_target_and_image_order():
    source = example()

    converted = make_llamafactory_stage1_example(source)

    assert converted["conversations"][1]["value"] == source["target_xml"]
    assert converted["images"] == ["/tmp/frame.png"]
    assert converted["conversations"][0]["value"].count("<image>") == 1


def test_sft_dry_run_executes_main_and_writes_manifest(tmp_path: Path):
    source = tmp_path / "samples.jsonl"
    output_dir = tmp_path / "dry-run"
    source.write_text(json.dumps(example()) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen3vl_stage1_sft.py",
            "--train-jsonl",
            str(source),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "qwen3vl_stage1_sft_dry_run: OK" in completed.stdout
    payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert payload["examples"] == 1
