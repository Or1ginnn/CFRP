import json
from pathlib import Path

import pytest

from vlnce_server.qwen3vl.sft_data import SFT_SCHEMA
from vlnce_server.qwen3vl.sft_manifest import load_stage1_sft_jsonl, local_file_uri, validate_stage1_sft_example


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
    with pytest.raises(ValueError, match="local file URI"):
        validate_stage1_sft_example(payload, check_images=True)


def test_load_manifest_reports_line_number(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(example()) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="samples.jsonl:2"):
        load_stage1_sft_jsonl(path)


def test_local_file_uri_decodes_paths():
    assert str(local_file_uri("file:///tmp/a%20frame.png")) == "/tmp/a frame.png"
