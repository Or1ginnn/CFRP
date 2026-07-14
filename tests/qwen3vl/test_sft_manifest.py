import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from scripts.convert_stage1_warmup_to_sft import (
    _export_images,
    _iter_unique_image_paths,
)
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
        "window_index": 1,
        "start_turn_index": 3,
        "end_turn_index": 3,
        "images": ["file:///tmp/frame.png"],
        "targets": [
            {
                "message_index": 2,
                "request_id": 1,
                "turn_index": 3,
                "initializes_plan": False,
                "target_xml": target,
            }
        ],
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

    assert normalized[0]["content"] == [{"type": "text", "text": "system"}]
    assert normalized[1]["content"][0]["image"] == "/tmp/frame.png"
    assert normalized[2]["content"] == [{"type": "text", "text": messages[2]["content"]}]
    assert messages[0]["content"] == "system"
    assert messages[1]["content"][0]["image"] == "file:///tmp/frame.png"


def test_llamafactory_export_preserves_target_and_image_order():
    source = example()

    converted = make_llamafactory_stage1_example(source)

    assert converted["conversations"][1]["value"] == source["targets"][0]["target_xml"]
    assert converted["images"] == ["/tmp/frame.png"]
    assert converted["conversations"][0]["value"].count("<image>") == 1


def test_warmup_image_export_reuses_identical_source_frame(tmp_path: Path, monkeypatch):
    class FakeImage:
        class Resampling:
            LANCZOS = "lanczos"

        @staticmethod
        def fromarray(_array):
            return FakeImage()

        def convert(self, _mode):
            return self

        def resize(self, _size, resample):
            assert resample == FakeImage.Resampling.LANCZOS
            return self

        def save(self, path):
            Path(path).write_bytes(b"fake-png")

    fake_numpy = ModuleType("numpy")
    fake_numpy.load = lambda _path: object()
    fake_pil = ModuleType("PIL")
    fake_pil.Image = FakeImage
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    source = tmp_path / "frame.npy"
    source.write_bytes(b"placeholder")
    record = {"model_input": {"visual_history_paths": [str(source), str(source)]}}

    exported = _export_images(record, tmp_path / "images", {})

    assert exported[0] == exported[1]
    assert len(list((tmp_path / "images").glob("*.png"))) == 1


def test_unique_image_scan_preserves_first_seen_order(tmp_path: Path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    source = tmp_path / "warmup.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "model_input": {
                        "visual_history_paths": [str(first), str(second), str(first)]
                    }
                }
            )
            for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    assert list(_iter_unique_image_paths(source)) == [
        str(first.resolve()),
        str(second.resolve()),
    ]


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
    assert payload["loss_weights"] == {"action": 5.0, "progress": 2.0, "subgoal": 0.25, "xml": 1.0}
