import json
from pathlib import Path

import pytest

from scripts.audit_action_sft import audit_action_sft
from vlnce_server.qwen3vl.action_sft import make_action_sft_example


def _write_episode(path: Path, actions: tuple[str, ...]) -> None:
    frame_uris = []
    rows = []
    for step, action in enumerate(actions):
        frame = path.parent / f"frame-{step:04d}.npy"
        frame.write_bytes(b"frame")
        frame_uris.append(frame.as_uri())
        rows.append(
            make_action_sft_example(
                episode_id="1",
                step_index=step,
                instruction="Walk to the door.",
                frame_uris=frame_uris,
                expert_action=action,
            )
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_action_audit_accepts_complete_primitive_episode(tmp_path: Path) -> None:
    source = tmp_path / "action.jsonl"
    _write_episode(source, ("MOVE_FORWARD", "TURN_LEFT", "STOP"))
    report = audit_action_sft(source, check_images=True)
    assert report["status"] == "passed"
    assert report["examples"] == 3
    assert report["episodes"] == 1
    assert report["action_counts"]["STOP"] == 1


def test_action_audit_rejects_episode_without_stop(tmp_path: Path) -> None:
    source = tmp_path / "action.jsonl"
    _write_episode(source, ("MOVE_FORWARD", "TURN_LEFT"))
    with pytest.raises(ValueError, match="does not end"):
        audit_action_sft(source)
