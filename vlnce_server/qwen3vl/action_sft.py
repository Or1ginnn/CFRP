"""JanusVLN-style action-only SFT examples for the CFRP Phase 0 baseline."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sft_manifest import local_image_path


ACTION_SFT_SCHEMA = "cfrp.qwen3vl.action_sft.v1"
ACTION_SFT_MAX_FRAMES = 9
ALLOWED_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")

ACTION_SYSTEM_PROMPT = (
    "You are a visual language navigation policy. Follow the navigation instruction "
    "using the observations. Return exactly one primitive action as "
    "<action>ACTION</action> and no other text."
)

_ACTION_PATTERN = re.compile(r"^<action>(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)</action>$")
_FORBIDDEN_TAGS = ("<plan", "<progress", "<subgoal", "<tool", "<actions")


def janus_frame_indices(current_index: int, max_frames: int = ACTION_SFT_MAX_FRAMES) -> tuple[int, ...]:
    """Select all early frames or uniformly sample history with current frame last.

    This matches JanusVLN's ``np.linspace(0, i, 9, dtype=int)`` rule without
    requiring NumPy in the manifest validator.
    """

    if current_index < 0:
        raise ValueError("current_index must not be negative")
    if max_frames < 2:
        raise ValueError("max_frames must be at least two")
    if current_index + 1 <= max_frames:
        return tuple(range(current_index + 1))
    return tuple(int(offset * current_index / (max_frames - 1)) for offset in range(max_frames))


def make_action_sft_example(
    *,
    episode_id: str,
    step_index: int,
    instruction: str,
    frame_uris: Sequence[str],
    expert_action: str,
) -> dict[str, Any]:
    """Build one independent expert-decision sample from an episode prefix."""

    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    if expert_action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported expert action: {expert_action!r}")
    if step_index < 0 or len(frame_uris) != step_index + 1:
        raise ValueError("frame_uris must contain every episode frame through step_index")

    selected_indices = janus_frame_indices(step_index)
    selected_images = [str(frame_uris[index]) for index in selected_indices]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Navigation instruction: {instruction.strip()}\n"
                "The observations are ordered from earlier to later. The last image is "
                "the current observation."
            ),
        }
    ]
    content.extend({"type": "image", "image": image} for image in selected_images)
    content.append(
        {
            "type": "text",
            "text": "Allowed actions: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.",
        }
    )
    target_xml = f"<action>{expert_action}</action>"
    example = {
        "schema": ACTION_SFT_SCHEMA,
        "episode_id": str(episode_id),
        "window_index": step_index,
        "step_index": step_index,
        "messages": [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": target_xml},
        ],
        "images": selected_images,
        "targets": [
            {
                "message_index": 2,
                "step_index": step_index,
                "action": expert_action,
                "target_xml": target_xml,
            }
        ],
        "visual_contract": {
            "sampling": "janus_uniform_episode_prefix",
            "max_frames": ACTION_SFT_MAX_FRAMES,
            "current_frame_last": True,
            "selected_frame_indices": list(selected_indices),
        },
    }
    validate_action_sft_example(example)
    return example


def load_action_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    examples: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                validate_action_sft_example(example)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid action SFT example at {source}:{line_number}: {exc}") from exc
            examples.append(example)
    if not examples:
        raise ValueError(f"action SFT manifest is empty: {source}")
    return examples


def validate_action_sft_example(example: Mapping[str, Any], *, check_images: bool = False) -> None:
    if example.get("schema") != ACTION_SFT_SCHEMA:
        raise ValueError(f"expected schema {ACTION_SFT_SCHEMA!r}")
    messages = example.get("messages")
    images = example.get("images")
    targets = example.get("targets")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("action SFT messages must be exactly system, user, assistant")
    if not isinstance(images, list) or not 1 <= len(images) <= ACTION_SFT_MAX_FRAMES:
        raise ValueError("action SFT requires one to nine images")
    image_blocks = [
        block.get("image")
        for block in messages[1].get("content", ())
        if isinstance(block, Mapping) and block.get("type") == "image"
    ]
    if image_blocks != images:
        raise ValueError("images must match user image blocks in temporal order")
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError("action SFT requires exactly one supervised target")
    target = targets[0]
    target_xml = target.get("target_xml")
    if target.get("message_index") != 2 or messages[2].get("content") != target_xml:
        raise ValueError("assistant response must match the single target")
    if not isinstance(target_xml, str) or _ACTION_PATTERN.fullmatch(target_xml) is None:
        raise ValueError("assistant response must contain exactly one primitive <action>")
    if any(tag in target_xml for tag in _FORBIDDEN_TAGS):
        raise ValueError("Phase 0 action SFT must not contain CFRP planning or tool tags")
    action = _ACTION_PATTERN.fullmatch(target_xml).group(1)  # type: ignore[union-attr]
    if target.get("action") != action:
        raise ValueError("target action metadata does not match assistant response")
    contract = example.get("visual_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("visual_contract is required")
    indices = contract.get("selected_frame_indices")
    step_index = int(example.get("step_index", -1))
    if (
        contract.get("sampling") != "janus_uniform_episode_prefix"
        or contract.get("max_frames") != ACTION_SFT_MAX_FRAMES
        or contract.get("current_frame_last") is not True
        or indices != list(janus_frame_indices(step_index))
        or indices[-1] != step_index
    ):
        raise ValueError("visual_contract does not match JanusVLN temporal sampling")
    if check_images:
        for source in images:
            if not local_image_path(source).is_file():
                raise ValueError(f"image file is missing: {local_image_path(source)}")
