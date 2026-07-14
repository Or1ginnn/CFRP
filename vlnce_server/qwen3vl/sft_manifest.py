"""Validation and loading for portable Stage 1 Qwen3-VL SFT manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

from vlnce_server.cfrp import MAX_STAGE1_ACTION_CHUNK, parse_cfrp_output

from .sft_data import SFT_SCHEMA


_ALLOWED_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def load_stage1_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate every line before a model runtime touches it."""

    source = Path(path)
    examples = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                validate_stage1_sft_example(example)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Stage 1 SFT example at {source}:{line_number}: {exc}") from exc
            examples.append(example)
    if not examples:
        raise ValueError(f"Stage 1 SFT manifest is empty: {source}")
    return examples


def validate_stage1_sft_example(example: Mapping[str, Any], *, check_images: bool = False) -> None:
    """Validate one bounded, model-visible multi-turn conversation window."""

    if example.get("schema") != SFT_SCHEMA:
        raise ValueError(f"expected schema {SFT_SCHEMA!r}")
    messages = example.get("messages")
    images = example.get("images")
    targets = example.get("targets")
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError("messages must contain system plus one or more user/assistant turns")
    expected_roles = ["system"] + [
        role for _ in range((len(messages) - 1) // 2) for role in ("user", "assistant")
    ]
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError("messages must alternate system, user, and assistant roles")
    if not isinstance(messages[0].get("content"), str):
        raise ValueError("system content must be text")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")
    if not isinstance(targets, list) or len(targets) != (len(messages) - 1) // 2:
        raise ValueError("targets must describe every assistant turn")

    prompt_images = []
    for message_index in range(1, len(messages), 2):
        user_content = messages[message_index].get("content")
        if not isinstance(user_content, list):
            raise ValueError("every user turn must contain multimodal content blocks")
        turn_images = [
            item.get("image")
            for item in user_content
            if isinstance(item, Mapping) and item.get("type") == "image"
        ]
        if not turn_images:
            raise ValueError("every user turn requires at least one image")
        prompt_images.extend(turn_images)
    if prompt_images != images:
        raise ValueError("images must match all user image blocks in conversation order")

    for target_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError("target metadata must be objects")
        message_index = int(target.get("message_index", -1))
        expected_index = 2 + target_index * 2
        if message_index != expected_index:
            raise ValueError("target message indices must match assistant turns")
        target_xml = target.get("target_xml")
        if not isinstance(target_xml, str) or messages[message_index].get("content") != target_xml:
            raise ValueError("assistant content must equal target_xml metadata")
        parsed = parse_cfrp_output(target_xml)
        initializes_plan = target.get("initializes_plan") is True
        if initializes_plan != (parsed.plan is not None):
            raise ValueError("only an explicitly marked first turn may initialize <plan>")
        actions = parsed.actions or (parsed.action,)
        if len(actions) > MAX_STAGE1_ACTION_CHUNK or any(
            action not in _ALLOWED_ACTIONS for action in actions
        ):
            raise ValueError(f"unsupported Stage 1 action chunk: {actions}")
        if "STOP" in actions and actions != ("STOP",):
            raise ValueError("STOP must be the only Stage 1 chunk action")
    if check_images:
        for image_uri in images:
            image_path = local_image_path(image_uri)
            if not image_path.is_file():
                raise ValueError(f"image file is missing: {image_path}")


def iter_stage1_targets(example: Mapping[str, Any]) -> Iterable[str]:
    """Yield every supervised assistant XML response in conversation order."""

    for target in example.get("targets", ()):
        if isinstance(target, Mapping) and isinstance(target.get("target_xml"), str):
            yield target["target_xml"]


def local_image_path(source: str) -> Path:
    """Resolve a local path or ``file://`` URI for model-runtime media loading."""

    if not isinstance(source, str) or not source:
        raise ValueError("Stage 1 SFT image source must be a non-empty string")
    parsed = urlparse(source)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Stage 1 SFT image must be local: {source!r}")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"Stage 1 SFT image must be a local path or file URI: {source!r}")
    return Path(source)


def local_file_uri(uri: str) -> Path:
    """Backward-compatible strict ``file://`` resolver."""

    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Stage 1 SFT image must be a local file URI: {uri!r}")
    return local_image_path(uri)


def iter_stage1_sft_examples(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from load_stage1_sft_jsonl(path)
