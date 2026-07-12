"""Validation and loading for portable Stage 1 Qwen3-VL SFT manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

from vlnce_server.cfrp import parse_cfrp_output

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
    """Validate the model-visible conversation and terminal XML target."""

    if example.get("schema") != SFT_SCHEMA:
        raise ValueError(f"expected schema {SFT_SCHEMA!r}")
    messages = example.get("messages")
    images = example.get("images")
    target_xml = example.get("target_xml")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("messages must contain system, user, and assistant entries")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("messages must be ordered system/user/assistant")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")
    if not isinstance(target_xml, str) or messages[-1].get("content") != target_xml:
        raise ValueError("assistant content must equal target_xml")
    parsed = parse_cfrp_output(target_xml)
    if parsed.action not in _ALLOWED_ACTIONS:
        raise ValueError(f"unsupported Stage 1 action: {parsed.action}")
    user_content = messages[1].get("content")
    if not isinstance(user_content, list):
        raise ValueError("user content must be multimodal content blocks")
    prompt_images = [item.get("image") for item in user_content if item.get("type") == "image"]
    if prompt_images != images:
        raise ValueError("images must match user image blocks in order")
    if check_images:
        for image_uri in images:
            image_path = local_file_uri(image_uri)
            if not image_path.is_file():
                raise ValueError(f"image file is missing: {image_path}")


def local_file_uri(uri: str) -> Path:
    """Resolve a portable local ``file://`` URI without accepting remote media."""

    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Stage 1 SFT image must be a local file URI: {uri!r}")
    return Path(unquote(parsed.path))


def iter_stage1_sft_examples(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from load_stage1_sft_jsonl(path)
