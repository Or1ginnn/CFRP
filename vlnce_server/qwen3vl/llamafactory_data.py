"""LLaMA-Factory ShareGPT export for canonical CFRP Stage 1 SFT data."""

from __future__ import annotations

from typing import Any, Mapping

from .sft_manifest import local_image_path, validate_stage1_sft_example


LLAMAFACTORY_SCHEMA = "cfrp.llamafactory.stage1_sft.v1"


def make_llamafactory_stage1_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one canonical multi-turn chat into LLaMA-Factory ShareGPT data."""

    validate_stage1_sft_example(example, check_images=False)
    messages = example["messages"]
    system = messages[0]["content"]
    if not isinstance(system, str):
        raise ValueError("canonical Stage 1 system message must contain text")
    conversations = []
    images = []
    for message in messages[1:]:
        if message["role"] == "user":
            prompt, turn_images = _sharegpt_prompt_and_images(message["content"])
            conversations.append({"from": "human", "value": prompt})
            images.extend(turn_images)
        else:
            conversations.append({"from": "gpt", "value": message["content"]})
    converted = {
        "schema": LLAMAFACTORY_SCHEMA,
        "episode_id": example["episode_id"],
        "window_index": example["window_index"],
        "start_turn_index": example["start_turn_index"],
        "end_turn_index": example["end_turn_index"],
        "conversations": conversations,
        "system": system,
        "images": images,
        "targets": example["targets"],
    }
    validate_llamafactory_stage1_example(converted)
    return converted


def validate_llamafactory_stage1_example(example: Mapping[str, Any]) -> None:
    if example.get("schema") != LLAMAFACTORY_SCHEMA:
        raise ValueError(f"expected schema {LLAMAFACTORY_SCHEMA!r}")
    conversations = example.get("conversations")
    images = example.get("images")
    if not isinstance(conversations, list) or not conversations or len(conversations) % 2:
        raise ValueError("ShareGPT example must contain one or more human/gpt turns")
    expected_roles = [
        role for _ in range(len(conversations) // 2) for role in ("human", "gpt")
    ]
    if [message.get("from") for message in conversations] != expected_roles:
        raise ValueError("ShareGPT roles must alternate human and gpt")
    if any(not isinstance(message.get("value"), str) for message in conversations):
        raise ValueError("ShareGPT message values must be strings")
    if not isinstance(images, list) or not images:
        raise ValueError("ShareGPT example requires images")
    if sum(
        message["value"].count("<image>")
        for message in conversations
        if message["from"] == "human"
    ) != len(images):
        raise ValueError("ShareGPT image token count must equal image path count")
    for image in images:
        path = local_image_path(image)
        if not path.is_absolute():
            raise ValueError(f"LLaMA-Factory image path must be absolute: {image!r}")


def _sharegpt_prompt_and_images(blocks: list[Mapping[str, Any]]) -> tuple[str, list[str]]:
    fragments: list[str] = []
    images: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("text content block requires text")
            fragments.append(text)
        elif block_type == "image":
            fragments.append("<image>")
            images.append(str(local_image_path(block.get("image"))))
        else:
            raise ValueError(f"unsupported Stage 1 content block: {block_type!r}")
    return "\n".join(fragments), images
