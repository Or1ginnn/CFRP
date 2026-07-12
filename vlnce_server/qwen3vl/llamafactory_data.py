"""LLaMA-Factory ShareGPT export for canonical CFRP Stage 1 SFT data."""

from __future__ import annotations

from typing import Any, Mapping

from .sft_manifest import local_image_path, validate_stage1_sft_example


LLAMAFACTORY_SCHEMA = "cfrp.llamafactory.stage1_sft.v1"


def make_llamafactory_stage1_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one canonical multimodal chat into LLaMA-Factory ShareGPT data."""

    validate_stage1_sft_example(example, check_images=False)
    messages = example["messages"]
    system = messages[0]["content"]
    user_blocks = messages[1]["content"]
    target_xml = example["target_xml"]
    if not isinstance(system, str) or not isinstance(user_blocks, list):
        raise ValueError("canonical Stage 1 messages must contain text system and multimodal user content")
    prompt, images = _sharegpt_prompt_and_images(user_blocks)
    converted = {
        "schema": LLAMAFACTORY_SCHEMA,
        "episode_id": example["episode_id"],
        "turn_index": example["turn_index"],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target_xml},
        ],
        "system": system,
        "images": images,
        "target_xml": target_xml,
    }
    validate_llamafactory_stage1_example(converted)
    return converted


def validate_llamafactory_stage1_example(example: Mapping[str, Any]) -> None:
    if example.get("schema") != LLAMAFACTORY_SCHEMA:
        raise ValueError(f"expected schema {LLAMAFACTORY_SCHEMA!r}")
    conversations = example.get("conversations")
    images = example.get("images")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("ShareGPT example must contain one human and one gpt message")
    if [message.get("from") for message in conversations] != ["human", "gpt"]:
        raise ValueError("ShareGPT roles must be human then gpt")
    prompt = conversations[0].get("value")
    target = conversations[1].get("value")
    if not isinstance(prompt, str) or not isinstance(target, str) or target != example.get("target_xml"):
        raise ValueError("ShareGPT prompt/target is malformed")
    if not isinstance(images, list) or not images:
        raise ValueError("ShareGPT example requires images")
    if prompt.count("<image>") != len(images):
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
