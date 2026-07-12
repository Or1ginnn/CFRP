"""Portable Qwen3-VL Stage 1 SFT manifest construction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vlnce_server.cfrp import Stage1RolloutRequest, parse_cfrp_output, validate_output

from .stage1 import build_stage1_messages


SFT_SCHEMA = "cfrp.qwen3vl.stage1_sft.v1"


def make_stage1_sft_example(record: Mapping[str, Any], image_uris: Sequence[str]) -> dict[str, Any]:
    """Turn one validated oracle record into a multimodal assistant example."""

    request_payload = record.get("model_input")
    target_xml = record.get("target_xml")
    if not isinstance(request_payload, Mapping) or not isinstance(target_xml, str):
        raise ValueError("warm-up record requires model_input and target_xml")
    request = Stage1RolloutRequest.from_dict(request_payload)
    if len(image_uris) != len(request.visual_history_paths):
        raise ValueError("image URI count must match visual history")
    output = parse_cfrp_output(target_xml)
    validate_output(output, request.allowed_actions, previous_plan=request.current_plan, mode="stage1")

    messages = build_stage1_messages(
        _stage1_request_with_images(request, tuple(image_uris))
    )
    messages.append({"role": "assistant", "content": target_xml})
    return {
        "schema": SFT_SCHEMA,
        "episode_id": request.episode_id,
        "turn_index": request.turn_index,
        "messages": messages,
        "images": list(image_uris),
        "target_xml": target_xml,
    }


def _stage1_request_with_images(request: Stage1RolloutRequest, image_uris: tuple[str, ...]):
    from .stage1 import Stage1ModelRequest

    return Stage1ModelRequest(
        instruction=request.instruction,
        current_plan=request.current_plan,
        visual_history=image_uris,
        action_history=request.action_history,
        allowed_actions=request.allowed_actions,
    )
