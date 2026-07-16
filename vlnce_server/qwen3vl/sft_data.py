"""Canonical multi-turn Qwen3-VL Stage 1 SFT conversation construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vlnce_server.cfrp import Stage1RolloutRequest, parse_cfrp_output, validate_output
from vlnce_server.cfrp.prompts import STAGE1_SYSTEM_PROMPT
from .stage1 import DEFAULT_STAGE1_STREAMING_TURNS, build_stage1_turn_content


SFT_SCHEMA = "cfrp.qwen3vl.stage1_streaming_sft.v2"
DEFAULT_CONVERSATION_TURNS = DEFAULT_STAGE1_STREAMING_TURNS
DEFAULT_STREAM_HISTORY_ANCHORS = 8
DEFAULT_STREAM_OBSERVATIONS_PER_TURN = 1


def make_stage1_sft_conversations(
    records: Sequence[Mapping[str, Any]],
    image_uris: Sequence[Sequence[str]],
    *,
    max_turns: int = DEFAULT_CONVERSATION_TURNS,
) -> list[dict[str, Any]]:
    """Convert one complete expert episode into bounded multi-turn windows.

    The first episode turn teaches plan initialization. A new window starts
    with up to eight uniformly sampled historical anchors plus its current
    observation. Later turns append exactly one new current observation. This
    mirrors StreamVLN's bounded fast dialogue while retaining CFRP's XML state.
    """

    if not records or len(records) != len(image_uris):
        raise ValueError("records and image URI rows must be non-empty and aligned")
    if max_turns < 1:
        raise ValueError("max_turns must be at least one")

    requests = [_validated_record(record) for record in records]
    episode_ids = {request.episode_id for request, _target in requests}
    if len(episode_ids) != 1:
        raise ValueError("one Stage 1 conversation source must contain exactly one episode")

    conversations = []
    for window_index, start in enumerate(range(0, len(records), max_turns)):
        stop = min(start + max_turns, len(records))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT}
        ]
        flattened_images: list[str] = []
        targets: list[dict[str, Any]] = []
        previous_turn_index: int | None = None

        for record_index in range(start, stop):
            request, target_xml = requests[record_index]
            row_images = tuple(str(uri) for uri in image_uris[record_index])
            if len(row_images) != len(request.visual_history_paths):
                raise ValueError("image URI count must match visual history")
            if previous_turn_index is not None and request.turn_index <= previous_turn_index:
                raise ValueError("conversation turn indices must increase")
            turn_images = _window_turn_images(
                row_images,
                first_in_window=record_index == start,
            )
            previous_turn_index = request.turn_index
            flattened_images.extend(turn_images)
            initialize_plan = record_index == 0 and request.turn_index == 0
            response = _assistant_response(request, target_xml, initialize_plan)
            messages.append(
                {
                    "role": "user",
                    "content": _turn_content(
                        request,
                        turn_images,
                        initialize_plan=initialize_plan,
                        first_in_window=record_index == start,
                    ),
                }
            )
            messages.append({"role": "assistant", "content": response})
            targets.append(
                {
                    "message_index": len(messages) - 1,
                    "request_id": request.request_id,
                    "turn_index": request.turn_index,
                    "initializes_plan": initialize_plan,
                    "target_xml": response,
                }
            )

        conversations.append(
            {
                "schema": SFT_SCHEMA,
                "episode_id": requests[0][0].episode_id,
                "window_index": window_index,
                "start_turn_index": targets[0]["turn_index"],
                "end_turn_index": targets[-1]["turn_index"],
                "visual_contract": {
                    "history_anchor_count": DEFAULT_STREAM_HISTORY_ANCHORS,
                    "new_observations_per_turn": DEFAULT_STREAM_OBSERVATIONS_PER_TURN,
                    "max_active_dialogue_turns": DEFAULT_CONVERSATION_TURNS,
                    "max_window_images": (
                        DEFAULT_STREAM_HISTORY_ANCHORS + DEFAULT_CONVERSATION_TURNS
                    ),
                },
                "messages": messages,
                "images": flattened_images,
                "targets": targets,
            }
        )
    return conversations


def make_stage1_sft_example(
    record: Mapping[str, Any], image_uris: Sequence[str]
) -> dict[str, Any]:
    """Backward-compatible helper for a one-turn conversation window."""

    return make_stage1_sft_conversations([record], [image_uris], max_turns=1)[0]


def _validated_record(
    record: Mapping[str, Any],
) -> tuple[Stage1RolloutRequest, str]:
    request_payload = record.get("model_input")
    target_xml = record.get("target_xml")
    if not isinstance(request_payload, Mapping) or not isinstance(target_xml, str):
        raise ValueError("warm-up record requires model_input and target_xml")
    request = Stage1RolloutRequest.from_dict(request_payload)
    if request.current_plan is None:
        raise ValueError("oracle warm-up record requires its deterministic plan label")
    output = parse_cfrp_output(target_xml)
    validate_output(
        output,
        request.allowed_actions,
        previous_plan=request.current_plan,
        mode="stage1",
    )
    return request, target_xml


def _assistant_response(
    request: Stage1RolloutRequest, target_xml: str, initialize_plan: bool
) -> str:
    if not initialize_plan:
        return target_xml
    assert request.current_plan is not None
    response = f"{request.current_plan.to_xml()}\n{target_xml}"
    output = parse_cfrp_output(response)
    validate_output(
        output,
        request.allowed_actions,
        previous_plan=None,
        mode="stage1",
    )
    return response


def _window_turn_images(
    row_images: Sequence[str], *, first_in_window: bool
) -> tuple[str, ...]:
    if not row_images:
        raise ValueError("every conversation turn requires at least one image")
    if first_in_window:
        if len(row_images) > DEFAULT_STREAM_HISTORY_ANCHORS + 1:
            raise ValueError("window context exceeds eight history anchors plus current observation")
        return tuple(row_images)
    return (str(row_images[-1]),)


def _turn_content(
    request: Stage1RolloutRequest,
    images: Sequence[str],
    *,
    initialize_plan: bool,
    first_in_window: bool,
) -> list[dict[str, Any]]:
    return build_stage1_turn_content(
        request,
        tuple(images),
        initialize_plan=initialize_plan,
        first_in_window=first_in_window,
    )
