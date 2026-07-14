"""Canonical multi-turn Qwen3-VL Stage 1 SFT conversation construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vlnce_server.cfrp import Stage1RolloutRequest, parse_cfrp_output, validate_output
from vlnce_server.cfrp.prompts import STAGE1_SYSTEM_PROMPT


SFT_SCHEMA = "cfrp.qwen3vl.stage1_multiturn_sft.v1"
DEFAULT_CONVERSATION_TURNS = 4


def make_stage1_sft_conversations(
    records: Sequence[Mapping[str, Any]],
    image_uris: Sequence[Sequence[str]],
    *,
    max_turns: int = DEFAULT_CONVERSATION_TURNS,
) -> list[dict[str, Any]]:
    """Convert one complete expert episode into bounded multi-turn windows.

    The first episode turn teaches plan initialization. Later turns expose the
    controller-owned plan and append only visual observations that have not
    already appeared in the current window. A new window is self-contained: it
    starts with the full visible slow-fast context from its first record.
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
            turn_images = _window_turn_images(
                row_images,
                first_in_window=record_index == start,
                new_frame_count=(
                    None
                    if previous_turn_index is None
                    else request.turn_index - previous_turn_index
                ),
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
    row_images: Sequence[str], *, first_in_window: bool, new_frame_count: int | None
) -> tuple[str, ...]:
    if not row_images:
        raise ValueError("every conversation turn requires at least one image")
    if first_in_window:
        return tuple(row_images)
    if new_frame_count is None or new_frame_count < 1:
        raise ValueError("conversation turn indices must increase")
    # Action chunks contain at most three primitives. The slow-fast contract
    # guarantees that its visible tail contains those newly arrived frames.
    return tuple(row_images[-min(new_frame_count, len(row_images)) :])


def _turn_content(
    request: Stage1RolloutRequest,
    images: Sequence[str],
    *,
    initialize_plan: bool,
    first_in_window: bool,
) -> list[dict[str, Any]]:
    recent_actions = ", ".join(request.action_history) if request.action_history else "None"
    if initialize_plan:
        plan_text = "None. Initialize one compact <plan> from the instruction."
    else:
        assert request.current_plan is not None
        plan_text = request.current_plan.to_xml()
    context_kind = "window context" if first_in_window else "new streaming observations"
    text = f"""Navigation instruction:
{request.instruction}

Current compact plan:
{plan_text}

Executed recent actions (oldest to newest):
{recent_actions}

Allowed actions:
{", ".join(request.allowed_actions)}

The images below are {context_kind}. Continue the same navigation episode and output only the required XML."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, image in enumerate(images, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Observation {index} of {len(images)} (oldest to newest):",
            }
        )
        content.append({"type": "image", "image": image})
    return content
