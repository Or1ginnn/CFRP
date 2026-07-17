"""Inference contract for the JanusVLN-style Phase 0 action policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .action_sft import ACTION_SYSTEM_PROMPT, ALLOWED_ACTIONS, janus_frame_indices


_ACTION_XML = re.compile(r"^<action>(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)</action>$")


@dataclass(frozen=True)
class ActionModelRequest:
    instruction: str
    visual_history: tuple[Any, ...]
    allowed_actions: tuple[str, ...] = ALLOWED_ACTIONS

    @classmethod
    def from_episode_history(
        cls,
        instruction: str,
        observations: Sequence[Any],
        allowed_actions: Sequence[str] = ALLOWED_ACTIONS,
    ) -> "ActionModelRequest":
        if not observations:
            raise ValueError("action policy requires at least one observation")
        indices = janus_frame_indices(len(observations) - 1)
        return cls(
            instruction=instruction,
            visual_history=tuple(observations[index] for index in indices),
            allowed_actions=tuple(allowed_actions),
        )


def parse_action_xml(raw_xml: str, allowed_actions: Sequence[str] = ALLOWED_ACTIONS) -> str:
    match = _ACTION_XML.fullmatch(raw_xml.strip())
    if match is None:
        raise ValueError("action policy output must be exactly one <action>ACTION</action>")
    action = match.group(1)
    if action not in allowed_actions:
        raise ValueError(f"action is not allowed in the current environment: {action}")
    return action


def build_action_messages(request: ActionModelRequest) -> list[dict[str, Any]]:
    if not request.instruction.strip() or not request.visual_history:
        raise ValueError("action model request requires instruction and visual history")
    unsupported = set(request.allowed_actions) - set(ALLOWED_ACTIONS)
    if unsupported:
        raise ValueError(f"unsupported action names: {sorted(unsupported)}")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Navigation instruction: {request.instruction.strip()}\n"
                "The observations are ordered from earlier to later. The last image is "
                "the current observation."
            ),
        }
    ]
    content.extend({"type": "image", "image": image} for image in request.visual_history)
    content.append(
        {
            "type": "text",
            "text": "Allowed actions: {}.".format(", ".join(request.allowed_actions)),
        }
    )
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
