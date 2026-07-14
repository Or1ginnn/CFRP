"""Qwen3-VL Stage 1 inference boundary.

This module intentionally has no top-level Torch, Transformers, or Habitat
imports.  It can therefore be imported by the lightweight tests and by the
Python 3.9 Habitat process without creating a model-runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from vlnce_server.cfrp import PlanState
from vlnce_server.cfrp.prompts import STAGE1_SYSTEM_PROMPT

from .vision import prepare_qwen3vl_image, qwen3vl_processor_kwargs
from vlnce_server.habitat030.temporal_history import (
    DEFAULT_HISTORY_ANCHOR_COUNT,
    DEFAULT_MODEL_VISUAL_FRAME_COUNT,
    DEFAULT_RECENT_CONTIGUOUS_COUNT,
)


DEFAULT_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


class Qwen3VLDependencyError(RuntimeError):
    """Raised when the optional Qwen3-VL inference stack is unavailable."""


@dataclass(frozen=True)
class Stage1ModelRequest:
    """The complete model-visible state for one Stage 1 navigation decision."""

    instruction: str
    current_plan: PlanState
    visual_history: Tuple[Any, ...]
    action_history: Tuple[str, ...]
    allowed_actions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("Stage 1 model request requires an instruction")
        if not self.visual_history:
            raise ValueError("Stage 1 model request requires at least one RGB observation")
        if not self.allowed_actions:
            raise ValueError("Stage 1 model request requires allowed actions")
        if any(not action for action in self.action_history):
            raise ValueError("Stage 1 action history must not contain empty actions")


def build_stage1_messages(request: Stage1ModelRequest) -> list[dict[str, Any]]:
    """Build Qwen's multimodal chat messages from strictly model-visible state.

    Frames remain in chronological order.  RGB payloads can be numpy arrays or
    PIL images, both of which are accepted by ``qwen_vl_utils`` at runtime.
    """

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _stage1_context_text(request),
        }
    ]
    for index, rgb in enumerate(request.visual_history, start=1):
        content.append(
            {
                "type": "text",
                "text": _frame_label(index, len(request.visual_history)),
            }
        )
        content.append({"type": "image", "image": prepare_qwen3vl_image(rgb)})

    return [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _frame_label(index: int, total: int) -> str:
    """Describe the temporal role without leaking any privileged state."""

    if total == DEFAULT_MODEL_VISUAL_FRAME_COUNT:
        if index <= DEFAULT_HISTORY_ANCHOR_COUNT:
            return (
                f"Route-history anchor {index} of {DEFAULT_HISTORY_ANCHOR_COUNT} "
                "(uniformly sampled, oldest to newest):"
            )
        recent_index = index - DEFAULT_HISTORY_ANCHOR_COUNT
        return (
            f"Recent consecutive observation {recent_index} of "
            f"{DEFAULT_RECENT_CONTIGUOUS_COUNT} (oldest to newest):"
        )
    return f"Observation frame {index} of {total} (oldest to newest):"


class Qwen3VLStage1Policy:
    """Generate one Stage 1 CFRP XML decision with Qwen3-VL.

    The policy returns raw XML deliberately.  ``Stage1EpisodeRunner`` remains
    the only component that parses, validates, and executes an action.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int = 128,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_QWEN3_VL_MODEL,
        *,
        device_map: str = "auto",
        max_new_tokens: int = 128,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        adapter_path: Optional[str] = None,
    ) -> "Qwen3VLStage1Policy":
        """Load Qwen3-VL only in the dedicated Python 3.10 model environment."""

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise Qwen3VLDependencyError(
                "Qwen3-VL inference requires torch and transformers>=4.57.0. "
                "Install them in the dedicated model environment, not the Habitat environment."
            ) from exc

        load_kwargs = dict(model_kwargs or {})
        load_kwargs.setdefault("dtype", torch.bfloat16)
        load_kwargs.setdefault("device_map", device_map)
        model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **load_kwargs)
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise Qwen3VLDependencyError(
                    "Loading a Qwen3-VL LoRA adapter requires peft in the model environment."
                ) from exc
            model = PeftModel.from_pretrained(model, adapter_path)
        processor = AutoProcessor.from_pretrained(model_name_or_path, **qwen3vl_processor_kwargs())
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

    def generate_xml(self, request: Stage1ModelRequest) -> str:
        """Generate raw XML for a single navigation turn without repairing it."""

        messages = build_stage1_messages(request)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_inputs_to_model_device(inputs, self.model)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        continuation_ids = _strip_prompt_tokens(generated_ids, inputs["input_ids"])
        decoded = self.processor.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != 1:
            raise RuntimeError(f"expected one Stage 1 generation, got {len(decoded)}")
        return decoded[0].strip()


def _stage1_context_text(request: Stage1ModelRequest) -> str:
    actions = ", ".join(request.allowed_actions)
    recent_actions = ", ".join(request.action_history) if request.action_history else "None"
    return f"""Navigation instruction:
{request.instruction}

Controller-owned compact plan (read-only):
{request.current_plan.to_xml()}

Executed recent actions (oldest to newest):
{recent_actions}

Allowed actions:
{actions}

Use the observation frames below. Output only the required Stage 1 XML."""


def _move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    move = getattr(inputs, "to", None)
    return move(device) if callable(move) else inputs


def _strip_prompt_tokens(generated_ids: Any, input_ids: Any) -> list[Any]:
    """Remove the prompt prefix for each batch item without requiring Torch."""

    return [
        output_ids[len(prompt_ids) :]
        for prompt_ids, output_ids in zip(input_ids, generated_ids)
    ]
