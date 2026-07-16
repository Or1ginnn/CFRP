"""OpenAI-compatible vLLM client for concurrent CFRP Stage 1 requests.

The client deliberately depends only on the standard library at import time so
the Habitat 0.3 runtime remains separate from the vLLM server runtime.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .stage1 import (
    DEFAULT_STAGE1_STREAMING_TURNS,
    Stage1ModelRequest,
    build_stage1_messages,
    build_stage1_turn_content,
)
from .vision import prepare_qwen3vl_image, qwen3vl_processor_kwargs


class VLLMRequestError(RuntimeError):
    """Raised when the local vLLM OpenAI endpoint cannot serve a request."""


class VLLMStage1Client:
    """Submit one Stage 1 request to a continuously-batching vLLM server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_new_tokens: int = 128,
        timeout_seconds: float = 600.0,
        seed: int = 123,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self._messages: List[Dict[str, Any]] = []
        self._turn_count = 0

    def generate_xml(self, stage1_request: Stage1ModelRequest) -> str:
        if self._turn_count % DEFAULT_STAGE1_STREAMING_TURNS == 0:
            self._messages = make_openai_messages(stage1_request)
        else:
            self._messages.append(make_openai_streaming_user_message(stage1_request))
        payload = {
            "model": self.model,
            "messages": self._messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
            "mm_processor_kwargs": qwen3vl_processor_kwargs(),
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VLLMRequestError("vLLM HTTP {}: {}".format(exc.code, detail)) from exc
        except URLError as exc:
            raise VLLMRequestError("vLLM connection failed: {}".format(exc.reason)) from exc
        try:
            output = str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMRequestError("vLLM response lacks choices[0].message.content: {}".format(body)) from exc
        self._messages.append({"role": "assistant", "content": output})
        self._turn_count += 1
        return output

    def reset(self) -> None:
        """Clear one episode's bounded streaming dialogue state."""

        self._messages = []
        self._turn_count = 0


def make_openai_messages(stage1_request: Stage1ModelRequest) -> List[Dict[str, Any]]:
    """Translate CFRP's model-visible messages to vLLM's OpenAI schema."""

    messages = build_stage1_messages(stage1_request)
    system = {"role": "system", "content": messages[0]["content"]}
    return [system, {"role": "user", "content": _openai_content(messages[1]["content"])}]


def make_openai_streaming_user_message(
    stage1_request: Stage1ModelRequest,
) -> Dict[str, Any]:
    """Append only the latest RGB observation inside an active eight-turn window."""

    content = build_stage1_turn_content(
        stage1_request,
        (stage1_request.visual_history[-1],),
        initialize_plan=False,
        first_in_window=False,
    )
    return {"role": "user", "content": _openai_content(content)}


def _openai_content(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content = []
    for item in items:
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_uri(prepare_qwen3vl_image(item["image"]))},
                }
            )
        else:
            raise ValueError("unsupported Stage 1 content type: {}".format(item["type"]))
    return content


def _png_data_uri(image: Any) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise VLLMRequestError("Pillow is required in the Habitat evaluation environment") from exc
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
