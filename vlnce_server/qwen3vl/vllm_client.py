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

from .stage1 import Stage1ModelRequest, build_stage1_messages


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

    def generate_xml(self, stage1_request: Stage1ModelRequest) -> str:
        payload = {
            "model": self.model,
            "messages": make_openai_messages(stage1_request),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
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
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMRequestError("vLLM response lacks choices[0].message.content: {}".format(body)) from exc


def make_openai_messages(stage1_request: Stage1ModelRequest) -> List[Dict[str, Any]]:
    """Translate CFRP's model-visible messages to vLLM's OpenAI schema."""

    messages = build_stage1_messages(stage1_request)
    system = {"role": "system", "content": messages[0]["content"]}
    content = []
    for item in messages[1]["content"]:
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image":
            content.append({"type": "image_url", "image_url": {"url": _png_data_uri(item["image"])}})
        else:
            raise ValueError("unsupported Stage 1 content type: {}".format(item["type"]))
    return [system, {"role": "user", "content": content}]


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
