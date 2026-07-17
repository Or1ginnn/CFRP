"""Stateless OpenAI-compatible vLLM client for action-only navigation."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .action_policy import ActionModelRequest, build_action_messages
from .vision import qwen3vl_processor_kwargs
from .vllm_client import VLLMRequestError, _openai_content


class VLLMActionClient:
    """Predict one primitive action from a complete sampled episode prefix."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_new_tokens: int = 32,
        timeout_seconds: float = 600.0,
        seed: int = 123,
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.seed = seed

    def generate_xml(self, request: ActionModelRequest) -> str:
        messages = build_action_messages(request)
        openai_messages = [
            messages[0],
            {"role": "user", "content": _openai_content(messages[1]["content"])},
        ]
        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
            "mm_processor_kwargs": qwen3vl_processor_kwargs(),
        }
        http_request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VLLMRequestError(f"vLLM HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise VLLMRequestError(f"vLLM connection failed: {exc.reason}") from exc
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMRequestError("vLLM response lacks choices[0].message.content") from exc
