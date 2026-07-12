"""Long-lived Qwen3-VL file worker for the split-runtime Stage 1 baseline."""

from __future__ import annotations

import time
from pathlib import Path

from vlnce_server.cfrp import (
    Stage1RolloutResponse,
    read_request,
    response_path,
    write_response,
)

from .stage1 import Qwen3VLStage1Policy, Stage1ModelRequest


def run_file_worker(
    exchange_dir: str | Path,
    *,
    model_name_or_path: str,
    max_new_tokens: int,
    adapter_path: str | None = None,
    poll_seconds: float = 0.1,
    max_requests: int | None = None,
) -> int:
    """Load the model once and serve ordered request files until stopped."""

    exchange = Path(exchange_dir)
    exchange.mkdir(parents=True, exist_ok=True)
    policy = Qwen3VLStage1Policy.from_pretrained(
        model_name_or_path,
        max_new_tokens=max_new_tokens,
        adapter_path=adapter_path,
    )
    handled = 0
    next_request_id = 0
    stop_file = exchange / "worker.stop"

    while not stop_file.exists():
        request_file = exchange / f"request-{next_request_id:06d}.json"
        if not request_file.exists():
            time.sleep(poll_seconds)
            continue
        response_file = response_path(exchange, next_request_id)
        if response_file.exists():
            next_request_id += 1
            continue

        request = read_request(request_file)
        try:
            request_model = Stage1ModelRequest(
                instruction=request.instruction,
                current_plan=request.current_plan,
                visual_history=tuple(_load_rgb(path) for path in request.visual_history_paths),
                action_history=request.action_history,
                allowed_actions=request.allowed_actions,
            )
            response = Stage1RolloutResponse(
                episode_id=request.episode_id,
                request_id=request.request_id,
                turn_index=request.turn_index,
                raw_xml=policy.generate_xml(request_model),
            )
        except Exception as exc:
            response = Stage1RolloutResponse(
                episode_id=request.episode_id,
                request_id=request.request_id,
                turn_index=request.turn_index,
                raw_xml="",
                error=f"{type(exc).__name__}: {exc}",
            )
        write_response(response_file, response)
        handled += 1
        next_request_id += 1
        if max_requests is not None and handled >= max_requests:
            break
    return handled


def _load_rgb(path: str):
    from PIL import Image

    import numpy as np

    return Image.fromarray(np.load(path))
