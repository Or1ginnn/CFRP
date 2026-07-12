"""File-based request/response records for split Habitat and Qwen runtimes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .protocol import PlanPoint, PlanState, validate_plan


REQUEST_SCHEMA = "cfrp.stage1.request.v1"
RESPONSE_SCHEMA = "cfrp.stage1.response.v1"
_FORBIDDEN_REQUEST_KEYS = {
    "pose",
    "agent_position",
    "agent_rotation",
    "goal_positions",
    "distance_to_goal",
    "reference_path",
    "expert_path",
}


@dataclass(frozen=True)
class Stage1RolloutRequest:
    """One model-visible turn, serializable across Python environments."""

    episode_id: str
    request_id: int
    turn_index: int
    instruction: str
    current_plan: PlanState
    visual_history_paths: Tuple[str, ...]
    action_history: Tuple[str, ...]
    allowed_actions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("rollout request requires an episode_id")
        if self.request_id < 0 or self.turn_index < 0:
            raise ValueError("rollout request turn_index must be non-negative")
        if not self.instruction.strip():
            raise ValueError("rollout request requires an instruction")
        validate_plan(self.current_plan)
        if not self.visual_history_paths:
            raise ValueError("rollout request requires at least one RGB path")
        if not self.allowed_actions:
            raise ValueError("rollout request requires allowed actions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "turn_index": self.turn_index,
            "instruction": self.instruction,
            "current_plan": _plan_to_dict(self.current_plan),
            "visual_history_paths": list(self.visual_history_paths),
            "action_history": list(self.action_history),
            "allowed_actions": list(self.allowed_actions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1RolloutRequest":
        _validate_request_payload(payload)
        return cls(
            episode_id=str(payload["episode_id"]),
            request_id=int(payload["request_id"]),
            turn_index=int(payload["turn_index"]),
            instruction=str(payload["instruction"]),
            current_plan=_plan_from_dict(payload["current_plan"]),
            visual_history_paths=tuple(str(path) for path in payload["visual_history_paths"]),
            action_history=tuple(str(action) for action in payload["action_history"]),
            allowed_actions=tuple(str(action) for action in payload["allowed_actions"]),
        )


@dataclass(frozen=True)
class Stage1RolloutResponse:
    """Raw model output returned to the Habitat controller for validation."""

    episode_id: str
    request_id: int
    turn_index: int
    raw_xml: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESPONSE_SCHEMA,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "turn_index": self.turn_index,
            "raw_xml": self.raw_xml,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1RolloutResponse":
        if payload.get("schema") != RESPONSE_SCHEMA:
            raise ValueError("unsupported Stage 1 response schema")
        raw_xml = payload.get("raw_xml")
        if not isinstance(raw_xml, str):
            raise ValueError("Stage 1 response raw_xml must be a string")
        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("Stage 1 response error must be a string or null")
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            request_id=int(payload.get("request_id", -1)),
            turn_index=int(payload.get("turn_index", -1)),
            raw_xml=raw_xml,
            error=error,
        )


def request_path(exchange_dir: str | Path, request_id: int) -> Path:
    return Path(exchange_dir) / f"request-{request_id:06d}.json"


def response_path(exchange_dir: str | Path, request_id: int) -> Path:
    return Path(exchange_dir) / f"response-{request_id:06d}.json"


def write_request(path: str | Path, request: Stage1RolloutRequest) -> None:
    _atomic_write_json(Path(path), request.to_dict())


def read_request(path: str | Path) -> Stage1RolloutRequest:
    return Stage1RolloutRequest.from_dict(_read_json(Path(path)))


def write_response(path: str | Path, response: Stage1RolloutResponse) -> None:
    _atomic_write_json(Path(path), response.to_dict())


def read_response(path: str | Path) -> Stage1RolloutResponse:
    return Stage1RolloutResponse.from_dict(_read_json(Path(path)))


def wait_for_response(path: str | Path, timeout_seconds: float, poll_seconds: float = 0.1) -> Stage1RolloutResponse:
    response_file = Path(path)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if response_file.exists():
            return read_response(response_file)
        time.sleep(poll_seconds)
    raise TimeoutError(f"timed out waiting for model response: {response_file}")


def _plan_to_dict(plan: PlanState) -> dict[str, Any]:
    return {
        "global_goal": plan.global_goal,
        "points": [
            {"id": point.id, "status": point.status, "text": point.text}
            for point in plan.points
        ],
    }


def _plan_from_dict(payload: Any) -> PlanState:
    if not isinstance(payload, Mapping):
        raise ValueError("rollout request current_plan must be an object")
    points = payload.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValueError("rollout request plan points must be a list")
    plan = PlanState(
        global_goal=str(payload.get("global_goal", "")),
        points=tuple(
            PlanPoint(
                id=str(point.get("id", "")),
                status=str(point.get("status", "")),
                text=str(point.get("text", "")),
            )
            for point in points
            if isinstance(point, Mapping)
        ),
    )
    if len(plan.points) != len(points):
        raise ValueError("rollout request plan points must be objects")
    validate_plan(plan)
    return plan


def _validate_request_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported Stage 1 request schema")
    leaked = sorted(_FORBIDDEN_REQUEST_KEYS.intersection(payload))
    if leaked:
        raise ValueError(f"rollout request leaked oracle fields: {leaked}")
    for key in (
        "episode_id",
        "request_id",
        "turn_index",
        "instruction",
        "current_plan",
        "visual_history_paths",
        "action_history",
        "allowed_actions",
    ):
        if key not in payload:
            raise ValueError(f"rollout request missing {key}")
    for key in ("visual_history_paths", "action_history", "allowed_actions"):
        value = payload[key]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"rollout request {key} must be a list")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object expected: {path}")
    return payload
