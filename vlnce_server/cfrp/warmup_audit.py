"""Semantic consistency checks for offline Stage 1 warm-up records."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import parse_cfrp_output, validate_output
from .rollout_exchange import Stage1RolloutRequest


def audit_stage1_warmup(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    check_frames: bool = False,
) -> dict[str, Any]:
    """Raise on any broken oracle trajectory, plan cursor, or prompt boundary."""

    if manifest.get("status") != "complete":
        raise ValueError("warm-up manifest must have status=complete")
    requested = tuple(str(value) for value in manifest.get("requested_episode_ids", ()))
    completed = tuple(str(value) for value in manifest.get("completed_episode_ids", ()))
    if not requested or requested != completed:
        raise ValueError("warm-up manifest requested and completed episode IDs must match")
    max_visual_history = int(manifest["max_visual_history"])
    max_action_history = int(manifest["max_action_history"])
    if max_visual_history < 1 or max_action_history < 0:
        raise ValueError("invalid history budgets in warm-up manifest")
    expected_frame_size = _expected_frame_size(manifest)
    temporal_spec = manifest.get("temporal_visual_history")
    _validate_temporal_spec(temporal_spec, max_visual_history)

    by_episode: dict[str, list[tuple[Stage1RolloutRequest, Any, Mapping[str, Any]]]] = defaultdict(list)
    seen_turns: set[tuple[str, int]] = set()
    action_counts: Counter[str] = Counter()
    progress_counts: Counter[str] = Counter()
    for line_number, record in enumerate(records, start=1):
        request_payload = record.get("model_input")
        target_xml = record.get("target_xml")
        oracle_only = record.get("oracle_only")
        if not isinstance(request_payload, Mapping) or not isinstance(target_xml, str):
            raise ValueError(f"line {line_number}: missing model_input or target_xml")
        if not isinstance(oracle_only, Mapping):
            raise ValueError(f"line {line_number}: missing oracle_only audit metadata")
        request = Stage1RolloutRequest.from_dict(request_payload)
        output = parse_cfrp_output(target_xml)
        validate_output(output, request.allowed_actions, previous_plan=request.current_plan, mode="stage1")
        key = (request.episode_id, request.turn_index)
        if key in seen_turns:
            raise ValueError(f"line {line_number}: duplicate episode/turn {key}")
        seen_turns.add(key)
        current_point = request.current_plan.current_points()[0]
        if output.subgoal != current_point.text:
            raise ValueError(f"line {line_number}: target subgoal does not match current plan point")
        if oracle_only.get("oracle_action") != output.action:
            raise ValueError(f"line {line_number}: target action does not match oracle action")
        if len(request.visual_history_paths) > max_visual_history:
            raise ValueError(f"line {line_number}: visual history exceeds manifest budget")
        if len(request.action_history) > max_action_history:
            raise ValueError(f"line {line_number}: action history exceeds manifest budget")
        if check_frames:
            _validate_frames(request.visual_history_paths, line_number, expected_frame_size)
        _validate_temporal_history_paths(request, temporal_spec, line_number)
        by_episode[request.episode_id].append((request, output, oracle_only))
        action_counts[output.action] += 1
        progress_counts[str(output.progress)] += 1

    if tuple(by_episode) != requested:
        raise ValueError("record episode IDs do not match manifest order")
    for episode_id, trajectory in by_episode.items():
        _audit_episode(episode_id, trajectory, max_action_history)
    return {
        "schema": "cfrp.stage1.warmup.audit.v1",
        "status": "passed",
        "records": len(records),
        "episodes": len(by_episode),
        "action_counts": dict(sorted(action_counts.items())),
        "progress_counts": dict(sorted(progress_counts.items())),
    }


def audit_sft_alignment(records: Sequence[Mapping[str, Any]], sft_examples: Sequence[Mapping[str, Any]]) -> None:
    """Ensure conversion preserved a one-to-one target ordering."""

    if len(records) != len(sft_examples):
        raise ValueError("warm-up and SFT record counts differ")
    for index, (record, example) in enumerate(zip(records, sft_examples), start=1):
        request = Stage1RolloutRequest.from_dict(record["model_input"])
        if example.get("episode_id") != request.episode_id or example.get("turn_index") != request.turn_index:
            raise ValueError(f"SFT example {index} does not match warm-up episode/turn")
        if example.get("target_xml") != record.get("target_xml"):
            raise ValueError(f"SFT example {index} target XML differs from warm-up record")


def _audit_episode(
    episode_id: str,
    trajectory: Sequence[tuple[Stage1RolloutRequest, Any, Mapping[str, Any]]],
    max_action_history: int,
) -> None:
    trajectory = sorted(trajectory, key=lambda item: item[0].turn_index)
    actions: list[str] = []
    previous_plan = None
    for expected_turn, (request, output, _oracle_only) in enumerate(trajectory):
        if request.turn_index != expected_turn or request.request_id != expected_turn:
            raise ValueError(f"episode {episode_id}: turns must be contiguous from zero")
        expected_history = tuple(actions[-max_action_history:]) if max_action_history else tuple()
        if request.action_history != expected_history:
            raise ValueError(f"episode {episode_id} turn {expected_turn}: action history is inconsistent")
        if previous_plan is not None:
            expected_plan = previous_plan.advance_current() if previous_progress == "advance" else previous_plan
            if request.current_plan != expected_plan:
                raise ValueError(f"episode {episode_id} turn {expected_turn}: plan cursor is inconsistent")
        previous_plan = request.current_plan
        previous_progress = output.progress
        actions.append(output.action)
    if actions.count("STOP") != 1 or actions[-1] != "STOP":
        raise ValueError(f"episode {episode_id}: trajectory must end with exactly one STOP")


def _expected_frame_size(manifest: Mapping[str, Any]) -> tuple[int, int] | None:
    contract = manifest.get("visual_contract")
    if not isinstance(contract, Mapping):
        return None
    size = contract.get("habitat_rgb_size")
    if not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError("visual contract must contain habitat_rgb_size=[width, height]")
    return (int(size[0]), int(size[1]))


def _validate_temporal_spec(value: Any, max_visual_history: int) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("temporal_visual_history must be an object")
    # Import lazily: Habitat's Stage 1 runner imports CFRP, while this audit
    # is only invoked after the package graph has finished initializing.
    from vlnce_server.habitat030.temporal_history import (
        DEFAULT_MODEL_VISUAL_FRAME_COUNT,
        temporal_history_spec,
    )

    expected = temporal_history_spec()
    if dict(value) != expected:
        raise ValueError("warm-up temporal visual history does not match the CFRP 6+3 contract")
    if max_visual_history != DEFAULT_MODEL_VISUAL_FRAME_COUNT:
        raise ValueError("6+3 temporal visual history requires max_visual_history=9")


def _validate_temporal_history_paths(
    request: Stage1RolloutRequest, temporal_spec: Any, line_number: int
) -> None:
    if temporal_spec is None:
        return
    from vlnce_server.habitat030.temporal_history import select_temporal_history

    paths = tuple(Path(path) for path in request.visual_history_paths)
    if not paths:
        raise ValueError(f"line {line_number}: missing temporal visual history")
    frames_dir = paths[-1].parent
    raw_paths = tuple(frames_dir / "frame-{:04d}.npy".format(index) for index in range(request.turn_index + 1))
    expected = select_temporal_history(raw_paths)
    if paths != expected:
        raise ValueError(f"line {line_number}: visual history does not match the slow-fast temporal contract")


def _validate_frames(
    paths: Sequence[str], line_number: int, expected_size: tuple[int, int] | None
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("frame audit requires numpy") from exc
    for path in paths:
        source = Path(path)
        if not source.is_file():
            raise ValueError(f"line {line_number}: missing RGB frame {source}")
        frame = np.load(source, mmap_mode="r")
        if frame.ndim != 3 or frame.shape[-1] != 3 or str(frame.dtype) != "uint8":
            raise ValueError(f"line {line_number}: invalid RGB frame {source}")
        if expected_size is not None:
            expected_width, expected_height = expected_size
            if tuple(frame.shape[:2]) != (expected_height, expected_width):
                raise ValueError(
                    f"line {line_number}: RGB frame shape {tuple(frame.shape)} does not match "
                    f"visual contract {expected_width}x{expected_height}"
                )
