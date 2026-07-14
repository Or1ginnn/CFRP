"""Shared temporal visual-context selection for Stage 1.

The model sees a compact nine-frame context: six uniformly spaced route
anchors plus the latest three consecutive control frames.  The raw rolling
context is deliberately longer, so training and rollout use the same view of
the preceding route without sending every frame to Qwen.
"""

from __future__ import annotations

from typing import Sequence, Tuple, TypeVar


DEFAULT_VISUAL_CONTEXT_WINDOW = 32
DEFAULT_HISTORY_ANCHOR_COUNT = 6
DEFAULT_RECENT_CONTIGUOUS_COUNT = 3
DEFAULT_MODEL_VISUAL_FRAME_COUNT = (
    DEFAULT_HISTORY_ANCHOR_COUNT + DEFAULT_RECENT_CONTIGUOUS_COUNT
)

T = TypeVar("T")


def temporal_history_indices(
    frame_count: int,
    *,
    context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW,
    history_anchor_count: int = DEFAULT_HISTORY_ANCHOR_COUNT,
    recent_contiguous_count: int = DEFAULT_RECENT_CONTIGUOUS_COUNT,
) -> Tuple[int, ...]:
    """Select chronological indices from the latest raw visual context.

    For a mature context this returns six uniform anchors from the earlier
    part of the 32-frame window followed by the last three adjacent frames.
    During the first eight turns it returns every available frame once rather
    than padding or duplicating observations.
    """

    if frame_count < 0:
        raise ValueError("frame_count must not be negative")
    if context_window < 1:
        raise ValueError("context_window must be at least 1")
    if history_anchor_count < 0 or recent_contiguous_count < 1:
        raise ValueError("invalid temporal history composition")
    if context_window < history_anchor_count + recent_contiguous_count:
        raise ValueError("context_window must fit the requested model-visible frames")
    if frame_count == 0:
        return tuple()

    start = max(0, frame_count - context_window)
    recent_start = max(start, frame_count - recent_contiguous_count)
    anchor_candidates = tuple(range(start, recent_start))
    anchors = _uniform_indices(anchor_candidates, history_anchor_count)
    recent = tuple(range(recent_start, frame_count))
    return anchors + recent


def select_temporal_history(
    values: Sequence[T],
    *,
    context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW,
    history_anchor_count: int = DEFAULT_HISTORY_ANCHOR_COUNT,
    recent_contiguous_count: int = DEFAULT_RECENT_CONTIGUOUS_COUNT,
) -> Tuple[T, ...]:
    """Return the model-visible temporal subset of chronological values."""

    context = tuple(values)[-context_window:]
    indices = temporal_history_indices(
        len(context),
        context_window=context_window,
        history_anchor_count=history_anchor_count,
        recent_contiguous_count=recent_contiguous_count,
    )
    return tuple(context[index] for index in indices)


def temporal_history_spec() -> dict[str, int | str]:
    """Machine-readable contract written into warm-up manifests."""

    return {
        "sampling": "uniform_history_anchors_plus_recent_contiguous",
        "visual_context_window": DEFAULT_VISUAL_CONTEXT_WINDOW,
        "history_anchor_count": DEFAULT_HISTORY_ANCHOR_COUNT,
        "recent_contiguous_count": DEFAULT_RECENT_CONTIGUOUS_COUNT,
        "model_visual_frame_count": DEFAULT_MODEL_VISUAL_FRAME_COUNT,
    }


def _uniform_indices(candidates: Sequence[int], count: int) -> Tuple[int, ...]:
    if count == 0 or not candidates:
        return tuple()
    take = min(count, len(candidates))
    if take == 1:
        return (candidates[-1],)
    last = len(candidates) - 1
    return tuple(candidates[(slot * last) // (take - 1)] for slot in range(take))
