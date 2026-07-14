"""Stateful slow-fast visual context for Stage 1 navigation.

The model receives a bounded visual snapshot: a slowly updated set of route
keyframes and the newest consecutive control observations.  Slow memory is
held stable between refreshes, unlike the previous per-turn re-sampling of a
rolling window.  This makes the context contract suitable for a future
stateful/KV-cached inference backend while keeping today's stateless SFT and
vLLM clients behaviourally identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, Tuple, TypeVar


DEFAULT_VISUAL_CONTEXT_WINDOW = 32
DEFAULT_HISTORY_ANCHOR_COUNT = 6
DEFAULT_RECENT_CONTIGUOUS_COUNT = 3
DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL = 5
DEFAULT_MODEL_VISUAL_FRAME_COUNT = (
    DEFAULT_HISTORY_ANCHOR_COUNT + DEFAULT_RECENT_CONTIGUOUS_COUNT
)

T = TypeVar("T")


@dataclass(frozen=True)
class SlowFastVisualHistory(Generic[T]):
    """Bounded raw context plus a slowly refreshed visual-memory snapshot.

    The first nine observations are visible without padding.  Once the visual
    budget is full, six slow-memory keyframes are refreshed every configured
    interval from the earlier portion of the raw context; the latest three
    observations remain a dense fast stream on every decision.
    """

    context: Tuple[T, ...] = tuple()
    slow_memory: Tuple[T, ...] = tuple()
    observation_count: int = 0
    slow_memory_updated_at: int = 0
    context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW
    history_anchor_count: int = DEFAULT_HISTORY_ANCHOR_COUNT
    recent_contiguous_count: int = DEFAULT_RECENT_CONTIGUOUS_COUNT
    slow_memory_update_interval: int = DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL

    @classmethod
    def create(
        cls,
        *,
        context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW,
        history_anchor_count: int = DEFAULT_HISTORY_ANCHOR_COUNT,
        recent_contiguous_count: int = DEFAULT_RECENT_CONTIGUOUS_COUNT,
        slow_memory_update_interval: int = DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL,
    ) -> "SlowFastVisualHistory[T]":
        _validate_composition(
            context_window,
            history_anchor_count,
            recent_contiguous_count,
            slow_memory_update_interval,
        )
        return cls(
            context_window=context_window,
            history_anchor_count=history_anchor_count,
            recent_contiguous_count=recent_contiguous_count,
            slow_memory_update_interval=slow_memory_update_interval,
        )

    @property
    def visible(self) -> Tuple[T, ...]:
        """Return chronological model-visible frames without duplicates."""

        if not self.context:
            return tuple()
        if self.history_anchor_count == 0:
            return self.context[-self.recent_contiguous_count :]
        if not self.slow_memory:
            return self.context
        return self.slow_memory + self.context[-self.recent_contiguous_count :]

    def reset(self, value: T) -> "SlowFastVisualHistory[T]":
        return SlowFastVisualHistory(
            context=(value,),
            context_window=self.context_window,
            history_anchor_count=self.history_anchor_count,
            recent_contiguous_count=self.recent_contiguous_count,
            slow_memory_update_interval=self.slow_memory_update_interval,
            observation_count=1,
        )

    def append(self, value: T) -> "SlowFastVisualHistory[T]":
        next_context = (self.context + (value,))[-self.context_window :]
        next_count = self.observation_count + 1
        slow_memory = self.slow_memory
        updated_at = self.slow_memory_updated_at
        if self._should_refresh_slow_memory(next_count):
            slow_memory = _uniform_values(
                next_context[: -self.recent_contiguous_count],
                self.history_anchor_count,
            )
            updated_at = next_count
        return SlowFastVisualHistory(
            context=next_context,
            slow_memory=slow_memory,
            observation_count=next_count,
            slow_memory_updated_at=updated_at,
            context_window=self.context_window,
            history_anchor_count=self.history_anchor_count,
            recent_contiguous_count=self.recent_contiguous_count,
            slow_memory_update_interval=self.slow_memory_update_interval,
        )

    def _should_refresh_slow_memory(self, next_count: int) -> bool:
        if self.history_anchor_count == 0:
            return False
        full_budget = self.history_anchor_count + self.recent_contiguous_count
        return next_count >= full_budget and (
            not self.slow_memory
            or next_count - self.slow_memory_updated_at >= self.slow_memory_update_interval
        )


def select_temporal_history(
    values: Sequence[T],
    *,
    context_window: int = DEFAULT_VISUAL_CONTEXT_WINDOW,
    history_anchor_count: int = DEFAULT_HISTORY_ANCHOR_COUNT,
    recent_contiguous_count: int = DEFAULT_RECENT_CONTIGUOUS_COUNT,
    slow_memory_update_interval: int = DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL,
) -> Tuple[T, ...]:
    """Replay the stateful slow-fast contract over chronological values.

    This helper is used by dataset conversion/audit paths which retain every
    raw frame, while online loops retain the equivalent ``SlowFastVisualHistory``
    state directly.
    """

    history = SlowFastVisualHistory[T].create(
        context_window=context_window,
        history_anchor_count=history_anchor_count,
        recent_contiguous_count=recent_contiguous_count,
        slow_memory_update_interval=slow_memory_update_interval,
    )
    for value in values:
        history = history.reset(value) if not history.context else history.append(value)
    return history.visible


def temporal_history_indices(
    frame_count: int,
    **kwargs: int,
) -> Tuple[int, ...]:
    """Return indices selected by the stateful slow-fast contract."""

    if frame_count < 0:
        raise ValueError("frame_count must not be negative")
    return tuple(select_temporal_history(tuple(range(frame_count)), **kwargs))


def temporal_history_spec() -> dict[str, int | str]:
    """Machine-readable visual contract written into warm-up manifests."""

    return {
        "sampling": "stateful_slow_memory_plus_recent_contiguous",
        "visual_context_window": DEFAULT_VISUAL_CONTEXT_WINDOW,
        "history_anchor_count": DEFAULT_HISTORY_ANCHOR_COUNT,
        "recent_contiguous_count": DEFAULT_RECENT_CONTIGUOUS_COUNT,
        "slow_memory_update_interval": DEFAULT_SLOW_MEMORY_UPDATE_INTERVAL,
        "model_visual_frame_count": DEFAULT_MODEL_VISUAL_FRAME_COUNT,
    }


def _validate_composition(
    context_window: int,
    history_anchor_count: int,
    recent_contiguous_count: int,
    slow_memory_update_interval: int,
) -> None:
    if context_window < 1:
        raise ValueError("context_window must be at least 1")
    if history_anchor_count < 0 or recent_contiguous_count < 1:
        raise ValueError("invalid temporal history composition")
    if context_window < history_anchor_count + recent_contiguous_count:
        raise ValueError("context_window must fit the requested model-visible frames")
    if slow_memory_update_interval < 1:
        raise ValueError("slow_memory_update_interval must be at least 1")


def _uniform_values(values: Sequence[T], count: int) -> Tuple[T, ...]:
    if count == 0 or not values:
        return tuple()
    take = min(count, len(values))
    if take == 1:
        return (values[-1],)
    last = len(values) - 1
    return tuple(values[(slot * last) // (take - 1)] for slot in range(take))
