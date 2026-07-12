"""Deterministic Stage 1 plan initialization from an R2R instruction."""

from __future__ import annotations

import re

from .protocol import PlanPoint, PlanState


_NAVIGATION_VERB = r"(?:go|walk|head|move|turn|enter|exit|leave|continue|pass|stop|take|proceed|follow|approach|cross)"
_CLAUSE_BOUNDARY = re.compile(
    rf"\s*(?:;|\.|\band then\b|\bthen\b|\band\s+(?={_NAVIGATION_VERB}\b)|,\s*(?={_NAVIGATION_VERB}\b))\s*",
    re.IGNORECASE,
)


def initialize_plan_from_instruction(instruction: str, max_points: int = 4) -> PlanState:
    """Make a compact read-only execution plan without route-oracle input.

    The initializer is deliberately deterministic: the same instruction always
    yields the same plan, which prevents a hidden planner from becoming an
    untracked source of supervision during Phase 0.
    """

    normalized = _normalize(instruction)
    if not normalized:
        raise ValueError("cannot initialize a plan from an empty instruction")
    if max_points < 1:
        raise ValueError("max_points must be at least 1")

    clauses = tuple(clause for clause in (_normalize(item) for item in _CLAUSE_BOUNDARY.split(normalized)) if clause)
    points_text = clauses[:max_points] or (normalized,)
    points = tuple(
        PlanPoint(
            id=f"p{index}",
            status="current" if index == 1 else "todo",
            text=text,
        )
        for index, text in enumerate(points_text, start=1)
    )
    return PlanState(global_goal=" -> ".join(points_text), points=points)


def advance_turn_indices(action_count: int, plan: PlanState) -> tuple[int, ...]:
    """Assign normal cursor advances evenly across non-STOP oracle actions."""

    if action_count < 1:
        raise ValueError("action_count must be positive")
    transitions = len(plan.points) - 1
    if transitions <= 0:
        return tuple()
    if action_count <= transitions:
        return tuple(range(action_count - 1))
    return tuple((action_count * index) // len(plan.points) - 1 for index in range(1, len(plan.points)))


def _normalize(text: str) -> str:
    return " ".join(text.strip(" ,;").split())
