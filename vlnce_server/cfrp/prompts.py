"""Prompt builders for the CFRP loop."""

from __future__ import annotations

from collections.abc import Sequence

from .protocol import PlanState


CFRP_SYSTEM_PROMPT = """You are a continuous Vision-and-Language Navigation planner.

Maintain a persistent structured plan state. The plan is a control state, not a reasoning trace.
Output only XML. Do not explain your reasoning.

At each step, choose one tool:
- continue: follow the current plan.
- replan: update the current and future plan when the current plan is no longer reliable.

STOP is an action, not a tool. Output a new <plan> only when initializing or replanning.
During replanning, you may instead output one <plan_update>. During normal execution, do not repeat the full plan.
Done plan points are immutable. A new plan must contain exactly one current point.
The <action> field must contain exactly one action selected from the current step's Allowed actions.
"""

STAGE1_SYSTEM_PROMPT = """You are a continuous Vision-and-Language Navigation agent.

The controller owns a persistent structured plan. Read it as execution context; do not rewrite it.
Output only XML with exactly one <progress>, one <subgoal>, and one <action> field.
Set <progress> to hold or advance. advance moves the normal plan cursor to the next point.
Do not output <tool>, <plan>, <plan_update>, or free-form reasoning.
STOP is a primitive action. The <action> must be selected from the current Allowed actions.
"""


def build_step_prompt(
    *,
    full_instruction: str,
    allowed_actions: Sequence[str],
    current_observation: str,
    recent_visual_history: Sequence[str] | None = None,
    recent_actions: Sequence[str] | None = None,
    current_plan: PlanState | None = None,
    active_instruction_excerpt: str | None = None,
    mode: str = "stage2",
) -> str:
    history_lines = _format_lines(recent_visual_history or (), empty="None")
    action_lines = _format_lines(recent_actions or (), empty="None")
    if mode not in {"stage1", "stage2"}:
        raise ValueError(f"invalid protocol mode: {mode}")
    if mode == "stage1" and current_plan is None:
        raise ValueError("Stage 1 prompt requires a controller-owned current plan")
    plan_text = current_plan.to_xml() if current_plan else "None. Please initialize the plan."
    excerpt = active_instruction_excerpt if active_instruction_excerpt else "None"

    return f"""Full instruction:
{full_instruction}

Allowed actions:
{", ".join(allowed_actions)}

Current observation:
{current_observation}

Recent visual history:
{history_lines}

Recent actions:
{action_lines}

Current compact plan:
{plan_text}

Active instruction excerpt:
{excerpt}
"""


def _format_lines(values: Sequence[str], empty: str) -> str:
    if not values:
        return empty
    return "\n".join(str(value) for value in values)
