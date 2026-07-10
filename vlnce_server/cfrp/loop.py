"""Habitat-free CFRP loop runner.

The runner is deliberately small: it connects prompt construction, model XML
output parsing, protocol validation, and persistent plan-state updates. Real
VLN simulation can later replace the scripted observations and model outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .controller import CFRPController
from .prompts import build_step_prompt
from .protocol import parse_cfrp_output


@dataclass(frozen=True)
class CFRPLoopTurn:
    turn_index: int
    prompt: str
    raw_output: str
    tool: str
    subgoal: str
    action: str
    current_plan_xml: str | None
    is_stop: bool


def run_scripted_cfrp_loop(
    *,
    full_instruction: str,
    observations: Sequence[str],
    model_outputs: Sequence[str],
    allowed_actions: Sequence[str],
    active_instruction_excerpt: str | None = None,
    max_history: int = 3,
) -> list[CFRPLoopTurn]:
    """Run a deterministic CFRP loop with scripted model XML outputs.

    This is the Phase 1 mock path: no Habitat, no torch, and no VLM. It verifies
    the control contract we will later insert before ``env.step(action)``.
    """

    if not observations:
        raise ValueError("observations must contain at least one item")
    if len(model_outputs) < len(observations):
        raise ValueError("model_outputs must cover every observation or an earlier stop")
    if max_history < 0:
        raise ValueError("max_history must be non-negative")

    controller = CFRPController(allowed_actions=set(allowed_actions))
    turns: list[CFRPLoopTurn] = []
    recent_actions: list[str] = []
    recent_observations: list[str] = []

    for turn_index, observation in enumerate(observations):
        prompt = build_step_prompt(
            full_instruction=full_instruction,
            allowed_actions=allowed_actions,
            current_observation=observation,
            recent_visual_history=recent_observations[-max_history:] if max_history else (),
            recent_actions=recent_actions[-max_history:] if max_history else (),
            current_plan=controller.current_plan,
            active_instruction_excerpt=active_instruction_excerpt,
        )
        raw_output = model_outputs[turn_index]
        result = controller.step(parse_cfrp_output(raw_output))
        current_plan_xml = result.current_plan.to_xml() if result.current_plan else None
        is_stop = result.action == "STOP"

        turns.append(
            CFRPLoopTurn(
                turn_index=turn_index,
                prompt=prompt,
                raw_output=raw_output.strip(),
                tool=result.tool,
                subgoal=result.subgoal,
                action=result.action,
                current_plan_xml=current_plan_xml,
                is_stop=is_stop,
            )
        )

        recent_actions.append(result.action)
        recent_observations.append(observation)
        if is_stop:
            break

    return turns
