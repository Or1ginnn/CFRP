"""Thin CFRP controller for maintaining persistent plan state."""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import CFRPOutput, CFRPProtocolError, PlanState, validate_output


@dataclass(frozen=True)
class ControllerStepResult:
    action: str
    current_plan: PlanState | None
    tool: str
    subgoal: str


@dataclass
class CFRPController:
    allowed_actions: set[str]
    current_plan: PlanState | None = None
    action_history: list[str] = field(default_factory=list)

    def step(self, output: CFRPOutput) -> ControllerStepResult:
        validate_output(
            output,
            allowed_actions=self.allowed_actions,
            previous_plan=self.current_plan,
        )

        if output.tool == "continue":
            if self.current_plan is None and output.plan is not None:
                self.current_plan = output.plan
            elif self.current_plan is None:
                raise CFRPProtocolError("continue requires an existing current plan")
        elif output.tool == "replan":
            self.current_plan = output.plan
        elif output.tool == "stop":
            pass
        else:
            raise CFRPProtocolError(f"invalid tool: {output.tool}")

        self.action_history.append(output.action)
        return ControllerStepResult(
            action=output.action,
            current_plan=self.current_plan,
            tool=output.tool,
            subgoal=output.subgoal,
        )
