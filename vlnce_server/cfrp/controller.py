"""Thin CFRP controller for maintaining persistent plan state."""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import CFRPOutput, CFRPProtocolError, PlanState, apply_plan_update, validate_output


@dataclass(frozen=True)
class ControllerStepResult:
    action: str
    actions: tuple[str, ...]
    current_plan: PlanState | None
    tool: str | None
    progress: str | None
    subgoal: str


@dataclass
class CFRPController:
    allowed_actions: set[str]
    current_plan: PlanState | None = None
    action_history: list[str] = field(default_factory=list)
    mode: str = "stage2"

    def step(self, output: CFRPOutput) -> ControllerStepResult:
        validate_output(
            output,
            allowed_actions=self.allowed_actions,
            previous_plan=self.current_plan,
            mode=self.mode,
        )

        if self.mode == "stage1":
            assert self.current_plan is not None
            if output.progress == "advance":
                self.current_plan = self.current_plan.advance_current()
        elif output.tool == "continue":
            if self.current_plan is None and output.plan is not None:
                self.current_plan = output.plan
            elif self.current_plan is None:
                raise CFRPProtocolError("continue requires an existing current plan")
        elif output.tool == "replan":
            if output.plan is not None:
                self.current_plan = output.plan
            else:
                assert self.current_plan is not None
                assert output.plan_update is not None
                self.current_plan = apply_plan_update(self.current_plan, output.plan_update)
        else:
            raise CFRPProtocolError(f"invalid tool: {output.tool}")

        actions = output.actions or (output.action,)
        self.action_history.extend(actions)
        return ControllerStepResult(
            action=output.action,
            actions=actions,
            current_plan=self.current_plan,
            tool=output.tool,
            progress=output.progress,
            subgoal=output.subgoal,
        )
