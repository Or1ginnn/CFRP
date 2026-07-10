"""Translate CFRP primitive actions into Habitat execution commands.

The adapter deliberately does not import Habitat. This keeps the CFRP control
contract testable on CPU while making the simulator/task distinction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .controller import ControllerStepResult


CFRP_PRIMITIVE_ACTIONS = frozenset({"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"})
_HABITAT_ACTIONS = {
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "STOP": "stop",
}


class CFRPActionAdapterError(ValueError):
    """Raised when a CFRP action cannot be executed by the target environment."""


@dataclass(frozen=True)
class HabitatActionCommand:
    """An action ready for either Habitat-Sim or Habitat-Lab execution."""

    primitive_action: str
    habitat_action: str | None
    target: Literal["simulator", "task"]
    terminate_episode: bool


class HabitatActionAdapter:
    """Map CFRP actions to the action names exposed by Habitat 0.3 navigation."""

    def __init__(
        self,
        *,
        simulator_actions: Iterable[str] = ("move_forward", "turn_left", "turn_right"),
        task_actions: Iterable[str] = ("move_forward", "turn_left", "turn_right", "stop"),
    ) -> None:
        self._simulator_actions = frozenset(simulator_actions)
        self._task_actions = frozenset(task_actions)

    def for_simulator(self, primitive_action: str) -> HabitatActionCommand:
        """Create a command for ``habitat_sim.Simulator.step``.

        ``STOP`` terminates the controller episode and intentionally has no
        simulator movement action.
        """

        habitat_action = self._habitat_action_for(primitive_action)
        if primitive_action == "STOP":
            return HabitatActionCommand(
                primitive_action=primitive_action,
                habitat_action=None,
                target="simulator",
                terminate_episode=True,
            )
        self._require_available(habitat_action, self._simulator_actions, "simulator")
        return HabitatActionCommand(
            primitive_action=primitive_action,
            habitat_action=habitat_action,
            target="simulator",
            terminate_episode=False,
        )

    def for_task(self, primitive_action: str) -> HabitatActionCommand:
        """Create a command for ``habitat.Env.step`` and its task action layer."""

        habitat_action = self._habitat_action_for(primitive_action)
        self._require_available(habitat_action, self._task_actions, "task")
        return HabitatActionCommand(
            primitive_action=primitive_action,
            habitat_action=habitat_action,
            target="task",
            terminate_episode=primitive_action == "STOP",
        )

    def controller_result_for_simulator(self, result: ControllerStepResult) -> HabitatActionCommand:
        return self.for_simulator(result.action)

    def controller_result_for_task(self, result: ControllerStepResult) -> HabitatActionCommand:
        return self.for_task(result.action)

    @staticmethod
    def _habitat_action_for(primitive_action: str) -> str:
        if primitive_action not in CFRP_PRIMITIVE_ACTIONS:
            raise CFRPActionAdapterError(f"unsupported CFRP primitive action: {primitive_action}")
        return _HABITAT_ACTIONS[primitive_action]

    @staticmethod
    def _require_available(action: str, available_actions: frozenset[str], target: str) -> None:
        if action not in available_actions:
            available = ", ".join(sorted(available_actions))
            raise CFRPActionAdapterError(
                f"Habitat {target} action unavailable: {action}; available actions: {available}"
            )
