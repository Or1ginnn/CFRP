#!/usr/bin/env python
"""Run a Habitat-free CFRP mock navigation loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlnce_server.cfrp import run_scripted_cfrp_loop


ALLOWED_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")

OBSERVATIONS = (
    "rgb=t0: agent is in a bedroom facing an open doorway.",
    "rgb=t1: agent has entered a hallway; a side room is visible on the left.",
    "rgb=t2: agent drifted into the side room and must recover to the hallway.",
    "rgb=t3: agent is back in the hallway near the kitchen entrance.",
    "rgb=t4: agent is facing the kitchen sink at close range.",
)

MODEL_OUTPUTS = (
    """
<plan>
  <global>bedroom -> hallway -> kitchen sink</global>
  <local>
    <p id="p1" status="current">exit the bedroom through the doorway</p>
    <p id="p2" status="todo">follow the hallway toward the kitchen</p>
    <p id="p3" status="todo">stop near the kitchen sink</p>
  </local>
</plan>
<tool>continue</tool>
<subgoal>exit the bedroom through the doorway</subgoal>
<action>MOVE_FORWARD</action>
""",
    """
<tool>continue</tool>
<subgoal>continue along the hallway toward the kitchen</subgoal>
<action>MOVE_FORWARD</action>
""",
    """
<plan>
  <global>bedroom -> hallway -> kitchen sink</global>
  <local>
    <p id="p1" status="done">exit the bedroom through the doorway</p>
    <p id="p2" status="abandoned">follow the hallway without entering side rooms</p>
    <p id="r1" status="current">turn left and return from the side room to the hallway</p>
    <p id="p3" status="todo">continue toward the kitchen sink</p>
  </local>
</plan>
<tool>replan</tool>
<subgoal>return from the side room to the hallway</subgoal>
<action>TURN_LEFT</action>
""",
    """
<tool>continue</tool>
<subgoal>move toward the kitchen sink</subgoal>
<action>MOVE_FORWARD</action>
""",
    """
<tool>stop</tool>
<subgoal>stop near the kitchen sink</subgoal>
<action>STOP</action>
""",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the turn trace as JSON")
    args = parser.parse_args()

    turns = run_scripted_cfrp_loop(
        full_instruction="Exit the bedroom, follow the hallway, and stop near the kitchen sink.",
        observations=OBSERVATIONS,
        model_outputs=MODEL_OUTPUTS,
        allowed_actions=ALLOWED_ACTIONS,
        active_instruction_excerpt="stop near the kitchen sink",
    )

    if args.json:
        print(json.dumps([turn.__dict__ for turn in turns], indent=2))
        return

    for turn in turns:
        print(
            f"turn={turn.turn_index} tool={turn.tool} "
            f"action={turn.action} subgoal={turn.subgoal}"
        )


if __name__ == "__main__":
    main()
