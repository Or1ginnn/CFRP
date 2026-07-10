"""Run one CFRP XML action through the controller and Habitat-Lab 0.3.0."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import habitat
from habitat.config.default import get_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlnce_server.cfrp import CFRPController, HabitatActionAdapter, parse_cfrp_output


PROJECT = Path(os.environ.get("CFRP_PROJECT", "/mnt/data1/zar/plan_vln"))
DATA = Path(os.environ.get("HABITAT030_DATA", PROJECT / "data" / "habitat030"))
CONFIG = PROJECT / "third_party" / "habitat-lab-0.3.0" / "habitat-lab" / "habitat" / "config" / "benchmark" / "nav" / "pointnav" / "pointnav_habitat_test.yaml"
DATASET = DATA / "datasets" / "pointnav" / "habitat-test-scenes" / "v1" / "train" / "train.json.gz"
SCENES_DIR = DATA / "scene_datasets"

ALLOWED_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
INITIAL_XML = """
<plan>
  <global>reach the target</global>
  <local><p id="p1" status="current">move forward</p></local>
</plan>
<tool>continue</tool>
<subgoal>move forward toward the target</subgoal>
<action>MOVE_FORWARD</action>
"""
STOP_XML = """
<tool>continue</tool>
<subgoal>stop at the current location</subgoal>
<action>STOP</action>
"""


def main() -> int:
    for path in (CONFIG, DATASET, SCENES_DIR):
        if not path.exists():
            raise FileNotFoundError(path)
    os.chdir(PROJECT)
    config = get_config(
        str(CONFIG),
        overrides=[
            f"habitat.dataset.data_path={DATASET}",
            f"habitat.dataset.scenes_dir={SCENES_DIR}",
            "habitat.dataset.split=train",
            "habitat.environment.iterator_options.shuffle=False",
        ],
    )
    env = habitat.Env(config=config)
    try:
        env.reset()
        task_actions = env.action_space.spaces.keys()
        controller = CFRPController(allowed_actions=ALLOWED_ACTIONS)
        adapter = HabitatActionAdapter(task_actions=task_actions)

        move_result = controller.step(parse_cfrp_output(INITIAL_XML))
        move_command = adapter.controller_result_for_task(move_result)
        env.step(move_command.habitat_action)
        print(f"xml_action={move_result.action} habitat_action={move_command.habitat_action} executed=OK")

        stop_result = controller.step(parse_cfrp_output(STOP_XML))
        stop_command = adapter.controller_result_for_task(stop_result)
        env.step(stop_command.habitat_action)
        print(
            f"xml_action={stop_result.action} habitat_action={stop_command.habitat_action} "
            f"episode_over={env.episode_over}"
        )
        if not env.episode_over:
            raise RuntimeError("CFRP STOP did not end the Habitat-Lab episode")
        print("cfrp_habitat_adapter_smoke: OK")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
