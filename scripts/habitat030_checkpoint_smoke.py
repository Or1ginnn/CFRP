"""Verify deterministic CFRP branch restores in a static Habitat-Sim scene."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import habitat_sim
import numpy as np
import quaternion as np_quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlnce_server.cfrp import (
    CFRPController,
    HabitatActionAdapter,
    capture_cfrp_checkpoint,
    parse_cfrp_output,
    restore_cfrp_checkpoint,
)


PROJECT = Path(os.environ.get("CFRP_PROJECT", "/mnt/data1/zar/plan_vln"))
DATA = Path(os.environ.get("HABITAT030_DATA", PROJECT / "data" / "habitat030"))
SCENE = DATA / "scene_datasets" / "habitat-test-scenes" / "skokloster-castle.glb"
ALLOWED_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
INIT_XML = """
<plan><global>test branch restore</global><local>
  <p id="p1" status="current">navigate through the room</p>
</local></plan>
<tool>continue</tool><subgoal>navigate through the room</subgoal><action>MOVE_FORWARD</action>
"""


def build_simulator() -> habitat_sim.Simulator:
    if not SCENE.exists():
        raise FileNotFoundError(SCENE)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(SCENE)
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_spec.resolution = [128, 128]
    sensor_spec.position = [0.0, 1.5, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def pose(simulator: habitat_sim.Simulator) -> tuple[np.ndarray, np.ndarray]:
    state = simulator.get_agent(0).get_state()
    return np.asarray(state.position).copy(), np_quaternion.as_float_array(state.rotation).copy()


def main() -> int:
    simulator = build_simulator()
    try:
        simulator.initialize_agent(0)
        simulator.reset()
        controller = CFRPController(allowed_actions=ALLOWED_ACTIONS)
        controller.step(parse_cfrp_output(INIT_XML))
        adapter = HabitatActionAdapter(
            simulator_actions=simulator.get_agent(0).agent_config.action_space.keys()
        )
        checkpoint = capture_cfrp_checkpoint(
            simulator,
            controller,
            recent_observation_history=["t0"],
            recent_action_history=controller.action_history,
            turn_index=1,
            cooldown_steps=2,
            episode_id="habitat-test-scene",
        )

        move_command = adapter.for_simulator("MOVE_FORWARD")
        move_observation = simulator.step(move_command.habitat_action)
        move_pose = pose(simulator)

        restored = restore_cfrp_checkpoint(
            simulator, controller, checkpoint, current_episode_id="habitat-test-scene"
        )
        if restored.cooldown_steps != 2:
            raise RuntimeError("checkpoint did not restore CFRP cooldown state")
        turn_command = adapter.for_simulator("TURN_LEFT")
        simulator.step(turn_command.habitat_action)
        turn_pose = pose(simulator)
        if np.allclose(move_pose[0], turn_pose[0]) and np.allclose(move_pose[1], turn_pose[1]):
            raise RuntimeError("counterfactual branches unexpectedly produced the same pose")

        restore_cfrp_checkpoint(
            simulator, controller, checkpoint, current_episode_id="habitat-test-scene"
        )
        replay_observation = simulator.step(move_command.habitat_action)
        replay_pose = pose(simulator)
        if not np.array_equal(move_observation["color_sensor"], replay_observation["color_sensor"]):
            raise RuntimeError("restored MOVE_FORWARD observation differs from the original branch")
        if not (np.allclose(move_pose[0], replay_pose[0]) and np.allclose(move_pose[1], replay_pose[1])):
            raise RuntimeError("restored MOVE_FORWARD pose differs from the original branch")

        print("branch_a=MOVE_FORWARD branch_b=TURN_LEFT diverged=OK")
        print("restore_then_replay=MOVE_FORWARD observation_match=OK pose_match=OK")
        print("control_memory_restore=cooldown_match=OK")
        print("cfrp_checkpoint_smoke: OK")
        return 0
    finally:
        simulator.close()


if __name__ == "__main__":
    raise SystemExit(main())
