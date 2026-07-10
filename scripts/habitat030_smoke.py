from __future__ import annotations

import os
import sys
from pathlib import Path

import habitat
import habitat_sim
import quaternion as np_quaternion
from habitat_sim.agent import AgentConfiguration


PROJECT = Path(os.environ.get("CFRP_PROJECT", "/mnt/data1/zar/plan_vln"))
DATA = Path(os.environ.get("HABITAT030_DATA", PROJECT / "data" / "habitat030"))
SCENE = DATA / "scene_datasets" / "habitat-test-scenes" / "skokloster-castle.glb"


def rotation_to_list(rotation):
    try:
        values = np_quaternion.as_float_array(rotation)
        if hasattr(values, "tolist"):
            return values.tolist()
    except Exception:
        pass
    if hasattr(rotation, "real") and hasattr(rotation, "imag"):
        imag = rotation.imag
        if hasattr(imag, "tolist"):
            imag_values = imag.tolist()
        else:
            imag_values = [getattr(imag, axis) for axis in ("x", "y", "z")]
        return [rotation.real, *imag_values]
    if hasattr(rotation, "tolist"):
        values = rotation.tolist()
        return values if isinstance(values, list) else [values]
    return [rotation]


def format_state(agent) -> str:
    state = agent.get_state()
    position = [float(x) for x in state.position]
    rotation = [float(x) for x in rotation_to_list(state.rotation)]
    return f"position={position} rotation=[w,x,y,z]={rotation}"


def summarize_observation(label: str, observations: dict, agent) -> None:
    if "color_sensor" not in observations:
        raise RuntimeError(f"color_sensor missing from observations: {sorted(observations.keys())}")
    rgb = observations["color_sensor"]
    print(
        f"{label}: color_sensor shape={tuple(rgb.shape)} dtype={rgb.dtype} "
        f"{format_state(agent)}"
    )


def build_simulator() -> habitat_sim.Simulator:
    if not SCENE.exists():
        raise FileNotFoundError(f"Habitat test scene not found: {SCENE}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(SCENE)
    sim_cfg.enable_physics = False

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_spec.resolution = [128, 128]
    sensor_spec.position = [0.0, 1.5, 0.0]

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    config = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(config)


def main() -> int:
    print("scene:", SCENE)
    print("python:", sys.version.replace("\n", " "))
    print("habitat:", getattr(habitat, "__version__", "unknown"))
    print("habitat_sim:", getattr(habitat_sim, "__version__", "unknown"))
    print("EGL_PLATFORM:", os.environ.get("EGL_PLATFORM", ""))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    sim = build_simulator()
    try:
        agent = sim.initialize_agent(0)
        action_names = sorted(agent.agent_config.action_space.keys())
        print("actual_actions:", action_names)
        for name in action_names:
            spec = agent.agent_config.action_space[name]
            print(f"action_spec: {name} amount={getattr(spec.actuation, 'amount', None)}")

        cfrp_action_map = {
            "MOVE_FORWARD": "move_forward",
            "TURN_LEFT": "turn_left",
            "TURN_RIGHT": "turn_right",
            "STOP": None,
        }
        print("cfrp_action_map:", cfrp_action_map)

        missing = [a for a in cfrp_action_map.values() if a is not None and a not in action_names]
        if missing:
            raise RuntimeError(f"Habitat action(s) missing for CFRP mapping: {missing}")

        observations = sim.reset()
        summarize_observation("reset", observations, agent)

        for primitive in ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]:
            habitat_action = cfrp_action_map[primitive]
            observations = sim.step(habitat_action)
            summarize_observation(f"{primitive}->{habitat_action}", observations, agent)
            print(f"executed: {primitive}=OK")

        print("STOP: controller terminates episode; no habitat_sim.step call")
        print("smoke_test: OK")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
