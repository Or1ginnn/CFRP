from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import habitat
from habitat.config.default import get_config


PROJECT = Path(os.environ.get("CFRP_PROJECT", "/mnt/data1/zar/plan_vln"))
DATA = Path(os.environ.get("HABITAT030_DATA", PROJECT / "data" / "habitat030"))
CONFIG = PROJECT / "third_party" / "habitat-lab-0.3.0" / "habitat-lab" / "habitat" / "config" / "benchmark" / "nav" / "pointnav" / "pointnav_habitat_test.yaml"
DATASET = DATA / "datasets" / "pointnav" / "habitat-test-scenes" / "v1" / "train" / "train.json.gz"
SCENES_DIR = DATA / "scene_datasets"
SCENE = SCENES_DIR / "habitat-test-scenes" / "skokloster-castle.glb"


def metric(metrics: Dict[str, Any], name: str) -> Any:
    value = metrics.get(name, None)
    if hasattr(value, "item"):
        return value.item()
    return value


def print_metrics(prefix: str, env: habitat.Env) -> None:
    metrics = env.get_metrics()
    print(
        f"{prefix}: episode_over={env.episode_over} "
        f"distance_to_goal={metric(metrics, 'distance_to_goal')} "
        f"success={metric(metrics, 'success')} spl={metric(metrics, 'spl')}"
    )


def describe_action_space(action_space: Any) -> None:
    print("action_space_repr:", action_space)
    spaces = getattr(action_space, "spaces", None)
    if spaces is not None:
        print("action_space_keys:", sorted(spaces.keys()))
        for name, space in sorted(spaces.items()):
            print(f"action_space_item: {name} -> {space}")
    else:
        print("action_space_type:", type(action_space))


def main() -> int:
    if not CONFIG.exists():
        raise FileNotFoundError(f"PointNav config not found: {CONFIG}")
    if not DATASET.exists():
        raise FileNotFoundError(f"PointNav dataset not found: {DATASET}")
    if not SCENE.exists():
        raise FileNotFoundError(f"Habitat test scene not found: {SCENE}")

    os.chdir(PROJECT)

    print("python:", sys.version.replace("\n", " "))
    print("habitat:", getattr(habitat, "__version__", "unknown"))
    print("config:", CONFIG)
    print("dataset:", DATASET)
    print("scenes_dir:", SCENES_DIR)
    print("scene:", SCENE)
    print("cwd:", Path.cwd())
    print("EGL_PLATFORM:", os.environ.get("EGL_PLATFORM", ""))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    config = get_config(
        str(CONFIG),
        overrides=[
            f"habitat.dataset.data_path={DATASET}",
            "habitat.dataset.split=train",
            f"habitat.dataset.scenes_dir={SCENES_DIR}",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=128",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=128",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=128",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=128",
        ],
    )

    env = habitat.Env(config=config)
    try:
        expected_scene_id = str(SCENE)
        skokloster_episodes = [ep for ep in env.episodes if ep.scene_id == expected_scene_id]
        if not skokloster_episodes:
            scenes = sorted({ep.scene_id for ep in env.episodes})
            raise RuntimeError(f"No episode found for scene_id={expected_scene_id}; available={scenes[:10]}")
        env.episodes = [skokloster_episodes[0]]
        observations = env.reset()
        print("reset_observation_keys:", sorted(observations.keys()))
        print("episode_id:", env.current_episode.episode_id)
        print("episode_scene_id:", env.current_episode.scene_id)
        describe_action_space(env.action_space)
        print_metrics("before_stop", env)

        stop_payload = "stop"
        print("stop_payload:", stop_payload)
        print("stop_path: habitat.Env.step('stop') -> task StopAction; no habitat_sim physical action")
        observations = env.step(stop_payload)
        print("post_stop_observation_keys:", sorted(observations.keys()))
        print_metrics("after_stop", env)

        if not env.episode_over:
            raise RuntimeError("STOP did not end the Habitat-Lab episode")
        metrics = env.get_metrics()
        for required in ("distance_to_goal", "success", "spl"):
            if required not in metrics:
                raise RuntimeError(f"Missing expected task metric after STOP: {required}")
        print("stop_task_smoke: OK")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
