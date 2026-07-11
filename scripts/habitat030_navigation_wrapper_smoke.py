from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import habitat
from habitat.config.default import get_config

from vlnce_server.habitat030 import Habitat030NavigationEnvironment, NavigationObservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Habitat 0.3 CFRP navigation wrapper.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--fallback-instruction", default="")
    return parser.parse_args()


def metric_keys(env: habitat.Env) -> list[str]:
    return sorted((env.get_metrics() or {}).keys())


def first_existing_scene_episode(env: habitat.Env, scenes_dir: Path):
    for episode in env.episodes:
        if Path(episode.scene_id).exists():
            return episode
    available = sorted({episode.scene_id for episode in env.episodes})[:10]
    raise RuntimeError(f"No episode scene exists under {scenes_dir}; available={available}")


def rgb_shape(rgb: Any) -> Any:
    return getattr(rgb, "shape", None)


def assert_oracle_free(observation: NavigationObservation) -> None:
    forbidden = ("agent_position", "agent_rotation", "goal_positions", "distance_to_goal", "expert_path")
    leaked = [name for name in forbidden if hasattr(observation, name)]
    if leaked:
        raise RuntimeError(f"NavigationObservation leaked privileged fields: {leaked}")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    dataset_path = Path(args.dataset)
    scenes_dir = Path(args.scenes_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if not scenes_dir.exists():
        raise FileNotFoundError(f"scenes-dir not found: {scenes_dir}")

    config = get_config(
        str(config_path),
        overrides=[
            f"habitat.dataset.data_path={dataset_path}",
            "habitat.dataset.split=train",
            f"habitat.dataset.scenes_dir={scenes_dir}",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=128",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=128",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=128",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=128",
        ],
    )

    env = habitat.Env(config=config)
    try:
        env.episodes = [first_existing_scene_episode(env, scenes_dir)]
        wrapper = Habitat030NavigationEnvironment(env, fallback_instruction=args.fallback_instruction)
        observation = wrapper.reset()
        assert_oracle_free(observation)
        print(f"episode_id={observation.episode_id}")
        print(f"rgb_shape={rgb_shape(observation.rgb)}")
        print(f"instruction={observation.instruction}")
        print(f"allowed_actions={observation.allowed_actions}")

        for action in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            step = wrapper.step(action)
            print(f"action={action} habitat_action={step.habitat_action} executed=OK")

        stop_step = wrapper.step("STOP")
        print(
            f"action=STOP habitat_action={stop_step.habitat_action} "
            f"episode_over={stop_step.episode_over}"
        )
        print(f"metrics_keys={metric_keys(env)}")
        print("navigation_observation_oracle_free=OK")
        print("habitat030_navigation_wrapper_smoke: OK")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
