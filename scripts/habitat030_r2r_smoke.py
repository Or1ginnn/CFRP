from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030 import Habitat030NavigationEnvironment, NavigationObservation
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real R2R-CE Habitat 0.3 smoke test.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def rgb_shape(rgb: Any) -> Any:
    return getattr(rgb, "shape", None)


def assert_oracle_free(observation: NavigationObservation) -> None:
    forbidden = (
        "pose",
        "agent_position",
        "agent_rotation",
        "goal_positions",
        "distance_to_goal",
        "reference_path",
        "expert_path",
    )
    leaked = [name for name in forbidden if hasattr(observation, name)]
    if leaked:
        raise RuntimeError(f"NavigationObservation leaked privileged fields: {leaked}")


def main() -> int:
    args = parse_args()
    env, record = create_r2r_habitat_env(
        config_path=args.config,
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
        split=args.split,
        episode_id=args.episode_id,
    )
    wrapper = Habitat030NavigationEnvironment(env)
    try:
        observation = wrapper.reset()
        assert_oracle_free(observation)
        print(f"r2r_episode_id={record.episode_id}")
        print(f"r2r_scene_path={record.scene_path}")
        print(f"r2r_instruction_present={bool(observation.instruction)}")
        print(f"instruction={observation.instruction}")
        print(f"rgb_shape={rgb_shape(observation.rgb)}")

        for action in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            step = wrapper.step(action)
            print(f"action={action} habitat_action={step.habitat_action} executed=OK")

        stop_step = wrapper.step("STOP")
        metrics = stop_step.metrics
        print(
            f"action=STOP habitat_action={stop_step.habitat_action} "
            f"episode_over={stop_step.episode_over}"
        )
        print(f"distance_to_goal={metrics.distance_to_goal}")
        print(f"success={metrics.success}")
        print(f"spl={metrics.spl}")
        print(f"path_length={metrics.path_length}")
        print("navigation_observation_oracle_free=OK")
        print("habitat030_r2r_smoke: OK")
        return 0
    finally:
        wrapper.close()


if __name__ == "__main__":
    raise SystemExit(main())
