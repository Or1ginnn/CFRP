"""Export one oracle-free R2R Habitat RGB observation for a model-process smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030 import Habitat030NavigationEnvironment
from vlnce_server.habitat030.r2r_environment import create_r2r_habitat_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("The Habitat environment must provide numpy") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        forbidden = ("pose", "goal_positions", "distance_to_goal", "reference_path", "expert_path")
        leaked = [name for name in forbidden if hasattr(observation, name)]
        if leaked:
            raise RuntimeError(f"navigation observation leaked privileged fields: {leaked}")

        rgb_path = output_dir / "r2r_stage1_rgb.npy"
        manifest_path = output_dir / "r2r_stage1_manifest.json"
        np.save(rgb_path, observation.rgb)
        manifest_path.write_text(
            json.dumps(
                {
                    "episode_id": observation.episode_id,
                    "instruction": observation.instruction,
                    "rgb_npy": str(rgb_path),
                    "allowed_actions": list(observation.allowed_actions),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"episode_id={record.episode_id}")
        print(f"rgb_npy={rgb_path}")
        print(f"manifest={manifest_path}")
        print("r2r_rgb_oracle_free=OK")
        print("habitat030_export_r2r_rgb: OK")
        return 0
    finally:
        wrapper.close()


if __name__ == "__main__":
    raise SystemExit(main())
