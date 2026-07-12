"""Factories for real R2R-CE Habitat 0.3 smoke environments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .r2r_dataset import R2REpisodeRecord, load_r2r_episode, make_habitat_dataset


def create_r2r_habitat_env(
    config_path: str,
    dataset_root: str,
    scenes_dir: str,
    split: str = "val_seen",
    episode_id: Optional[str] = None,
    seed: Optional[int] = None,
    success_distance: float = 3.0,
) -> Tuple[object, R2REpisodeRecord]:
    import habitat
    from habitat.config.default import get_config

    record = load_r2r_episode(
        dataset_root=dataset_root,
        split=split,
        episode_id=episode_id,
        scenes_dir=scenes_dir,
    )
    dataset = make_habitat_dataset((record,))
    overrides = [
        f"habitat.dataset.data_path={Path(dataset_root) / split / (split + '.json.gz')}",
        f"habitat.dataset.scenes_dir={scenes_dir}",
        f"habitat.dataset.split={split}",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=128",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=128",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=128",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=128",
    ]
    if seed is not None:
        overrides.append(f"habitat.seed={seed}")
    if success_distance <= 0:
        raise ValueError("success_distance must be positive")
    # R2R-CE/RxR-CE use a 3m navigation success threshold.  The PointNav
    # smoke config otherwise carries its own, different default.
    overrides.append(f"habitat.task.measurements.success.success_distance={success_distance}")
    config = get_config(
        str(config_path),
        overrides=overrides,
    )
    return habitat.Env(config=config, dataset=dataset), record
