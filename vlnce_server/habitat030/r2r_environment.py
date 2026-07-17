"""Factories for real R2R-CE Habitat 0.3 smoke environments."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from vlnce_server.qwen3vl.vision import HABITAT_RGB_HEIGHT, HABITAT_RGB_HFOV, HABITAT_RGB_WIDTH

from .r2r_dataset import R2REpisodeRecord, load_r2r_episode, make_habitat_dataset


R2R_MAX_EPISODE_STEPS = 500
R2R_FORWARD_STEP_SIZE = 0.25
R2R_TURN_ANGLE = 15
R2R_SUCCESS_DISTANCE = 3.0
JANUS_INTERMEDIATE_WAYPOINT_RADIUS = 1.8
JANUS_FINAL_WAYPOINT_RADIUS = 0.25


def r2r_camera_overrides() -> List[str]:
    """Return JanusVLN's high-resolution R2R camera configuration."""

    return [
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width={}".format(HABITAT_RGB_WIDTH),
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height={}".format(HABITAT_RGB_HEIGHT),
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov={}".format(HABITAT_RGB_HFOV),
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width={}".format(HABITAT_RGB_WIDTH),
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height={}".format(HABITAT_RGB_HEIGHT),
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov={}".format(HABITAT_RGB_HFOV),
    ]


def r2r_simulator_overrides() -> List[str]:
    """Return the complete JanusVLN simulator contract used by train and eval."""

    return r2r_camera_overrides() + [
        f"habitat.simulator.forward_step_size={R2R_FORWARD_STEP_SIZE}",
        f"habitat.simulator.turn_angle={R2R_TURN_ANGLE}",
    ]


def janus_r2r_simulator_contract() -> dict[str, object]:
    """Return the machine-readable contract embedded in CFRP data artifacts."""

    return {
        "source": "JanusVLN/config/vln_r2r.yaml",
        "rgb_size": [HABITAT_RGB_WIDTH, HABITAT_RGB_HEIGHT],
        "depth_size": [HABITAT_RGB_WIDTH, HABITAT_RGB_HEIGHT],
        "hfov": HABITAT_RGB_HFOV,
        "forward_step_size": R2R_FORWARD_STEP_SIZE,
        "turn_angle": R2R_TURN_ANGLE,
        "max_episode_steps": R2R_MAX_EPISODE_STEPS,
        "success_distance": R2R_SUCCESS_DISTANCE,
    }


def janus_r2r_oracle_contract() -> dict[str, object]:
    """Return JanusVLN's force-expert reference-path follower contract."""

    return {
        "route": "r2r_reference_path",
        "intermediate_waypoint_radius": JANUS_INTERMEDIATE_WAYPOINT_RADIUS,
        "final_waypoint_radius": JANUS_FINAL_WAYPOINT_RADIUS,
    }


def create_r2r_habitat_env(
    config_path: str,
    dataset_root: str,
    scenes_dir: str,
    split: str = "val_seen",
    episode_id: Optional[str] = None,
    seed: Optional[int] = None,
    success_distance: float = R2R_SUCCESS_DISTANCE,
    include_top_down_map: bool = False,
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
        f"habitat.environment.max_episode_steps={R2R_MAX_EPISODE_STEPS}",
    ]
    overrides.extend(r2r_simulator_overrides())
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
    if include_top_down_map:
        from habitat.config import read_write
        from habitat.config.default_structured_configs import (
            CollisionsMeasurementConfig,
            FogOfWarConfig,
            TopDownMapMeasurementConfig,
        )

        with read_write(config):
            config.habitat.task.measurements.top_down_map = TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
            )
            config.habitat.task.measurements.collisions = CollisionsMeasurementConfig()
    return habitat.Env(config=config, dataset=dataset), record
