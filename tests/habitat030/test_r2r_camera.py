from vlnce_server.habitat030.r2r_environment import (
    janus_r2r_simulator_contract,
    r2r_camera_overrides,
    r2r_simulator_overrides,
)


def test_r2r_camera_uses_janus_resolution_and_hfov():
    assert r2r_camera_overrides() == [
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=640",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=480",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=79",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=640",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=480",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=79",
    ]


def test_r2r_simulator_uses_janus_action_geometry():
    assert r2r_simulator_overrides()[-2:] == [
        "habitat.simulator.forward_step_size=0.25",
        "habitat.simulator.turn_angle=15",
    ]
    assert janus_r2r_simulator_contract() == {
        "source": "JanusVLN/config/vln_r2r.yaml",
        "rgb_size": [640, 480],
        "depth_size": [640, 480],
        "hfov": 79,
        "forward_step_size": 0.25,
        "turn_angle": 15,
        "max_episode_steps": 500,
        "success_distance": 3.0,
    }
