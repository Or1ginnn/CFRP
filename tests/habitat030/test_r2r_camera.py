from vlnce_server.habitat030.r2r_environment import r2r_camera_overrides


def test_r2r_camera_uses_activevln_resolution_and_hfov():
    assert r2r_camera_overrides() == [
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=640",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=480",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=640",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=480",
    ]
