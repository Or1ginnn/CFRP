from __future__ import annotations

from dataclasses import dataclass

import pytest

from vlnce_server.habitat030 import Habitat030NavigationEnvironment, NavigationObservation


@dataclass
class FakeInstruction:
    instruction_text: str


@dataclass
class FakeGoal:
    position: tuple[float, ...]


@dataclass
class FakeEpisode:
    episode_id: str = "episode-1"
    instruction: object = None
    goals: tuple[FakeGoal, ...] = (FakeGoal((3.0, 0.0, 4.0)),)
    expert_path: tuple[tuple[float, ...], ...] = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))


@dataclass
class FakeAgentState:
    position: tuple[float, ...] = (1.0, 2.0, 3.0)
    rotation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)


class FakeSimulator:
    def get_agent_state(self):
        return FakeAgentState()


class StrictFakeEnv:
    def __init__(self, *, reset_observation=None, step_observation=None, metrics=None, episode=None):
        self.current_episode = episode or FakeEpisode(instruction=FakeInstruction("episode instruction"))
        self.sim = FakeSimulator()
        self.episode_over = False
        self.closed = False
        self.step_calls = []
        self.reset_observation = reset_observation or {
            "rgb": "rgb-frame",
            "instruction": {"text": "dict instruction"},
        }
        self.step_observation = step_observation or {
            "rgb": "next-rgb",
            "instruction": "step instruction",
        }
        self.metric_values = metrics or {
            "distance_to_goal": 2.5,
            "success": 0.0,
            "spl": 0.0,
            "path_length": 1.25,
            "ndtw": 0.75,
            "not_numeric": "skip",
        }

    def __getattribute__(self, name):
        if name.startswith("_"):
            raise AssertionError(f"private Habitat field accessed: {name}")
        return object.__getattribute__(self, name)

    def reset(self):
        return self.reset_observation

    def step(self, action):
        self.step_calls.append(action)
        if action == "stop":
            self.episode_over = True
        return self.step_observation

    def get_metrics(self):
        return self.metric_values

    def close(self):
        self.closed = True


def test_reset_returns_model_visible_fields_only():
    wrapper = Habitat030NavigationEnvironment(StrictFakeEnv())

    observation = wrapper.reset()

    assert observation == NavigationObservation(
        episode_id="episode-1",
        instruction="dict instruction",
        rgb="rgb-frame",
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )
    assert not hasattr(observation, "agent_position")
    assert not hasattr(observation, "goal_positions")
    assert not hasattr(observation, "distance_to_goal")
    assert not hasattr(observation, "expert_path")


def test_action_mapping_for_movement_primitives():
    env = StrictFakeEnv()
    wrapper = Habitat030NavigationEnvironment(env)
    wrapper.reset()

    forward = wrapper.step("MOVE_FORWARD")
    left = wrapper.step("TURN_LEFT")
    right = wrapper.step("TURN_RIGHT")

    assert env.step_calls == ["move_forward", "turn_left", "turn_right"]
    assert forward.habitat_action == "move_forward"
    assert left.habitat_action == "turn_left"
    assert right.habitat_action == "turn_right"


def test_stop_calls_task_stop_and_marks_episode_over():
    env = StrictFakeEnv()
    wrapper = Habitat030NavigationEnvironment(env)
    wrapper.reset()

    step = wrapper.step("STOP")

    assert env.step_calls == ["stop"]
    assert step.action == "STOP"
    assert step.habitat_action == "stop"
    assert step.episode_over is True


def test_metrics_extract_standard_and_extra_numeric_values():
    wrapper = Habitat030NavigationEnvironment(StrictFakeEnv())

    metrics = wrapper.metrics()

    assert metrics.distance_to_goal == 2.5
    assert metrics.success == 0.0
    assert metrics.spl == 0.0
    assert metrics.path_length == 1.25
    assert metrics.extra == (("ndtw", 0.75),)


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ({"text": "dict text"}, "dict text"),
        ({"instruction_text": "dict instruction"}, "dict instruction"),
        (type("InstructionObject", (), {"text": "object text"})(), "object text"),
        ("string text", "string text"),
    ],
)
def test_instruction_formats_from_observation(instruction, expected):
    env = StrictFakeEnv(reset_observation={"rgb": "rgb", "instruction": instruction})
    wrapper = Habitat030NavigationEnvironment(env)

    assert wrapper.reset().instruction == expected


def test_instruction_falls_back_to_episode_then_configured_text():
    episode_env = StrictFakeEnv(reset_observation={"rgb": "rgb"})
    assert Habitat030NavigationEnvironment(episode_env).reset().instruction == "episode instruction"

    no_episode_instruction = FakeEpisode(instruction=None)
    fallback_env = StrictFakeEnv(reset_observation={"rgb": "rgb"}, episode=no_episode_instruction)
    wrapper = Habitat030NavigationEnvironment(fallback_env, fallback_instruction="fallback text")
    assert wrapper.reset().instruction == "fallback text"


def test_privileged_state_contains_pose_goals_and_expert_path():
    wrapper = Habitat030NavigationEnvironment(StrictFakeEnv())

    state = wrapper.privileged_state()

    assert state.episode_id == "episode-1"
    assert state.agent_position == (1.0, 2.0, 3.0)
    assert state.agent_rotation == (1.0, 0.0, 0.0, 0.0)
    assert state.goal_positions == ((3.0, 0.0, 4.0),)
    assert state.expert_path == ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))


def test_agent_pose_returns_public_simulator_state():
    wrapper = Habitat030NavigationEnvironment(StrictFakeEnv())

    assert wrapper.agent_pose() == ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))


def test_invalid_action_is_rejected():
    wrapper = Habitat030NavigationEnvironment(StrictFakeEnv())

    with pytest.raises(ValueError):
        wrapper.step("JUMP")


def test_close_forwards_to_env_close():
    env = StrictFakeEnv()
    wrapper = Habitat030NavigationEnvironment(env)

    wrapper.close()

    assert env.closed is True


@dataclass
class MovingFakeAgentState:
    position: tuple[float, ...]
    rotation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)


class MovingFakeSimulator:
    def __init__(self):
        self.position = (0.0, 0.0, 0.0)

    def get_agent_state(self):
        return MovingFakeAgentState(self.position)


class PathLengthFakeEnv(StrictFakeEnv):
    def __init__(self):
        super().__init__(metrics={"distance_to_goal": 10.0, "success": 0.0, "spl": 0.0})
        self.sim = MovingFakeSimulator()

    def reset(self):
        self.sim.position = (0.0, 0.0, 0.0)
        return self.reset_observation

    def step(self, action):
        self.step_calls.append(action)
        if action == "move_forward":
            x, y, z = self.sim.position
            self.sim.position = (x + 3.0, y, z + 4.0)
        if action == "stop":
            self.episode_over = True
        return self.step_observation


def test_path_length_fallback_accumulates_forward_motion_only():
    env = PathLengthFakeEnv()
    wrapper = Habitat030NavigationEnvironment(env)
    wrapper.reset()

    forward = wrapper.step("MOVE_FORWARD")
    left = wrapper.step("TURN_LEFT")
    right = wrapper.step("TURN_RIGHT")

    assert forward.metrics.path_length == 5.0
    assert left.metrics.path_length == 5.0
    assert right.metrics.path_length == 5.0


def test_path_length_fallback_reset_clears_accumulator():
    env = PathLengthFakeEnv()
    wrapper = Habitat030NavigationEnvironment(env)
    wrapper.reset()
    wrapper.step("MOVE_FORWARD")

    observation = wrapper.reset()

    assert observation.episode_id == "episode-1"
    assert wrapper.metrics().path_length == 0.0