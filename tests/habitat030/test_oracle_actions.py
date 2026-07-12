from __future__ import annotations

import pytest

from vlnce_server.habitat030.oracle_actions import (
    OracleActionError,
    _cfrp_action_from_registry_id,
    cfrp_action_from_habitat_oracle,
)


class FakeAction:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.parametrize("name", ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"))
def test_maps_named_habitat_oracle_actions(name: str):
    assert cfrp_action_from_habitat_oracle(FakeAction(name)) == name


def test_rejects_oracle_action_outside_cfrp_space():
    with pytest.raises(OracleActionError, match="not a CFRP primitive"):
        cfrp_action_from_habitat_oracle(FakeAction("LOOK_UP"))


def test_rejects_missing_oracle_action():
    with pytest.raises(OracleActionError, match="no action"):
        cfrp_action_from_habitat_oracle(None)


class FakeHabitatRegistry:
    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3


@pytest.mark.parametrize(
    ("action_id", "expected"),
    ((0, "STOP"), (1, "MOVE_FORWARD"), (2, "TURN_LEFT"), (3, "TURN_RIGHT")),
)
def test_maps_habitat_singleton_ids_by_registered_action_names(action_id: int, expected: str):
    assert _cfrp_action_from_registry_id(action_id, FakeHabitatRegistry()) == expected


class KeyErrorRegistry(FakeHabitatRegistry):
    def __getattr__(self, name: str):
        raise KeyError(name)


def test_maps_registry_ids_when_missing_aliases_raise_key_error():
    assert _cfrp_action_from_registry_id(1, KeyErrorRegistry()) == "MOVE_FORWARD"
