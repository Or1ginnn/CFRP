from __future__ import annotations

import pytest

from vlnce_server.habitat030.oracle_actions import OracleActionError, cfrp_action_from_habitat_oracle


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
