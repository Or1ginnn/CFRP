from __future__ import annotations

import json

import pytest

from vlnce_server.cfrp import (
    PlanPoint,
    PlanState,
    Stage1RolloutRequest,
    Stage1RolloutResponse,
    read_request,
    read_response,
    request_path,
    response_path,
    write_request,
    write_response,
)


def plan() -> PlanState:
    return PlanState(
        global_goal="reach the kitchen",
        points=(
            PlanPoint(id="p1", status="current", text="leave the bedroom"),
            PlanPoint(id="p2", status="todo", text="enter the kitchen"),
        ),
    )


def request() -> Stage1RolloutRequest:
    return Stage1RolloutRequest(
        episode_id="7",
        request_id=11,
        turn_index=3,
        instruction="Leave the bedroom and enter the kitchen.",
        current_plan=plan(),
        visual_history_paths=("/tmp/frame-1.npy", "/tmp/frame-2.npy"),
        action_history=("TURN_LEFT",),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )


def test_request_round_trip_is_oracle_free(tmp_path):
    path = request_path(tmp_path, 11)
    write_request(path, request())

    restored = read_request(path)

    assert restored == request()
    payload = json.loads(path.read_text())
    assert payload["schema"] == "cfrp.stage1.request.v1"
    assert payload["visual_history_paths"][-1] == "/tmp/frame-2.npy"
    for forbidden in ("pose", "goal_positions", "distance_to_goal", "reference_path", "expert_path"):
        assert forbidden not in payload


def test_request_rejects_oracle_field(tmp_path):
    path = request_path(tmp_path, 11)
    payload = request().to_dict()
    payload["distance_to_goal"] = 1.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="oracle fields"):
        read_request(path)


def test_first_turn_request_round_trips_without_a_plan(tmp_path):
    first = Stage1RolloutRequest(
        episode_id="7",
        request_id=0,
        turn_index=0,
        instruction="Leave the bedroom.",
        current_plan=None,
        visual_history_paths=("/tmp/frame-0.npy",),
        action_history=tuple(),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )
    path = request_path(tmp_path, 0)

    write_request(path, first)

    assert read_request(path) == first
    assert json.loads(path.read_text())["current_plan"] is None


def test_response_round_trip(tmp_path):
    path = response_path(tmp_path, 11)
    response = Stage1RolloutResponse(
        episode_id="7",
        request_id=11,
        turn_index=3,
        raw_xml="<progress>hold</progress><subgoal>look</subgoal><action>TURN_LEFT</action>",
    )
    write_response(path, response)

    assert read_response(path) == response
