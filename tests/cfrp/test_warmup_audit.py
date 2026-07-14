from vlnce_server.cfrp import PlanPoint, PlanState, Stage1RolloutRequest, audit_stage1_warmup


def _record(turn: int, action: str, progress: str, plan: PlanState, action_history=()):
    request = Stage1RolloutRequest(
        episode_id="1",
        request_id=turn,
        turn_index=turn,
        instruction="Leave the room.",
        current_plan=plan,
        visual_history_paths=("/tmp/frame.npy",),
        action_history=action_history,
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )
    current = plan.current_points()[0]
    return {
        "model_input": request.to_dict(),
        "target_xml": f"<progress>{progress}</progress><subgoal>{current.text}</subgoal><action>{action}</action>",
        "oracle_only": {"oracle_action": action},
    }


def _manifest():
    return {
        "status": "complete",
        "requested_episode_ids": ["1"],
        "completed_episode_ids": ["1"],
        "max_visual_history": 4,
        "max_action_history": 3,
        "visual_contract": {"habitat_rgb_size": [640, 480]},
    }


def test_audit_accepts_complete_consistent_trajectory():
    plan = PlanState("leave", (PlanPoint("p1", "current", "leave room"), PlanPoint("p2", "todo", "stop")))
    records = [
        _record(0, "MOVE_FORWARD", "advance", plan),
        _record(1, "STOP", "hold", plan.advance_current(), ("MOVE_FORWARD",)),
    ]

    summary = audit_stage1_warmup(records, _manifest())

    assert summary["records"] == 2
    assert summary["action_counts"] == {"MOVE_FORWARD": 1, "STOP": 1}


def test_audit_rejects_target_not_matching_oracle_action():
    plan = PlanState("leave", (PlanPoint("p1", "current", "leave room"), PlanPoint("p2", "todo", "stop")))
    record = _record(0, "STOP", "hold", plan)
    record["oracle_only"]["oracle_action"] = "MOVE_FORWARD"

    try:
        audit_stage1_warmup([record], _manifest())
    except ValueError as exc:
        assert "do not match oracle actions" in str(exc)
    else:
        raise AssertionError("expected semantic audit to fail")
