from vlnce_server.cfrp import advance_turn_indices, initialize_plan_from_instruction


def test_plan_initializer_makes_compact_clause_plan():
    plan = initialize_plan_from_instruction(
        "Leave the bedroom, then turn left down the hallway; stop by the kitchen."
    )

    assert plan.global_goal == "Leave the bedroom -> turn left down the hallway -> stop by the kitchen"
    assert [point.text for point in plan.points] == [
        "Leave the bedroom",
        "turn left down the hallway",
        "stop by the kitchen",
    ]
    assert [point.status for point in plan.points] == ["current", "todo", "todo"]


def test_plan_initializer_caps_plan_points_and_is_deterministic():
    instruction = "Go forward. Then turn left. Then turn right. Then stop."

    assert initialize_plan_from_instruction(instruction, max_points=2) == initialize_plan_from_instruction(
        instruction, max_points=2
    )
    assert len(initialize_plan_from_instruction(instruction, max_points=2).points) == 2


def test_advance_turn_indices_spread_cursor_updates_before_stop():
    plan = initialize_plan_from_instruction("Leave the room. Then cross the hall. Then stop.")

    assert advance_turn_indices(12, plan) == (3, 7)
    assert advance_turn_indices(4, plan) == (0, 1)
