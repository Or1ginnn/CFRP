from vlnce_server.cfrp import PlanPoint, PlanState, Stage1RolloutRequest
from vlnce_server.qwen3vl.sft_data import SFT_SCHEMA, make_stage1_sft_example


def test_sft_example_preserves_multimodal_stage1_contract():
    request = Stage1RolloutRequest(
        episode_id="1",
        request_id=0,
        turn_index=0,
        instruction="Leave the bedroom and stop in the hallway.",
        current_plan=PlanState(
            global_goal="leave bedroom -> stop hallway",
            points=(
                PlanPoint("p1", "current", "leave the bedroom"),
                PlanPoint("p2", "todo", "stop in the hallway"),
            ),
        ),
        visual_history_paths=("/source/frame.npy",),
        action_history=tuple(),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )
    example = make_stage1_sft_example(
        {"model_input": request.to_dict(), "target_xml": "<progress>hold</progress><subgoal>leave the bedroom</subgoal><action>MOVE_FORWARD</action>"},
        ("file:///output/frame.png",),
    )

    assert example["schema"] == SFT_SCHEMA
    assert example["images"] == ["file:///output/frame.png"]
    assert example["messages"][-1]["role"] == "assistant"
    assert example["messages"][-1]["content"].endswith("<action>MOVE_FORWARD</action>")
    user_images = [item["image"] for item in example["messages"][1]["content"] if item["type"] == "image"]
    assert user_images == ["file:///output/frame.png"]
