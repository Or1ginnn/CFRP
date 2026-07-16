from vlnce_server.cfrp import PlanPoint, PlanState, Stage1RolloutRequest
from vlnce_server.qwen3vl.sft_data import (
    SFT_SCHEMA,
    make_stage1_sft_conversations,
    make_stage1_sft_example,
)


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
    assert example["visual_contract"] == {
        "history_anchor_count": 8,
        "new_observations_per_turn": 1,
        "max_active_dialogue_turns": 8,
        "max_window_images": 16,
    }
    assert example["images"] == ["file:///output/frame.png"]
    assert example["messages"][-1]["role"] == "assistant"
    assert example["messages"][-1]["content"].startswith("<plan>")
    assert example["messages"][-1]["content"].endswith("<action>MOVE_FORWARD</action>")
    assert example["targets"][0]["initializes_plan"] is True
    user_images = [item["image"] for item in example["messages"][1]["content"] if item["type"] == "image"]
    assert user_images == ["file:///output/frame.png"]


def test_episode_becomes_bounded_multiturn_windows_with_incremental_images():
    current_plan = PlanState(
        global_goal="leave bedroom -> stop hallway",
        points=(
            PlanPoint("p1", "current", "leave the bedroom"),
            PlanPoint("p2", "todo", "stop in the hallway"),
        ),
    )

    def record(request_id: int, turn_index: int, frames: tuple[str, ...], action: str):
        request = Stage1RolloutRequest(
            episode_id="1",
            request_id=request_id,
            turn_index=turn_index,
            instruction="Leave the bedroom and stop in the hallway.",
            current_plan=current_plan,
            visual_history_paths=frames,
            action_history=tuple(),
            allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
        )
        return {
            "model_input": request.to_dict(),
            "target_xml": (
                "<progress>hold</progress><subgoal>leave the bedroom</subgoal>"
                f"<action>{action}</action>"
            ),
        }

    records = [
        record(0, 0, ("/source/f0.npy",), "MOVE_FORWARD"),
        record(1, 1, ("/source/f0.npy", "/source/f1.npy"), "TURN_LEFT"),
        record(2, 2, ("/source/f0.npy", "/source/f1.npy", "/source/f2.npy"), "STOP"),
    ]
    windows = make_stage1_sft_conversations(
        records,
        [
            ("file:///output/f0.png",),
            ("file:///output/f0.png", "file:///output/f1.png"),
            ("file:///output/f0.png", "file:///output/f1.png", "file:///output/f2.png"),
        ],
        max_turns=2,
    )

    assert [len(window["targets"]) for window in windows] == [2, 1]
    assert windows[0]["images"] == [
        "file:///output/f0.png",
        "file:///output/f1.png",
    ]
    assert windows[1]["images"] == [
        "file:///output/f0.png",
        "file:///output/f1.png",
        "file:///output/f2.png",
    ]
    assert windows[0]["targets"][0]["initializes_plan"] is True
    assert windows[0]["targets"][1]["initializes_plan"] is False


def test_eight_turn_window_appends_only_one_current_observation_per_turn():
    current_plan = PlanState(
        global_goal="reach hallway",
        points=(PlanPoint("p1", "current", "reach hallway"),),
    )
    records = []
    image_rows = []
    for turn in range(8):
        request = Stage1RolloutRequest(
            episode_id="stream",
            request_id=turn,
            turn_index=turn * 3,
            instruction="Reach the hallway.",
            current_plan=current_plan,
            visual_history_paths=tuple(f"/source/a-{index}.npy" for index in range(8))
            + (f"/source/current-{turn}.npy",),
            action_history=tuple(),
            allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
        )
        records.append(
            {
                "model_input": request.to_dict(),
                "target_xml": (
                    "<progress>hold</progress><subgoal>reach hallway</subgoal>"
                    "<action>MOVE_FORWARD</action>"
                ),
            }
        )
        image_rows.append(
            tuple(f"file:///anchors/a-{index}.png" for index in range(8))
            + (f"file:///current/{turn}.png",)
        )

    windows = make_stage1_sft_conversations(records, image_rows)

    assert len(windows) == 1
    user_messages = windows[0]["messages"][1::2]
    image_counts = [
        sum(item["type"] == "image" for item in message["content"])
        for message in user_messages
    ]
    assert image_counts == [9, 1, 1, 1, 1, 1, 1, 1]
    assert len(windows[0]["images"]) == 16
