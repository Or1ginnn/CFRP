from __future__ import annotations

from dataclasses import dataclass

from vlnce_server.cfrp import PlanPoint, PlanState
from vlnce_server.qwen3vl import Qwen3VLStage1Policy, Stage1ModelRequest, VLLMStage1Client, build_stage1_messages, make_openai_messages


def plan() -> PlanState:
    return PlanState(
        global_goal="reach the kitchen",
        points=(
            PlanPoint(id="p1", status="current", text="leave the bedroom"),
            PlanPoint(id="p2", status="todo", text="enter the kitchen"),
        ),
    )


def request() -> Stage1ModelRequest:
    return Stage1ModelRequest(
        instruction="Leave the bedroom and enter the kitchen.",
        current_plan=plan(),
        visual_history=("rgb-oldest", "rgb-newest"),
        action_history=("TURN_LEFT",),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )


def test_stage1_messages_include_only_model_visible_context_in_order():
    messages = build_stage1_messages(request())

    assert messages[0]["role"] == "system"
    assert "Output only XML" in messages[0]["content"]
    content = messages[1]["content"]
    assert content[0]["type"] == "text"
    assert "Leave the bedroom and enter the kitchen." in content[0]["text"]
    assert '<p id="p1" status="current">leave the bedroom</p>' in content[0]["text"]
    assert "Executed recent actions (oldest to newest):\nTURN_LEFT" in content[0]["text"]
    assert [item["image"] for item in content if item["type"] == "image"] == [
        "rgb-oldest",
        "rgb-newest",
    ]
    rendered = "\n".join(str(item) for item in content)
    for forbidden in ("goal_positions", "distance_to_goal", "reference_path", "expert_path", "pose"):
        assert forbidden not in rendered


def test_vllm_messages_preserve_text_and_image_order(monkeypatch):
    monkeypatch.setattr("vlnce_server.qwen3vl.vllm_client._png_data_uri", lambda image: "data:" + image)

    messages = make_openai_messages(request())

    assert messages[0]["role"] == "system"
    assert [item["type"] for item in messages[1]["content"]] == ["text", "text", "image_url", "text", "image_url"]
    assert messages[1]["content"][2]["image_url"]["url"] == "data:rgb-oldest"


def test_vllm_client_keeps_a_fixed_request_seed(monkeypatch):
    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"choices": [{"message": {"content": "<action>STOP</action>"}}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_urlopen(http_request, timeout):
        captured["payload"] = http_request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("vlnce_server.qwen3vl.vllm_client.urlopen", fake_urlopen)
    monkeypatch.setattr("vlnce_server.qwen3vl.vllm_client.make_openai_messages", lambda _request: [])
    client = VLLMStage1Client("http://127.0.0.1:8000", "cfrp-stage1", seed=77)

    client.generate_xml(request())

    assert b'"seed": 77' in captured["payload"]
    assert b'"max_pixels": 150528' in captured["payload"]


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=[[10, 11, 12]])
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


@dataclass
class FakeModel:
    device: str = "cuda:0"

    def __post_init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[10, 11, 12, 20, 21]]


class FakeProcessor:
    def __init__(self):
        self.messages = None
        self.call_kwargs = None
        self.inputs = FakeInputs()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        self.messages = messages
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is True
        assert return_tensors == "pt"
        return self.inputs

    def batch_decode(self, continuation_ids, **kwargs):
        assert continuation_ids == [[20, 21]]
        assert kwargs["skip_special_tokens"] is True
        return ["<progress>hold</progress><subgoal>look ahead</subgoal><action>MOVE_FORWARD</action>"]


def test_policy_generates_xml_with_injected_runtime_without_torch():
    model = FakeModel()
    processor = FakeProcessor()
    policy = Qwen3VLStage1Policy(model, processor, max_new_tokens=64)
    raw_xml = policy.generate_xml(request())

    assert raw_xml == "<progress>hold</progress><subgoal>look ahead</subgoal><action>MOVE_FORWARD</action>"
    assert processor.inputs.moved_to == "cuda:0"
    assert model.generate_calls[0]["max_new_tokens"] == 64
    assert model.generate_calls[0]["do_sample"] is False
