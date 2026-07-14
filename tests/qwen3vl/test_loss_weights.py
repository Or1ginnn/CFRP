from vlnce_server.qwen3vl.loss_weights import (
    DEFAULT_ACTION_LOSS_WEIGHT,
    DEFAULT_PROGRESS_LOSS_WEIGHT,
    DEFAULT_SUBGOAL_LOSS_WEIGHT,
    locate_target_token_weights,
    target_xml_region_weights,
)


TARGET = "<progress>hold</progress><subgoal>walk past the table</subgoal><actions><action>MOVE_FORWARD</action><action>TURN_LEFT</action></actions>"


def _char_offsets(text: str):
    return [(index, index + 1) for index in range(len(text))]


def _char_tokenizer(text: str, **_kwargs):
    return {
        "input_ids": [ord(char) for char in text],
        "offset_mapping": _char_offsets(text),
    }


def test_target_xml_weights_emphasize_actions_and_downweight_subgoal():
    weights = target_xml_region_weights(TARGET, _char_offsets(TARGET))

    assert weights[TARGET.index("hold")] == DEFAULT_PROGRESS_LOSS_WEIGHT
    assert weights[TARGET.index("walk")] == DEFAULT_SUBGOAL_LOSS_WEIGHT
    assert weights[TARGET.index("MOVE_FORWARD")] == DEFAULT_ACTION_LOSS_WEIGHT
    assert weights[TARGET.index("TURN_LEFT")] == DEFAULT_ACTION_LOSS_WEIGHT
    assert weights[TARGET.index("<progress>")] == 1.0


def test_locate_target_weights_preserves_template_prefix_and_suffix():
    target_ids = [999] + [ord(char) for char in TARGET] + [998]

    start, weights = locate_target_token_weights(TARGET, target_ids, _char_tokenizer)

    assert start == 1
    assert weights[TARGET.index("MOVE_FORWARD")] == DEFAULT_ACTION_LOSS_WEIGHT


def test_target_weights_reject_non_positive_values():
    try:
        target_xml_region_weights(TARGET, _char_offsets(TARGET), action_weight=0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected positive-weight validation")
