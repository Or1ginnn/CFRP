from vlnce_server.habitat030.temporal_history import (
    select_temporal_history,
    temporal_history_indices,
    temporal_history_spec,
)


def test_temporal_history_uses_uniform_route_anchors_then_recent_control_frames():
    assert temporal_history_indices(51) == (19, 24, 30, 35, 41, 47, 48, 49, 50)
    assert select_temporal_history(tuple(range(51))) == (19, 24, 30, 35, 41, 47, 48, 49, 50)


def test_temporal_history_does_not_pad_or_duplicate_early_frames():
    assert temporal_history_indices(1) == (0,)
    assert temporal_history_indices(8) == tuple(range(8))
    assert temporal_history_indices(9) == tuple(range(9))


def test_temporal_history_contract_is_the_explicit_six_plus_three_schema():
    assert temporal_history_spec() == {
        "sampling": "uniform_history_anchors_plus_recent_contiguous",
        "visual_context_window": 32,
        "history_anchor_count": 6,
        "recent_contiguous_count": 3,
        "model_visual_frame_count": 9,
    }
