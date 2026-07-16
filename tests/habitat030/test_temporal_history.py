from vlnce_server.habitat030.temporal_history import (
    SlowFastVisualHistory,
    select_temporal_history,
    temporal_history_indices,
    temporal_history_spec,
)


def test_temporal_history_uses_stable_slow_memory_then_recent_control_frames():
    assert temporal_history_indices(51) == (0, 7, 14, 21, 28, 35, 42, 49, 50)
    assert select_temporal_history(tuple(range(51))) == (0, 7, 14, 21, 28, 35, 42, 49, 50)


def test_temporal_history_does_not_pad_or_duplicate_early_frames():
    assert temporal_history_indices(1) == (0,)
    assert temporal_history_indices(8) == tuple(range(8))
    assert temporal_history_indices(9) == tuple(range(9))


def test_slow_memory_is_held_between_refreshes_then_rebuilt_deterministically():
    history = SlowFastVisualHistory[int].create().reset(0)
    for value in range(1, 10):
        history = history.append(value)
    assert history.slow_memory == (0, 1, 2, 3, 4, 5, 6, 8)
    assert history.visible == (0, 1, 2, 3, 4, 5, 6, 8, 9)

    history = history.append(10)
    assert history.slow_memory == (0, 1, 2, 3, 5, 6, 7, 9)
    assert history.visible == (0, 1, 2, 3, 5, 6, 7, 9, 10)

    for value in range(11, 18):
        history = history.append(value)
    assert history.slow_memory == (0, 2, 4, 6, 9, 11, 13, 16)
    assert history.visible == (0, 2, 4, 6, 9, 11, 13, 16, 17)


def test_temporal_history_contract_is_eight_route_anchors_plus_current_frame():
    assert temporal_history_spec() == {
        "sampling": "stateful_slow_memory_plus_recent_contiguous",
        "visual_context_window": 160,
        "history_anchor_count": 8,
        "recent_contiguous_count": 1,
        "slow_memory_update_interval": 1,
        "model_visual_frame_count": 9,
    }
