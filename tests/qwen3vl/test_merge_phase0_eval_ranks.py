import json

from scripts.merge_phase0_eval_ranks import _aggregate, _read_rank_progress


def test_rank_progress_merge_and_aggregate(tmp_path):
    records = [
        {"scene_id": "a", "episode_id": 1, "success": 1.0, "spl": 0.5, "os": 1.0, "ne": 1.0},
        {"scene_id": "b", "episode_id": 2, "success": 0.0, "spl": 0.0, "os": 1.0, "ne": 5.0},
    ]
    for rank, record in enumerate(records):
        (tmp_path / "progress_rank{}.json".format(rank)).write_text(json.dumps(record) + "\n")

    loaded = list(_read_rank_progress(tmp_path))
    assert loaded == records
    assert _aggregate(loaded) == {
        "sucs_all": 0.5,
        "spls_all": 0.25,
        "oss_all": 1.0,
        "nes_all": 3.0,
        "length": 2,
    }
