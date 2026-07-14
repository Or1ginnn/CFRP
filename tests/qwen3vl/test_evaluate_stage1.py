import json

from scripts.evaluate_stage1 import (
    _merge_and_aggregate_results,
    _partition_episode_ids,
    _summarize_records,
    _validate_adapter_rank,
)


def test_partition_episode_ids_keeps_order_and_balances_ranks():
    assert _partition_episode_ids(["1", "2", "3", "4", "5"], 3) == [
        ["1", "4"],
        ["2", "5"],
        ["3"],
    ]


def test_summarize_records_uses_rollout_metrics():
    records = [
        {
            "final_metrics": {"success": 1.0, "spl": 0.5},
            "navigation_error": 1.0,
            "oracle_success": True,
            "invalid_output": False,
            "stop_correct": True,
            "environment_steps": 2,
            "steps": [{}, {}],
        },
        {
            "final_metrics": {"success": 0.0, "spl": 0.0},
            "navigation_error": 3.0,
            "oracle_success": False,
            "invalid_output": True,
            "stop_correct": False,
            "environment_steps": 1,
            "steps": [{}, {}, {}],
        },
    ]

    assert _summarize_records(records) == {
        "episodes": 2.0,
        "sr": 0.5,
        "spl": 0.25,
        "navigation_error": 2.0,
        "oracle_success": 0.5,
        "invalid_output_rate": 0.5,
        "stop_correct_rate": 0.5,
        "average_steps": 1.5,
    }


def test_adapter_rank_must_fit_vllm_limit(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32}))

    _validate_adapter_rank(tmp_path, 32)


def test_merge_and_aggregate_writes_internnav_artifacts(tmp_path):
    trajectory = {
        "episode_id": "7",
        "scene_id": "mp3d/scene/scene.glb",
        "final_metrics": {"success": 1.0, "spl": 0.5},
        "navigation_error": 1.0,
        "oracle_success": True,
        "invalid_output": False,
        "stop_correct": True,
        "steps": [{}],
    }
    progress = {
        "episode_id": 7,
        "scene_id": "scene",
        "success": 1.0,
        "spl": 0.5,
        "os": 1.0,
        "ne": 1.0,
    }
    (tmp_path / "trajectories_rank1.jsonl").write_text(json.dumps(trajectory) + "\n")
    (tmp_path / "progress_rank1.json").write_text(json.dumps(progress) + "\n")

    summary = _merge_and_aggregate_results(tmp_path, ["7"])

    assert summary["sr"] == 1.0
    assert (tmp_path / "trajectories.jsonl").is_file()
    assert (tmp_path / "progress.json").is_file()
    assert json.loads((tmp_path / "result.json").read_text())["length"] == 1
