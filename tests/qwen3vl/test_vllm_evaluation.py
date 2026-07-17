from types import SimpleNamespace

from scripts.habitat030_r2r_vllm_eval import _make_job, _stage1_termination_reason


def test_vllm_evaluation_job_preserves_batch_runtime_configuration(tmp_path):
    args = SimpleNamespace(
        dataset_root="/data/r2r", scenes_dir="/data/scenes", config="/config.yaml", split="val_seen", seed=123,
        max_steps=160, max_visual_history=6, max_action_history=8, success_distance=3.0,
        vllm_base_url="http://127.0.0.1:8000", vllm_model="cfrp-stage1", max_new_tokens=128, response_timeout=600.0,
        save_frames=False, save_video=True, save_oracle_trace=False,
        action_queue_mode="rolling", max_actions_during_inference=1, rank=3,
    )

    job = _make_job(args, tmp_path, "7", 1)

    assert job.episode_id == "7"
    assert job.repeat_index == 1
    assert job.max_visual_history == 6
    assert job.vllm_model == "cfrp-stage1"
    assert job.save_frames is False
    assert job.save_video is True
    assert job.action_queue_mode == "rolling"
    assert job.max_actions_during_inference == 1
    assert job.rank == 3


def test_habitat_time_limit_is_not_reported_as_model_stop():
    assert _stage1_termination_reason(SimpleNamespace(action="MOVE_FORWARD", episode_over=True)) == (
        "environment_episode_over"
    )
    assert _stage1_termination_reason(SimpleNamespace(action="STOP", episode_over=True)) == "stop"
