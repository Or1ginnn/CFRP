from pathlib import Path

from scripts.convert_stage1_sft_to_llamafactory import _write_train_config


def test_llamafactory_config_enables_wandb_for_formal_runs(tmp_path: Path):
    path = tmp_path / "train_lora.yaml"

    _write_train_config(path, tmp_path, "cfrp_stage1", tmp_path / "adapter", "stage1-pilot")

    config = path.read_text(encoding="utf-8")
    assert "report_to: wandb" in config
    assert "run_name: stage1-pilot" in config
