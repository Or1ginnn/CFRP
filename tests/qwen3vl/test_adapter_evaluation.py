from pathlib import Path
from types import SimpleNamespace

from scripts.habitat030_r2r_qwen_baseline import _worker_command


def test_worker_command_passes_adapter_only_when_configured():
    args = SimpleNamespace(
        model_python="/env/bin/python",
        model="Qwen/Qwen3-VL-4B-Instruct",
        adapter="/runs/adapter",
        max_new_tokens=128,
    )

    command = _worker_command(args, Path("/exchange"), Path("/worker.py"))

    assert command[-4:] == ["--adapter", "/runs/adapter", "--max-new-tokens", "128"]


def test_worker_command_omits_adapter_for_base_model():
    args = SimpleNamespace(
        model_python="/env/bin/python",
        model="Qwen/Qwen3-VL-4B-Instruct",
        adapter=None,
        max_new_tokens=128,
    )

    command = _worker_command(args, Path("/exchange"), Path("/worker.py"))

    assert "--adapter" not in command
